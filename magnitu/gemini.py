"""Shared Gemini client with JSON repair pipeline.

Adapted from transcribe-v3 ``mac/v3/gemini.py``; HTTP uses ``httpx`` (project standard).
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List, Optional

import httpx
from json_repair import repair_json

from magnitu.gemini_config import GeminiConfig

_BASE = "https://generativelanguage.googleapis.com/v1beta"


def _log(msg: str) -> None:
    print(f"[gemini] {msg}", flush=True)


# ---------------------------------------------------------------------------
# JSON repair pipeline (ported from transcribe-v3)
# ---------------------------------------------------------------------------


def _extract_markdown_json(text: str) -> str:
    stripped = (text or "").strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        return stripped[start : end + 1]
    return stripped


def _remove_trailing_commas(text: str) -> str:
    return re.sub(r",\s*([}\]])", r"\1", text)


def _insert_missing_commas(text: str) -> str:
    if not text:
        return text
    repaired = re.sub(r"}\s*{", "},{", text)
    repaired = re.sub(r"]\s*\[", "],[", repaired)
    lines = repaired.splitlines()
    if len(lines) < 2:
        return repaired
    out: List[str] = []
    for idx, line in enumerate(lines):
        out.append(line)
        if idx >= len(lines) - 1:
            continue
        curr = line.rstrip()
        nxt = lines[idx + 1].lstrip()
        curr_s = curr.strip()
        if not curr_s or not nxt:
            continue
        if curr_s.endswith(","):
            continue
        if nxt.startswith(("}", "]")):
            continue
        if curr_s.endswith(("{", "[", ":")):
            continue
        if ":" in curr_s and nxt.startswith('"'):
            out[-1] = curr + ","
            continue
        if curr_s.endswith(("}", "]", '"')) and nxt.startswith(("{", "[", '"')):
            out[-1] = curr + ","
    return "\n".join(out)


def _extract_balanced_json_object(text: str) -> str:
    s = (text or "").strip()
    start = s.find("{")
    if start == -1:
        return s
    stack: List[str] = []
    in_string = False
    escaped = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in ("{", "["):
            stack.append(ch)
        elif ch == "}":
            if stack and stack[-1] == "{":
                stack.pop()
            if not stack:
                return s[start : i + 1]
        elif ch == "]":
            if stack and stack[-1] == "[":
                stack.pop()
    closing = "".join("]" if c == "[" else "}" for c in reversed(stack))
    return s[start:] + closing


def parse_json_lenient(text: str) -> Dict[str, Any]:
    """Multi-layer lenient JSON parser. Raises on total failure."""
    candidates: List[str] = []
    base = _extract_markdown_json(text)
    candidates.append(base)
    candidates.append(_extract_balanced_json_object(base))
    candidates.append(_remove_trailing_commas(candidates[-1]))
    candidates.append(_insert_missing_commas(candidates[-1]))
    candidates.append(_remove_trailing_commas(candidates[-1]))
    candidates.append(_insert_missing_commas(_extract_balanced_json_object(base)))

    last_err: Optional[Exception] = None
    for candidate in candidates:
        candidate = (candidate or "").strip()
        if not candidate:
            continue
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except Exception as e:
            last_err = e

    try:
        repaired = repair_json(base, return_objects=True)
        if isinstance(repaired, dict):
            return repaired
    except Exception as e:
        last_err = e

    raise ValueError(
        "Gemini output is not valid JSON: %s; text=%s" % (last_err, base[:2000])
    )


# ---------------------------------------------------------------------------
# Gemini request helpers
# ---------------------------------------------------------------------------


class GeminiClient:
    def __init__(self, cfg: GeminiConfig):
        self.cfg = cfg
        self._client = httpx.Client(headers={"Content-Type": "application/json"})

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> GeminiClient:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def request_json(
        self,
        prompt: str,
        *,
        label: str = "",
        response_schema: Optional[Dict[str, Any]] = None,
        model_override: Optional[str] = None,
        timeout_override: int = 0,
        system_instruction: Optional[str] = None,
    ) -> Dict[str, Any]:
        model = self.cfg.effective_model(model_override)
        url = "%s/%s:generateContent" % (_BASE, model)
        json_retries = max(0, self.cfg.json_retries)
        max_retries = max(0, self.cfg.max_retries)
        timeout_s = max(10, timeout_override or self.cfg.timeout_s)
        backoff_s = max(0.5, self.cfg.retry_backoff_s)

        total_attempts = max_retries + json_retries + 1
        # Request structured output whenever the caller supplies a schema (synthetic
        # single + batch). If the API rejects responseSchema, we fall back to JSON MIME only.
        use_schema = bool(response_schema)

        for attempt in range(total_attempts):
            strict_note = ""
            if attempt > max_retries:
                strict_note = (
                    "\n\nIMPORTANT JSON REQUIREMENT:\n"
                    "- Return ONLY one complete JSON object.\n"
                    "- No markdown fences.\n"
                    "- No trailing commas.\n"
                    "- Ensure valid closing braces/brackets.\n"
                )
            payload: Dict[str, Any] = {
                "contents": [
                    {"role": "user", "parts": [{"text": prompt + strict_note}]}
                ],
                "generationConfig": {
                    "temperature": max(0.0, self.cfg.temperature),
                    "maxOutputTokens": max(256, self.cfg.max_output_tokens),
                    "responseMimeType": "application/json",
                },
            }
            if system_instruction and system_instruction.strip():
                payload["systemInstruction"] = {
                    "parts": [{"text": system_instruction.strip()}],
                }
            if use_schema:
                payload["generationConfig"]["responseSchema"] = response_schema

            r: Optional[httpx.Response] = None
            try:
                r = self._client.post(
                    url,
                    params={"key": self.cfg.api_key},
                    json=payload,
                    timeout=timeout_s,
                )
                if r.status_code in {429, 500, 502, 503, 504} and attempt < (
                    total_attempts - 1
                ):
                    wait_s = backoff_s * (attempt + 1)
                    _log(
                        "Transient HTTP %s for %s; retry %s/%s in %.1fs."
                        % (
                            r.status_code,
                            label,
                            attempt + 1,
                            total_attempts - 1,
                            wait_s,
                        )
                    )
                    time.sleep(wait_s)
                    continue
                if r.status_code >= 400:
                    if r.status_code == 400 and use_schema:
                        _log(
                            "Strict responseSchema rejected for %s; falling back to MIME-only JSON mode."
                            % label
                        )
                        use_schema = False
                        continue
                    raise ValueError(
                        "Gemini request failed (%s): %s"
                        % (r.status_code, (r.text or "")[:1000])
                    )
            except (httpx.TimeoutException, httpx.ConnectError, httpx.TransportError) as e:
                if attempt >= (total_attempts - 1):
                    raise ValueError(
                        "Gemini network timeout/connection error: %s" % e
                    ) from e
                wait_s = backoff_s * (attempt + 1)
                _log(
                    "Network issue for %s: %s. Retry %s/%s in %.1fs."
                    % (label, e, attempt + 1, total_attempts - 1, wait_s)
                )
                time.sleep(wait_s)
                continue

            if r is None:
                continue
            data = r.json()
            try:
                text = data["candidates"][0]["content"]["parts"][0]["text"]
            except Exception:
                if attempt >= (total_attempts - 1):
                    raise ValueError(
                        "Gemini response missing content text: %s"
                        % (json.dumps(data)[:1200],)
                    )
                wait_s = backoff_s * (attempt + 1)
                _log(
                    "Missing content text for %s; retry %s/%s in %.1fs."
                    % (label, attempt + 1, total_attempts - 1, wait_s)
                )
                time.sleep(wait_s)
                continue

            try:
                return parse_json_lenient(text)
            except Exception as e:
                if attempt >= (total_attempts - 1):
                    raise
                wait_s = backoff_s * (attempt + 1)
                _log(
                    "Malformed JSON for %s: %s. Retry %s/%s in %.1fs."
                    % (label, e, attempt + 1, total_attempts - 1, wait_s)
                )
                time.sleep(wait_s)

        raise ValueError("Gemini request did not produce a valid response.")

    def request_text(
        self,
        prompt: str,
        *,
        label: str = "",
        model_override: Optional[str] = None,
        max_output_tokens: int = 0,
        timeout_override: int = 0,
        system_instruction: Optional[str] = None,
    ) -> str:
        model = self.cfg.effective_model(model_override)
        url = "%s/%s:generateContent" % (_BASE, model)
        max_retries = max(0, self.cfg.max_retries)
        timeout_s = max(10, timeout_override or self.cfg.timeout_s)
        backoff_s = max(0.5, self.cfg.retry_backoff_s)
        tokens = max_output_tokens or max(256, self.cfg.max_output_tokens)

        for attempt in range(max_retries + 1):
            payload: Dict[str, Any] = {
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.0, "maxOutputTokens": tokens},
            }
            if system_instruction and system_instruction.strip():
                payload["systemInstruction"] = {
                    "parts": [{"text": system_instruction.strip()}],
                }
            try:
                r = self._client.post(
                    url,
                    params={"key": self.cfg.api_key},
                    json=payload,
                    timeout=timeout_s,
                )
                if r.status_code in {429, 500, 502, 503, 504} and attempt < max_retries:
                    wait_s = backoff_s * (attempt + 1)
                    _log(
                        "Transient HTTP %s for %s; retry in %.1fs."
                        % (r.status_code, label, wait_s)
                    )
                    time.sleep(wait_s)
                    continue
                if r.status_code >= 400:
                    raise ValueError(
                        "Gemini text request failed (%s): %s"
                        % (r.status_code, (r.text or "")[:1000])
                    )
            except (httpx.TimeoutException, httpx.ConnectError, httpx.TransportError) as e:
                if attempt >= max_retries:
                    raise ValueError("Gemini text network error: %s" % e) from e
                wait_s = backoff_s * (attempt + 1)
                _log("Network issue for %s: %s. Retry in %.1fs." % (label, e, wait_s))
                time.sleep(wait_s)
                continue

            data = r.json()
            try:
                return data["candidates"][0]["content"]["parts"][0]["text"]
            except Exception:
                if attempt >= max_retries:
                    raise ValueError(
                        "Gemini text response missing content: %s"
                        % (json.dumps(data)[:1200],)
                    )
                time.sleep(backoff_s * (attempt + 1))

        raise ValueError("Gemini text request did not produce a response.")
