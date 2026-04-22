"""Gemini REST settings — env names match transcribe-v3 / `.env.example`."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


def _env(key: str, default: str = "") -> str:
    v = os.environ.get(key)
    if v is None:
        return default
    return v.strip()


def _env_int(key: str, default: int, min_value: int = 0) -> int:
    raw = _env(key, "")
    if not raw:
        return max(min_value, default)
    try:
        return max(min_value, int(raw))
    except ValueError:
        return max(min_value, default)


def _env_float(key: str, default: float, min_value: float = 0.0) -> float:
    raw = _env(key, "")
    if not raw:
        return max(min_value, default)
    try:
        return max(min_value, float(raw))
    except ValueError:
        return max(min_value, default)


def _env_bool(key: str, default: str) -> bool:
    raw = _env(key, default).lower()
    return raw in ("1", "true", "yes", "on")


@dataclass
class GeminiConfig:
    api_key: str = ""
    model: str = "models/gemini-2.5-flash"
    chunk_model: str = ""
    temperature: float = 0.0
    max_output_tokens: int = 8192
    timeout_s: int = 120
    max_retries: int = 4
    retry_backoff_s: float = 8.0
    json_retries: int = 3
    # Unused for gating: callers pass response_schema; GeminiClient always requests it
    # when set, with HTTP-400 fallback to JSON MIME only.
    strict_schema: bool = False

    @classmethod
    def from_env(cls) -> GeminiConfig:
        # Check global app config (magnitu_config.json) first
        try:
            from config import get_config
            app_cfg = get_config()
        except ImportError:
            app_cfg = {}

        api_key = app_cfg.get("gemini_api_key") or _env("GEMINI_API_KEY")
        model = app_cfg.get("gemini_model") or _env("GEMINI_MODEL", "models/gemini-2.5-flash")

        return cls(
            api_key=api_key,
            model=model,
            chunk_model=_env("GEMINI_CHUNK_MODEL", ""),
            temperature=_env_float("GEMINI_TEMPERATURE", 0.0),
            max_output_tokens=_env_int("GEMINI_MAX_OUTPUT_TOKENS", 8192, min_value=256),
            timeout_s=_env_int("GEMINI_TIMEOUT_S", 120, min_value=10),
            max_retries=_env_int("GEMINI_MAX_RETRIES", 4),
            retry_backoff_s=_env_float("GEMINI_RETRY_BACKOFF_S", 8.0, min_value=0.5),
            json_retries=_env_int("GEMINI_JSON_RETRIES", 3),
            strict_schema=_env_bool("GEMINI_STRICT_SCHEMA", "0"),
        )

    def effective_model(self, override: Optional[str] = None) -> str:
        if override and override.strip():
            return override.strip().strip("/")
        if self.chunk_model and self.chunk_model.strip():
            return self.chunk_model.strip().strip("/")
        return self.model.strip().strip("/")
