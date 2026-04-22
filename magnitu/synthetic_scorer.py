"""Single-entry Gemini synthetic labeling: prompt → API → validate (+ one retry rule)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from magnitu.gemini import GeminiClient
from magnitu.prompts import (
    INVESTIGATION_LEAD_REASONING_RETRY_SUFFIX,
    SCHEMA_SYNTHETIC_LABEL,
    SCHEMA_SYNTHETIC_LABEL_BATCH,
    SYSTEM_SWISS_TRADE_ANALYST,
    build_synthetic_label_user_prompt,
    build_synthetic_label_batch_prompt,
    should_retry_investigation_lead_empty_reasoning,
    validate_synthetic_label_output,
)


def call_gemini_for_synthetic_label(
    client: GeminiClient,
    *,
    title: str = "",
    description: str = "",
    content: str = "",
    link: str = "",
    author: str = "",
    source_name: str = "",
    source_category: str = "",
    source_type: str = "",
    published_date: str = "",
    system_instruction: Optional[str] = None,
) -> Tuple[str, str]:
    """Call Gemini once; retry at most once if investigation_lead has empty reasoning.

    Returns (label, reasoning). Raises ValueError if validation fails after retry.
    """
    if system_instruction is None:
        system_instruction = SYSTEM_SWISS_TRADE_ANALYST
    
    fields: dict[str, Any] = {
        "title": title,
        "description": description,
        "content": content,
        "link": link,
        "author": author,
        "source_name": source_name,
        "source_category": source_category,
        "source_type": source_type,
        "published_date": published_date,
    }
    user = build_synthetic_label_user_prompt(**fields)
    raw = client.request_json(
        user,
        label="synthetic_label",
        response_schema=SCHEMA_SYNTHETIC_LABEL,
        system_instruction=system_instruction,
    )
    if should_retry_investigation_lead_empty_reasoning(raw):
        user_retry = user + "\n\n" + INVESTIGATION_LEAD_REASONING_RETRY_SUFFIX
        raw = client.request_json(
            user_retry,
            label="synthetic_label_retry",
            response_schema=SCHEMA_SYNTHETIC_LABEL,
            system_instruction=system_instruction,
        )
    return validate_synthetic_label_output(raw)


def call_gemini_for_synthetic_label_batch(
    client: GeminiClient,
    entries: List[Dict[str, Any]],
    *,
    system_instruction: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Call Gemini for a batch of entries. Returns list of {entry_type, entry_id, label, reasoning}."""
    if not entries:
        return []
    if system_instruction is None:
        system_instruction = SYSTEM_SWISS_TRADE_ANALYST
    
    user = build_synthetic_label_batch_prompt(entries)
    n = len(entries)
    raw = client.request_json(
        user,
        label="synthetic_label_batch",
        response_schema=SCHEMA_SYNTHETIC_LABEL_BATCH,
        system_instruction=system_instruction,
        timeout_override=max(0, 90 + 35 * n),
    )
    # raw should be {"labels": [...]}
    labels = raw.get("labels")
    if not isinstance(labels, list):
        raise ValueError("Gemini batch response missing 'labels' list")
    return labels
