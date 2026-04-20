"""Single-entry Gemini synthetic labeling: prompt → API → validate (+ one retry rule)."""

from __future__ import annotations

from typing import Any, Tuple

from magnitu.gemini import GeminiClient
from magnitu.prompts import (
    INVESTIGATION_LEAD_REASONING_RETRY_SUFFIX,
    SCHEMA_SYNTHETIC_LABEL,
    SYSTEM_SWISS_TRADE_ANALYST,
    build_synthetic_label_user_prompt,
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
) -> Tuple[str, str]:
    """Call Gemini once; retry at most once if investigation_lead has empty reasoning.

    Returns (label, reasoning). Raises ValueError if validation fails after retry.
    """
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
        system_instruction=SYSTEM_SWISS_TRADE_ANALYST,
    )
    if should_retry_investigation_lead_empty_reasoning(raw):
        user_retry = user + "\n\n" + INVESTIGATION_LEAD_REASONING_RETRY_SUFFIX
        raw = client.request_json(
            user_retry,
            label="synthetic_label_retry",
            response_schema=SCHEMA_SYNTHETIC_LABEL,
            system_instruction=SYSTEM_SWISS_TRADE_ANALYST,
        )
    return validate_synthetic_label_output(raw)
