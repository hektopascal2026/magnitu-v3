"""Magnitu Gemini prompts and response schemas for synthetic labeling.

Interview extraction templates live in transcribe-v3 ``mac/v3/extraction/prompts.py``;
this module is Magnitu-specific (four label classes + Swiss trade persona).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

# Seismo / Magnitu label enums (exact strings)
LABEL_INVESTIGATION_LEAD = "investigation_lead"
LABEL_IMPORTANT = "important"
LABEL_BACKGROUND = "background"
LABEL_NOISE = "noise"

MAGNITU_LABELS: List[str] = [
    LABEL_INVESTIGATION_LEAD,
    LABEL_IMPORTANT,
    LABEL_BACKGROUND,
    LABEL_NOISE,
]

# Human-readable names for prompts only
LABEL_DISPLAY: Dict[str, str] = {
    LABEL_INVESTIGATION_LEAD: "Investigation Lead",
    LABEL_IMPORTANT: "Important",
    LABEL_BACKGROUND: "Background",
    LABEL_NOISE: "Noise",
}

SYSTEM_SWISS_TRADE_ANALYST = """You are a Swiss economic and trade analyst helping prioritize \
news and legal signals for a professional monitoring desk.

Prioritize items that affect Switzerland, Swiss firms, or Swiss policy — including when \
the text does NOT name Switzerland explicitly.

Pay special attention to exclusion language that leaves Switzerland outside a regime, for example:
- EU or EEA-only scope, \"Member States\" only, internal market / Binnenmarkt without third countries
- \"Drittstaaten\" / third-country rules that imply non-EU treatment
- \"third countries\" or \"non-EU\" where Switzerland would follow third-country treatment

Be factual. Do not invent facts not supported by the text."""


SCHEMA_SYNTHETIC_LABEL: Dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "label": {"type": "STRING"},
        "reasoning": {"type": "STRING"},
    },
    "required": ["label", "reasoning"],
}

# Appended to the user prompt on the single retry when label is investigation_lead
# but reasoning was empty (workspace rule: never persist that combination).
INVESTIGATION_LEAD_REASONING_RETRY_SUFFIX = (
    "CRITICAL: Your last reply used label investigation_lead but reasoning was "
    "missing or empty. Respond again with the SAME JSON shape only: "
    '{"label": "investigation_lead", "reasoning": "<non-empty string>"}. '
    "The reasoning must explain the lead, including any EU/EEA-only, third-country, "
    "or exclusion angle if the text supports it. No markdown fences."
)


def build_synthetic_label_user_prompt(
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
) -> str:
    """Build the user message for one entry (JSON response: label + reasoning)."""
    label_line = ", ".join(
        "%s (%s)" % (LABEL_DISPLAY[k], k) for k in MAGNITU_LABELS
    )
    parts = [
        "Classify the following entry into exactly one label.",
        "Allowed labels: %s." % label_line,
        'Return ONLY valid JSON: {"label": "<enum>", "reasoning": "<string>"}.',
        "Use the enum strings exactly: investigation_lead, important, background, noise.",
        "reasoning must be concise (1-4 sentences), in English or German as appropriate.",
        (
            "If label is investigation_lead, reasoning must explain why it is a lead "
            "(including exclusion/third-country angle if relevant)."
        ),
        "",
        "ENTRY",
        "----",
    ]
    if title.strip():
        parts.append("title: %s" % title.strip())
    if description.strip():
        parts.append("description: %s" % description.strip())
    if content.strip():
        parts.append("content: %s" % content.strip())
    if link.strip():
        parts.append("link: %s" % link.strip())
    if author.strip():
        parts.append("author: %s" % author.strip())
    if source_name.strip():
        parts.append("source_name: %s" % source_name.strip())
    if source_category.strip():
        parts.append("source_category: %s" % source_category.strip())
    if source_type.strip():
        parts.append("source_type: %s" % source_type.strip())
    if published_date.strip():
        parts.append("published_date: %s" % published_date.strip())
    parts.append("")
    parts.append("No markdown fences. No trailing commas.")
    return "\n".join(parts)


def normalize_synthetic_label(raw: Dict[str, Any]) -> Optional[str]:
    """Return canonical enum or None if invalid."""
    label = (raw.get("label") or "").strip()
    if label in MAGNITU_LABELS:
        return label
    return None


def _reasoning_stripped(raw: Dict[str, Any]) -> str:
    reasoning = raw.get("reasoning")
    if reasoning is None:
        return ""
    if isinstance(reasoning, str):
        return reasoning.strip()
    return str(reasoning).strip()


def should_retry_investigation_lead_empty_reasoning(raw: Dict[str, Any]) -> bool:
    """True when parsed JSON is investigation_lead with missing/blank reasoning (one retry allowed)."""
    if not isinstance(raw, dict):
        return False
    if normalize_synthetic_label(raw) != LABEL_INVESTIGATION_LEAD:
        return False
    return len(_reasoning_stripped(raw)) == 0


def validate_synthetic_label_output(raw: Dict[str, Any]) -> Tuple[str, str]:
    """Validate Gemini synthetic JSON. Returns (label_enum, reasoning).

    Raises ValueError on unknown label or empty/missing reasoning.
    """
    if not isinstance(raw, dict):
        raise ValueError("Gemini response must be a JSON object")
    label = normalize_synthetic_label(raw)
    if label is None:
        raise ValueError("Invalid label: %r" % (raw.get("label"),))
    reasoning = _reasoning_stripped(raw)
    if not reasoning:
        raise ValueError("Missing or empty reasoning")
    return label, reasoning
