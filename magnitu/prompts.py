"""Magnitu Gemini prompts and response schemas for synthetic labeling.

Interview extraction templates live in transcribe-v3 ``mac/v3/extraction/prompts.py``;
this module is Magnitu-specific (four label classes + Swiss trade persona).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

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
