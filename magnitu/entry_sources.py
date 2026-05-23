"""
Seismo entry types and Label-page source tabs.

Legislation tab: ``lex_item`` and ``calendar_event`` (Lex / Leg) only.
Email tab: ``email`` entries only. RSS/substack feeds appear under All.
"""
from typing import List, Optional, Tuple

# (Seismo magnitu_entries ``type`` param, magnitu_status count key)
SEISMO_ENTRY_PULL_SPECS = (
    ("feed_item", "feed_items"),
    ("email", "emails"),
    ("lex_item", "lex_items"),
    ("calendar_event", "calendar_events"),
)

SEISMO_ENTRY_TYPES = tuple(spec[0] for spec in SEISMO_ENTRY_PULL_SPECS)


def is_leg_source_type(source_type: Optional[str]) -> bool:
    st = (source_type or "").strip().lower()
    if not st:
        return False
    return st.startswith("leg_") or st == "parl_press"


def normalize_source_filter(source: Optional[str]) -> Optional[str]:
    """Return ``lex``, ``email``, or ``None`` (all)."""
    s = (source or "").strip().lower()
    if s in ("lex", "leg", "legislation"):
        return "lex"
    if s in ("email", "emails"):
        return "email"
    return None


def sql_source_filter_clause(source: Optional[str]) -> Tuple[str, List]:
    """SQL fragment + params for entries alias ``e`` (Label page tabs)."""
    sf = normalize_source_filter(source)
    if sf == "lex":
        return (
            " AND e.entry_type IN ('lex_item', 'calendar_event') ",
            [],
        )
    if sf == "email":
        return (
            " AND e.entry_type = 'email' ",
            [],
        )
    return "", []


def entry_matches_source_filter(entry: dict, source: Optional[str]) -> bool:
    """Python-side filter (mini client, tests)."""
    sf = normalize_source_filter(source)
    if not sf:
        return True
    et = (entry.get("entry_type") or "").strip()
    if sf == "lex":
        return et in ("lex_item", "calendar_event")
    if sf == "email":
        return et == "email"
    return True
