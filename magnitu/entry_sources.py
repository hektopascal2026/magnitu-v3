"""
Seismo entry types and Legislation vs News source grouping.

Legislation includes statutory lex items, parliamentary calendar events, and
feed items whose ``source_type`` is ``leg_*`` or ``parl_press``.
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
    """Return ``lex``, ``news``, or ``None`` (all)."""
    s = (source or "").strip().lower()
    if s in ("lex", "leg", "legislation"):
        return "lex"
    if s in ("news",):
        return "news"
    return None


def sql_source_filter_clause(source: Optional[str]) -> Tuple[str, List]:
    """SQL fragment + params for entries alias ``e`` (Leg vs News tabs)."""
    sf = normalize_source_filter(source)
    if sf == "lex":
        return (
            """
            AND (
                e.entry_type IN ('lex_item', 'calendar_event')
                OR (
                    e.entry_type = 'feed_item'
                    AND (
                        LOWER(COALESCE(e.source_type, '')) LIKE 'leg_%'
                        OR LOWER(COALESCE(e.source_type, '')) = 'parl_press'
                    )
                )
            )
            """,
            [],
        )
    if sf == "news":
        return (
            """
            AND (
                e.entry_type = 'email'
                OR (
                    e.entry_type = 'feed_item'
                    AND LOWER(COALESCE(e.source_type, '')) NOT LIKE 'leg_%'
                    AND LOWER(COALESCE(e.source_type, '')) != 'parl_press'
                )
            )
            """,
            [],
        )
    return "", []


def entry_matches_source_filter(entry: dict, source: Optional[str]) -> bool:
    """Python-side filter (mini client, tests)."""
    sf = normalize_source_filter(source)
    if not sf:
        return True
    et = (entry.get("entry_type") or "").strip()
    st = entry.get("source_type")
    if sf == "lex":
        if et in ("lex_item", "calendar_event"):
            return True
        return et == "feed_item" and is_leg_source_type(st)
    if sf == "news":
        if et == "email":
            return True
        if et == "feed_item":
            return not is_leg_source_type(st)
        return False
    return True
