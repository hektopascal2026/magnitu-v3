"""
Seismo timeline source-pill label text (see docs/magnitu-entry-pills.md).

Returns human-readable strings only — Magnitu keeps its own CSS classes.
"""
from typing import Dict, List, Optional

_PILL_LABEL_MAX = 32

_CALENDAR_EVENT_TYPE_LABELS = {
    "session": "Session",
    "Motion": "Motion",
    "Postulat": "Postulat",
    "Interpellation": "Interpellation",
    "Dringliche Interpellation": "Interpellation",
    "Einfache Anfrage": "Anfrage",
    "Dringliche Einfache Anfrage": "Anfrage",
    "Parlamentarische Initiative": "Parl. Initiative",
    "Standesinitiative": "Standesinitiative",
    "Geschaeft des Bundesrates": "Bundesratsgeschäft",
    "Geschäft des Bundesrates": "Bundesratsgeschäft",
    "Geschaeft des Parlaments": "Parlamentsgeschäft",
    "Geschäft des Parlaments": "Parlamentsgeschäft",
    "Petition": "Petition",
    "Empfehlung": "Empfehlung",
    "Fragestunde. Frage": "Fragestunde",
}

_COUNCIL_LABELS = {
    "NR": "Nationalrat",
    "SR": "Ständerat",
    "BR": "Bundesrat",
}

_LEX_JURISDICTION = {
    "ch_bger": ("⚖️", "BGer"),
    "ch_bge": ("⚖️", "BGE"),
    "ch_bvger": ("⚖️", "BVGer"),
    "de": ("🇩🇪", "DE"),
    "ch": ("🇨🇭", "CH"),
    "fr": ("🇫🇷", "FR"),
}


def entry_pill_texts(entry: dict) -> List[str]:
    """0..n pill labels matching Seismo dashboard logic."""
    et = (entry.get("entry_type") or "").strip()
    if et == "feed_item":
        return _feed_item_pills(entry)
    if et == "email":
        return _email_pills(entry)
    if et == "lex_item":
        return _lex_item_pills(entry)
    if et == "calendar_event":
        return _calendar_event_pills(entry)
    return []


def _truncate_label(text: str, max_len: int = _PILL_LABEL_MAX) -> str:
    s = (text or "").strip()
    if len(s) <= max_len:
        return s
    return s[:max_len] + "…"


def _label_from_category_or_title(category: Optional[str], title: Optional[str]) -> str:
    c = (category or "").strip()
    if c and c.lower() != "unsortiert":
        return c
    return (title or "").strip()


def _lex_source_key(entry: dict) -> str:
    st = (entry.get("source_type") or "").strip().lower()
    if st.startswith("lex_"):
        return st[4:]
    return st


def _is_parl_swiss_lex(entry: dict, lex_source: str) -> bool:
    if lex_source in ("parl_mm", "parl_sda"):
        return True
    link = (entry.get("link") or "").strip().lower()
    title = (entry.get("title") or "").strip().lower()
    for hint in (link, title):
        if hint.startswith("parl_mm:") or hint.startswith("parl_sda:"):
            return True
    return False


def _parl_press_is_sda(entry: dict) -> bool:
    cat = (entry.get("source_category") or "").strip().lower()
    if cat == "parl_sda":
        return True
    link = (entry.get("link") or "").strip().lower()
    if link.startswith("parl_sda:"):
        return True
    return False


def _parl_press_meta_label(entry: dict, is_sda: bool) -> str:
    if is_sda:
        return "Session"
    cat = (entry.get("source_category") or "").strip()
    if cat and cat.lower() not in ("unsortiert", "parl_sda", "parl_mm"):
        return cat
    return "Medienmitteilung"


def _feed_item_pills(entry: dict) -> List[str]:
    st = (entry.get("source_type") or "").strip().lower()
    if st == "parl_press":
        is_sda = _parl_press_is_sda(entry)
        primary = "🇨🇭 Parl SDA" if is_sda else "🇨🇭 Parl MM"
        return [primary, _parl_press_meta_label(entry, is_sda)]
    if st == "scraper":
        name = (entry.get("source_name") or "").strip() or "Scraper"
        return ["🌐 " + name]
    label = _truncate_label(
        _label_from_category_or_title(
            entry.get("source_category"),
            entry.get("source_name"),
        )
    )
    if not label:
        return []
    return [label]


def _email_pills(entry: dict) -> List[str]:
    tag = (entry.get("source_category") or "").strip()
    name = (entry.get("source_name") or "").strip()
    if tag.lower() == "unclassified" and not name:
        return []
    text = name if name else tag
    return [text] if text else []


def _lex_item_pills(entry: dict) -> List[str]:
    lex_source = _lex_source_key(entry)
    doc_type = (entry.get("source_category") or "").strip() or "Legislation"
    if lex_source == "eu" or _is_parl_swiss_lex(entry, lex_source):
        mark = "🇨🇭 CH" if _is_parl_swiss_lex(entry, lex_source) else "🇪🇺 EU"
        return [mark, doc_type]
    if lex_source == "ch":
        return ["🇨🇭 CH", doc_type]
    emoji, short = _LEX_JURISDICTION.get(lex_source, ("🇪🇺", "EU"))
    return ["%s %s" % (emoji, short), doc_type]


def _calendar_event_type_label(event_type: Optional[str]) -> str:
    raw = (event_type or "").strip()
    if not raw:
        return "Event"
    return _CALENDAR_EVENT_TYPE_LABELS.get(raw, raw)


def _council_label(code: Optional[str]) -> str:
    c = (code or "").strip()
    if not c:
        return ""
    return _COUNCIL_LABELS.get(c, c)


def _calendar_event_pills(entry: dict) -> List[str]:
    pills = [_calendar_event_type_label(entry.get("source_category"))]
    council = _council_label(entry.get("author"))
    if council:
        pills.append(council)
    return pills
