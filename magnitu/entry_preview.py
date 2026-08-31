"""
Entry card preview text aligned with Seismo 0.6 timeline / score cards.

Seismo dashboard uses {@see LexCardPreview} for lex rows (synopsis + corpus
excerpt heuristics, not raw ``content``) and {@see seismo_calendar_event_body_text}
for Leg rows. Magnitu score / labeling cards should show the same snippets.
"""
from __future__ import annotations

import html
import re
from html.parser import HTMLParser
from typing import Any, Dict, Optional

TIMELINE_EXCERPT_CHARS = 8192
LEX_CARD_PREVIEW_CHARS = 300
CALENDAR_CARD_PREVIEW_CHARS = 200
FEED_CARD_PREVIEW_CHARS = 200
EMAIL_CARD_PREVIEW_CHARS = 200


class _StripTags(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self._parts.append(data)

    def get_text(self) -> str:
        return "".join(self._parts)


def _strip_tags(raw: str) -> str:
    if not raw:
        return ""
    parser = _StripTags()
    try:
        parser.feed(raw)
        parser.close()
        text = parser.get_text()
    except Exception:
        text = re.sub(r"<[^>]+>", "", raw)
    return html.unescape(text)


def _normalize_plain_text(text: str) -> str:
    text = text.replace("\xc2\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _plain_excerpt(raw: str) -> str:
    raw = (raw or "").strip()
    if not raw:
        return ""
    if "<" in raw and re.search(r"<[a-z][\s\S]*>", raw, re.IGNORECASE):
        return _normalize_plain_text(_strip_tags(raw))
    return _normalize_plain_text(raw)


def _lead(excerpt: str, max_chars: int) -> str:
    excerpt = excerpt.strip()
    if not excerpt:
        return ""
    if len(excerpt) <= max_chars:
        return excerpt
    return excerpt[:max_chars].rstrip() + "…"


def _lex_source(entry: Dict[str, Any]) -> str:
    source_type = (entry.get("source_type") or "").strip().lower()
    if source_type.startswith("lex_"):
        return source_type[4:]
    return (entry.get("source") or "").strip().lower()


def _excerpt_from_row(row: Dict[str, Any]) -> str:
    for key in ("content_excerpt", "content"):
        raw = (row.get(key) or "").strip()
        if raw:
            excerpt = _plain_excerpt(raw)
            if len(excerpt) > TIMELINE_EXCERPT_CHARS:
                excerpt = excerpt[:TIMELINE_EXCERPT_CHARS]
            return excerpt
    return ""


def _looks_like_eu_instrument_text(excerpt: str) -> bool:
    head = excerpt[:800]
    return bool(
        re.match(
            r"^(?:COMMISSION|COUNCIL|EUROPEAN PARLIAMENT|REGULATION|DIRECTIVE|DECISION)\b",
            excerpt,
            re.IGNORECASE,
        )
        or re.search(
            r"\b(?:REGULATION|DIRECTIVE|DECISION)\s*\((?:EU|EC|EEC)\)\s+\d{4}/\d+",
            head,
            re.IGNORECASE,
        )
    )


def _eu_title_block_offset(excerpt: str) -> int:
    offset = len(excerpt)
    patterns = [
        r"\nTHE EUROPEAN COMMISSION,?\s*\n",
        r"\nTHE COUNCIL(?: OF THE EUROPEAN UNION)?,?\s*\n",
        r"\nTHE EUROPEAN PARLIAMENT(?: AND OF THE COUNCIL)?,?\s*\n",
        r"\nTHE EUROPEAN PARLIAMENT AND THE COUNCIL,?\s*\n",
        r"\nHaving regard to\b",
        r"\nWhereas:\s*\n",
        r"\nWhereas\s*\n",
        r"\nDie Europäische Kommission,?\s*\n",
        r"\nLa Commission européenne,?\s*\n",
    ]
    for pattern in patterns:
        match = re.search(pattern, excerpt, re.IGNORECASE)
        if match:
            offset = min(offset, match.start())
    if offset == len(excerpt):
        match = re.search(
            r"\n(?:of|du|vom)\s+\d{1,2}\s+\w+\s+\d{4}\s*\n",
            excerpt,
            re.IGNORECASE,
        )
        if match and match.start() < 1500:
            offset = min(offset, match.end())
    return 0 if offset == len(excerpt) else offset


def _eu_body_from_excerpt(excerpt: str) -> str:
    if not _looks_like_eu_instrument_text(excerpt):
        return excerpt
    start = _eu_title_block_offset(excerpt)
    if start <= 0:
        return excerpt
    body = excerpt[start:].strip()
    return body if body and len(body) >= 40 else excerpt


def _eu_preamble(description: str, excerpt: str) -> str:
    if not excerpt:
        return description
    excerpt = _eu_body_from_excerpt(excerpt)
    cut = len(excerpt)
    patterns = [
        r"\n\s*Article\s+1\b",
        r"\n\s*Artikel\s+1\b",
        r"\n\s*Article\s+1[\.\—\-]",
        r"\n\s*Artikel\s+1[\.\—\-]",
        r"\n\s*HAVE ADOPTED\b",
        r"\n\s*HABEN FOLGENDES\b",
        r"\n\s*CHAPTER\s+I\b",
        r"\n\s*KAPITEL\s+I\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, excerpt, re.IGNORECASE)
        if match:
            cut = min(cut, match.start())
    preamble = excerpt[:cut].strip()
    if preamble and len(preamble) >= 80:
        return preamble
    return description if description else _lead(excerpt, 600)


def _looks_like_french_jorf_text(excerpt: str) -> bool:
    head = excerpt[:4000]
    return bool(
        re.search(
            r"\b(?:Assemblée nationale|promulgue la loi|Travaux préparatoires)\b",
            head,
            re.IGNORECASE,
        )
        or re.search(r"\b(?:LOI|DÉCRET|ORDONNANCE|ARRÊTÉ)\s+n[°o]\s+\d+", head, re.IGNORECASE)
    )


def _looks_like_french_jorf_boilerplate(text: str) -> bool:
    if _looks_like_french_jorf_text(text):
        return True
    return bool(
        re.search(
            r"\b(?:promulgue la loi dont la teneur suit|Assemblée nationale et le Sénat)\b",
            text,
            re.IGNORECASE,
        )
    )


def _fr_trim_travaux_preparatoires(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    match = re.search(r"\n\s*\(\d+\)\s*Travaux préparatoires\b", text, re.IGNORECASE)
    if match:
        return text[: match.start()].strip()
    match = re.search(r"\n\s*Travaux préparatoires\s*:\s*\n", text, re.IGNORECASE)
    if match:
        return text[: match.start()].strip()
    return text


def _fr_body_offset(excerpt: str) -> int:
    offset = len(excerpt)
    patterns = [
        r"\n\s*Article\s+(?:1er|premier|[1](?:[\s\.]|$))",
        r"\n\s*Chapitre\s+(?:I|1\b|premier)",
        r"\n\s*Titre\s+(?:I|1\b|premier)",
        r"\n\s*Section\s+(?:I|1\b|première)",
        r"\n\s*Partie\s+(?:I|1\b|première)",
        r"\n\s*Livre\s+(?:I|1\b|premier)",
    ]
    for pattern in patterns:
        match = re.search(pattern, excerpt, re.IGNORECASE)
        if match:
            offset = min(offset, match.start())
    if offset == len(excerpt):
        match = re.search(
            r"(?:promulgue la loi dont la teneur suit|promulgue l[\u2019']ordonnance dont la teneur suit)\s*:\s*\n",
            excerpt,
            re.IGNORECASE,
        )
        if match:
            after = match.end()
            rest = excerpt[after:]
            title_match = re.match(
                r"^(?:LOI|DÉCRET|ORDONNANCE|ARRÊTÉ)\s+n[°o]\s+\d{4}-\d+[^\n]*\n",
                rest,
                re.IGNORECASE,
            )
            if title_match:
                after += len(title_match.group(0))
            offset = min(offset, after)
    if offset == len(excerpt):
        match = re.search(
            r"\n(?:LOI|DÉCRET|ORDONNANCE|ARRÊTÉ)\s+n[°o]\s+\d{4}-\d+[^\n]*\n",
            excerpt,
            re.IGNORECASE,
        )
        if match and match.start() < 4000:
            offset = min(offset, match.end())
    return 0 if offset == len(excerpt) else offset


def _fr_body_from_excerpt(excerpt: str) -> str:
    excerpt = _fr_trim_travaux_preparatoires(excerpt)
    if not excerpt:
        return ""
    if not _looks_like_french_jorf_text(excerpt):
        return excerpt
    start = _fr_body_offset(excerpt)
    if start <= 0:
        return excerpt
    body = excerpt[start:].strip()
    return body if body and len(body) >= 40 else excerpt


def _fr_summary(description: str, excerpt: str) -> str:
    description = _fr_trim_travaux_preparatoires(_plain_excerpt(description))
    if description and not _looks_like_french_jorf_boilerplate(description):
        return description
    return _lead(_fr_body_from_excerpt(excerpt), 600)


def _looks_like_bgbl_pdf_text(excerpt: str) -> bool:
    head = excerpt[:2000]
    return bool(re.match(r"^(?:Bundesgesetzblatt\b|BGBl\.)", excerpt, re.IGNORECASE)) or (
        "Ausgegeben zu Bonn" in head
    )


def _de_bgbl_body_offset(excerpt: str) -> int:
    offset = len(excerpt)
    patterns = [
        r"\n\s*(?:Auf Grund des|Aufgrund des)\b",
        r"\n\s*Der Bundestag hat\b",
        r"\n\s*Der Bundesrat hat\b",
        r"\n\s*(?:Es verordnet|Es wird verordnet)\s*:",
        r"\n\s*Die Bevollmächtigte der Bundesregierung\b",
        r"\n\s*Die Bundesregierung verordnet\b",
        r"\n\s*Art(?:ikel|\.)?\s*1(?:[\s\.]|$)",
        r"\n\s*§\s*1(?:[\s\.]|$)",
        r"\n\s*Diese Verordnung tritt\b",
        r"\n\s*Anlage\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, excerpt, re.IGNORECASE)
        if match:
            offset = min(offset, match.start())
    if offset == len(excerpt):
        match = re.search(r"\nVom \d{1,2}\. \w+\s+\d{4}\s*\n", excerpt, re.IGNORECASE)
        if match and match.start() < 3000:
            offset = min(offset, match.end())
    return 0 if offset == len(excerpt) else offset


def _de_body_from_excerpt(excerpt: str) -> str:
    if not _looks_like_bgbl_pdf_text(excerpt):
        return excerpt
    start = _de_bgbl_body_offset(excerpt)
    if start <= 0:
        return excerpt
    body = excerpt[start:].strip()
    return body if body and len(body) >= 40 else excerpt


def _de_lead(description: str, excerpt: str) -> str:
    if description:
        return description
    return _lead(_de_body_from_excerpt(excerpt), 450)


def _default_lex_preview(description: str, excerpt: str) -> str:
    if description:
        return description
    return _lead(excerpt, 500)


LEGAL_SOURCE_TYPES = frozenset({"lex_eu", "lex_ch", "leg_eu", "leg_ch"})

# Analytical long-form sources: Substack newsletters and scraper-extracted
# articles. These routinely carry 7K-47K chars of substantive analysis. The
# default 3K cap feeds E5 only the intro/chitchat; the analytical tier gives
# them the same corpus budget as legal entries.
# SRF is excluded: its content is full broadcast subtitle transcripts (20K
# chars of "Mit Live-Untertiteln..."), not analysis. The sendungsbeschrieb
# (description, ~150 chars) is the actual signal — the 3K default cap is fine.
ANALYTICAL_SOURCE_TYPES = frozenset({"substack", "scraper"})


def is_legal_training_entry(entry: Dict[str, Any]) -> bool:
    """True for lex / Leg rows that carry long statutory body text from Seismo."""
    entry_type = (entry.get("entry_type") or "").strip()
    if entry_type in ("lex_item", "calendar_event"):
        return True
    source_type = (entry.get("source_type") or "").strip().lower()
    if source_type in LEGAL_SOURCE_TYPES:
        return True
    return source_type.startswith("lex_") or source_type.startswith("leg_")


def is_analytical_training_entry(entry: Dict[str, Any]) -> bool:
    """True for long-form analytical sources (Substack, SRF, scraper).

    These sources routinely publish 7K-47K char articles where the
    substantive analysis starts well past the 3K default cap. The
    analytical tier gives them a higher content cap and more embed
    chunks so E5 sees the analysis, not just the intro.
    """
    source_type = (entry.get("source_type") or "").strip().lower()
    return source_type in ANALYTICAL_SOURCE_TYPES


def training_corpus_text(entry: Dict[str, Any]) -> str:
    """
    Plain-text corpus for ML training / embeddings (not UI card previews).

    Merges description and content, strips HTML, and for lex corpora skips
    boilerplate headers so substantive articles reach the model.
    """
    entry_type = (entry.get("entry_type") or "").strip()

    if entry_type == "lex_item":
        description = _plain_excerpt((entry.get("description") or "").strip())
        excerpt = _excerpt_from_row(entry)
        if not excerpt and not description:
            return ""
        source = _lex_source(entry)
        body = excerpt
        if excerpt:
            if source == "eu":
                body = _eu_body_from_excerpt(excerpt)
            elif source == "fr":
                body = _fr_body_from_excerpt(excerpt)
            elif source == "de":
                body = _de_body_from_excerpt(excerpt)
        parts = []
        if description:
            parts.append(description)
        if body:
            desc_norm = description.strip()
            body_norm = body.strip()
            if not desc_norm or body_norm != desc_norm:
                if desc_norm and body_norm.startswith(desc_norm) and len(body_norm) <= len(desc_norm) + 40:
                    pass
                else:
                    parts.append(body)
        return "\n\n".join(parts).strip()

    if entry_type == "calendar_event":
        return _plain_excerpt(calendar_event_body_text(entry))

    description = _plain_excerpt((entry.get("description") or "").strip())
    content = _plain_excerpt((entry.get("content") or "").strip())
    if content and content == description:
        content = ""
    if description and content:
        return "{}\n\n{}".format(description, content)
    return content or description or ""


def lex_card_preview_text(entry: Dict[str, Any]) -> str:
    """Card body for lex rows — mirrors Seismo ``LexCardPreview::previewText``."""
    source = _lex_source(entry)
    description = (entry.get("description") or "").strip()
    row = dict(entry)
    row["source"] = source
    excerpt = _excerpt_from_row(row)

    if source == "eu":
        return _eu_preamble(description, excerpt)
    if source == "fr":
        return _fr_summary(description, excerpt)
    if source == "de":
        return _de_lead(description, excerpt)
    return _default_lex_preview(description, excerpt)


def lex_card_heading_title(entry: Dict[str, Any]) -> str:
    """Primary heading for lex cards — mirrors ``seismo_lex_card_heading_title``."""
    source = _lex_source(entry)
    title = (entry.get("title") or "").strip()
    celex = re.sub(r"\s+", "", (entry.get("celex") or "").upper())
    description = (entry.get("description") or "").strip()

    if source == "eu" and description:
        t_norm = re.sub(r"\s+", "", title.upper())
        if not title or t_norm == celex or re.match(r"^\d{4,}[A-Z][0-9A-Z]+$", title, re.IGNORECASE):
            return description
    if title:
        return title
    return (entry.get("celex") or "").strip()


def calendar_event_body_text(entry: Dict[str, Any]) -> str:
    """Display body for Leg / calendar_events cards."""
    description = _plain_excerpt((entry.get("description") or "").strip())
    body_text = _plain_excerpt((entry.get("content") or "").strip())
    if body_text == description:
        body_text = ""
    if not body_text:
        return description
    if description:
        return description + "\n\n" + body_text
    return body_text


def _feed_full_text(entry: Dict[str, Any]) -> str:
    from_content = _plain_excerpt((entry.get("content") or "").strip())
    from_desc = _plain_excerpt((entry.get("description") or "").strip())
    full = from_content or from_desc
    if not full:
        full = (entry.get("title") or "").strip()
    return full


def _email_full_text(entry: Dict[str, Any]) -> str:
    body = (entry.get("content") or "").strip()
    if not body:
        body = (entry.get("description") or "").strip()
    return _plain_excerpt(body)


def _truncate_preview(full_text: str, limit: int, flatten: bool = False) -> Dict[str, Any]:
    if not full_text:
        return {"full": "", "preview": "", "has_more": False}
    compare = re.sub(r"\s+", " ", full_text).strip() if flatten else full_text
    preview = compare[:limit]
    has_more = len(compare) > limit
    if has_more:
        preview += "..."
    return {"full": full_text, "preview": preview, "has_more": has_more}


def entry_card_body(entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Preview payload for a score / labeling card.

    Returns ``None`` when no body should render. Otherwise:
    ``full``, ``preview``, ``has_more``, and optional ``skip`` (lex title dup).
    """
    entry_type = entry.get("entry_type") or ""

    if entry_type == "lex_item":
        full = lex_card_preview_text(entry)
        if not full:
            return None
        heading = lex_card_heading_title(entry)
        result = _truncate_preview(full, LEX_CARD_PREVIEW_CHARS)
        if heading and full and heading == full:
            result["skip"] = True
        else:
            result["skip"] = False
        return result

    if entry_type == "calendar_event":
        full = calendar_event_body_text(entry)
        if not full:
            return None
        result = _truncate_preview(full, CALENDAR_CARD_PREVIEW_CHARS)
        result["skip"] = False
        return result

    if entry_type == "email":
        full = _email_full_text(entry)
        if not full:
            return None
        result = _truncate_preview(full, EMAIL_CARD_PREVIEW_CHARS, flatten=True)
        result["skip"] = False
        return result

    if entry_type == "feed_item":
        full = _feed_full_text(entry)
        if not full:
            return None
        result = _truncate_preview(full, FEED_CARD_PREVIEW_CHARS)
        result["skip"] = False
        return result

    description = (entry.get("description") or "").strip()
    content = (entry.get("content") or "").strip()
    full = description or content
    if not full:
        return None
    result = _truncate_preview(full, FEED_CARD_PREVIEW_CHARS)
    result["skip"] = False
    return result
