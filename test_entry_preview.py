"""Tests for Seismo-aligned entry card preview text."""
from magnitu.entry_preview import (
    calendar_event_body_text,
    entry_card_body,
    lex_card_preview_text,
)


def test_lex_prefers_description_over_raw_content():
    entry = {
        "entry_type": "lex_item",
        "source_type": "lex_ch",
        "description": "Bundesgesetz über die Krankenversicherung",
        "content": "Art. 1\nGegenstand\nDieses Gesetz regelt ..." * 50,
    }
    assert lex_card_preview_text(entry) == "Bundesgesetz über die Krankenversicherung"
    body = entry_card_body(entry)
    assert body is not None
    assert body["full"] == "Bundesgesetz über die Krankenversicherung"
    assert body["has_more"] is False


def test_lex_de_uses_description_not_bgbl_corpus():
    entry = {
        "entry_type": "lex_item",
        "source_type": "lex_de",
        "description": "Verordnung summary of the act",
        "content": "Bundesgesetzblatt\nTeil I\nAusgegeben zu Bonn\n\nAuf Grund des ...",
    }
    assert lex_card_preview_text(entry) == "Verordnung summary of the act"


def test_calendar_merges_description_and_content():
    long_body = "Eingereichter Text des Vorstosses. " * 12
    entry = {
        "entry_type": "calendar_event",
        "description": "Ausgangslage",
        "content": long_body.strip(),
    }
    full = calendar_event_body_text(entry)
    assert full.startswith("Ausgangslage\n\n")
    body = entry_card_body(entry)
    assert body is not None
    assert body["preview"].endswith("...")
    assert body["has_more"] is True


def test_feed_uses_content_when_present():
    entry = {
        "entry_type": "feed_item",
        "title": "Headline",
        "description": "RSS summary",
        "content": "Full article body text here",
    }
    body = entry_card_body(entry)
    assert body is not None
    assert body["full"] == "Full article body text here"


def test_lex_skips_preview_when_same_as_heading():
    text = "Same title and body synopsis"
    entry = {
        "entry_type": "lex_item",
        "source_type": "lex_ch",
        "title": text,
        "description": text,
        "content": "",
    }
    body = entry_card_body(entry)
    assert body is not None
    assert body["skip"] is True
