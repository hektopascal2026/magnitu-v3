"""Entry pill label text (Seismo parity)."""
import sys

from magnitu.entry_pills import entry_pill_texts

PASS = FAIL = 0


def ok():
    global PASS
    PASS += 1


def fail(msg):
    global FAIL
    FAIL += 1
    print("  FAIL:", msg)


def test_feed_rss_category():
    pills = entry_pill_texts({
        "entry_type": "feed_item",
        "source_type": "rss",
        "source_category": "NZZ",
        "source_name": "Neue Zürcher Zeitung",
    })
    if pills != ["NZZ"]:
        fail("rss category: %r" % pills)
    else:
        ok()


def test_feed_rss_title_when_unsortiert():
    pills = entry_pill_texts({
        "entry_type": "feed_item",
        "source_type": "rss",
        "source_category": "unsortiert",
        "source_name": "Long Feed Title Here",
    })
    if pills != ["Long Feed Title Here"]:
        fail("unsortiert fallback: %r" % pills)
    else:
        ok()


def test_scraper():
    pills = entry_pill_texts({
        "entry_type": "feed_item",
        "source_type": "scraper",
        "source_name": "My Scraper",
    })
    if pills != ["🌐 My Scraper"]:
        fail("scraper: %r" % pills)
    else:
        ok()


def test_parl_press_mm():
    pills = entry_pill_texts({
        "entry_type": "feed_item",
        "source_type": "parl_press",
        "source_category": "WK",
    })
    if pills != ["🇨🇭 Parl MM", "WK"]:
        fail("parl mm: %r" % pills)
    else:
        ok()


def test_parl_press_sda():
    pills = entry_pill_texts({
        "entry_type": "feed_item",
        "source_type": "parl_press",
        "source_category": "parl_sda",
    })
    if pills != ["🇨🇭 Parl SDA", "Session"]:
        fail("parl sda: %r" % pills)
    else:
        ok()


def test_email_no_pill():
    pills = entry_pill_texts({
        "entry_type": "email",
        "source_type": "email",
        "source_category": "unclassified",
        "source_name": "",
    })
    if pills != []:
        fail("email empty: %r" % pills)
    else:
        ok()


def test_email_name():
    pills = entry_pill_texts({
        "entry_type": "email",
        "source_type": "email",
        "source_category": "unclassified",
        "source_name": "Admin.ch",
    })
    if pills != ["Admin.ch"]:
        fail("email name: %r" % pills)
    else:
        ok()


def test_lex_eu():
    pills = entry_pill_texts({
        "entry_type": "lex_item",
        "source_type": "lex_eu",
        "source_category": "Directive",
    })
    if pills != ["🇪🇺 EU", "Directive"]:
        fail("lex eu: %r" % pills)
    else:
        ok()


def test_lex_ch_fedlex():
    pills = entry_pill_texts({
        "entry_type": "lex_item",
        "source_type": "lex_ch",
        "source_category": "Ordonnance",
    })
    if pills != ["🇨🇭 CH", "Ordonnance"]:
        fail("lex ch: %r" % pills)
    else:
        ok()


def test_lex_bger():
    pills = entry_pill_texts({
        "entry_type": "lex_item",
        "source_type": "lex_ch_bger",
        "source_category": "Urteil",
    })
    if pills != ["⚖️ BGer", "Urteil"]:
        fail("lex bger: %r" % pills)
    else:
        ok()


def test_calendar_event():
    pills = entry_pill_texts({
        "entry_type": "calendar_event",
        "source_type": "leg_parliament_ch",
        "source_category": "Motion",
        "author": "NR",
    })
    if pills != ["Motion · NR"]:
        fail("calendar: %r" % pills)
    else:
        ok()


if __name__ == "__main__":
    print("=== entry_pills ===")
    test_feed_rss_category()
    test_feed_rss_title_when_unsortiert()
    test_scraper()
    test_parl_press_mm()
    test_parl_press_sda()
    test_email_no_pill()
    test_email_name()
    test_lex_eu()
    test_lex_ch_fedlex()
    test_lex_bger()
    test_calendar_event()
    print("Results: %d passed, %d failed" % (PASS, FAIL))
    sys.exit(1 if FAIL else 0)
