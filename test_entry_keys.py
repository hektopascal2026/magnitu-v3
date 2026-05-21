"""Tests for canonical entry_key() joins (int/str entry_id drift)."""
import db


def test_entry_key_coerces_numeric_strings():
    assert db.entry_key("feed_item", 42) == db.entry_key("feed_item", "42")
    assert db.entry_key("feed_item", "42")[1] == 42


def test_entry_key_strips_entry_type():
    assert db.entry_key("  feed_item ", 1) == ("feed_item", 1)


def test_entry_key_from_mapping_dict():
    row = {"entry_type": "email", "entry_id": "99"}
    assert db.entry_key_from_mapping(row) == ("email", 99)


def test_labeled_keys_match_score_row():
    score = {"entry_type": "feed_item", "entry_id": "123"}
    label = {"entry_type": "feed_item", "entry_id": 123}
    assert db.entry_key_from_mapping(score) == db.entry_key_from_mapping(label)


if __name__ == "__main__":
    test_entry_key_coerces_numeric_strings()
    test_entry_key_strips_entry_type()
    test_entry_key_from_mapping_dict()
    test_labeled_keys_match_score_row()
    print("test_entry_keys: ok")
