"""Unit tests for memory-bounded recipe distillation sampling."""
import numpy as np

from pipeline import _sample_entries_for_distillation


def test_sample_under_cap_returns_all():
    entries = [{"entry_type": "feed_item", "entry_id": i} for i in range(5)]
    out = _sample_entries_for_distillation(
        entries, labeled_keys=set(), max_entries=8000, rng=np.random.RandomState(1),
    )
    assert out == entries


def test_sample_zero_cap_returns_all():
    entries = [{"entry_type": "feed_item", "entry_id": i} for i in range(5)]
    out = _sample_entries_for_distillation(
        entries, labeled_keys=set(), max_entries=0, rng=np.random.RandomState(1),
    )
    assert out == entries


def test_sample_prefers_all_labels_then_fills():
    entries = [{"entry_type": "feed_item", "entry_id": i} for i in range(30)]
    labeled = {("feed_item", i) for i in range(10)}
    out = _sample_entries_for_distillation(
        entries, labeled_keys=labeled, max_entries=15, rng=np.random.RandomState(2),
    )
    assert len(out) == 15
    keys = {(e["entry_type"], e["entry_id"]) for e in out}
    assert labeled.issubset(keys)


def test_sample_when_labels_exceed_cap():
    entries = [{"entry_type": "feed_item", "entry_id": i} for i in range(20)]
    labeled = {("feed_item", i) for i in range(20)}
    out = _sample_entries_for_distillation(
        entries, labeled_keys=labeled, max_entries=5, rng=np.random.RandomState(3),
    )
    assert len(out) == 5
    assert all(("feed_item", e["entry_id"]) in labeled for e in out)


def test_sample_works_on_key_only_rows():
    """ML window samples keys before loading full text bodies."""
    entries = [{"entry_type": "feed_item", "entry_id": i} for i in range(50)]
    labeled = {("feed_item", 1), ("feed_item", 2)}
    out = _sample_entries_for_distillation(
        entries, labeled_keys=labeled, max_entries=10, rng=np.random.RandomState(4),
    )
    assert len(out) == 10
    keys = {(e["entry_type"], e["entry_id"]) for e in out}
    assert labeled.issubset(keys)
