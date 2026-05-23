"""Stable train/test holdout: same entry stays in same fold when dataset grows."""
import sys

import numpy as np

import db
from pipeline import (
    _holdout_test_fraction,
    _stable_split_bucket,
    _stable_train_test_split,
)

PASS = FAIL = 0


def ok():
    global PASS
    PASS += 1


def fail(msg):
    global FAIL
    FAIL += 1
    print("  FAIL:", msg)


def test_bucket_stable():
    b1 = _stable_split_bucket("feed_item", 42)
    b2 = _stable_split_bucket("feed_item", "42")
    b3 = _stable_split_bucket("feed_item", 42)
    if b1 != b2 or b1 != b3:
        fail("bucket not stable for same entry")
    else:
        ok()


def test_growing_dataset_preserves_fold():
    rows = []
    labels = []
    for i in range(80):
        et = "feed_item" if i % 2 == 0 else "email"
        rows.append({"entry_type": et, "entry_id": i})
        labels.append(
            ["investigation_lead", "important", "background", "noise"][i % 4]
        )
    X = np.arange(len(rows))[:, None]
    sw = np.ones(len(rows))
    ts = _holdout_test_fraction(5, len(rows))

    _, X_test_small, _, _, _, _ = _stable_train_test_split(
        X[:60], labels[:60], sw[:60], rows[:60], test_size=ts
    )
    _, X_test_big, _, _, _, _ = _stable_train_test_split(
        X, labels, sw, rows, test_size=ts
    )
    small_test_idx = set(int(x[0]) for x in X_test_small)
    big_test_idx = set(int(x[0]) for x in X_test_big)
    if small_test_idx != (big_test_idx & set(range(60))):
        fail("fold changed for early entries: small=%s big=%s" % (
            sorted(small_test_idx)[:8], sorted(big_test_idx & set(range(60)))[:8],
        ))
    else:
        ok()


def test_each_class_represented_in_test():
    rows = []
    labels = []
    for i in range(40):
        rows.append({"entry_type": "feed_item", "entry_id": i})
        labels.append(
            ["investigation_lead", "important", "background", "noise"][i % 4]
        )
    X = np.zeros((len(rows), 2))
    sw = np.ones(len(rows))
    _, _, _, y_test, _, _ = _stable_train_test_split(
        X, labels, sw, rows, test_size=0.2
    )
    if len(y_test) < 4:
        fail("test fold too small: %d" % len(y_test))
        return
    if len(set(y_test)) < 4:
        fail("not all classes in test: %s" % set(y_test))
    else:
        ok()


if __name__ == "__main__":
    print("=== stable_split ===")
    test_bucket_stable()
    test_growing_dataset_preserves_fold()
    test_each_class_represented_in_test()
    print("Results: %d passed, %d failed" % (PASS, FAIL))
    sys.exit(1 if FAIL else 0)
