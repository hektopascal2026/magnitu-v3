"""
Tests for the P0-3 recent-holdout exclusion cap in pipeline._train_transformer.

The exclusion caps at ``len(labeled) - min_labels`` so a desk with 21..119
labels does not exclude all of them (top-100 by created_at covers everything)
and fail to train.  Before the fix, training was impossible in the 21..119
label range — a months-long dead zone after cold start at 15-20 labels.
"""
import pytest
from unittest.mock import patch, MagicMock

import pipeline


def _make_labels(n):
    """N fake label dicts with unique entry keys."""
    return [
        {"entry_type": "feed_item", "entry_id": i, "label": "background"}
        for i in range(n)
    ]


def _setup_db_mock(mock_db, n_labels, min_labels=20):
    """Configure the pipeline.db mock for _train_transformer."""
    mock_db.get_effective_config.return_value = {
        "min_labels_to_train": min_labels,
        "embedding_dim": 768,
        "embedding_l2_normalize": False,
        "classifier_c": 0.01,
        "classifier_apply_prior": False,
        "label_time_decay_days": 0,
        "reasoning_weight_boost": 1.0,
        "synthetic_label_weight": 1.0,
    }
    mock_db.get_all_labels.return_value = _make_labels(n_labels)
    mock_db.get_next_model_version.return_value = 1
    mock_db.save_model_record.return_value = 1
    mock_db.entry_key_from_mapping.side_effect = lambda m: (m["entry_type"], m["entry_id"])
    mock_db.get_eval_reserve_keys.return_value = set()
    mock_db.roll_eval_reserve.return_value = {
        "before": 0, "admitted": 0, "after": 0, "target": 100,
    }

    holdout_limits = []

    conn = MagicMock()
    mock_db.get_db.return_value = conn

    def execute(sql, params=()):
        if "ORDER BY l.created_at DESC" in sql:
            limit = params[1] if len(params) > 1 else 0
            holdout_limits.append(limit)
            return [
                {"entry_type": "feed_item", "entry_id": i}
                for i in range(limit)
            ]
        if "embedding IS NOT NULL" in sql:
            return []
        return MagicMock()
    conn.execute.side_effect = execute

    return holdout_limits


@patch("pipeline.db")
@patch("pipeline.get_config")
def test_recent_holdout_cap_prevents_dead_zone_50_labels(mock_config, mock_db):
    """With 50 labels (min_labels=20, GATE_N_RECENT=100), the exclusion
    must cap at 50-20=30, not 100.  Before the fix it excluded all 50
    and training failed with 'need 20 labels'.
    """
    mock_config.return_value = {
        "embedding_l2_normalize": False, "classifier_c": 0.01,
        "classifier_apply_prior": False, "label_time_decay_days": 0,
        "reasoning_weight_boost": 1.0, "synthetic_label_weight": 1.0,
        "legal_signal_patterns": [],
    }
    holdout_limits = _setup_db_mock(mock_db, 50, min_labels=20)

    # Training will fail later (no embeddings mocked), but we only need
    # to verify the holdout query LIMIT was capped correctly.
    try:
        pipeline._train_transformer(
            profile_id=1, activate=False, recent_holdout_n=100,
        )
    except Exception:
        pass  # expected — no embeddings / model mocking

    assert holdout_limits, "recent-holdout query was not issued"
    assert holdout_limits[0] == 30, \
        "exclusion must cap at len(labels)-min_labels=30, got %d" % holdout_limits[0]


@patch("pipeline.db")
@patch("pipeline.get_config")
def test_recent_holdout_skipped_when_below_min_labels(mock_config, mock_db):
    """With 18 labels (below min_labels=20), no exclusion happens at all.
    This is the cold-start path: train on all 18, gate is in-sample.
    """
    mock_config.return_value = {}
    holdout_limits = _setup_db_mock(mock_db, 18, min_labels=20)

    try:
        pipeline._train_transformer(
            profile_id=1, activate=False, recent_holdout_n=100,
        )
    except Exception:
        pass  # expected

    assert not holdout_limits, \
        "recent-holdout query must not run when len(labels) <= min_labels"


@patch("pipeline.db")
@patch("pipeline.get_config")
def test_recent_holdout_full_exclusion_above_120_labels(mock_config, mock_db):
    """With 150 labels (min_labels=20, GATE_N_RECENT=100), the exclusion
    uses the full 100 — the cap (150-20=130) doesn't limit it.
    """
    mock_config.return_value = {}
    holdout_limits = _setup_db_mock(mock_db, 150, min_labels=20)

    try:
        pipeline._train_transformer(
            profile_id=1, activate=False, recent_holdout_n=100,
        )
    except Exception:
        pass  # expected

    assert holdout_limits, "recent-holdout query was not issued"
    assert holdout_limits[0] == 100, \
        "exclusion must use full GATE_N_RECENT=100 when cap allows it, got %d" % holdout_limits[0]


@patch("pipeline.db")
@patch("pipeline.get_config")
def test_recent_holdout_cap_at_boundary_21_labels(mock_config, mock_db):
    """At exactly 21 labels (min_labels=20), the cap is 1 — only 1 row
    excluded, 20 remain for training.  Before the fix, all 21 were
    excluded and training failed.
    """
    mock_config.return_value = {}
    holdout_limits = _setup_db_mock(mock_db, 21, min_labels=20)

    try:
        pipeline._train_transformer(
            profile_id=1, activate=False, recent_holdout_n=100,
        )
    except Exception:
        pass  # expected

    assert holdout_limits, "recent-holdout query was not issued"
    assert holdout_limits[0] == 1, \
        "at 21 labels with min=20, cap must be 1, got %d" % holdout_limits[0]
