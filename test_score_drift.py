"""WP3: pre-push score-drift EWMA tripwire."""
from unittest.mock import MagicMock, patch

import ml_window


def test_score_batch_stats_percentiles():
    scores = [{"relevance_score": float(i)} for i in range(1, 11)]
    stats = ml_window._score_batch_stats(scores)
    assert stats is not None
    assert stats["count"] == 10
    assert abs(stats["mean"] - 5.5) < 1e-9
    assert stats["p50"] == 5.5
    assert 1.0 <= stats["p10"] <= 3.0
    assert 8.0 <= stats["p90"] <= 10.0


def test_first_window_inits_baseline_without_alert():
    mock_db = MagicMock()
    mock_db.get_score_drift_baseline.return_value = None
    scores = [{"relevance_score": 0.4}, {"relevance_score": 0.5}]
    with patch.object(ml_window, "db", mock_db):
        fired = ml_window._record_push_score_drift(7, scores, False)
    assert fired is None
    mock_db.log_sync.assert_not_called()
    mock_db.upsert_score_drift_baseline.assert_called_once()
    args = mock_db.upsert_score_drift_baseline.call_args[0]
    assert args[0] == 7
    assert abs(args[1] - 0.45) < 1e-9
    assert args[2] == 1
    assert args[3] == 0


def test_mean_shift_trips_score_drift():
    mock_db = MagicMock()
    mock_db.get_score_drift_baseline.return_value = {
        "ewma_mean": 0.70,
        "window_count": 5,
        "last_rank_normalize": 0,
    }
    scores = [{"relevance_score": 0.50}] * 8
    with patch.object(ml_window, "db", mock_db):
        fired = ml_window._record_push_score_drift(1, scores, False)
    assert fired is not None
    assert fired.startswith("score_drift")
    mock_db.log_sync.assert_called_once()
    assert mock_db.log_sync.call_args[0][0] == "score_drift"
    upsert = mock_db.upsert_score_drift_baseline.call_args[0]
    assert upsert[2] == 6
    expected_ewma = (
        ml_window.SCORE_DRIFT_EWMA_ALPHA * 0.50
        + (1.0 - ml_window.SCORE_DRIFT_EWMA_ALPHA) * 0.70
    )
    assert abs(upsert[1] - expected_ewma) < 1e-9


def test_rank_normalize_flip_resets_without_false_alarm():
    mock_db = MagicMock()
    mock_db.get_score_drift_baseline.return_value = {
        "ewma_mean": 0.80,
        "window_count": 12,
        "last_rank_normalize": 1,
    }
    scores = [{"relevance_score": 0.30}] * 5
    with patch.object(ml_window, "db", mock_db):
        fired = ml_window._record_push_score_drift(2, scores, False)
    assert fired == "drift baseline re-initialized (semantics change)"
    mock_db.log_sync.assert_called_once()
    details = mock_db.log_sync.call_args[0][2]
    assert "semantics change" in details
    upsert = mock_db.upsert_score_drift_baseline.call_args[0]
    assert upsert[2] == 1
    assert abs(upsert[1] - 0.30) < 1e-9
    assert upsert[3] == 0
