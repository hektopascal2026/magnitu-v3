import pytest
from unittest.mock import patch, MagicMock
import ml_window

def test_rank_normalize_push_scores():
    scores = [
        {"id": 1, "relevance_score": 0.1},
        {"id": 2, "relevance_score": 0.9},
        {"id": 3, "relevance_score": 0.1},
        {"id": 4, "relevance_score": 0.5},
    ]
    # Ranks will be:
    # 0.1 and 0.1 (ties at rank 1 and 2, mean rank = 1.5) -> 1.5 / 4 = 0.375
    # 0.5 (rank 3) -> 3 / 4 = 0.75
    # 0.9 (rank 4) -> 4 / 4 = 1.0
    normalized = ml_window._rank_normalize_push_scores(scores)
    
    id_to_score = {s["id"]: s["relevance_score"] for s in normalized}
    assert id_to_score[1] == 0.375
    assert id_to_score[3] == 0.375
    assert id_to_score[4] == 0.75
    assert id_to_score[2] == 1.0

def test_empty_or_single_score():
    assert ml_window._rank_normalize_push_scores([]) == []
    assert ml_window._rank_normalize_push_scores([{"id": 1, "relevance_score": 0.5}]) == [{"id": 1, "relevance_score": 0.5}]


def test_should_promote_cold_start():
    assert ml_window._should_promote(None, {"precision_at_30": 0.1, "f1_score": 0.2})


def test_should_promote_p30_path():
    old = {"precision_at_30": 0.5, "f1_score": 0.5}
    assert ml_window._should_promote(old, {"precision_at_30": 0.52, "f1_score": 0.5})
    assert not ml_window._should_promote(old, {"precision_at_30": 0.505, "f1_score": 0.5})
    # Clear p@30 win with small F1 dip inside slack (live EU v6-like)
    assert ml_window._should_promote(
        {"precision_at_30": 0.133, "f1_score": 0.322},
        {"precision_at_30": 0.192, "f1_score": 0.274},
    )


def test_should_promote_f1_path_with_ranking_slack():
    # Live EU-like: p@30 dipped within slack, F1 clearly up → promote
    old = {"precision_at_30": 0.133, "f1_score": 0.322}
    assert ml_window._should_promote(
        old, {"precision_at_30": 0.111, "f1_score": 0.405}
    )
    # Equal p@30, better F1 → promote (was rejected under strict p@30-only)
    assert ml_window._should_promote(
        {"precision_at_30": 0.2, "f1_score": 0.3},
        {"precision_at_30": 0.2, "f1_score": 0.4},
    )
    # Ranking collapsed beyond slack → reject even if F1 up a bit
    assert not ml_window._should_promote(
        {"precision_at_30": 0.5, "f1_score": 0.4},
        {"precision_at_30": 0.4, "f1_score": 0.42},
    )

@patch("ml_window.os.path.exists")
@patch("ml_window.open")
@patch("ml_window.json.load")
@patch("ml_window.db")
@patch("ml_window.sync")
@patch("ml_window.pipeline")
@patch("ml_window.distiller")
@patch("ml_window.export_model")
def test_ml_window_main_promotes(mock_export, mock_distiller, mock_pipeline, mock_sync, mock_db, mock_json_load, mock_open, mock_exists, monkeypatch):
    monkeypatch.setenv("SEISMO_DESKS_JSON", '[{"seismo_url": "foo", "api_key": "bar"}]')
    monkeypatch.setenv("MAGNITU_VAULT_PASSWORD", "vault-secret")
    
    # Mock config
    ml_window.get_config = MagicMock(return_value={"seismo_url": "mother", "api_key": "secret"})
    
    # Mock db
    mock_db.slugify.return_value = "foo"
    mock_db.get_profile_by_slug.return_value = {"id": 1}
    mock_db.get_all_labels.return_value = [1] * 20
    
    # Database connection mock
    mock_conn = MagicMock()
    mock_db.get_db.return_value = mock_conn
    mock_conn.execute.return_value.fetchone.return_value = [20]
    
    # Mock model
    mock_db.get_active_model.side_effect = [
        {"trained_at": "2020-01-01T00:00:00Z", "precision_at_30": 0.5, "f1_score": 0.5, "version": 1}, # before train
        {"trained_at": "2020-01-01T00:00:00Z", "precision_at_30": 0.5, "f1_score": 0.5, "version": 2, "recipe_path": "recipe.json"}, # after promote
        {"trained_at": "2020-01-01T00:00:00Z", "precision_at_30": 0.5, "f1_score": 0.5, "version": 2, "recipe_path": "recipe.json"}  # for score pushing
    ]
    
    # Mock train returns success + better metrics
    mock_pipeline.train.return_value = {
        "success": True,
        "precision_at_30": 0.6,
        "f1_score": 0.6,
        "version": 2
    }
    
    # Mock sync compute pending (1st call returns 1, 2nd call returns 0)
    mock_sync._compute_pending_embeddings.side_effect = [1, 0]
    
    # Mock get_recent_entries
    mock_db.get_recent_entries.return_value = [{"id": 1}]
    mock_pipeline.score_entries.return_value = [{"id": 1, "relevance_score": 0.5}]
    
    mock_exists.return_value = True
    
    # Run
    res = ml_window.main()
    
    assert res == 0
    mock_pipeline.train.assert_called_once_with(profile_id=1, activate=False)
    mock_distiller.distill_recipe.assert_called_once_with(profile_id=1)
    mock_export.assert_called_once_with(profile_id=1)
    mock_sync.vault_upload.assert_called_once()
    mock_sync.push_scores.assert_called_once()
    
    # Ensure compute embeddings looped twice
    assert mock_sync._compute_pending_embeddings.call_count == 2
    mock_sync.vault_upload.assert_called_once_with(vault_password="vault-secret", package_path=mock_export.return_value, overwrite=True)

@patch("ml_window.db")
@patch("ml_window.sync")
@patch("ml_window.pipeline")
def test_ml_window_main_rejects(mock_pipeline, mock_sync, mock_db, monkeypatch):
    monkeypatch.setenv("SEISMO_DESKS_JSON", '[{"seismo_url": "foo", "api_key": "bar"}]')
    monkeypatch.setenv("MAGNITU_VAULT_PASSWORD", "vault-secret")
    
    # Mock config
    ml_window.get_config = MagicMock(return_value={"seismo_url": "mother", "api_key": "secret"})
    
    # Mock db
    mock_db.slugify.return_value = "foo"
    mock_db.get_profile_by_slug.return_value = {"id": 1}
    mock_db.get_all_labels.return_value = [1] * 20
    
    # Database connection mock
    mock_conn = MagicMock()
    mock_db.get_db.return_value = mock_conn
    mock_conn.execute.return_value.fetchone.return_value = [20]
    
    # Mock model
    mock_db.get_active_model.return_value = {"trained_at": "2020-01-01T00:00:00Z", "precision_at_30": 0.5, "f1_score": 0.5, "version": 1}
    
    # Mock train returns success but worse metrics -> REJECT
    mock_pipeline.train.return_value = {
        "success": True,
        "precision_at_30": 0.4, # worse
        "f1_score": 0.4,
        "version": 2
    }
    
    # Mock sync compute pending
    mock_sync._compute_pending_embeddings.side_effect = [0]
    
    # Mock get_recent_entries
    mock_db.get_recent_entries.return_value = []
    
    # Run
    res = ml_window.main()
    
    assert res == 0
    mock_pipeline.train.assert_called_once_with(profile_id=1, activate=False)
    mock_db.log_sync.assert_called_once_with("train_rejected", 1, "Kept older model, new version 2 rejected.", profile_id=1)
    mock_sync.vault_upload.assert_not_called()
