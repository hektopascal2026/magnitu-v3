import math
import pytest
from unittest.mock import patch, MagicMock
import ml_window


def test_is_oom_kill_returncode():
    assert ml_window._is_oom_kill_returncode(137)
    assert ml_window._is_oom_kill_returncode(-9)
    assert not ml_window._is_oom_kill_returncode(1)
    assert not ml_window._is_oom_kill_returncode(0)


def test_sort_prepared_desks_unbaked_first():
    prepared = [
        {"slug": "eu", "do_train": False, "report": {"labels_since_train": 0}},
        {"slug": "digital", "do_train": True, "report": {"labels_since_train": 15}},
        {"slug": "sicherheit", "do_train": True, "report": {"labels_since_train": 20}},
        {"slug": "seismo", "do_train": False, "report": {"labels_since_train": 3}},
    ]
    ordered = ml_window._sort_prepared_desks(prepared)
    assert [d["slug"] for d in ordered] == ["sicherheit", "digital", "seismo", "eu"]


def test_should_promote_cold_start():
    assert ml_window._should_promote(None, {"precision_at_30": 0.1, "f1_score": 0.2})


def test_should_promote_p30_path():
    old = {"precision_at_30": 0.5, "f1_score": 0.5}
    assert ml_window._should_promote(old, {"precision_at_30": 0.52, "f1_score": 0.5})
    assert not ml_window._should_promote(old, {"precision_at_30": 0.505, "f1_score": 0.5})
    # EU-like p@30 win with tiny F1 dip inside PROMOTE_MARGIN
    assert ml_window._should_promote(
        {"precision_at_30": 0.133, "f1_score": 0.322},
        {"precision_at_30": 0.192, "f1_score": 0.315},
    )
    # Digital-like: p@30 up but F1 drops beyond margin → reject
    assert not ml_window._should_promote(
        {"precision_at_30": 0.167, "f1_score": 0.415},
        {"precision_at_30": 0.200, "f1_score": 0.382},
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


def test_evaluate_model_update_cold_start():
    assert ml_window.evaluate_model_update(None, {"precision_at_30": 0.1, "f1_score": 0.2})


def test_evaluate_model_update_incident_promotes():
    assert ml_window.evaluate_model_update(
        {"precision_at_30": 0.167, "f1_score": 0.415},
        {"precision_at_30": 0.300, "f1_score": 0.335},
    )


def test_evaluate_model_update_catastrophe_cap_rejects():
    old = {"precision_at_30": 0.167, "f1_score": 0.415}
    assert not ml_window.evaluate_model_update(
        old, {"precision_at_30": 0.300, "f1_score": 0.295}
    )


def test_evaluate_model_update_big_win_boundaries():
    old = {
        "precision_at_30": 0.0,
        "f1_score": ml_window.F1_HARD_DROP_LIMIT,
    }
    # Gain exactly 0.05 and dip exactly -0.10 both promote.
    assert ml_window.evaluate_model_update(
        old,
        {
            "precision_at_30": ml_window.PROMOTE_BIG_P30_WIN,
            "f1_score": 0.0,
        },
    )
    assert not ml_window.evaluate_model_update(
        old,
        {
            "precision_at_30": math.nextafter(ml_window.PROMOTE_BIG_P30_WIN, 0.0),
            "f1_score": 0.0,
        },
    )
    assert not ml_window.evaluate_model_update(
        old,
        {
            "precision_at_30": ml_window.PROMOTE_BIG_P30_WIN,
            "f1_score": -1e-6,
        },
    )


def test_evaluate_model_update_lead_recall_veto_big_win():
    assert not ml_window.evaluate_model_update(
        {
            "precision_at_30": 0.167,
            "f1_score": 0.415,
            "lead_recall_at_30": 0.8,
        },
        {
            "precision_at_30": 0.300,
            "f1_score": 0.335,
            "lead_recall_at_30": 0.4,
        },
    )


def test_evaluate_model_update_lead_recall_veto_legacy_f1_path():
    assert not ml_window.evaluate_model_update(
        {"precision_at_30": 0.20, "f1_score": 0.30, "lead_recall_at_30": 0.8},
        {"precision_at_30": 0.20, "f1_score": 0.40, "lead_recall_at_30": 0.4},
    )


def test_train_reject_log_names_lead_recall_crater():
    old = {"precision_at_30": 0.20, "f1_score": 0.30, "lead_recall_at_30": 0.8}
    new = {"precision_at_30": 0.20, "f1_score": 0.40, "lead_recall_at_30": 0.4}
    msg = ml_window._train_reject_log(old, new)
    assert "lead_recall_at_30 crater" in msg
    assert "0.800" in msg and "0.400" in msg
    assert ml_window._train_reject_log(
        {"precision_at_30": 0.5, "f1_score": 0.5},
        {"precision_at_30": 0.4, "f1_score": 0.4},
    ) == "Model rejected. Keeping older model."


def test_evaluate_model_update_lead_recall_guard_skipped_when_missing_or_zero():
    incident_old = {"precision_at_30": 0.167, "f1_score": 0.415}
    incident_new = {"precision_at_30": 0.300, "f1_score": 0.335}
    assert ml_window.evaluate_model_update(incident_old, incident_new)
    assert ml_window.evaluate_model_update(
        dict(incident_old, lead_recall_at_30=0.0),
        dict(incident_new, lead_recall_at_30=0.4),
    )
    assert ml_window.evaluate_model_update(
        incident_old,
        dict(incident_new, lead_recall_at_30=0.4),
    )


@patch("ml_window.os.path.exists")
@patch("ml_window.open")
@patch("ml_window.json.load")
@patch("ml_window.db")
@patch("ml_window.sync")
@patch("ml_window.pipeline")
@patch("ml_window._distill_recipe_in_subprocess", return_value=0)
@patch("ml_window.export_model")
def test_ml_window_main_promotes(
    mock_export,
    mock_distill_sub,
    mock_pipeline,
    mock_sync,
    mock_db,
    mock_json_load,
    mock_open,
    mock_exists,
    monkeypatch,
):
    monkeypatch.setenv("SEISMO_DESKS_JSON", '[{"seismo_url": "foo", "api_key": "bar"}]')
    monkeypatch.setenv("MAGNITU_VAULT_PASSWORD", "vault-secret")
    
    # Mock config
    ml_window.get_config = MagicMock(return_value={"seismo_url": "mother", "api_key": "secret"})
    
    # Mock db
    mock_db.slugify.return_value = "foo"
    mock_db.derive_profile_identity_from_push_url.return_value = ("Foo", "foo")
    mock_db.get_profile_by_slug.return_value = {"id": 1}
    mock_db.get_all_labels.return_value = [1] * 20
    mock_db.get_model_profile.return_value = {
        "model_name": "Foo",
        "model_uuid": "uuid-1",
        "description": "desk",
    }

    # Database connection mock
    mock_conn = MagicMock()
    mock_db.get_db.return_value = mock_conn
    mock_conn.execute.return_value.fetchone.return_value = [20]
    
    # Mock model
    active_v2 = {
        "trained_at": "2020-01-01T00:00:00Z",
        "precision_at_30": 0.5,
        "f1_score": 0.5,
        "version": 2,
        "recipe_path": "recipe.json",
        "accuracy": 0.5,
        "label_count": 20,
        "architecture": "tfidf",
    }
    mock_db.get_active_model.side_effect = [
        {"trained_at": "2020-01-01T00:00:00Z", "precision_at_30": 0.5, "f1_score": 0.5, "version": 1},  # before train
        active_v2,  # after promote
        active_v2,  # score push
        active_v2,  # post-promote recipe path
        active_v2,  # report label counts
    ]
    
    # Mock train returns success
    mock_pipeline.train.return_value = {
        "success": True,
        "precision_at_30": 0.6,
        "f1_score": 0.6,
        "version": 2
    }

    # Mock evaluate_on_recent for the recent-items gate.
    # new model better than old on recent set → promote
    mock_pipeline.evaluate_on_recent.side_effect = [
        {"success": True, "precision_at_30": 0.6, "lead_recall_at_30": 0.8,
         "util_at_30": 0.55, "n_recent": 20, "n_leads": 4},  # new
        {"success": True, "precision_at_30": 0.5, "lead_recall_at_30": 0.8,
         "util_at_30": 0.45, "n_recent": 20, "n_leads": 4},  # old
    ]
    
    # Mock sync compute pending (1st call returns 1, 2nd call returns 0)
    mock_sync._compute_pending_embeddings.side_effect = [1, 0]
    mock_sync.backfill_orphan_label_entries.return_value = (0, 0)
    mock_sync.entry_store_watermarks.return_value = {}

    # Mock get_recent_entries
    mock_db.get_recent_entries.return_value = [{"id": 1}]
    mock_pipeline.score_entries.return_value = [{"id": 1, "relevance_score": 0.5}]
    
    mock_exists.return_value = True
    order = []
    mock_sync.push_scores.side_effect = lambda *a, **k: order.append("push")
    mock_distill_sub.side_effect = lambda *a, **k: order.append("distill") or 0

    # Run
    res = ml_window.main()
    
    assert res == 0
    mock_pipeline.train.assert_called_once_with(profile_id=1, activate=False, recent_holdout_n=ml_window.GATE_N_RECENT)
    # Recent-items gate: evaluate_on_recent called for both new and old models
    assert mock_pipeline.evaluate_on_recent.call_count == 2
    mock_distill_sub.assert_called_once_with(1)
    mock_pipeline.release_embedder.assert_called()
    mock_export.assert_called_once_with(profile_id=1)
    mock_sync.vault_upload.assert_called_once()
    mock_sync.push_scores.assert_called_once()
    assert order == ["push", "distill"]
    push_kwargs = mock_sync.push_scores.call_args.kwargs
    assert push_kwargs.get("model_meta") is not None
    assert "model_trained_at" in push_kwargs["model_meta"]
    # model_trained_at must be UTC, not Zurich time (1-2h ahead in summer).
    assert push_kwargs["model_meta"]["model_trained_at"] == "2020-01-01 00:00:00"

    # Ensure compute embeddings looped twice
    assert mock_sync._compute_pending_embeddings.call_count == 2
    mock_db.activate_model.assert_called_once_with(
        profile_id=1, version=2, origin="window", ungated=False,
    )
    mock_sync.vault_upload.assert_called_once_with(vault_password="vault-secret", package_path=mock_export.return_value, overwrite=True)
    mock_db.get_recent_entries.assert_called_with(days=14, include_embedding=False)

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
    mock_db.derive_profile_identity_from_push_url.return_value = ("Foo", "foo")
    mock_db.get_profile_by_slug.return_value = {"id": 1}
    mock_db.get_all_labels.return_value = [1] * 20
    mock_db.get_model_profile.return_value = {
        "model_name": "Foo",
        "model_uuid": "uuid-1",
        "description": "desk",
    }

    # Database connection mock
    mock_conn = MagicMock()
    mock_db.get_db.return_value = mock_conn
    mock_conn.execute.return_value.fetchone.return_value = [20]
    
    # Mock model
    mock_db.get_active_model.return_value = {"trained_at": "2020-01-01T00:00:00Z", "precision_at_30": 0.5, "f1_score": 0.5, "version": 1}
    
    # Mock train returns success
    mock_pipeline.train.return_value = {
        "success": True,
        "precision_at_30": 0.4, # worse
        "f1_score": 0.4,
        "version": 2
    }

    # Mock evaluate_on_recent — new model worse on recent set → REJECT
    mock_pipeline.evaluate_on_recent.side_effect = [
        {"success": True, "precision_at_30": 0.3, "lead_recall_at_30": 0.4,
         "util_at_30": 0.20, "n_recent": 20, "n_leads": 4},  # new
        {"success": True, "precision_at_30": 0.5, "lead_recall_at_30": 0.8,
         "util_at_30": 0.55, "n_recent": 20, "n_leads": 4},  # old
    ]
    
    # Mock sync compute pending
    mock_sync._compute_pending_embeddings.side_effect = [0]
    mock_sync.backfill_orphan_label_entries.return_value = (0, 0)
    mock_sync.entry_store_watermarks.return_value = {}

    # Mock get_recent_entries
    mock_db.get_recent_entries.return_value = []
    
    # Run
    res = ml_window.main()
    
    assert res == 0
    mock_pipeline.train.assert_called_once_with(profile_id=1, activate=False, recent_holdout_n=ml_window.GATE_N_RECENT)
    # Recent-items gate: evaluate_on_recent called for both new and old models
    assert mock_pipeline.evaluate_on_recent.call_count == 2
    # Reject logged with train_rejected
    mock_db.log_sync.assert_called_once()
    log_args = mock_db.log_sync.call_args
    assert log_args.args[0] == "train_rejected"
    assert log_args.kwargs.get("profile_id") == 1
    assert "Recent gate rejected v2" in log_args.args[2]
    mock_sync.vault_upload.assert_not_called()


def test_score_push_days_defaults_and_config():
    assert ml_window._score_push_days({}) == 14
    assert ml_window._score_push_days({"score_push_days": 7}) == 7
    assert ml_window._score_push_days({"score_push_days": 0}) == 1


# ── _label_counts filters confirmed labels (no pending Gemini rows) ──

def test_label_counts_filters_confirmed_labels():
    """_label_counts must exclude pending Gemini rows, matching get_all_labels."""
    from unittest.mock import MagicMock

    ml_window.db = MagicMock()
    conn = MagicMock()
    ml_window.db.get_db.return_value = conn
    ml_window.db._labels_confirmed_sql = lambda alias: (
        "(%spending_gemini_job_id IS NULL OR TRIM(COALESCE(%spending_gemini_job_id,''))='')"
        % (alias + ".", alias + ".")
    )

    # Each execute() returns a row; fetchone()[0] gives the count.
    # total=30, trainable=28, since_train=12
    conn.execute.side_effect = [
        MagicMock(fetchone=MagicMock(return_value=[30])),  # total
        MagicMock(fetchone=MagicMock(return_value=[28])),  # trainable
        MagicMock(fetchone=MagicMock(return_value=[12])),  # since_train
    ]

    counts = ml_window._label_counts(profile_id=1, trained_at="2025-01-01 00:00:00")
    assert counts == {
        "labels_total": 30,
        "labels_trainable": 28,
        "labels_orphan": 2,
        "labels_since_train": 12,
    }

    # All three queries must include the confirmed-label filter.
    sqls = [call.args[0] for call in conn.execute.call_args_list]
    assert all("pending_gemini_job_id" in sql for sql in sqls), \
        "every _label_counts query must filter confirmed labels"


def test_label_counts_since_train_defaults_to_total_when_no_model():
    """When trained_at is None, since_train = total (confirmed)."""
    from unittest.mock import MagicMock

    ml_window.db = MagicMock()
    conn = MagicMock()
    ml_window.db.get_db.return_value = conn
    ml_window.db._labels_confirmed_sql = lambda alias: "1=1"

    conn.execute.side_effect = [
        MagicMock(fetchone=MagicMock(return_value=[25])),  # total
        MagicMock(fetchone=MagicMock(return_value=[23])),  # trainable
    ]

    counts = ml_window._label_counts(profile_id=1, trained_at=None)
    assert counts["labels_since_train"] == 25
    assert counts["labels_total"] == 25


# ── model_trained_at sent as UTC, not Zurich time ──

def test_model_meta_trained_at_is_utc_not_zurich():
    """model_trained_at must be naive UTC so Seismo's normaliseTimestamp
    (which treats bare YYYY-MM-DD HH:MM:SS as UTC) compares correctly.
    A Zurich conversion would shift it 1-2h ahead in summer.
    """
    from unittest.mock import MagicMock

    ml_window.db = MagicMock()
    ml_window.db.get_model_profile.return_value = {"model_name": "X", "model_uuid": "u", "description": "d"}
    model_row = {"version": 5, "trained_at": "2025-07-15T10:30:00Z", "accuracy": 0.9, "f1_score": 0.8, "label_count": 50, "architecture": "transformer"}

    meta = ml_window._model_meta_for_push(profile_id=1, model_row=model_row)
    assert meta is not None
    # UTC 10:30, not Zurich 12:30 (CEST, +2h in July)
    assert meta["model_trained_at"] == "2025-07-15 10:30:00"


def test_model_meta_trained_at_handles_naive_sql_timestamp():
    """Naive SQLite datetime strings (already UTC) pass through as-is."""
    from unittest.mock import MagicMock

    ml_window.db = MagicMock()
    ml_window.db.get_model_profile.return_value = {"model_name": "X", "model_uuid": "u", "description": "d"}
    model_row = {"version": 5, "trained_at": "2025-07-15 10:30:00", "accuracy": 0.9, "f1_score": 0.8, "label_count": 50, "architecture": "transformer"}

    meta = ml_window._model_meta_for_push(profile_id=1, model_row=model_row)
    assert meta["model_trained_at"] == "2025-07-15 10:30:00"


def test_model_meta_trained_at_empty_on_garbage():
    from unittest.mock import MagicMock

    ml_window.db = MagicMock()
    ml_window.db.get_model_profile.return_value = {"model_name": "X", "model_uuid": "u", "description": "d"}
    model_row = {"version": 5, "trained_at": "", "accuracy": 0.9, "f1_score": 0.8, "label_count": 50, "architecture": "transformer"}

    meta = ml_window._model_meta_for_push(profile_id=1, model_row=model_row)
    assert meta["model_trained_at"] == ""
