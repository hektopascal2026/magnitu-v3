"""WP2: Spearman cap objective + recipe_quality_floor holdback."""
from unittest.mock import patch, MagicMock

import distiller
import ml_window


def test_spearman_monotonic_identity():
    xs = [0.1, 0.2, 0.3, 0.9]
    m = distiller.paired_rank_metrics(xs, xs)
    assert abs(m["spearman"] - 1.0) < 1e-9
    assert abs(m["pearson"] - 1.0) < 1e-9
    assert abs(m["top30_overlap"] - 1.0) < 1e-9


def test_spearman_vs_pearson_on_rank_preserving_nonlinear():
    model = [i for i in range(1, 41)]
    recipe = [v * v for v in model]
    m = distiller.paired_rank_metrics(model, recipe)
    assert m["spearman"] > 0.99
    assert m["pearson"] < 0.97


def test_top30_overlap_tiebreak_shape():
    model = list(range(40, 0, -1))
    recipe_match = list(model)
    recipe_shift = [0.0] * 30 + list(range(10, 0, -1))
    match = distiller.paired_rank_metrics(model, recipe_match)
    shift = distiller.paired_rank_metrics(model, recipe_shift)
    assert match["top30_overlap"] > shift["top30_overlap"]


def test_recipe_quality_floor_holdback_skips_push():
    recipe = {"metrics": {"recipe_quality": 0.12}}
    mock_db = MagicMock()
    with patch.object(ml_window, "db", mock_db), \
         patch.object(ml_window, "get_config", return_value={"recipe_quality_floor": 0.30}):
        held = ml_window._hold_recipe_below_floor(recipe, profile_id=3)
    assert held is True
    mock_db.log_sync.assert_called_once()
    args, kwargs = mock_db.log_sync.call_args
    assert args[0] == "recipe_quality_below_floor"
    assert kwargs.get("profile_id") == 3 or (len(args) >= 4 and args[3] == 3)


def test_recipe_quality_floor_zero_disables():
    recipe = {"metrics": {"recipe_quality": 0.01}}
    mock_db = MagicMock()
    with patch.object(ml_window, "db", mock_db):
        held = ml_window._hold_recipe_below_floor(
            recipe, profile_id=1, cfg={"recipe_quality_floor": 0}
        )
    assert held is False
    mock_db.log_sync.assert_not_called()


def test_recipe_quality_above_floor_pushes():
    recipe = {"metrics": {"recipe_quality": 0.55}}
    mock_db = MagicMock()
    with patch.object(ml_window, "db", mock_db), \
         patch.object(ml_window, "get_config", return_value={"recipe_quality_floor": 0.30}):
        held = ml_window._hold_recipe_below_floor(recipe, 1)
    assert held is False
    mock_db.log_sync.assert_not_called()


def test_post_promote_below_floor_does_not_push_recipe(tmp_path):
    recipe_path = tmp_path / "recipe.json"
    recipe_path.write_text(
        '{"metrics": {"recipe_quality": 0.05, "recipe_pearson": 0.4}}',
        encoding="utf-8",
    )
    mock_sync = MagicMock()
    mock_db = MagicMock()
    mock_db.get_active_model.return_value = {
        "recipe_path": str(recipe_path),
        "version": 2,
    }
    with patch.object(ml_window, "db", mock_db), \
         patch.object(ml_window, "sync", mock_sync), \
         patch.object(ml_window, "pipeline"), \
         patch.object(ml_window, "_distill_recipe_in_subprocess", return_value=0), \
         patch.object(ml_window, "export_model", return_value=str(tmp_path / "m.zip")), \
         patch.object(ml_window, "get_config", return_value={"recipe_quality_floor": 0.30}):
        ok = ml_window._post_promote_recipe_and_vault(
            1, "http://example", {"id": 1}, "pw"
        )
    assert ok is True
    mock_sync.push_recipe.assert_not_called()
    mock_sync.vault_upload.assert_called_once()
    mock_db.log_sync.assert_called()
    assert mock_db.log_sync.call_args[0][0] == "recipe_quality_below_floor"
