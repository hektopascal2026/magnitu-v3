"""Tests for PHP-parity recipe scoring and scoring-fix plan acceptance criteria."""
import unittest
from unittest import mock

import db
import distiller
import explainer
import pipeline
import sync


class TestSeismoTokenizer(unittest.TestCase):
    def test_hyphen_split(self):
        words = distiller._seismo_tokenize("E-Commerce-Verordnung")
        self.assertEqual(words, ["e", "commerce", "verordnung"])
        self.assertIn("e commerce", distiller._seismo_tokens("E-Commerce-Verordnung"))

    def test_normalize_recipe_key(self):
        self.assertEqual(distiller._normalize_recipe_key("Third-Country"), "third country")

    def test_accent_preservation(self):
        tokens = distiller._seismo_tokens("Überwachung der Märkte")
        self.assertIn("überwachung", tokens)


class TestRecipeCompositeParity(unittest.TestCase):
    def test_once_per_token(self):
        keywords = {"signal": {"important": 0.5}}
        entry = {
            "entry_type": "feed_item",
            "title": "signal signal signal signal signal",
            "description": "",
            "content": "",
            "source_type": "rss",
        }
        classes = ["investigation_lead", "important", "background", "noise"]
        class_wts = pipeline.class_weight_list(classes)
        scores = distiller._accumulate_recipe_class_scores(
            entry, keywords, {}, classes
        )
        self.assertAlmostEqual(scores["important"], 0.5)

    def test_lex_synopsis_only(self):
        keywords = {
            "third country": {"investigation_lead": 1.0},
            "member states only": {"investigation_lead": 1.0},
        }
        entry = {
            "entry_type": "lex_item",
            "title": "t",
            "description": "third country",
            "content": "member states only",
            "source_type": "lex_eu",
        }
        classes = ["investigation_lead", "important", "background", "noise"]
        scores = distiller._accumulate_recipe_class_scores(
            entry, keywords, {}, classes
        )
        self.assertAlmostEqual(scores["investigation_lead"], 1.0)


class TestReasoningBoost(unittest.TestCase):
    def test_single_boost_not_compounded(self):
        keywords = {"trade": {"important": 0.08}}
        labels = [
            {"reasoning": "trade deal", "label": "important"}
            for _ in range(10)
        ]
        with mock.patch.object(db, "get_all_reasoning_texts", return_value=labels):
            out = distiller._boost_from_reasoning(dict(keywords), profile_id=1)
        self.assertAlmostEqual(out["trade"]["important"], 0.12)

    def test_negative_coefficient_becomes_positive_seed(self):
        keywords = {"trade": {"important": -0.2}}
        labels = [{"reasoning": "trade deal", "label": "important"}]
        with mock.patch.object(db, "get_all_reasoning_texts", return_value=labels):
            out = distiller._boost_from_reasoning(dict(keywords), profile_id=1)
        self.assertGreater(out["trade"]["important"], 0)


class TestOptimizeCapsFloor(unittest.TestCase):
    def test_floor_applied_during_optimization(self):
        keywords = {"member states only": {"investigation_lead": 0.1}}
        source_weights = {}
        labels = [
            {"entry_type": "feed_item", "entry_id": i, "label": "important"}
            for i in range(5)
        ]
        entries = [
            {
                "entry_type": "feed_item",
                "entry_id": i,
                "title": "news",
                "description": "",
                "content": "text",
                "source_type": "rss",
            }
            for i in range(5)
        ]
        scores = [
            {
                "entry_type": "feed_item",
                "entry_id": i,
                "relevance_score": 0.5,
            }
            for i in range(5)
        ]
        with mock.patch.object(db, "get_all_labels", return_value=labels), \
             mock.patch.object(db, "get_all_entries", return_value=entries), \
             mock.patch.object(distiller, "score_entries", return_value=scores):
            kw, _sw, meta = distiller._optimize_recipe_caps(
                keywords, source_weights, profile_id=1
            )
        self.assertIsNotNone(meta)
        self.assertGreaterEqual(
            kw.get("member states only", {}).get("investigation_lead", 0), 0.55
        )


class TestExplainerCache(unittest.TestCase):
    def test_classifier_loaded_once(self):
        load_count = {"n": 0}
        fake_path = "/fake/model.joblib"

        def counting_load(path):
            load_count["n"] += 1
            return object()

        stat_mock = mock.Mock(st_mtime=1.0)
        with mock.patch.object(explainer.joblib, "load", side_effect=counting_load), \
             mock.patch.object(explainer.Path, "stat", return_value=stat_mock):
            explainer._CLF_CACHE.update({"path": None, "mtime": None, "clf": None})
            explainer._load_classifier_cached(fake_path)
            explainer._load_classifier_cached(fake_path)
        self.assertEqual(load_count["n"], 1)


class TestLabelTimestampNormalize(unittest.TestCase):
    def test_iso_t_does_not_beat_later_space_format(self):
        remote = "2026-06-10T08:00:00"
        local = "2026-06-10 09:00:00"
        self.assertFalse(
            sync._normalize_label_ts(remote) > sync._normalize_label_ts(local)
        )


if __name__ == "__main__":
    unittest.main()
