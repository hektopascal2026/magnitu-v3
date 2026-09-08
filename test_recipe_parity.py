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
        with mock.patch.object(db, "get_all_reasoning_texts", return_value=labels), \
             mock.patch.object(db, "get_all_labels", return_value=[]):
            out = distiller._boost_from_reasoning(dict(keywords), profile_id=1)
        self.assertAlmostEqual(out["trade"]["important"], 0.12)

    def test_negative_coefficient_preserved(self):
        # Phase 2: a reasoning mention alone doesn't reverse a learned negative.
        # "no connection to defence procurement" mentions the term contrastively.
        keywords = {"trade": {"important": -0.2}}
        labels = [{"reasoning": "trade deal", "label": "important"}]
        with mock.patch.object(db, "get_all_reasoning_texts", return_value=labels), \
             mock.patch.object(db, "get_all_labels", return_value=[]):
            out = distiller._boost_from_reasoning(dict(keywords), profile_id=1)
        self.assertAlmostEqual(out["trade"]["important"], -0.2)

    def test_new_unigram_not_seeded(self):
        # Reasoning unigrams are too generic to seed as new recipe keywords.
        keywords = {}
        labels = [
            {"entry_type": "feed_item", "entry_id": i,
             "reasoning": "tariff impact", "label": "important"}
            for i in range(5)
        ]
        with mock.patch.object(db, "get_all_reasoning_texts", return_value=labels), \
             mock.patch.object(db, "get_all_labels", return_value=[]):
            out = distiller._boost_from_reasoning(dict(keywords), profile_id=1)
        self.assertNotIn("tariff", out)
        self.assertNotIn("impact", out)

    def test_new_phrase_below_recurrence_threshold_not_seeded(self):
        # A phrase mentioned in only one entry's reasoning is prose, not signal.
        keywords = {}
        labels = [
            {"entry_type": "feed_item", "entry_id": 1,
             "reasoning": "focused on bilateral", "label": "important"}
        ]
        with mock.patch.object(db, "get_all_reasoning_texts", return_value=labels), \
             mock.patch.object(db, "get_all_labels", return_value=[]):
            out = distiller._boost_from_reasoning(dict(keywords), profile_id=1)
        self.assertNotIn("focused on", out)
        self.assertNotIn("on bilateral", out)

    def test_new_phrase_at_recurrence_threshold_seeded(self):
        # A phrase recurring across >=3 distinct entries is real signal.
        # (No labeled entries => discriminative gate falls back to recurrence-only.)
        keywords = {}
        labels = [
            {"entry_type": "feed_item", "entry_id": i,
             "reasoning": "third country exclusion", "label": "important"}
            for i in range(3)
        ]
        with mock.patch.object(db, "get_all_reasoning_texts", return_value=labels), \
             mock.patch.object(db, "get_all_labels", return_value=[]):
            out = distiller._boost_from_reasoning(dict(keywords), profile_id=1)
        self.assertIn("third country", out)
        self.assertIn("country exclusion", out)
        self.assertGreater(out["third country"]["important"], 0)

    def test_new_low_signal_phrase_not_seeded_even_if_recurring(self):
        # All-stopword n-grams are dropped by the low-signal filter.
        keywords = {}
        labels = [
            {"entry_type": "feed_item", "entry_id": i,
             "reasoning": "on the and", "label": "important"}
            for i in range(5)
        ]
        with mock.patch.object(db, "get_all_reasoning_texts", return_value=labels), \
             mock.patch.object(db, "get_all_labels", return_value=[]):
            out = distiller._boost_from_reasoning(dict(keywords), profile_id=1)
        self.assertNotIn("on the", out)
        self.assertNotIn("the and", out)

    def test_existing_unigram_boosted_regardless_of_recurrence(self):
        # Tokens already in the recipe with positive weight are always boosted.
        keywords = {"tariff": {"important": 0.08}}
        labels = [
            {"entry_type": "feed_item", "entry_id": 1,
             "reasoning": "tariff impact", "label": "important"}
        ]
        with mock.patch.object(db, "get_all_reasoning_texts", return_value=labels), \
             mock.patch.object(db, "get_all_labels", return_value=[]):
            out = distiller._boost_from_reasoning(dict(keywords), profile_id=1)
        self.assertAlmostEqual(out["tariff"]["important"], 0.12)

    def test_relabel_same_entry_does_not_inflate_recurrence(self):
        # Re-annotating one entry 5 times must count as 1 distinct entry.
        keywords = {}
        labels = [
            {"entry_type": "feed_item", "entry_id": 1,
             "reasoning": "third country exclusion", "label": "important"}
            for _ in range(5)
        ]
        with mock.patch.object(db, "get_all_reasoning_texts", return_value=labels), \
             mock.patch.object(db, "get_all_labels", return_value=[]):
            out = distiller._boost_from_reasoning(dict(keywords), profile_id=1)
        self.assertNotIn("third country", out)

    def test_numeric_date_fragment_dropped(self):
        # N-grams made only of numbers (dates, quantities) are not signal.
        self.assertTrue(distiller._is_low_signal_feature("08 31"))
        self.assertTrue(distiller._is_low_signal_feature("2026 08"))
        self.assertTrue(distiller._is_low_signal_feature("2026 08 31"))
        # A number mixed with a real word is kept.
        self.assertFalse(distiller._is_low_signal_feature("section 3"))
        self.assertFalse(distiller._is_low_signal_feature("article 2026"))

    def test_boilerplate_phrase_blocked_by_discrim_gate(self):
        # "this article discusses" recurs in reasoning but appears in entries
        # of ALL labels equally => not discriminative => not seeded.
        keywords = {}
        reasoning_labels = [
            {"entry_type": "feed_item", "entry_id": i,
             "reasoning": "this article discusses trade", "label": "important"}
            for i in range(3)
        ]
        # Labeled entries: "this article discusses" appears in 3 important
        # AND 3 noise entries equally => ratio 1.0 < 2.0.
        all_labels = [
            {"entry_type": "feed_item", "entry_id": i,
             "label": "important", "title": "this article discusses policy",
             "description": "", "content": "", "source_type": "rss"}
            for i in range(3)
        ] + [
            {"entry_type": "feed_item", "entry_id": i + 10,
             "label": "noise", "title": "this article discusses weather",
             "description": "", "content": "", "source_type": "rss"}
            for i in range(3)
        ]
        with mock.patch.object(db, "get_all_reasoning_texts", return_value=reasoning_labels), \
             mock.patch.object(db, "get_all_labels", return_value=all_labels):
            out = distiller._boost_from_reasoning(dict(keywords), profile_id=1)
        self.assertNotIn("this article", out)
        self.assertNotIn("article discusses", out)

    def test_discriminative_phrase_seeded(self):
        # "third country exclusion" appears in 3 important entries and 0 noise
        # entries => maximally discriminative => seeded.
        keywords = {}
        reasoning_labels = [
            {"entry_type": "feed_item", "entry_id": i,
             "reasoning": "third country exclusion applies", "label": "important"}
            for i in range(3)
        ]
        all_labels = [
            {"entry_type": "feed_item", "entry_id": i,
             "label": "important", "title": "third country exclusion in trade",
             "description": "", "content": "", "source_type": "rss"}
            for i in range(3)
        ] + [
            {"entry_type": "feed_item", "entry_id": i + 10,
             "label": "noise", "title": "local weather report",
             "description": "", "content": "", "source_type": "rss"}
            for i in range(3)
        ]
        with mock.patch.object(db, "get_all_reasoning_texts", return_value=reasoning_labels), \
             mock.patch.object(db, "get_all_labels", return_value=all_labels):
            out = distiller._boost_from_reasoning(dict(keywords), profile_id=1)
        self.assertIn("third country", out)
        self.assertGreater(out["third country"]["important"], 0)

    def test_non_discriminative_phrase_blocked_even_with_recurrence(self):
        # Phrase recurs in 5 entries' reasoning but appears in 2 important
        # and 4 noise entries => P(phrase|important)=2/5, P(phrase|noise)=4/5
        # ratio = 0.5 < 2.0 => blocked.
        keywords = {}
        reasoning_labels = [
            {"entry_type": "feed_item", "entry_id": i,
             "reasoning": "market reaction follows", "label": "important"}
            for i in range(5)
        ]
        all_labels = [
            {"entry_type": "feed_item", "entry_id": i,
             "label": "important", "title": "market reaction follows policy",
             "description": "", "content": "", "source_type": "rss"}
            for i in range(2)
        ] + [
            {"entry_type": "feed_item", "entry_id": i + 10,
             "label": "noise", "title": "market reaction follows earnings",
             "description": "", "content": "", "source_type": "rss"}
            for i in range(4)
        ] + [
            {"entry_type": "feed_item", "entry_id": i + 20,
             "label": "important", "title": "unrelated story",
             "description": "", "content": "", "source_type": "rss"}
            for i in range(3)
        ]
        with mock.patch.object(db, "get_all_reasoning_texts", return_value=reasoning_labels), \
             mock.patch.object(db, "get_all_labels", return_value=all_labels):
            out = distiller._boost_from_reasoning(dict(keywords), profile_id=1)
        self.assertNotIn("market reaction", out)


class TestLegalTemplateGroups(unittest.TestCase):
    def test_compliance_group_excluded(self):
        # Excluding "compliance" drops "ce marking" but keeps Gold-critical
        # "third country" (trade_market group).
        keywords = {}
        with mock.patch.object(distiller, "get_config",
                                return_value={"legal_template_group_exclusions": ["compliance"]}):
            out = distiller._boost_legal_templates(dict(keywords))
        self.assertNotIn("ce marking", out)
        self.assertNotIn("conformity assessment", out)
        self.assertIn("third country", out)
        self.assertIn("member states only", out)

    def test_trade_market_group_excluded(self):
        # Excluding "trade_market" drops Gold-critical phrases.
        keywords = {}
        with mock.patch.object(distiller, "get_config",
                                return_value={"legal_template_group_exclusions": ["trade_market"]}):
            out = distiller._boost_legal_templates(dict(keywords))
        self.assertNotIn("third country", out)
        self.assertNotIn("member states only", out)
        self.assertIn("ce marking", out)

    def test_procedural_noise_group_excluded(self):
        # Excluding "procedural_noise" drops "implementing act" noise prior.
        keywords = {}
        with mock.patch.object(distiller, "get_config",
                                return_value={"legal_template_group_exclusions": ["procedural_noise"]}):
            out = distiller._boost_legal_templates(dict(keywords))
        self.assertNotIn("implementing act", out)
        self.assertNotIn("corrigendum", out)
        self.assertIn("third country", out)

    def test_no_exclusions_applies_all_groups(self):
        # Default: all groups apply (backward compatible).
        keywords = {}
        with mock.patch.object(distiller, "get_config",
                                return_value={"legal_template_group_exclusions": []}):
            out = distiller._boost_legal_templates(dict(keywords))
        self.assertIn("third country", out)
        self.assertIn("ce marking", out)
        self.assertIn("implementing act", out)

    def test_floor_respects_exclusions(self):
        # Floor pass must skip excluded groups too — no floor without seed.
        keywords = {"ce marking": {"important": 0.01}}
        with mock.patch.object(distiller, "get_config",
                                return_value={"legal_template_group_exclusions": ["compliance"]}):
            out = distiller._apply_floor_weights(dict(keywords))
        # "ce marking" was already in keywords (from TF-IDF) but its group is
        # excluded — floor should NOT raise it to 0.25.
        self.assertLess(out["ce marking"]["important"], 0.25)
        # "third country" (trade_market, not excluded) should still be floored.
        self.assertGreaterEqual(
            out.get("third country", {}).get("investigation_lead", 0), 0.45
        )


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
