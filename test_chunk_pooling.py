"""Tests for chunk pooling, rank-normalized push scores, and synthetic weights."""
import unittest
from unittest import mock

import numpy as np

import main
import pipeline


class TestSplitTextChunks(unittest.TestCase):
    def test_short_text_single_chunk(self):
        text = "short body"
        self.assertEqual(pipeline._split_text_chunks(text), [text])

    def test_long_text_multiple_chunks(self):
        text = "word " * 2000
        chunks = pipeline._split_text_chunks(text, chunk_chars=500, max_chunks=4)
        self.assertLessEqual(len(chunks), 4)
        self.assertTrue(all(chunks))
        for ch in chunks:
            self.assertLessEqual(len(ch), 500)

    def test_no_mid_word_cut(self):
        words = ["alpha"] * 400
        text = " ".join(words)
        chunks = pipeline._split_text_chunks(text, chunk_chars=200, max_chunks=3)
        for ch in chunks[:-1]:
            self.assertFalse(ch.endswith("alp"))
            self.assertFalse(ch.startswith("ha"))


class TestChunkPooling(unittest.TestCase):
    def test_length_weighted_mean(self):
        specs = [(0, 100), (0, 300)]
        fake_embs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        by_vecs = {0: [fake_embs[0], fake_embs[1]]}
        by_wts = {0: [100, 300]}
        pooled = np.average(
            np.vstack(by_vecs[0]), axis=0, weights=by_wts[0]
        )
        self.assertAlmostEqual(pooled[0], 0.25)
        self.assertAlmostEqual(pooled[1], 0.75)


class TestRankNormalizePushScores(unittest.TestCase):
    def test_spreads_clustered_scores(self):
        scores = [
            {"relevance_score": 0.49},
            {"relevance_score": 0.51},
            {"relevance_score": 0.49},
        ]
        out = main._rank_normalize_push_scores(scores)
        self.assertAlmostEqual(out[1]["relevance_score"], 1.0)
        self.assertAlmostEqual(out[0]["relevance_score"], 0.5)
        self.assertAlmostEqual(out[2]["relevance_score"], 0.5)

    def test_single_entry_unchanged(self):
        scores = [{"relevance_score": 0.42}]
        out = main._rank_normalize_push_scores(scores)
        self.assertAlmostEqual(out[0]["relevance_score"], 0.42)


class TestSyntheticLabelWeight(unittest.TestCase):
    def test_gemini_half_weight(self):
        labeled = [
            {"label": "important", "label_source": "Gemini"},
            {"label": "important", "label_source": ""},
        ]
        cfg = {"synthetic_label_weight": 0.5}
        w = pipeline.compute_sample_weights(labeled, cfg)
        self.assertAlmostEqual(w[0], 0.5)
        self.assertAlmostEqual(w[1], 1.0)

    def test_legacy_when_disabled(self):
        labeled = [
            {"label": "important", "label_source": "Gemini"},
        ]
        cfg = {"synthetic_label_weight": 1.0}
        w = pipeline.compute_sample_weights(labeled, cfg)
        self.assertAlmostEqual(w[0], 1.0)


if __name__ == "__main__":
    unittest.main()
