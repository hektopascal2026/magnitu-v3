import sqlite3
from unittest.mock import Mock

import numpy as np
import pytest

import db
import ml_window
import pipeline
import sync
from score_push_checkpoint import fingerprint


class FakeClassifier:
    classes_ = np.array(['investigation_lead', 'important', 'background', 'noise'])

    def __init__(self):
        self.inputs = []

    def predict_proba(self, values):
        self.inputs.append(values)
        return np.tile([0.1, 0.2, 0.3, 0.4], (len(values), 1))

    def decision_function(self, values):
        self.inputs.append(values)
        return np.tile([0.0, 1.0, 2.0, 3.0], (len(values), 1))


@pytest.fixture(params=['tfidf', 'transformer'])
def scorer(request, monkeypatch, tmp_path):
    # Older tests replace module references during import. Bind the actual
    # dependencies here so these integration tests exercise the real scoring
    # functions regardless of collection order, then restore prior bindings.
    monkeypatch.setattr(ml_window, 'db', db)
    monkeypatch.setattr(pipeline, 'db', db)
    monkeypatch.setattr(sync, 'db', db)
    monkeypatch.setattr(ml_window, 'pipeline', pipeline)
    monkeypatch.setattr(ml_window, 'sync', sync)
    path = tmp_path / 'worker.sqlite'
    def connect():
        conn = sqlite3.connect(str(path))
        conn.row_factory = sqlite3.Row
        return conn
    with connect() as conn:
        conn.execute('CREATE TABLE entries (entry_type TEXT, entry_id INTEGER, embedding BLOB)')
        conn.execute("INSERT INTO entries VALUES ('feed_item',1,NULL)")
    cfg = {'classifier_apply_prior': False, 'embedding_dim': 3, 'embedding_content_cap': 3000}
    model_path = tmp_path / 'model.joblib'
    model_path.write_text('fake model')
    model = {'profile_id': 1, 'version': 1, 'architecture': request.param,
             'model_path': str(model_path), 'embedding_l2_normalize': False}
    classifier = FakeClassifier()
    calibration = [None]
    monkeypatch.setattr(db, 'get_db', connect)
    monkeypatch.setattr(db, 'get_active_model', lambda *_, **__: model)
    monkeypatch.setattr(db, 'get_effective_config', lambda _: dict(cfg))
    monkeypatch.setattr(ml_window, 'get_config', lambda: dict(cfg))
    monkeypatch.setattr(pipeline, 'get_config', lambda: dict(cfg))
    monkeypatch.setattr(pipeline.joblib, 'load', lambda _: classifier)
    monkeypatch.setattr(pipeline, 'load_calibration', lambda _: calibration[0])
    encoder = Mock(side_effect=lambda texts, **_: np.ones((len(texts), 3), dtype=np.float32))
    monkeypatch.setattr(pipeline, 'compute_embeddings', encoder)
    monkeypatch.setattr(ml_window, '_model_meta_for_push', lambda *_: {})
    pushes = Mock(return_value={'success': True, 'errors': 0})
    monkeypatch.setattr(ml_window.sync, 'push_scores', pushes)
    profile = {'id': 1, 'seismo_url': 'https://example.test/desk', 'api_key': 'test'}
    item = {'entry_type': 'feed_item', 'entry_id': 1, 'title': 'Title',
            'content': 'words ' * 1000, 'source_type': 'rss'}
    def run():
        ml_window._score_and_push_changed_entries(1, profile, model, [item])
    return cfg, model, profile, item, classifier, encoder, calibration, pushes, run


def test_actual_scorers_skip_replay_for_unrelated_settings(scorer):
    cfg, _, profile, _, classifier, encoder, _, pushes, run = scorer
    run()
    assert pushes.call_count == 1
    first_input_count = len(classifier.inputs)
    for key, value in {
        'active_profile_id': 99, 'seismo_pruning_days': 90,
        'gemini_model': 'different', 'auto_train_after_n_labels': 100,
        'distillation_max_entries': 2000, 'seismo_push_batch_size': 5,
        'seismo_push_max_body_bytes': 2048, 'sync_watermark': 'later',
        'score_push_days': 7, 'seismo_url': 'https://other-mothership.test',
    }.items():
        cfg[key] = value
        run()
    profile['api_key'] = 'rotated credential'
    run()
    assert pushes.call_count == 1
    assert len(classifier.inputs) == first_input_count
    assert encoder.call_count <= 1  # Real transformer path, fake encoder only.


def test_actual_scorers_replay_on_scoring_inputs_and_calibration(scorer):
    cfg, model, profile, item, _, _, calibration, pushes, run = scorer
    run()
    cfg['embedding_content_cap'] = 500
    run()
    item['title'] = 'Revised title'
    run()
    model['version'] = 2
    run()
    profile['seismo_url'] = 'https://example.test/other-desk'
    run()
    calibration[0] = {'method': 'temperature', 'temperature': 2.0,
                      'class_names': list(FakeClassifier.classes_),
                      'prior_fit': {'prior_log_offsets': {'investigation_lead': 2.0, 'noise': -2.0}}}
    run()
    before_prior = pushes.call_args.args[0][0]['probabilities']
    cfg['classifier_apply_prior'] = True
    run()
    assert pushes.call_count == 7
    assert pushes.call_args.args[0][0]['probabilities'] != before_prior


def test_required_embedding_settings_and_dimension_remain_context_inputs(scorer):
    cfg, model, profile, _, _, _, _, _, _ = scorer
    baseline = fingerprint(ml_window._score_input_context(1, profile, model))
    for key in ml_window.SCORE_INPUT_CONFIG_KEYS:
        original = dict(cfg)
        cfg[key] = 'changed context input'
        assert fingerprint(ml_window._score_input_context(1, profile, model)) != baseline
        cfg.clear()
        cfg.update(original)
    cfg['embedding_dim'] = 4
    assert fingerprint(ml_window._score_input_context(1, profile, model)) != baseline


def test_fallback_destination_change_invalidates_same_input(scorer):
    cfg, model, _, _, _, _, _, _, _ = scorer
    cfg['seismo_url'] = 'https://first-mothership.test'
    blank_profile = {'id': 1, 'seismo_url': '', 'api_key': ''}
    first = ml_window._score_input_context(1, blank_profile, model)
    cfg['seismo_url'] = 'https://second-mothership.test'
    second = ml_window._score_input_context(1, blank_profile, model)
    assert first['target'] != second['target']
    assert fingerprint(first) != fingerprint(second)
