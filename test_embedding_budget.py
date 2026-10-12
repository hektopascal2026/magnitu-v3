from unittest.mock import Mock, MagicMock

import numpy as np
import pytest

import ml_window
import pipeline
import sync


def entries(count):
    return [{'entry_type': 'feed_item', 'entry_id': i} for i in range(count)]


def fake_embeddings(monkeypatch, count=25):
    now = [0.0]
    monkeypatch.setattr(sync.time, 'monotonic', lambda: now[0])
    pending = Mock(return_value=entries(count))
    store = Mock()
    embed = Mock(side_effect=lambda rows, **_: [b'embedding'] * len(rows))
    release = Mock()
    monkeypatch.setattr(sync.db, 'get_entries_without_embeddings', pending)
    monkeypatch.setattr(sync.db, 'store_embeddings_batch', store)
    monkeypatch.setattr(pipeline, 'embed_entries', embed)
    monkeypatch.setattr(pipeline, 'release_embedder', release)
    return now, pending, store, embed, release


def test_expired_budget_starts_no_query_or_inference_and_releases(monkeypatch):
    _, pending, store, embed, release = fake_embeddings(monkeypatch)
    assert sync._compute_pending_embeddings(deadline=0) == 0
    pending.assert_not_called()
    store.assert_not_called()
    embed.assert_not_called()
    release.assert_called_once_with()


def test_finished_sub_batch_persists_even_when_it_overruns_budget(monkeypatch):
    now, _, store, embed, release = fake_embeddings(monkeypatch)
    def finish_after_deadline(rows, **_):
        now[0] = 200
        return [b'embedding'] * len(rows)
    embed.side_effect = finish_after_deadline
    assert sync._compute_pending_embeddings(limit=200, deadline=100) == 10
    assert len(embed.call_args.args[0]) == 10
    store.assert_called_once_with([(b'embedding', 'feed_item', i) for i in range(10)])
    release.assert_called_once_with()


def test_query_consumes_budget_before_first_sub_batch(monkeypatch):
    now, pending, store, embed, release = fake_embeddings(monkeypatch)
    def slow_query(**_):
        now[0] = 100
        return entries(25)
    pending.side_effect = slow_query
    assert sync._compute_pending_embeddings(deadline=100) == 0
    store.assert_not_called()
    embed.assert_not_called()
    release.assert_called_once_with()


def test_sub_batch_exception_preserves_prior_updates_and_releases(monkeypatch):
    _, _, store, embed, release = fake_embeddings(monkeypatch)
    embed.side_effect = [[b'embedding'] * 10, RuntimeError('fake inference failure')]
    assert sync._compute_pending_embeddings(deadline=100) == 10
    assert store.call_count == 1
    assert embed.call_count == 2
    release.assert_called_once_with()


def test_callback_exception_releases_embedder(monkeypatch):
    _, _, store, embed, release = fake_embeddings(monkeypatch)
    callback = Mock(side_effect=RuntimeError('fake callback failure'))
    assert sync._compute_pending_embeddings(progress_cb=callback, deadline=100) == 0
    embed.assert_not_called()
    store.assert_not_called()
    release.assert_called_once_with()


def test_database_read_failure_still_propagates_and_releases(monkeypatch):
    _, pending, _, embed, release = fake_embeddings(monkeypatch)
    pending.side_effect = RuntimeError('fake database failure')
    with pytest.raises(RuntimeError, match='fake database failure'):
        sync._compute_pending_embeddings(deadline=100)
    embed.assert_not_called()
    release.assert_called_once_with()


def test_interruption_releases_embedder_and_propagates(monkeypatch):
    _, _, _, embed, release = fake_embeddings(monkeypatch)
    embed.side_effect = KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        sync._compute_pending_embeddings(deadline=100)
    release.assert_called_once_with()


def test_full_window_retains_fifty_entry_batches_without_clock_checks(monkeypatch):
    _, _, store, embed, release = fake_embeddings(monkeypatch, 55)
    monkeypatch.setattr(sync.time, 'monotonic', Mock(side_effect=AssertionError('no deadline')))
    assert sync._compute_pending_embeddings() == 55
    assert [len(call.args[0]) for call in embed.call_args_list] == [50, 5]
    assert store.call_count == 2
    release.assert_called_once_with()


def test_full_window_embedding_loop_keeps_existing_call_shape(monkeypatch):
    compute = Mock(side_effect=[2, 0])
    monkeypatch.setattr(ml_window.sync, '_compute_pending_embeddings', compute)
    ml_window._embed_pending_until_done()
    assert [call.kwargs for call in compute.call_args_list] == [{'limit': 1000}, {'limit': 1000}]


def test_score_only_deadline_is_anchored_before_slow_shared_sync(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(ml_window.time, 'monotonic', lambda: now[0])
    database = MagicMock()
    database.derive_profile_identity_from_push_url.return_value = ('a', 'a')
    database.get_profile_by_slug.return_value = {'id': 1}
    database.get_active_model.return_value = None
    shared_sync = MagicMock()
    shared_sync.pull_all_entry_types.side_effect = lambda **_: now.__setitem__(0, 251.0)
    monkeypatch.setattr(ml_window, 'db', database)
    monkeypatch.setattr(ml_window, 'sync', shared_sync)
    monkeypatch.setattr(ml_window, '_load_worker_secrets_file', lambda: None)
    monkeypatch.setattr(ml_window, 'enforce_embedding_store_cap', lambda: None)
    monkeypatch.setattr(ml_window, 'get_config', lambda: {'seismo_url': 'mother', 'api_key': 'test'})
    monkeypatch.setattr(ml_window, '_label_counts', lambda *_: {})
    monkeypatch.setattr(ml_window, '_score_only_desk_order', lambda prepared: prepared)
    admission = Mock()
    monkeypatch.setattr(ml_window, '_record_score_only_desk_attempt', admission)
    monkeypatch.setenv('MAGNITU_ML_SCORE_ONLY', '1')
    monkeypatch.setenv('MAGNITU_VAULT_PASSWORD', 'test')
    monkeypatch.setenv('SEISMO_DESKS_JSON', '[{"seismo_url":"a","api_key":"test"}]')
    assert ml_window.main() == 0
    shared_sync._compute_pending_embeddings.assert_not_called()
    admission.assert_called_once_with('a')


def test_document_chunk_limits_unchanged_with_fake_encoder(monkeypatch):
    monkeypatch.setattr(pipeline, '_build_entry_text', lambda *_args, **_kwargs: 'word ' * 10000)
    monkeypatch.setattr(pipeline, 'get_config', lambda: {})
    encoded = []
    def fake_encode(texts, **_):
        encoded.append(len(texts))
        return np.zeros((len(texts), 768), dtype=np.float32)
    monkeypatch.setattr(pipeline, 'compute_embeddings', fake_encode)
    for item, expected_chunks in [
        ({'entry_type': 'feed_item'}, 4),
        ({'entry_type': 'lex_item'}, 6),
        ({'entry_type': 'feed_item', 'source_type': 'substack'}, 6),
    ]:
        assert len(pipeline.embed_entries([item])) == 1
        assert encoded[-1] == expected_chunks
