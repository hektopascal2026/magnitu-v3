"""Recipe reads preserve evidence without loading the entire cached corpus."""
from unittest.mock import Mock, patch
import sqlite3

import pytest

import db
import distiller


def test_text_stream_fetches_bounded_batches_and_closes():
    conn = Mock()
    cursor = conn.execute.return_value
    cursor.fetchmany.side_effect = [[{'id': 1}, {'id': 2}], [{'id': 3}], []]
    with patch.object(db, 'get_db', return_value=conn):
        assert list(db.iter_entries_text(batch_size=2)) == [{'id': 1}, {'id': 2}, {'id': 3}]
    query = conn.execute.call_args.args[0]
    assert 'embedding' not in query and 'SELECT *' not in query
    assert 'ORDER BY id' in query
    assert cursor.fetchmany.call_args_list == [((2,),), ((2,),), ((2,),)]
    cursor.fetchall.assert_not_called()
    conn.close.assert_called_once()


def test_text_stream_closes_when_consumer_stops():
    conn = Mock()
    conn.execute.return_value.fetchmany.return_value = [{'id': 1}]
    with patch.object(db, 'get_db', return_value=conn):
        stream = db.iter_entries_text()
        assert next(stream) == {'id': 1}
        stream.close()
    conn.close.assert_called_once()


def test_real_text_stream_preserves_complete_population_without_blobs():
    conn = sqlite3.connect(':memory:')
    conn.row_factory = sqlite3.Row
    columns = db._ENTRY_TEXT_COLUMNS.split(', ')
    conn.execute('CREATE TABLE entries (' + ', '.join(
        'id INTEGER PRIMARY KEY' if name == 'id' else name + ' TEXT'
        for name in columns
    ) + ', embedding BLOB)')
    for number in range(1, 12):
        values = [number if name == 'id' else
                  ('2026-10-%02d' % (30 - number) if name == 'published_date' else
                   'ä complete %s %d' % (name, number)) for name in columns]
        conn.execute('INSERT INTO entries VALUES (' + ','.join(['?'] * (len(columns) + 1)) + ')', values + [b'not a text field'])
    baseline = [dict(row) for row in conn.execute(
        'SELECT ' + db._ENTRY_TEXT_COLUMNS + ' FROM entries ORDER BY published_date DESC')]
    with patch.object(db, 'get_db', return_value=conn):
        actual = list(db.iter_entries_text(batch_size=3))
    assert sorted(actual, key=lambda row: row['id']) == sorted(baseline, key=lambda row: row['id'])
    assert all('embedding' not in row for row in actual)
    with pytest.raises(sqlite3.ProgrammingError):
        conn.execute('SELECT 1')


def test_text_stream_closes_on_query_failure():
    conn = Mock()
    conn.execute.side_effect = sqlite3.OperationalError('fixture')
    with patch.object(db, 'get_db', return_value=conn), pytest.raises(sqlite3.OperationalError):
        list(db.iter_entries_text())
    conn.close.assert_called_once()


def test_text_stream_rejects_unbounded_batch():
    with patch.object(db, 'get_db') as connect:
        with pytest.raises(ValueError):
            list(db.iter_entries_text(0))
    connect.assert_not_called()


def test_normalization_streams_every_row_with_identical_result():
    rows = [{'entry_id': i, 'content': 'energy ' * (i + 1)} for i in range(7)]
    keywords = {'energy': {'important': 0.2}}
    with patch.object(db, 'get_all_entries', side_effect=AssertionError('whole corpus load')), \
         patch.object(db, 'iter_entries_text', side_effect=lambda: iter(rows)), \
         patch.object(distiller, 'get_config', return_value={}):
        forward = distiller._normalize_weights(keywords, {})
    with patch.object(db, 'iter_entries_text', side_effect=lambda: reversed(rows)), \
         patch.object(distiller, 'get_config', return_value={}):
        reverse = distiller._normalize_weights(keywords, {})
    assert forward == reverse
    magnitudes = [max(abs(x) for x in distiller._accumulate_recipe_class_scores(
        row, keywords, {}, ['investigation_lead', 'important', 'background', 'noise']).values()) for row in rows]
    magnitudes = sorted(x for x in magnitudes if x > 0)
    scale = 2.0 / magnitudes[len(magnitudes) // 2]
    assert forward[0]['energy']['important'] == round(0.2 * scale, 4)


def test_quality_fetches_only_original_sample_keys_in_original_order():
    keys = [('feed_item', i) for i in range(20)]
    rows = [{'entry_type': et, 'entry_id': eid} for et, eid in keys[:3]]
    scores = [{'entry_type': et, 'entry_id': eid, 'relevance_score': eid / 20} for et, eid in keys[:3]]
    with patch.object(db, 'get_all_entries', side_effect=AssertionError('whole corpus load')), \
         patch.object(db, 'get_all_entry_keys', return_value=keys), \
         patch.object(db, 'get_entries_text_by_keys', return_value=rows) as fetch, \
         patch.object(distiller, 'score_entries', return_value=scores), \
         patch.object(distiller, '_recipe_composite', side_effect=[0.0, 0.05, 0.1]):
        result = distiller.evaluate_recipe_rank_metrics({}, sample_size=3, profile_id=4)
    fetch.assert_called_once_with(keys[:3])
    assert result['spearman'] == 1.0
