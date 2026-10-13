import json
import sqlite3
from unittest.mock import Mock
import pytest
import score_repair_scan as scan
import score_text_scan
import ml_window
import sync


def test_durable_admission_survives_restart_and_rotates_failed_keys(tmp_path):
    path = tmp_path/'repair.sqlite'
    keys = [('email',i) for i in range(1,13)]
    c = sqlite3.connect(path)
    assert scan.admit(c,1,3,'next-page',keys) == keys
    c.close()
    c = sqlite3.connect(path)
    assert scan.load(c,1,3)['pending'] == keys
    batch = scan.admit(c,1,3,'later-page',[('lex_item',99)])
    assert batch == keys[:10]+[('lex_item',99)]
    assert scan.load(c,1,3)['pending'][:3] == keys[10:]+[('lex_item',99)]
    scan.complete(c,1,3,keys[:5])
    assert set(scan.load(c,1,3)['pending']) == set(keys[5:]+[('lex_item',99)])
    assert scan.load(c,2,3) == {'cursor':'','pending':[]}
    assert scan.load(c,1,4) == {'cursor':'','pending':[]}


def test_pending_capacity_is_explicit_and_cannot_advance_cursor(tmp_path):
    c=sqlite3.connect(tmp_path/'repair.sqlite')
    scan.load(c,1,3)
    keys=[['email',i] for i in range(1,201)]
    c.execute('INSERT INTO worker_score_repair_scan VALUES(1,3,?,?)',('prior',json.dumps(keys)));c.commit()
    with pytest.raises(ValueError,match='bounds'):
        scan.admit(c,1,3,'next',[('email',201)])
    assert scan.load(c,1,3)['cursor']=='prior'
    with pytest.raises(ValueError,match='key'):
        scan.admit(c,1,3,'next',[('email',True)])


@pytest.fixture
def repair_env(tmp_path,monkeypatch):
    path=tmp_path/'worker.sqlite'
    connect=lambda:sqlite3.connect(path)
    monkeypatch.setattr(ml_window.db,'get_db',connect)
    monkeypatch.setattr(ml_window,'_score_input_context',lambda *_:{})
    monkeypatch.setattr(ml_window,'_model_meta_for_push',lambda *_:{})
    page=Mock(return_value={'keys':[],'sweep_cursor':'next','sweep_examined':200,'sweep_completed':False})
    monkeypatch.setattr(sync,'pull_score_repair_page',page)
    pull=Mock(side_effect=lambda t,ids,**_:[(t,i) for i in ids])
    monkeypatch.setattr(sync,'pull_entries_by_ids',pull)
    monkeypatch.setattr(score_text_scan,'admit_repair_text_keys',lambda _,keys:(keys,0))
    monkeypatch.setattr(score_text_scan,'by_keys',lambda _,keys:[{'entry_type':t,'entry_id':i,'content':'body'} for t,i in keys])
    monkeypatch.setattr(score_text_scan,'embedding_ready_keys',lambda _,keys:set(keys))
    scorer=Mock(side_effect=lambda rows,**_:rows)
    monkeypatch.setattr(ml_window.pipeline,'score_entries',scorer)
    push=Mock(return_value={'success':True,'errors':0})
    monkeypatch.setattr(sync,'push_scores',push)
    return connect,page,pull,scorer,push


def pending(connect):
    c=connect()
    try:return scan.load(c,1,3)['pending']
    finally:c.close()


def run():
    ml_window._repair_remote_scores(1,{}, {'version':3})


def test_failed_push_keeps_atomic_cursor_and_retry_then_completes(repair_env):
    connect,page,pull,scorer,push=repair_env
    keys=[('email',7),('feed_item',8)]
    page.return_value['keys']=keys
    push.return_value={'success':False}
    with pytest.raises(RuntimeError,match='acknowledged'):
        run()
    assert pending(connect)==keys
    page.return_value={'keys':[],'sweep_cursor':'later','sweep_examined':0,'sweep_completed':False}
    push.return_value={'success':True,'errors':0}
    run()
    assert page.call_args.kwargs['cursor']=='next'
    assert pending(connect)==[]
    run()
    assert scorer.call_count==2  # Healthy delivery is not repeated.


def test_embedding_quota_defers_without_false_failure_and_drains(repair_env,monkeypatch):
    connect,page,pull,scorer,push=repair_env
    keys=[('email',i) for i in range(1,23)]
    page.return_value['keys']=keys
    monkeypatch.setattr(score_text_scan,'embedding_ready_keys',lambda *_:set())
    run()
    assert pending(connect)==keys[10:]
    assert len(scorer.call_args.args[0])==ml_window.pipeline.MAX_ONTHEFLY_EMBEDDINGS
    page.return_value['keys']=[]
    run();run()
    assert pending(connect)==[]
    assert [len(c.args[0]) for c in scorer.call_args_list]==[10,10,2]


def test_true_scoring_omission_is_an_error_and_preserves_only_missing(repair_env):
    connect,page,pull,scorer,push=repair_env
    page.return_value['keys']=[('email',7),('email',8)]
    scorer.side_effect=lambda rows,**_:rows[:1]
    with pytest.raises(RuntimeError,match='omitted'):
        run()
    assert pending(connect)==[('email',8)]


def test_confirmed_source_removal_is_retired_without_scoring_stale_cache(repair_env):
    connect,page,pull,scorer,push=repair_env
    page.return_value['keys']=[('email',7),('email',8)]
    pull.side_effect=lambda *_,**__:[('email',7)]
    run()
    assert pending(connect)==[]
    assert [r['entry_id'] for r in scorer.call_args.args[0]]==[7]


def test_deadline_after_admission_leaves_durable_work(repair_env):
    connect,page,pull,scorer,push=repair_env
    page.return_value['keys']=[('email',7)]
    with pytest.raises(RuntimeError,match='deadline'):
        ml_window._repair_remote_scores(1,{}, {'version':3},deadline=0)
    assert pending(connect)==[('email',7)]
    pull.assert_not_called()


def test_text_deferral_is_retained_and_does_not_load_deferred_bodies(repair_env,monkeypatch):
    connect,page,pull,scorer,push=repair_env
    page.return_value['keys']=[('email',7),('email',8)]
    monkeypatch.setattr(score_text_scan,'admit_repair_text_keys',lambda _,keys:(keys[:1],1))
    run()
    assert pending(connect)==[('email',8)]
    assert len(scorer.call_args.args[0])==1


def test_byte_admission_is_metadata_only_and_preserves_one_oversized_row():
    c=sqlite3.connect(':memory:')
    fields=['id INTEGER PRIMARY KEY','entry_id INTEGER']+[f+" TEXT DEFAULT ''" for f in score_text_scan.TEXT_FIELDS if f not in ('id','entry_id')]
    c.execute('CREATE TABLE entries('+','.join(fields)+')')
    c.executemany("INSERT INTO entries(id,entry_type,entry_id,content) VALUES(?,'email',?,?)",[(1,1,'large first'),(2,2,'second')])
    assert score_text_scan.admit_repair_text_keys(c,[('email',1),('email',2)],max_bytes=1)==([('email',1)],1)
    with pytest.raises(RuntimeError,match='metadata incomplete'):
        score_text_scan.admit_repair_text_keys(c,[('email',3)])
