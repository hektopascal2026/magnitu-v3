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
    assert scan.load(c,1,4) == scan.load(c,1,3)


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


def test_retry_age_attempts_and_causes_survive_restart(tmp_path):
    path = tmp_path/'repair.sqlite'
    c = sqlite3.connect(path)
    keys = [('email', 7), ('email', 8)]
    scan.admit(c, 1, 3, 'next', keys, now=100)
    scan.outcome(c, 1, 3, [keys[1]], 'embedding_budget')
    c.close()
    c = sqlite3.connect(path)
    h = scan.health(c, 1, 3, now=200)
    assert h == {'pending': 2, 'oldest_seconds': 100, 'max_attempts': 1,
                 'cursor_stale_seconds': 100, 'causes': {'interrupted': 1, 'embedding_budget': 1}}
    scan.admit(c, 1, 3, 'next', [], now=300)
    assert scan.health(c, 1, 3, now=400)['oldest_seconds'] == 300
    assert scan.health(c, 1, 3, now=400)['max_attempts'] == 2
    assert scan.health(c, 1, 3, now=400)['cursor_stale_seconds'] == 300
    scan.complete(c, 1, 3, keys)
    assert scan.health(c, 1, 3, now=400)['causes'] == scan.health(c, 1, 4, now=400)['causes']
    assert c.execute('SELECT COUNT(*) FROM worker_score_repair_retry').fetchone()[0] == 0


def test_legacy_queue_migration_is_idempotent_and_age_is_observation_time(tmp_path):
    c = sqlite3.connect(tmp_path/'repair.sqlite')
    c.execute('CREATE TABLE worker_score_repair_scan(profile_id INTEGER PRIMARY KEY,model_version INTEGER,cursor TEXT,pending TEXT)')
    c.execute('INSERT INTO worker_score_repair_scan VALUES(1,3,?,?)', ('prior', json.dumps([['email',7]])))
    c.commit()
    scan.admit(c, 1, 3, 'next', [], now=100)
    scan.load(c, 1, 3)
    assert scan.health(c, 1, 3, now=200)['oldest_seconds'] == 100
    scan.admit(c, 1, 4, 'fresh', [('email',8)], now=300)
    assert scan.health(c, 1, 4, now=400)['oldest_seconds'] == 300
    assert c.execute('SELECT COUNT(*) FROM worker_score_repair_retry').fetchone()[0] == 2
    assert scan.health(c, 1, 3, now=400)['causes'] == scan.health(c, 1, 4, now=400)['causes']


@pytest.mark.parametrize('failure,cause', [('refetch','refetch_failure'), ('push','scoring_delivery_failure'),
                                          ('omission','scoring_omission'), ('deadline','deadline_exhausted')])
def test_failure_cause_is_durable(repair_env, failure, cause):
    connect,page,pull,scorer,push = repair_env
    page.return_value['keys'] = [('email',7)]
    if failure == 'refetch':
        pull.side_effect = RuntimeError('transport failed')
    elif failure == 'push':
        push.return_value = {'success':False}
    elif failure == 'omission':
        scorer.side_effect = lambda *_args, **_kwargs: []
    with pytest.raises(RuntimeError):
        ml_window._repair_remote_scores(1, {}, {'version':3}, deadline=0 if failure == 'deadline' else None)
    c = connect()
    assert scan.health(c, 1, 3)['causes'] == {cause:1}


def test_budget_deferral_causes_are_not_overwritten_by_scoring_failure(repair_env, monkeypatch):
    connect,page,pull,scorer,push = repair_env
    page.return_value['keys'] = [('email',i) for i in range(1,14)]
    monkeypatch.setattr(score_text_scan, 'admit_repair_text_keys', lambda _,keys:(keys[:-1],1))
    monkeypatch.setattr(score_text_scan, 'embedding_ready_keys', lambda *_:set())
    push.return_value = {'success':False}
    with pytest.raises(RuntimeError):
        run()
    c = connect()
    assert scan.health(c, 1, 3)['causes'] == {'scoring_delivery_failure':10, 'embedding_budget':2, 'text_budget':1}


def test_health_warns_on_old_retry_and_stalled_cursor(repair_env, caplog):
    connect,page,*_ = repair_env
    c = connect()
    scan.admit(c, 1, 3, 'next', [('email',7)], now=1)
    c.close()
    run()
    assert 'old_retry=0' in caplog.text  # Successful removal eliminates the old retry condition.
    assert 'stalled_cursor=1' in caplog.text


def test_failure_logging_does_not_mask_original_exception(repair_env, monkeypatch):
    connect,page,pull,scorer,push = repair_env
    page.return_value['keys'] = [('email',7)]
    pull.side_effect = RuntimeError('original transport failure')
    monkeypatch.setattr(scan, 'outcome', Mock(side_effect=RuntimeError('telemetry broke')))
    with pytest.raises(RuntimeError, match='original transport failure'):
        run()


def test_failed_old_retry_warns_even_on_exception(repair_env, caplog):
    connect,page,pull,*_ = repair_env
    c = connect()
    scan.admit(c, 1, 3, 'next', [('email',7)], now=1)
    c.close()
    pull.side_effect = RuntimeError('transport failure')
    with pytest.raises(RuntimeError):
        run()
    assert 'old_retry=1' in caplog.text
    assert 'stalled_cursor=1' in caplog.text


def test_telemetry_and_cursor_admission_roll_back_together(tmp_path):
    c = sqlite3.connect(tmp_path/'repair.sqlite')
    scan.admit(c, 1, 3, 'prior', [('email',7)], now=100)
    c.execute("CREATE TRIGGER fail_telemetry BEFORE INSERT ON worker_score_repair_retry BEGIN SELECT RAISE(ABORT,'synthetic telemetry failure'); END")
    c.commit()
    with pytest.raises(sqlite3.IntegrityError):
        scan.admit(c, 1, 3, 'next', [('email',8)], now=200)
    assert scan.load(c, 1, 3) == {'cursor':'prior', 'pending':[('email',7)]}
    assert scan.health(c, 1, 3, now=300)['max_attempts'] == 1


def test_saturated_queue_warns_and_retains_full_queue(repair_env, caplog):
    connect,page,pull,*_ = repair_env
    c = connect()
    scan.load(c,1,3)
    keys = [('email',i) for i in range(1,201)]
    c.execute('INSERT INTO worker_score_repair_scan VALUES(1,3,?,?)', ('prior',json.dumps(keys)))
    c.commit();c.close()
    pull.side_effect = RuntimeError('transport failure')
    with pytest.raises(RuntimeError):
        run()
    page.assert_not_called()
    assert len(pending(connect)) == 200
    assert 'saturated=1' in caplog.text


def test_explicit_interruption_is_persisted_and_reraised(repair_env):
    connect,page,pull,*_ = repair_env
    page.return_value['keys'] = [('email',7)]
    pull.side_effect = KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        run()
    c = connect()
    assert scan.health(c, 1, 3)['causes'] == {'interrupted':1}


def test_promotions_do_not_reset_cursor_or_discard_pending_work(tmp_path):
    c = sqlite3.connect(tmp_path/'repair.sqlite')
    scan.admit(c, 1, 3, 'interior-page', [('email',7)], now=100)
    scan.outcome(c, 1, 3, [('email',7)], 'embedding_budget')
    for model in range(4, 10):
        assert scan.load(c, 1, model) == {'cursor':'interior-page', 'pending':[('email',7)]}
        scan.admit(c, 1, model, 'interior-page', [], now=100+model)
    h = scan.health(c, 1, 9, now=200)
    assert h['oldest_seconds'] == 100
    assert h['max_attempts'] == 7
    assert h['cursor_stale_seconds'] == 100


def test_stale_model_acknowledgement_cannot_remove_new_model_retry(tmp_path):
    c = sqlite3.connect(tmp_path/'repair.sqlite')
    scan.admit(c,1,3,'prior',[('email',7)],now=100)
    scan.admit(c,1,4,'next',[],now=200)
    with pytest.raises(RuntimeError, match='Stale model'):
        scan.complete(c,1,3,[('email',7)])
    assert scan.load(c,1,4) == {'cursor':'next','pending':[('email',7)]}
    scan.outcome(c,1,3,[('email',7)],'refetch_failure')
    assert scan.health(c,1,4,now=300)['causes'] == {'interrupted':1}
    scan.complete(c,1,4,[('email',7)])
    assert scan.load(c,1,4)['pending'] == []


def test_promotion_adoption_and_age_preservation_are_atomic(tmp_path):
    c = sqlite3.connect(tmp_path/'repair.sqlite')
    scan.admit(c,1,3,'prior',[('email',7)],now=100)
    c.execute("CREATE TRIGGER fail_adoption BEFORE UPDATE ON worker_score_repair_retry BEGIN SELECT RAISE(ABORT,'synthetic adoption failure'); END")
    c.commit()
    with pytest.raises(sqlite3.IntegrityError):
        scan.admit(c,1,4,'next',[],now=200)
    assert c.execute('SELECT model_version,cursor FROM worker_score_repair_scan').fetchone() == (3,'prior')
    assert scan.health(c,1,3,now=300)['oldest_seconds'] == 200


def test_pending_work_is_scored_with_new_model_after_promotion(repair_env, monkeypatch):
    connect,page,pull,scorer,push = repair_env
    c = connect()
    scan.admit(c,1,2,'interior-page',[('email',7)],now=100)
    c.close()
    observed = []
    def score_and_ack(profile, prof, model, rows, **kwargs):
        observed.append(model['version'])
        kwargs['on_acknowledged']([(r['entry_type'],r['entry_id']) for r in rows])
        return {'acknowledged':len(rows)}
    monkeypatch.setattr(ml_window,'_score_and_push_changed_entries',score_and_ack)
    run()
    assert page.call_args.kwargs['cursor'] == 'interior-page'
    assert observed == [3]
    assert pending(connect) == []


def test_new_model_sees_retry_age_before_successful_adoption(tmp_path):
    c = sqlite3.connect(tmp_path/'repair.sqlite')
    scan.admit(c,1,3,'interior',[('email',7)],now=100)
    scan.outcome(c,1,3,[('email',7)],'embedding_budget')
    h = scan.health(c,1,4,now=200)
    assert h['pending'] == 1
    assert h['oldest_seconds'] == 100
    assert h['cursor_stale_seconds'] == 100
    assert h['causes'] == {'embedding_budget':1}
    assert c.execute('SELECT model_version FROM worker_score_repair_scan').fetchone()[0] == 3


def test_repair_rounds_share_embedding_quota_and_attempt_each_key_once(repair_env,monkeypatch):
    connect,page,pull,scorer,push = repair_env
    monkeypatch.setattr(score_text_scan,'embedding_ready_keys',lambda *_:set())
    context = {'attempted':set(),'embedding_used':0}
    for round_id in range(4):
        page.return_value = {'keys':[('email',i) for i in range(1+20*round_id,21+20*round_id)],
                             'sweep_cursor':'page-'+str(round_id),'sweep_examined':200,'sweep_completed':False}
        ml_window._repair_remote_scores(1,{}, {'version':3},context=context)
    assert context['embedding_used'] == 10
    assert len(context['attempted']) == 80
    assert scorer.call_count == 1
    assert len(scorer.call_args.args[0]) == 10
    c = connect()
    assert len(scan.load(c,1,3)['pending']) == 70
    assert scan.health(c,1,3)['max_attempts'] == 1
    assert scan.health(c,1,3)['causes'] == {'embedding_budget':70}


def test_cached_embeddings_allow_four_fifty_key_pages_without_repeat_delivery(repair_env):
    connect,page,pull,scorer,push = repair_env
    context = {'attempted':set(),'embedding_used':0}
    for round_id in range(4):
        page.return_value = {'keys':[('email',i) for i in range(1+50*round_id,51+50*round_id)],
                             'sweep_cursor':'page-'+str(round_id),'sweep_examined':200,'sweep_completed':False}
        ml_window._repair_remote_scores(1,{}, {'version':3},context=context)
    assert len(context['attempted']) == 200
    assert context['embedding_used'] == 0
    assert pending(connect) == []
    assert push.call_count == 4
    assert len({(r['entry_type'],r['entry_id']) for call in scorer.call_args_list for r in call.args[0]}) == 200


def test_excluded_retries_keep_position_age_and_capacity(tmp_path):
    c = sqlite3.connect(tmp_path/'repair.sqlite')
    keys=[('email',i) for i in range(1,21)]
    scan.admit(c,1,3,'prior',keys,now=100)
    assert scan.admit(c,1,3,'next',[],now=200,exclude=set(keys[:10])) == keys[10:]
    assert set(scan.load(c,1,3)['pending']) == set(keys)
    h=scan.health(c,1,3,now=300)
    assert h['oldest_seconds'] == 200
    assert h['max_attempts'] == 2
