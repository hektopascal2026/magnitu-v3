from collections import Counter
from unittest.mock import Mock
import score_delivery


def test_round_robin_more_than_one_page_with_no_work_or_failed_desk_removed():
    calls = []; counts = Counter()
    def attempt(desk):
        calls.append(desk); counts[desk] += 1
        return desk != 'failed' and counts[desk] < 3
    score_delivery.drain_fresh(['a', 'failed', 'b'], attempt, 10, clock=lambda: 0)
    assert calls == ['a', 'failed', 'b', 'a', 'b', 'a', 'b']


def test_deadline_prevents_next_desk_admission():
    attempt = Mock(return_value=True)
    times = iter([0, 0, 10])
    score_delivery.drain_fresh(['a', 'b', 'c'], attempt, 10, clock=lambda: next(times))
    assert [c.args[0] for c in attempt.call_args_list] == ['a', 'b']


def test_per_desk_page_cap_is_finite_even_when_clock_does_not_advance():
    attempt = Mock(return_value=True)
    score_delivery.drain_fresh(['a', 'b'], attempt, 10, clock=lambda: 0)
    assert attempt.call_count == 2 * (score_delivery.MAX_FRESH_PAGES_PER_DESK - 1)


def test_fresh_rounds_precede_each_repair_round_and_all_desks_get_admission():
    events = []
    fresh = lambda d: events.append(('fresh',d)) or True
    repair = lambda d: events.append(('repair',d)) or True
    historical = lambda d: events.append(('historical',d))
    score_delivery.drain_score_only(['a','b'],fresh,repair,historical,10,clock=lambda:0)
    assert events[:6] == [('fresh','a'),('fresh','b'),('fresh','a'),('fresh','b'),('repair','a'),('repair','b')]
    assert events[6:8] == [('historical','a'),('historical','b')]
    counts = Counter(events)
    assert counts[('repair','a')] == counts[('repair','b')] == 4
    assert counts[('historical','a')] == counts[('historical','b')] == 1
    assert counts[('fresh','a')] == counts[('fresh','b')] == 8


def test_deadline_during_fresh_pass_admits_no_repair_or_historical_work():
    repair,historical = Mock(),Mock()
    times = iter([0,10])
    score_delivery.drain_score_only(['a','b'],Mock(return_value=True),repair,historical,10,clock=lambda:next(times))
    repair.assert_not_called(); historical.assert_not_called()


def test_failed_repair_desk_does_not_block_other_repairs_or_fresh_scoring():
    calls = []
    repair = lambda d: calls.append(d) or d != 'failed'
    fresh = Mock(return_value=True)
    score_delivery.drain_score_only(['failed','good'],fresh,repair,Mock(),10,clock=lambda:0)
    assert calls == ['failed','good','good','good','good']
    assert fresh.call_count == 16


def test_completed_repair_sweep_is_not_restarted_in_same_window():
    repair = Mock(return_value=False)
    score_delivery.drain_score_only(['a'],Mock(return_value=False),repair,Mock(),10,clock=lambda:0)
    assert repair.call_count == 1
