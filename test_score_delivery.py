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
