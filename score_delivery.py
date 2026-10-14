"""Fair, finite extra fresh-score admission within an existing window budget."""
import time

MAX_FRESH_PAGES_PER_DESK = 100


def drain_fresh(desks, attempt, deadline, *, clock=time.monotonic):
    """One page per desk per round; never admit work after the deadline.

    attempt advances that desk's in-window descending cursor and returns false
    on exhaustion/failure. Each page owns its acknowledgement, not this loop.
    """
    active = list(desks)
    for _ in range(MAX_FRESH_PAGES_PER_DESK - 1):
        if not active:
            break
        remaining = []
        for desk in active:
            if clock() >= deadline:
                return
            if attempt(desk):
                remaining.append(desk)
        active = remaining


MAX_REPAIR_PAGES_PER_DESK = 4
FRESH_ROUNDS_BEFORE_REPAIR = 2


def drain_score_only(desks, fresh_attempt, repair_attempt, historical_attempt, deadline, *, clock=time.monotonic):
    """All desks already received initial fresh admission. Fresh work leads each repair round.

    Keep one normal historical checkpoint page after the first repair round,
    and admit at most four 50-key repair pages per desk. Each callback owns bounded text and durable acknowledgement.
    The same deadline gates every admission; no full-sweep drain is attempted.
    """
    fresh = list(desks)
    repair = list(desks)
    for round_id in range(MAX_REPAIR_PAGES_PER_DESK):
        for _ in range(FRESH_ROUNDS_BEFORE_REPAIR):
            remaining = []
            for item in fresh:
                if clock() >= deadline:
                    return
                if fresh_attempt(item):
                    remaining.append(item)
            fresh = remaining
        remaining = []
        for item in repair:
            if clock() >= deadline:
                return
            if repair_attempt(item):
                remaining.append(item)
        repair = remaining
        if round_id == 0:
            for item in desks:
                if clock() >= deadline:
                    return
                historical_attempt(item)
        if not repair:
            break
