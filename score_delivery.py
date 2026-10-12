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
