"""
Display timestamps in Europe/Zurich.

Magnitu stores naive UTC strings from SQLite ``datetime('now')``. UI and
Seismo-facing metadata convert to local wall clock for Switzerland.
"""
from datetime import datetime, timezone
from typing import Optional
from zoneinfo import ZoneInfo

ZURICH_TZ = ZoneInfo("Europe/Zurich")


def parse_stored_timestamp(value: Optional[str]) -> Optional[datetime]:
    """Parse a timestamp from SQLite or ISO; naive values are treated as UTC."""
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None

    try:
        iso = s.replace("Z", "+00:00")
        if "T" in iso:
            dt = datetime.fromisoformat(iso)
            if dt.tzinfo is not None:
                return dt
            return dt.replace(tzinfo=timezone.utc)
    except ValueError:
        pass

    for fmt, length in (
        ("%Y-%m-%d %H:%M:%S", 19),
        ("%Y-%m-%dT%H:%M:%S", 19),
        ("%Y-%m-%d %H:%M", 16),
    ):
        try:
            dt = datetime.strptime(s[:length], fmt)
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def to_zurich(value: Optional[str]) -> Optional[datetime]:
    dt = parse_stored_timestamp(value)
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(ZURICH_TZ)


def format_zurich_datetime(value: Optional[str], with_tz_label: bool = True) -> str:
    """
    Human-readable Zurich time, e.g. ``23.05.2026 14:29 CEST``.
    """
    local = to_zurich(value)
    if local is None:
        return ""
    base = local.strftime("%d.%m.%Y %H:%M")
    if not with_tz_label:
        return base
    return "{} {}".format(base, local.strftime("%Z"))


def format_seismo_timestamp(value: Optional[str]) -> str:
    """
    Wall-clock string for Seismo recipe / model metadata (``YYYY-MM-DD HH:MM:SS``).
    """
    local = to_zurich(value)
    if local is None:
        return ""
    return local.strftime("%Y-%m-%d %H:%M:%S")


def utc_now_sql() -> str:
    """Naive UTC timestamp string for SQLite inserts."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
