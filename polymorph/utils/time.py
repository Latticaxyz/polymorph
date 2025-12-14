from __future__ import annotations

from datetime import datetime, timezone


def utc() -> datetime:
    return datetime.now(timezone.utc)


def months_ago(n: int) -> datetime:
    dt = utc()
    month = dt.month - n
    year = dt.year + (month - 1) // 12
    month = ((month - 1) % 12) + 1
    return dt.replace(year=year, month=month)


def utc_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def months_ago_ms(n: int) -> int:
    dt = months_ago(n)
    return int(dt.timestamp() * 1000)


def datetime_to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def ms_to_datetime(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
