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
