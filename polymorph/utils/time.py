from __future__ import annotations

from datetime import datetime, timedelta, timezone


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


def time_delta_ms(
    minutes: int = 0,
    hours: int = 0,
    days: int = 0,
    weeks: int = 0,
    months: int = 0,
    years: int = 0,
) -> int:
    now = utc()

    if months > 0:
        dt = months_ago(months)
    else:
        dt = now

    delta = timedelta(
        minutes=minutes,
        hours=hours,
        days=days + (weeks * 7) + (years * 365),
    )

    return int((dt - delta).timestamp() * 1000)


def datetime_to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def ms_to_datetime(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


def parse_iso_to_ms(iso_str: str) -> int:
    dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
    return int(dt.timestamp() * 1000)
