from __future__ import annotations

from datetime import datetime, timedelta, timezone

from polymorph.utils.constants import MS_PER_SECOND


def utc() -> datetime:
    return datetime.now(timezone.utc)


def months_ago(n: int) -> datetime:
    dt = utc()
    month = dt.month - n
    year = dt.year + (month - 1) // 12
    month = ((month - 1) % 12) + 1
    return dt.replace(year=year, month=month)


def utc_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * MS_PER_SECOND)


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

    return int((dt - delta).timestamp() * MS_PER_SECOND)


def datetime_to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * MS_PER_SECOND)


def ms_to_datetime(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / MS_PER_SECOND, tz=timezone.utc)


def parse_iso_to_ms(iso_str: str) -> int:
    dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
    return int(dt.timestamp() * MS_PER_SECOND)


def parse_iso_to_ms_or_none(iso_str: str | None) -> int | None:
    if not iso_str:
        return None
    try:
        return parse_iso_to_ms(iso_str)
    except ValueError:
        return None
