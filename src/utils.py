from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, List

from dateutil.relativedelta import relativedelta

ISO = "%Y-%m-%dT%H:%M:%SZ"


def months_ago_utc(n: int) -> datetime:
    now = datetime.now(timezone.utc)
    return (now - relativedelta(months=n)).replace(microsecond=0)


def to_iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime(ISO)


def chunked(seq: List[str], size: int) -> Iterable[List[str]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


@dataclass
class RateLimiter:
    rps: float = 5.0

    def sleep(self):
        time.sleep(1.0 / self.rps)
