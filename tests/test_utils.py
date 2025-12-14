from __future__ import annotations

from datetime import timezone

import pytest

from polymorph.utils.time import months_ago, utc


def test_utc_returns_timezone_aware_utc() -> None:
    now = utc()
    assert now.tzinfo is not None
    assert now.utcoffset() == timezone.utc.utcoffset(now)


@pytest.mark.parametrize("n", [0, 1, 2, 11, 12, 13, 24, 120, 240])
def test_months_ago_not_in_future(n: int) -> None:
    past = months_ago(n)
    now = utc()
    assert past <= now
    assert past.tzinfo == now.tzinfo
    # Rough check that we moved at least n months back
    delta_months = (now.year - past.year) * 12 + (now.month - past.month)
    assert delta_months >= n
