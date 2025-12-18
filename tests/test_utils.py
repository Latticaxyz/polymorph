from __future__ import annotations

from datetime import datetime, timezone

import pytest

from polymorph.utils.time import datetime_to_ms, months_ago, ms_to_datetime, time_delta_ms, utc, utc_ms


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
    delta_months = (now.year - past.year) * 12 + (now.month - past.month)
    assert delta_months >= n


# ============================================================================
# TIME DELTA TESTS
# ============================================================================


def test_time_delta_ms_minutes() -> None:
    """Test time_delta_ms with minutes parameter."""
    now_ms = utc_ms()
    result = time_delta_ms(minutes=30)

    expected_ms = 30 * 60 * 1000
    diff = now_ms - result

    assert diff >= expected_ms - 1000, "Should be at least 30 minutes ago (within 1s tolerance)"
    assert diff <= expected_ms + 1000, "Should be at most 30 minutes ago (within 1s tolerance)"


def test_time_delta_ms_hours() -> None:
    """Test time_delta_ms with hours parameter."""
    now_ms = utc_ms()
    result = time_delta_ms(hours=2)

    expected_ms = 2 * 60 * 60 * 1000
    diff = now_ms - result

    assert diff >= expected_ms - 1000, "Should be at least 2 hours ago (within 1s tolerance)"
    assert diff <= expected_ms + 1000, "Should be at most 2 hours ago (within 1s tolerance)"


def test_time_delta_ms_days() -> None:
    """Test time_delta_ms with days parameter."""
    now_ms = utc_ms()
    result = time_delta_ms(days=7)

    expected_ms = 7 * 24 * 60 * 60 * 1000
    diff = now_ms - result

    assert diff >= expected_ms - 1000, "Should be at least 7 days ago (within 1s tolerance)"
    assert diff <= expected_ms + 1000, "Should be at most 7 days ago (within 1s tolerance)"


def test_time_delta_ms_weeks() -> None:
    """Test time_delta_ms with weeks parameter."""
    now_ms = utc_ms()
    result = time_delta_ms(weeks=2)

    expected_ms = 2 * 7 * 24 * 60 * 60 * 1000
    diff = now_ms - result

    assert diff >= expected_ms - 1000, "Should be at least 2 weeks ago (within 1s tolerance)"
    assert diff <= expected_ms + 1000, "Should be at most 2 weeks ago (within 1s tolerance)"


def test_time_delta_ms_months() -> None:
    """Test time_delta_ms with months parameter."""
    now_ms = utc_ms()
    result = time_delta_ms(months=3)

    now_dt = ms_to_datetime(now_ms)
    result_dt = ms_to_datetime(result)

    assert result_dt <= now_dt, "Result should be in the past"

    delta_months = (now_dt.year - result_dt.year) * 12 + (now_dt.month - result_dt.month)
    assert delta_months >= 3, "Should be at least 3 months ago"
    assert delta_months <= 3, "Should be exactly 3 months ago"


def test_time_delta_ms_years() -> None:
    """Test time_delta_ms with years parameter."""
    now_ms = utc_ms()
    result = time_delta_ms(years=1)

    expected_ms = 365 * 24 * 60 * 60 * 1000
    diff = now_ms - result

    assert diff >= expected_ms - 1000, "Should be at least 1 year (365 days) ago (within 1s tolerance)"
    assert diff <= expected_ms + 1000, "Should be at most 1 year (365 days) ago (within 1s tolerance)"


def test_time_delta_ms_zero_params() -> None:
    """Test time_delta_ms with no parameters returns current time."""
    now_ms = utc_ms()
    result = time_delta_ms()

    diff = abs(now_ms - result)
    assert diff < 1000, "With no params should return current time (within 1s)"


def test_time_delta_ms_combined_params() -> None:
    """Test time_delta_ms with multiple parameters combined."""
    now_ms = utc_ms()
    result = time_delta_ms(days=1, hours=2, minutes=30)

    expected_ms = (1 * 24 * 60 * 60 + 2 * 60 * 60 + 30 * 60) * 1000
    diff = now_ms - result

    assert diff >= expected_ms - 1000, "Should correctly combine time deltas (within 1s tolerance)"
    assert diff <= expected_ms + 1000, "Should correctly combine time deltas (within 1s tolerance)"


def test_time_delta_ms_months_with_other_params() -> None:
    """Test time_delta_ms with months combined with other parameters."""
    now_ms = utc_ms()
    result = time_delta_ms(months=1, days=5, hours=3)

    now_dt = ms_to_datetime(now_ms)
    result_dt = ms_to_datetime(result)

    assert result_dt <= now_dt, "Result should be in the past"

    delta_months = (now_dt.year - result_dt.year) * 12 + (now_dt.month - result_dt.month)
    assert delta_months >= 1, "Should be at least 1 month ago"

    time_diff_ms = now_ms - result
    expected_additional_ms = (5 * 24 * 60 * 60 + 3 * 60 * 60) * 1000

    assert time_diff_ms >= expected_additional_ms, "Should include additional days and hours beyond month"


def test_time_delta_ms_large_values() -> None:
    """Test time_delta_ms with large time values."""
    now_ms = utc_ms()
    result = time_delta_ms(months=12, weeks=4, days=3)

    now_dt = ms_to_datetime(now_ms)
    result_dt = ms_to_datetime(result)

    assert result_dt <= now_dt, "Result should be in the past"

    delta_months = (now_dt.year - result_dt.year) * 12 + (now_dt.month - result_dt.month)
    assert delta_months >= 12, "Should be at least 12 months ago"


# ============================================================================
# DATETIME CONVERSION TESTS
# ============================================================================


def test_datetime_to_ms_conversion() -> None:
    """Test datetime_to_ms converts correctly."""
    dt = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    result = datetime_to_ms(dt)

    assert isinstance(result, int), "Should return integer milliseconds"
    assert result == 1704067200000, "Should convert to correct millisecond timestamp"


def test_ms_to_datetime_conversion() -> None:
    """Test ms_to_datetime converts correctly."""
    ms = 1704067200000
    result = ms_to_datetime(ms)

    assert isinstance(result, datetime), "Should return datetime object"
    assert result.year == 2024, "Should have correct year"
    assert result.month == 1, "Should have correct month"
    assert result.day == 1, "Should have correct day"
    assert result.tzinfo == timezone.utc, "Should be UTC timezone"


def test_datetime_ms_roundtrip() -> None:
    """Test converting datetime to ms and back preserves value."""
    original = datetime(2024, 6, 15, 12, 30, 45, tzinfo=timezone.utc)
    ms = datetime_to_ms(original)
    result = ms_to_datetime(ms)

    assert result.year == original.year
    assert result.month == original.month
    assert result.day == original.day
    assert result.hour == original.hour
    assert result.minute == original.minute
    assert result.second == original.second
    assert result.tzinfo == original.tzinfo
