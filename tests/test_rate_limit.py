from __future__ import annotations

from typing import cast

import httpx
import pytest

from polymorph.core.rate_limit import (
    CLOB_RATE_LIMIT,
    GAMMA_RATE_LIMIT,
    RateLimiter,
    RateLimitError,
)
from polymorph.core.retry import with_retry


@pytest.mark.asyncio
async def test_rate_limiter_singleton_instances() -> None:
    rl1 = await RateLimiter.get_instance("gamma", **GAMMA_RATE_LIMIT)
    rl2 = await RateLimiter.get_instance("gamma", **GAMMA_RATE_LIMIT)
    rl3 = await RateLimiter.get_instance("clob", **CLOB_RATE_LIMIT)
    assert rl1 is rl2
    assert rl1 is not rl3


@pytest.mark.asyncio
async def test_rate_limiter_throttles_requests() -> None:
    limiter = await RateLimiter.get_instance("test-throttle", max_requests=2, time_window_seconds=0.1)

    await limiter.acquire()
    await limiter.acquire()
    await limiter.acquire()

    stats = limiter.get_stats()
    current = cast(int, stats["current_count"])
    max_requests = cast(int, stats["max_requests"])
    assert current <= max_requests


@pytest.mark.asyncio
async def test_with_retry_retries_and_succeeds_on_rate_limit_error() -> None:
    call_count = 0

    @with_retry(max_attempts=3, min_wait=0.01, max_wait=0.05)
    async def sometimes_fails() -> str:
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise RateLimitError("temporary")
        return "ok"

    result = await sometimes_fails()
    assert result == "ok"
    assert call_count == 3


@pytest.mark.asyncio
async def test_with_retry_does_not_retry_on_client_error() -> None:
    @with_retry(max_attempts=5, min_wait=0.01, max_wait=0.05)
    async def always_400() -> None:
        request = httpx.Request("GET", "https://example.com")
        response = httpx.Response(400, request=request)
        raise httpx.HTTPStatusError("bad request", request=request, response=response)

    with pytest.raises(httpx.HTTPStatusError):
        await always_400()
