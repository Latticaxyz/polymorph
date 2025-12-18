from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import pytest

from polymorph.config import config as base_config
from polymorph.core.base import PipelineContext, RuntimeConfig
from polymorph.sources.clob import CLOB
from polymorph.utils.time import utc


def _make_context(tmp_path: Path) -> PipelineContext:
    runtime_cfg = RuntimeConfig(http_timeout=None, max_concurrency=None, data_dir=str(tmp_path))
    return PipelineContext(
        config=base_config,
        run_timestamp=utc(),
        data_dir=tmp_path,
        runtime_config=runtime_cfg,
    )


@pytest.mark.asyncio
async def test_fetch_orderbook_retries_on_network_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that fetch_orderbook retries on network errors."""
    context = _make_context(tmp_path)
    clob = CLOB(context)

    call_count = 0

    async def fake_get(
        url: str, params: dict[str, str | int | float | bool], use_data_api: bool = False
    ) -> dict[str, list[dict[str, str | int]] | int]:
        nonlocal call_count
        call_count += 1

        if call_count < 3:
            raise httpx.ConnectError("Connection refused")

        return {
            "bids": [{"price": "0.5", "size": "100"}],
            "asks": [{"price": "0.6", "size": "50"}],
            "timestamp": 1704067200000,
        }

    monkeypatch.setattr(clob, "_get", fake_get)

    orderbook = await clob.fetch_orderbook("TEST_TOKEN")

    assert call_count == 3, "Should retry twice then succeed"
    assert orderbook.token_id == "TEST_TOKEN"
    assert orderbook.best_bid == 0.5
    assert orderbook.best_ask == 0.6


@pytest.mark.asyncio
async def test_fetch_orderbook_retries_on_server_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that fetch_orderbook retries on 5xx server errors."""
    context = _make_context(tmp_path)
    clob = CLOB(context)

    call_count = 0

    async def fake_get(
        url: str, params: dict[str, str | int | float | bool], use_data_api: bool = False
    ) -> dict[str, list[dict[str, str | int]] | int]:
        nonlocal call_count
        call_count += 1

        if call_count < 3:
            request = httpx.Request("GET", url)
            response = httpx.Response(503, request=request)
            raise httpx.HTTPStatusError("Service Unavailable", request=request, response=response)

        return {
            "bids": [{"price": "0.4", "size": "200"}],
            "asks": [{"price": "0.7", "size": "100"}],
            "timestamp": 1704067200000,
        }

    monkeypatch.setattr(clob, "_get", fake_get)

    orderbook = await clob.fetch_orderbook("TEST_TOKEN")

    assert call_count == 3, "Should retry twice on 503 then succeed"
    assert orderbook.best_bid == 0.4


@pytest.mark.asyncio
async def test_fetch_orderbook_retries_on_rate_limit_429(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that fetch_orderbook retries on HTTP 429 rate limit responses."""
    context = _make_context(tmp_path)
    clob = CLOB(context)

    call_count = 0

    async def fake_get(
        url: str, params: dict[str, str | int | float | bool], use_data_api: bool = False
    ) -> dict[str, list[dict[str, str | int]] | int]:
        nonlocal call_count
        call_count += 1

        if call_count < 2:
            request = httpx.Request("GET", url)
            response = httpx.Response(429, request=request)
            raise httpx.HTTPStatusError("Too Many Requests", request=request, response=response)

        return {
            "bids": [{"price": "0.55", "size": "150"}],
            "asks": [{"price": "0.65", "size": "75"}],
            "timestamp": 1704067200000,
        }

    monkeypatch.setattr(clob, "_get", fake_get)

    orderbook = await clob.fetch_orderbook("TEST_TOKEN")

    assert call_count == 2, "Should retry once on 429 then succeed"
    assert orderbook.mid_price == pytest.approx(0.6)


@pytest.mark.asyncio
async def test_fetch_orderbook_does_not_retry_on_client_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that fetch_orderbook does NOT retry on 4xx client errors."""
    context = _make_context(tmp_path)
    clob = CLOB(context)

    call_count = 0

    async def fake_get(
        url: str, params: dict[str, str | int | float | bool], use_data_api: bool = False
    ) -> dict[str, list[dict[str, str | int]] | int]:
        nonlocal call_count
        call_count += 1

        request = httpx.Request("GET", url)
        response = httpx.Response(404, request=request)
        raise httpx.HTTPStatusError("Not Found", request=request, response=response)

    monkeypatch.setattr(clob, "_get", fake_get)

    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        await clob.fetch_orderbook("NONEXISTENT_TOKEN")

    assert call_count == 1, "Should NOT retry on 404 client error"
    assert exc_info.value.response.status_code == 404


@pytest.mark.asyncio
async def test_fetch_trades_paged_retries_on_network_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that fetch_trades_paged retries on network errors."""
    context = _make_context(tmp_path)
    clob = CLOB(context)

    call_count = 0

    async def fake_get(
        url: str, params: dict[str, str | int | float | bool], use_data_api: bool = True
    ) -> list[dict[str, str | int | float]]:
        nonlocal call_count
        call_count += 1

        if call_count < 3:
            raise httpx.RequestError("Connection reset by peer")

        return [
            {"timestamp": 1704067200000, "size": 1.0, "price": 0.5, "conditionId": "c1"},
            {"timestamp": 1704153600000, "size": 2.0, "price": 0.6, "conditionId": "c2"},
        ]

    monkeypatch.setattr(clob, "_get", fake_get)

    trades = await clob.fetch_trades_paged(limit=1000, offset=0)

    assert call_count == 3, "Should retry twice on network error then succeed"
    assert len(trades) == 2


@pytest.mark.asyncio
async def test_fetch_trades_paged_retries_on_timeout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that fetch_trades_paged retries on timeout errors."""
    context = _make_context(tmp_path)
    clob = CLOB(context)

    call_count = 0

    async def fake_get(
        url: str, params: dict[str, str | int | float | bool], use_data_api: bool = True
    ) -> list[dict[str, str | int | float]]:
        nonlocal call_count
        call_count += 1

        if call_count < 2:
            raise asyncio.TimeoutError("Request timed out")

        return [
            {"timestamp": 1704067200000, "size": 3.0, "price": 0.7, "conditionId": "c3"},
        ]

    monkeypatch.setattr(clob, "_get", fake_get)

    trades = await clob.fetch_trades_paged(limit=1000, offset=0)

    assert call_count == 2, "Should retry once on timeout then succeed"
    assert len(trades) == 1


@pytest.mark.asyncio
async def test_retry_exponential_backoff_timing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that retry logic uses exponential backoff correctly."""
    context = _make_context(tmp_path)
    clob = CLOB(context)

    sleep_times: list[float] = []

    original_sleep = asyncio.sleep

    async def track_sleep(seconds: float) -> None:
        sleep_times.append(seconds)
        await original_sleep(0.001)

    monkeypatch.setattr(asyncio, "sleep", track_sleep)

    call_count = 0

    async def fake_get(
        url: str, params: dict[str, str | int | float | bool], use_data_api: bool = False
    ) -> dict[str, list[dict[str, str | int]] | int]:
        nonlocal call_count
        call_count += 1

        if call_count < 3:
            raise httpx.ConnectError("Connection refused")

        return {
            "bids": [{"price": "0.5", "size": "100"}],
            "asks": [{"price": "0.6", "size": "50"}],
            "timestamp": 1704067200000,
        }

    monkeypatch.setattr(clob, "_get", fake_get)

    await clob.fetch_orderbook("TEST_TOKEN")

    assert len(sleep_times) >= 2, "Should have at least 2 retry delays"
    assert sleep_times[1] > sleep_times[0], "Second delay should be longer (exponential backoff)"
    assert all(t <= 10.0 for t in sleep_times), "All delays should be capped at max_wait (10s)"


@pytest.mark.asyncio
async def test_fetch_price_history_chunk_has_retry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression test: verify _fetch_price_history_chunk still has retry decorator."""
    context = _make_context(tmp_path)
    clob = CLOB(context)

    call_count = 0

    async def fake_get(
        url: str, params: dict[str, str | int | float | bool], use_data_api: bool = False
    ) -> dict[str, list[dict[str, int | str]]]:
        nonlocal call_count
        call_count += 1

        if call_count < 2:
            raise httpx.ConnectError("Connection refused")

        return {
            "history": [
                {"t": 1577836800000, "p": "0.5"},
                {"t": 1577836800001, "p": "0.6"},
            ]
        }

    monkeypatch.setattr(clob, "_get", fake_get)

    df = await clob._fetch_price_history_chunk("TEST_TOKEN", 1577836800000, 1577836800001, 60)

    assert call_count == 2, "_fetch_price_history_chunk should still have retry"
    assert df.height == 2
