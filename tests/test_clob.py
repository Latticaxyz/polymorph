from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import polars as pl
import pytest

from polymorph import __version__
from polymorph.config import config as base_config
from polymorph.core.base import PipelineContext, RuntimeConfig
from polymorph.models.api import OrderBook, OrderBookLevel
from polymorph.sources.clob import CLOB
from polymorph.sources.gamma import Gamma
from polymorph.utils.constants import CLOB_MAX_PRICE_HISTORY_MS
from polymorph.utils.time import utc


def _make_context(tmp_path: Path) -> PipelineContext:
    """Create a test pipeline context."""
    runtime_cfg = RuntimeConfig(http_timeout=None, max_concurrency=None, data_dir=str(tmp_path))
    return PipelineContext(
        config=base_config,
        run_timestamp=utc(),
        data_dir=tmp_path,
        runtime_config=runtime_cfg,
    )


@pytest.mark.asyncio
async def test_clob_fetch_price_history_single_chunk(tmp_path: Path) -> None:
    """Test fetching price history for a single time chunk."""
    context = _make_context(tmp_path)
    clob = CLOB(context, clob_base_url="https://example.test", data_api_url="https://example-data.test")

    # Use valid millisecond timestamps (2020-01-01 range)
    t1 = 1577836800000  # 2020-01-01 00:00:00
    t2 = 1577836800001  # 2020-01-01 00:00:00.001

    async def fake_get(
        url: str, params: dict[str, int | str | bool], use_data_api: bool = False
    ) -> dict[str, list[dict[str, str | int | float]]]:
        """Mock HTTP GET that returns price data."""
        _ = use_data_api
        assert "prices-history" in url
        assert params["market"] == "YES"
        return {
            "history": [
                {"t": t1, "p": 0.1},
                {"t": t2, "p": 0.2},
            ]
        }

    clob._get = fake_get  # type: ignore[method-assign]

    df = await clob.fetch_prices_history("YES", start_ts=t1, end_ts=t2, fidelity=60)

    assert df.height == 2
    assert set(df.columns) >= {"t", "p", "token_id"}
    assert set(df["token_id"].to_list()) == {"YES"}

    timestamps = df["t"].to_list()
    prices = df["p"].to_list()
    assert timestamps == [t1, t2]
    assert prices == [0.1, 0.2]


@pytest.mark.asyncio
async def test_clob_fetch_price_history_sends_seconds_to_api(tmp_path: Path) -> None:
    """Test that timestamps are converted to seconds when sent to API."""
    context = _make_context(tmp_path)
    clob = CLOB(context, clob_base_url="https://example.test", data_api_url="https://example-data.test")

    start_ts_ms = 1577836800000  # 2020-01-01 00:00:00 in milliseconds
    end_ts_ms = 1577923200000  # 2020-01-02 00:00:00 in milliseconds
    expected_start_s = 1577836800  # Same timestamp in seconds
    expected_end_s = 1577923200

    received_params: dict[str, int | str | bool] = {}

    async def fake_get(
        url: str, params: dict[str, int | str | bool], use_data_api: bool = False
    ) -> dict[str, list[dict[str, str | int | float]]]:
        nonlocal received_params
        received_params = dict(params)
        return {"history": [{"t": expected_start_s, "p": 0.5}]}

    clob._get = fake_get  # type: ignore[method-assign]

    await clob.fetch_prices_history("TOKEN", start_ts=start_ts_ms, end_ts=end_ts_ms, fidelity=60)

    assert (
        received_params["startTs"] == expected_start_s
    ), f"API should receive seconds, not milliseconds. Got {received_params['startTs']}"
    assert (
        received_params["endTs"] == expected_end_s
    ), f"API should receive seconds, not milliseconds. Got {received_params['endTs']}"


@pytest.mark.asyncio
async def test_clob_fetch_price_history_chunking(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that long time spans are split into multiple chunks."""
    context = _make_context(tmp_path)
    clob = CLOB(context)

    calls: list[tuple[int, int]] = []

    async def fake_chunk(token_id: str, start_ts: int, end_ts: int, fidelity: int) -> pl.DataFrame:
        """Mock chunk fetcher that records calls."""
        _ = fidelity
        calls.append((start_ts, end_ts))
        return pl.DataFrame({"t": [start_ts], "p": [1.0], "token_id": [token_id]})

    monkeypatch.setattr(clob, "_fetch_price_history_chunk", fake_chunk)  # type: ignore[arg-type]

    # Use milliseconds (2x the max window + some extra)
    start_ts = 1577836800000  # 2020-01-01
    span_ms = CLOB_MAX_PRICE_HISTORY_MS * 2 + 10000  # > 28 days
    end_ts = start_ts + span_ms

    df = await clob.fetch_prices_history("YES", start_ts=start_ts, end_ts=end_ts, fidelity=60)

    assert len(calls) >= 2, f"Expected multiple chunks for long time span, got {len(calls)} calls: {calls}"
    assert df.height == len(calls), "Should have one row per chunk"
    assert set(df["token_id"].to_list()) == {"YES"}

    all_start_times = [start for start, _ in calls]
    all_end_times = [end for _, end in calls]
    assert min(all_start_times) == start_ts, f"First chunk should start at {start_ts}"
    assert max(all_end_times) >= end_ts, f"Last chunk should cover end_ts {end_ts}"


@pytest.mark.asyncio
async def test_clob_fetch_price_history_with_interval(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test fetching price history using interval parameter instead of timestamps."""
    context = _make_context(tmp_path)
    clob = CLOB(context)

    t1 = 1577836800000
    t2 = 1577923200000

    async def fake_get(
        url: str, params: dict[str, int | str | bool], use_data_api: bool = False
    ) -> dict[str, list[dict[str, str | int | float]]]:
        """Mock HTTP GET that returns price data."""
        _ = use_data_api
        assert "prices-history" in url
        assert params["market"] == "YES"
        assert params["interval"] == "1w"
        assert params["fidelity"] == 60
        return {
            "history": [
                {"t": t1, "p": 0.3},
                {"t": t2, "p": 0.7},
            ]
        }

    monkeypatch.setattr(clob, "_get", fake_get)

    df = await clob.fetch_prices_history("YES", interval="1w", fidelity=60)

    assert df.height == 2
    assert set(df.columns) >= {"t", "p", "token_id"}
    assert set(df["token_id"].to_list()) == {"YES"}
    timestamps = df["t"].to_list()
    prices = df["p"].to_list()
    assert timestamps == [t1, t2]
    assert prices == [0.3, 0.7]


@pytest.mark.asyncio
async def test_clob_fetch_price_history_interval_options(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test various interval options (all, max, 1d, 1w, etc.)."""
    context = _make_context(tmp_path)
    clob = CLOB(context)

    intervals_tested: list[str] = []

    async def fake_get(
        url: str, params: dict[str, int | str | bool], use_data_api: bool = False
    ) -> dict[str, list[dict[str, str | int | float]]]:
        """Mock HTTP GET that tracks interval values."""
        _ = use_data_api
        intervals_tested.append(str(params["interval"]))
        return {"history": [{"t": 1577836800000, "p": 0.5}]}

    monkeypatch.setattr(clob, "_get", fake_get)

    test_intervals = ["all", "max", "1d", "1w", "1m"]
    for interval in test_intervals:
        await clob.fetch_prices_history("TOKEN", interval=interval)

    assert intervals_tested == test_intervals, "Should test all interval types"


@pytest.mark.asyncio
async def test_clob_fetch_price_history_requires_either_interval_or_timestamps(
    tmp_path: Path,
) -> None:
    """Test that fetch_prices_history requires either interval or timestamps."""
    context = _make_context(tmp_path)
    clob = CLOB(context)

    with pytest.raises(ValueError, match="Either 'interval' or both 'start_ts' and 'end_ts' must be provided"):
        await clob.fetch_prices_history("TOKEN")

    with pytest.raises(ValueError, match="Either 'interval' or both 'start_ts' and 'end_ts' must be provided"):
        await clob.fetch_prices_history("TOKEN", start_ts=1000000)

    with pytest.raises(ValueError, match="Either 'interval' or both 'start_ts' and 'end_ts' must be provided"):
        await clob.fetch_prices_history("TOKEN", end_ts=2000000)


@pytest.mark.asyncio
async def test_clob_fetch_trades_parses_created_at_and_filters(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that trades are parsed from created_at and filtered by timestamp."""
    context = _make_context(tmp_path)
    clob = CLOB(context)

    t1 = 1704067200000
    t2 = 1704153600000
    mid_ts = (t1 + t2) // 2

    async def fake_paged(
        limit: int,
        offset: int,
        market_ids: list[str] | None = None,
    ) -> list[dict[str, str | int | float]]:
        _ = market_ids
        assert limit == 1000
        assert offset == 0
        return [
            {"timestamp": t1, "size": 1, "price": 0.4, "conditionId": "c1"},
            {"timestamp": t2, "size": 2, "price": 0.5, "conditionId": "c2"},
        ]

    monkeypatch.setattr(clob, "fetch_trades_paged", fake_paged)  # type: ignore[arg-type]

    df = await clob.fetch_trades(market_ids=None, since_ts=mid_ts)

    # Verify filtering
    assert df.height == 1, "Should filter to trades after mid_ts"
    assert "timestamp" in df.columns, "Should have timestamp column"

    # Type-safe timestamp checking
    min_timestamp = df["timestamp"].min()
    assert isinstance(min_timestamp, int), "Timestamp should be an integer"
    assert min_timestamp >= mid_ts, f"Min timestamp {min_timestamp} should be >= {mid_ts}"

    # Verify correct trade was kept
    assert set(df["conditionId"].to_list()) == {"c2"}, "Should keep only the second trade"

    # Verify trade data (stored as Float64)
    assert df["size"].to_list() == [2.0]
    assert df["price"].to_list() == [0.5]


@pytest.mark.asyncio
async def test_clob_orderbook_dataframe_and_spread(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test converting orderbook to DataFrame and calculating spread."""
    context = _make_context(tmp_path)
    clob = CLOB(context)

    async def fake_fetch_orderbook(token_id: str) -> OrderBook:
        """Mock orderbook fetcher with realistic data."""
        bids = [OrderBookLevel(price=0.4, size=10.0)]
        asks = [OrderBookLevel(price=0.6, size=5.0)]
        ob = OrderBook(
            token_id=token_id,
            timestamp=1577836800000,
            bids=bids,
            asks=asks,
            best_bid=0.4,
            best_ask=0.6,
        )
        ob.mid_price = ob.calculate_mid_price()
        ob.spread = ob.calculate_spread()
        return ob

    monkeypatch.setattr(clob, "fetch_orderbook", fake_fetch_orderbook)  # type: ignore[arg-type]

    # Test DataFrame conversion
    df = await clob.fetch_orderbooks(["YES"])
    assert set(df.columns) == {"token_id", "timestamp", "side", "price", "size"}
    assert df.height == 2, "Should have 2 rows (1 bid + 1 ask)"
    assert set(df["side"].to_list()) == {"bid", "ask"}

    # Verify bid data
    bid_rows = df.filter(pl.col("side") == "bid")
    assert bid_rows.height == 1
    assert bid_rows["price"].to_list() == [0.4]
    assert bid_rows["size"].to_list() == [10.0]

    # Verify ask data
    ask_rows = df.filter(pl.col("side") == "ask")
    assert ask_rows.height == 1
    assert ask_rows["price"].to_list() == [0.6]
    assert ask_rows["size"].to_list() == [5.0]

    # Test spread calculation
    spread = await clob.fetch_spread("YES")
    assert spread["token_id"] == "YES"
    assert spread["bid"] == pytest.approx(0.4)
    assert spread["ask"] == pytest.approx(0.6)
    assert spread["mid"] == pytest.approx(0.5)
    assert spread["spread"] == pytest.approx(0.2)
    assert "timestamp" in spread


# ============================================================================
# EDGE CASE TESTS
# ============================================================================


@pytest.mark.asyncio
async def test_clob_fetch_orderbook_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test handling of completely empty orderbook with no bids or asks."""
    context = _make_context(tmp_path)
    clob = CLOB(context)

    async def fake_get(
        url: str, params: dict[str, int | str | bool], use_data_api: bool = False
    ) -> dict[str, list[dict[str, float]] | int]:
        """Mock GET returning empty orderbook."""
        return {"bids": [], "asks": [], "timestamp": 1577836800000}

    monkeypatch.setattr(clob, "_get", fake_get)

    orderbook = await clob.fetch_orderbook("EMPTY")

    assert orderbook.token_id == "EMPTY"
    assert len(orderbook.bids) == 0
    assert len(orderbook.asks) == 0
    assert orderbook.best_bid is None
    assert orderbook.best_ask is None
    assert orderbook.mid_price is None
    assert orderbook.spread is None


@pytest.mark.asyncio
async def test_clob_fetch_orderbook_malformed_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test handling of malformed orderbook data with missing/invalid fields."""
    context = _make_context(tmp_path)
    clob = CLOB(context)

    async def fake_get(
        url: str, params: dict[str, int | str | bool], use_data_api: bool = False
    ) -> dict[str, list[dict[str, object]] | str]:
        """Mock GET returning malformed orderbook with invalid price/size types."""
        return {
            "bids": [
                {"price": "invalid", "size": 10.0},  # Invalid price
                {"price": 0.5, "size": None},  # Invalid size
                {"price": 0.4},  # Missing size
            ],
            "asks": [
                {"size": 5.0},  # Missing price
                {"price": 0.7, "size": "bad"},  # Invalid size type
            ],
            "timestamp": "not_an_int",  # Invalid timestamp
        }

    monkeypatch.setattr(clob, "_get", fake_get)

    orderbook = await clob.fetch_orderbook("MALFORMED")

    # Should skip invalid entries and only parse valid ones
    assert orderbook.token_id == "MALFORMED"
    assert len(orderbook.bids) == 0, "All bid entries were malformed"
    assert len(orderbook.asks) == 0, "All ask entries were malformed"
    assert orderbook.timestamp == 0, "Invalid timestamp should default to 0"


@pytest.mark.asyncio
async def test_clob_fetch_price_history_empty_response(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test handling of empty price history response."""
    context = _make_context(tmp_path)
    clob = CLOB(context)

    async def fake_get(
        url: str, params: dict[str, int | str | bool], use_data_api: bool = False
    ) -> dict[str, list[dict[str, int | float]]]:
        """Mock GET returning empty history dict."""
        return {"history": []}

    monkeypatch.setattr(clob, "_get", fake_get)

    df = await clob.fetch_prices_history("EMPTY", start_ts=0, end_ts=100, fidelity=60)

    assert df.height == 0, "Empty response should return empty DataFrame"
    assert isinstance(df, pl.DataFrame), "Should still return DataFrame type"


@pytest.mark.asyncio
async def test_clob_fetch_price_history_deduplicates_timestamps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that duplicate timestamps are deduplicated when chunking."""
    context = _make_context(tmp_path)
    clob = CLOB(context)

    call_count = 0

    async def fake_chunk(token_id: str, start_ts: int, end_ts: int, fidelity: int) -> pl.DataFrame:
        """Mock chunk fetcher returning overlapping timestamps."""
        nonlocal call_count
        call_count += 1

        if call_count == 1:
            return pl.DataFrame({"t": [0, 50, 100], "p": [0.5, 0.6, 0.7], "token_id": [token_id] * 3})
        else:
            return pl.DataFrame({"t": [100, 150, 200], "p": [0.7, 0.8, 0.9], "token_id": [token_id] * 3})

    monkeypatch.setattr(clob, "_fetch_price_history_chunk", fake_chunk)

    # Request long enough to trigger chunking
    span = CLOB_MAX_PRICE_HISTORY_MS * 2
    df = await clob.fetch_prices_history("TOKEN", start_ts=0, end_ts=span, fidelity=60)

    # Verify deduplication occurred (line 153 in clob.py)
    timestamps = df["t"].to_list()
    assert len(timestamps) == len(set(timestamps)), "Timestamps should be deduplicated"
    assert timestamps == sorted(timestamps), "Timestamps should be in order"


@pytest.mark.asyncio
async def test_clob_fetch_trades_stops_at_max_trades(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that fetch_trades respects max_trades limit."""
    context = _make_context(tmp_path)
    clob = CLOB(context, max_trades=2500)  # Set low limit

    call_count = 0

    async def fake_paged(
        limit: int, offset: int, market_ids: list[str] | None = None
    ) -> list[dict[str, str | int | float]]:
        """Mock that always returns full pages."""
        nonlocal call_count
        call_count += 1

        # Use integer timestamps in milliseconds
        base_ts = 1704067200000  # 2025-01-01 in milliseconds
        return [
            {
                "timestamp": base_ts + offset + i,
                "size": 1.0,
                "price": 0.5,
                "conditionId": f"c{i}",
            }
            for i in range(limit)
        ]

    monkeypatch.setattr(clob, "fetch_trades_paged", fake_paged)

    df = await clob.fetch_trades(market_ids=None, since_ts=None)

    # Should stop at max_trades even though pages still have data
    assert df.height <= clob.max_trades, "Should respect max_trades limit"
    assert call_count <= 4, "Should stop pagination early due to max_trades"


@pytest.mark.asyncio
async def test_clob_fetch_trades_filters_by_since_ts_boundary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test exact boundary behavior of since_ts filtering."""
    context = _make_context(tmp_path)
    clob = CLOB(context)

    # Use timestamp that matches the middle trade (2025-01-12T13:46:40)
    boundary_ts = 1736689600000  # This is the exact timestamp for the "exact" trade in milliseconds

    async def fake_paged(
        limit: int, offset: int, market_ids: list[str] | None = None
    ) -> list[dict[str, str | int | float]]:
        """Mock returning trades at and around boundary."""
        return [
            {"timestamp": 1704067200000, "size": 1.0, "price": 0.5, "conditionId": "before"},
            {
                "timestamp": 1736689600000,
                "size": 2.0,
                "price": 0.6,
                "conditionId": "exact",
            },
            {"timestamp": 1736689601000, "size": 3.0, "price": 0.7, "conditionId": "after"},
        ]

    monkeypatch.setattr(clob, "fetch_trades_paged", fake_paged)

    df = await clob.fetch_trades(market_ids=None, since_ts=boundary_ts)

    # Filter at line 356: >= since_ts, so should include boundary
    assert df.height == 2, "Should include trades at exact boundary and after"
    condition_ids = set(df["conditionId"].to_list())
    assert "exact" in condition_ids, "Should include trade at exact boundary timestamp"
    assert "after" in condition_ids, "Should include trade after boundary"
    assert "before" not in condition_ids, "Should exclude trade before boundary"


@pytest.mark.asyncio
async def test_clob_fetch_trades_batches_market_ids(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that fetch_trades batches market_ids to avoid URL length issues."""
    context = _make_context(tmp_path)
    clob = CLOB(context, max_trades=200000)

    received_market_ids: list[list[str] | None] = []

    async def fake_paged(
        limit: int, offset: int, market_ids: list[str] | None = None
    ) -> list[dict[str, str | int | float]]:
        received_market_ids.append(market_ids)
        if offset > 0:
            return []
        return [
            {"timestamp": 1704067200000, "size": 1.0, "price": 0.5, "conditionId": "c1"},
        ]

    monkeypatch.setattr(clob, "fetch_trades_paged", fake_paged)

    long_market_ids = [f"0x{'a' * 64}_{i:04d}" for i in range(120)]
    await clob.fetch_trades(market_ids=long_market_ids, since_ts=None)

    assert len(received_market_ids) >= 3, "Should batch 120 IDs into multiple calls (50 per batch)"
    for batch in received_market_ids:
        if batch is not None:
            assert len(batch) <= 50, "Each batch should have at most 50 market IDs"


# ============================================================================
# HEADER TESTS
# ============================================================================


@pytest.mark.asyncio
async def test_clob_client_has_user_agent_header(tmp_path: Path) -> None:
    """Test that CLOB client includes User-Agent header."""
    context = _make_context(tmp_path)
    clob = CLOB(context)

    client = await clob._get_client()

    user_agent = client.headers.get("User-Agent")
    assert user_agent is not None, "User-Agent header should be set"
    assert "polymorph" in user_agent.lower(), "User-Agent should contain 'polymorph'"

    await clob.close()


@pytest.mark.asyncio
async def test_user_agent_format(tmp_path: Path) -> None:
    """Test that User-Agent header follows expected format."""
    context = _make_context(tmp_path)
    clob = CLOB(context)

    client = await clob._get_client()

    user_agent = client.headers.get("User-Agent")
    assert user_agent is not None

    assert f"polymorph/{__version__}" in user_agent, "User-Agent should include version number"
    assert "httpx" in user_agent.lower(), "User-Agent should mention httpx"

    await clob.close()


@pytest.mark.asyncio
async def test_gamma_client_has_user_agent_header(tmp_path: Path) -> None:
    """Test that Gamma client includes User-Agent header for consistency."""
    context = _make_context(tmp_path)
    gamma = Gamma(context)

    client = await gamma._get_client()

    user_agent = client.headers.get("User-Agent")
    assert user_agent is not None, "User-Agent header should be set"
    assert f"polymorph/{__version__}" in user_agent, "User-Agent should include version number"
    assert "httpx" in user_agent.lower(), "User-Agent should mention httpx"

    await gamma.__aexit__(None, None, None)


# ============================================================================
# RETRY TESTS
# ============================================================================


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
                {"t": 1577836800000, "p": 0.5},
                {"t": 1577836800001, "p": 0.6},
            ]
        }

    monkeypatch.setattr(clob, "_get", fake_get)

    df = await clob._fetch_price_history_chunk("TEST_TOKEN", 1577836800000, 1577836800001, 60)

    assert call_count == 2, "_fetch_price_history_chunk should still have retry"
    assert df.height == 2
