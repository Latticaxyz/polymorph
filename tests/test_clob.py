from __future__ import annotations

from datetime import datetime
from pathlib import Path

import polars as pl
import pytest

from polymorph.config import config as base_config
from polymorph.core.base import PipelineContext, RuntimeConfig
from polymorph.models.api import OrderBook, OrderBookLevel
from polymorph.sources.clob import CLOB, MAX_PRICE_HISTORY_SECONDS
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

    async def fake_get(
        url: str, params: dict[str, int | str | bool], use_data_api: bool = False
    ) -> list[dict[str, str | int | float]]:
        """Mock HTTP GET that returns price data."""
        _ = use_data_api
        assert "prices-history" in url
        assert params["market"] == "YES"
        return [
            {"t": 1, "p": 0.1},
            {"t": 2, "p": 0.2},
        ]

    clob._get = fake_get  # type: ignore[method-assign]

    df = await clob.fetch_prices_history("YES", start_ts=0, end_ts=10, fidelity=60)

    # Verify structure
    assert df.height == 2
    assert set(df.columns) >= {"t", "p", "token_id"}
    assert set(df["token_id"].to_list()) == {"YES"}

    # Verify data
    timestamps = df["t"].to_list()
    prices = df["p"].to_list()
    assert timestamps == [1, 2]
    assert prices == [0.1, 0.2]


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

    span = MAX_PRICE_HISTORY_SECONDS * 2 + 10
    df = await clob.fetch_prices_history("YES", start_ts=0, end_ts=span, fidelity=60)

    assert len(calls) >= 2, "Expected multiple chunks for long time span"
    assert df.height == len(calls), "Should have one row per chunk"
    assert set(df["token_id"].to_list()) == {"YES"}

    all_start_times = [start for start, _ in calls]
    all_end_times = [end for _, end in calls]
    assert min(all_start_times) == 0, "First chunk should start at 0"
    assert max(all_end_times) >= span, "Last chunk should cover the end"


@pytest.mark.asyncio
async def test_clob_fetch_trades_parses_created_at_and_filters(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that trades are parsed from created_at and filtered by timestamp."""
    context = _make_context(tmp_path)
    clob = CLOB(context)

    created1 = "2025-01-01T00:00:00+00:00"
    created2 = "2025-01-02T00:00:00+00:00"
    t1 = int(datetime.fromisoformat(created1).timestamp())
    t2 = int(datetime.fromisoformat(created2).timestamp())
    mid_ts = (t1 + t2) // 2

    async def fake_paged(
        limit: int,
        offset: int,
        market_ids: list[str] | None = None,
    ) -> list[dict[str, str | int | float]]:
        """Mock paginated trade fetcher."""
        _ = market_ids
        assert limit == 1000
        assert offset == 0
        return [
            {"created_at": created1, "size": 1.0, "price": 0.4, "conditionId": "c1"},
            {"created_at": created2, "size": 2.0, "price": 0.5, "conditionId": "c2"},
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

    # Verify trade data
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
            timestamp=123456789,
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
    df = await clob.fetch_orderbook_to_dataframe("YES")
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
        return {"bids": [], "asks": [], "timestamp": 123456789}

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
    ) -> list[dict[str, int | float]]:
        """Mock GET returning empty list."""
        return []

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
            # First chunk: timestamps 0-100, with overlap at 100
            return pl.DataFrame({"t": [0, 50, 100], "p": [0.5, 0.6, 0.7], "token_id": [token_id] * 3})
        else:
            # Second chunk: timestamps 100-200, overlap at 100
            return pl.DataFrame({"t": [100, 150, 200], "p": [0.7, 0.8, 0.9], "token_id": [token_id] * 3})

    monkeypatch.setattr(clob, "_fetch_price_history_chunk", fake_chunk)

    # Request long enough to trigger chunking
    span = MAX_PRICE_HISTORY_SECONDS * 2
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

        # Always return full page to test limit enforcement
        return [
            {
                "created_at": f"2025-01-{(offset // 1000) + 1:02d}T00:00:00+00:00",
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
    boundary_ts = 1736689600  # This is the exact timestamp for the "exact" trade

    async def fake_paged(
        limit: int, offset: int, market_ids: list[str] | None = None
    ) -> list[dict[str, str | int | float]]:
        """Mock returning trades at and around boundary."""
        return [
            {"created_at": "2025-01-01T00:00:00+00:00", "size": 1.0, "price": 0.5, "conditionId": "before"},
            {
                "created_at": "2025-01-12T13:46:40+00:00",
                "size": 2.0,
                "price": 0.6,
                "conditionId": "exact",
            },  # exactly boundary_ts
            {"created_at": "2025-01-12T13:46:41+00:00", "size": 3.0, "price": 0.7, "conditionId": "after"},
        ]

    monkeypatch.setattr(clob, "fetch_trades_paged", fake_paged)

    df = await clob.fetch_trades(market_ids=None, since_ts=boundary_ts)

    # Filter at line 356: >= since_ts, so should include boundary
    assert df.height == 2, "Should include trades at exact boundary and after"
    condition_ids = set(df["conditionId"].to_list())
    assert "exact" in condition_ids, "Should include trade at exact boundary timestamp"
    assert "after" in condition_ids, "Should include trade after boundary"
    assert "before" not in condition_ids, "Should exclude trade before boundary"
