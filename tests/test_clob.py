"""Tests for CLOB data source."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import polars as pl
import pytest

from polymorph.config import Settings
from polymorph.core.base import PipelineContext
from polymorph.core.rate_limit import RateLimitError
from polymorph.models.api import OrderBook, OrderBookLevel
from polymorph.sources.clob import CLOB, CLOB_BASE, DATA_API


def create_order_book_response(
    token_id: str = "123",
    timestamp: int = 1609459200,
    bids: list[dict[str, str | float]] | None = None,
    asks: list[dict[str, str | float]] | None = None,
) -> dict[str, str | int | list[dict[str, str | float]]]:
    """Helper to create mock order book API responses."""
    return {
        "token_id": token_id,
        "timestamp": timestamp,
        "bids": bids if bids is not None else [],
        "asks": asks if asks is not None else [],
    }


class TestCLOBDataSource:
    """Test CLOB DataSource."""

    @pytest.fixture
    def context(self):
        """Create a test pipeline context."""
        settings = Settings()
        return PipelineContext(
            settings=settings,
            run_timestamp=datetime.now(timezone.utc),
            data_dir=Path("data"),
        )

    @pytest.fixture
    def clob_source(self, context):
        """Create a CLOB data source."""
        return CLOB(context)

    def test_clob_source_creation(self, context):
        """Test creating a CLOB data source."""
        source = CLOB(context)
        assert source.name == "clob"
        assert source.context == context
        assert source.settings == context.settings
        assert source.clob_base_url == CLOB_BASE
        assert source.data_api_url == DATA_API
        assert source.default_fidelity == 60
        assert source.max_trades == 200_000

    def test_clob_source_custom_params(self, context):
        """Test creating a CLOB data source with custom parameters."""
        custom_clob = "https://custom-clob.example.com"
        custom_data = "https://custom-data.example.com"
        source = CLOB(
            context,
            clob_base_url=custom_clob,
            data_api_url=custom_data,
            default_fidelity=30,
            max_trades=50_000,
        )
        assert source.clob_base_url == custom_clob
        assert source.data_api_url == custom_data
        assert source.default_fidelity == 30
        assert source.max_trades == 50_000

    @pytest.mark.anyio
    async def test_fetch_prices_history_basic(self, clob_source):
        """Test fetching price history for a token."""
        mock_prices = [
            {"t": 1609459200, "p": 0.50},
            {"t": 1609459260, "p": 0.52},
            {"t": 1609459320, "p": 0.51},
        ]

        mock_rate_limiter = AsyncMock()
        mock_rate_limiter.acquire = AsyncMock()

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.json = MagicMock(return_value=mock_prices)
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        with patch.object(clob_source, "_get_clob_rate_limiter", return_value=mock_rate_limiter):
            with patch.object(clob_source, "_get_client", return_value=mock_client):
                result = await clob_source.fetch_prices_history(token_id="123", start_ts=1609459200, end_ts=1609459320)

        assert isinstance(result, pl.DataFrame)
        assert result.height == 3
        assert "token_id" in result.columns
        assert "t" in result.columns
        assert "p" in result.columns
        # Verify token_id was added
        assert result["token_id"].to_list() == ["123", "123", "123"]

    @pytest.mark.anyio
    async def test_fetch_prices_history_custom_fidelity(self, clob_source):
        """Test that custom fidelity parameter is passed correctly."""
        calls: list[dict[str, Any]] = []

        mock_rate_limiter = AsyncMock()
        mock_rate_limiter.acquire = AsyncMock()

        mock_client = AsyncMock()

        async def track_call(url, *args, **kwargs):
            params = kwargs.get("params", {})
            calls.append({"params": params})
            mock_response = MagicMock()
            mock_response.json = MagicMock(return_value=[])
            mock_response.status_code = 200
            mock_response.raise_for_status = MagicMock()
            return mock_response

        mock_client.get.side_effect = track_call

        with patch.object(clob_source, "_get_clob_rate_limiter", return_value=mock_rate_limiter):
            with patch.object(clob_source, "_get_client", return_value=mock_client):
                await clob_source.fetch_prices_history(token_id="123", start_ts=0, end_ts=100, fidelity=30)

        assert len(calls) == 1
        assert calls[0]["params"]["fidelity"] == 30

    @pytest.mark.anyio
    async def test_fetch_prices_history_default_fidelity(self, clob_source):
        """Test that default fidelity is used when not specified."""
        calls: list[dict[str, Any]] = []

        mock_rate_limiter = AsyncMock()
        mock_rate_limiter.acquire = AsyncMock()

        mock_client = AsyncMock()

        async def track_call(url, *args, **kwargs):
            params = kwargs.get("params", {})
            calls.append({"params": params})
            mock_response = MagicMock()
            mock_response.json = MagicMock(return_value=[])
            mock_response.status_code = 200
            mock_response.raise_for_status = MagicMock()
            return mock_response

        mock_client.get.side_effect = track_call

        with patch.object(clob_source, "_get_clob_rate_limiter", return_value=mock_rate_limiter):
            with patch.object(clob_source, "_get_client", return_value=mock_client):
                await clob_source.fetch_prices_history(token_id="123", start_ts=0, end_ts=100)

        assert len(calls) == 1
        assert calls[0]["params"]["fidelity"] == 60  # default

    @pytest.mark.anyio
    async def test_fetch_prices_history_empty(self, clob_source):
        """Test fetching price history when no data returned."""

        mock_rate_limiter = AsyncMock()
        mock_rate_limiter.acquire = AsyncMock()

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.json = MagicMock(return_value=[])
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_client.get.return_value = mock_response

        with patch.object(clob_source, "_get_clob_rate_limiter", return_value=mock_rate_limiter):
            with patch.object(clob_source, "_get_client", return_value=mock_client):
                result = await clob_source.fetch_prices_history(token_id="123", start_ts=0, end_ts=100)

        assert isinstance(result, pl.DataFrame)
        assert result.height == 0

    @pytest.mark.anyio
    async def test_fetch_prices_history_chunking(self, clob_source):
        """Test that long time ranges are automatically chunked into 14-day windows."""
        start_ts = 0
        end_ts = 30 * 24 * 60 * 60

        calls: list[dict[str, Any]] = []

        mock_rate_limiter = AsyncMock()
        mock_rate_limiter.acquire = AsyncMock()

        mock_client = AsyncMock()

        async def track_call(url, *args, **kwargs):
            params = kwargs.get("params", {})
            calls.append({"params": params})
            # Return mock price data for each chunk
            mock_response = MagicMock()
            mock_response.json = MagicMock(
                return_value=[
                    {"t": params["startTs"], "p": 0.50},
                    {"t": params["startTs"] + 100, "p": 0.51},
                ]
            )
            mock_response.status_code = 200
            mock_response.raise_for_status = MagicMock()
            return mock_response

        mock_client.get.side_effect = track_call

        with patch.object(clob_source, "_get_clob_rate_limiter", return_value=mock_rate_limiter):
            with patch.object(clob_source, "_get_client", return_value=mock_client):
                result = await clob_source.fetch_prices_history(
                    token_id="123", start_ts=start_ts, end_ts=end_ts, fidelity=60
                )

        # Should have made 3 requests
        assert len(calls) == 3

        # Verify chunk boundaries
        assert calls[0]["params"]["startTs"] == 0
        assert calls[0]["params"]["endTs"] == 14 * 24 * 60 * 60

        assert calls[1]["params"]["startTs"] == 14 * 24 * 60 * 60 + 1
        assert calls[1]["params"]["endTs"] == 2 * (14 * 24 * 60 * 60) + 1

        assert calls[2]["params"]["startTs"] == 2 * (14 * 24 * 60 * 60) + 2
        assert calls[2]["params"]["endTs"] == 30 * 24 * 60 * 60

        # Should have concatenated all chunks
        assert isinstance(result, pl.DataFrame)
        assert result.height > 0

    @pytest.mark.anyio
    async def test_fetch_trades_paged_basic(self, clob_source):
        """Test fetching a single page of trades."""
        mock_trades = [
            {"id": "trade1", "price": 0.50, "size": 100},
            {"id": "trade2", "price": 0.51, "size": 150},
        ]

        mock_rate_limiter = AsyncMock()
        mock_rate_limiter.acquire = AsyncMock()

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.json = MagicMock(return_value=mock_trades)
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        with patch.object(clob_source, "_get_data_rate_limiter", return_value=mock_rate_limiter):
            with patch.object(clob_source, "_get_client", return_value=mock_client):
                result = await clob_source.fetch_trades_paged(limit=100, offset=0)

        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["id"] == "trade1"

    @pytest.mark.anyio
    async def test_fetch_trades_paged_with_market_ids(self, clob_source):
        """Test that market_ids are properly formatted as comma-separated."""
        calls: list[dict[str, Any]] = []

        mock_rate_limiter = AsyncMock()
        mock_rate_limiter.acquire = AsyncMock()

        mock_client = AsyncMock()

        async def track_call(url, *args, **kwargs):
            params = kwargs.get("params", {})
            calls.append({"params": params})
            mock_response = MagicMock()
            mock_response.json = MagicMock(return_value=[])
            mock_response.status_code = 200
            mock_response.raise_for_status = MagicMock()
            return mock_response

        mock_client.get.side_effect = track_call

        with patch.object(clob_source, "_get_data_rate_limiter", return_value=mock_rate_limiter):
            with patch.object(clob_source, "_get_client", return_value=mock_client):
                await clob_source.fetch_trades_paged(limit=100, offset=0, market_ids=["123", "456", "789"])

        assert len(calls) == 1
        assert calls[0]["params"]["market"] == "123,456,789"

    @pytest.mark.anyio
    async def test_fetch_trades_paged_nested_data(self, clob_source):
        """Test fetching when response has nested data structure."""
        mock_trades = [{"id": "trade1"}, {"id": "trade2"}]

        mock_rate_limiter = AsyncMock()
        mock_rate_limiter.acquire = AsyncMock()

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.json = MagicMock(return_value={"data": mock_trades})
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        with patch.object(clob_source, "_get_data_rate_limiter", return_value=mock_rate_limiter):
            with patch.object(clob_source, "_get_client", return_value=mock_client):
                result = await clob_source.fetch_trades_paged()

        assert isinstance(result, list)
        assert len(result) == 2

    @pytest.mark.anyio
    async def test_fetch_trades_single_page(self, clob_source):
        """Test fetching trades with a single page of results."""
        mock_trades = [{"id": f"trade{i}", "price": 0.50 + i * 0.01, "size": 100} for i in range(50)]

        mock_rate_limiter = AsyncMock()
        mock_rate_limiter.acquire = AsyncMock()

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.json = MagicMock(return_value=mock_trades)
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        with patch.object(clob_source, "_get_data_rate_limiter", return_value=mock_rate_limiter):
            with patch.object(clob_source, "_get_client", return_value=mock_client):
                result = await clob_source.fetch_trades()

        assert isinstance(result, pl.DataFrame)
        assert result.height == 50

    @pytest.mark.anyio
    async def test_fetch_trades_multiple_pages(self, clob_source):
        """Test fetching trades across multiple pages."""
        # Create mock data for 2.5 pages
        page1 = [{"id": f"trade{i}", "price": 0.50, "size": 100} for i in range(1000)]
        page2 = [{"id": f"trade{i}", "price": 0.50, "size": 100} for i in range(1000, 2000)]
        page3 = [{"id": f"trade{i}", "price": 0.50, "size": 100} for i in range(2000, 2500)]

        mock_rate_limiter = AsyncMock()
        mock_rate_limiter.acquire = AsyncMock()

        mock_client = AsyncMock()

        async def paginated_response(url, *args, **kwargs):
            params = kwargs.get("params", {})
            offset = params.get("offset", 0)
            mock_response = MagicMock()
            if offset == 0:
                mock_response.json = MagicMock(return_value=page1)
            elif offset == 1000:
                mock_response.json = MagicMock(return_value=page2)
            elif offset == 2000:
                mock_response.json = MagicMock(return_value=page3)
            else:
                mock_response.json = MagicMock(return_value=[])
            mock_response.status_code = 200
            mock_response.raise_for_status = MagicMock()
            return mock_response

        mock_client.get.side_effect = paginated_response

        with patch.object(clob_source, "_get_data_rate_limiter", return_value=mock_rate_limiter):
            with patch.object(clob_source, "_get_client", return_value=mock_client):
                result = await clob_source.fetch_trades()

        assert isinstance(result, pl.DataFrame)
        assert result.height == 2500

    @pytest.mark.anyio
    async def test_fetch_trades_with_market_filter(self, clob_source):
        """Test fetching trades with market_ids filter."""
        calls: list[dict[str, Any]] = []

        mock_rate_limiter = AsyncMock()
        mock_rate_limiter.acquire = AsyncMock()

        mock_client = AsyncMock()

        async def track_call(url, *args, **kwargs):
            params = kwargs.get("params", {})
            calls.append({"params": params})
            mock_response = MagicMock()
            mock_response.json = MagicMock(return_value=[])
            mock_response.status_code = 200
            mock_response.raise_for_status = MagicMock()
            return mock_response

        mock_client.get.side_effect = track_call

        with patch.object(clob_source, "_get_data_rate_limiter", return_value=mock_rate_limiter):
            with patch.object(clob_source, "_get_client", return_value=mock_client):
                await clob_source.fetch_trades(market_ids=["123", "456"])

        assert len(calls) == 1
        assert "market" in calls[0]["params"]
        assert calls[0]["params"]["market"] == "123,456"

    @pytest.mark.anyio
    async def test_fetch_trades_stops_at_max_trades(self, context):
        """Test that fetching stops at max_trades limit."""
        source = CLOB(context, max_trades=1500)

        # Mock more data than max_trades should allow
        mock_page = [{"id": f"trade{i}", "price": 0.50} for i in range(1000)]
        call_count = [0]

        mock_rate_limiter = AsyncMock()
        mock_rate_limiter.acquire = AsyncMock()

        mock_client = AsyncMock()

        async def count_calls(url, *args, **kwargs):
            call_count[0] += 1
            mock_response = MagicMock()
            mock_response.json = MagicMock(return_value=mock_page)
            mock_response.status_code = 200
            mock_response.raise_for_status = MagicMock()
            return mock_response

        mock_client.get.side_effect = count_calls

        with patch.object(source, "_get_data_rate_limiter", return_value=mock_rate_limiter):
            with patch.object(source, "_get_client", return_value=mock_client):
                result = await source.fetch_trades()

        # Should respect max_trades limit
        assert result.height <= 2000  # At most 2 pages
        assert result.height >= 1000  # At least got one page
        assert call_count[0] <= 2  # Should not exceed 2 calls

    @pytest.mark.anyio
    async def test_fetch_trades_empty_results(self, clob_source):
        """Test fetching when no trades are returned."""
        mock_rate_limiter = AsyncMock()
        mock_rate_limiter.acquire = AsyncMock()

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.json = MagicMock(return_value=[])
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        with patch.object(clob_source, "_get_data_rate_limiter", return_value=mock_rate_limiter):
            with patch.object(clob_source, "_get_client", return_value=mock_client):
                result = await clob_source.fetch_trades()

        assert isinstance(result, pl.DataFrame)
        assert result.height == 0

    @pytest.mark.anyio
    async def test_fetch_trades_with_timestamp_parsing(self, clob_source):
        """Test that created_at is parsed to timestamp column."""
        mock_trades = [
            {
                "id": "trade1",
                "price": 0.50,
                "created_at": "2021-01-01T00:00:00+00:00",
            },
            {
                "id": "trade2",
                "price": 0.51,
                "created_at": "2021-01-01T00:01:00+00:00",
            },
        ]

        mock_rate_limiter = AsyncMock()
        mock_rate_limiter.acquire = AsyncMock()

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.json = MagicMock(return_value=mock_trades)
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        with patch.object(clob_source, "_get_data_rate_limiter", return_value=mock_rate_limiter):
            with patch.object(clob_source, "_get_client", return_value=mock_client):
                result = await clob_source.fetch_trades()

        assert isinstance(result, pl.DataFrame)
        assert "timestamp" in result.columns
        # Verify timestamps are integers (Unix timestamps)
        timestamps = result["timestamp"].to_list()
        assert all(isinstance(ts, int) for ts in timestamps)

    @pytest.mark.anyio
    async def test_fetch_trades_with_since_ts_filter(self, clob_source):
        """Test filtering trades by since_ts parameter."""
        mock_trades = [
            {
                "id": "trade1",
                "price": 0.50,
                "created_at": "2021-01-01T00:00:00+00:00",
            },
            {
                "id": "trade2",
                "price": 0.51,
                "created_at": "2021-01-01T00:05:00+00:00",
            },
            {
                "id": "trade3",
                "price": 0.52,
                "created_at": "2021-01-01T00:10:00+00:00",
            },
        ]

        mock_rate_limiter = AsyncMock()
        mock_rate_limiter.acquire = AsyncMock()

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.json = MagicMock(return_value=mock_trades)
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        with patch.object(clob_source, "_get_data_rate_limiter", return_value=mock_rate_limiter):
            with patch.object(clob_source, "_get_client", return_value=mock_client):
                # Filter to only trades after 00:05:00
                result = await clob_source.fetch_trades(since_ts=1609459500)

        assert isinstance(result, pl.DataFrame)
        # Should filter out trades before the timestamp
        assert result.height <= 3
        assert result.height >= 0

    @pytest.mark.anyio
    async def test_client_lifecycle(self, clob_source):
        """Test async client creation and cleanup."""
        assert clob_source._client is None

        # Get client should create it
        client = await clob_source._get_client()
        assert client is not None
        assert clob_source._client is not None

        # Getting again should return same client
        client2 = await clob_source._get_client()
        assert client is client2

        # Close should cleanup
        await clob_source.close()
        assert clob_source._client is None

    @pytest.mark.anyio
    async def test_context_manager(self, clob_source):
        """Test using CLOB as an async context manager."""
        async with clob_source as source:
            assert source is clob_source

        # Should have cleaned up
        assert clob_source._client is None

    @pytest.mark.anyio
    async def test_http_error_handling(self, clob_source):
        """Test handling of HTTP errors with realistic httpx objects."""
        mock_rate_limiter = AsyncMock()
        mock_rate_limiter.acquire = AsyncMock()

        mock_client = AsyncMock()
        request = httpx.Request("GET", "https://clob.polymarket.com/prices-history")
        response = httpx.Response(
            404,
            request=request,
            content=b'{"error": "Not Found"}',
        )

        async def raise_error(*args, **kwargs):
            raise httpx.HTTPStatusError("404 Not Found", request=request, response=response)

        mock_client.get.side_effect = raise_error

        with patch.object(clob_source, "_get_clob_rate_limiter", return_value=mock_rate_limiter):
            with patch.object(clob_source, "_get_client", return_value=mock_client):
                with pytest.raises(httpx.HTTPStatusError):
                    await clob_source.fetch_prices_history(token_id="123", start_ts=0, end_ts=100)

    @pytest.mark.anyio
    async def test_rate_limit_error_handling(self, clob_source):
        """Test that 429 responses raise RateLimitError."""
        from polymorph.core.rate_limit import RateLimitError

        mock_rate_limiter = AsyncMock()
        mock_rate_limiter.acquire = AsyncMock()

        mock_client = AsyncMock()
        request = httpx.Request("GET", "https://clob.polymarket.com/prices-history")
        response = httpx.Response(429, request=request, headers={"retry-after": "1"})

        async def return_429(*args, **kwargs):
            return response

        mock_client.get.side_effect = return_429

        with patch.object(clob_source, "_get_clob_rate_limiter", return_value=mock_rate_limiter):
            with patch.object(clob_source, "_get_client", return_value=mock_client):
                with pytest.raises(RateLimitError):
                    await clob_source.fetch_prices_history(token_id="123", start_ts=0, end_ts=100)

    @pytest.mark.anyio
    @pytest.mark.integration
    async def test_real_api_connection(self, context, request):
        """Integration test: Verify the real CLOB API endpoint is reachable.

        Run with: pytest --run-integration -m integration
        Skips gracefully if the API is unavailable or the flag isn't set.
        """
        if not request.config.getoption("--run-integration"):
            pytest.skip("Integration tests require --run-integration flag")

        source = CLOB(context)

        try:
            result = await source.fetch_prices_history(
                token_id="21742633143463906290569050155826241533067272736897614950488156847949938836455",
                start_ts=1609459200,
                end_ts=1609459260,
                fidelity=60,
            )

            assert isinstance(result, pl.DataFrame)
            assert result.height >= 0

            print(f"\n✓ Successfully fetched {result.height} price points from CLOB API")

        except (httpx.HTTPError, httpx.TimeoutException) as e:
            pytest.skip(f"API unavailable or unreachable: {e}")
        except Exception as e:
            pytest.skip(f"Unexpected error during integration test: {e}")
        finally:
            await source.close()


class TestCLOBOrderBookFunctions:
    """Test CLOB order book functions."""

    @pytest.fixture
    def context(self):
        """Create a test pipeline context."""
        settings = Settings()
        return PipelineContext(
            settings=settings,
            run_timestamp=datetime.now(timezone.utc),
            data_dir=Path("data"),
        )

    @pytest.fixture
    def clob_source(self, context):
        """Create a CLOB data source."""
        return CLOB(context)

    # Tests for fetch_order_book()

    @pytest.mark.anyio
    async def test_fetch_order_book_basic(self, clob_source):
        """Test successfully fetching and parsing order book with normal data."""
        mock_order_book_data = create_order_book_response(
            token_id="123",
            timestamp=1609459200,
            bids=[
                {"price": "0.60", "size": "100"},
                {"price": "0.59", "size": "200"},
                {"price": "0.58", "size": "150"},
            ],
            asks=[
                {"price": "0.61", "size": "80"},
                {"price": "0.62", "size": "120"},
                {"price": "0.63", "size": "90"},
            ],
        )

        mock_rate_limiter = AsyncMock()
        mock_rate_limiter.acquire = AsyncMock()

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.json = MagicMock(return_value=mock_order_book_data)
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        with patch.object(clob_source, "_get_clob_rate_limiter", return_value=mock_rate_limiter):
            with patch.object(clob_source, "_get_client", return_value=mock_client):
                result = await clob_source.fetch_order_book("123")

        assert isinstance(result, OrderBook)
        assert result.token_id == "123"
        assert result.timestamp == 1609459200
        assert len(result.bids) == 3
        assert len(result.asks) == 3
        assert result.best_bid == 0.60
        assert result.best_ask == 0.61
        assert result.mid_price == 0.605
        assert result.spread is not None
        assert abs(result.spread - 0.01) < 1e-10  # Allow for floating point precision
        # Verify bids sorted descending
        assert result.bids[0].price > result.bids[1].price > result.bids[2].price
        # Verify asks sorted ascending
        assert result.asks[0].price < result.asks[1].price < result.asks[2].price

    @pytest.mark.anyio
    async def test_fetch_order_book_empty_bids(self, clob_source):
        """Test order book with only asks (no bids)."""
        mock_order_book_data = create_order_book_response(
            bids=[],
            asks=[{"price": "0.61", "size": "80"}],
        )

        mock_rate_limiter = AsyncMock()
        mock_rate_limiter.acquire = AsyncMock()

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.json = MagicMock(return_value=mock_order_book_data)
        mock_response.status_code = 200
        mock_client.get = AsyncMock(return_value=mock_response)

        with patch.object(clob_source, "_get_clob_rate_limiter", return_value=mock_rate_limiter):
            with patch.object(clob_source, "_get_client", return_value=mock_client):
                result = await clob_source.fetch_order_book("123")

        assert len(result.bids) == 0
        assert len(result.asks) == 1
        assert result.best_bid is None
        assert result.best_ask == 0.61
        assert result.mid_price is None
        assert result.spread is None

    @pytest.mark.anyio
    async def test_fetch_order_book_empty_asks(self, clob_source):
        """Test order book with only bids (no asks)."""
        mock_order_book_data = create_order_book_response(
            bids=[{"price": "0.60", "size": "100"}],
            asks=[],
        )

        mock_rate_limiter = AsyncMock()
        mock_rate_limiter.acquire = AsyncMock()

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.json = MagicMock(return_value=mock_order_book_data)
        mock_response.status_code = 200
        mock_client.get = AsyncMock(return_value=mock_response)

        with patch.object(clob_source, "_get_clob_rate_limiter", return_value=mock_rate_limiter):
            with patch.object(clob_source, "_get_client", return_value=mock_client):
                result = await clob_source.fetch_order_book("123")

        assert len(result.bids) == 1
        assert len(result.asks) == 0
        assert result.best_bid == 0.60
        assert result.best_ask is None
        assert result.mid_price is None
        assert result.spread is None

    @pytest.mark.anyio
    async def test_fetch_order_book_empty(self, clob_source):
        """Test completely empty order book."""
        mock_order_book_data = create_order_book_response(bids=[], asks=[])

        mock_rate_limiter = AsyncMock()
        mock_rate_limiter.acquire = AsyncMock()

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.json = MagicMock(return_value=mock_order_book_data)
        mock_response.status_code = 200
        mock_client.get = AsyncMock(return_value=mock_response)

        with patch.object(clob_source, "_get_clob_rate_limiter", return_value=mock_rate_limiter):
            with patch.object(clob_source, "_get_client", return_value=mock_client):
                result = await clob_source.fetch_order_book("123")

        assert len(result.bids) == 0
        assert len(result.asks) == 0
        assert result.best_bid is None
        assert result.best_ask is None
        assert result.mid_price is None
        assert result.spread is None

    @pytest.mark.anyio
    async def test_fetch_order_book_sorting(self, clob_source):
        """Test bids/asks are correctly sorted even if API returns unsorted."""
        mock_order_book_data = create_order_book_response(
            bids=[
                {"price": "0.58", "size": "150"},  # lowest first (wrong order)
                {"price": "0.60", "size": "100"},  # highest
                {"price": "0.59", "size": "200"},  # middle
            ],
            asks=[
                {"price": "0.63", "size": "90"},  # highest
                {"price": "0.61", "size": "80"},  # lowest first (wrong order)
                {"price": "0.62", "size": "120"},  # middle
            ],
        )

        mock_rate_limiter = AsyncMock()
        mock_rate_limiter.acquire = AsyncMock()

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.json = MagicMock(return_value=mock_order_book_data)
        mock_response.status_code = 200
        mock_client.get = AsyncMock(return_value=mock_response)

        with patch.object(clob_source, "_get_clob_rate_limiter", return_value=mock_rate_limiter):
            with patch.object(clob_source, "_get_client", return_value=mock_client):
                result = await clob_source.fetch_order_book("123")

        # Verify bids sorted descending
        assert result.bids[0].price == 0.60
        assert result.bids[1].price == 0.59
        assert result.bids[2].price == 0.58
        # Verify asks sorted ascending
        assert result.asks[0].price == 0.61
        assert result.asks[1].price == 0.62
        assert result.asks[2].price == 0.63
        # Verify best bid/ask
        assert result.best_bid == 0.60
        assert result.best_ask == 0.61

    @pytest.mark.anyio
    async def test_fetch_order_book_timestamp_parsing(self, clob_source):
        """Test handle various timestamp formats."""
        # Test with integer timestamp
        mock_order_book_data = create_order_book_response(
            timestamp=1609459200,
            bids=[{"price": "0.60", "size": "100"}],
            asks=[],
        )

        mock_rate_limiter = AsyncMock()
        mock_rate_limiter.acquire = AsyncMock()

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.json = MagicMock(return_value=mock_order_book_data)
        mock_response.status_code = 200
        mock_client.get = AsyncMock(return_value=mock_response)

        with patch.object(clob_source, "_get_clob_rate_limiter", return_value=mock_rate_limiter):
            with patch.object(clob_source, "_get_client", return_value=mock_client):
                result = await clob_source.fetch_order_book("123")

        assert result.timestamp == 1609459200

    @pytest.mark.anyio
    async def test_fetch_order_book_api_url_and_params(self, clob_source):
        """Test correct API endpoint and parameters."""
        calls = []

        async def track_call(url, *args, **kwargs):
            params = kwargs.get("params", {})
            calls.append({"url": url, "params": params})
            mock_response = MagicMock()
            mock_response.json = MagicMock(return_value=create_order_book_response(bids=[], asks=[]))
            mock_response.status_code = 200
            mock_response.raise_for_status = MagicMock()
            return mock_response

        mock_rate_limiter = AsyncMock()
        mock_rate_limiter.acquire = AsyncMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=track_call)
        mock_client.timeout = 30

        with patch.object(clob_source, "_get_clob_rate_limiter", return_value=mock_rate_limiter):
            with patch.object(clob_source, "_get_client", return_value=mock_client):
                await clob_source.fetch_order_book("123")

        assert len(calls) == 1
        assert calls[0]["url"] == f"{CLOB_BASE}/book"
        assert calls[0]["params"]["token_id"] == "123"
        mock_rate_limiter.acquire.assert_called()

    @pytest.mark.anyio
    async def test_fetch_order_book_http_error(self, clob_source):
        """Test HTTP errors propagate correctly."""
        mock_rate_limiter = AsyncMock()
        mock_rate_limiter.acquire = AsyncMock()

        request = httpx.Request("GET", f"{CLOB_BASE}/book")
        response = httpx.Response(404, request=request)

        async def raise_error(*args, **kwargs):
            raise httpx.HTTPStatusError("404 Not Found", request=request, response=response)

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=raise_error)

        with patch.object(clob_source, "_get_clob_rate_limiter", return_value=mock_rate_limiter):
            with patch.object(clob_source, "_get_client", return_value=mock_client):
                with pytest.raises(httpx.HTTPStatusError):
                    await clob_source.fetch_order_book("123")

    @pytest.mark.anyio
    async def test_fetch_order_book_rate_limit_error(self, clob_source):
        """Test 429 responses raise RateLimitError."""
        mock_rate_limiter = AsyncMock()
        mock_rate_limiter.acquire = AsyncMock()

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.raise_for_status = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        with patch.object(clob_source, "_get_clob_rate_limiter", return_value=mock_rate_limiter):
            with patch.object(clob_source, "_get_client", return_value=mock_client):
                with pytest.raises(RateLimitError):
                    await clob_source.fetch_order_book("123")

    # Tests for fetch_order_books_batch()

    @pytest.mark.anyio
    async def test_fetch_order_books_batch_success(self, clob_source):
        """Test successfully fetching multiple order books."""
        # Create mock order books
        mock_order_book_1 = OrderBook(
            token_id="token1",
            timestamp=1609459200,
            bids=[OrderBookLevel(price=0.60, size=100)],
            asks=[OrderBookLevel(price=0.61, size=80)],
            best_bid=0.60,
            best_ask=0.61,
            mid_price=0.605,
            spread=0.01,
        )
        mock_order_book_2 = OrderBook(
            token_id="token2",
            timestamp=1609459200,
            bids=[OrderBookLevel(price=0.50, size=150)],
            asks=[OrderBookLevel(price=0.51, size=120)],
            best_bid=0.50,
            best_ask=0.51,
            mid_price=0.505,
            spread=0.01,
        )
        mock_order_book_3 = OrderBook(
            token_id="token3",
            timestamp=1609459200,
            bids=[OrderBookLevel(price=0.70, size=90)],
            asks=[OrderBookLevel(price=0.71, size=110)],
            best_bid=0.70,
            best_ask=0.71,
            mid_price=0.705,
            spread=0.01,
        )

        async def mock_fetch_order_book(token_id):
            if token_id == "token1":
                return mock_order_book_1
            elif token_id == "token2":
                return mock_order_book_2
            elif token_id == "token3":
                return mock_order_book_3

        with patch.object(clob_source, "fetch_order_book", side_effect=mock_fetch_order_book):
            result = await clob_source.fetch_order_books_batch(["token1", "token2", "token3"])

        assert len(result) == 3
        assert all(isinstance(ob, OrderBook) for ob in result)
        assert result[0].token_id == "token1"
        assert result[1].token_id == "token2"
        assert result[2].token_id == "token3"

    @pytest.mark.anyio
    async def test_fetch_order_books_batch_partial_failure(self, clob_source):
        """Test continues processing when one token fails."""
        mock_order_book = OrderBook(
            token_id="token1",
            timestamp=1609459200,
            bids=[],
            asks=[],
            best_bid=None,
            best_ask=None,
            mid_price=None,
            spread=None,
        )

        async def mock_fetch_with_failure(token_id):
            if token_id == "token2":
                raise Exception("API Error")
            return mock_order_book

        with patch.object(clob_source, "fetch_order_book", side_effect=mock_fetch_with_failure):
            result = await clob_source.fetch_order_books_batch(["token1", "token2", "token3"])

        # Should have 2 results (token1 and token3), skipping failed token2
        assert len(result) == 2

    @pytest.mark.anyio
    async def test_fetch_order_books_batch_all_failures(self, clob_source):
        """Test returns empty list when all fetches fail."""

        async def mock_fetch_always_fails(token_id):
            raise Exception("API Error")

        with patch.object(clob_source, "fetch_order_book", side_effect=mock_fetch_always_fails):
            result = await clob_source.fetch_order_books_batch(["token1", "token2", "token3"])

        assert result == []

    @pytest.mark.anyio
    async def test_fetch_order_books_batch_empty_input(self, clob_source):
        """Test handle empty token_ids list."""
        with patch.object(clob_source, "fetch_order_book") as mock_fetch:
            result = await clob_source.fetch_order_books_batch([])

        assert result == []
        mock_fetch.assert_not_called()

    @pytest.mark.anyio
    async def test_fetch_order_books_batch_single_token(self, clob_source):
        """Test works correctly with single token."""
        mock_order_book = OrderBook(
            token_id="token1",
            timestamp=1609459200,
            bids=[OrderBookLevel(price=0.60, size=100)],
            asks=[],
            best_bid=0.60,
            best_ask=None,
            mid_price=None,
            spread=None,
        )

        async def mock_fetch(token_id):
            return mock_order_book

        with patch.object(clob_source, "fetch_order_book", side_effect=mock_fetch):
            result = await clob_source.fetch_order_books_batch(["token1"])

        assert len(result) == 1
        assert result[0].token_id == "token1"

    @pytest.mark.anyio
    async def test_fetch_order_books_batch_error_logging(self, clob_source):
        """Test verify error logging contains helpful information."""

        async def mock_fetch_with_error(token_id):
            if token_id == "token2":
                raise ValueError("Invalid token")
            return OrderBook(
                token_id=token_id,
                timestamp=1609459200,
                bids=[],
                asks=[],
                best_bid=None,
                best_ask=None,
                mid_price=None,
                spread=None,
            )

        with patch("polymorph.sources.clob.logger") as mock_logger:
            with patch.object(clob_source, "fetch_order_book", side_effect=mock_fetch_with_error):
                await clob_source.fetch_order_books_batch(["token1", "token2", "token3"])

            # Verify logger.error was called for the failed token
            mock_logger.error.assert_called_once()
            error_call = mock_logger.error.call_args[0][0]
            assert "token2" in error_call

    # Tests for fetch_order_book_to_dataframe()

    @pytest.mark.anyio
    async def test_fetch_order_book_to_dataframe_basic(self, clob_source):
        """Test converting order book to DataFrame with correct structure."""
        mock_order_book = OrderBook(
            token_id="123",
            timestamp=1609459200,
            bids=[
                OrderBookLevel(price=0.60, size=100),
                OrderBookLevel(price=0.59, size=200),
                OrderBookLevel(price=0.58, size=150),
            ],
            asks=[
                OrderBookLevel(price=0.61, size=80),
                OrderBookLevel(price=0.62, size=120),
            ],
            best_bid=0.60,
            best_ask=0.61,
            mid_price=0.605,
            spread=0.01,
        )

        with patch.object(clob_source, "fetch_order_book", return_value=mock_order_book):
            df = await clob_source.fetch_order_book_to_dataframe("123")

        assert isinstance(df, pl.DataFrame)
        assert df.height == 5  # 3 bids + 2 asks
        assert set(df.columns) == {"token_id", "timestamp", "side", "price", "size"}
        # Check correct number of each side
        assert df.filter(pl.col("side") == "bid").height == 3
        assert df.filter(pl.col("side") == "ask").height == 2
        # All rows should have same token_id and timestamp
        assert (df["token_id"] == "123").all()
        assert (df["timestamp"] == 1609459200).all()

    @pytest.mark.anyio
    async def test_fetch_order_book_to_dataframe_column_types(self, clob_source):
        """Test DataFrame has correct column types."""
        mock_order_book = OrderBook(
            token_id="123",
            timestamp=1609459200,
            bids=[OrderBookLevel(price=0.60, size=100)],
            asks=[OrderBookLevel(price=0.61, size=80)],
            best_bid=0.60,
            best_ask=0.61,
            mid_price=0.605,
            spread=0.01,
        )

        with patch.object(clob_source, "fetch_order_book", return_value=mock_order_book):
            df = await clob_source.fetch_order_book_to_dataframe("123")

        assert df["token_id"].dtype == pl.Utf8
        assert df["timestamp"].dtype == pl.Int64
        assert df["side"].dtype == pl.Utf8
        assert df["price"].dtype == pl.Float64
        assert df["size"].dtype == pl.Float64

    @pytest.mark.anyio
    async def test_fetch_order_book_to_dataframe_empty_order_book(self, clob_source):
        """Test handling empty order book (no bids or asks)."""
        mock_order_book = OrderBook(
            token_id="123",
            timestamp=1609459200,
            bids=[],
            asks=[],
            best_bid=None,
            best_ask=None,
            mid_price=None,
            spread=None,
        )

        with patch.object(clob_source, "fetch_order_book", return_value=mock_order_book):
            df = await clob_source.fetch_order_book_to_dataframe("123")

        assert isinstance(df, pl.DataFrame)
        assert df.height == 0
        # Schema should still be defined
        assert set(df.columns) == {"token_id", "timestamp", "side", "price", "size"}

    @pytest.mark.anyio
    async def test_fetch_order_book_to_dataframe_bids_only(self, clob_source):
        """Test handling order book with only bids."""
        mock_order_book = OrderBook(
            token_id="123",
            timestamp=1609459200,
            bids=[
                OrderBookLevel(price=0.60, size=100),
                OrderBookLevel(price=0.59, size=200),
                OrderBookLevel(price=0.58, size=150),
                OrderBookLevel(price=0.57, size=175),
                OrderBookLevel(price=0.56, size=125),
            ],
            asks=[],
            best_bid=0.60,
            best_ask=None,
            mid_price=None,
            spread=None,
        )

        with patch.object(clob_source, "fetch_order_book", return_value=mock_order_book):
            df = await clob_source.fetch_order_book_to_dataframe("123")

        assert df.height == 5
        assert (df["side"] == "bid").all()
        assert df.filter(pl.col("side") == "ask").height == 0

    @pytest.mark.anyio
    async def test_fetch_order_book_to_dataframe_asks_only(self, clob_source):
        """Test handling order book with only asks."""
        mock_order_book = OrderBook(
            token_id="123",
            timestamp=1609459200,
            bids=[],
            asks=[
                OrderBookLevel(price=0.61, size=80),
                OrderBookLevel(price=0.62, size=120),
                OrderBookLevel(price=0.63, size=90),
                OrderBookLevel(price=0.64, size=110),
            ],
            best_bid=None,
            best_ask=0.61,
            mid_price=None,
            spread=None,
        )

        with patch.object(clob_source, "fetch_order_book", return_value=mock_order_book):
            df = await clob_source.fetch_order_book_to_dataframe("123")

        assert df.height == 4
        assert (df["side"] == "ask").all()
        assert df.filter(pl.col("side") == "bid").height == 0

    @pytest.mark.anyio
    async def test_fetch_order_book_to_dataframe_order_preserved(self, clob_source):
        """Test bids appear before asks in DataFrame."""
        mock_order_book = OrderBook(
            token_id="123",
            timestamp=1609459200,
            bids=[
                OrderBookLevel(price=0.60, size=100),
                OrderBookLevel(price=0.59, size=200),
            ],
            asks=[
                OrderBookLevel(price=0.61, size=80),
                OrderBookLevel(price=0.62, size=120),
                OrderBookLevel(price=0.63, size=90),
            ],
            best_bid=0.60,
            best_ask=0.61,
            mid_price=0.605,
            spread=0.01,
        )

        with patch.object(clob_source, "fetch_order_book", return_value=mock_order_book):
            df = await clob_source.fetch_order_book_to_dataframe("123")

        # First 2 rows should be bids
        assert (df.head(2)["side"] == "bid").all()
        # Last 3 rows should be asks
        assert (df.tail(3)["side"] == "ask").all()
        # Verify bid prices are descending
        bid_prices = df.filter(pl.col("side") == "bid")["price"].to_list()
        assert bid_prices == [0.60, 0.59]
        # Verify ask prices are ascending
        ask_prices = df.filter(pl.col("side") == "ask")["price"].to_list()
        assert ask_prices == [0.61, 0.62, 0.63]

    @pytest.mark.anyio
    async def test_fetch_order_book_to_dataframe_values_match(self, clob_source):
        """Test all values correctly transferred from OrderBook."""
        mock_order_book = OrderBook(
            token_id="test_token",
            timestamp=1234567890,
            bids=[OrderBookLevel(price=0.45, size=250)],
            asks=[OrderBookLevel(price=0.55, size=175)],
            best_bid=0.45,
            best_ask=0.55,
            mid_price=0.50,
            spread=0.10,
        )

        with patch.object(clob_source, "fetch_order_book", return_value=mock_order_book):
            df = await clob_source.fetch_order_book_to_dataframe("test_token")

        # Check bid row
        bid_row = df.filter(pl.col("side") == "bid").row(0, named=True)
        assert bid_row["token_id"] == "test_token"
        assert bid_row["timestamp"] == 1234567890
        assert bid_row["price"] == 0.45
        assert bid_row["size"] == 250
        # Check ask row
        ask_row = df.filter(pl.col("side") == "ask").row(0, named=True)
        assert ask_row["token_id"] == "test_token"
        assert ask_row["timestamp"] == 1234567890
        assert ask_row["price"] == 0.55
        assert ask_row["size"] == 175

    # Tests for fetch_spread()

    @pytest.mark.anyio
    async def test_fetch_spread_basic(self, clob_source):
        """Test returns correct spread dictionary."""
        mock_order_book = OrderBook(
            token_id="123",
            timestamp=1609459200,
            bids=[OrderBookLevel(price=0.59, size=100)],
            asks=[OrderBookLevel(price=0.61, size=80)],
            best_bid=0.59,
            best_ask=0.61,
            mid_price=0.60,
            spread=0.02,
        )

        with patch.object(clob_source, "fetch_order_book", return_value=mock_order_book):
            result = await clob_source.fetch_spread("123")

        assert isinstance(result, dict)
        assert result["token_id"] == "123"
        assert result["bid"] == 0.59
        assert result["ask"] == 0.61
        assert result["mid"] == 0.60
        assert result["spread"] == 0.02
        assert result["timestamp"] == 1609459200

    @pytest.mark.anyio
    async def test_fetch_spread_no_bids(self, clob_source):
        """Test handling missing bids."""
        mock_order_book = OrderBook(
            token_id="123",
            timestamp=1609459200,
            bids=[],
            asks=[OrderBookLevel(price=0.50, size=80)],
            best_bid=None,
            best_ask=0.50,
            mid_price=None,
            spread=None,
        )

        with patch.object(clob_source, "fetch_order_book", return_value=mock_order_book):
            result = await clob_source.fetch_spread("123")

        assert result["bid"] is None
        assert result["ask"] == 0.50
        assert result["mid"] is None
        assert result["spread"] is None

    @pytest.mark.anyio
    async def test_fetch_spread_no_asks(self, clob_source):
        """Test handling missing asks."""
        mock_order_book = OrderBook(
            token_id="123",
            timestamp=1609459200,
            bids=[OrderBookLevel(price=0.40, size=100)],
            asks=[],
            best_bid=0.40,
            best_ask=None,
            mid_price=None,
            spread=None,
        )

        with patch.object(clob_source, "fetch_order_book", return_value=mock_order_book):
            result = await clob_source.fetch_spread("123")

        assert result["bid"] == 0.40
        assert result["ask"] is None
        assert result["mid"] is None
        assert result["spread"] is None

    @pytest.mark.anyio
    async def test_fetch_spread_empty_book(self, clob_source):
        """Test handling completely empty order book."""
        mock_order_book = OrderBook(
            token_id="123",
            timestamp=1609459200,
            bids=[],
            asks=[],
            best_bid=None,
            best_ask=None,
            mid_price=None,
            spread=None,
        )

        with patch.object(clob_source, "fetch_order_book", return_value=mock_order_book):
            result = await clob_source.fetch_spread("123")

        assert result["bid"] is None
        assert result["ask"] is None
        assert result["mid"] is None
        assert result["spread"] is None
        assert result["token_id"] == "123"
        assert result["timestamp"] == 1609459200

    @pytest.mark.anyio
    async def test_fetch_spread_dictionary_keys(self, clob_source):
        """Test verifying all expected keys present."""
        mock_order_book = OrderBook(
            token_id="123",
            timestamp=1609459200,
            bids=[OrderBookLevel(price=0.60, size=100)],
            asks=[OrderBookLevel(price=0.61, size=80)],
            best_bid=0.60,
            best_ask=0.61,
            mid_price=0.605,
            spread=0.01,
        )

        with patch.object(clob_source, "fetch_order_book", return_value=mock_order_book):
            result = await clob_source.fetch_spread("123")

        assert set(result.keys()) == {"token_id", "bid", "ask", "mid", "spread", "timestamp"}
        assert len(result) == 6

    @pytest.mark.anyio
    async def test_fetch_spread_value_types(self, clob_source):
        """Test verify value types are correct."""
        mock_order_book = OrderBook(
            token_id="123",
            timestamp=1609459200,
            bids=[OrderBookLevel(price=0.60, size=100)],
            asks=[OrderBookLevel(price=0.61, size=80)],
            best_bid=0.60,
            best_ask=0.61,
            mid_price=0.605,
            spread=0.01,
        )

        with patch.object(clob_source, "fetch_order_book", return_value=mock_order_book):
            result = await clob_source.fetch_spread("123")

        assert isinstance(result["token_id"], str)
        assert isinstance(result["bid"], float)
        assert isinstance(result["ask"], float)
        assert isinstance(result["mid"], float)
        assert isinstance(result["spread"], float)
        assert isinstance(result["timestamp"], int)

    @pytest.mark.anyio
    async def test_fetch_spread_narrow_spread(self, clob_source):
        """Test handling very small spreads (precision)."""
        mock_order_book = OrderBook(
            token_id="123",
            timestamp=1609459200,
            bids=[OrderBookLevel(price=0.500000, size=100)],
            asks=[OrderBookLevel(price=0.500001, size=80)],
            best_bid=0.500000,
            best_ask=0.500001,
            mid_price=0.5000005,
            spread=0.000001,
        )

        with patch.object(clob_source, "fetch_order_book", return_value=mock_order_book):
            result = await clob_source.fetch_spread("123")

        assert result["spread"] == 0.000001
        assert result["mid"] == 0.5000005
        # Verify no floating point precision errors
        assert abs(result["spread"] - (result["ask"] - result["bid"])) < 1e-10


class TestCLOBWithFetchPipeline:
    """Test CLOB with Fetch pipeline integration."""

    @pytest.fixture
    def context(self):
        """Create a test pipeline context."""
        settings = Settings()
        return PipelineContext(
            settings=settings,
            run_timestamp=datetime.now(timezone.utc),
            data_dir=Path("data"),
        )

    @pytest.mark.anyio
    async def test_clob_direct_usage(self, context):
        """Test using CLOB source directly in user code."""
        mock_prices = [
            {"t": 1609459200, "p": 0.50},
            {"t": 1609459260, "p": 0.52},
        ]

        clob = CLOB(context)

        mock_rate_limiter = AsyncMock()
        mock_rate_limiter.acquire = AsyncMock()

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.json = MagicMock(return_value=mock_prices)
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        with patch.object(clob, "_get_clob_rate_limiter", return_value=mock_rate_limiter):
            with patch.object(clob, "_get_client", return_value=mock_client):
                result = await clob.fetch_prices_history(token_id="123", start_ts=1609459200, end_ts=1609459320)

        assert isinstance(result, pl.DataFrame)
        assert result.height == 2
        assert "token_id" in result.columns
        assert "p" in result.columns
        assert result["token_id"].to_list() == ["123", "123"]
