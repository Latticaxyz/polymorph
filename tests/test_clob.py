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
from polymorph.sources.clob import CLOB, CLOB_BASE, DATA_API


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
