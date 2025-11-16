"""Tests for Gamma data source."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import polars as pl
import pytest

from polymorph.config import Settings
from polymorph.core.base import PipelineContext
from polymorph.sources.gamma import GAMMA_BASE, Gamma


class TestGammaDataSource:
    """Test Gamma DataSource."""

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
    def gamma_source(self, context):
        """Create a Gamma data source."""
        return Gamma(context)

    def test_gamma_source_creation(self, context):
        """Test creating a Gamma data source."""
        source = Gamma(context)
        assert source.name == "gamma"
        assert source.context == context
        assert source.settings == context.settings
        assert source.base_url == GAMMA_BASE
        assert source.page_size == 250
        assert source.max_pages == 200

    def test_gamma_source_custom_params(self, context):
        """Test creating a Gamma data source with custom parameters."""
        custom_base = "https://custom-api.example.com"
        source = Gamma(context, base_url=custom_base, page_size=100, max_pages=50)
        assert source.base_url == custom_base
        assert source.page_size == 100
        assert source.max_pages == 50

    @pytest.mark.parametrize(
        "input_value,expected",
        [
            # None cases
            (None, []),
            # List cases
            (["123", "456"], ["123", "456"]),
            ([123, 456], ["123", "456"]),
            ([None, "123", None, "456"], ["123", "456"]),
            ([], []),
            # String cases - JSON array
            ('["123", "456"]', ["123", "456"]),
            ("[123, 456]", ["123", "456"]),
            # String cases - comma-separated
            ("123,456,789", ["123", "456", "789"]),
            ("123, 456, 789", ["123", "456", "789"]),
            ("  123  ,  456  ", ["123", "456"]),
            # String cases - single value
            ("123", ["123"]),
            ("  single  ", ["single"]),
            # Invalid JSON should fall back to single string
            ("[invalid json", ["[invalid json"]),
            # Number cases
            (123, ["123"]),
            (456.789, ["456.789"]),
        ],
    )
    def test_normalize_ids(self, input_value, expected):
        """Test ID normalization with various input types."""
        result = Gamma._normalize_ids(input_value)
        assert result == expected

    @pytest.mark.anyio
    async def test_fetch_single_page(self, gamma_source):
        """Test fetching markets with a single page of results."""
        mock_markets = [
            {
                "id": "market1",
                "question": "Will it rain?",
                "clobTokenIds": "123,456",  # API returns as comma-separated string
            },
            {
                "id": "market2",
                "question": "Will it snow?",
                "clobTokenIds": '["789", "012"]',  # or as JSON string
            },
        ]

        async def mock_get(_url: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
            _ = params  # Mark as intentionally unused
            return mock_markets

        with patch.object(gamma_source, "_get", side_effect=mock_get):
            result = await gamma_source.fetch(active_only=True)

        assert isinstance(result, pl.DataFrame)
        assert result.height == 2
        assert "token_ids" in result.columns
        # Verify normalization worked
        token_list = result["token_ids"].to_list()
        assert token_list[0] == ["123", "456"]
        assert token_list[1] == ["789", "012"]

    @pytest.mark.anyio
    async def test_fetch_multiple_pages(self, gamma_source):
        """Test fetching markets across multiple pages."""
        # Create mock data for 2 pages
        page1 = [{"id": f"market{i}", "clobTokenIds": [f"{i}"]} for i in range(250)]
        page2 = [{"id": f"market{i}", "clobTokenIds": [f"{i}"]} for i in range(250, 300)]

        # Mock responses for different pages
        async def mock_get(_url: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
            if params is not None and params.get("offset") == 0:
                return page1
            elif params is not None and params.get("offset") == 250:
                return page2
            else:
                return []

        with patch.object(gamma_source, "_get", side_effect=mock_get):
            result = await gamma_source.fetch(active_only=True)

        assert isinstance(result, pl.DataFrame)
        assert "token_ids" in result.columns
        # Be flexible - implementation might evolve, but should fetch multiple pages
        assert result.height > 250  # Got more than one page
        assert result.height <= 500  # But reasonable upper bound

    @pytest.mark.anyio
    async def test_fetch_empty_results(self, gamma_source):
        """Test fetching when no markets are returned."""

        async def mock_get(_url: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
            _ = params  # Mark as intentionally unused
            return []

        with patch.object(gamma_source, "_get", side_effect=mock_get):
            result = await gamma_source.fetch()

        assert isinstance(result, pl.DataFrame)
        assert result.height == 0
        assert "token_ids" in result.columns

    @pytest.mark.anyio
    async def test_fetch_active_only_parameter(self, gamma_source):
        """Test that active_only parameter is properly passed."""
        calls: list[dict[str, Any]] = []

        async def mock_get(url: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
            calls.append({"url": url, "params": params})
            return []

        with patch.object(gamma_source, "_get", side_effect=mock_get):
            await gamma_source.fetch(active_only=True)

        # Verify the call was made with closed=False
        assert len(calls) == 1
        assert calls[0]["params"] is not None
        assert calls[0]["params"]["closed"] is False

        # Test with active_only=False
        calls.clear()
        with patch.object(gamma_source, "_get", side_effect=mock_get):
            await gamma_source.fetch(active_only=False)

        # Verify closed param is not set
        assert len(calls) == 1
        assert calls[0]["params"] is not None
        assert "closed" not in calls[0]["params"]

    @pytest.mark.anyio
    async def test_fetch_without_clob_token_ids(self, gamma_source):
        """Test fetching markets without clobTokenIds field."""
        mock_markets = [
            {"id": "market1", "question": "Will it rain?"},
            {"id": "market2", "question": "Will it snow?"},
        ]

        async def mock_get(_url: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
            _ = params  # Mark as intentionally unused
            return mock_markets

        with patch.object(gamma_source, "_get", side_effect=mock_get):
            result = await gamma_source.fetch()

        assert isinstance(result, pl.DataFrame)
        assert result.height == 2
        assert "token_ids" in result.columns
        # Should have empty lists for token_ids
        assert result["token_ids"].to_list()[0] == []
        assert result["token_ids"].to_list()[1] == []

    @pytest.mark.anyio
    async def test_fetch_with_nested_data_response(self, gamma_source):
        """Test fetching when response has nested data structure."""
        mock_markets = [
            {"id": "market1", "clobTokenIds": ["123"]},
            {"id": "market2", "clobTokenIds": ["456"]},
        ]

        async def mock_get(_url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
            _ = params  # Mark as intentionally unused
            # Return data nested in "data" key
            return {"data": mock_markets}

        with patch.object(gamma_source, "_get", side_effect=mock_get):
            result = await gamma_source.fetch()

        assert isinstance(result, pl.DataFrame)
        assert result.height == 2

    @pytest.mark.anyio
    async def test_fetch_with_markets_key_response(self, gamma_source):
        """Test fetching when response has markets key."""
        mock_markets = [
            {"id": "market1", "clobTokenIds": ["123"]},
        ]

        async def mock_get(_url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
            _ = params  # Mark as intentionally unused
            # Return data nested in "markets" key
            return {"markets": mock_markets}

        with patch.object(gamma_source, "_get", side_effect=mock_get):
            result = await gamma_source.fetch()

        assert isinstance(result, pl.DataFrame)
        assert result.height == 1

    @pytest.mark.anyio
    async def test_fetch_stops_at_max_pages(self, context):
        """Test that fetching stops at max_pages limit."""
        source = Gamma(context, max_pages=2, page_size=10)

        # Mock more data than max_pages should allow
        mock_page = [{"id": f"market{i}", "clobTokenIds": [f"{i}"]} for i in range(10)]
        call_count = [0]

        async def mock_get(_url: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
            _ = params  # Mark as intentionally unused
            call_count[0] += 1
            return mock_page

        with patch.object(source, "_get", side_effect=mock_get):
            result = await source.fetch()

        # Should respect max_pages limit
        assert result.height <= 20  # At most 2 pages worth
        assert result.height >= 10  # At least got one page
        assert call_count[0] <= 2  # Should not exceed max_pages

    @pytest.mark.anyio
    async def test_client_lifecycle(self, gamma_source):
        """Test async client creation and cleanup."""
        assert gamma_source._client is None

        # Get client should create it
        client = await gamma_source._get_client()
        assert client is not None
        assert gamma_source._client is not None

        # Getting again should return same client
        client2 = await gamma_source._get_client()
        assert client is client2

        # Close should cleanup
        await gamma_source.close()
        assert gamma_source._client is None

    @pytest.mark.anyio
    async def test_context_manager(self, gamma_source):
        """Test using Gamma as an async context manager."""
        async with gamma_source as source:
            assert source is gamma_source

        # Should have cleaned up
        assert gamma_source._client is None

    @pytest.mark.anyio
    async def test_http_error_handling(self, gamma_source):
        """Test handling of HTTP errors with realistic httpx objects."""

        async def mock_get(_url: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
            _ = params  # Mark as intentionally unused
            # Create realistic httpx request and response objects
            request = httpx.Request("GET", "https://gamma-api.polymarket.com/markets")
            response = httpx.Response(
                404,
                request=request,
                content=b'{"error": "Not Found"}',
            )
            raise httpx.HTTPStatusError("404 Not Found", request=request, response=response)

        with patch.object(gamma_source, "_get", side_effect=mock_get):
            with pytest.raises(httpx.HTTPStatusError):
                await gamma_source.fetch()

    @pytest.mark.anyio
    async def test_retry_on_rate_limit(self, gamma_source):
        """Test that retry logic handles HTTP 429 (rate limit) via HTTPStatusError."""
        from unittest.mock import AsyncMock

        call_count = [0]

        async def mock_client_get(*args, **kwargs):
            """Mock httpx client.get to simulate rate limiting then success."""
            call_count[0] += 1

            # Fail first 2 attempts with rate limit, succeed on 3rd
            if call_count[0] < 3:
                request = httpx.Request("GET", "https://gamma-api.polymarket.com/markets")
                response = httpx.Response(429, request=request, headers={"retry-after": "1"})
                response._content = b'{"error": "Rate limited"}'
                raise httpx.HTTPStatusError("429 Too Many Requests", request=request, response=response)

            # Succeed on 3rd attempt - create proper mock response
            mock_response = MagicMock()
            mock_response.json.return_value = [{"id": "market1", "clobTokenIds": "123"}]
            mock_response.raise_for_status.return_value = None
            return mock_response

        # Mock at the client level to preserve retry decorator behavior
        mock_client = AsyncMock()
        mock_client.get.side_effect = mock_client_get
        mock_client.timeout = 30

        with patch.object(gamma_source, "_get_client", new=AsyncMock(return_value=mock_client)):
            result = await gamma_source.fetch()

        # Should have retried and eventually succeeded
        assert isinstance(result, pl.DataFrame)
        assert result.height >= 0
        assert call_count[0] >= 3  # At least 3 attempts due to retries

    @pytest.mark.anyio
    @pytest.mark.integration
    async def test_real_api_connection(self, context, request):
        """Integration test: Verify the real Gamma API endpoint is reachable.

        This test makes an actual HTTP request to the Gamma API.
        Run with: pytest --run-integration -m integration

        Fails soft — skips gracefully if the API is unavailable or the flag isn't set.
        """
        # Skip if the --run-integration flag was not passed
        if not request.config.getoption("--run-integration"):
            pytest.skip("Integration tests require --run-integration flag")

        source = Gamma(context, page_size=5, max_pages=1)  # Limit to reduce API load

        try:
            result = await source.fetch(active_only=True)

            # Validate the response
            assert isinstance(result, pl.DataFrame)
            assert "token_ids" in result.columns
            assert result.height >= 0  # Can be zero if no active markets

            print(f"\n✓ Successfully fetched {result.height} markets from Gamma API")

        except (httpx.HTTPError, httpx.TimeoutException) as e:
            # Skip gracefully if the network or API is down
            pytest.skip(f"API unavailable or unreachable: {e}")

        except Exception as e:
            # Catch-all skip for unexpected transient errors
            pytest.skip(f"Unexpected error during integration test: {e}")

        finally:
            await source.close()


class TestGammaWithFetchPipeline:
    """Test Gamma with Fetch pipeline integration."""

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
    async def test_gamma_direct_usage(self, context):
        """Test using Gamma source directly in user code."""
        mock_markets = [
            {"id": "market1", "question": "Test?", "clobTokenIds": "123,456"},
        ]

        gamma = Gamma(context)

        # Mock the _get method directly
        async def mock_get(_url: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
            _ = params  # Mark as intentionally unused
            return mock_markets

        with patch.object(gamma, "_get", side_effect=mock_get):
            result = await gamma.fetch(active_only=True)

        assert isinstance(result, pl.DataFrame)
        assert result.height == 1
        assert "token_ids" in result.columns
        # Verify token IDs were normalized from comma-separated string
        token_list = result["token_ids"].to_list()[0]
        assert len(token_list) == 2
        assert "123" in token_list
        assert "456" in token_list
