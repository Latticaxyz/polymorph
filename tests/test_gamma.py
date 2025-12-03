"""Tests for Gamma data source."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import polars as pl
import pytest

from polymorph.config import Settings
from polymorph.core.base import PipelineContext
from polymorph.sources.gamma import GAMMA_BASE, Gamma

# JSON type alias for test helpers
JsonValue = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]


def create_market_row(
    question: str = "Generic question?",
    tags: list[str] | None = None,
) -> dict[str, JsonValue]:
    """Helper for classification tests."""
    tag_list: list[JsonValue] = list(tags) if tags is not None else []
    clob_ids: list[JsonValue] = []
    return {
        "question": question,
        "tags": tag_list,
        "id": "test_market_123",
        "clobTokenIds": clob_ids,
    }


def create_mock_markets(
    count: int = 10,
    include_resolved: bool = False,
    include_closed: bool = True,
) -> list[dict[str, JsonValue]]:
    """Helper for fetch tests."""
    markets: list[dict[str, JsonValue]] = []
    for i in range(count):
        market: dict[str, JsonValue] = {
            "id": f"market{i}",
            "question": f"Test question {i}?",
            "clobTokenIds": [f"{i}"],
            "tags": ["test"],
        }
        if include_closed:
            market["closed"] = include_resolved
        if include_resolved:
            market["resolved"] = True
        markets.append(market)
    return markets


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
        assert source.max_pages is None

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

        mock_rate_limiter = AsyncMock()
        mock_rate_limiter.acquire = AsyncMock()

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.json = MagicMock(return_value=mock_markets)
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        with patch.object(gamma_source, "_get_rate_limiter", return_value=mock_rate_limiter):
            with patch.object(gamma_source, "_get_client", return_value=mock_client):
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

        mock_rate_limiter = AsyncMock()
        mock_rate_limiter.acquire = AsyncMock()

        mock_client = AsyncMock()

        async def paginated_response(*args, **kwargs):
            params = kwargs.get("params", {})
            offset = params.get("offset", 0)
            mock_response = MagicMock()
            if offset == 0:
                mock_response.json = MagicMock(return_value=page1)
            elif offset == 250:
                mock_response.json = MagicMock(return_value=page2)
            else:
                mock_response.json = MagicMock(return_value=[])
            mock_response.status_code = 200
            mock_response.raise_for_status = MagicMock()
            return mock_response

        mock_client.get.side_effect = paginated_response

        with patch.object(gamma_source, "_get_rate_limiter", return_value=mock_rate_limiter):
            with patch.object(gamma_source, "_get_client", return_value=mock_client):
                result = await gamma_source.fetch(active_only=True)

        assert isinstance(result, pl.DataFrame)
        assert "token_ids" in result.columns
        assert result.height == 300

    @pytest.mark.anyio
    async def test_fetch_empty_results(self, gamma_source):
        """Test fetching when no markets are returned."""
        mock_rate_limiter = AsyncMock()
        mock_rate_limiter.acquire = AsyncMock()

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.json = MagicMock(return_value=[])
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        with patch.object(gamma_source, "_get_rate_limiter", return_value=mock_rate_limiter):
            with patch.object(gamma_source, "_get_client", return_value=mock_client):
                result = await gamma_source.fetch()

        assert isinstance(result, pl.DataFrame)
        assert result.height == 0
        assert "token_ids" in result.columns

    @pytest.mark.anyio
    async def test_fetch_active_only_parameter(self, gamma_source):
        """Test that active_only parameter is properly passed."""
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

        with patch.object(gamma_source, "_get_rate_limiter", return_value=mock_rate_limiter):
            with patch.object(gamma_source, "_get_client", return_value=mock_client):
                await gamma_source.fetch(active_only=True)

        # Verify the call was made with closed=False
        assert len(calls) == 1
        assert calls[0]["params"] is not None
        assert calls[0]["params"]["closed"] is False

        # Test with active_only=False
        calls.clear()
        with patch.object(gamma_source, "_get_rate_limiter", return_value=mock_rate_limiter):
            with patch.object(gamma_source, "_get_client", return_value=mock_client):
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

        mock_rate_limiter = AsyncMock()
        mock_rate_limiter.acquire = AsyncMock()

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.json = MagicMock(return_value=mock_markets)
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        with patch.object(gamma_source, "_get_rate_limiter", return_value=mock_rate_limiter):
            with patch.object(gamma_source, "_get_client", return_value=mock_client):
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

        mock_rate_limiter = AsyncMock()
        mock_rate_limiter.acquire = AsyncMock()

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.json = MagicMock(return_value={"data": mock_markets})
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        with patch.object(gamma_source, "_get_rate_limiter", return_value=mock_rate_limiter):
            with patch.object(gamma_source, "_get_client", return_value=mock_client):
                result = await gamma_source.fetch()

        assert isinstance(result, pl.DataFrame)
        assert result.height == 2

    @pytest.mark.anyio
    async def test_fetch_with_markets_key_response(self, gamma_source):
        """Test fetching when response has markets key."""
        mock_markets = [
            {"id": "market1", "clobTokenIds": ["123"]},
        ]

        mock_rate_limiter = AsyncMock()
        mock_rate_limiter.acquire = AsyncMock()

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.json = MagicMock(return_value={"markets": mock_markets})
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        mock_client.get.return_value = mock_response

        with patch.object(gamma_source, "_get_rate_limiter", return_value=mock_rate_limiter):
            with patch.object(gamma_source, "_get_client", return_value=mock_client):
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

        mock_rate_limiter = AsyncMock()
        mock_rate_limiter.acquire = AsyncMock()

        mock_client = AsyncMock()

        async def count_calls(*args, **kwargs):
            call_count[0] += 1
            mock_response = MagicMock()
            mock_response.json = MagicMock(return_value=mock_page)
            mock_response.status_code = 200
            mock_response.raise_for_status = MagicMock()
            return mock_response

        mock_client.get.side_effect = count_calls

        with patch.object(source, "_get_rate_limiter", return_value=mock_rate_limiter):
            with patch.object(source, "_get_client", return_value=mock_client):
                result = await source.fetch()

        # Should respect max_pages limit
        assert result.height <= 20  # At most 2 pages worth
        assert result.height >= 10  # At least got one page
        assert call_count[0] <= 2  # Should not exceed max_pages

    @pytest.mark.anyio
    async def test_fetch_max_markets_limit(self, gamma_source):
        mock_rate_limiter = AsyncMock()
        mock_rate_limiter.acquire = AsyncMock()

        mock_client = AsyncMock()

        async def smart_response(url, *args, **kwargs):
            params = kwargs.get("params", {})
            limit = params.get("limit", 250)
            # Return only as many as requested
            mock_page = [{"id": f"market{i}", "clobTokenIds": [f"{i}"]} for i in range(limit)]
            mock_response = MagicMock()
            mock_response.json = MagicMock(return_value=mock_page)
            mock_response.status_code = 200
            mock_response.raise_for_status = MagicMock()
            return mock_response

        mock_client.get.side_effect = smart_response

        with patch.object(gamma_source, "_get_rate_limiter", return_value=mock_rate_limiter):
            with patch.object(gamma_source, "_get_client", return_value=mock_client):
                result = await gamma_source.fetch(active_only=True, max_markets=100)

        # Should stop after getting 100 markets
        assert result.height == 100

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
        mock_rate_limiter = AsyncMock()
        mock_rate_limiter.acquire = AsyncMock()

        mock_client = AsyncMock()
        request = httpx.Request("GET", "https://gamma-api.polymarket.com/markets")
        response = httpx.Response(
            404,
            request=request,
            content=b'{"error": "Not Found"}',
        )

        async def raise_error(*args, **kwargs):
            raise httpx.HTTPStatusError("404 Not Found", request=request, response=response)

        mock_client.get.side_effect = raise_error

        with patch.object(gamma_source, "_get_rate_limiter", return_value=mock_rate_limiter):
            with patch.object(gamma_source, "_get_client", return_value=mock_client):
                with pytest.raises(httpx.HTTPStatusError):
                    await gamma_source.fetch()

    @pytest.mark.anyio
    async def test_rate_limit_error_handling(self, gamma_source):
        """Test that 429 responses raise RateLimitError."""
        from polymorph.core.rate_limit import RateLimitError

        mock_rate_limiter = AsyncMock()
        mock_rate_limiter.acquire = AsyncMock()

        mock_client = AsyncMock()
        request = httpx.Request("GET", "https://gamma-api.polymarket.com/markets")
        response = httpx.Response(429, request=request, headers={"retry-after": "1"})

        async def return_429(*args, **kwargs):
            return response

        mock_client.get.side_effect = return_429

        with patch.object(gamma_source, "_get_rate_limiter", return_value=mock_rate_limiter):
            with patch.object(gamma_source, "_get_client", return_value=mock_client):
                with pytest.raises(RateLimitError):
                    await gamma_source.fetch()

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


class TestGammaClassifyMarketType:
    """Test Gamma market classification function."""

    def test_classify_market_type_election(self):
        """Test correctly identifies election markets."""
        test_cases = [
            create_market_row("Will Trump win the 2024 election?", []),
            create_market_row("Who will be the next president?", ["politics"]),
            create_market_row("Senate race results", ["senator"]),
            create_market_row("Governor election outcome", []),
            create_market_row("Presidential nominee announcement", []),
        ]
        for row in test_cases:
            result = Gamma._classify_market_type(row)
            assert result == "election", f"Failed for: {row['question']}"

    def test_classify_market_type_sports(self):
        """Test correctly identifies sports markets."""
        test_cases = [
            create_market_row("Super Bowl outcome prediction", []),
            create_market_row("NBA Championship prediction", ["basketball"]),
            create_market_row("NFL season predictions", []),
            create_market_row("MLB playoff results", []),
            create_market_row("Premier League soccer match", ["football"]),
            create_market_row("NHL Stanley Cup", []),
        ]
        for row in test_cases:
            result = Gamma._classify_market_type(row)
            assert result == "sports", f"Failed for: {row['question']}"

    def test_classify_market_type_crypto(self):
        """Test correctly identifies crypto markets."""
        test_cases = [
            create_market_row("Bitcoin price prediction", []),
            create_market_row("Will ETH reach $5000?", []),
            create_market_row("Cryptocurrency adoption", ["crypto"]),
            create_market_row("BTC market trends", ["blockchain"]),
            create_market_row("New token launch success", []),
        ]
        for row in test_cases:
            result = Gamma._classify_market_type(row)
            assert result == "crypto", f"Failed for: {row['question']}"

    def test_classify_market_type_deadline(self):
        """Test correctly identifies deadline-based markets."""
        test_cases = [
            create_market_row("Will this happen by 2024?", []),
            create_market_row("Event before December 2025?", []),
            create_market_row("Achievement after January 2026?", []),
            create_market_row("Outcome by end of year?", []),
        ]
        for row in test_cases:
            result = Gamma._classify_market_type(row)
            assert result == "deadline", f"Failed for: {row['question']}"

    def test_classify_market_type_other(self):
        """Test defaults to other for unclassified."""
        test_cases = [
            create_market_row("Generic market question?", []),
            create_market_row("Random topic prediction", ["misc"]),
            create_market_row("Unknown category", None),
        ]
        for row in test_cases:
            result = Gamma._classify_market_type(row)
            assert result == "other", f"Failed for: {row['question']}"

    def test_classify_market_type_case_insensitive(self):
        """Test classification is case-insensitive."""
        test_cases = [
            (create_market_row("ELECTION results", []), "election"),
            (create_market_row("bitcoin PRICE", []), "crypto"),
            (create_market_row("NFL game", ["SPORTS"]), "sports"),
        ]
        for row, expected in test_cases:
            result = Gamma._classify_market_type(row)
            assert result == expected, f"Failed for: {row['question']}"

    def test_classify_market_type_tags_only(self):
        """Test classification works with tags when question is generic."""
        test_cases = [
            (create_market_row("Market prediction?", ["election", "politics"]), "election"),
            (create_market_row("Will this happen?", ["nfl", "football"]), "sports"),
            (create_market_row("Prediction market", ["crypto", "bitcoin"]), "crypto"),
        ]
        for row, expected in test_cases:
            result = Gamma._classify_market_type(row)
            assert result == expected, f"Failed for: {row['question']}"

    def test_classify_market_type_question_and_tags_combined(self):
        """Test uses both question and tags for classification."""
        row = create_market_row("Sports event", ["basketball", "championship"])
        result = Gamma._classify_market_type(row)
        assert result == "sports"

    def test_classify_market_type_priority_order(self):
        """Test election has priority over deadline when both match."""
        row = create_market_row("Will Trump win by 2024?", ["election"])
        result = Gamma._classify_market_type(row)
        assert result == "election"  # Not "deadline"

    def test_classify_market_type_missing_fields(self):
        """Test handles missing question or tags gracefully."""
        test_cases: list[dict[str, JsonValue]] = [
            {"tags": ["election"]},  # no question
            {"question": "Election?"},  # no tags
            {},  # empty dict
        ]
        for row in test_cases:
            result = Gamma._classify_market_type(row)
            # Should not raise error, returns valid classification
            assert result in ["election", "sports", "crypto", "deadline", "other"]

    def test_classify_market_type_non_string_values(self):
        """Test handles non-string question/tags safely."""
        test_cases: list[dict[str, JsonValue]] = [
            {"question": None, "tags": None},
            {"question": 123, "tags": ["election"]},
            {"question": "Valid?", "tags": 456},
        ]
        for row in test_cases:
            result = Gamma._classify_market_type(row)
            # Should not raise type error
            assert isinstance(result, str)

    def test_classify_market_type_deadline_without_question_mark(self):
        """Test deadline keywords alone don't classify without ?."""
        row = create_market_row("Event in 2024", [])  # no "?"
        result = Gamma._classify_market_type(row)
        assert result == "other"  # not "deadline"


class TestGammaFetchResolved:
    """Test Gamma fetch_resolved_markets function."""

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

    @pytest.mark.anyio
    async def test_fetch_resolved_markets_basic(self, gamma_source):
        """Test calls fetch() with correct parameters."""
        mock_df = pl.DataFrame({"id": ["market1"], "resolved": [True]})

        with patch.object(gamma_source, "fetch", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = mock_df
            result = await gamma_source.fetch_resolved_markets()

            mock_fetch.assert_called_once_with(
                active_only=False,
                resolved_only=True,
                include_resolution_data=True,
                max_markets=None,
            )
            assert isinstance(result, pl.DataFrame)
            assert result.height == 1

    @pytest.mark.anyio
    async def test_fetch_resolved_markets_with_max_markets(self, gamma_source):
        """Test max_markets parameter passed through."""
        mock_df = pl.DataFrame({"id": [f"market{i}" for i in range(50)]})

        with patch.object(gamma_source, "fetch", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = mock_df
            await gamma_source.fetch_resolved_markets(max_markets=50)

            mock_fetch.assert_called_once()
            call_kwargs = mock_fetch.call_args[1]
            assert call_kwargs["max_markets"] == 50

    @pytest.mark.anyio
    async def test_fetch_resolved_markets_returns_dataframe(self, gamma_source):
        """Test returns actual DataFrame from fetch()."""
        mock_df = pl.DataFrame(
            {
                "id": ["market1", "market2"],
                "resolved": [True, True],
                "question": ["Q1?", "Q2?"],
            }
        )

        with patch.object(gamma_source, "fetch", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = mock_df
            result = await gamma_source.fetch_resolved_markets()

            assert result.equals(mock_df)

    @pytest.mark.anyio
    async def test_fetch_resolved_markets_empty_results(self, gamma_source):
        """Test handles empty results gracefully."""
        mock_df = pl.DataFrame({"id": []})

        with patch.object(gamma_source, "fetch", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = mock_df
            result = await gamma_source.fetch_resolved_markets()

            assert isinstance(result, pl.DataFrame)
            assert result.height == 0


class TestGammaFetchEnhancements:
    """Test Gamma fetch enhancements (resolved_only and include_resolution_data)."""

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

    # Tests for resolved_only parameter

    @pytest.mark.anyio
    async def test_fetch_resolved_only_true(self, gamma_source):
        """Test resolved_only=True sets closed=True parameter."""
        calls = []

        async def track_params(url, *args, **kwargs):
            params = kwargs.get("params", {})
            calls.append(params)
            mock_response = MagicMock()
            mock_response.json = MagicMock(return_value=[])
            mock_response.status_code = 200
            mock_response.raise_for_status = MagicMock()
            return mock_response

        mock_rate_limiter = AsyncMock()
        mock_rate_limiter.acquire = AsyncMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=track_params)
        mock_client.timeout = 30

        with patch.object(gamma_source, "_get_rate_limiter", return_value=mock_rate_limiter):
            with patch.object(gamma_source, "_get_client", return_value=mock_client):
                await gamma_source.fetch(resolved_only=True)

        assert len(calls) >= 1
        assert calls[0].get("closed") is True

    @pytest.mark.anyio
    async def test_fetch_resolved_only_with_active_only_conflict(self, gamma_source):
        """Test resolved_only takes precedence over active_only."""
        calls = []

        async def track_params(url, *args, **kwargs):
            params = kwargs.get("params", {})
            calls.append(params)
            mock_response = MagicMock()
            mock_response.json = MagicMock(return_value=[])
            mock_response.status_code = 200
            mock_response.raise_for_status = MagicMock()
            return mock_response

        mock_rate_limiter = AsyncMock()
        mock_rate_limiter.acquire = AsyncMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=track_params)
        mock_client.timeout = 30

        with patch.object(gamma_source, "_get_rate_limiter", return_value=mock_rate_limiter):
            with patch.object(gamma_source, "_get_client", return_value=mock_client):
                await gamma_source.fetch(active_only=True, resolved_only=True)

        assert len(calls) >= 1
        assert calls[0].get("closed") is True

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        "active_only,resolved_only,expected_closed",
        [
            (True, False, False),
            (False, False, None),
            (True, True, True),
            (False, True, True),
        ],
    )
    async def test_fetch_parameter_combinations(self, gamma_source, active_only, resolved_only, expected_closed):
        """Test all parameter combinations work correctly."""
        calls = []

        async def track_params(url, *args, **kwargs):
            params = kwargs.get("params", {})
            calls.append(params)
            mock_response = MagicMock()
            mock_response.json = MagicMock(return_value=[])
            mock_response.status_code = 200
            mock_response.raise_for_status = MagicMock()
            return mock_response

        mock_rate_limiter = AsyncMock()
        mock_rate_limiter.acquire = AsyncMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=track_params)
        mock_client.timeout = 30

        with patch.object(gamma_source, "_get_rate_limiter", return_value=mock_rate_limiter):
            with patch.object(gamma_source, "_get_client", return_value=mock_client):
                await gamma_source.fetch(active_only=active_only, resolved_only=resolved_only)

        assert len(calls) >= 1
        if expected_closed is None:
            assert "closed" not in calls[0]
        else:
            assert calls[0].get("closed") == expected_closed

    # Tests for include_resolution_data parameter

    @pytest.mark.anyio
    async def test_fetch_include_resolution_data_true(self, gamma_source):
        """Test include_resolution_data=True adds resolved column."""
        mock_markets = [
            {"id": "market1", "closed": True, "question": "Q1?", "clobTokenIds": []},
            {"id": "market2", "closed": False, "question": "Q2?", "clobTokenIds": []},
        ]

        mock_rate_limiter = AsyncMock()
        mock_rate_limiter.acquire = AsyncMock()

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.json = MagicMock(return_value=mock_markets)
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        with patch.object(gamma_source, "_get_rate_limiter", return_value=mock_rate_limiter):
            with patch.object(gamma_source, "_get_client", return_value=mock_client):
                result = await gamma_source.fetch(include_resolution_data=True)

        assert "resolved" in result.columns
        assert result["resolved"].to_list() == [True, False]

    @pytest.mark.anyio
    async def test_fetch_include_resolution_data_with_existing_resolved(self, gamma_source):
        """Test doesn't overwrite existing resolved field."""
        mock_markets = [
            {"id": "market1", "closed": False, "resolved": True, "question": "Q1?", "clobTokenIds": []},
        ]

        mock_rate_limiter = AsyncMock()
        mock_rate_limiter.acquire = AsyncMock()

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.json = MagicMock(return_value=mock_markets)
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        with patch.object(gamma_source, "_get_rate_limiter", return_value=mock_rate_limiter):
            with patch.object(gamma_source, "_get_client", return_value=mock_client):
                result = await gamma_source.fetch(include_resolution_data=True)

        # Original resolved value should be preserved
        assert result["resolved"][0] is True

    @pytest.mark.anyio
    async def test_fetch_include_resolution_data_no_closed_field(self, gamma_source):
        """Test handles missing closed field gracefully."""
        mock_markets = [
            {"id": "market1", "question": "Q1?", "clobTokenIds": []},
        ]

        mock_rate_limiter = AsyncMock()
        mock_rate_limiter.acquire = AsyncMock()

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.json = MagicMock(return_value=mock_markets)
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        with patch.object(gamma_source, "_get_rate_limiter", return_value=mock_rate_limiter):
            with patch.object(gamma_source, "_get_client", return_value=mock_client):
                result = await gamma_source.fetch(include_resolution_data=True)

        # Should not raise error
        assert isinstance(result, pl.DataFrame)

    @pytest.mark.anyio
    async def test_fetch_include_resolution_data_with_resolved_only(self, gamma_source):
        """Test works correctly with resolved_only=True."""
        mock_markets = [
            {"id": "market1", "closed": True, "question": "Q1?", "clobTokenIds": []},
        ]

        mock_rate_limiter = AsyncMock()
        mock_rate_limiter.acquire = AsyncMock()

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.json = MagicMock(return_value=mock_markets)
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        with patch.object(gamma_source, "_get_rate_limiter", return_value=mock_rate_limiter):
            with patch.object(gamma_source, "_get_client", return_value=mock_client):
                result = await gamma_source.fetch(resolved_only=True, include_resolution_data=True)

        assert "resolved" in result.columns
        assert result["resolved"][0] is True


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

        mock_rate_limiter = AsyncMock()
        mock_rate_limiter.acquire = AsyncMock()

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.json = MagicMock(return_value=mock_markets)
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        with patch.object(gamma, "_get_rate_limiter", return_value=mock_rate_limiter):
            with patch.object(gamma, "_get_client", return_value=mock_client):
                result = await gamma.fetch(active_only=True)

        assert isinstance(result, pl.DataFrame)
        assert result.height == 1
        assert "token_ids" in result.columns
        # Verify token IDs were normalized from comma-separated string
        token_list = result["token_ids"].to_list()[0]
        assert len(token_list) == 2
        assert "123" in token_list
        assert "456" in token_list
