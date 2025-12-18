from __future__ import annotations

from pathlib import Path

import pytest

from polymorph.config import config as base_config
from polymorph.core.base import PipelineContext, RuntimeConfig
from polymorph.sources.gamma import GAMMA_BASE, Gamma
from polymorph.utils.time import utc


@pytest.mark.asyncio
async def test_gamma_fetch_markets_paginates_and_parses(tmp_path: Path) -> None:
    """Test that fetch_markets paginates and parses clobTokenIds correctly."""
    runtime_cfg = RuntimeConfig(http_timeout=None, max_concurrency=None, data_dir=str(tmp_path))
    context = PipelineContext(
        config=base_config,
        run_timestamp=utc(),
        data_dir=tmp_path,
        runtime_config=runtime_cfg,
    )

    gamma = Gamma(context, base_url=GAMMA_BASE, page_size=2, max_pages=2)

    pages: list[list[dict[str, object]]] = [
        [
            {
                "id": "m1",
                "question": "Will test pass?",
                "clobTokenIds": '["1", "2"]',  # Stringified JSON (actual API format)
                "closed": False,
                "resolved": False,
            },
            {
                "id": "m2",
                "question": "Is this resolved?",
                "clobTokenIds": '["3", "4"]',  # Stringified JSON
                "closed": True,
                "resolved": True,
            },
        ],
        [],
    ]

    async def fake_get(_url: str, params: dict[str, int | bool] | None = None):
        offset = params.get("offset", 0) if params else 0
        limit = params.get("limit", 2) if params else 2
        page_idx = offset // limit
        if page_idx >= len(pages):
            return []
        return pages[page_idx]

    gamma._get = fake_get  # type: ignore[method-assign]

    df = await gamma.fetch_markets(resolved_only=False)
    assert df.height == 2
    assert "id" in df.columns
    assert "question" in df.columns
    assert "closed" in df.columns
    assert "resolved" in df.columns
    assert "token_ids" in df.columns

    # Verify token IDs were parsed from stringified JSON
    token_ids = df["token_ids"].to_list()
    assert token_ids[0] == ["1", "2"]
    assert token_ids[1] == ["3", "4"]

    # Verify closed and resolved fields
    closed_values = df["closed"].to_list()
    assert closed_values == [False, True]
    resolved_values = df["resolved"].to_list()
    assert resolved_values == [False, True]


@pytest.mark.asyncio
async def test_gamma_fetch_respects_max_pages(tmp_path: Path) -> None:
    """Test that fetch_markets stops at max_pages limit."""
    runtime_cfg = RuntimeConfig(http_timeout=None, max_concurrency=None, data_dir=str(tmp_path))
    context = PipelineContext(
        config=base_config,
        run_timestamp=utc(),
        data_dir=tmp_path,
        runtime_config=runtime_cfg,
    )

    gamma = Gamma(context, page_size=10, max_pages=2)

    call_count = 0

    async def fake_get(_url: str, params: dict[str, int | bool] | None = None) -> list[dict[str, object]]:
        """Mock returning full pages."""
        nonlocal call_count
        call_count += 1

        # Always return full page of 10 markets
        markets = [
            {
                "id": f"m{i + (call_count - 1) * 10}",
                "question": f"Market {i}",
                "clobTokenIds": '["1", "2"]',
                "closed": False,
            }
            for i in range(10)
        ]
        return markets

    gamma._get = fake_get  # type: ignore[method-assign]

    df = await gamma.fetch_markets(resolved_only=False)

    # Should fetch exactly 2 pages (max_pages=2) with 10 markets each = 20 markets
    assert df.height == 20, f"Should fetch exactly 20 markets (2 pages), got {df.height}"
    assert call_count == 2, f"Should make exactly 2 API calls (max_pages=2), made {call_count}"


@pytest.mark.asyncio
async def test_gamma_fetch_empty_response_formats(tmp_path: Path) -> None:
    """Test handling of various empty response formats from API."""
    runtime_cfg = RuntimeConfig(http_timeout=None, max_concurrency=None, data_dir=str(tmp_path))
    context = PipelineContext(
        config=base_config,
        run_timestamp=utc(),
        data_dir=tmp_path,
        runtime_config=runtime_cfg,
    )

    gamma = Gamma(context)

    # Test: Empty list response (actual API format)
    async def fake_get_empty_list(_url: str, params: dict[str, int | bool] | None = None) -> list[dict[str, object]]:
        return []

    gamma._get = fake_get_empty_list  # type: ignore[method-assign]
    df = await gamma.fetch_markets(resolved_only=False)
    assert df.height == 0, "Empty list should return empty DataFrame"
    # Verify schema is correct even for empty DataFrame
    assert "id" in df.columns
    assert "closed" in df.columns
    assert "token_ids" in df.columns


@pytest.mark.asyncio
async def test_gamma_fetch_markets_filters_by_time_range(tmp_path: Path) -> None:
    """Test that fetch_markets filters markets by created_at timestamp."""
    runtime_cfg = RuntimeConfig(http_timeout=None, max_concurrency=None, data_dir=str(tmp_path))
    context = PipelineContext(
        config=base_config,
        run_timestamp=utc(),
        data_dir=tmp_path,
        runtime_config=runtime_cfg,
    )

    gamma = Gamma(context, page_size=10, max_pages=1)

    markets = [
        {
            "id": "m1",
            "question": "Old market",
            "clobTokenIds": '["1"]',
            "createdAt": "2024-01-01T00:00:00Z",  # Jan 1, 2024 = 1704067200000 ms
        },
        {
            "id": "m2",
            "question": "Mid market",
            "clobTokenIds": '["2"]',
            "createdAt": "2024-06-01T00:00:00Z",  # Jun 1, 2024 = 1717200000000 ms
        },
        {
            "id": "m3",
            "question": "Recent market",
            "clobTokenIds": '["3"]',
            "createdAt": "2024-12-01T00:00:00Z",  # Dec 1, 2024 = 1733011200000 ms
        },
    ]

    async def fake_get(_url: str, params: dict[str, int | bool] | None = None):
        return markets

    gamma._get = fake_get  # type: ignore[method-assign]

    # Test: Filter to markets created between June and December 2024
    start_ts = 1717200000000  # Jun 1, 2024
    end_ts = 1735689600000  # Jan 1, 2025
    df = await gamma.fetch_markets(resolved_only=False, start_ts=start_ts, end_ts=end_ts)

    assert df.height == 2, f"Should return 2 markets (m2 and m3), got {df.height}"
    market_ids = df["id"].to_list()
    assert "m2" in market_ids, "Should include m2 (mid market)"
    assert "m3" in market_ids, "Should include m3 (recent market)"
    assert "m1" not in market_ids, "Should exclude m1 (too old)"


@pytest.mark.asyncio
async def test_gamma_fetch_markets_filters_with_start_ts_only(tmp_path: Path) -> None:
    """Test that fetch_markets filters with only start_ts (no end_ts)."""
    runtime_cfg = RuntimeConfig(http_timeout=None, max_concurrency=None, data_dir=str(tmp_path))
    context = PipelineContext(
        config=base_config,
        run_timestamp=utc(),
        data_dir=tmp_path,
        runtime_config=runtime_cfg,
    )

    gamma = Gamma(context, page_size=10, max_pages=1)

    markets = [
        {"id": "m1", "question": "Old", "clobTokenIds": '["1"]', "createdAt": "2024-01-01T00:00:00Z"},
        {"id": "m2", "question": "Recent", "clobTokenIds": '["2"]', "createdAt": "2024-12-01T00:00:00Z"},
    ]

    async def fake_get(_url: str, params: dict[str, int | bool] | None = None):
        return markets

    gamma._get = fake_get  # type: ignore[method-assign]

    # Test: Only filter by minimum creation date
    start_ts = 1717200000000  # Jun 1, 2024
    df = await gamma.fetch_markets(resolved_only=False, start_ts=start_ts, end_ts=None)

    assert df.height == 1, f"Should return 1 market (m2), got {df.height}"
    assert df["id"][0] == "m2", "Should only include m2 (recent market)"


@pytest.mark.asyncio
async def test_gamma_fetch_markets_handles_missing_created_at(tmp_path: Path) -> None:
    """Test that markets without created_at are excluded when time filtering."""
    runtime_cfg = RuntimeConfig(http_timeout=None, max_concurrency=None, data_dir=str(tmp_path))
    context = PipelineContext(
        config=base_config,
        run_timestamp=utc(),
        data_dir=tmp_path,
        runtime_config=runtime_cfg,
    )

    gamma = Gamma(context, page_size=10, max_pages=1)

    markets = [
        {"id": "m1", "question": "Has timestamp", "clobTokenIds": '["1"]', "createdAt": "2024-06-01T00:00:00Z"},
        {"id": "m2", "question": "No timestamp", "clobTokenIds": '["2"]'},  # Missing createdAt
    ]

    async def fake_get(_url: str, params: dict[str, int | bool] | None = None):
        return markets

    gamma._get = fake_get  # type: ignore[method-assign]

    # Test: With time filtering, market without created_at is excluded
    start_ts = 1704067200000  # Jan 1, 2024
    df = await gamma.fetch_markets(resolved_only=False, start_ts=start_ts)

    assert df.height == 1, f"Should return 1 market (m1), got {df.height}"
    assert df["id"][0] == "m1", "Should only include m1 (has created_at)"


@pytest.mark.asyncio
async def test_gamma_fetch_markets_handles_invalid_created_at(tmp_path: Path) -> None:
    """Test that markets with invalid created_at are excluded when time filtering."""
    runtime_cfg = RuntimeConfig(http_timeout=None, max_concurrency=None, data_dir=str(tmp_path))
    context = PipelineContext(
        config=base_config,
        run_timestamp=utc(),
        data_dir=tmp_path,
        runtime_config=runtime_cfg,
    )

    gamma = Gamma(context, page_size=10, max_pages=1)

    markets = [
        {"id": "m1", "question": "Valid timestamp", "clobTokenIds": '["1"]', "createdAt": "2024-06-01T00:00:00Z"},
        {"id": "m2", "question": "Invalid timestamp", "clobTokenIds": '["2"]', "createdAt": "not-a-date"},
    ]

    async def fake_get(_url: str, params: dict[str, int | bool] | None = None):
        return markets

    gamma._get = fake_get  # type: ignore[method-assign]

    # Test: With time filtering, market with invalid created_at is excluded
    start_ts = 1704067200000  # Jan 1, 2024
    df = await gamma.fetch_markets(resolved_only=False, start_ts=start_ts)

    assert df.height == 1, f"Should return 1 market (m1), got {df.height}"
    assert df["id"][0] == "m1", "Should only include m1 (valid created_at)"


@pytest.mark.asyncio
async def test_gamma_fetch_markets_no_time_filter_includes_all(tmp_path: Path) -> None:
    """Test that without time filtering, all markets are included (including those without created_at)."""
    runtime_cfg = RuntimeConfig(http_timeout=None, max_concurrency=None, data_dir=str(tmp_path))
    context = PipelineContext(
        config=base_config,
        run_timestamp=utc(),
        data_dir=tmp_path,
        runtime_config=runtime_cfg,
    )

    gamma = Gamma(context, page_size=10, max_pages=1)

    markets = [
        {"id": "m1", "question": "Has timestamp", "clobTokenIds": '["1"]', "createdAt": "2024-06-01T00:00:00Z"},
        {"id": "m2", "question": "No timestamp", "clobTokenIds": '["2"]'},  # Missing createdAt
    ]

    async def fake_get(_url: str, params: dict[str, int | bool] | None = None):
        return markets

    gamma._get = fake_get  # type: ignore[method-assign]

    # Test: Without time filtering, all markets are included
    df = await gamma.fetch_markets(resolved_only=False)

    assert df.height == 2, f"Should return 2 markets, got {df.height}"
    market_ids = df["id"].to_list()
    assert "m1" in market_ids
    assert "m2" in market_ids
