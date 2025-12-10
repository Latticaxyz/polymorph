from __future__ import annotations

from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from polymorph.config import config as base_config
from polymorph.core.base import PipelineContext, RuntimeConfig
from polymorph.sources.gamma import GAMMA_BASE, Gamma
from polymorph.utils.time import utc


@given(
    st.one_of(
        st.none(),
        st.text(),
        st.integers(),
        st.floats(allow_nan=False, allow_infinity=False),
        st.lists(st.one_of(st.text(), st.integers(), st.none()), max_size=5),
    )
)
def test_gamma_normalize_ids_returns_strings(v: object) -> None:
    ids = Gamma._normalize_ids(v)
    assert isinstance(ids, list)
    assert all(isinstance(s, str) for s in ids)


def test_gamma_classify_market_type_basic_categories() -> None:
    assert Gamma._classify_market_type({"question": "Who will win the election?"}) == "election"
    assert Gamma._classify_market_type({"question": "NBA finals", "tags": ["sports"]}) == "sports"
    assert Gamma._classify_market_type({"question": "BTC price above 100k?", "tags": []}) == "crypto"
    assert Gamma._classify_market_type({"question": "Will BTC hit 50k by 2025?"}) == "deadline"
    assert Gamma._classify_market_type({"question": "Some weird market"}) == "other"


@pytest.mark.asyncio
async def test_gamma_fetch_paginates_and_normalizes(tmp_path: Path) -> None:
    runtime_cfg = RuntimeConfig(http_timeout=None, max_concurrency=None, data_dir=str(tmp_path))
    context = PipelineContext(
        config=base_config,
        run_timestamp=utc(),
        data_dir=tmp_path,
        runtime_config=runtime_cfg,
    )

    gamma = Gamma(context, base_url=GAMMA_BASE, page_size=2, max_pages=None)

    pages: list[list[dict[str, object]]] = [
        [
            {
                "id": "m1",
                "question": "Who will win the election?",
                "tags": ["politics"],
                "clobTokenIds": [1, 2],
                "closed": False,
            },
            {
                "id": "m2",
                "question": "NBA finals winner",
                "tags": ["NBA"],
                "clobTokenIds": "[3,4]",
                "closed": True,
            },
        ],
        [],
    ]

    async def fake_get(_url: str, params: dict[str, int | bool] | None = None):
        offset = params.get("offset", 0) if params else 0
        limit = params.get("limit", 2) if params else 2
        page_idx = offset // limit
        if page_idx >= len(pages):
            return {"markets": []}
        return {"markets": pages[page_idx]}

    gamma._get = fake_get  # type: ignore[method-assign]

    df = await gamma.fetch(active_only=True, max_markets=None, resolved_only=False, include_resolution_data=True)
    assert df.height == 2
    assert "token_ids" in df.columns
    token_ids = df["token_ids"].to_list()
    assert token_ids[0] == ["1", "2"]
    assert token_ids[1] == ["3", "4"]
    assert "market_type" in df.columns
    assert set(df["market_type"].to_list()) == {"election", "sports"}
    assert "resolved" in df.columns
    assert df["resolved"].to_list() == [False, True]


# ============================================================================
# EDGE CASE TESTS
# ============================================================================


def test_gamma_normalize_ids_edge_cases() -> None:
    """Test _normalize_ids handles all edge cases correctly."""
    # Empty/null inputs
    assert Gamma._normalize_ids(None) == []
    assert Gamma._normalize_ids("") == [""]  # Empty string becomes list with empty string
    assert Gamma._normalize_ids([]) == []

    # JSON array formats
    assert Gamma._normalize_ids("[1,2,3]") == ["1", "2", "3"]
    assert Gamma._normalize_ids('["a","b"]') == ["a", "b"]
    assert Gamma._normalize_ids("[null,1,null,2]") == ["1", "2"]  # Filters nulls

    # CSV formats
    assert Gamma._normalize_ids("a,b,c") == ["a", "b", "c"]
    assert Gamma._normalize_ids(" a , b , c ") == ["a", "b", "c"]  # Strips whitespace
    assert Gamma._normalize_ids("single") == ["single"]

    # List formats
    assert Gamma._normalize_ids([1, 2, None, 3]) == ["1", "2", "3"]  # Filters nulls
    assert Gamma._normalize_ids(["a", "b"]) == ["a", "b"]

    # Numeric formats
    assert Gamma._normalize_ids(42) == ["42"]
    assert Gamma._normalize_ids(3.14) == ["3.14"]

    # Malformed JSON (should fall back to single string)
    assert Gamma._normalize_ids("[invalid json") == ["[invalid json"]


def test_gamma_classify_market_type_edge_cases() -> None:
    """Test _classify_market_type handles edge cases and priority."""
    # Empty/minimal data
    assert Gamma._classify_market_type({}) == "other"
    assert Gamma._classify_market_type({"question": ""}) == "other"
    assert Gamma._classify_market_type({"question": None}) == "other"

    # Priority testing (election > deadline > sports > crypto)
    assert (
        Gamma._classify_market_type({"question": "Will the presidential election affect crypto by 2025?"}) == "election"
    ), "Election should take highest priority"

    assert (
        Gamma._classify_market_type({"question": "Will BTC hit 100k by 2025?"}) == "deadline"
    ), "Deadline should override crypto"

    assert (
        Gamma._classify_market_type({"question": "Will crypto affect NBA finals?"}) == "sports"
    ), "Sports should override crypto"

    # Case insensitivity
    assert Gamma._classify_market_type({"question": "ELECTION 2024"}) == "election"
    assert Gamma._classify_market_type({"question": "nba FINALS"}) == "sports"

    # Tags-only classification
    assert Gamma._classify_market_type({"question": "Random market", "tags": ["NBA"]}) == "sports"
    assert Gamma._classify_market_type({"question": "Random market", "tags": None}) == "other"

    # Deadline detection requires question mark
    assert Gamma._classify_market_type({"question": "Will it happen by 2025?"}) == "deadline"
    assert Gamma._classify_market_type({"question": "It will happen by 2025"}) == "other"  # No question mark


@pytest.mark.asyncio
async def test_gamma_fetch_respects_max_markets_mid_page(tmp_path: Path) -> None:
    """Test that fetch stops exactly at max_markets even mid-page."""
    runtime_cfg = RuntimeConfig(http_timeout=None, max_concurrency=None, data_dir=str(tmp_path))
    context = PipelineContext(
        config=base_config,
        run_timestamp=utc(),
        data_dir=tmp_path,
        runtime_config=runtime_cfg,
    )

    gamma = Gamma(context, page_size=10, max_pages=None)

    call_count = 0

    async def fake_get(_url: str, params: dict[str, int | bool] | None = None) -> dict[str, list[dict[str, object]]]:
        """Mock returning full pages."""
        nonlocal call_count
        call_count += 1

        # Always return full page of 10 markets
        markets = [
            {
                "id": f"m{i + (call_count - 1) * 10}",
                "question": f"Question {i}",
                "clobTokenIds": ["1", "2"],
                "closed": False,
            }
            for i in range(10)
        ]
        return {"markets": markets}

    gamma._get = fake_get  # type: ignore[method-assign]

    # Request exactly 15 markets (should stop mid-second page)
    df = await gamma.fetch(active_only=True, max_markets=15, resolved_only=False, include_resolution_data=True)

    assert df.height == 15, "Should stop exactly at max_markets"
    assert call_count == 2, "Should make exactly 2 API calls for 15 markets with page_size=10"


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

    # Test 1: Empty list response
    async def fake_get_empty_list(_url: str, params: dict[str, int | bool] | None = None) -> list[dict[str, object]]:
        return []

    gamma._get = fake_get_empty_list  # type: ignore[method-assign]
    df1 = await gamma.fetch()
    assert df1.height == 0, "Empty list should return empty DataFrame"

    # Test 2: Empty dict with markets key
    async def fake_get_empty_dict(
        _url: str, params: dict[str, int | bool] | None = None
    ) -> dict[str, list[dict[str, object]]]:
        return {"markets": []}

    gamma._get = fake_get_empty_dict  # type: ignore[method-assign]
    df2 = await gamma.fetch()
    assert df2.height == 0, "Empty markets list should return empty DataFrame"

    # Test 3: Missing expected keys
    async def fake_get_no_markets_key(_url: str, params: dict[str, int | bool] | None = None) -> dict[str, object]:
        return {"some_other_key": "value"}

    gamma._get = fake_get_no_markets_key  # type: ignore[method-assign]
    df3 = await gamma.fetch()
    assert df3.height == 0, "Response without markets key should return empty DataFrame"
