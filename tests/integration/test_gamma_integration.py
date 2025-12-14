from __future__ import annotations

import pytest

from polymorph.sources.gamma import Gamma


@pytest.mark.integration
@pytest.mark.asyncio
async def test_gamma_markets_response_format(gamma_client: Gamma) -> None:
    """Verify actual Gamma /markets endpoint response format."""
    df = await gamma_client.fetch_markets(resolved_only=False)

    # Should get some markets back
    assert df.height > 0, "Expected at least some markets from API"

    # Verify required columns exist
    required_columns = ["id", "token_ids"]
    for col in required_columns:
        assert col in df.columns, f"Missing required column: {col}"

    # Verify optional columns exist (may have null values)
    optional_columns = [
        "question",
        "description",
        "market_slug",
        "condition_id",
        "outcomes",
        "active",
        "closed",
        "archived",
        "created_at",
        "end_date",
        "resolved",
        "resolution_date",
        "resolution_outcome",
        "tags",
        "category",
    ]
    for col in optional_columns:
        assert col in df.columns, f"Missing optional column: {col}"

    # Verify types
    import polars as pl

    assert df["id"].dtype == pl.String, f"Expected id to be String, got {df['id'].dtype}"

    # Get a sample market to inspect
    sample = df.row(0, named=True)
    print("\nSample market structure:")
    print(f"  id: {sample['id']}")
    print(f"  question: {sample.get('question')}")
    print(f"  closed: {sample.get('closed')}")
    print(f"  resolved: {sample.get('resolved')}")
    print(f"  token_ids: {sample['token_ids']} (count: {len(sample['token_ids']) if sample['token_ids'] else 0})")
    print(f"  outcomes: {sample.get('outcomes')}")
    print(f"  category: {sample.get('category')}")

    # Verify token_ids is a list of strings
    assert isinstance(
        sample["token_ids"], list
    ), f"Expected token_ids to be list, got {type(sample['token_ids']).__name__}"
    if sample["token_ids"]:
        assert all(isinstance(tid, str) for tid in sample["token_ids"]), "All token IDs should be strings"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_gamma_clob_token_ids_format(gamma_client: Gamma) -> None:
    """Verify the actual format of clobTokenIds from API.

    This test specifically checks whether the API returns:
    - Direct list: ["token1", "token2"]
    - Stringified JSON: '["token1", "token2"]'
    """
    # Fetch raw API response to inspect actual format

    client = await gamma_client._get_client()
    url = f"{gamma_client.base_url}/markets"
    params = {"limit": 10, "offset": 0}

    resp = await client.get(url, params=params)
    resp.raise_for_status()
    data = resp.json()

    # Verify response is array
    assert isinstance(data, list), f"Expected direct array response, got {type(data).__name__}"
    assert len(data) > 0, "Expected at least one market in response"

    # Inspect first market's clobTokenIds field
    first_market = data[0]
    print(f"\nFirst market keys: {list(first_market.keys())}")

    if "clobTokenIds" in first_market:
        clob_token_ids = first_market["clobTokenIds"]
        print(f"clobTokenIds raw value: {repr(clob_token_ids)}")
        print(f"clobTokenIds type: {type(clob_token_ids).__name__}")

        # Check if it's stringified JSON or direct list
        if isinstance(clob_token_ids, str):
            print("✓ CONFIRMED: clobTokenIds is stringified JSON")
            import json

            try:
                parsed = json.loads(clob_token_ids)
                print(f"  Parsed value: {parsed}")
                print(f"  Parsed type: {type(parsed).__name__}")
            except json.JSONDecodeError as e:
                print(f"  ERROR parsing JSON: {e}")
        elif isinstance(clob_token_ids, list):
            print("✓ CONFIRMED: clobTokenIds is direct list")
            print(f"  List contents: {clob_token_ids}")
        else:
            print(f"✗ UNEXPECTED: clobTokenIds is {type(clob_token_ids).__name__}")
    else:
        print("WARNING: clobTokenIds field not present in first market")
        print(f"Available fields: {list(first_market.keys())}")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_gamma_pagination(gamma_client: Gamma) -> None:
    """Verify Gamma API pagination behavior."""
    # Fetch with small page size to test pagination
    df = await gamma_client.fetch_markets(resolved_only=False)

    # Should have fetched up to max_pages * page_size markets
    # Configured in conftest: max_pages=1, page_size=10
    assert df.height <= 10, f"Expected at most 10 markets with page_size=10, max_pages=1, got {df.height}"

    print(f"\nFetched {df.height} markets with pagination (max_pages=1, page_size=10)")
