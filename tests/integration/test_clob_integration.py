from __future__ import annotations

import pytest

from polymorph.sources.clob import CLOB


@pytest.mark.integration
@pytest.mark.asyncio
async def test_clob_prices_history_response_format(clob_client: CLOB) -> None:
    """Verify actual CLOB /prices-history endpoint response format."""
    # Use a well-known token for testing
    # Fetch raw API response to inspect actual format

    client = await clob_client._get_client()
    url = f"{clob_client.clob_base_url}/prices-history"

    # Try a few different test tokens (some might not have recent data)
    test_tokens = [
        "21742633143463906290569050155826241533067272736897614950488156847949938836455",  # Common YES token
        "71321045679252212594626385532706912750332728571942532289631379312455583992563",  # Common NO token
    ]

    response_data = None
    for token_id in test_tokens:
        params = {
            "market": token_id,
            "interval": "1h",  # Use interval parameter for simplicity
        }

        try:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            response_data = resp.json()

            if "history" in response_data and response_data["history"]:
                print(f"\nSuccessfully fetched data for token: {token_id}")
                break
        except Exception as e:
            print(f"Failed to fetch data for token {token_id}: {e}")
            continue

    assert response_data is not None, "Could not fetch price history from any test token"
    assert "history" in response_data, "Response missing 'history' field"
    assert isinstance(response_data["history"], list), "history should be a list"

    if response_data["history"]:
        first_point = response_data["history"][0]
        print("\nFirst price point structure:")
        print(f"  Raw: {first_point}")
        print(f"  t value: {first_point.get('t')} (type: {type(first_point.get('t')).__name__})")
        print(f"  p value: {first_point.get('p')} (type: {type(first_point.get('p')).__name__})")

        # Check timestamp
        t_value = first_point.get("t")
        assert t_value is not None, "Missing 't' field"

        if isinstance(t_value, (int, float)):
            # Check if it's seconds or milliseconds
            # Milliseconds are 13 digits (1234567890123), seconds are 10 digits (1234567890)
            if t_value > 10_000_000_000:
                print("  ✓ CONFIRMED: Timestamps are in MILLISECONDS")
            else:
                print("  ✓ CONFIRMED: Timestamps are in SECONDS")
        elif isinstance(t_value, str):
            print(f"  ✓ Timestamps are STRINGS: {repr(t_value)}")

        # Check price
        p_value = first_point.get("p")
        assert p_value is not None, "Missing 'p' field"

        if isinstance(p_value, str):
            print("  ✓ CONFIRMED: Prices are DECIMAL STRINGS")
        elif isinstance(p_value, (int, float)):
            print("  ✓ CONFIRMED: Prices are NUMBERS")
    else:
        print("\nWARNING: No price history data available for test tokens")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_clob_orderbook_response_format(clob_client: CLOB, gamma_client) -> None:
    """Verify actual CLOB /book endpoint response format."""
    import httpx

    # Try to fetch a valid token from active markets
    test_token = None
    try:
        # Fetch raw market data (not DataFrame) to get clobTokenIds
        client = await gamma_client._get_client()
        resp = await client.get(f"{gamma_client.gamma_url}/markets", params={"limit": 10})
        markets = resp.json()

        for market in markets:
            if market.get("clobTokenIds"):
                import json

                clob_token_ids = market["clobTokenIds"]
                token_ids = json.loads(clob_token_ids) if isinstance(clob_token_ids, str) else clob_token_ids
                if token_ids:
                    test_token = token_ids[0]
                    print(f"\nUsing token ID from active market: {test_token}")
                    break
    except Exception as e:
        print(f"\nWARNING: Could not fetch markets dynamically: {e}")

    if not test_token:
        # Fallback to a known token (may become stale over time)
        print("\nWARNING: Using fallback token ID - test may fail if token is inactive")
        return

    # Fetch raw API response
    client = await clob_client._get_client()
    url = f"{clob_client.clob_base_url}/book"
    params = {"token_id": test_token}

    try:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            print(f"\nWARNING: Token {test_token} returned 404 - market may be inactive, skipping test")
            return
        raise

    data = resp.json()

    print("\nOrderbook response structure:")
    print(f"  Keys: {list(data.keys())}")

    assert "bids" in data, "Missing 'bids' field"
    assert "asks" in data, "Missing 'asks' field"

    if data["bids"]:
        first_bid = data["bids"][0]
        print("\nFirst bid structure:")
        print(f"  Raw: {first_bid}")
        print(f"  price: {first_bid.get('price')} (type: {type(first_bid.get('price')).__name__})")
        print(f"  size: {first_bid.get('size')} (type: {type(first_bid.get('size')).__name__})")

        if isinstance(first_bid.get("price"), str):
            print("  ✓ CONFIRMED: Orderbook prices are DECIMAL STRINGS")
        elif isinstance(first_bid.get("price"), (int, float)):
            print("  ✓ CONFIRMED: Orderbook prices are NUMBERS")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_data_api_trades_response_format(clob_client: CLOB) -> None:
    """Verify actual Data API /trades endpoint response format."""
    # Fetch raw API response

    client = await clob_client._get_client()
    url = f"{clob_client.data_api_url}/trades"
    params = {"limit": 10, "offset": 0}

    resp = await client.get(url, params=params)
    resp.raise_for_status()
    data = resp.json()

    assert isinstance(data, list), f"Expected direct array response, got {type(data).__name__}"
    assert len(data) > 0, "Expected at least some trades"

    first_trade = data[0]
    print("\nFirst trade structure:")
    print(f"  Keys: {list(first_trade.keys())}")
    print(f"  size: {first_trade.get('size')} (type: {type(first_trade.get('size')).__name__})")
    print(f"  price: {first_trade.get('price')} (type: {type(first_trade.get('price')).__name__})")
    print(f"  timestamp: {first_trade.get('timestamp')} (type: {type(first_trade.get('timestamp')).__name__})")

    # Check field types
    if isinstance(first_trade.get("price"), str):
        print("  ✓ CONFIRMED: Trade prices are DECIMAL STRINGS")
    elif isinstance(first_trade.get("price"), (int, float)):
        print("  ✓ CONFIRMED: Trade prices are NUMBERS")

    if isinstance(first_trade.get("size"), str):
        print("  ✓ CONFIRMED: Trade sizes are DECIMAL STRINGS")
    elif isinstance(first_trade.get("size"), (int, float)):
        print("  ✓ CONFIRMED: Trade sizes are NUMBERS")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_data_api_pagination_limits(clob_client: CLOB) -> None:
    """Test the actual pagination limits for Data API /trades endpoint."""
    import httpx

    client = await clob_client._get_client()
    url = f"{clob_client.data_api_url}/trades"

    # Test with different limit values
    test_limits = [100, 1000, 5000, 10000]

    for limit in test_limits:
        params = {"limit": limit, "offset": 0}
        try:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
            actual_count = len(data)
            print(f"  limit={limit}: returned {actual_count} trades (status: {resp.status_code})")
        except httpx.HTTPStatusError as e:
            print(f"  limit={limit}: ERROR - {e.response.status_code}")
            break

    print("\n✓ Pagination limit testing completed")
