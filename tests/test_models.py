from __future__ import annotations

import pytest

from polymorph.models.api import Market, OrderBook, OrderBookLevel


def test_market_normalizes_clob_token_ids_from_list() -> None:
    """Test Market normalizes token IDs from a list."""
    m = Market(
        id="test-market-1",
        question="Will this test pass?",
        clobTokenIds=["1", "2"],
    )
    assert m.clob_token_ids == ["1", "2"]
    assert m.id == "test-market-1"
    assert m.question == "Will this test pass?"


def test_market_normalizes_clob_token_ids_from_json_string() -> None:
    """Test Market normalizes token IDs from JSON string."""
    m = Market(
        id="test-market-2",
        question="Can we parse JSON?",
        clobTokenIds=["a", "b"],
    )
    assert m.clob_token_ids == ["a", "b"]


def test_market_normalizes_clob_token_ids_from_csv_string() -> None:
    """Test Market normalizes token IDs from CSV string."""
    m = Market(
        id="test-market-3",
        question="Can we parse CSV?",
        clobTokenIds=["x", "y", "z"],
    )
    assert m.clob_token_ids == ["x", "y", "z"]


def test_market_with_complete_data() -> None:
    """Test Market with comprehensive data to verify all fields work."""
    m = Market(
        id="comprehensive-market",
        question="Who will win the 2024 election?",
        description="Presidential election prediction market",
        marketSlug="2024-election",
        conditionId="condition-123",
        clobTokenIds=["token-yes", "token-no"],
        outcomes=["YES", "NO"],
        active=True,
        closed=False,
        archived=False,
        createdAt="2024-01-01T00:00:00Z",
        endDate="2024-11-05T23:59:59Z",
        resolved=False,
        tags=["politics", "election"],
        category="politics",
    )

    assert m.id == "comprehensive-market"
    assert m.question == "Who will win the 2024 election?"
    assert m.market_slug == "2024-election"
    assert m.condition_id == "condition-123"
    assert m.clob_token_ids == ["token-yes", "token-no"]
    assert m.outcomes == ["YES", "NO"]
    assert m.active is True
    assert m.closed is False
    assert m.resolved is False
    assert len(m.tags) == 2


def test_orderbook_mid_spread_and_depth() -> None:
    """Test OrderBook calculations for mid price, spread, and depth."""
    bids = [OrderBookLevel(price=0.4, size=10.0), OrderBookLevel(price=0.3, size=5.0)]
    asks = [OrderBookLevel(price=0.6, size=4.0), OrderBookLevel(price=0.7, size=6.0)]

    ob = OrderBook(
        token_id="YES",
        timestamp=1577836800000,
        bids=bids,
        asks=asks,
        best_bid=0.4,
        best_ask=0.6,
    )

    ob.mid_price = ob.calculate_mid_price()
    assert ob.mid_price == pytest.approx(0.5)

    ob.spread = ob.calculate_spread()
    assert ob.spread == pytest.approx(0.2)

    depth_bid = ob.get_depth_at_distance(distance=0.1, side="bid")
    depth_ask = ob.get_depth_at_distance(distance=0.1, side="ask")
    depth_both = ob.get_depth_at_distance(distance=0.1, side="both")

    assert depth_bid > 0.0
    assert depth_ask > 0.0
    assert depth_both == pytest.approx(depth_bid + depth_ask)


def test_orderbook_empty_book() -> None:
    """Test OrderBook with no orders."""
    ob = OrderBook(
        token_id="TEST",
        timestamp=1577836800000,
    )

    assert ob.calculate_mid_price() is None
    assert ob.calculate_spread() is None
    assert ob.get_depth_at_distance(0.1, "both") == 0.0
