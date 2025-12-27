from __future__ import annotations

from pathlib import Path

import pytest

from polymorph.core.gamma_cache import GammaMarketCache
from polymorph.models.api import Market


@pytest.fixture
def cache(tmp_path: Path) -> GammaMarketCache:
    return GammaMarketCache(tmp_path / ".gamma_cache.db")


def test_cache_stores_and_retrieves_resolved_market(cache: GammaMarketCache) -> None:
    """Resolved markets should be retrievable from cache."""
    market = Market(id="m1", question="Test?", resolved=True, closed=True)
    cache.upsert_markets([market])

    resolved = cache.get_resolved_markets()
    assert len(resolved) == 1
    assert resolved[0].id == "m1"
    assert resolved[0].question == "Test?"
    assert resolved[0].resolved is True


def test_cache_resolved_market_immutable(cache: GammaMarketCache) -> None:
    """Resolved markets should never be overwritten."""
    market = Market(id="m1", question="Original", resolved=True, closed=True)
    cache.upsert_markets([market])

    updated = Market(id="m1", question="Changed", resolved=False, closed=True)
    cache.upsert_markets([updated])

    resolved = cache.get_resolved_markets()
    assert len(resolved) == 1
    assert resolved[0].question == "Original"
    assert resolved[0].resolved is True


def test_cache_updates_non_resolved_to_resolved(cache: GammaMarketCache) -> None:
    """Non-resolved markets should become resolved when updated."""
    market = Market(id="m1", question="Test", resolved=False, closed=True)
    cache.upsert_markets([market])

    resolved_market = Market(id="m1", question="Test", resolved=True, closed=True)
    cache.upsert_markets([resolved_market])

    resolved = cache.get_resolved_markets()
    assert len(resolved) == 1
    assert resolved[0].resolved is True


def test_get_resolved_market_ids_fast_lookup(cache: GammaMarketCache) -> None:
    """Should return set of resolved market IDs for fast membership testing."""
    markets = [
        Market(id="m1", resolved=True, closed=True),
        Market(id="m2", resolved=False, closed=True),
        Market(id="m3", resolved=True, closed=True),
    ]
    cache.upsert_markets(markets)

    ids = cache.get_resolved_market_ids()
    assert ids == {"m1", "m3"}


def test_cache_stats(cache: GammaMarketCache) -> None:
    """Should return accurate cache statistics."""
    markets = [
        Market(id="m1", resolved=True, closed=True),
        Market(id="m2", resolved=False, closed=True),
        Market(id="m3", resolved=False, closed=False),
    ]
    cache.upsert_markets(markets)

    stats = cache.get_cache_stats()
    assert stats["total"] == 3
    assert stats["resolved"] == 1
    assert stats["closed_unresolved"] == 1
    assert stats["active"] == 1


def test_upsert_returns_counts(cache: GammaMarketCache) -> None:
    """upsert_markets should return (inserted, updated) counts."""
    m1 = Market(id="m1", resolved=False, closed=False)
    inserted, updated = cache.upsert_markets([m1])
    assert inserted == 1
    assert updated == 0

    m1_updated = Market(id="m1", resolved=True, closed=True)
    m2 = Market(id="m2", resolved=False, closed=False)
    inserted, updated = cache.upsert_markets([m1_updated, m2])
    assert inserted == 1
    assert updated == 1


def test_empty_cache_returns_empty(cache: GammaMarketCache) -> None:
    """Empty cache should return empty collections."""
    assert cache.get_resolved_markets() == []
    assert cache.get_resolved_market_ids() == set()
    stats = cache.get_cache_stats()
    assert stats["total"] == 0


def test_cache_close_and_reopen(tmp_path: Path) -> None:
    """Cache should persist data after close and reopen."""
    cache_path = tmp_path / ".gamma_cache.db"

    cache1 = GammaMarketCache(cache_path)
    cache1.upsert_markets([Market(id="m1", resolved=True, closed=True)])
    cache1.close()

    cache2 = GammaMarketCache(cache_path)
    resolved = cache2.get_resolved_markets()
    assert len(resolved) == 1
    assert resolved[0].id == "m1"
    cache2.close()


def test_cache_preserves_market_fields(cache: GammaMarketCache) -> None:
    """Cache should preserve all market fields including lists."""
    market = Market(
        id="m1",
        question="Will X happen?",
        description="A test market",
        market_slug="will-x-happen",
        condition_id="cond123",
        clob_token_ids=["token1", "token2"],
        outcomes=["Yes", "No"],
        active=False,
        closed=True,
        archived=False,
        created_at="2024-01-01T00:00:00Z",
        end_date="2024-12-31T23:59:59Z",
        resolved=True,
        resolution_date="2024-06-15T12:00:00Z",
        resolution_outcome="Yes",
        tags=["politics", "election"],
        category="Politics",
    )
    cache.upsert_markets([market])

    resolved = cache.get_resolved_markets()
    assert len(resolved) == 1
    m = resolved[0]
    assert m.id == "m1"
    assert m.question == "Will X happen?"
    assert m.clob_token_ids == ["token1", "token2"]
    assert m.outcomes == ["Yes", "No"]
    assert m.resolution_date == "2024-06-15T12:00:00Z"
    assert m.tags == ["politics", "election"]
