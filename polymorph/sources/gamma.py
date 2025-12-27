from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from typing import cast

import httpx
import polars as pl

from polymorph import __version__
from polymorph.core.base import PipelineContext
from polymorph.core.gamma_cache import GammaMarketCache
from polymorph.core.rate_limit import GAMMA_RATE_LIMIT, RateLimiter, RateLimitError
from polymorph.core.retry import with_retry
from polymorph.models.api import Market
from polymorph.utils.constants import (
    CONNECT_TIMEOUT_SECONDS,
    GAMMA_PAGE_SIZE,
    KEEPALIVE_EXPIRY_SECONDS,
    MS_PER_SECOND,
)
from polymorph.utils.logging import get_logger
from polymorph.utils.time import parse_iso_to_ms_or_none
from polymorph.utils.types import JsonValue

logger = get_logger(__name__)

ProgressCallback = Callable[[int], None]

GAMMA_BASE = "https://gamma-api.polymarket.com"

MARKETS_SCHEMA = pl.Schema(
    [
        ("id", pl.Utf8),
        ("question", pl.Utf8),
        ("description", pl.Utf8),
        ("market_slug", pl.Utf8),
        ("condition_id", pl.Utf8),
        ("token_ids", pl.List(pl.Utf8)),
        ("outcomes", pl.List(pl.Utf8)),
        ("active", pl.Boolean),
        ("closed", pl.Boolean),
        ("archived", pl.Boolean),
        ("created_at", pl.Utf8),
        ("end_date", pl.Utf8),
        ("resolved", pl.Boolean),
        ("resolution_date", pl.Utf8),
        ("resolution_outcome", pl.Utf8),
        ("tags", pl.List(pl.Utf8)),
        ("category", pl.Utf8),
    ]
)


def _ts_to_iso(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / MS_PER_SECOND, tz=timezone.utc).isoformat()


def _filter_markets_by_creation_date(markets: list[Market], end_ts: int) -> list[Market]:
    filtered: list[Market] = []
    filtered_count = 0
    for market in markets:
        created_ms = parse_iso_to_ms_or_none(market.created_at)
        if created_ms is None or created_ms <= end_ts:
            filtered.append(market)
        else:
            filtered_count += 1
    if filtered_count > 0:
        logger.info(f"Filtered out {filtered_count} active markets created after time period")
    return filtered


def _dedupe_markets(markets: list[Market]) -> list[Market]:
    seen_ids: set[str] = set()
    unique: list[Market] = []
    for market in markets:
        if market.id not in seen_ids:
            seen_ids.add(market.id)
            unique.append(market)
    if len(markets) != len(unique):
        logger.info(f"Removed {len(markets) - len(unique)} duplicate markets")
    return unique


def _parse_clob_token_ids(token_ids_raw: JsonValue) -> list[str]:
    import json

    if not isinstance(token_ids_raw, str):
        return (
            [] if token_ids_raw is None else [str(x) for x in token_ids_raw] if isinstance(token_ids_raw, list) else []
        )

    try:
        parsed = json.loads(token_ids_raw)
        if isinstance(parsed, list):
            return [str(x) for x in parsed if x is not None]
        logger.warning(f"Parsed clobTokenIds is not a list: {type(parsed).__name__}")
        return []
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse clobTokenIds as JSON: {token_ids_raw} - {e}")
        return []


def _markets_to_dataframe(markets: list[Market]) -> pl.DataFrame:
    if not markets:
        return pl.DataFrame(schema=MARKETS_SCHEMA)
    rows = [
        {
            "id": m.id,
            "question": m.question,
            "description": m.description,
            "market_slug": m.market_slug,
            "condition_id": m.condition_id,
            "token_ids": m.clob_token_ids,
            "outcomes": m.outcomes,
            "active": m.active,
            "closed": m.closed,
            "archived": m.archived,
            "created_at": m.created_at,
            "end_date": m.end_date,
            "resolved": m.resolved,
            "resolution_date": m.resolution_date,
            "resolution_outcome": m.resolution_outcome,
            "tags": m.tags,
            "category": m.category,
        }
        for m in markets
    ]
    return pl.DataFrame(rows, schema=MARKETS_SCHEMA)


class Gamma:
    def __init__(
        self,
        context: PipelineContext,
        base_url: str = GAMMA_BASE,
        max_pages: int | None = None,
        page_size: int = GAMMA_PAGE_SIZE,
        use_cache: bool = True,
    ):
        self.context = context
        self.config = context.config
        self.base_url = base_url
        self.max_pages = max_pages if max_pages is not None else context.config.general.gamma_max_pages
        self.page_size = page_size
        self._client: httpx.AsyncClient | None = None
        self._rate_limiter: RateLimiter | None = None
        self._use_cache = use_cache
        self._cache: GammaMarketCache | None = None

    def _get_cache(self) -> GammaMarketCache | None:
        """Lazy-initialize cache if enabled."""
        if not self._use_cache:
            return None
        if self._cache is None:
            cache_path = self.context.data_dir / ".gamma_cache.db"
            self._cache = GammaMarketCache(cache_path)
        return self._cache

    async def _get_rate_limiter(self) -> RateLimiter:
        if self._rate_limiter is None:
            self._rate_limiter = await RateLimiter.get_instance(
                name="gamma",
                max_requests=GAMMA_RATE_LIMIT["max_requests"],
                time_window_seconds=GAMMA_RATE_LIMIT["time_window_seconds"],
            )
        return self._rate_limiter

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            max_conn = self.context.config.general.gamma_max_conn
            ka_conn = self.context.config.general.gamma_ka_conn
            limits = httpx.Limits(
                max_connections=max_conn,
                max_keepalive_connections=ka_conn,
                keepalive_expiry=KEEPALIVE_EXPIRY_SECONDS,
            )
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.context.http_timeout, connect=CONNECT_TIMEOUT_SECONDS),
                http2=True,
                limits=limits,
                headers={
                    "User-Agent": f"polymorph/{__version__} (httpx; +https://github.com/lattica/polymorph)",
                },
            )
        return self._client

    async def __aenter__(self) -> "Gamma":
        _ = await self._get_client()
        return self

    async def __aexit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: object,
    ) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        if self._cache is not None:
            self._cache.close()
            self._cache = None

    async def _get(
        self,
        url: str,
        params: Mapping[str, str | int | float | bool] | None = None,
    ) -> JsonValue:
        limiter = await self._get_rate_limiter()
        try:
            await limiter.acquire()
        except RateLimitError:
            raise

        client = await self._get_client()
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        return cast(JsonValue, resp.json())

    async def _fetch_markets_with_params(
        self,
        params: dict[str, str | int | float | bool],
        on_progress: ProgressCallback | None = None,
    ) -> list[Market]:
        markets: list[Market] = []
        page = 1

        while self.max_pages is None or page <= self.max_pages:
            page_params = {**params, "limit": self.page_size, "offset": (page - 1) * self.page_size}
            data = await self._get(f"{self.base_url}/markets", params=page_params)

            if not isinstance(data, list) or not data:
                break

            page += 1

            for item in data:
                if not isinstance(item, dict):
                    continue

                if "clobTokenIds" in item:
                    item["clobTokenIds"] = cast(JsonValue, _parse_clob_token_ids(item["clobTokenIds"]))

                try:
                    markets.append(Market.model_validate(item))
                except Exception as e:
                    logger.warning(f"Failed to parse market {item.get('id', 'unknown')}: {e}")

            if on_progress is not None:
                on_progress(len(markets))

        return markets

    async def _fetch_closed_markets(
        self,
        start_date: str,
        end_date: str,
        active_count: int,
        on_progress: ProgressCallback | None,
    ) -> list[Market]:
        logger.info(f"Fetching markets that ended between {start_date} and {end_date}...")
        closed_params: dict[str, str | int | float | bool] = {
            "end_date_min": start_date,
            "end_date_max": end_date,
        }

        def offset_progress(count: int) -> None:
            if on_progress is not None:
                on_progress(active_count + count)

        closed_markets = await self._fetch_markets_with_params(
            closed_params, on_progress=offset_progress if on_progress else None
        )
        logger.info(f"Fetched {len(closed_markets)} closed markets")
        return closed_markets

    @with_retry()
    async def fetch_markets(
        self,
        *,
        resolved_only: bool = False,
        start_ts: int | None = None,
        end_ts: int | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> pl.DataFrame:
        cache = self._get_cache()
        cached_resolved: list[Market] = []
        cached_resolved_ids: set[str] = set()

        if cache is not None:
            cached_resolved = cache.get_resolved_markets()
            cached_resolved_ids = {m.id for m in cached_resolved}
            if cached_resolved:
                stats = cache.get_cache_stats()
                logger.info(
                    f"Loaded {len(cached_resolved)} resolved markets from cache "
                    f"(total cached: {stats['total']}, active: {stats['active']})"
                )

        start_date = _ts_to_iso(start_ts) if start_ts is not None else None
        end_date = _ts_to_iso(end_ts) if end_ts is not None else None

        logger.info("Fetching active markets from API...")
        active_params: dict[str, str | int | float | bool] = {"closed": False}
        active_markets = await self._fetch_markets_with_params(active_params, on_progress=on_progress)

        if end_ts is not None:
            active_markets = _filter_markets_by_creation_date(active_markets, end_ts)

        logger.info(f"Fetched {len(active_markets)} active markets from API")

        closed_markets: list[Market] = []
        if start_date is not None and end_date is not None:
            closed_markets = await self._fetch_closed_markets(start_date, end_date, len(active_markets), on_progress)
            closed_before_filter = len(closed_markets)
            closed_markets = [m for m in closed_markets if m.id not in cached_resolved_ids]
            if closed_before_filter > len(closed_markets):
                logger.info(
                    f"Skipped {closed_before_filter - len(closed_markets)} " "closed markets already cached as resolved"
                )

        all_api_markets = active_markets + closed_markets
        if cache is not None and all_api_markets:
            inserted, updated = cache.upsert_markets(all_api_markets)
            newly_resolved = sum(1 for m in all_api_markets if m.resolved and m.id not in cached_resolved_ids)
            if newly_resolved > 0:
                logger.info(f"Cached {newly_resolved} newly resolved markets permanently")
            if inserted > 0 or updated > 0:
                logger.debug(f"Cache update: {inserted} inserted, {updated} updated")

        markets: list[Market] = list(active_markets)

        if start_ts is not None and end_ts is not None:
            for m in cached_resolved:
                resolution_ms = parse_iso_to_ms_or_none(m.resolution_date)
                end_ms = parse_iso_to_ms_or_none(m.end_date)
                relevant_ts = resolution_ms or end_ms
                if relevant_ts and start_ts <= relevant_ts <= end_ts:
                    markets.append(m)

        markets.extend(closed_markets)

        markets = _dedupe_markets(markets)
        logger.info(f"Total unique markets: {len(markets)}")

        if resolved_only:
            markets = [m for m in markets if m.resolved is True]

        return _markets_to_dataframe(markets)
