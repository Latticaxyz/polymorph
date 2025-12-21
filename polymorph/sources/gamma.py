from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import cast

import httpx
import polars as pl

from polymorph.core.base import DataSource, PipelineContext
from polymorph.core.rate_limit import GAMMA_RATE_LIMIT, RateLimiter, RateLimitError
from polymorph.core.retry import with_retry
from polymorph.models.api import Market
from polymorph.utils.logging import get_logger

logger = get_logger(__name__)

ProgressCallback = Callable[[int], None]

JsonValue = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
JsonDict = dict[str, JsonValue]
JsonList = list[JsonValue]

GAMMA_BASE = "https://gamma-api.polymarket.com"


class Gamma(DataSource[pl.DataFrame]):
    def __init__(
        self,
        context: PipelineContext,
        base_url: str = GAMMA_BASE,
        max_pages: int | None = None,
        page_size: int = 100,
    ):
        super().__init__(context)
        self.base_url = base_url
        self.max_pages = max_pages if max_pages is not None else context.config.general.gamma_max_pages
        self.page_size = page_size
        self._client: httpx.AsyncClient | None = None
        self._rate_limiter: RateLimiter | None = None

    @property
    def name(self) -> str:
        return "gamma"

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
            self._client = httpx.AsyncClient(
                timeout=self.context.http_timeout,
                http2=True,
                headers={
                    "User-Agent": "polymorph/0.2.1 (httpx; +https://github.com/lattica/polymorph)",
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

        while True:
            if self.max_pages is not None and page > self.max_pages:
                break

            page_params = {
                **params,
                "limit": self.page_size,
                "offset": (page - 1) * self.page_size,
            }

            url = f"{self.base_url}/markets"
            data = await self._get(url, params=page_params)

            # Gamma API returns array directly
            if not isinstance(data, list):
                raise ValueError(f"Expected list response from Gamma API, got {type(data).__name__}")

            if not data:
                break

            page += 1

            for item in data:
                if not isinstance(item, dict):
                    raise ValueError(f"Market item must be dict, got {type(item).__name__}")

                if "clobTokenIds" in item:
                    token_ids_raw = item["clobTokenIds"]
                    if isinstance(token_ids_raw, str):
                        import json

                        try:
                            parsed = json.loads(token_ids_raw)
                            if isinstance(parsed, list):
                                item["clobTokenIds"] = [str(x) for x in parsed if x is not None]
                            else:
                                logger.warning(f"Parsed clobTokenIds is not a list: {type(parsed).__name__}")
                                item["clobTokenIds"] = []
                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to parse clobTokenIds as JSON: {token_ids_raw} - {e}")
                            item["clobTokenIds"] = []

                try:
                    market = Market.model_validate(item)
                except Exception as e:
                    logger.warning(f"Failed to parse market {item.get('id', 'unknown')}: {e}")
                    continue

                markets.append(market)

            if on_progress is not None:
                on_progress(len(markets))

        return markets

    @with_retry()
    async def fetch_markets(
        self,
        *,
        resolved_only: bool = False,
        start_ts: int | None = None,
        end_ts: int | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> pl.DataFrame:
        from datetime import datetime, timezone

        markets: list[Market] = []

        start_date = None
        end_date = None
        if start_ts is not None:
            start_date = datetime.fromtimestamp(start_ts / 1000, tz=timezone.utc).isoformat()
        if end_ts is not None:
            end_date = datetime.fromtimestamp(end_ts / 1000, tz=timezone.utc).isoformat()

        # Fetch both active markets AND markets that ended during the period
        # Ensures we get markets that were "in existence" during the time range

        # Fetch active markets (not closed yet)
        logger.info("Fetching active markets...")
        active_params: dict[str, str | int | float | bool] = {"closed": False}
        active_markets = await self._fetch_markets_with_params(active_params, on_progress=on_progress)

        # Filter active markets by creation date
        # API doesn't support created_at_max parameter
        if end_ts is not None:
            filtered_count = 0
            filtered_active = []
            for market in active_markets:
                if market.created_at:
                    try:
                        created_dt = datetime.fromisoformat(market.created_at.replace("Z", "+00:00"))
                        created_ms = int(created_dt.timestamp() * 1000)
                        if created_ms <= end_ts:
                            filtered_active.append(market)
                        else:
                            filtered_count += 1
                    except (ValueError, AttributeError):
                        # Include markets with invalid created_at to be safe
                        filtered_active.append(market)
                else:
                    filtered_active.append(market)

            active_markets = filtered_active
            if filtered_count > 0:
                logger.info(f"Filtered out {filtered_count} active markets created after time period")

        markets.extend(active_markets)
        logger.info(f"Fetched {len(active_markets)} active markets")

        # Fetch markets that ended during the time period
        if start_date is not None and end_date is not None:
            logger.info(f"Fetching markets that ended between {start_date} and {end_date}...")
            closed_params: dict[str, str | int | float | bool] = {
                "end_date_min": start_date,
                "end_date_max": end_date,
            }
            active_count = len(markets)

            def offset_progress(count: int) -> None:
                if on_progress is not None:
                    on_progress(active_count + count)

            closed_markets = await self._fetch_markets_with_params(
                closed_params, on_progress=offset_progress if on_progress else None
            )
            markets.extend(closed_markets)
            logger.info(f"Fetched {len(closed_markets)} closed markets")

        # Deduplicate markets (overlap between active and closed sets)
        # Happens when markets transition from active to closed
        seen_ids: set[str] = set()
        unique_markets: list[Market] = []
        for market in markets:
            if market.id not in seen_ids:
                seen_ids.add(market.id)
                unique_markets.append(market)

        if len(markets) != len(unique_markets):
            logger.info(f"Removed {len(markets) - len(unique_markets)} duplicate markets")

        markets = unique_markets
        logger.info(f"Total unique markets: {len(markets)}")

        if resolved_only:
            markets = [m for m in markets if m.resolved is True]

        schema = pl.Schema(
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

        if not markets:
            return pl.DataFrame(schema=schema)

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

        return pl.DataFrame(rows, schema=schema)
