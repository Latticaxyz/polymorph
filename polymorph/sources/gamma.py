"""Per verified API behavior:
- Markets returned as direct array (not wrapped in object)
- clobTokenIds field is stringified JSON array, not direct list
- Requires json.loads() to parse
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import httpx
import polars as pl

from polymorph.core.base import DataSource, PipelineContext
from polymorph.core.rate_limit import GAMMA_RATE_LIMIT, RateLimiter, RateLimitError
from polymorph.core.retry import with_retry
from polymorph.models.api import Market
from polymorph.utils.logging import get_logger

logger = get_logger(__name__)

JsonValue = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
JsonDict = dict[str, JsonValue]
JsonList = list[JsonValue]

GAMMA_BASE = "https://gamma-api.polymarket.com"


class Gamma(DataSource[pl.DataFrame]):
    def __init__(
        self,
        context: PipelineContext,
        base_url: str = GAMMA_BASE,
        max_pages: int = 250,
        page_size: int = 100,
    ):
        super().__init__(context)
        self.base_url = base_url
        self.max_pages = max_pages
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

    @with_retry()
    async def fetch_markets(
        self, *, resolved_only: bool = False, start_ts: int | None = None, end_ts: int | None = None
    ) -> pl.DataFrame:
        markets: list[Market] = []

        for page in range(1, self.max_pages + 1):
            params: dict[str, str | int | float | bool] = {
                "limit": self.page_size,
                "offset": (page - 1) * self.page_size,
            }

            url = f"{self.base_url}/markets"
            data = await self._get(url, params=params)

            # Gamma API returns array directly (NOT wrapped in {"markets": [...]})
            if not isinstance(data, list):
                raise ValueError(f"Expected list response from Gamma API, got {type(data).__name__}")

            if not data:
                break

            for item in data:
                if not isinstance(item, dict):
                    raise ValueError(f"Market item must be dict, got {type(item).__name__}")

                # Handle stringified JSON clobTokenIds before passing to Market model
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

                # Filter by resolved_only if requested
                if resolved_only and market.resolved is not True:
                    continue

                # Filter by time range if requested
                if start_ts is not None or end_ts is not None:
                    if market.created_at is None:
                        continue

                    from datetime import datetime

                    try:
                        created_dt = datetime.fromisoformat(market.created_at.replace("Z", "+00:00"))
                        created_ms = int(created_dt.timestamp() * 1000)

                        if start_ts is not None and created_ms < start_ts:
                            continue
                        if end_ts is not None and created_ms > end_ts:
                            continue
                    except (ValueError, AttributeError) as e:
                        logger.warning(f"Skipping market {market.id}: invalid created_at '{market.created_at}' - {e}")
                        continue

                markets.append(market)

        if not markets:
            # Return empty DataFrame with full schema
            return pl.DataFrame(
                schema={
                    "id": pl.Utf8,
                    "question": pl.Utf8,
                    "description": pl.Utf8,
                    "market_slug": pl.Utf8,
                    "condition_id": pl.Utf8,
                    "token_ids": pl.List(pl.Utf8),
                    "outcomes": pl.List(pl.Utf8),
                    "active": pl.Boolean,
                    "closed": pl.Boolean,
                    "archived": pl.Boolean,
                    "created_at": pl.Utf8,
                    "end_date": pl.Utf8,
                    "resolved": pl.Boolean,
                    "resolution_date": pl.Utf8,
                    "resolution_outcome": pl.Utf8,
                    "tags": pl.List(pl.Utf8),
                    "category": pl.Utf8,
                }
            )

        # Convert Market objects to dictionaries for DataFrame
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

        return pl.DataFrame(rows)
