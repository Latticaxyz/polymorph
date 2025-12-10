from __future__ import annotations

import json

import httpx
import polars as pl

from polymorph.core.base import DataSource, PipelineContext
from polymorph.core.rate_limit import GAMMA_RATE_LIMIT, RateLimiter, RateLimitError
from polymorph.core.retry import with_retry
from polymorph.utils.logging import get_logger

logger = get_logger(__name__)

JsonValue = str | int | float | bool | None | list["JsonValue"] | list[str] | dict[str, "JsonValue"]
JsonDict = dict[str, JsonValue]
JsonList = list[JsonValue]

GAMMA_BASE = "https://gamma-api.polymarket.com"

MAX_MARKETS_PER_REQUESTS = 500


class Gamma(DataSource[pl.DataFrame]):
    def __init__(
        self,
        context: PipelineContext,
        base_url: str = GAMMA_BASE,
        page_size: int = 250,
        max_pages: int | None = None,
    ):
        super().__init__(context)
        self.base_url = base_url
        self.page_size = min(page_size, MAX_MARKETS_PER_REQUESTS)
        self.max_pages = max_pages
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
            )
        return self._client

    @with_retry(max_attempts=5, min_wait=1.0, max_wait=30.0)
    async def _get(self, url: str, params: dict[str, int | bool] | None = None) -> JsonDict | JsonList:
        rate_limiter = await self._get_rate_limiter()
        await rate_limiter.acquire()

        client = await self._get_client()
        r = await client.get(url, params=params, timeout=client.timeout)

        if r.status_code == 429:
            logger.warning("Rate limit exceeded (429), raising RateLimitError")
            raise RateLimitError("Rate limit exceeded")

        r.raise_for_status()
        result: JsonDict | JsonList = r.json()
        return result

    @staticmethod
    def _normalize_ids(v: object) -> list[str]:
        if v is None:
            return []
        if isinstance(v, list):
            return [str(x) for x in v if x is not None]
        if isinstance(v, str):
            s = v.strip()
            if s.startswith("[") and s.endswith("]"):
                try:
                    arr = json.loads(s)
                    if isinstance(arr, list):
                        return [str(x) for x in arr if x is not None]
                except Exception:
                    return [s]
            if "," in s:
                return [t.strip() for t in s.split(",") if t.strip()]
            return [s]
        return [str(v)]

    @staticmethod
    def _classify_market_type(row: JsonDict) -> str:
        question_value = row.get("question", "")
        tags_value = row.get("tags", None)

        question_lower = question_value.lower() if isinstance(question_value, str) else ""

        tag_str = ""
        if tags_value is not None:
            if isinstance(tags_value, list):
                tag_str = " ".join(str(t) for t in tags_value).lower()
            elif isinstance(tags_value, str):
                tag_str = tags_value.lower()

        combined = f"{question_lower} {tag_str}"

        if any(
            kw in combined for kw in ["election", "president", "senator", "governor", "vote", "nominee", "electoral"]
        ):
            return "election"

        if any(kw in combined for kw in ["by", "before", "after", "2024", "2025", "2026"]) and "?" in question_lower:
            return "deadline"

        if any(
            kw in combined
            for kw in ["nfl", "nba", "mlb", "nhl", "soccer", "football", "basketball", "championship", "super bowl"]
        ):
            return "sports"

        if any(kw in combined for kw in ["bitcoin", "eth", "crypto", "btc", "coin", "token", "blockchain"]):
            return "crypto"

        return "other"

    async def fetch(
        self,
        active_only: bool = True,
        max_markets: int | None = None,
        resolved_only: bool = False,
        include_resolution_data: bool = True,
    ) -> pl.DataFrame:
        logger.info(f"Fetching markets from Gamma API (active_only={active_only}, resolved_only={resolved_only})")

        url = f"{self.base_url}/markets"
        offset = 0
        markets_data: list[JsonDict] = []
        page = 0

        while True:
            if self.max_pages is not None and page >= self.max_pages:
                logger.info(f"Reached max_pages limit: {self.max_pages}")
                break

            if max_markets is not None and len(markets_data) >= max_markets:
                logger.info(f"Reached max_markets limit: {max_markets}")
                break

            batch_size = self.page_size
            if max_markets is not None:
                remaining = max_markets - len(markets_data)
                batch_size = min(batch_size, remaining)

            params: dict[str, int | bool] = {"limit": batch_size, "offset": offset}

            if active_only and not resolved_only:
                params["closed"] = False
            elif resolved_only:
                params["closed"] = True

            payload = await self._get(url, params=params)

            if isinstance(payload, list):
                items: list[JsonDict] = [obj for obj in payload if isinstance(obj, dict)]
            else:
                data_value = payload.get("data")
                markets_value = payload.get("markets")
                if isinstance(data_value, list):
                    items = [obj for obj in data_value if isinstance(obj, dict)]
                elif isinstance(markets_value, list):
                    items = [obj for obj in markets_value if isinstance(obj, dict)]
                else:
                    items = []

            if not items:
                logger.debug(f"No more items at page {page}")
                break

            if max_markets is not None and len(markets_data) + len(items) > max_markets:
                remaining = max_markets - len(markets_data)
                markets_data.extend(items[:remaining])
                logger.info(f"Reached max_markets limit: {max_markets}")
                break

            markets_data.extend(items)
            logger.debug(f"Fetched page {page + 1}: {len(items)} markets (total: {len(markets_data)})")

            if len(items) < batch_size:
                logger.debug(f"Received {len(items)} < {batch_size}, assuming end of data")
                break

            offset += batch_size
            page += 1

        logger.info(f"Fetched {len(markets_data)} total markets")

        if not markets_data:
            return pl.DataFrame({"token_ids": pl.Series([], dtype=pl.List(pl.Utf8))})

        for market in markets_data:
            if "clobTokenIds" in market:
                market["token_ids"] = self._normalize_ids(market["clobTokenIds"])
                del market["clobTokenIds"]
            else:
                market["token_ids"] = []

        df = pl.DataFrame(markets_data)

        if "question" in df.columns:
            if "tags" not in df.columns:
                df = df.with_columns(pl.lit([]).alias("tags"))

            df = df.with_columns(
                pl.struct(["question", "tags"])
                .map_elements(lambda x: self._classify_market_type(x), return_dtype=pl.Utf8)
                .alias("market_type")
            )

        if include_resolution_data and "closed" in df.columns:
            if "resolved" not in df.columns:
                df = df.with_columns(pl.col("closed").alias("resolved"))

        return df

    async def fetch_resolved_markets(self, max_markets: int | None = None) -> pl.DataFrame:
        return await self.fetch(
            active_only=False,
            resolved_only=True,
            include_resolution_data=True,
            max_markets=max_markets,
        )

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> Gamma:
        return self

    async def __aexit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: object,
    ) -> None:
        await self.close()
