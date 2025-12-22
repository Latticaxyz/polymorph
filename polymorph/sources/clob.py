from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import cast

import httpx
import polars as pl

from polymorph import __version__
from polymorph.core.base import DataSource, PipelineContext
from polymorph.core.fetch_cache import CacheKey, FetchCache
from polymorph.core.rate_limit import CLOB_RATE_LIMIT, DATA_API_RATE_LIMIT, RateLimiter, RateLimitError
from polymorph.core.retry import with_retry
from polymorph.models.api import OrderBook, OrderBookLevel
from polymorph.utils.constants import CLOB_MAX_PRICE_HISTORY_MS
from polymorph.utils.logging import get_logger
from polymorph.utils.parse import parse_timestamp_ms

logger = get_logger(__name__)

ProgressCallback = Callable[[int], None]

JsonValue = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
JsonDict = dict[str, JsonValue]
JsonList = list[JsonValue]

CLOB_BASE = "https://clob.polymarket.com"
DATA_API = "https://data-api.polymarket.com"


class CLOB(DataSource[pl.DataFrame]):
    def __init__(
        self,
        context: PipelineContext,
        clob_base_url: str = CLOB_BASE,
        data_api_url: str = DATA_API,
        default_fidelity: int = 60,
        max_trades: int = 200_000,
    ):
        super().__init__(context)
        self.clob_base_url = clob_base_url
        self.data_api_url = data_api_url
        self.default_fidelity = default_fidelity
        self.max_trades = max_trades
        self._client: httpx.AsyncClient | None = None
        self._clob_rate_limiter: RateLimiter | None = None
        self._data_rate_limiter: RateLimiter | None = None

    @property
    def name(self) -> str:
        return "clob"

    async def _get_clob_rate_limiter(self) -> RateLimiter:
        if self._clob_rate_limiter is None:
            self._clob_rate_limiter = await RateLimiter.get_instance(
                name="clob",
                max_requests=CLOB_RATE_LIMIT["max_requests"],
                time_window_seconds=CLOB_RATE_LIMIT["time_window_seconds"],
            )
        return self._clob_rate_limiter

    async def _get_data_rate_limiter(self) -> RateLimiter:
        if self._data_rate_limiter is None:
            self._data_rate_limiter = await RateLimiter.get_instance(
                name="data_api",
                max_requests=DATA_API_RATE_LIMIT["max_requests"],
                time_window_seconds=DATA_API_RATE_LIMIT["time_window_seconds"],
            )
        return self._data_rate_limiter

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            limits = httpx.Limits(
                max_connections=100,
                max_keepalive_connections=50,
                keepalive_expiry=30.0,
            )
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.context.http_timeout, connect=10.0),
                http2=False,
                limits=limits,
                headers={
                    "User-Agent": f"polymorph/{__version__} (httpx; +https://github.com/lattica/polymorph)",
                },
            )
        return self._client

    async def _get(
        self,
        url: str,
        params: Mapping[str, str | int | float | bool] | None = None,
        *,
        use_data_api: bool = True,
    ) -> JsonValue:
        limiter = await (self._get_data_rate_limiter() if use_data_api else self._get_clob_rate_limiter())

        try:
            await limiter.acquire()
        except RateLimitError:
            raise

        client = await self._get_client()
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        return cast(JsonValue, resp.json())

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "CLOB":
        _ = await self._get_client()
        return self

    async def __aexit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: object,
    ) -> None:
        await self.close()

    @with_retry()
    async def _fetch_price_history_chunk(
        self,
        token_id: str,
        start_ts: int,  # Unix milliseconds
        end_ts: int,  # Unix milliseconds
        fidelity: int,  # Seconds
    ) -> pl.DataFrame:
        url = f"{self.clob_base_url}/prices-history"
        params: dict[str, str | int | float | bool] = {
            "market": token_id,
            "startTs": start_ts // 1000,  # API expects seconds
            "endTs": end_ts // 1000,
            "fidelity": fidelity,
        }

        data = await self._get(url, params=params, use_data_api=False)

        if not isinstance(data, dict):
            raise ValueError(f"Expected dict response, got {type(data).__name__}")

        hist = data.get("history")
        if hist is None:
            raise ValueError("Response missing 'history' field")
        if not isinstance(hist, list):
            raise ValueError(f"'history' must be list, got {type(hist).__name__}")

        if not hist:
            return pl.DataFrame(schema={"token_id": pl.Utf8, "t": pl.Int64, "p": pl.Float64})

        rows: list[dict[str, object]] = []
        for item in hist:
            if not isinstance(item, dict):
                raise ValueError(f"History item must be dict, got {type(item).__name__}")

            if "t" not in item or "p" not in item:
                raise ValueError(f"History item missing required fields: {item}")

            t_seconds = item["t"]
            if isinstance(t_seconds, (int, float)):
                t = int(t_seconds * 1000) if t_seconds < 10000000000 else int(t_seconds)
            else:
                t = parse_timestamp_ms(t_seconds)

            p_val = item["p"]
            if not isinstance(p_val, (int, float)):
                raise ValueError(f"Price must be numeric, got {type(p_val).__name__}")
            p = float(p_val)

            rows.append({"token_id": token_id, "t": t, "p": p})

        return pl.DataFrame(rows)

    @with_retry()
    async def _fetch_price_history_interval(
        self,
        token_id: str,
        interval: str,  # 'all', 'max', '1d', '1w', etc.
        fidelity: int,
    ) -> pl.DataFrame:
        url = f"{self.clob_base_url}/prices-history"
        params: dict[str, str | int | float | bool] = {
            "market": token_id,
            "interval": interval,
            "fidelity": fidelity,
        }

        data = await self._get(url, params=params, use_data_api=False)

        if not isinstance(data, dict):
            raise ValueError(f"Expected dict response, got {type(data).__name__}")

        hist = data.get("history")
        if hist is None:
            raise ValueError("Response missing 'history' field")
        if not isinstance(hist, list):
            raise ValueError(f"'history' must be list, got {type(hist).__name__}")

        if not hist:
            return pl.DataFrame(schema={"token_id": pl.Utf8, "t": pl.Int64, "p": pl.Float64})

        rows: list[dict[str, object]] = []
        for item in hist:
            if not isinstance(item, dict):
                raise ValueError(f"History item must be dict, got {type(item).__name__}")

            if "t" not in item or "p" not in item:
                raise ValueError(f"History item missing required fields: {item}")

            t_seconds = item["t"]
            if isinstance(t_seconds, (int, float)):
                t = int(t_seconds * 1000) if t_seconds < 10000000000 else int(t_seconds)
            else:
                t = parse_timestamp_ms(t_seconds)

            p_val = item["p"]
            if not isinstance(p_val, (int, float)):
                raise ValueError(f"Price must be numeric, got {type(p_val).__name__}")
            p = float(p_val)

            rows.append({"token_id": token_id, "t": t, "p": p})

        return pl.DataFrame(rows)

    async def fetch_prices_history(
        self,
        token_id: str,
        start_ts: int | None = None,
        end_ts: int | None = None,
        fidelity: int | None = None,
        interval: str | None = None,
        empty_chunk_limit: int = 5,
        cache: FetchCache | None = None,
        created_at_ts: int | None = None,
    ) -> pl.DataFrame:
        fidelity = fidelity if fidelity is not None else self.default_fidelity

        if interval is not None:
            return await self._fetch_price_history_interval(token_id, interval, fidelity)

        if start_ts is None or end_ts is None:
            raise ValueError("Either 'interval' or both 'start_ts' and 'end_ts' must be provided")

        chunks_to_fetch: list[tuple[int, int]]
        if cache:
            chunks_to_fetch = list(
                cache.get_pending_chunks(token_id, start_ts, end_ts, fidelity, CLOB_MAX_PRICE_HISTORY_MS)
            )
        else:
            chunks_to_fetch = []
            current = start_ts
            while current < end_ts:
                chunk_end = min(current + CLOB_MAX_PRICE_HISTORY_MS, end_ts)
                chunks_to_fetch.append((current, chunk_end))
                current = chunk_end + 1

        results: list[pl.DataFrame] = []
        consecutive_empty = 0

        for chunk_start, chunk_end in chunks_to_fetch:
            df = await self._fetch_price_history_chunk(token_id, chunk_start, chunk_end, fidelity)

            if cache:
                cache.mark_completed(CacheKey(token_id, chunk_start, chunk_end, fidelity), df.height)

            if df.height > 0:
                results.append(df)
                consecutive_empty = 0
            else:
                consecutive_empty += 1
                can_early_stop = created_at_ts is None or chunk_start >= created_at_ts
                if can_early_stop and consecutive_empty >= empty_chunk_limit:
                    break

        if not results:
            return pl.DataFrame(schema={"token_id": pl.Utf8, "t": pl.Int64, "p": pl.Float64})

        combined = pl.concat(results, how="vertical")

        if "t" in combined.columns:
            combined = combined.unique(subset=["t"], maintain_order=True)

        return combined

    @with_retry()
    async def fetch_orderbook(self, token_id: str) -> OrderBook:
        url = f"{self.clob_base_url}/book"
        params: dict[str, str | int | float | bool] = {"token_id": token_id}

        data = await self._get(url, params=params, use_data_api=False)

        if not isinstance(data, dict):
            raise ValueError(f"Expected dict response, got {type(data).__name__}")

        if "bids" not in data or "asks" not in data:
            raise ValueError(f"Response missing bids/asks: {list(data.keys())}")

        bids_data = data["bids"]
        asks_data = data["asks"]

        if not isinstance(bids_data, list) or not isinstance(asks_data, list):
            raise ValueError("bids and asks must be lists")

        bids: list[OrderBookLevel] = []
        for level in bids_data:
            if not isinstance(level, dict):
                continue
            if "price" not in level or "size" not in level:
                continue

            try:
                price_val = level["price"]
                size_val = level["size"]
                if isinstance(price_val, str):
                    price_val = float(price_val)
                if isinstance(size_val, str):
                    size_val = float(size_val)
                if not isinstance(price_val, (int, float)) or not isinstance(size_val, (int, float)):
                    continue
                bids.append(OrderBookLevel(price=float(price_val), size=float(size_val)))
            except (ValueError, TypeError):
                continue

        asks: list[OrderBookLevel] = []
        for level in asks_data:
            if not isinstance(level, dict):
                continue
            if "price" not in level or "size" not in level:
                continue

            try:
                price_val = level["price"]
                size_val = level["size"]
                if isinstance(price_val, str):
                    price_val = float(price_val)
                if isinstance(size_val, str):
                    size_val = float(size_val)
                if not isinstance(price_val, (int, float)) or not isinstance(size_val, (int, float)):
                    continue
                asks.append(OrderBookLevel(price=float(price_val), size=float(size_val)))
            except (ValueError, TypeError):
                continue

        bids = sorted(bids, key=lambda x: x.price, reverse=True)
        asks = sorted(asks, key=lambda x: x.price)

        best_bid = bids[0].price if bids else None
        best_ask = asks[0].price if asks else None

        try:
            timestamp = parse_timestamp_ms(data.get("timestamp", 0))
        except (ValueError, TypeError):
            timestamp = 0

        ob = OrderBook(
            token_id=token_id,
            timestamp=timestamp,
            bids=bids,
            asks=asks,
            best_bid=best_bid,
            best_ask=best_ask,
        )
        ob.mid_price = ob.calculate_mid_price()
        ob.spread = ob.calculate_spread()
        return ob

    async def fetch_orderbooks(self, token_ids: list[str]) -> pl.DataFrame:
        rows: list[dict[str, object]] = []

        for token_id in token_ids:
            try:
                ob = await self.fetch_orderbook(token_id)
            except Exception as e:
                logger.error(f"Failed to fetch order book for {token_id}: {e}")
                continue

            for level in ob.bids:
                rows.append(
                    {
                        "token_id": ob.token_id,
                        "timestamp": ob.timestamp,
                        "side": "bid",
                        "price": level.price,
                        "size": level.size,
                    }
                )

            for level in ob.asks:
                rows.append(
                    {
                        "token_id": ob.token_id,
                        "timestamp": ob.timestamp,
                        "side": "ask",
                        "price": level.price,
                        "size": level.size,
                    }
                )

        return pl.DataFrame(rows) if rows else pl.DataFrame()

    async def fetch_spread(self, token_id: str) -> dict[str, str | float | int | None]:
        ob = await self.fetch_orderbook(token_id)
        return {
            "token_id": token_id,
            "bid": ob.best_bid,
            "ask": ob.best_ask,
            "mid": ob.mid_price,
            "spread": ob.spread,
            "timestamp": ob.timestamp,
        }

    @with_retry()
    async def fetch_trades_paged(
        self,
        limit: int,
        offset: int,
        market_ids: list[str] | None = None,
    ) -> list[dict[str, str | int | float]]:
        params: dict[str, str | int | float | bool] = {"limit": limit, "offset": offset}
        if market_ids:
            params["market"] = ",".join(market_ids)

        url = f"{self.data_api_url}/trades"
        data = await self._get(url, params=params, use_data_api=True)

        # API returns array directly (NOT wrapped)
        if isinstance(data, list):
            return cast(list[dict[str, str | int | float]], [x for x in data if isinstance(x, dict)])

        # Fallback: handle wrapped format
        if isinstance(data, dict):
            data_list = data.get("data")
            if isinstance(data_list, list):
                out: list[dict[str, str | int | float]] = []
                for item in data_list:
                    if isinstance(item, dict):
                        out.append(item)  # type: ignore[arg-type]
                return out

        return []

    async def _fetch_trades_for_market_batch(
        self,
        market_ids: list[str] | None,
        max_rows: int,
        on_progress: ProgressCallback | None = None,
        progress_offset: int = 0,
    ) -> list[dict[str, str | int | float]]:
        rows: list[dict[str, str | int | float]] = []
        offset = 0
        limit = 1000

        while True:
            batch = await self.fetch_trades_paged(limit=limit, offset=offset, market_ids=market_ids)
            if not batch:
                break

            if len(rows) + len(batch) > max_rows:
                remaining = max_rows - len(rows)
                rows.extend(batch[:remaining])
                if on_progress is not None:
                    on_progress(progress_offset + len(rows))
                break

            rows.extend(batch)
            offset += limit

            if on_progress is not None:
                on_progress(progress_offset + len(rows))

            if len(batch) < limit:
                break

        return rows

    async def fetch_trades(
        self,
        market_ids: list[str] | None = None,
        since_ts: int | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> pl.DataFrame:
        rows: list[dict[str, str | int | float]] = []
        market_batch_size = 50

        if market_ids and len(market_ids) > market_batch_size:
            for i in range(0, len(market_ids), market_batch_size):
                if len(rows) >= self.max_trades:
                    break
                chunk = market_ids[i : i + market_batch_size]
                remaining_budget = self.max_trades - len(rows)
                batch_rows = await self._fetch_trades_for_market_batch(
                    market_ids=chunk,
                    max_rows=remaining_budget,
                    on_progress=on_progress,
                    progress_offset=len(rows),
                )
                rows.extend(batch_rows)
        else:
            rows = await self._fetch_trades_for_market_batch(
                market_ids=market_ids,
                max_rows=self.max_trades,
                on_progress=on_progress,
            )

        if not rows:
            schema = {
                "id": pl.Utf8,
                "market": pl.Utf8,
                "asset_id": pl.Utf8,
                "side": pl.Utf8,
                "size": pl.Float64,
                "price": pl.Float64,
                "timestamp": pl.Int64,
            }
            return pl.DataFrame(schema=schema)

        df = pl.DataFrame(rows)

        if "timestamp" in df.columns:
            df = df.with_columns(pl.col("timestamp").cast(pl.Int64))
        else:
            raise ValueError("Trades data missing required 'timestamp' field")

        if since_ts is not None:
            df = df.filter(pl.col("timestamp") >= since_ts)

        logger.info(f"Fetched {len(df)} total trades")
        return df
