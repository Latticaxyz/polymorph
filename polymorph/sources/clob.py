"""CLOB (Central Limit Order Book) data source for Polymarket.

Per verified API behavior:
- CLOB /prices-history: Returns timestamps/prices as numbers, converted to strings
- CLOB /book: Returns price/size as strings (decimal format)
- Data API /trades: Returns price/size/timestamp as numbers, converted to strings

All timestamps stored as Int64 milliseconds.
All prices/sizes stored as Utf8 decimal strings for precision.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import httpx
import polars as pl

from polymorph.core.base import DataSource, PipelineContext
from polymorph.core.rate_limit import CLOB_RATE_LIMIT, DATA_API_RATE_LIMIT, RateLimiter, RateLimitError
from polymorph.core.retry import with_retry
from polymorph.models.api import OrderBook, OrderBookLevel
from polymorph.utils.constants import CLOB_MAX_PRICE_HISTORY_MS
from polymorph.utils.logging import get_logger
from polymorph.utils.parse import parse_decimal_string, parse_timestamp_ms

logger = get_logger(__name__)

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
            self._client = httpx.AsyncClient(
                timeout=self.context.http_timeout,
                http2=True,
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
            "token_id": token_id,
            "startTs": start_ts,  # Send milliseconds directly
            "endTs": end_ts,  # Send milliseconds directly
            "fidelity": fidelity,  # Fidelity is in seconds
        }

        data = await self._get(url, params=params, use_data_api=False)

        # Strict response validation
        if not isinstance(data, dict):
            raise ValueError(f"Expected dict response, got {type(data).__name__}")

        hist = data.get("history")
        if hist is None:
            raise ValueError("Response missing 'history' field")
        if not isinstance(hist, list):
            raise ValueError(f"'history' must be list, got {type(hist).__name__}")

        if not hist:
            return pl.DataFrame(schema={"token_id": pl.Utf8, "t": pl.Int64, "p": pl.Utf8})

        rows: list[dict[str, object]] = []
        for item in hist:
            if not isinstance(item, dict):
                raise ValueError(f"History item must be dict, got {type(item).__name__}")

            # Strict parsing - must have both 't' and 'p'
            if "t" not in item or "p" not in item:
                raise ValueError(f"History item missing required fields: {item}")

            # Parse strictly using our parsers
            t = parse_timestamp_ms(item["t"])
            p = parse_decimal_string(item["p"])

            rows.append({"token_id": token_id, "t": t, "p": p})

        return pl.DataFrame(rows)

    async def fetch_prices_history(
        self,
        token_id: str,
        start_ts: int,  # Unix milliseconds
        end_ts: int,  # Unix milliseconds
        fidelity: int | None = None,
    ) -> pl.DataFrame:
        fidelity = fidelity if fidelity is not None else self.default_fidelity

        results: list[pl.DataFrame] = []
        current_start = start_ts

        while current_start < end_ts:
            # Chunk in milliseconds
            current_end = min(current_start + CLOB_MAX_PRICE_HISTORY_MS, end_ts)
            df = await self._fetch_price_history_chunk(token_id, current_start, current_end, fidelity)
            if df.height > 0:
                results.append(df)
            # Move to next chunk (add 1 ms to avoid overlap)
            current_start = current_end + 1

        if not results:
            return pl.DataFrame(schema={"token_id": pl.Utf8, "t": pl.Int64, "p": pl.Utf8})

        combined = pl.concat(results, how="vertical")

        # Deduplicate by timestamp to avoid overlaps
        if "t" in combined.columns:
            combined = combined.unique(subset=["t"], maintain_order=True)

        return combined

    async def fetch_orderbook(self, token_id: str) -> OrderBook:
        url = f"{self.clob_base_url}/book"
        params: dict[str, str | int | float | bool] = {"token_id": token_id}

        data = await self._get(url, params=params, use_data_api=False)

        # Strict validation
        if not isinstance(data, dict):
            raise ValueError(f"Expected dict response, got {type(data).__name__}")

        if "bids" not in data or "asks" not in data:
            raise ValueError(f"Response missing bids/asks: {list(data.keys())}")

        bids_data = data["bids"]
        asks_data = data["asks"]

        if not isinstance(bids_data, list) or not isinstance(asks_data, list):
            raise ValueError("bids and asks must be lists")

        # Parse bids
        bids: list[OrderBookLevel] = []
        for level in bids_data:
            if not isinstance(level, dict):
                continue  # Skip non-dict entries
            if "price" not in level or "size" not in level:
                continue  # Skip entries missing required fields

            try:
                bids.append(
                    OrderBookLevel(price=parse_decimal_string(level["price"]), size=parse_decimal_string(level["size"]))
                )
            except (ValueError, TypeError):
                # Skip entries with invalid price/size values
                continue

        # Parse asks
        asks: list[OrderBookLevel] = []
        for level in asks_data:
            if not isinstance(level, dict):
                continue  # Skip non-dict entries
            if "price" not in level or "size" not in level:
                continue  # Skip entries missing required fields

            try:
                asks.append(
                    OrderBookLevel(price=parse_decimal_string(level["price"]), size=parse_decimal_string(level["size"]))
                )
            except (ValueError, TypeError):
                # Skip entries with invalid price/size values
                continue

        # Sort by price (convert to float for comparison)
        bids = sorted(bids, key=lambda x: float(x.price), reverse=True)
        asks = sorted(asks, key=lambda x: float(x.price))

        # Calculate best bid/ask from sorted levels
        best_bid = float(bids[0].price) if bids else None
        best_ask = float(asks[0].price) if asks else None

        # Parse timestamp in milliseconds
        try:
            timestamp = parse_timestamp_ms(data.get("timestamp", 0))
        except (ValueError, TypeError):
            # Default to 0 for invalid/missing timestamps
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

            # Add bid levels
            for level in ob.bids:
                rows.append(
                    {
                        "token_id": ob.token_id,
                        "timestamp": ob.timestamp,
                        "side": "bid",
                        "price": level.price,  # String
                        "size": level.size,  # String
                    }
                )

            # Add ask levels
            for level in ob.asks:
                rows.append(
                    {
                        "token_id": ob.token_id,
                        "timestamp": ob.timestamp,
                        "side": "ask",
                        "price": level.price,  # String
                        "size": level.size,  # String
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

        # Fallback: handle wrapped format if API changes
        if isinstance(data, dict):
            data_list = data.get("data")
            if isinstance(data_list, list):
                out: list[dict[str, str | int | float]] = []
                for item in data_list:
                    if isinstance(item, dict):
                        out.append(item)  # type: ignore[arg-type]
                return out

        return []

    async def fetch_trades(self, market_ids: list[str] | None = None, since_ts: int | None = None) -> pl.DataFrame:
        rows: list[dict[str, str | int | float]] = []
        offset = 0
        limit = 1000

        while True:
            batch = await self.fetch_trades_paged(limit=limit, offset=offset, market_ids=market_ids)
            if not batch:
                break

            if len(rows) + len(batch) > self.max_trades:
                remaining = self.max_trades - len(rows)
                rows.extend(batch[:remaining])
                break

            rows.extend(batch)
            offset += limit

            if len(batch) < limit:
                break

        if not rows:
            # Schema matches actual Data API response types
            # Note: API returns price/size as numbers, not strings
            schema = {
                "id": pl.Utf8,
                "market": pl.Utf8,
                "asset_id": pl.Utf8,
                "side": pl.Utf8,
                "size": pl.Float64,  # API returns as number
                "price": pl.Float64,  # API returns as number
                "timestamp": pl.Int64,
            }
            return pl.DataFrame(schema=schema)

        df = pl.DataFrame(rows)

        # Convert price/size to strings for decimal precision (if present)
        if "price" in df.columns:
            df = df.with_columns(pl.col("price").cast(pl.Utf8).alias("price"))
        if "size" in df.columns:
            df = df.with_columns(pl.col("size").cast(pl.Utf8).alias("size"))

        # Ensure timestamp is Int64 milliseconds
        if "timestamp" in df.columns:
            df = df.with_columns(pl.col("timestamp").cast(pl.Int64))
        else:
            raise ValueError("Trades data missing required 'timestamp' field")

        # Filter by timestamp if provided (milliseconds)
        if since_ts is not None:
            df = df.filter(pl.col("timestamp") >= since_ts)

        logger.info(f"Fetched {len(df)} total trades")
        return df
