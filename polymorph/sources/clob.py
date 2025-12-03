import httpx
import polars as pl

from polymorph.core.base import DataSource, PipelineContext
from polymorph.core.rate_limit import CLOB_RATE_LIMIT, DATA_API_RATE_LIMIT, RateLimiter, RateLimitError
from polymorph.core.retry import with_retry
from polymorph.models.api import OrderBook, OrderBookLevel
from polymorph.utils.logging import get_logger

logger = get_logger(__name__)

# JSON type aliases for strict typing
JsonValue = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
JsonDict = dict[str, JsonValue]
JsonList = list[JsonValue]

CLOB_BASE = "https://clob.polymarket.com"
DATA_API = "https://data-api.polymarket.com"

# CLOB API has a max time window of ~14 days for price history
MAX_PRICE_HISTORY_DAYS = 14
MAX_PRICE_HISTORY_SECONDS = MAX_PRICE_HISTORY_DAYS * 24 * 60 * 60


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

    @with_retry(max_attempts=5, min_wait=2.0, max_wait=30.0)
    async def _get(
        self, url: str, params: dict[str, int | str | bool] | None = None, use_data_api: bool = False
    ) -> JsonDict | JsonList:
        if use_data_api:
            rate_limiter = await self._get_data_rate_limiter()
        else:
            rate_limiter = await self._get_clob_rate_limiter()

        await rate_limiter.acquire()

        client = await self._get_client()
        r = await client.get(url, params=params, timeout=client.timeout)

        if r.status_code == 429:
            logger.warning("Rate limit exceeded (429), raising RateLimitError")
            raise RateLimitError("Rate limit exceeded")

        r.raise_for_status()
        result: JsonDict | JsonList = r.json()
        return result

    async def _fetch_price_history_chunk(
        self, token_id: str, start_ts: int, end_ts: int, fidelity: int
    ) -> pl.DataFrame:
        url = f"{self.clob_base_url}/prices-history"
        params: dict[str, int | str] = {
            "market": token_id,
            "startTs": start_ts,
            "endTs": end_ts,
            "fidelity": fidelity,
        }

        data = await self._get(url, params=params, use_data_api=False)

        if not data:
            return pl.DataFrame()

        df = pl.DataFrame(data)
        df = df.with_columns([pl.lit(token_id).alias("token_id")])

        return df

    async def fetch_prices_history(
        self,
        token_id: str,
        start_ts: int,
        end_ts: int,
        fidelity: int | None = None,
    ) -> pl.DataFrame:
        fidelity = fidelity or self.default_fidelity

        time_span = end_ts - start_ts

        if time_span <= MAX_PRICE_HISTORY_SECONDS:
            return await self._fetch_price_history_chunk(token_id, start_ts, end_ts, fidelity)

        logger.debug(
            f"Chunking price history for {token_id}: "
            f"{time_span / 86400:.1f} days -> {time_span // MAX_PRICE_HISTORY_SECONDS + 1} chunks"
        )

        results: list[pl.DataFrame] = []
        current_start = start_ts

        while current_start < end_ts:
            current_end = min(current_start + MAX_PRICE_HISTORY_SECONDS, end_ts)

            df = await self._fetch_price_history_chunk(token_id, current_start, current_end, fidelity)
            if df.height > 0:
                results.append(df)

            current_start = current_end + 1

        if not results:
            return pl.DataFrame()

        combined = pl.concat(results, how="vertical")

        if "t" in combined.columns:
            combined = combined.unique(subset=["t"], maintain_order=True)

        logger.debug(f"Fetched {combined.height} price points for {token_id}")

        return combined

    async def fetch_order_book(self, token_id: str) -> OrderBook:
        url = f"{self.clob_base_url}/book"
        params: dict[str, int | str | bool] = {"token_id": token_id}

        data = await self._get(url, params=params, use_data_api=False)

        if not isinstance(data, dict):
            raise ValueError(f"Unexpected order book response format: {type(data)}")

        # Parse bids and asks
        bids_data = data.get("bids", [])
        asks_data = data.get("asks", [])

        bids = []
        if isinstance(bids_data, list):
            for level in bids_data:
                if isinstance(level, dict):
                    price = level.get("price")
                    size = level.get("size")
                    if (
                        price is not None
                        and size is not None
                        and isinstance(price, (int, float, str))
                        and isinstance(size, (int, float, str))
                    ):
                        bids.append(OrderBookLevel(price=float(price), size=float(size)))

        asks = []
        if isinstance(asks_data, list):
            for level in asks_data:
                if isinstance(level, dict):
                    price = level.get("price")
                    size = level.get("size")
                    if (
                        price is not None
                        and size is not None
                        and isinstance(price, (int, float, str))
                        and isinstance(size, (int, float, str))
                    ):
                        asks.append(OrderBookLevel(price=float(price), size=float(size)))

        # Sort bids descending, asks ascending
        bids = sorted(bids, key=lambda x: x.price, reverse=True)
        asks = sorted(asks, key=lambda x: x.price)

        # Extract best bid/ask
        best_bid = bids[0].price if bids else None
        best_ask = asks[0].price if asks else None

        # FIXED: Parse timestamp safely
        timestamp_val = data.get("timestamp", 0)
        timestamp = int(timestamp_val) if isinstance(timestamp_val, (int, float)) else 0

        # Create order book object
        order_book = OrderBook(
            token_id=token_id,
            timestamp=timestamp,
            bids=bids,
            asks=asks,
            best_bid=best_bid,
            best_ask=best_ask,
        )

        # Calculate derived metrics
        order_book.mid_price = order_book.calculate_mid_price()
        order_book.spread = order_book.calculate_spread()

        return order_book

    async def fetch_order_books_batch(self, token_ids: list[str]) -> list[OrderBook]:
        results: list[OrderBook] = []

        for token_id in token_ids:
            try:
                order_book = await self.fetch_order_book(token_id)
                results.append(order_book)
            except Exception as e:
                logger.error(f"Failed to fetch order book for {token_id}: {e}")

        return results

    async def fetch_order_book_to_dataframe(self, token_id: str) -> pl.DataFrame:
        order_book = await self.fetch_order_book(token_id)

        bid_rows = [
            {
                "token_id": order_book.token_id,
                "timestamp": order_book.timestamp,
                "side": "bid",
                "price": level.price,
                "size": level.size,
            }
            for level in order_book.bids
        ]

        ask_rows = [
            {
                "token_id": order_book.token_id,
                "timestamp": order_book.timestamp,
                "side": "ask",
                "price": level.price,
                "size": level.size,
            }
            for level in order_book.asks
        ]

        all_rows = bid_rows + ask_rows

        if not all_rows:
            return pl.DataFrame(
                schema={
                    "token_id": pl.Utf8,
                    "timestamp": pl.Int64,
                    "side": pl.Utf8,
                    "price": pl.Float64,
                    "size": pl.Float64,
                }
            )

        return pl.DataFrame(all_rows)

    async def fetch_spread(self, token_id: str) -> dict[str, str | float | int | None]:
        order_book = await self.fetch_order_book(token_id)
        return {
            "token_id": token_id,
            "bid": order_book.best_bid,
            "ask": order_book.best_ask,
            "mid": order_book.mid_price,
            "spread": order_book.spread,
            "timestamp": order_book.timestamp,
        }

    async def fetch_trades_paged(
        self,
        limit: int = 1000,
        offset: int = 0,
        market_ids: list[str] | None = None,
    ) -> JsonList:
        params: dict[str, str | int] = {"limit": limit, "offset": offset}
        if market_ids:
            params["market"] = ",".join(market_ids)

        url = f"{self.data_api_url}/trades"
        data = await self._get(url, params=params)

        if isinstance(data, list):
            return data
        else:
            data_field = data.get("data")
            return data_field if isinstance(data_field, list) else []

    async def fetch_trades(
        self,
        market_ids: list[str] | None = None,
        since_ts: int | None = None,
    ) -> pl.DataFrame:
        logger.info(f"Fetching trades (markets={len(market_ids) if market_ids else 'all'})")

        rows: JsonList = []
        offset = 0
        limit = 1000

        while True:
            batch = await self.fetch_trades_paged(limit=limit, offset=offset, market_ids=market_ids)

            if not batch:
                logger.debug(f"No more trades at offset {offset}")
                break

            rows.extend(batch)
            logger.debug(f"Fetched {len(batch)} trades (total: {len(rows)})")

            offset += limit

            if len(batch) < limit or offset > self.max_trades:
                if offset > self.max_trades:
                    logger.warning(f"Reached max trades limit: {self.max_trades}")
                break

        if not rows:
            return pl.DataFrame()

        df = pl.DataFrame(rows)

        # Parse timestamp from created_at if needed
        if "timestamp" not in df.columns and "created_at" in df.columns:
            df = df.with_columns(
                pl.col("created_at")
                .str.strptime(pl.Datetime, strict=False, format="%Y-%m-%dT%H:%M:%S%z")
                .cast(pl.Int64)
                .alias("timestamp")
            )

        # Filter by timestamp if provided
        if since_ts is not None and "timestamp" in df.columns:
            df = df.filter(pl.col("timestamp") >= since_ts)

        logger.info(f"Fetched {len(df)} total trades")

        return df

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "CLOB":
        return self

    async def __aexit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: object,
    ) -> None:
        await self.close()
