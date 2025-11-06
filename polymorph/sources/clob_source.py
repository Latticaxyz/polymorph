"""CLOB API data source for price and trade data."""

from typing import Any

import httpx
import polars as pl

from polymorph.core.base import DataSource, PipelineContext
from polymorph.core.retry import with_retry, RateLimitError
from polymorph.models.api import PricePoint, Trade
from polymorph.utils.logging import get_logger

logger = get_logger(__name__)

CLOB_BASE = "https://clob.polymarket.com"
DATA_API = "https://data-api.polymarket.com"


class CLOBSource(DataSource[pl.DataFrame]):
    """Data source for Polymarket CLOB API.

    Fetches price history and trade data from the CLOB
    and data APIs.
    """

    def __init__(
        self,
        context: PipelineContext,
        clob_base_url: str = CLOB_BASE,
        data_api_url: str = DATA_API,
        default_fidelity: int = 60,
        max_trades: int = 200_000,
    ):
        """Initialize CLOB source.

        Args:
            context: Pipeline context
            clob_base_url: Base URL for CLOB API
            data_api_url: Base URL for data API
            default_fidelity: Default price fidelity in seconds
            max_trades: Maximum number of trades to fetch
        """
        super().__init__(context)
        self.clob_base_url = clob_base_url
        self.data_api_url = data_api_url
        self.default_fidelity = default_fidelity
        self.max_trades = max_trades
        self._client: httpx.AsyncClient | None = None

    @property
    def name(self) -> str:
        return "clob"

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self.settings.http_timeout,
                http2=True,
            )
        return self._client

    @with_retry(max_attempts=5, min_wait=1.0, max_wait=10.0)
    async def _get(
        self, url: str, params: dict[str, Any] | None = None
    ) -> dict | list:
        """Make GET request with retry logic.

        Args:
            url: URL to fetch
            params: Query parameters

        Returns:
            JSON response

        Raises:
            RateLimitError: If rate limit is exceeded
        """
        client = await self._get_client()
        r = await client.get(url, params=params, timeout=client.timeout)

        if r.status_code == 429:
            raise RateLimitError("Rate limit exceeded")

        r.raise_for_status()
        return r.json()

    async def fetch_prices_history(
        self,
        token_id: str,
        start_ts: int,
        end_ts: int,
        fidelity: int | None = None,
    ) -> pl.DataFrame:
        """Fetch price history for a token.

        Args:
            token_id: Token ID to fetch prices for
            start_ts: Start timestamp (Unix seconds)
            end_ts: End timestamp (Unix seconds)
            fidelity: Price fidelity in seconds (default: 60)

        Returns:
            DataFrame with columns: t, p, token_id
        """
        url = f"{self.clob_base_url}/prices-history"
        params = {
            "market": token_id,
            "startTs": start_ts,
            "endTs": end_ts,
            "fidelity": fidelity or self.default_fidelity,
        }

        data = await self._get(url, params=params)

        if not data:
            return pl.DataFrame()

        df = pl.DataFrame(data)
        df = df.with_columns([pl.lit(token_id).alias("token_id")])

        return df

    async def fetch_trades_paged(
        self,
        limit: int = 1000,
        offset: int = 0,
        market_ids: list[str] | None = None,
    ) -> list[dict]:
        """Fetch a page of trades.

        Args:
            limit: Maximum number of trades per page
            offset: Offset for pagination
            market_ids: Filter by market IDs

        Returns:
            List of trade dictionaries
        """
        params: dict[str, str | int] = {"limit": limit, "offset": offset}
        if market_ids:
            params["market"] = ",".join(market_ids)

        url = f"{self.data_api_url}/trades"
        data = await self._get(url, params=params)

        return data if isinstance(data, list) else data.get("data", [])

    async def fetch_trades(
        self,
        market_ids: list[str] | None = None,
        since_ts: int | None = None,
    ) -> pl.DataFrame:
        """Fetch trades with pagination.

        Args:
            market_ids: Filter by market IDs
            since_ts: Only fetch trades after this timestamp

        Returns:
            DataFrame with trade data
        """
        logger.info(
            f"Fetching trades (markets={len(market_ids) if market_ids else 'all'})"
        )

        rows: list[dict] = []
        offset = 0
        limit = 1000

        while True:
            batch = await self.fetch_trades_paged(
                limit=limit, offset=offset, market_ids=market_ids
            )

            if not batch:
                logger.debug(f"No more trades at offset {offset}")
                break

            rows.extend(batch)
            logger.debug(
                f"Fetched {len(batch)} trades (total: {len(rows)})"
            )

            offset += limit

            if len(batch) < limit or offset > self.max_trades:
                if offset > self.max_trades:
                    logger.warning(
                        f"Reached max trades limit: {self.max_trades}"
                    )
                break

        if not rows:
            return pl.DataFrame()

        df = pl.DataFrame(rows)

        # Parse timestamp from created_at if needed
        if "timestamp" not in df.columns and "created_at" in df.columns:
            df = df.with_columns(
                pl.col("created_at")
                .str.strptime(
                    pl.Datetime, strict=False, format="%Y-%m-%dT%H:%M:%S%z"
                )
                .cast(pl.Int64)
                .alias("timestamp")
            )

        # Filter by timestamp if provided
        if since_ts is not None and "timestamp" in df.columns:
            df = df.filter(pl.col("timestamp") >= since_ts)

        logger.info(f"Fetched {len(df)} total trades")

        return df

    async def fetch(self, **kwargs) -> dict[str, pl.DataFrame]:
        """Fetch all data from CLOB source.

        This is a placeholder - typically you'd call fetch_prices_history
        or fetch_trades directly based on your needs.

        Args:
            **kwargs: Additional parameters

        Returns:
            Dictionary with 'prices' and 'trades' DataFrames
        """
        return {
            "prices": pl.DataFrame(),
            "trades": pl.DataFrame(),
        }

    async def close(self):
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self):
        """Context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        await self.close()
