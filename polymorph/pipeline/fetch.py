import asyncio
from pathlib import Path
from typing import Awaitable, TypeVar

import polars as pl
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from polymorph.core.base import PipelineContext, PipelineStage
from polymorph.models.api import OrderBook
from polymorph.models.pipeline import FetchResult
from polymorph.sources.clob import CLOB
from polymorph.sources.gamma import Gamma
from polymorph.utils.logging import get_logger
from polymorph.utils.time import datetime_to_ms, time_delta_ms, utc

T = TypeVar("T")

logger = get_logger(__name__)


class FetchStage(PipelineStage[None, FetchResult]):
    def __init__(
        self,
        context: PipelineContext,
        minutes: int = 0,
        hours: int = 0,
        days: int = 0,
        weeks: int = 0,
        months: int = 0,
        years: int = 0,
        include_gamma: bool = True,
        include_prices: bool = True,
        include_trades: bool = True,
        include_orderbooks: bool = False,
        include_spreads: bool = False,
        resolved_only: bool = False,
        max_concurrency: int | None = None,
    ):
        super().__init__(context)
        self.minutes = minutes
        self.hours = hours
        self.days = days
        self.weeks = weeks
        self.months = months
        self.years = years
        self.include_gamma = include_gamma
        self.include_prices = include_prices
        self.include_trades = include_trades
        self.include_orderbooks = include_orderbooks
        self.include_spreads = include_spreads
        self.resolved_only = resolved_only
        self.max_concurrency = max_concurrency or context.max_concurrency

        self.storage = context.storage
        self.gamma = Gamma(context)
        self.clob = CLOB(context)

    @property
    def name(self) -> str:
        return "fetch"

    def _stamp(self) -> str:
        return self.context.run_timestamp.strftime("%Y%m%dT%H%M%SZ")

    async def execute(self, _input: None = None) -> FetchResult:
        start_ts = time_delta_ms(
            minutes=self.minutes,
            hours=self.hours,
            days=self.days,
            weeks=self.weeks,
            months=self.months,
            years=self.years,
        )
        end_ts = datetime_to_ms(utc())
        stamp = self._stamp()

        result = FetchResult(run_timestamp=self.context.run_timestamp)
        sem = asyncio.Semaphore(self.max_concurrency)

        async def limited(coro: Awaitable[T]) -> T:
            async with sem:
                return await coro

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task("fetch", total=6)

            markets_df = None
            token_ids: list[str] = []

            if self.include_gamma:
                progress.update(task, advance=1, description="gamma markets")
                async with self.gamma:
                    markets_df = await self.gamma.fetch_markets(
                        resolved_only=self.resolved_only, start_ts=start_ts, end_ts=end_ts
                    )

                if markets_df.height > 0:
                    markets_df = markets_df.with_columns(
                        [
                            pl.lit("gamma-api.polymarket.com").alias("_source_api"),
                            pl.lit(self.context.run_timestamp).alias("_fetch_timestamp"),
                            pl.lit("/markets").alias("_api_endpoint"),
                        ]
                    )
                    path = Path("raw/gamma") / f"{stamp}_markets.parquet"
                    self.storage.write(markets_df, path)
                    result.markets_path = self.storage._resolve_path(path)
                    result.market_count = markets_df.height

                    token_ids = (
                        markets_df.select("token_ids").explode("token_ids").drop_nulls().unique().to_series().to_list()
                    )
                    result.token_count = len(token_ids)

            if self.include_prices and token_ids:
                progress.update(task, advance=1, description="prices")
                async with self.clob:
                    dfs = await asyncio.gather(
                        *[limited(self.clob.fetch_prices_history(tid, interval="all")) for tid in token_ids],
                        return_exceptions=True,
                    )
                    # TODO: RATE LIMITING - Concurrent token fetching + chunking overwhelms rate limiter
                    # TODO: Each token may require multiple chunked requests, happening concurrently across tokens
                    # TODO: Fix rate limiter race condition and consider chunking strategy

                valid_dfs: list[pl.DataFrame] = [df for df in dfs if isinstance(df, pl.DataFrame) and df.height > 0]
                if valid_dfs:
                    df = pl.concat(valid_dfs)
                    df = df.with_columns(
                        [
                            pl.lit("clob.polymarket.com").alias("_source_api"),
                            pl.lit(self.context.run_timestamp).alias("_fetch_timestamp"),
                            pl.lit("/prices-history").alias("_api_endpoint"),
                        ]
                    )
                    path = Path("raw/clob") / f"{stamp}_prices.parquet"
                    self.storage.write(df, path)
                    result.prices_path = self.storage._resolve_path(path)
                    result.price_point_count = df.height

            if self.include_orderbooks and token_ids:
                progress.update(task, advance=1, description="orderbooks")
                async with self.clob:
                    orderbook_results = await asyncio.gather(
                        *[limited(self.clob.fetch_orderbook(tid)) for tid in token_ids],
                        return_exceptions=True,
                    )

                orderbook_rows: list[dict[str, object]] = []
                for result_item in orderbook_results:
                    if isinstance(result_item, Exception):
                        logger.warning(f"Failed to fetch orderbook: {result_item}")
                        continue
                    if not isinstance(result_item, OrderBook):
                        continue

                    ob = result_item
                    for level in ob.bids:
                        orderbook_rows.append(
                            {
                                "token_id": ob.token_id,
                                "timestamp": ob.timestamp,
                                "side": "bid",
                                "price": level.price,
                                "size": level.size,
                            }
                        )
                    for level in ob.asks:
                        orderbook_rows.append(
                            {
                                "token_id": ob.token_id,
                                "timestamp": ob.timestamp,
                                "side": "ask",
                                "price": level.price,
                                "size": level.size,
                            }
                        )

                if orderbook_rows:
                    df = pl.DataFrame(orderbook_rows)
                    df = df.with_columns(
                        [
                            pl.lit("clob.polymarket.com").alias("_source_api"),
                            pl.lit(self.context.run_timestamp).alias("_fetch_timestamp"),
                            pl.lit("/book").alias("_api_endpoint"),
                        ]
                    )
                    path = Path("raw/clob") / f"{stamp}_orderbooks.parquet"
                    self.storage.write(df, path)
                    result.orderbooks_path = self.storage._resolve_path(path)
                    result.orderbook_levels = df.height

            if self.include_spreads and token_ids:
                progress.update(task, advance=1, description="spreads")
                async with self.clob:
                    spread_results = await asyncio.gather(
                        *[limited(self.clob.fetch_spread(tid)) for tid in token_ids],
                        return_exceptions=True,
                    )

                rows: list[dict[str, str | float | int | None]] = [r for r in spread_results if isinstance(r, dict)]
                if rows:
                    df = pl.DataFrame(rows)
                    df = df.with_columns(
                        [
                            pl.lit("clob.polymarket.com").alias("_source_api"),
                            pl.lit(self.context.run_timestamp).alias("_fetch_timestamp"),
                            pl.lit("/book").alias("_api_endpoint"),
                        ]
                    )
                    path = Path("raw/clob") / f"{stamp}_spreads.parquet"
                    self.storage.write(df, path)
                    result.spreads_path = self.storage._resolve_path(path)
                    result.spreads_count = df.height

            if self.include_trades:
                progress.update(task, advance=1, description="trades")
                market_ids = (
                    markets_df.select("id").drop_nulls().to_series().to_list() if markets_df is not None else None
                )
                async with self.clob:
                    df = await self.clob.fetch_trades(market_ids=market_ids, since_ts=start_ts)

                if df.height > 0:
                    df = df.with_columns(
                        [
                            pl.lit("data-api.polymarket.com").alias("_source_api"),
                            pl.lit(self.context.run_timestamp).alias("_fetch_timestamp"),
                            pl.lit("/trades").alias("_api_endpoint"),
                        ]
                    )
                    path = Path("raw/data_api") / f"{stamp}_trades.parquet"
                    self.storage.write(df, path)
                    result.trades_path = self.storage._resolve_path(path)
                    result.trade_count = df.height

            progress.update(task, advance=1, description="done")

        return result
