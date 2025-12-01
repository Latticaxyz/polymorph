import asyncio
from datetime import datetime
from pathlib import Path

import polars as pl
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from polymorph.core.base import PipelineContext, PipelineStage
from polymorph.core.storage import ParquetStorage
from polymorph.models.pipeline import FetchResult
from polymorph.sources.clob import CLOB
from polymorph.sources.gamma import Gamma
from polymorph.utils.logging import get_logger
from polymorph.utils.time import months_ago, utc

logger = get_logger(__name__)


class FetchStage(PipelineStage[None, FetchResult]):
    def __init__(
        self,
        context: PipelineContext,
        n_months: int = 1,
        include_gamma: bool = True,
        include_prices: bool = True,
        include_trades: bool = True,
        max_concurrency: int | None = None,
    ):
        super().__init__(context)
        self.n_months = n_months
        self.include_gamma = include_gamma
        self.include_prices = include_prices
        self.include_trades = include_trades
        self.max_concurrency = max_concurrency or context.max_concurrency

        self.storage = ParquetStorage(context.data_dir)

        self.gamma_source = Gamma(context)
        self.clob_source = CLOB(context)

    @property
    def name(self) -> str:
        return "fetch"

    def _ts(self, dt: datetime) -> int:
        return int(dt.timestamp())

    def _run_timestamp_str(self) -> str:
        return self.context.run_timestamp.strftime("%Y%m%dT%H%M%SZ")

    async def execute(self, _input_data: None = None) -> FetchResult:
        logger.info(
            f"Starting fetch stage: {self.n_months} months "
            f"(gamma={self.include_gamma}, prices={self.include_prices}, "
            f"trades={self.include_trades})"
        )

        stamp = self._run_timestamp_str()
        start = months_ago(self.n_months)
        end = utc()
        start_ts, end_ts = self._ts(start), self._ts(end)

        result = FetchResult(
            run_timestamp=self.context.run_timestamp,
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            transient=False,
        ) as progress:
            market_df = pl.DataFrame()
            token_ids: list[str] = []

            if self.include_gamma:
                task = progress.add_task("Fetching markets (gamma)", total=None)
                progress.log("[cyan]Starting gamma fetch[/cyan]")

                try:
                    async with self.gamma_source:
                        market_df = await self.gamma_source.fetch()

                    if market_df.height > 0:
                        markets_path = Path("raw/gamma") / f"{stamp}_markets.parquet"
                        self.storage.write(market_df, markets_path)
                        result.markets_path = self.storage._resolve_path(markets_path)
                        result.market_count = market_df.height

                        if "token_ids" in market_df.columns:
                            tokens_df = market_df.select(pl.col("token_ids")).explode("token_ids").drop_nulls().unique()
                            token_ids = [str(x) for x in tokens_df["token_ids"].to_list() if x]
                            result.token_count = len(token_ids)

                        progress.log(
                            f"[green]✓[/green] Gamma: {result.market_count} markets, " f"{result.token_count} tokens"
                        )
                    else:
                        progress.log("[yellow]•[/yellow] Gamma returned 0 rows")

                except Exception as e:
                    logger.error(f"Gamma fetch failed: {e}", exc_info=True)
                    progress.log(f"[red]✗[/red] Gamma fetch failed: {e!r}")

                finally:
                    progress.update(task, visible=False)

            if self.include_prices and token_ids:
                task = progress.add_task(f"Prices history ({self.n_months}m)", total=len(token_ids))

                sem = asyncio.Semaphore(self.max_concurrency)
                prices_out: list[pl.DataFrame] = []

                async def fetch_token_prices(token_id: str) -> None:
                    subtask = progress.add_task(f"[cyan]{token_id}", total=None)
                    try:
                        async with sem:
                            async with self.clob_source:
                                df = await self.clob_source.fetch_prices_history(
                                    token_id, start_ts, end_ts, fidelity=60
                                )

                        if df.height > 0:
                            prices_out.append(df)
                            progress.log(f"[green]✓[/green] {token_id}: {df.height} prices")
                        else:
                            progress.log(f"[yellow]•[/yellow] {token_id}: empty")

                    except Exception as e:
                        logger.error(f"Price fetch failed for {token_id}: {e}", exc_info=True)
                        progress.log(f"[red]✗[/red] {token_id}: {e!r}")

                    finally:
                        progress.update(subtask, visible=False)
                        progress.advance(task)

                await asyncio.gather(*[fetch_token_prices(t) for t in token_ids])

                if prices_out:
                    combined = pl.concat(prices_out, how="vertical")
                    prices_path = Path("raw/clob") / f"{stamp}_prices.parquet"
                    self.storage.write(combined, prices_path)
                    result.prices_path = self.storage._resolve_path(prices_path)
                    result.price_point_count = combined.height
                    progress.log(f"[green]✓[/green] Prices: {result.price_point_count} rows")

                progress.update(task, visible=False)

            elif self.include_prices and not token_ids:
                progress.log("[yellow]•[/yellow] No token IDs available; skipping prices")

            if self.include_trades:
                progress.log("[cyan]Starting trades backfill[/cyan]")
                try:
                    async with self.clob_source:
                        trades_df = await self.clob_source.fetch_trades(market_ids=None, since_ts=start_ts)

                    if trades_df.height > 0:
                        trades_path = Path("raw/clob") / f"{stamp}_trades.parquet"
                        self.storage.write(trades_df, trades_path)
                        result.trades_path = self.storage._resolve_path(trades_path)
                        result.trade_count = trades_df.height
                        progress.log(f"[green]✓[/green] Trades: {result.trade_count} rows")
                    else:
                        progress.log("[yellow]•[/yellow] Trades returned 0 rows")

                except Exception as e:
                    logger.error(f"Trades fetch failed: {e}", exc_info=True)
                    progress.log(f"[red]✗[/red] Trades fetch failed: {e!r}")

            progress.log("[bold green]Fetch complete[/bold green]")

        logger.info(
            f"Fetch stage complete: {result.market_count} markets, "
            f"{result.token_count} tokens, {result.price_point_count} prices, "
            f"{result.trade_count} trades"
        )

        return result
