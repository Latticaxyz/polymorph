from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Awaitable, TypeVar

import polars as pl
from rich.progress import Progress, ProgressColumn, SpinnerColumn, Task, TaskID, TextColumn, TimeElapsedColumn
from rich.text import Text

from polymorph.core.base import PipelineContext
from polymorph.core.fetch_cache import FetchCache
from polymorph.core.rate_limit import RateLimiter
from polymorph.core.storage import PathStorage
from polymorph.models.api import OrderBook
from polymorph.models.pipeline import FetchResult
from polymorph.sources.clob import CLOB
from polymorph.sources.gamma import Gamma
from polymorph.utils.logging import get_logger
from polymorph.utils.time import datetime_to_ms, parse_iso_to_ms, time_delta_ms, utc

T = TypeVar("T")

logger = get_logger(__name__)


def add_metadata_columns(df: pl.DataFrame, source: str, endpoint: str, timestamp: datetime) -> pl.DataFrame:
    return df.with_columns(
        [
            pl.lit(source).alias("_source_api"),
            pl.lit(timestamp).alias("_fetch_timestamp"),
            pl.lit(endpoint).alias("_api_endpoint"),
        ]
    )


class RateLimiterColumn(ProgressColumn):
    def __init__(self, rate_limiter: RateLimiter | None = None) -> None:
        super().__init__()
        self._rate_limiter = rate_limiter

    def render(self, _task: Task) -> Text:
        if self._rate_limiter is None:
            return Text("")
        return Text(f"| {self._rate_limiter.get_rps():.1f} req/s")


def create_progress(
    label: str,
    total: int | None = None,
    rate_limiter: RateLimiter | None = None,
) -> tuple[Progress, TaskID]:
    columns: list[ProgressColumn] = [
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TextColumn("{task.completed}"),
    ]
    if total is not None:
        columns.append(TextColumn("/ {task.total} fetched"))
    else:
        columns.append(TextColumn("fetched"))
    columns.append(RateLimiterColumn(rate_limiter))
    columns.append(TimeElapsedColumn())

    progress = Progress(*columns, refresh_per_second=10)
    task_id = progress.add_task(label, total=total)
    return progress, task_id


@dataclass
class TokenFetchJob:
    token_id: str
    start_ts: int
    end_ts: int
    market_id: str
    created_at_ts: int


class BatchResultWriter:
    def __init__(
        self,
        storage: PathStorage,
        base_path: Path,
        batch_size: int = 1000,
        flush_interval_seconds: float = 30.0,
    ):
        self.storage = storage
        self.base_path = base_path
        self.batch_size = batch_size
        self.flush_interval = flush_interval_seconds
        self._buffer: list[pl.DataFrame] = []
        self._last_flush = time.monotonic()
        self._total_written = 0
        self._part_number = 0

    def add(self, df: pl.DataFrame) -> None:
        if df.height > 0:
            self._buffer.append(df)

        should_flush = (
            len(self._buffer) >= self.batch_size or (time.monotonic() - self._last_flush) >= self.flush_interval
        )

        if should_flush and self._buffer:
            self._flush()

    def _flush(self) -> None:
        if not self._buffer:
            return

        combined = pl.concat(self._buffer, how="vertical")
        self._buffer = []
        self._last_flush = time.monotonic()

        part_path = self.base_path.parent / f"{self.base_path.stem}_part{self._part_number:04d}.parquet"
        self.storage.write(combined, part_path)
        self._part_number += 1
        self._total_written += combined.height

    def finalize(self) -> tuple[int, int]:
        if self._buffer:
            self._flush()
        return self._total_written, self._part_number


class FetchStage:
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
        use_gamma_cache: bool = True,
    ):
        self.context = context
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
        self.gamma = Gamma(context, use_cache=use_gamma_cache)
        self.clob = CLOB(context)
        self._pending_consolidations: list[tuple[Path, str]] = []

    def _stamp(self) -> str:
        return self.context.run_timestamp.strftime("%Y%m%dT%H%M%SZ")

    def _build_token_jobs(
        self,
        markets_df: pl.DataFrame,
        global_start_ts: int,
        global_end_ts: int,
    ) -> list[TokenFetchJob]:
        jobs: list[TokenFetchJob] = []
        for row in markets_df.iter_rows(named=True):
            market_id: str = row["id"]
            token_ids: list[str] = row.get("token_ids") or []

            created_at = row.get("created_at")
            end_date = row.get("end_date")
            resolution_date = row.get("resolution_date")

            created_at_ts = parse_iso_to_ms(created_at) if created_at else global_start_ts
            market_end_ms = global_end_ts
            if resolution_date:
                market_end_ms = min(market_end_ms, parse_iso_to_ms(resolution_date))
            elif end_date:
                market_end_ms = min(market_end_ms, parse_iso_to_ms(end_date))

            effective_start = max(global_start_ts, created_at_ts)
            effective_end = min(global_end_ts, market_end_ms)

            if effective_start >= effective_end:
                continue

            for tid in token_ids:
                jobs.append(
                    TokenFetchJob(
                        token_id=tid,
                        start_ts=effective_start,
                        end_ts=effective_end,
                        market_id=market_id,
                        created_at_ts=created_at_ts,
                    )
                )
        return jobs

    def _consolidate_part_files(self, run_dir: Path, prefix: str) -> Path | None:
        resolved_dir = self.storage._resolve_path(run_dir)
        pattern = f"{prefix}_part*.parquet"
        part_files = sorted(resolved_dir.glob(pattern))
        if not part_files:
            return None

        run_name = run_dir.name if run_dir.name else run_dir.parts[-1]
        processed_dir = Path("processed") / f"{run_name}_processed"

        logger.info(f"Consolidating {len(part_files)} {prefix} part files from {run_dir} to {processed_dir}...")

        merged_path = processed_dir / f"{prefix}.parquet"

        dfs = [pl.read_parquet(f) for f in part_files]
        merged = pl.concat(dfs, how="vertical")
        self.storage.write(merged, merged_path)

        logger.info(
            f"Consolidated {prefix} into {merged_path} ({merged.height:,} rows). " f"Part files retained in {run_dir}."
        )
        return self.storage._resolve_path(merged_path)

    def _copy_non_part_files_to_processed(self, run_dir: Path) -> None:
        resolved_dir = self.storage._resolve_path(run_dir)
        run_name = run_dir.name if run_dir.name else run_dir.parts[-1]
        processed_dir = Path("processed") / f"{run_name}_processed"

        files_to_copy = ["markets.parquet", "trades.parquet", "orderbooks.parquet", "spreads.parquet"]

        for filename in files_to_copy:
            src_path = resolved_dir / filename
            if src_path.exists():
                try:
                    df = pl.read_parquet(src_path)
                    dst_path = processed_dir / filename
                    self.storage.write(df, dst_path)
                    logger.info(f"Copied {filename} to {processed_dir}")
                except Exception as e:
                    logger.warning(f"Failed to copy {filename} to processed dir: {e}")

    def _run_pending_consolidations(self) -> tuple[Path | None, bool]:
        processed_dir: Path | None = None
        any_consolidated = False

        for run_dir, prefix in self._pending_consolidations:
            try:
                consolidated_path = self._consolidate_part_files(run_dir, prefix)
                if consolidated_path:
                    any_consolidated = True
                    run_name = run_dir.name if run_dir.name else run_dir.parts[-1]
                    processed_dir = self.storage._resolve_path(Path("processed") / f"{run_name}_processed")
            except Exception as e:
                logger.warning(f"Failed to consolidate {prefix} part files in {run_dir}: {e}")

        if any_consolidated and self._pending_consolidations:
            run_dir = self._pending_consolidations[0][0]
            try:
                self._copy_non_part_files_to_processed(run_dir)
            except Exception as e:
                logger.warning(f"Failed to copy non-part files: {e}")

        self._pending_consolidations.clear()
        return processed_dir, any_consolidated

    async def _fetch_with_progress(
        self,
        label: str,
        coros: Sequence[Awaitable[T]],
        sem: asyncio.Semaphore,
    ) -> list[T | BaseException]:
        total = len(coros)
        progress, task_id = create_progress(label, total)
        results: list[T | BaseException] = [None] * total  # type: ignore[list-item]

        async def tracked(idx: int, coro: Awaitable[T]) -> None:
            async with sem:
                try:
                    results[idx] = await coro
                except Exception as e:
                    results[idx] = e
                progress.update(task_id, advance=1)

        tasks = [asyncio.create_task(tracked(i, c)) for i, c in enumerate(coros)]

        with progress:
            while not all(t.done() for t in tasks):
                await asyncio.sleep(0.1)

        return results

    async def _fetch_prices_concurrent(
        self,
        jobs: list[TokenFetchJob],
        clob: CLOB,
        progress: Progress,
        task_id: TaskID,
        cache: FetchCache | None,
        writer: BatchResultWriter | None,
        run_timestamp: datetime,
    ) -> list[pl.DataFrame | BaseException]:
        sem = asyncio.Semaphore(self.max_concurrency)

        async def fetch_one(job: TokenFetchJob) -> pl.DataFrame | BaseException:
            async with sem:
                try:
                    df = await clob.fetch_prices_history(
                        job.token_id,
                        start_ts=job.start_ts,
                        end_ts=job.end_ts,
                        cache=cache,
                        created_at_ts=job.created_at_ts,
                    )
                    if writer and df.height > 0:
                        writer.add(add_metadata_columns(df, "clob.polymarket.com", "/prices-history", run_timestamp))
                    progress.advance(task_id)
                    return df
                except Exception as e:
                    progress.advance(task_id)
                    return e

        return await asyncio.gather(*[fetch_one(job) for job in jobs])

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

        try:
            result = await self._execute_fetch(start_ts, end_ts, stamp, result, sem)
        finally:
            processed_dir, _ = self._run_pending_consolidations()
            if processed_dir:
                result.processed_dir = processed_dir

        return result

    async def _execute_fetch(
        self,
        start_ts: int,
        end_ts: int,
        stamp: str,
        result: FetchResult,
        sem: asyncio.Semaphore,
    ) -> FetchResult:
        markets_df = None
        token_ids: list[str] = []
        run_dir = Path("raw") / stamp

        if self.include_gamma:
            progress, task_id = create_progress("markets")

            def on_markets_progress(count: int) -> None:
                progress.update(task_id, completed=count)

            async def fetch_markets_coro() -> pl.DataFrame | None:
                async with self.gamma:
                    return await self.gamma.fetch_markets(
                        resolved_only=self.resolved_only,
                        start_ts=start_ts,
                        end_ts=end_ts,
                        on_progress=on_markets_progress,
                    )

            with progress:
                markets_df = await fetch_markets_coro()

            if markets_df is not None and markets_df.height > 0:
                markets_df = add_metadata_columns(
                    markets_df, "gamma-api.polymarket.com", "/markets", self.context.run_timestamp
                )
                path = run_dir / "markets.parquet"
                self.storage.write(markets_df, path)
                result.markets_path = self.storage._resolve_path(path)
                result.market_count = markets_df.height

                token_ids = (
                    markets_df.select("token_ids").explode("token_ids").drop_nulls().unique().to_series().to_list()
                )
                result.token_count = len(token_ids)

        token_jobs: list[TokenFetchJob] = []
        if self.include_prices and markets_df is not None and markets_df.height > 0:
            token_jobs = self._build_token_jobs(markets_df, start_ts, end_ts)
            logger.info(f"Built {len(token_jobs)} token fetch jobs with per-market time bounds")

        if self.include_prices and token_jobs:
            cache_path = self.context.data_dir / ".fetch_cache.db"
            cache = FetchCache(cache_path)
            cached_count = cache.get_total_completed()
            if cached_count > 0:
                logger.info(f"Resuming with {cached_count} cached chunk windows")

            base_path = run_dir / "prices.parquet"
            self._pending_consolidations.append((run_dir, "prices"))
            writer = BatchResultWriter(
                storage=self.storage,
                base_path=base_path,
                batch_size=1000,
                flush_interval_seconds=30.0,
            )

            try:
                async with self.clob:
                    rate_limiter = await self.clob._get_clob_rate_limiter()
                    progress, task_id = create_progress("prices", len(token_jobs), rate_limiter)

                    with progress:
                        await self._fetch_prices_concurrent(
                            jobs=token_jobs,
                            clob=self.clob,
                            progress=progress,
                            task_id=task_id,
                            cache=cache,
                            writer=writer,
                            run_timestamp=self.context.run_timestamp,
                        )

                total_written, part_count = writer.finalize()
                if total_written > 0:
                    result.prices_path = self.storage._resolve_path(run_dir)
                    result.price_point_count = total_written
                    logger.info(f"Wrote {total_written} price points across {part_count} part files")
            finally:
                cache.close()

        if self.include_orderbooks and token_ids:
            async with self.clob:
                ob_coros: Sequence[Awaitable[OrderBook]] = [self.clob.fetch_orderbook(tid) for tid in token_ids]
                orderbook_results = await self._fetch_with_progress("orderbooks", ob_coros, sem)

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
                df = add_metadata_columns(df, "clob.polymarket.com", "/book", self.context.run_timestamp)
                path = run_dir / "orderbooks.parquet"
                self.storage.write(df, path)
                result.orderbooks_path = self.storage._resolve_path(path)
                result.orderbook_levels = df.height

        if self.include_spreads and token_ids:
            async with self.clob:
                spread_coros: Sequence[Awaitable[dict[str, str | float | int | None]]] = [
                    self.clob.fetch_spread(tid) for tid in token_ids
                ]
                spread_results = await self._fetch_with_progress("spreads", spread_coros, sem)

            rows: list[dict[str, str | float | int | None]] = [
                r for r in spread_results if isinstance(r, dict) and not isinstance(r, BaseException)
            ]
            if rows:
                df = pl.DataFrame(rows)
                df = add_metadata_columns(df, "clob.polymarket.com", "/book", self.context.run_timestamp)
                path = run_dir / "spreads.parquet"
                self.storage.write(df, path)
                result.spreads_path = self.storage._resolve_path(path)
                result.spreads_count = df.height

        if self.include_trades:
            progress, task_id = create_progress("trades")

            def on_trades_progress(count: int) -> None:
                progress.update(task_id, completed=count)

            async def fetch_trades_coro() -> pl.DataFrame:
                market_ids = (
                    markets_df.select("id").drop_nulls().to_series().to_list() if markets_df is not None else None
                )
                async with self.clob:
                    return await self.clob.fetch_trades(
                        market_ids=market_ids, since_ts=start_ts, on_progress=on_trades_progress
                    )

            with progress:
                trades_df = await fetch_trades_coro()

            if trades_df is not None and trades_df.height > 0:
                trades_df = add_metadata_columns(
                    trades_df, "data-api.polymarket.com", "/trades", self.context.run_timestamp
                )
                path = run_dir / "trades.parquet"
                self.storage.write(trades_df, path)
                result.trades_path = self.storage._resolve_path(path)
                result.trade_count = trades_df.height

        return result
