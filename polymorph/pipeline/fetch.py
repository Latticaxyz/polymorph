from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Awaitable, TypeVar

import polars as pl
from rich.live import Live
from rich.text import Text

from polymorph.core.base import PipelineContext, PipelineStage
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

SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]


@dataclass
class TokenFetchJob:
    token_id: str
    start_ts: int
    end_ts: int
    market_id: str
    created_at_ts: int


@dataclass
class WorkItem:
    index: int
    job: TokenFetchJob


@dataclass
class WorkResult:
    index: int
    result: pl.DataFrame | BaseException


class FetchProgress:
    def __init__(
        self,
        label: str,
        total: int | None = None,
        rate_limiter: RateLimiter | None = None,
    ):
        self.label = label
        self.total = total
        self.completed = 0
        self._start_time = time.monotonic()
        self._rate_limiter = rate_limiter

    def increment(self) -> None:
        self.completed += 1

    def elapsed(self) -> str:
        seconds = time.monotonic() - self._start_time
        hours, remainder = divmod(int(seconds), 3600)
        minutes, secs = divmod(remainder, 60)
        if hours:
            return f"{hours}:{minutes:02d}:{secs:02d}"
        return f"{minutes}:{secs:02d}"

    def render(self) -> Text:
        frame_idx = int(time.monotonic() * 10) % len(SPINNER_FRAMES)
        spinner = SPINNER_FRAMES[frame_idx]

        if self.total is not None:
            remaining = self.total - self.completed
            status = f"{self.completed}/{self.total} fetched, {remaining} remaining"
        else:
            status = f"{self.completed} fetched"

        rps_str = ""
        if self._rate_limiter:
            rps = self._rate_limiter.get_rps()
            rps_str = f" | {rps:.1f} req/s"

        return Text(f"{spinner} {self.label} {status}{rps_str} [{self.elapsed()}]")


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


class PricesFetchOrchestrator:
    def __init__(
        self,
        clob: CLOB,
        num_workers: int,
        progress: FetchProgress,
        cache: FetchCache | None = None,
        writer: BatchResultWriter | None = None,
        run_timestamp: datetime | None = None,
    ):
        self.clob = clob
        self.num_workers = num_workers
        self.progress = progress
        self.cache = cache
        self.writer = writer
        self.run_timestamp = run_timestamp

    async def fetch_all(self, jobs: list[TokenFetchJob]) -> list[pl.DataFrame | BaseException]:
        total = len(jobs)
        if total == 0:
            return []

        job_queue: asyncio.Queue[WorkItem | None] = asyncio.Queue()
        result_queue: asyncio.Queue[WorkResult] = asyncio.Queue()

        for i, job in enumerate(jobs):
            await job_queue.put(WorkItem(index=i, job=job))

        for _ in range(self.num_workers):
            await job_queue.put(None)

        workers = [asyncio.create_task(self._worker(job_queue, result_queue)) for _ in range(self.num_workers)]

        results: list[pl.DataFrame | BaseException] = [
            pl.DataFrame(schema={"token_id": pl.Utf8, "t": pl.Int64, "p": pl.Float64}) for _ in range(total)
        ]

        collected = 0
        while collected < total:
            work_result = await result_queue.get()
            results[work_result.index] = work_result.result
            collected += 1
            self.progress.increment()

            if self.writer and isinstance(work_result.result, pl.DataFrame) and work_result.result.height > 0:
                df_with_meta = work_result.result.with_columns(
                    [
                        pl.lit("clob.polymarket.com").alias("_source_api"),
                        pl.lit(self.run_timestamp).alias("_fetch_timestamp"),
                        pl.lit("/prices-history").alias("_api_endpoint"),
                    ]
                )
                self.writer.add(df_with_meta)

        await asyncio.gather(*workers)
        return results

    async def _worker(
        self,
        job_queue: asyncio.Queue[WorkItem | None],
        result_queue: asyncio.Queue[WorkResult],
    ) -> None:
        while True:
            item = await job_queue.get()
            if item is None:
                job_queue.task_done()
                break

            try:
                df = await self.clob.fetch_prices_history(
                    item.job.token_id,
                    start_ts=item.job.start_ts,
                    end_ts=item.job.end_ts,
                    cache=self.cache,
                    created_at_ts=item.job.created_at_ts,
                )
                await result_queue.put(WorkResult(index=item.index, result=df))
            except Exception as e:
                await result_queue.put(WorkResult(index=item.index, result=e))

            job_queue.task_done()


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

    async def _fetch_with_progress(
        self,
        label: str,
        coros: Sequence[Awaitable[T]],
        sem: asyncio.Semaphore,
    ) -> list[T | BaseException]:
        total = len(coros)
        progress = FetchProgress(label, total)
        results: list[T | BaseException] = [None] * total  # type: ignore[list-item]

        async def tracked(idx: int, coro: Awaitable[T]) -> None:
            async with sem:
                try:
                    results[idx] = await coro
                except Exception as e:
                    results[idx] = e
                progress.increment()

        tasks = [asyncio.create_task(tracked(i, c)) for i, c in enumerate(coros)]

        with Live(progress.render(), refresh_per_second=10) as live:
            while not all(t.done() for t in tasks):
                live.update(progress.render())
                await asyncio.sleep(0.1)
            live.update(progress.render())

        return results

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

        markets_df = None
        token_ids: list[str] = []

        if self.include_gamma:
            progress = FetchProgress("markets")

            def on_markets_progress(count: int) -> None:
                progress.completed = count

            async def fetch_markets_with_live() -> pl.DataFrame | None:
                async with self.gamma:
                    return await self.gamma.fetch_markets(
                        resolved_only=self.resolved_only,
                        start_ts=start_ts,
                        end_ts=end_ts,
                        on_progress=on_markets_progress,
                    )

            with Live(progress.render(), refresh_per_second=10) as live:
                markets_task: asyncio.Task[pl.DataFrame | None] = asyncio.create_task(fetch_markets_with_live())
                while not markets_task.done():
                    live.update(progress.render())
                    await asyncio.sleep(0.1)
                markets_df = await markets_task
                live.update(progress.render())

            if markets_df is not None and markets_df.height > 0:
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

        token_jobs: list[TokenFetchJob] = []
        if self.include_prices and markets_df is not None and markets_df.height > 0:
            token_jobs = self._build_token_jobs(markets_df, start_ts, end_ts)
            logger.info(f"Built {len(token_jobs)} token fetch jobs with per-market time bounds")

        if self.include_prices and token_jobs:
            num_workers = min(self.max_concurrency, 200)
            cache_path = self.context.data_dir / ".fetch_cache.db"
            cache = FetchCache(cache_path)
            cached_count = cache.get_total_completed()
            if cached_count > 0:
                logger.info(f"Resuming with {cached_count} cached chunk windows")

            base_path = Path("raw/clob") / f"{stamp}_prices.parquet"
            writer = BatchResultWriter(
                storage=self.storage,
                base_path=base_path,
                batch_size=1000,
                flush_interval_seconds=30.0,
            )

            try:
                async with self.clob:
                    rate_limiter = await self.clob._get_clob_rate_limiter()
                    progress = FetchProgress("prices", len(token_jobs), rate_limiter=rate_limiter)

                    orchestrator = PricesFetchOrchestrator(
                        clob=self.clob,
                        num_workers=num_workers,
                        progress=progress,
                        cache=cache,
                        writer=writer,
                        run_timestamp=self.context.run_timestamp,
                    )

                    with Live(progress.render(), refresh_per_second=10) as live:
                        fetch_task = asyncio.create_task(orchestrator.fetch_all(token_jobs))
                        while not fetch_task.done():
                            live.update(progress.render())
                            await asyncio.sleep(0.1)
                        await fetch_task
                        live.update(progress.render())

                total_written, part_count = writer.finalize()
                if total_written > 0:
                    result.prices_path = self.storage._resolve_path(base_path.parent)
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
            progress = FetchProgress("trades")

            def on_trades_progress(count: int) -> None:
                progress.completed = count

            async def fetch_trades_with_live() -> pl.DataFrame:
                market_ids = (
                    markets_df.select("id").drop_nulls().to_series().to_list() if markets_df is not None else None
                )
                async with self.clob:
                    return await self.clob.fetch_trades(
                        market_ids=market_ids, since_ts=start_ts, on_progress=on_trades_progress
                    )

            with Live(progress.render(), refresh_per_second=10) as live:
                trades_task: asyncio.Task[pl.DataFrame] = asyncio.create_task(fetch_trades_with_live())
                while not trades_task.done():
                    live.update(progress.render())
                    await asyncio.sleep(0.1)
                trades_df = await trades_task
                live.update(progress.render())

            if trades_df is not None and trades_df.height > 0:
                trades_df = trades_df.with_columns(
                    [
                        pl.lit("data-api.polymarket.com").alias("_source_api"),
                        pl.lit(self.context.run_timestamp).alias("_fetch_timestamp"),
                        pl.lit("/trades").alias("_api_endpoint"),
                    ]
                )
                path = Path("raw/data_api") / f"{stamp}_trades.parquet"
                self.storage.write(trades_df, path)
                result.trades_path = self.storage._resolve_path(path)
                result.trade_count = trades_df.height

        return result
