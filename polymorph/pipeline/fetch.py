from __future__ import annotations
import asyncio
from datetime import datetime
from pathlib import Path
import httpx
import polars as pl
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
)
from polymorph import sources, io, config, utils


def _ts(dt: datetime) -> int:
    return int(dt.timestamp())


async def last_n_months(
    n_months: int,
    out_root: Path,
    include_trades: bool,
    include_prices: bool,
    include_gamma: bool,
    max_concurrency: int | None,
):
    out_root = out_root.resolve()
    http_timeout = config.settings.http_timeout
    concurrency = max_concurrency or config.settings.max_concurrency
    async with httpx.AsyncClient(timeout=http_timeout, http2=True) as client:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
        ) as progress:
            market_df = pl.DataFrame()
            if include_gamma:
                t = progress.add_task("Fetching markets (gamma)", total=None)
                market_df = await sources.gamma.fetch_markets(client)
                progress.remove_task(t)
                if market_df.height:
                    io.storage.write_parquet(
                        market_df, out_root / "raw" / "gamma" / "markets.parquet"
                    )

            token_ids = []
            if market_df.height and "token_id" in market_df.columns:
                token_ids = [str(x) for x in market_df["token_id"].to_list() if x]

            start = utils.time.months_ago(n_months)
            end = utils.time.utc()
            start_ts, end_ts = _ts(start), _ts(end)

            if include_prices and token_ids:
                sem = asyncio.Semaphore(concurrency)
                prices_out: list[pl.DataFrame] = []

                async def worker(tok: str):
                    async with sem:
                        df = await sources.clob.fetch_prices_history(
                            client, tok, start_ts, end_ts, fidelity=60
                        )
                        if df.height:
                            prices_out.append(df)

                task = progress.add_task(
                    "Fetching prices-history", total=len(token_ids)
                )
                tasks = [asyncio.create_task(worker(tok)) for tok in token_ids]
                for coro in asyncio.as_completed(tasks):
                    await coro
                    progress.advance(task)
                if prices_out:
                    prices = pl.concat(prices_out, how="diagonal_relaxed")
                    io.storage.write_parquet(
                        prices, out_root / "raw" / "clob" / "prices_history.parquet"
                    )

            if include_trades:
                trades = await sources.clob.backfill_trades(
                    client, market_ids=token_ids or None, since_ts=start_ts
                )
                if trades.height:
                    io.storage.write_parquet(
                        trades, out_root / "raw" / "data_api" / "trades.parquet"
                    )
