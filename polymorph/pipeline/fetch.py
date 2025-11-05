from __future__ import annotations
import asyncio
from datetime import datetime, timezone
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


def _run_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


async def last_n_months(
    n_months: int,
    out_root: Path,
    include_trades: bool,
    include_prices: bool,
    include_gamma: bool,
    max_concurrency: int | None,
) -> None:
    out_root = out_root.resolve()
    stamp = _run_ts()
    http_timeout = httpx.Timeout(
        connect=config.settings.http_timeout,
        read=config.settings.http_timeout,
        write=config.settings.http_timeout,
        pool=config.settings.http_timeout,
    )
    concurrency = max_concurrency or config.settings.max_concurrency
    start = utils.time.months_ago(n_months)
    end = utils.time.utc()
    start_ts, end_ts = _ts(start), _ts(end)
    async with httpx.AsyncClient(timeout=http_timeout, http2=True) as client:
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            transient=False,
        ) as progress:
            market_df = pl.DataFrame()
            token_ids: list[str] = []

            if include_gamma:
                t_markets = progress.add_task("Fetching markets (gamma)", total=None)
                progress.log("[cyan]starting gamma fetch[/cyan]")
                try:
                    market_df = await sources.gamma.fetch_markets(client)
                    if market_df.height:
                        io.storage.write_parquet(
                            market_df,
                            out_root / "raw" / "gamma" / f"{stamp}_markets.parquet",
                        )
                        if "token_ids" in market_df.columns:
                            tokens_df = (
                                market_df.select(pl.col("token_ids"))
                                .explode("token_ids")
                                .drop_nulls()
                                .unique()
                            )
                            token_ids = [
                                str(x) for x in tokens_df["token_ids"].to_list() if x
                            ]
                        elif "outcomes" in market_df.columns:
                            tokens_df = (
                                market_df.select(pl.col("outcomes"))
                                .explode("outcomes")
                                .select(
                                    pl.col("outcomes")
                                    .struct.field("tokenId")
                                    .alias("token_id")
                                )
                                .drop_nulls()
                                .unique()
                            )
                            token_ids = [
                                str(x) for x in tokens_df["token_id"].to_list() if x
                            ]
                        progress.log(
                            f"[green]✓[/green] gamma markets: {market_df.height}, tokens: {len(token_ids)}"
                        )
                    else:
                        progress.log("[yellow]•[/yellow] gamma markets returned 0 rows")
                except Exception as e:
                    progress.log(f"[red]✗[/red] gamma fetch failed: {e!r}")
                finally:
                    progress.update(t_markets, visible=False)

            if include_prices:
                if token_ids:
                    t_prices = progress.add_task(
                        f"Prices-history {n_months}m", total=len(token_ids)
                    )
                    sem = asyncio.Semaphore(concurrency)
                    prices_out: list[pl.DataFrame] = []

                    async def worker(tok: str) -> None:
                        sub = progress.add_task(f"[cyan]{tok}", total=None)
                        try:
                            async with sem:
                                df = await sources.clob.fetch_prices_history(
                                    client, tok, start_ts, end_ts, fidelity=60
                                )
                            if df.height:
                                prices_out.append(df)
                                progress.log(
                                    f"[green]✓[/green] {tok} prices rows={df.height}"
                                )
                            else:
                                progress.log(f"[yellow]•[/yellow] {tok} empty")
                        except Exception as e:
                            progress.log(f"[red]✗[/red] {tok} error: {e!r}")
                        finally:
                            progress.update(sub, visible=False)
                            progress.advance(t_prices)

                    await asyncio.gather(*[worker(t) for t in token_ids])
                    if prices_out:
                        combined = pl.concat(prices_out, how="vertical")
                        io.storage.write_parquet(
                            combined,
                            out_root / "raw" / "clob" / f"{stamp}_prices.parquet",
                        )
                        progress.log(
                            f"[green]✓[/green] prices wrote {combined.height} rows"
                        )
                    progress.update(t_prices, visible=False)
                else:
                    progress.log(
                        "[yellow]•[/yellow] no token_ids available for prices; skipping"
                    )

            if include_trades:
                progress.log("[cyan]starting trades backfill[/cyan]")
                try:
                    trades_df = await sources.clob.backfill_trades(
                        client, market_ids=None, since_ts=start_ts
                    )
                    if trades_df.height:
                        io.storage.write_parquet(
                            trades_df,
                            out_root / "raw" / "clob" / f"{stamp}_trades.parquet",
                        )
                        progress.log(
                            f"[green]✓[/green] trades rows: {trades_df.height}"
                        )
                    else:
                        progress.log("[yellow]•[/yellow] trades returned 0 rows")
                except Exception as e:
                    progress.log(f"[red]✗[/red] trades fetch failed: {e!r}")

            progress.log("[bold green]Done.[/bold green]")
