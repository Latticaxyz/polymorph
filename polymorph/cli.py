from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path

import click
import typer
from rich.console import Console
from rich.table import Table

from polymorph import __version__
from polymorph.config import config
from polymorph.core.base import PipelineContext, RuntimeConfig
from polymorph.models.pipeline import ProcessResult
from polymorph.pipeline import FetchStage, ProcessStage
from polymorph.utils.logging import setup as setup_logging

click.Context.formatter_class = click.HelpFormatter

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    rich_markup_mode=None,
    pretty_exceptions_enable=False,
)
console = Console()

_DEFAULT_DATA_DIR = Path(config.general.data_dir)
_DEFAULT_HTTP_TIMEOUT = config.general.http_timeout
_DEFAULT_MAX_CONCURRENCY = config.general.max_concurrency


def create_context(
    data_dir: Path,
    runtime_config: RuntimeConfig | None = None,
) -> PipelineContext:
    return PipelineContext(
        config=config,
        run_timestamp=datetime.now(timezone.utc),
        data_dir=data_dir,
        runtime_config=runtime_config or RuntimeConfig(),
    )


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"polymorph v{__version__}")
        raise typer.Exit()


@app.callback()
def init(
    ctx: typer.Context,
    _version: bool = typer.Option(
        False,
        "--version",
        "-V",
        "-v",
        help="Show version and exit",
        callback=_version_callback,
        is_eager=True,
    ),
    data_dir: Path = typer.Option(
        _DEFAULT_DATA_DIR,
        "--data-dir",
        "-d",
        help="Base data directory (overrides TOML config for this command)",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help="Enable verbose (DEBUG) logging",
    ),
    http_timeout: int = typer.Option(
        _DEFAULT_HTTP_TIMEOUT,
        "--http-timeout",
        help="HTTP timeout in seconds (overrides TOML config for this command)",
    ),
    max_concurrency: int = typer.Option(
        _DEFAULT_MAX_CONCURRENCY,
        "--max-concurrency",
        help="Max concurrent HTTP requests (overrides TOML config for this command)",
    ),
) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    setup_logging(level=level)

    runtime_config = RuntimeConfig(
        http_timeout=http_timeout if http_timeout != _DEFAULT_HTTP_TIMEOUT else None,
        max_concurrency=max_concurrency if max_concurrency != _DEFAULT_MAX_CONCURRENCY else None,
        data_dir=str(data_dir) if data_dir != _DEFAULT_DATA_DIR else None,
    )
    ctx.obj = runtime_config


@app.command(help="Fetch and store Gamma & CLOB API data")
def fetch(
    ctx: typer.Context,
    minutes: int = typer.Option(
        0,
        "--minutes",
        help="Fetch markets active/traded in the past n minutes (mutually exclusive with other time options)",
    ),
    hours: int = typer.Option(
        0,
        "--hours",
        help="Fetch markets active/traded in the past n hours (mutually exclusive with other time options)",
    ),
    days: int = typer.Option(
        0, "--days", help="Fetch markets active/traded in the past n days (mutually exclusive with other time options)"
    ),
    weeks: int = typer.Option(
        0,
        "--weeks",
        help="Fetch markets active/traded in the past n weeks (mutually exclusive with other time options)",
    ),
    months: int = typer.Option(
        0,
        "--months",
        "-m",
        help="Fetch markets active/traded in the past n months (mutually exclusive with other time options)",
    ),
    years: int = typer.Option(
        0,
        "--years",
        help="Fetch markets active/traded in the past n years (mutually exclusive with other time options)",
    ),
    out: Path = typer.Option(_DEFAULT_DATA_DIR, "--out", help="Root output dir for raw data"),
    include_trades: bool = typer.Option(True, "--trades/--no-trades", help="Include recent trades via Data-API"),
    include_prices: bool = typer.Option(True, "--prices/--no-prices", help="Include price history via CLOB"),
    include_gamma: bool = typer.Option(True, "--gamma/--no-gamma", help="Include Gamma markets snapshot"),
    include_orderbooks: bool = typer.Option(
        False, "--orderbooks/--no-orderbooks", help="Include current orderbook snapshots (not historical)"
    ),
    include_spreads: bool = typer.Option(
        False, "--spreads/--no-spreads", help="Include current spread snapshots (not historical)"
    ),
    resolved_only: bool = typer.Option(False, "--resolved-only", help="Gamma: only resolved markets"),
    max_concurrency: int = typer.Option(
        _DEFAULT_MAX_CONCURRENCY,
        "--max-concurrency",
        help="Max concurrent HTTP requests (overrides TOML/config for this command)",
    ),
    gamma_max_pages: int | None = typer.Option(
        None,
        "--gamma-max-pages",
        help="Max pages to fetch from Gamma API (None = unbounded, 100 records per page)",
    ),
) -> None:
    time_params = [minutes, hours, days, weeks, months, years]
    time_param_count = sum(1 for p in time_params if p > 0)

    if time_param_count > 1:
        console.print("[red]Error: Only one time period parameter can be specified at a time.[/red]")
        raise typer.Exit(1)

    if time_param_count == 0:
        months = 1

    time_period_str = (
        f"{minutes} minutes"
        if minutes > 0
        else (
            f"{hours} hours"
            if hours > 0
            else (
                f"{days} days"
                if days > 0
                else (
                    f"{weeks} weeks"
                    if weeks > 0
                    else f"{months} months" if months > 0 else f"{years} years" if years > 0 else "1 month (default)"
                )
            )
        )
    )

    runtime_config = ctx.obj if ctx and ctx.obj else RuntimeConfig()
    if gamma_max_pages is not None:
        runtime_config.gamma_max_pages = gamma_max_pages

    effective_max_concurrency = max_concurrency
    if max_concurrency == _DEFAULT_MAX_CONCURRENCY and runtime_config.max_concurrency is not None:
        effective_max_concurrency = runtime_config.max_concurrency

    console.log(
        f"time_period={time_period_str}, out={out}, gamma={include_gamma}, "
        f"prices={include_prices}, trades={include_trades}, "
        f"order_books={include_orderbooks}, spreads={include_spreads}, "
        f"resolved_only={resolved_only}, max_concurrency={effective_max_concurrency}, "
        f"gamma_max_pages={gamma_max_pages}"
    )

    context = create_context(out, runtime_config=runtime_config)

    stage = FetchStage(
        context=context,
        minutes=minutes,
        hours=hours,
        days=days,
        weeks=weeks,
        months=months,
        years=years,
        include_gamma=include_gamma,
        include_prices=include_prices,
        include_trades=include_trades,
        include_orderbooks=include_orderbooks,
        include_spreads=include_spreads,
        resolved_only=resolved_only,
        max_concurrency=effective_max_concurrency,
    )

    asyncio.run(stage.execute())

    console.print("Fetch complete.")


@app.command(help="Process raw data into analytical formats")
def process(
    ctx: typer.Context,
    raw_dir: Path = typer.Option(
        None,
        "--raw-dir",
        "-r",
        help="Input directory for raw data (default: data/raw)",
    ),
    out: Path = typer.Option(
        None,
        "--out",
        "-o",
        help="Output directory for processed data (default: data/processed)",
    ),
    enriched: bool = typer.Option(True, "--enriched/--no-enriched", help="Build enriched raw prices"),
    returns: bool = typer.Option(True, "--returns/--no-returns", help="Build daily returns"),
    panel: bool = typer.Option(True, "--panel/--no-panel", help="Build wide-format price panel"),
    trades: bool = typer.Option(True, "--trades/--no-trades", help="Build trade aggregates"),
) -> None:
    runtime_config = ctx.obj if ctx and ctx.obj else RuntimeConfig()
    data_dir = Path(runtime_config.data_dir) if runtime_config.data_dir else _DEFAULT_DATA_DIR
    context = create_context(data_dir, runtime_config=runtime_config)

    console.log(
        f"data_dir={data_dir}, raw_dir={raw_dir}, out={out}, "
        f"enriched={enriched}, returns={returns}, panel={panel}, trades={trades}"
    )

    stage = ProcessStage(
        context=context,
        raw_dir=raw_dir,
        processed_dir=out,
    )

    result = ProcessResult(run_timestamp=context.run_timestamp)

    if enriched:
        r = stage.build_enriched_prices()
        result.prices_enriched_path = r.prices_enriched_path
        result.enriched_count = r.enriched_count

    if returns:
        r = stage.build_daily_returns()
        result.daily_returns_path = r.daily_returns_path
        result.returns_count = r.returns_count

    if panel:
        r = stage.build_price_panel()
        result.price_panel_path = r.price_panel_path
        result.panel_days = r.panel_days
        result.panel_tokens = r.panel_tokens

    if trades:
        r = stage.build_trade_aggregates()
        result.trades_daily_agg_path = r.trades_daily_agg_path
        result.trade_agg_count = r.trade_agg_count

    table = Table(title="Process Results")
    table.add_column("Output")
    table.add_column("Path")
    table.add_column("Count")

    if result.prices_enriched_path:
        table.add_row("Enriched Prices", str(result.prices_enriched_path), str(result.enriched_count))
    if result.daily_returns_path:
        table.add_row("Daily Returns", str(result.daily_returns_path), str(result.returns_count))
    if result.price_panel_path:
        table.add_row(
            "Price Panel", str(result.price_panel_path), f"{result.panel_days} days x {result.panel_tokens} tokens"
        )
    if result.trades_daily_agg_path:
        table.add_row("Trade Aggregates", str(result.trades_daily_agg_path), str(result.trade_agg_count))

    console.print(table)
    console.print("Process complete.")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
