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
from polymorph.pipeline import FetchStage
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


@app.callback()
def init(
    ctx: typer.Context,
    data_dir: Path = typer.Option(
        _DEFAULT_DATA_DIR,
        "--data-dir",
        "-d",
        help="Base data directory (overrides TOML config for this command)",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
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


@app.command()
def version(ctx: typer.Context) -> None:
    runtime_config = ctx.obj if ctx and ctx.obj else RuntimeConfig()
    temp_context = create_context(_DEFAULT_DATA_DIR, runtime_config=runtime_config)
    effective_config = temp_context.effective_config

    table = Table(title="polymorph")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Version", __version__)
    table.add_row("Data dir", str(temp_context.data_dir))
    table.add_row("HTTP timeout", str(effective_config.http_timeout))
    table.add_row("Max concurrency", str(effective_config.max_concurrency))
    console.print(table)


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


def main() -> None:
    app()


if __name__ == "__main__":
    main()
