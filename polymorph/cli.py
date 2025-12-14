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

    ctx.obj = RuntimeConfig(
        http_timeout=http_timeout if http_timeout != _DEFAULT_HTTP_TIMEOUT else None,
        max_concurrency=max_concurrency if max_concurrency != _DEFAULT_MAX_CONCURRENCY else None,
        data_dir=str(data_dir) if data_dir != _DEFAULT_DATA_DIR else None,
    )

    console.log(
        f"polymorph v{__version__} "
        f"(data_dir={data_dir}, timeout={http_timeout}s, max_concurrency={max_concurrency})"
    )


@app.command()
def version() -> None:
    table = Table(title="polymorph")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Version", __version__)
    table.add_row("Data dir", config.general.data_dir)
    table.add_row("HTTP timeout", str(config.general.http_timeout))
    table.add_row("Max concurrency", str(config.general.max_concurrency))
    console.print(table)


@app.command(help="Fetch and store Gamma & CLOB API data")
def fetch(
    ctx: typer.Context,
    months: int = typer.Option(1, "--months", "-m", help="Number of months to backfill"),
    out: Path = typer.Option(_DEFAULT_DATA_DIR, "--out", help="Root output dir for raw data"),
    include_trades: bool = typer.Option(True, "--trades/--no-trades", help="Include recent trades via Data-API"),
    include_prices: bool = typer.Option(True, "--prices/--no-prices", help="Include price history via CLOB"),
    include_gamma: bool = typer.Option(True, "--gamma/--no-gamma", help="Include Gamma markets snapshot"),
    include_orderbooks: bool = typer.Option(False, "--orderbooks/--no-orderbooks", help="Include orderbooks via CLOB"),
    include_spreads: bool = typer.Option(False, "--spreads/--no-spreads", help="Include spreads via CLOB"),
    resolved_only: bool = typer.Option(False, "--resolved-only", help="Gamma: only resolved markets"),
    max_concurrency: int = typer.Option(
        _DEFAULT_MAX_CONCURRENCY,
        "--max-concurrency",
        help="Max concurrent HTTP requests (overrides TOML/config for this command)",
    ),
) -> None:
    console.log(
        f"months={months}, out={out}, gamma={include_gamma}, "
        f"prices={include_prices}, trades={include_trades}, "
        f"order_books={include_orderbooks}, spreads={include_spreads}, "
        f"resolved_only={resolved_only}"
    )

    runtime_config = ctx.obj if ctx and ctx.obj else RuntimeConfig()
    context = create_context(out, runtime_config=runtime_config)

    stage = FetchStage(
        context=context,
        n_months=months,
        include_gamma=include_gamma,
        include_prices=include_prices,
        include_trades=include_trades,
        include_orderbooks=include_orderbooks,
        include_spreads=include_spreads,
        resolved_only=resolved_only,
        max_concurrency=max_concurrency,
    )

    asyncio.run(stage.execute())
    console.print("Fetch complete.")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
