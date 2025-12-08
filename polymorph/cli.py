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
from polymorph.pipeline import FetchStage, ProcessStage
from polymorph.sims import MonteCarloSimulator, ParameterSearcher
from polymorph.utils.logging import setup as setup_logging

click.Context.formatter_class = click.HelpFormatter


app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    rich_markup_mode=None,
    pretty_exceptions_enable=False,
)
console = Console()

# Module-level defaults to avoid B008 flake8 errors
_DEFAULT_DATA_DIR = Path(config.data_dir)
_DEFAULT_HTTP_TIMEOUT = config.http_timeout
_DEFAULT_MAX_CONCURRENCY = config.max_concurrency
_DEFAULT_RAW_DIR = Path(config.data_dir) / "raw"
_DEFAULT_PROCESSED_DIR = Path(config.data_dir) / "processed"


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
    table.add_row("Data dir", config.data_dir)
    table.add_row("HTTP timeout", str(config.http_timeout))
    table.add_row("Max concurrency", str(config.max_concurrency))
    console.print(table)


@app.command(help="Fetch data and store to Parquet files")
def fetch(
    ctx: typer.Context,
    months: int = typer.Option(1, "--months", "-m", help="Number of months to backfill"),
    out: Path = typer.Option(_DEFAULT_DATA_DIR, "--out", help="Root output dir for raw data"),
    include_trades: bool = typer.Option(True, "--trades/--no-trades", help="Include recent trades via Data-API"),
    include_prices: bool = typer.Option(True, "--prices/--no-prices", help="Include prices-history for each token"),
    include_gamma: bool = typer.Option(True, "--gamma/--no-gamma", help="Fetch market metadata from Gamma"),
    include_orderbooks: bool = typer.Option(
        False, "--orderbooks/--no-orderbooks", help="Fetch real-time order book snapshots for all tokens"
    ),
    include_spreads: bool = typer.Option(
        False,
        "--spreads/--no-spreads",
        help="Fetch bid-ask spread data for all tokens",
    ),
    resolved_only: bool = typer.Option(
        False, "--resolved", help="Fetch only resolved/closed markets (when used with --gamma)"
    ),
    max_concurrency: int | None = typer.Option(
        None,
        "--local-max-concurrency",
        help="Override global max concurrency just for this run",
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


@app.command(help="Processing tools and algorithms (ex. Monte Carlo simulations")
def process(
    ctx: typer.Context,
    in_: Path = typer.Option(
        _DEFAULT_RAW_DIR,
        "--in",
        help="Input directory with raw parquet data",
    ),
    out: Path = typer.Option(
        _DEFAULT_PROCESSED_DIR,
        "--out",
        help="Output directory for processed data",
    ),
) -> None:
    console.log(f"in={in_}, out={out}")
    runtime_config = ctx.obj if ctx and ctx.obj else RuntimeConfig()

    context = create_context(Path(config.data_dir), runtime_config=runtime_config)
    stage = ProcessStage(context=context, raw_dir=in_, processed_dir=out)
    asyncio.run(stage.execute())
    console.print("Processing complete.")


mc_app = typer.Typer(help="Monte Carlo tooling")
app.add_typer(mc_app, name="mc")


@mc_app.command("run")
def mc_run(
    market_id: str = typer.Option(..., "--market-id"),
    trials: int = typer.Option(10000, "--trials"),
    horizon_days: int = typer.Option(7, "--horizon-days"),
    in_: Path = typer.Option(
        _DEFAULT_PROCESSED_DIR,
        "--in",
        help="Processed data directory",
    ),
) -> None:
    simulator = MonteCarloSimulator(processed_dir=in_)
    result = simulator.run(market_id, trials, horizon_days)
    table = Table(title="Monte Carlo Result")
    table.add_column("Metric")
    table.add_column("Value")
    for k, v in result.items():
        table.add_row(k, f"{v:.6f}" if isinstance(v, float) else str(v))
    console.print(table)


@app.command(help="Hyperparameter searchig and tuning (Optima)")
def tune(
    study: str = typer.Option("polymorph", "--study"),
    n_trials: int = typer.Option(20, "--n-trials"),
    in_: Path = typer.Option(
        _DEFAULT_PROCESSED_DIR,
        "--in",
        help="Processed data directory",
    ),
) -> None:
    searcher = ParameterSearcher(processed_dir=in_)
    searcher.run(study, n_trials)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
