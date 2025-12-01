from __future__ import annotations

import asyncio
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import click
import typer
from rich.console import Console
from rich.table import Table

from polymorph import __version__
from polymorph.config import settings
from polymorph.core.base import PipelineContext
from polymorph.pipeline import FetchStage, ProcessStage
from polymorph.remote import deploy_and_run
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
_DEFAULT_DATA_DIR = Path(settings.data_dir)
_DEFAULT_HTTP_TIMEOUT = settings.http_timeout
_DEFAULT_MAX_CONCURRENCY = settings.max_concurrency
_DEFAULT_RAW_DIR = Path(settings.data_dir) / "raw"
_DEFAULT_PROCESSED_DIR = Path(settings.data_dir) / "processed"


def create_context(data_dir: Path) -> PipelineContext:
    return PipelineContext(
        settings=settings,
        run_timestamp=datetime.now(timezone.utc),
        data_dir=data_dir,
    )


@app.callback()
def init(
    data_dir: Path = typer.Option(
        _DEFAULT_DATA_DIR,
        "--data-dir",
        "-d",
        help="Base data directory (overrides POLYMORPH_DATA_DIR)",
        envvar="POLYMORPH_DATA_DIR",
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
        help="HTTP timeout in seconds (overrides POLYMORPH_HTTP_TIMEOUT)",
        envvar="POLYMORPH_HTTP_TIMEOUT",
    ),
    max_concurrency: int = typer.Option(
        _DEFAULT_MAX_CONCURRENCY,
        "--max-concurrency",
        help="Max concurrent HTTP requests (overrides POLYMORPH_MAX_CONCURRENCY)",
        envvar="POLYMORPH_MAX_CONCURRENCY",
    ),
    remote: bool = typer.Option(
        False,
        "--remote",
        "-r",
        help="Execute command on remote server (requires POLYMORPH_REMOTE_* env vars)",
    ),
) -> None:
    if remote:
        args = sys.argv[1:]
        cleaned_args = [arg for arg in args if arg != "--remote"]
        exit_code = deploy_and_run(cleaned_args)
        sys.exit(exit_code)

    level = logging.DEBUG if verbose else logging.INFO
    setup_logging(level=level)
    settings.data_dir = str(data_dir)
    settings.http_timeout = http_timeout
    settings.max_concurrency = max_concurrency
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
    table.add_row("Data dir", settings.data_dir)
    table.add_row("HTTP timeout", str(settings.http_timeout))
    table.add_row("Max concurrency", str(settings.max_concurrency))
    console.print(table)


@app.command(help="Fetch data and store to Parquet files")
def fetch(
    months: int = typer.Option(1, "--months", "-m", help="Number of months to backfill"),
    out: Path = typer.Option(_DEFAULT_DATA_DIR, "--out", help="Root output dir for raw data"),
    include_trades: bool = typer.Option(True, "--trades/--no-trades", help="Include recent trades via Data-API"),
    include_prices: bool = typer.Option(True, "--prices/--no-prices", help="Include prices-history for each token"),
    include_gamma: bool = typer.Option(True, "--gamma/--no-gamma", help="Fetch market metadata from Gamma"),
    max_concurrency: int | None = typer.Option(
        None,
        "--local-max-concurrency",
        help="Override global max concurrency just for this run",
    ),
) -> None:
    console.log(
        f"months={months}, out={out}, gamma={include_gamma}, " f"prices={include_prices}, trades={include_trades}"
    )
    context = create_context(out)
    stage = FetchStage(
        context=context,
        n_months=months,
        include_gamma=include_gamma,
        include_prices=include_prices,
        include_trades=include_trades,
        max_concurrency=max_concurrency,
    )
    asyncio.run(stage.execute())
    console.print("Fetch complete.")


@app.command(help="Processing tools and algorithms (ex. Monte Carlo simulations")
def process(
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
    context = create_context(Path(settings.data_dir))
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
