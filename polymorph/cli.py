from __future__ import annotations
import asyncio
from datetime import datetime, timezone
from pathlib import Path
import typer
from rich.console import Console
from rich.table import Table

from polymorph.config import settings
from polymorph.core.base import PipelineContext
from polymorph.pipeline import FetchStage, ProcessStage
from polymorph.sims import MonteCarloSimulator, ParameterSearcher

app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()


def create_context(data_dir: Path) -> PipelineContext:
    """Create a pipeline context.

    Args:
        data_dir: Data directory path

    Returns:
        PipelineContext instance
    """
    return PipelineContext(
        settings=settings,
        run_timestamp=datetime.now(timezone.utc),
        data_dir=data_dir,
    )


@app.command()
def fetch(
    months: int = typer.Option(
        1, "--months", "-m", help="Number of months to backfill"
    ),
    out: Path = typer.Option(Path("data"), "--out", help="Root output dir"),
    include_trades: bool = typer.Option(
        True, help="Also pull recent trades via Data-API"
    ),
    include_prices: bool = typer.Option(
        True, help="Pull prices-history for each token"
    ),
    include_gamma: bool = typer.Option(True, help="Fetch market metadata from Gamma"),
    max_concurrency: int = typer.Option(None, help="Override default concurrency"),
):
    """Fetch last N months of Polymarket data into partitioned Parquet files."""
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
    console.print("[green]Fetch complete.[/green]")


@app.command()
def process(
    in_: Path = typer.Option(Path("data/raw"), "--in"),
    out: Path = typer.Option(Path("data/processed"), "--out"),
):
    """Build features/aggregations from raw parquet."""
    context = create_context(Path("data"))
    stage = ProcessStage(context=context, raw_dir=in_, processed_dir=out)
    asyncio.run(stage.execute())
    console.print("[green]Processing complete.[/green]")


mc_app = typer.Typer(help="Monte Carlo tooling")
app.add_typer(mc_app, name="mc")


@mc_app.command("run")
def mc_run(
    market_id: str = typer.Option(..., "--market-id"),
    trials: int = typer.Option(10000, "--trials"),
    horizon_days: int = typer.Option(7, "--horizon-days"),
    in_: Path = typer.Option(Path("data/processed"), "--in"),
):
    """Run a simple MC on empirical daily return/jump distribution for a market token."""
    simulator = MonteCarloSimulator(processed_dir=in_)
    result = simulator.run(market_id, trials, horizon_days)

    table = Table(title="Monte Carlo Result")
    table.add_column("Metric")
    table.add_column("Value")
    for k, v in result.items():
        table.add_row(k, f"{v:.6f}" if isinstance(v, float) else str(v))
    console.print(table)


@app.command()
def tune(
    study: str = typer.Option("polymorph", "--study"),
    n_trials: int = typer.Option(20, "--n-trials"),
    in_: Path = typer.Option(Path("data/processed"), "--in"),
):
    """Run Optuna parameter search using a simple lending-PnL objective over processed data."""
    searcher = ParameterSearcher(processed_dir=in_)
    searcher.run(study, n_trials)


def main():
    app()


if __name__ == "__main__":
    main()
