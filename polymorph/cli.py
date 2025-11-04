from __future__ import annotations
import asyncio
from pathlib import Path
import typer
from rich.console import Console
from rich.table import Table

from polymorph import pipeline, sims

app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()


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
    asyncio.run(
        pipeline.fetch.last_n_months(
            months, out, include_trades, include_prices, include_gamma, max_concurrency
        )
    )
    console.print("[green]Fetch complete.[/green]")


@app.command()
def process(
    in_: Path = typer.Option(Path("data/raw"), "--in"),
    out: Path = typer.Option(Path("data/processed"), "--out"),
):
    """Build features/aggregations from raw parquet."""
    pipeline.process.build_features(in_, out)
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
    result = sims.monte_carlo.run(market_id, trials, horizon_days, in_)
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
    sims.param_search.run(study, n_trials, in_)


def main():
    app()


if __name__ == "__main__":
    main()
