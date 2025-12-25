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
from polymorph.models.pipeline import ProcessInputConfig, ProcessResult
from polymorph.pipeline import FetchStage, ProcessStage
from polymorph.utils.logging import setup as setup_logging
from polymorph.utils.schema import (
    discover_files,
    validate_markets_schema,
    validate_prices_schema,
    validate_trades_schema,
)

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    rich_markup_mode="rich",
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
    run_dir: Path = typer.Option(
        None,
        "--dir",
        "-d",
        help="Directory to scan for parquet files",
        rich_help_panel="Input Sources",
    ),
    markets_file: Path = typer.Option(
        None,
        "--markets",
        "-m",
        help="Markets file",
        rich_help_panel="Input Sources",
    ),
    prices_file: Path = typer.Option(
        None,
        "--prices",
        "-p",
        help="Prices file",
        rich_help_panel="Input Sources",
    ),
    trades_input_file: Path = typer.Option(
        None,
        "--trades-file",
        "-t",
        help="Trades file",
        rich_help_panel="Input Sources",
    ),
    out: Path = typer.Option(
        None,
        "--out",
        "-o",
        help="Output directory (default: data/processed)",
        rich_help_panel="Output",
    ),
    enriched: str = typer.Option(
        "true",
        "--enriched",
        help="Build enriched prices (true/false)",
        rich_help_panel="Processing Steps",
    ),
    returns: str = typer.Option(
        "true",
        "--returns",
        help="Build daily returns (true/false)",
        rich_help_panel="Processing Steps",
    ),
    panel: str = typer.Option(
        "true",
        "--panel",
        help="Build price panel (true/false)",
        rich_help_panel="Processing Steps",
    ),
    trades: str = typer.Option(
        "true",
        "--trades",
        help="Build trade aggregates (true/false)",
        rich_help_panel="Processing Steps",
    ),
) -> None:
    do_enriched = enriched.lower() in ("true", "1", "yes", "on")
    do_returns = returns.lower() in ("true", "1", "yes", "on")
    do_panel = panel.lower() in ("true", "1", "yes", "on")
    do_trades = trades.lower() in ("true", "1", "yes", "on")
    runtime_config = ctx.obj if ctx and ctx.obj else RuntimeConfig()
    data_dir = Path(runtime_config.data_dir) if runtime_config.data_dir else _DEFAULT_DATA_DIR
    context = create_context(data_dir, runtime_config=runtime_config)

    has_run_dir = run_dir is not None
    has_file_paths = any([markets_file, prices_file, trades_input_file])

    if not has_run_dir and not has_file_paths:
        console.print("[red]Error: Must specify either --dir or explicit file paths[/red]")
        console.print("\nExamples:")
        console.print("  polymorph process --dir raw/20251223T201757Z")
        console.print("  polymorph process --markets my_markets.parquet --prices my_prices.parquet")
        raise typer.Exit(1)

    if has_run_dir and has_file_paths:
        console.print("[red]Error: --dir cannot be combined with --markets, --prices, or --trades-file[/red]")
        raise typer.Exit(1)

    errors: list[str] = []
    discovered_markets: Path | None = None
    discovered_prices: list[Path] = []
    discovered_trades: Path | None = None

    if has_run_dir:
        resolved_run_dir = run_dir if run_dir.is_absolute() else data_dir / run_dir
        if not resolved_run_dir.exists():
            console.print(f"[red]Error: Directory not found: {resolved_run_dir}[/red]")
            raise typer.Exit(1)

        discovered = discover_files(resolved_run_dir)
        discovered_markets = discovered.markets
        discovered_prices = discovered.prices
        discovered_trades = discovered.trades

        if discovered.unknown:
            console.print(f"[yellow]Unrecognized files (skipped): {[f.name for f in discovered.unknown]}[/yellow]")

        if do_enriched and not discovered.has_markets():
            errors.append("No markets file found (needs columns: id, token_ids)")
        if do_enriched and not discovered.has_prices():
            errors.append("No prices file found (needs columns: token_id + t/p or timestamp/price)")
        if do_returns and not discovered.has_prices():
            errors.append("No prices file found (needs columns: token_id + t/p or timestamp/price)")
        if do_panel and not discovered.has_prices():
            errors.append("No prices file found (needs columns: token_id + t/p or timestamp/price)")
        if do_trades and not discovered.has_trades():
            errors.append("No trades file found (needs columns: timestamp, size, price, conditionId)")

        seen_errors: set[str] = set()
        unique_errors = []
        for e in errors:
            if e not in seen_errors:
                seen_errors.add(e)
                unique_errors.append(e)
        errors = unique_errors

        if errors:
            console.print(f"[red]Missing required files in {resolved_run_dir}:[/red]")
            for err in errors:
                console.print(f"  [red]• {err}[/red]")
            raise typer.Exit(1)

        if discovered.has_markets():
            console.print(f"  Markets: {discovered.markets.name}")  # type: ignore[union-attr]
        if discovered.has_prices():
            console.print(f"  Prices:  {[p.name for p in discovered.prices]}")
        if discovered.has_trades():
            console.print(f"  Trades:  {discovered.trades.name}")  # type: ignore[union-attr]

    if has_file_paths:
        needs_markets = do_enriched
        needs_prices = do_enriched or do_returns or do_panel
        needs_trades = do_trades

        if needs_markets:
            if markets_file is None:
                errors.append("--markets required (for --enriched)")
            elif not markets_file.exists():
                errors.append(f"File not found: {markets_file}")
            else:
                valid, msg = validate_markets_schema(markets_file)
                if not valid:
                    errors.append(f"Invalid markets file: {msg}")

        if needs_prices:
            if prices_file is None:
                errors.append("--prices required (for --enriched/--returns/--panel)")
            elif not prices_file.exists():
                errors.append(f"File not found: {prices_file}")
            else:
                valid, msg = validate_prices_schema(prices_file)
                if not valid:
                    errors.append(f"Invalid prices file: {msg}")

        if needs_trades:
            if trades_input_file is None:
                errors.append("--trades-file required (for --trades)")
            elif not trades_input_file.exists():
                errors.append(f"File not found: {trades_input_file}")
            else:
                valid, msg = validate_trades_schema(trades_input_file)
                if not valid:
                    errors.append(f"Invalid trades file: {msg}")

        if errors:
            console.print("[red]Input validation failed:[/red]")
            for err in errors:
                console.print(f"  [red]• {err}[/red]")
            raise typer.Exit(1)

    if has_run_dir:
        input_config = ProcessInputConfig(
            markets_file=discovered_markets,
            prices_file=discovered_prices[0] if discovered_prices else None,
            trades_file=discovered_trades,
        )
    else:
        input_config = ProcessInputConfig(
            markets_file=markets_file,
            prices_file=prices_file,
            trades_file=trades_input_file,
        )

    console.log(f"Processing: mode={input_config.mode}, out={out or 'data/processed'}")

    stage = ProcessStage(
        context=context,
        processed_dir=out,
        input_config=input_config,
    )

    result = ProcessResult(run_timestamp=context.run_timestamp)

    if do_enriched:
        r = stage.build_enriched_prices()
        result.prices_enriched_path = r.prices_enriched_path
        result.enriched_count = r.enriched_count

    if do_returns:
        r = stage.build_daily_returns()
        result.daily_returns_path = r.daily_returns_path
        result.returns_count = r.returns_count

    if do_panel:
        r = stage.build_price_panel()
        result.price_panel_path = r.price_panel_path
        result.panel_days = r.panel_days
        result.panel_tokens = r.panel_tokens

    if do_trades:
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
    try:
        app(standalone_mode=False)
    except click.UsageError as e:
        console.print(f"[red]Error: {e.format_message()}[/red]")
        raise SystemExit(1) from None
    except click.exceptions.Exit as e:
        raise SystemExit(e.exit_code) from None


if __name__ == "__main__":
    main()
