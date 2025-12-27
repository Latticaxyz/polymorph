from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from polymorph import __version__
from polymorph.config import config
from polymorph.core.base import PipelineContext, RuntimeConfig
from polymorph.models.pipeline import FilterConfig, FilterResult, ProcessInputConfig, ProcessResult
from polymorph.pipeline import FetchStage, FilterStage, ProcessStage
from polymorph.utils.logging import setup as setup_logging
from polymorph.utils.paths import unique_path
from polymorph.utils.schema import (
    DiscoveredFiles,
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

_TRUTHY_VALUES = frozenset(("true", "1", "yes", "on"))


def _format_time_period(minutes: int, hours: int, days: int, weeks: int, months: int, years: int) -> str:
    if minutes > 0:
        return f"{minutes} minutes"
    if hours > 0:
        return f"{hours} hours"
    if days > 0:
        return f"{days} days"
    if weeks > 0:
        return f"{weeks} weeks"
    if months > 0:
        return f"{months} months"
    if years > 0:
        return f"{years} years"
    return "1 month (default)"


def _parse_bool_option(value: str) -> bool:
    return value.lower() in _TRUTHY_VALUES


def _validate_inputs(
    do_join: bool,
    do_returns: bool,
    do_panel: bool,
    do_trades: bool,
    discovered: DiscoveredFiles | None = None,
    markets_file: Path | None = None,
    prices_file: Path | None = None,
    trades_file: Path | None = None,
) -> list[str]:
    errors: list[str] = []
    needs_markets = do_join
    needs_prices = do_join or do_returns or do_panel
    needs_trades = do_trades

    if discovered is not None:
        if needs_markets and not discovered.has_markets():
            errors.append("No markets file found (needs columns: id, token_ids)")
        if needs_prices and not discovered.has_prices():
            errors.append("No prices file found (needs columns: token_id + t/p or timestamp/price)")
        if needs_trades and not discovered.has_trades():
            errors.append("No trades file found (needs columns: timestamp, size, price, conditionId)")
    else:
        if needs_markets:
            if markets_file is None:
                errors.append("--markets required (for --join)")
            elif not markets_file.exists():
                errors.append(f"File not found: {markets_file}")
            else:
                valid, msg = validate_markets_schema(markets_file)
                if not valid:
                    errors.append(f"Invalid markets file: {msg}")

        if needs_prices:
            if prices_file is None:
                errors.append("--prices required (for --join/--returns/--panel)")
            elif not prices_file.exists():
                errors.append(f"File not found: {prices_file}")
            else:
                valid, msg = validate_prices_schema(prices_file)
                if not valid:
                    errors.append(f"Invalid prices file: {msg}")

        if needs_trades:
            if trades_file is None:
                errors.append("--trades-file required (for --trades)")
            elif not trades_file.exists():
                errors.append(f"File not found: {trades_file}")
            else:
                valid, msg = validate_trades_schema(trades_file)
                if not valid:
                    errors.append(f"Invalid trades file: {msg}")

    return list(dict.fromkeys(errors))


def _build_results_table(result: ProcessResult) -> Table:
    table = Table(title="Process Results")
    table.add_column("Output")
    table.add_column("Path")
    table.add_column("Count")

    if result.prices_joined_path:
        table.add_row("Joined Prices", str(result.prices_joined_path), str(result.joined_count))
    if result.daily_returns_path:
        table.add_row("Daily Returns", str(result.daily_returns_path), str(result.returns_count))
    if result.price_panel_path:
        table.add_row(
            "Price Panel", str(result.price_panel_path), f"{result.panel_days} days x {result.panel_tokens} tokens"
        )
    if result.trades_daily_agg_path:
        table.add_row("Trade Aggregates", str(result.trades_daily_agg_path), str(result.trade_agg_count))

    return table


def _build_filter_results_table(result: FilterResult) -> Table:
    table = Table(title="Filter Results")
    table.add_column("Metric")
    table.add_column("Value")

    table.add_row("Input", str(result.input_path) if result.input_path else "-")
    table.add_row("Output", str(result.output_path) if result.output_path else "-")
    table.add_row("Input rows", str(result.input_count))
    table.add_row("Output rows", str(result.output_count))
    table.add_row("Rows filtered", str(result.input_count - result.output_count))

    if result.filters_applied:
        table.add_row("Filters applied", ", ".join(result.filters_applied))

    return table


def _parse_date(date_str: str) -> datetime:
    return datetime.strptime(date_str, "%Y-%m-%d")


def _resolve_input_config(
    run_dir: Path | None,
    data_dir: Path,
    markets_file: Path | None,
    prices_file: Path | None,
    trades_input_file: Path | None,
    do_join: bool,
    do_returns: bool,
    do_panel: bool,
    do_trades: bool,
    has_run_dir: bool,
) -> ProcessInputConfig:
    if has_run_dir:
        assert run_dir is not None
        resolved_run_dir = run_dir if run_dir.is_absolute() else data_dir / run_dir
        if not resolved_run_dir.exists():
            console.print(f"[red]Error: Directory not found: {resolved_run_dir}[/red]")
            raise typer.Exit(1)

        discovered = discover_files(resolved_run_dir)

        if discovered.unknown:
            console.print(f"[yellow]Unrecognized files (skipped): {[f.name for f in discovered.unknown]}[/yellow]")

        errors = _validate_inputs(do_join, do_returns, do_panel, do_trades, discovered=discovered)
        if errors:
            console.print(f"[red]Missing required files in {resolved_run_dir}:[/red]")
            for err in errors:
                console.print(f"  [red]* {err}[/red]")
            raise typer.Exit(1)

        if discovered.has_markets():
            console.print(f"  Markets: {discovered.markets.name}")  # type: ignore[union-attr]
        if discovered.has_prices():
            console.print(f"  Prices:  {[p.name for p in discovered.prices]}")
        if discovered.has_trades():
            console.print(f"  Trades:  {discovered.trades.name}")  # type: ignore[union-attr]

        return ProcessInputConfig(
            markets_file=discovered.markets,
            prices_file=discovered.prices[0] if discovered.prices else None,
            trades_file=discovered.trades,
        )

    errors = _validate_inputs(
        do_join,
        do_returns,
        do_panel,
        do_trades,
        markets_file=markets_file,
        prices_file=prices_file,
        trades_file=trades_input_file,
    )
    if errors:
        console.print("[red]Input validation failed:[/red]")
        for err in errors:
            console.print(f"  [red]* {err}[/red]")
        raise typer.Exit(1)

    return ProcessInputConfig(
        markets_file=markets_file,
        prices_file=prices_file,
        trades_file=trades_input_file,
    )


def _execute_processing_steps(
    stage: ProcessStage,
    context: PipelineContext,
    do_join: bool,
    do_returns: bool,
    do_panel: bool,
    do_trades: bool,
) -> ProcessResult:
    result = ProcessResult(run_timestamp=context.run_timestamp)

    if do_join:
        r = stage.build_joined_prices()
        result.prices_joined_path = r.prices_joined_path
        result.joined_count = r.joined_count

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

    return result


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
    no_cache: bool = typer.Option(
        False,
        "--no-cache",
        help="Disable Gamma market cache (force fresh fetch from API)",
    ),
) -> None:
    time_params = [minutes, hours, days, weeks, months, years]
    time_param_count = sum(1 for p in time_params if p > 0)

    if time_param_count > 1:
        console.print("[red]Error: Only one time period parameter can be specified at a time.[/red]")
        raise typer.Exit(1)

    if time_param_count == 0:
        months = 1

    time_period_str = _format_time_period(minutes, hours, days, weeks, months, years)

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
        f"gamma_max_pages={gamma_max_pages}, cache={not no_cache}"
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
        use_gamma_cache=not no_cache,
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
    join: str = typer.Option(
        "true",
        "--join",
        help="Build joined prices (true/false)",
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
    do_join = _parse_bool_option(join)
    do_returns = _parse_bool_option(returns)
    do_panel = _parse_bool_option(panel)
    do_trades = _parse_bool_option(trades)

    runtime_config = ctx.obj if ctx and ctx.obj else RuntimeConfig()
    data_dir = Path(runtime_config.data_dir) if runtime_config.data_dir else _DEFAULT_DATA_DIR
    context = create_context(data_dir, runtime_config=runtime_config)

    has_run_dir = run_dir is not None
    has_file_paths = any([markets_file, prices_file, trades_input_file])

    if not has_run_dir and not has_file_paths:
        console.print("[red]Error: Must specify either --dir or explicit file paths[/red]")
        console.print("\nExamples:")
        console.print("  polymorph process --dir raw/swift-falcon-1227")
        console.print("  polymorph process --markets my_markets.parquet --prices my_prices.parquet")
        raise typer.Exit(1)

    if has_run_dir and has_file_paths:
        console.print("[red]Error: --dir cannot be combined with --markets, --prices, or --trades-file[/red]")
        raise typer.Exit(1)

    input_config = _resolve_input_config(
        run_dir=run_dir,
        data_dir=data_dir,
        markets_file=markets_file,
        prices_file=prices_file,
        trades_input_file=trades_input_file,
        do_join=do_join,
        do_returns=do_returns,
        do_panel=do_panel,
        do_trades=do_trades,
        has_run_dir=has_run_dir,
    )

    console.log(f"Processing: mode={input_config.mode}, out={out or 'data/processed'}")

    stage = ProcessStage(
        context=context,
        processed_dir=out,
        input_config=input_config,
    )

    result = _execute_processing_steps(stage, context, do_join, do_returns, do_panel, do_trades)

    console.print(_build_results_table(result))
    console.print("Process complete.")


@app.command(help="Filter joined data by various criteria")
def filter(
    ctx: typer.Context,
    input_file: Path = typer.Argument(..., help="Path to joined parquet"),
    out: Path = typer.Option(
        None,
        "--out",
        "-o",
        help="Output file path (default: INPUT_FILE directory + prices_filtered.parquet)",
    ),
    start_date: str = typer.Option(
        None,
        "--start-date",
        help="Include prices on or after this date (YYYY-MM-DD format)",
        rich_help_panel="Date Filtering",
    ),
    end_date: str = typer.Option(
        None,
        "--end-date",
        help="Include prices on or before this date (YYYY-MM-DD format)",
        rich_help_panel="Date Filtering",
    ),
    resolved_only: bool = typer.Option(
        False,
        "--resolved-only",
        help="Only include prices from resolved markets",
        rich_help_panel="Resolution Filtering",
    ),
    unresolved_only: bool = typer.Option(
        False,
        "--unresolved-only",
        help="Only include prices from unresolved markets",
        rich_help_panel="Resolution Filtering",
    ),
    min_age_days: int = typer.Option(
        None,
        "--min-age-days",
        help="Only markets at least N days old",
        rich_help_panel="Market Age Filtering",
    ),
    max_age_days: int = typer.Option(
        None,
        "--max-age-days",
        help="Only markets at most N days old",
        rich_help_panel="Market Age Filtering",
    ),
    category: list[str] = typer.Option(
        [],
        "--category",
        help="Include category (can be repeated)",
        rich_help_panel="Category Filtering",
    ),
    exclude_category: list[str] = typer.Option(
        [],
        "--exclude-category",
        help="Exclude category (can be repeated)",
        rich_help_panel="Category Filtering",
    ),
    market_id: list[str] = typer.Option(
        [],
        "--market-id",
        help="Include specific market ID (can be repeated)",
        rich_help_panel="Market Selection",
    ),
    exclude_market: list[str] = typer.Option(
        [],
        "--exclude-market",
        help="Exclude specific market ID (can be repeated)",
        rich_help_panel="Market Selection",
    ),
    compute_jumps: bool = typer.Option(
        False,
        "--compute-jumps",
        help="Compute and append jump_abs and jump_pct columns",
        rich_help_panel="Jump Filtering",
    ),
    min_jump_pct: float = typer.Option(
        None,
        "--min-jump-pct",
        help="Keep rows where |jump_pct| >= threshold (e.g., 0.05 for 5%)",
        rich_help_panel="Jump Filtering",
    ),
    max_jump_pct: float = typer.Option(
        None,
        "--max-jump-pct",
        help="Keep rows where |jump_pct| <= threshold (filter outliers)",
        rich_help_panel="Jump Filtering",
    ),
    min_jump_abs: float = typer.Option(
        None,
        "--min-jump-abs",
        help="Keep rows where |jump_abs| >= threshold",
        rich_help_panel="Jump Filtering",
    ),
    max_jump_abs: float = typer.Option(
        None,
        "--max-jump-abs",
        help="Keep rows where |jump_abs| <= threshold",
        rich_help_panel="Jump Filtering",
    ),
) -> None:
    if not input_file.exists():
        console.print(f"[red]Error: Input file not found: {input_file}[/red]")
        raise typer.Exit(1)

    if resolved_only and unresolved_only:
        console.print("[red]Error: --resolved-only and --unresolved-only are mutually exclusive[/red]")
        raise typer.Exit(1)

    resolved_input = input_file.resolve()
    base_output = out.resolve() if out is not None else resolved_input.parent / "prices_filtered.parquet"
    output_file = unique_path(base_output)

    parsed_start_date = _parse_date(start_date).date() if start_date else None
    parsed_end_date = _parse_date(end_date).date() if end_date else None

    filter_config = FilterConfig(
        start_date=parsed_start_date,
        end_date=parsed_end_date,
        resolved_only=resolved_only,
        unresolved_only=unresolved_only,
        min_age_days=min_age_days,
        max_age_days=max_age_days,
        categories=list(category),
        exclude_categories=list(exclude_category),
        market_ids=list(market_id),
        exclude_market_ids=list(exclude_market),
        compute_jumps=compute_jumps,
        min_jump_pct=min_jump_pct,
        max_jump_pct=max_jump_pct,
        min_jump_abs=min_jump_abs,
        max_jump_abs=max_jump_abs,
    )

    runtime_config = ctx.obj if ctx and ctx.obj else RuntimeConfig()
    data_dir = Path(runtime_config.data_dir) if runtime_config.data_dir else _DEFAULT_DATA_DIR
    context = create_context(data_dir, runtime_config=runtime_config)

    stage = FilterStage(
        context=context,
        filter_config=filter_config,
        input_path=resolved_input,
        output_path=output_file,
    )

    result = asyncio.run(stage.execute())

    console.print(_build_filter_results_table(result))
    console.print("Filter complete.")


def main() -> None:
    try:
        app(standalone_mode=False)
    except typer.Exit as e:
        raise SystemExit(e.exit_code) from None
    except typer.Abort:
        console.print("[red]Aborted.[/red]")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
