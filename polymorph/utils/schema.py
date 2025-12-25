from pathlib import Path

import polars as pl

MARKETS_REQUIRED_COLS = {"id", "token_ids"}
PRICES_REQUIRED_COLS_V1 = {"token_id", "t", "p"}
PRICES_REQUIRED_COLS_V2 = {"token_id", "timestamp", "price"}
TRADES_REQUIRED_COLS = {"timestamp", "size", "price", "conditionId"}


def get_parquet_columns(path: Path) -> set[str]:
    try:
        schema = pl.read_parquet_schema(path)
        return set(schema.keys())
    except Exception:
        return set()


def is_markets_file(path: Path) -> bool:
    cols = get_parquet_columns(path)
    return MARKETS_REQUIRED_COLS.issubset(cols)


def is_prices_file(path: Path) -> bool:
    cols = get_parquet_columns(path)
    return PRICES_REQUIRED_COLS_V1.issubset(cols) or PRICES_REQUIRED_COLS_V2.issubset(cols)


def is_trades_file(path: Path) -> bool:
    cols = get_parquet_columns(path)
    return TRADES_REQUIRED_COLS.issubset(cols)


def validate_markets_schema(path: Path) -> tuple[bool, str]:
    cols = get_parquet_columns(path)
    if not cols:
        return False, f"Could not read schema from {path}"
    missing = MARKETS_REQUIRED_COLS - cols
    if missing:
        return False, f"Missing required columns: {missing}"
    return True, ""


def validate_prices_schema(path: Path) -> tuple[bool, str]:
    cols = get_parquet_columns(path)
    if not cols:
        return False, f"Could not read schema from {path}"
    if PRICES_REQUIRED_COLS_V1.issubset(cols) or PRICES_REQUIRED_COLS_V2.issubset(cols):
        return True, ""
    return False, f"Missing required columns. Need {PRICES_REQUIRED_COLS_V1} or {PRICES_REQUIRED_COLS_V2}, got {cols}"


def validate_trades_schema(path: Path) -> tuple[bool, str]:
    cols = get_parquet_columns(path)
    if not cols:
        return False, f"Could not read schema from {path}"
    missing = TRADES_REQUIRED_COLS - cols
    if missing:
        return False, f"Missing required columns: {missing}"
    return True, ""


class DiscoveredFiles:
    def __init__(self) -> None:
        self.markets: Path | None = None
        self.prices: list[Path] = []
        self.trades: Path | None = None
        self.unknown: list[Path] = []

    def has_markets(self) -> bool:
        return self.markets is not None

    def has_prices(self) -> bool:
        return len(self.prices) > 0

    def has_trades(self) -> bool:
        return self.trades is not None


def discover_files(directory: Path) -> DiscoveredFiles:
    result = DiscoveredFiles()

    parquet_files = sorted(directory.glob("*.parquet"))

    for f in parquet_files:
        if is_markets_file(f):
            if result.markets is None:
                result.markets = f
        elif is_prices_file(f):
            result.prices.append(f)
        elif is_trades_file(f):
            if result.trades is None:
                result.trades = f
        else:
            result.unknown.append(f)

    return result
