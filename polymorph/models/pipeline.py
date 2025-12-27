from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

from pydantic import BaseModel


@dataclass
class FilterConfig:
    start_date: date | None = None
    end_date: date | None = None
    resolved_only: bool = False
    unresolved_only: bool = False
    min_age_days: int | None = None
    max_age_days: int | None = None
    categories: list[str] = field(default_factory=list)
    exclude_categories: list[str] = field(default_factory=list)
    market_ids: list[str] = field(default_factory=list)
    exclude_market_ids: list[str] = field(default_factory=list)
    compute_jumps: bool = False
    min_jump_pct: float | None = None
    max_jump_pct: float | None = None
    min_jump_abs: float | None = None
    max_jump_abs: float | None = None


@dataclass
class FilterResult:
    input_path: Path | None = None
    output_path: Path | None = None
    input_count: int = 0
    output_count: int = 0
    filters_applied: list[str] = field(default_factory=list)


@dataclass
class ProcessInputConfig:
    raw_dir: Path | None = None
    run_dir: Path | None = None
    markets_file: Path | None = None
    prices_file: Path | None = None
    trades_file: Path | None = None

    @property
    def mode(self) -> str:
        if self.run_dir is not None:
            return "run_dir"
        if any([self.markets_file, self.prices_file, self.trades_file]):
            return "explicit_files"
        return "scan"


class FetchResult(BaseModel):
    markets_path: Path | None = None
    prices_path: Path | None = None
    trades_path: Path | None = None
    run_timestamp: datetime
    market_count: int = 0
    orderbooks_path: Path | None = None
    orderbook_levels: int = 0
    spreads_path: Path | None = None
    spreads_count: int = 0
    token_count: int = 0
    trade_count: int = 0
    price_point_count: int = 0
    processed_dir: Path | None = None

    model_config = {"arbitrary_types_allowed": True}


class ProcessResult(BaseModel):
    daily_returns_path: Path | None = None
    trades_daily_agg_path: Path | None = None
    prices_joined_path: Path | None = None
    price_panel_path: Path | None = None
    run_timestamp: datetime
    returns_count: int = 0
    trade_agg_count: int = 0
    joined_count: int = 0
    panel_days: int = 0
    panel_tokens: int = 0

    model_config = {"arbitrary_types_allowed": True}
