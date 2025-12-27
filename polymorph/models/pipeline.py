from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel


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
    prices_enriched_path: Path | None = None
    price_panel_path: Path | None = None
    run_timestamp: datetime
    returns_count: int = 0
    trade_agg_count: int = 0
    enriched_count: int = 0
    panel_days: int = 0
    panel_tokens: int = 0

    model_config = {"arbitrary_types_allowed": True}
