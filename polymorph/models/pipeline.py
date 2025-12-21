from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


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
    metadata: dict[str, Any] = Field(default_factory=dict)

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
    metadata: dict[str, object] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}


class AnalysisResult(BaseModel):
    simulation_results: dict[str, Any] = Field(default_factory=dict)
    optimization_results: dict[str, Any] = Field(default_factory=dict)
    run_timestamp: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}
