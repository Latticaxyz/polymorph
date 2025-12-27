from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from polymorph.config import config as base_config
from polymorph.core.base import PipelineContext, RuntimeConfig
from polymorph.models.api import OrderBook, OrderBookLevel
from polymorph.models.pipeline import FetchResult
from polymorph.pipeline.fetch import FetchStage
from polymorph.pipeline.process import ProcessStage
from polymorph.utils.time import utc


def _make_context(tmp_path: Path) -> PipelineContext:
    runtime_cfg = RuntimeConfig(http_timeout=None, max_concurrency=None, data_dir=str(tmp_path))
    return PipelineContext(
        config=base_config,
        run_timestamp=utc(),
        data_dir=tmp_path,
        runtime_config=runtime_cfg,
    )


@pytest.mark.asyncio
async def test_fetch_and_process_pipeline_with_fake_sources(tmp_path: Path) -> None:
    context = _make_context(tmp_path)

    class DummyGamma:
        async def __aenter__(self) -> "DummyGamma":
            return self

        async def __aexit__(self, _exc_type, _exc, _tb) -> None:
            return None

        async def fetch_markets(
            self,
            *,
            resolved_only: bool = False,
            start_ts: int | None = None,
            end_ts: int | None = None,
            on_progress: object = None,
        ) -> pl.DataFrame:
            return pl.DataFrame(
                {
                    "id": ["m1", "m2"],
                    "question": ["Market 1", "Market 2"],
                    "description": [None, None],
                    "market_slug": [None, None],
                    "condition_id": ["c1", "c2"],
                    "token_ids": [["YES1", "NO1"], ["YES2", "NO2"]],
                    "outcomes": [["YES", "NO"], ["YES", "NO"]],
                    "active": [True, False],
                    "closed": [False, True],
                    "archived": [False, False],
                    "created_at": [None, None],
                    "end_date": [None, None],
                    "resolved": [False, True],
                    "resolution_date": [None, None],
                    "resolution_outcome": [None, "YES"],
                    "tags": [[], []],
                    "category": [None, None],
                }
            )

    class DummyRateLimiter:
        def get_rps(self, window_seconds: float = 10.0) -> float:
            return 0.0

    class DummyClob:
        def __init__(self) -> None:
            self.price_calls: list[str] = []
            self.price_params: list[dict[str, int | str | None]] = []
            self.orderbook_calls: list[str] = []
            self.spread_calls: list[str] = []
            self.trades_called = False
            self._rate_limiter = DummyRateLimiter()

        async def __aenter__(self) -> "DummyClob":
            return self

        async def __aexit__(self, _exc_type, _exc, _tb) -> None:
            return None

        async def _get_clob_rate_limiter(self) -> DummyRateLimiter:
            return self._rate_limiter

        async def fetch_prices_history(
            self,
            token_id: str,
            start_ts: int | None = None,
            end_ts: int | None = None,
            interval: str | None = None,
            fidelity: int | None = None,
            cache: object = None,
            created_at_ts: int | None = None,
        ) -> pl.DataFrame:
            self.price_calls.append(token_id)
            self.price_params.append(
                {
                    "start_ts": start_ts,
                    "end_ts": end_ts,
                    "interval": interval,
                }
            )
            return pl.DataFrame(
                {
                    "t": [1704067200000, 1704153600000],
                    "p": [0.4, 0.6],
                    "token_id": [token_id, token_id],
                }
            )

        async def fetch_orderbook(self, token_id: str) -> OrderBook:
            """Fetch orderbook for a single token."""
            self.orderbook_calls.append(token_id)
            return OrderBook(
                token_id=token_id,
                timestamp=1704067200000,
                bids=[OrderBookLevel(price=0.4, size=10.0)],
                asks=[OrderBookLevel(price=0.6, size=5.0)],
                best_bid=0.4,
                best_ask=0.6,
                mid_price=0.5,
                spread=0.2,
            )

        async def fetch_orderbooks(self, token_ids: list[str]) -> pl.DataFrame:
            """Fetch orderbooks for multiple tokens (used by FetchStage)."""
            rows = []
            for token_id in token_ids:
                self.orderbook_calls.append(token_id)
                rows.extend(
                    [
                        {"token_id": token_id, "timestamp": 123, "side": "bid", "price": 0.4, "size": 10.0},
                        {"token_id": token_id, "timestamp": 123, "side": "ask", "price": 0.6, "size": 5.0},
                    ]
                )
            return pl.DataFrame(rows)

        async def fetch_spread(self, token_id: str) -> dict[str, str | float | int | None]:
            self.spread_calls.append(token_id)
            return {
                "token_id": token_id,
                "bid": 0.4,
                "ask": 0.6,
                "mid": 0.5,
                "spread": 0.2,
                "timestamp": 123,
            }

        async def fetch_trades(
            self, market_ids: object = None, since_ts: int | None = None, on_progress: object = None
        ) -> pl.DataFrame:
            self.trades_called = True
            base_ts = since_ts or 1704067200000
            return pl.DataFrame(
                {
                    "timestamp": [base_ts, base_ts + 86_400_000],
                    "size": [1.0, 2.0],
                    "price": [0.4, 0.6],
                    "conditionId": ["c1", "c1"],
                }
            )

    fetch_stage = FetchStage(
        context,
        months=1,
        include_gamma=True,
        include_prices=True,
        include_trades=True,
        include_orderbooks=True,
        include_spreads=True,
        resolved_only=False,
        max_concurrency=4,
    )
    # Override with dummy sources to avoid real API calls
    fetch_stage.gamma = DummyGamma()  # type: ignore[assignment]
    clob = DummyClob()
    fetch_stage.clob = clob  # type: ignore[assignment]

    fetch_result: FetchResult = await fetch_stage.execute(None)

    assert fetch_result.market_count == 2
    assert fetch_result.token_count == 4

    assert fetch_result.markets_path is not None and fetch_result.markets_path.exists()
    assert fetch_result.prices_path is not None and fetch_result.prices_path.exists()
    assert fetch_result.trades_path is not None and fetch_result.trades_path.exists()
    assert fetch_result.orderbooks_path is not None and fetch_result.orderbooks_path.exists()
    assert fetch_result.spreads_path is not None and fetch_result.spreads_path.exists()

    assert fetch_result.price_point_count > 0
    assert fetch_result.trade_count > 0
    assert fetch_result.orderbook_levels > 0
    assert fetch_result.spreads_count == 4

    assert sorted(clob.price_calls) == ["NO1", "NO2", "YES1", "YES2"]
    assert sorted(clob.orderbook_calls) == ["NO1", "NO2", "YES1", "YES2"]
    assert sorted(clob.spread_calls) == ["NO1", "NO2", "YES1", "YES2"]
    assert clob.trades_called

    for params in clob.price_params:
        assert params["start_ts"] is not None, "FetchStage must pass start_ts, not interval"
        assert params["end_ts"] is not None, "FetchStage must pass end_ts, not interval"
        assert params["interval"] is None, "FetchStage must not use interval parameter"

    process_stage = ProcessStage(context)
    process_result = await process_stage.execute(fetch_result)

    assert process_result.prices_joined_path is not None
    assert process_result.prices_joined_path.exists()
    assert process_result.joined_count > 0

    joined_df = pl.read_parquet(process_result.prices_joined_path)
    assert "market_id" in joined_df.columns
    assert "outcome_name" in joined_df.columns
    assert "question" in joined_df.columns

    assert process_result.daily_returns_path is not None
    assert process_result.daily_returns_path.exists()
    assert process_result.returns_count > 0

    returns_df = pl.read_parquet(process_result.daily_returns_path)
    assert "ret" in returns_df.columns
    assert "market_id" in returns_df.columns

    assert process_result.price_panel_path is not None
    assert process_result.price_panel_path.exists()
    assert process_result.panel_days > 0
    assert process_result.panel_tokens > 0

    panel_df = pl.read_parquet(process_result.price_panel_path)
    assert "day_ts" in panel_df.columns
    assert panel_df.width > 1

    assert process_result.trades_daily_agg_path is not None
    assert process_result.trades_daily_agg_path.exists()
    assert process_result.trade_agg_count > 0
