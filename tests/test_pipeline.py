from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from polymorph.config import config as base_config
from polymorph.core.base import PipelineContext, RuntimeConfig
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

        async def fetch_markets(self, *, resolved_only: bool = False) -> pl.DataFrame:
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

    class DummyClob:
        def __init__(self) -> None:
            self.price_calls: list[str] = []
            self.orderbook_calls: list[str] = []
            self.spread_calls: list[str] = []
            self.trades_called = False

        async def __aenter__(self) -> "DummyClob":
            return self

        async def __aexit__(self, _exc_type, _exc, _tb) -> None:
            return None

        async def fetch_prices_history(
            self,
            token_id: str,
            start_ts: int,
            end_ts: int,
            fidelity: int | None = None,
        ) -> pl.DataFrame:
            self.price_calls.append(token_id)
            return pl.DataFrame(
                {
                    "t": [start_ts, end_ts],
                    "p": ["0.4", "0.6"],  # Prices are strings per API spec
                    "token_id": [token_id, token_id],
                }
            )

        async def fetch_orderbooks(self, token_ids: list[str]) -> pl.DataFrame:
            """Fetch orderbooks for multiple tokens (used by FetchStage)."""
            rows = []
            for token_id in token_ids:
                self.orderbook_calls.append(token_id)
                rows.extend(
                    [
                        {"token_id": token_id, "timestamp": 123, "side": "bid", "price": "0.4", "size": "10.0"},
                        {"token_id": token_id, "timestamp": 123, "side": "ask", "price": "0.6", "size": "5.0"},
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

        async def fetch_trades(self, market_ids=None, since_ts: int | None = None) -> pl.DataFrame:
            self.trades_called = True
            base_ts = since_ts or 1704067200000  # Valid timestamp in milliseconds
            return pl.DataFrame(
                {
                    "timestamp": [base_ts, base_ts + 86_400_000],  # Milliseconds
                    "size": ["1.0", "2.0"],  # Strings per API spec
                    "price": ["0.4", "0.6"],  # Strings per API spec
                    "conditionId": ["c1", "c1"],
                }
            )

    fetch_stage = FetchStage(
        context,
        n_months=1,
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

    process_stage = ProcessStage(context)
    process_result = await process_stage.execute(fetch_result)

    assert process_result.daily_returns_path is not None
    assert process_result.daily_returns_path.exists()
    assert process_result.trades_daily_agg_path is not None
    assert process_result.trades_daily_agg_path.exists()
    assert process_result.returns_count > 0
    assert process_result.trade_agg_count > 0
