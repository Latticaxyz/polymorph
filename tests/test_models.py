"""Tests for data models."""

from datetime import datetime
from pathlib import Path

import pytest

from polymorph.models.api import Market, Token, PricePoint, Trade
from polymorph.models.pipeline import FetchResult, ProcessResult, AnalysisResult
from polymorph.models.analysis import SimulationResult, OptimizationResult


class TestAPIModels:
    """Test API response models."""

    def test_market_basic(self):
        """Test basic Market model."""
        market = Market(
            id="123",
            question="Will it rain?",
            condition_id="abc",
            clob_token_ids=["token1", "token2"],
        )

        assert market.id == "123"
        assert market.question == "Will it rain?"
        assert market.condition_id == "abc"
        assert market.clob_token_ids == ["token1", "token2"]

    def test_market_normalize_token_ids_list(self):
        """Test Market normalizes token IDs from list."""
        market = Market(clob_token_ids=[1, 2, 3])
        assert market.clob_token_ids == ["1", "2", "3"]

    def test_market_normalize_token_ids_json_string(self):
        """Test Market normalizes token IDs from JSON string."""
        market = Market(clob_token_ids='["token1", "token2"]')
        assert market.clob_token_ids == ["token1", "token2"]

    def test_market_normalize_token_ids_csv_string(self):
        """Test Market normalizes token IDs from CSV string."""
        market = Market(clob_token_ids="token1, token2, token3")
        assert market.clob_token_ids == ["token1", "token2", "token3"]

    def test_token(self):
        """Test Token model."""
        token = Token(tokenId="abc123", outcome="YES")
        assert token.token_id == "abc123"
        assert token.outcome == "YES"

    def test_price_point(self):
        """Test PricePoint model."""
        price = PricePoint(t=1234567890, p=0.65, tokenId="abc")
        assert price.t == 1234567890
        assert price.p == 0.65
        assert price.token_id == "abc"

    def test_trade(self):
        """Test Trade model."""
        trade = Trade(
            id="trade123",
            market="market1",
            assetId="asset1",
            side="BUY",
            size=100.0,
            price=0.5,
        )

        assert trade.id == "trade123"
        assert trade.market == "market1"
        assert trade.asset_id == "asset1"
        assert trade.side == "BUY"
        assert trade.size == 100.0
        assert trade.price == 0.5


class TestPipelineModels:
    """Test pipeline data models."""

    def test_fetch_result(self):
        """Test FetchResult model."""
        now = datetime.now()
        result = FetchResult(
            run_timestamp=now,
            markets_path=Path("data/markets.parquet"),
            market_count=100,
            token_count=200,
        )

        assert result.run_timestamp == now
        assert result.markets_path == Path("data/markets.parquet")
        assert result.market_count == 100
        assert result.token_count == 200
        assert result.metadata == {}

    def test_process_result(self):
        """Test ProcessResult model."""
        now = datetime.now()
        result = ProcessResult(
            run_timestamp=now,
            daily_returns_path=Path("data/returns.parquet"),
            returns_count=1000,
        )

        assert result.run_timestamp == now
        assert result.daily_returns_path == Path("data/returns.parquet")
        assert result.returns_count == 1000

    def test_analysis_result(self):
        """Test AnalysisResult model."""
        now = datetime.now()
        result = AnalysisResult(
            run_timestamp=now,
            simulation_results={"token1": {"median": 1.05}},
            optimization_results={"best_params": {"leverage": 2.0}},
        )

        assert result.run_timestamp == now
        assert result.simulation_results == {"token1": {"median": 1.05}}
        assert result.optimization_results == {"best_params": {"leverage": 2.0}}


class TestAnalysisModels:
    """Test analysis result models."""

    def test_simulation_result(self):
        """Test SimulationResult model."""
        result = SimulationResult(
            token_id="token123",
            n_trials=10000,
            n_days=7,
            median_return=1.05,
            percentile_5=0.95,
            percentile_95=1.15,
            prob_negative=0.3,
        )

        assert result.token_id == "token123"
        assert result.n_trials == 10000
        assert result.n_days == 7
        assert result.median_return == 1.05
        assert result.percentile_5 == 0.95
        assert result.percentile_95 == 1.15
        assert result.prob_negative == 0.3

    def test_optimization_result(self):
        """Test OptimizationResult model."""
        result = OptimizationResult(
            study_name="test_study",
            n_trials=20,
            best_params={"leverage": 2.0, "threshold": 0.01},
            best_value=1.5,
        )

        assert result.study_name == "test_study"
        assert result.n_trials == 20
        assert result.best_params == {"leverage": 2.0, "threshold": 0.01}
        assert result.best_value == 1.5
        assert result.optimization_history == []
