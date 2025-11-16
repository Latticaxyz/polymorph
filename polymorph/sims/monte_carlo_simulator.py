from pathlib import Path

import numpy as np
import polars as pl

from polymorph.core.base import PipelineContext
from polymorph.core.storage import ParquetStorage
from polymorph.models.analysis import SimulationResult
from polymorph.utils.logging import get_logger

logger = get_logger(__name__)


class MonteCarloSimulator:
    def __init__(
        self,
        context: PipelineContext | None = None,
        processed_dir: str | Path | None = None,
        clip_min: float = -0.99,
        clip_max: float = 1.0,
    ):
        self.context = context
        self.clip_min = clip_min
        self.clip_max = clip_max

        # Set up storage
        self.storage: ParquetStorage | None
        if context:
            self.storage = ParquetStorage(context.data_dir)
            self.processed_dir = context.data_dir / "processed"
        else:
            self.storage = None
            self.processed_dir = Path(processed_dir) if processed_dir else Path("data/processed")

    def load_returns(self, token_id: str) -> np.ndarray:
        returns_path = self.processed_dir / "daily_returns.parquet"

        if not returns_path.exists():
            raise FileNotFoundError(f"daily_returns.parquet not found at {returns_path}. " "Run process stage first.")

        df = pl.read_parquet(returns_path)
        returns = df.filter(pl.col("token_id") == token_id)["ret"].fill_null(0.0).to_numpy()

        if returns.size == 0:
            raise ValueError(f"No returns found for token_id={token_id}")

        # Clip extreme returns
        returns = np.clip(returns, self.clip_min, self.clip_max)

        logger.info(
            f"Loaded {len(returns)} returns for {token_id} " f"(mean={returns.mean():.4f}, std={returns.std():.4f})"
        )

        return returns

    def simulate(
        self,
        token_id: str,
        trials: int,
        horizon_days: int,
        initial_price: float | None = None,
    ) -> SimulationResult:
        logger.info(f"Running Monte Carlo: token={token_id}, trials={trials}, " f"horizon={horizon_days} days")

        # Load historical returns
        returns = self.load_returns(token_id)

        # Sample returns with replacement
        sims = np.random.choice(returns, size=(trials, horizon_days), replace=True)

        # Calculate path returns (compound growth)
        path_returns = (1.0 + sims).prod(axis=1)

        # Calculate statistics
        median_return = float(np.median(path_returns))
        p05_return = float(np.percentile(path_returns, 5))
        p95_return = float(np.percentile(path_returns, 95))
        prob_negative = float((path_returns < 1.0).mean())

        result = SimulationResult(
            token_id=token_id,
            n_trials=trials,
            n_days=horizon_days,
            median_return=median_return,
            percentile_5=p05_return,
            percentile_95=p95_return,
            prob_negative=prob_negative,
            initial_price=initial_price,
            metadata={
                "returns_mean": float(returns.mean()),
                "returns_std": float(returns.std()),
                "returns_count": len(returns),
            },
        )

        logger.info(
            f"Simulation complete: median={median_return:.4f}, "
            f"p05={p05_return:.4f}, p95={p95_return:.4f}, "
            f"prob_negative={prob_negative:.2%}"
        )

        return result

    def run(
        self,
        token_id: str,
        trials: int = 10000,
        horizon_days: int = 30,
        initial_price: float | None = None,
    ) -> dict[str, str | int | float]:
        result = self.simulate(token_id, trials, horizon_days, initial_price)

        return {
            "token_id": result.token_id,
            "trials": result.n_trials,
            "horizon_days": result.n_days,
            "median_growth": result.median_return,
            "p05_growth": result.percentile_5,
            "p95_growth": result.percentile_95,
            "prob_negative": result.prob_negative,
        }
