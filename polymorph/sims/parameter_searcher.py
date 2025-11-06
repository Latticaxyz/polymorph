from pathlib import Path
from typing import Any, Callable

import numpy as np
import optuna
import polars as pl

from polymorph.core.base import PipelineContext
from polymorph.core.storage import ParquetStorage
from polymorph.models.analysis import OptimizationResult
from polymorph.utils.logging import get_logger

logger = get_logger(__name__)


class ParameterSearcher:
    def __init__(
        self,
        context: PipelineContext | None = None,
        processed_dir: str | Path | None = None,
    ):
        self.context = context

        # Set up storage
        if context:
            self.storage = ParquetStorage(context.data_dir)
            self.processed_dir = context.data_dir / "processed"
        else:
            self.storage = None
            self.processed_dir = (
                Path(processed_dir) if processed_dir else Path("data/processed")
            )

    def load_returns(self) -> pl.DataFrame:
        returns_path = self.processed_dir / "daily_returns.parquet"

        if not returns_path.exists():
            raise FileNotFoundError(
                f"daily_returns.parquet not found at {returns_path}. "
                "Run process stage first."
            )

        df = pl.read_parquet(returns_path)
        logger.info(f"Loaded {len(df)} return observations")

        return df

    def default_objective(self, trial: optuna.Trial, df: pl.DataFrame) -> float:
        # Suggest parameters
        ret_threshold = trial.suggest_float("ret_threshold", 0.005, 0.05, step=0.0025)
        leverage = trial.suggest_float("leverage", 0.5, 3.0)

        # Apply strategy
        mask = df["ret"].abs() < ret_threshold
        returns = (
            1 + (df["ret"].fill_null(0.0) * leverage * mask.cast(pl.Float64))
        ).to_numpy()

        # Calculate compound PnL
        pnl = float(np.prod(returns))

        return pnl

    def optimize(
        self,
        study_name: str,
        n_trials: int,
        objective_fn: Callable[[optuna.Trial, pl.DataFrame], float] | None = None,
        direction: str = "maximize",
        sampler: optuna.samplers.BaseSampler | None = None,
    ) -> OptimizationResult:
        logger.info(
            f"Starting optimization: study={study_name}, trials={n_trials}, "
            f"direction={direction}"
        )

        # Load data
        df = self.load_returns()

        # Use default objective if none provided
        objective = objective_fn or self.default_objective

        # Create study
        study = optuna.create_study(
            direction=direction,
            study_name=study_name,
            sampler=sampler,
        )

        # Optimize
        study.optimize(lambda trial: objective(trial, df), n_trials=n_trials)

        # Build optimization history
        history = [
            {
                "trial": t.number,
                "value": t.value,
                "params": t.params,
                "state": t.state.name,
            }
            for t in study.trials
        ]

        result = OptimizationResult(
            study_name=study_name,
            n_trials=n_trials,
            best_params=study.best_params,
            best_value=study.best_value,
            optimization_history=history,
            metadata={
                "direction": direction,
                "n_observations": len(df),
            },
        )

        logger.info(
            f"Optimization complete: best_value={result.best_value:.6f}, "
            f"best_params={result.best_params}"
        )

        return result

    def run(
        self,
        study_name: str,
        n_trials: int,
        objective_fn: Callable[[optuna.Trial, pl.DataFrame], float] | None = None,
    ) -> None:
        result = self.optimize(study_name, n_trials, objective_fn)

        print(
            f"Best value: {result.best_value:.6f} " f"with params: {result.best_params}"
        )
