from __future__ import annotations
from pathlib import Path
import optuna
import numpy as np
import polars as pl


def _objective(trial: optuna.Trial, df: pl.DataFrame) -> float:
    thr = trial.suggest_float("ret_threshold", 0.005, 0.05, step=0.0025)
    leverage = trial.suggest_float("leverage", 0.5, 3.0)
    mask = df["ret"].abs() < thr
    ret = (1 + (df["ret"].fill_null(0.0) * leverage * mask.cast(pl.Float64))).to_numpy()
    pnl = float(np.prod(ret))
    return pnl


def run(study_name: str, n_trials: int, in_root: Path) -> None:
    returns_path = Path(in_root) / "daily_returns.parquet"
    if not returns_path.exists():
        raise FileNotFoundError(
            "daily_returns.parquet not found. Run `polymorph process` first."
        )
    df = pl.read_parquet(returns_path)
    study = optuna.create_study(direction="maximize", study_name=study_name)
    study.optimize(lambda tr: _objective(tr, df), n_trials=n_trials)
    print(f"Best value: {study.best_value:.6f} with params: {study.best_params}")
