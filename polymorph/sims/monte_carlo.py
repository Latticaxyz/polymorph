from __future__ import annotations
from pathlib import Path
import polars as pl
import numpy as np


def run(market_id: str, trials: int, horizon_days: int, in_root: Path) -> dict:
    returns_path = Path(in_root) / "daily_returns.parquet"
    if not returns_path.exists():
        raise FileNotFoundError(
            "daily_returns.parquet not found. Run `polymorph process` first."
        )

    df = pl.read_parquet(returns_path)
    sample = df.filter(pl.col("token_id") == market_id)["ret"].fill_null(0.0).to_numpy()

    if sample.size == 0:
        raise ValueError(f"No returns found for market_id={market_id}")

    sample = np.clip(sample, -0.99, 1.0)
    sims = np.random.choice(sample, size=(trials, horizon_days), replace=True)
    path_rets = (1.0 + sims).prod(axis=1)
    return {
        "trials": trials,
        "horizon_days": horizon_days,
        "median_growth": float(np.median(path_rets)),
        "p05_growth": float(np.percentile(path_rets, 5)),
        "p95_growth": float(np.percentile(path_rets, 95)),
        "prob_negative": float((path_rets < 1.0).mean()),
    }
