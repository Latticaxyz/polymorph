from __future__ import annotations
from pathlib import Path
import polars as pl


def build_features(in_root: Path, out_root: Path) -> None:
    out_root.mkdir(parents=True, exist_ok=True)

    prices_path = Path(in_root) / "clob" / "prices_history.parquet"
    if prices_path.exists():
        df = pl.read_parquet(prices_path)

        if {"timestamp", "price", "token_id"}.issubset(df.columns):
            lf = (
                df.lazy()
                .with_columns(
                    (pl.col("timestamp").cast(pl.Int64) // 86_400 * 86_400).alias(
                        "day_ts"
                    )
                )
                .group_by(["token_id", "day_ts"])
                .agg(pl.col("price").mean().alias("price_day"))
                .sort(["token_id", "day_ts"])
                .with_columns(
                    pl.col("price_day").pct_change().over("token_id").alias("ret")
                )
            )
            lf.collect().write_parquet(out_root / "daily_returns.parquet")

    trades_path = Path(in_root) / "data_api" / "trades.parquet"
    if trades_path.exists():
        t = pl.read_parquet(trades_path)
        if {"timestamp", "size", "price", "conditionId"}.issubset(t.columns):
            vol = (
                t.lazy()
                .with_columns(
                    [
                        (pl.col("timestamp").cast(pl.Int64) // 86_400 * 86_400).alias(
                            "day_ts"
                        ),
                        (pl.col("size") * pl.col("price")).alias("notional"),
                    ]
                )
                .group_by(["conditionId", "day_ts"])
                .agg(
                    [
                        pl.len().alias("trades"),
                        pl.col("size").sum().alias("size_sum"),
                        pl.col("notional").sum().alias("notional_sum"),
                    ]
                )
                .collect()
            )
            vol.write_parquet(out_root / "trades_daily_agg.parquet")
