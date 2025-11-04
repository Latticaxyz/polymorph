from __future__ import annotations
from pathlib import Path
import polars as pl


def write_parquet(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)


def scan_parquet_dir(path: Path) -> pl.LazyFrame:
    return pl.scan_parquet(str(path / "*.parquet"))
