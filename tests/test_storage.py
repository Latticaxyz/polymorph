from __future__ import annotations

from pathlib import Path

import polars as pl

from polymorph.config import Config, GeneralConfig, StorageConfig
from polymorph.core.storage import ParquetStorage
from polymorph.core.storage_factory import make_storage


def test_parquet_storage_round_trip(tmp_path: Path) -> None:
    storage = ParquetStorage(tmp_path)
    df = pl.DataFrame({"x": [1, 2, 3], "y": [4.0, 5.0, 6.0]})

    rel_path = Path("raw") / "test.parquet"
    storage.write(df, rel_path)

    assert storage.exists(rel_path)
    out = storage.read(rel_path)
    assert out.shape == df.shape
    assert out.to_dict(as_series=False) == df.to_dict(as_series=False)


def test_parquet_storage_resolve_absolute(tmp_path: Path) -> None:
    storage = ParquetStorage(tmp_path)
    df = pl.DataFrame({"x": [1]})

    abs_path = tmp_path / "abs.parquet"
    storage.write(df, abs_path)

    # _resolve_path leaves absolute paths untouched
    resolved = storage._resolve_path(abs_path)
    assert resolved == abs_path

    out = storage.read(abs_path)
    assert out.shape == (1, 1)


def test_parquet_storage_scan_glob_pattern(tmp_path: Path) -> None:
    """Test that scan() works with glob patterns to read multiple files."""
    storage = ParquetStorage(tmp_path)
    df1 = pl.DataFrame({"x": [1, 2]})
    df2 = pl.DataFrame({"x": [3, 4]})

    storage.write(df1, "raw/a.parquet")
    storage.write(df2, "raw/b.parquet")

    # Scan with glob pattern should combine both files
    lf = storage.scan("raw/*.parquet")
    collected = lf.collect()

    # Verify combined data from both files
    assert collected.shape == (4, 1), "Should combine rows from both parquet files"
    assert set(collected["x"].to_list()) == {1, 2, 3, 4}, "Should have all values from both files"


def test_make_storage_uses_config_data_dir_when_root_not_provided(tmp_path: Path) -> None:
    cfg = Config(
        general=GeneralConfig(http_timeout=30, max_concurrency=8, data_dir=str(tmp_path)),
        storage=StorageConfig(backend="parquet"),
    )
    storage = make_storage(cfg)
    assert storage.root == tmp_path
