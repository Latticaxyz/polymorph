from __future__ import annotations

from pathlib import Path

import polars as pl

from polymorph.config import Config, GeneralConfig, StorageConfig
from polymorph.core.storage import HybridStorage, ParquetStorage, SQLPathStorage
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


# ============================================================================
# SQL STORAGE TESTS
# ============================================================================


def _sqlite_url(path: Path) -> str:
    return f"sqlite+pysqlite:///{path.as_posix()}"


def test_sql_storage_write_read_roundtrip(tmp_path: Path) -> None:
    db_path = tmp_path / "polymorph_test.db"
    storage = SQLPathStorage(sql_url=_sqlite_url(db_path), sql_schema=None, root=tmp_path)

    df = pl.DataFrame(
        {
            "token_id": ["YES", "NO"],
            "t": [1, 2],
            "p": [0.12, 0.87],
            "flag": [True, False],
        }
    )
    rel_path = Path("raw/clob/20250101T000000Z_prices.parquet")

    assert storage.exists(rel_path) is False
    storage.write(df, rel_path)
    assert storage.exists(rel_path) is True

    out = storage.read(rel_path)
    assert out.height == df.height
    assert set(out.columns) == set(df.columns)

    out_sorted = out.sort(["token_id", "t"])
    df_sorted = df.sort(["token_id", "t"])
    assert out_sorted["token_id"].to_list() == df_sorted["token_id"].to_list()
    assert out_sorted["t"].to_list() == df_sorted["t"].to_list()
    assert out_sorted["p"].to_list() == df_sorted["p"].to_list()
    assert out_sorted["flag"].to_list() == df_sorted["flag"].to_list()


def test_sql_storage_scan_matches_glob(tmp_path: Path) -> None:
    db_path = tmp_path / "polymorph_test.db"
    storage = SQLPathStorage(sql_url=_sqlite_url(db_path), sql_schema=None, root=tmp_path)

    df1 = pl.DataFrame({"token_id": ["A"], "t": [1], "p": [0.1]})
    df2 = pl.DataFrame({"token_id": ["B"], "t": [2], "p": [0.2]})
    df3 = pl.DataFrame({"token_id": ["C"], "t": [3], "p": [0.3]})

    storage.write(df1, Path("raw/clob/20250101T000000Z_prices.parquet"))
    storage.write(df2, Path("raw/clob/20250102T000000Z_prices.parquet"))
    storage.write(df3, Path("raw/clob/20250102T000000Z_trades.parquet"))

    lf = storage.scan(Path("raw/clob/*_prices.parquet"))
    out = lf.collect()

    assert out.height == 2
    assert set(out["token_id"].to_list()) == {"A", "B"}
    assert set(out.columns) == {"token_id", "t", "p"}


def test_hybrid_storage_writes_both_and_fallback_reads_from_sql(tmp_path: Path) -> None:
    db_path = tmp_path / "polymorph_test.db"
    parquet = ParquetStorage(tmp_path)
    sql = SQLPathStorage(sql_url=_sqlite_url(db_path), sql_schema=None, root=tmp_path)
    storage = HybridStorage(primary=parquet, secondary=sql)

    df = pl.DataFrame({"token_id": ["YES"], "t": [123], "p": [0.42]})
    rel_path = Path("raw/clob/20250101T000000Z_prices.parquet")

    storage.write(df, rel_path)

    parquet_path = tmp_path / rel_path
    assert parquet_path.exists() is True
    assert sql.exists(rel_path) is True

    out_primary = storage.read(rel_path).sort(["token_id", "t"])
    assert out_primary.height == 1
    assert out_primary["p"].to_list() == [0.42]

    parquet_path.unlink()
    assert parquet_path.exists() is False

    out_fallback = storage.read(rel_path).sort(["token_id", "t"])
    assert out_fallback.height == 1
    assert out_fallback["p"].to_list() == [0.42]
