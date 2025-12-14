from __future__ import annotations

from pathlib import Path

from polymorph.config import Config
from polymorph.core.storage import (
    HybridStorage,
    ParquetDuckDBStorage,
    ParquetStorage,
    PathStorage,
    SQLPathStorage,
)


def _resolve_root(base_root: Path, parquet_root: str) -> Path:
    if not parquet_root:
        return base_root
    p = Path(parquet_root)
    return p if p.is_absolute() else base_root / p


def _resolve_duckdb_path(root: Path, duckdb_path: str) -> Path:
    if not duckdb_path:
        return root / "catalog.duckdb"
    p = Path(duckdb_path)
    return p if p.is_absolute() else root / p


def make_storage(cfg: Config, *, root: Path | None = None) -> PathStorage:
    base_root = root or Path(cfg.general.data_dir)
    parquet_root = _resolve_root(base_root, cfg.storage.parquet_root)
    parquet_root.mkdir(parents=True, exist_ok=True)

    def make_sql() -> SQLPathStorage:
        if not cfg.storage.sql_url:
            raise ValueError("storage.sql_url is required when using SQL storage")
        return SQLPathStorage(
            sql_url=cfg.storage.sql_url,
            sql_schema=cfg.storage.sql_schema or None,
            root=parquet_root,
        )

    if cfg.storage.backend == "sql":
        return make_sql()

    if cfg.storage.backend == "parquet":
        primary: PathStorage = ParquetStorage(parquet_root)
        if cfg.storage.sql_url:
            return HybridStorage(primary, make_sql())
        return primary

    if cfg.storage.backend == "parquet_duckdb":
        catalog_path = _resolve_duckdb_path(parquet_root, cfg.storage.duckdb_path)
        primary = ParquetDuckDBStorage(parquet_root, catalog_path=catalog_path)
        if cfg.storage.sql_url:
            return HybridStorage(primary, make_sql())
        return primary

    raise ValueError(cfg.storage.backend)
