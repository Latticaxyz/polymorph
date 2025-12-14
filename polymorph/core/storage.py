from __future__ import annotations

import fnmatch
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import polars as pl
from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    MetaData,
    Table,
    Text,
    create_engine,
    inspect,
)

from polymorph.core.catalog import DuckDBCatalog
from polymorph.utils.logging import get_logger

logger = get_logger(__name__)


def _sanitize(name: str) -> str:
    name = name.strip().replace("-", "_")
    name = re.sub(r"[^a-zA-Z0-9_]+", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_") or "dataset"


class PathStorage(ABC):
    def __init__(self, root: Path) -> None:
        self.root = root

    def _resolve_path(self, path: str | Path) -> Path:
        p = Path(path)
        return p if p.is_absolute() else self.root / p

    @abstractmethod
    def exists(self, path: str | Path) -> bool:
        raise NotImplementedError

    @abstractmethod
    def write(self, df: pl.DataFrame, path: str | Path) -> None:
        raise NotImplementedError

    @abstractmethod
    def read(self, path: str | Path) -> pl.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def scan(self, pattern: str | Path) -> pl.LazyFrame:
        raise NotImplementedError


class ParquetStorage(PathStorage):
    def exists(self, path: str | Path) -> bool:
        return self._resolve_path(path).exists()

    def write(self, df: pl.DataFrame, path: str | Path) -> None:
        p = self._resolve_path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(p)

    def read(self, path: str | Path) -> pl.DataFrame:
        p = self._resolve_path(path)
        return pl.read_parquet(p)

    def scan(self, pattern: str | Path) -> pl.LazyFrame:
        pat = self._resolve_path(pattern)
        return pl.scan_parquet(str(pat))


class ParquetDuckDBStorage(ParquetStorage):
    def __init__(self, root: Path, *, catalog_path: Path) -> None:
        super().__init__(root=root)
        self.catalog = DuckDBCatalog(catalog_path)

    def _dataset_name(self, path: str | Path) -> str:
        p = self._resolve_path(path)
        rel = p.relative_to(self.root) if p.is_relative_to(self.root) else p
        return rel.with_suffix("").as_posix().replace("/", ".")

    def write(self, df: pl.DataFrame, path: str | Path) -> None:
        p = self._resolve_path(path)
        super().write(df, p)
        try:
            self.catalog.register(self._dataset_name(p), p, df)
        except Exception as e:
            logger.warning(f"Failed to register dataset {p} in catalog: {e}", exc_info=True)


class SQLPathStorage(PathStorage):
    def __init__(self, *, sql_url: str, sql_schema: str | None, root: Path) -> None:
        super().__init__(root=root)
        self.engine = create_engine(sql_url)
        self.schema = sql_schema or None

    def _dataset_from_path(self, path: str | Path) -> str:
        p = self._resolve_path(path)
        rel = p.relative_to(self.root) if p.is_relative_to(self.root) else p
        return rel.with_suffix("").as_posix().replace("/", ".")

    def _ensure(self, table: str, df: pl.DataFrame) -> Table:
        insp = inspect(self.engine)

        if insp.has_table(table, schema=self.schema):
            md = MetaData(schema=self.schema)
            return Table(table, md, autoload_with=self.engine)

        def map_type(t: pl.DataType) -> Any:
            if isinstance(t, pl.List):
                return Text()

            type_mapping: dict[Any, Any] = {
                pl.Utf8: Text(),
                pl.Int64: BigInteger(),
                pl.Int32: Integer(),
                pl.Int16: Integer(),
                pl.Int8: Integer(),
                pl.Float64: Float(),
                pl.Float32: Float(),
                pl.Boolean: Boolean(),
                pl.Datetime: DateTime(),
            }
            return type_mapping.get(t, Text())

        md = MetaData(schema=self.schema)
        table_obj = Table(
            table,
            md,
            *[Column(c, map_type(t)) for c, t in df.schema.items()],
        )
        md.create_all(self.engine)
        return table_obj

    def exists(self, path: str | Path) -> bool:
        table = _sanitize(self._dataset_from_path(path))
        insp = inspect(self.engine)
        return insp.has_table(table, schema=self.schema)

    def write(self, df: pl.DataFrame, path: str | Path) -> None:
        dataset = _sanitize(self._dataset_from_path(path))
        table_obj = self._ensure(dataset, df)

        rows = df.to_dicts()
        if not rows:
            return

        with self.engine.begin() as con:
            con.execute(table_obj.insert(), rows)

    def read(self, path: str | Path) -> pl.DataFrame:
        dataset = _sanitize(self._dataset_from_path(path))
        md = MetaData(schema=self.schema)
        table_obj = Table(dataset, md, autoload_with=self.engine)

        with self.engine.connect() as con:
            rows = con.execute(table_obj.select()).mappings().all()
        return pl.DataFrame(rows)

    def scan(self, pattern: str | Path) -> pl.LazyFrame:
        pat = self._resolve_path(pattern)
        rel = pat.relative_to(self.root) if pat.is_relative_to(self.root) else pat

        dot_pat = rel.with_suffix("").as_posix().replace("/", ".")
        dot_pat = _sanitize(dot_pat).replace("_", "*")

        insp = inspect(self.engine)
        tables = [t for t in insp.get_table_names(schema=self.schema) if fnmatch.fnmatch(t, dot_pat)]
        if not tables:
            return pl.LazyFrame()

        dfs: list[pl.DataFrame] = []
        md = MetaData(schema=self.schema)
        for table_name in tables:
            table_obj = Table(table_name, md, autoload_with=self.engine)
            with self.engine.connect() as con:
                rows = con.execute(table_obj.select()).mappings().all()
            if rows:
                dfs.append(pl.DataFrame(rows))

        if not dfs:
            return pl.LazyFrame()

        return pl.concat(dfs, how="vertical").lazy()


class HybridStorage(PathStorage):
    def __init__(self, primary: PathStorage, secondary: PathStorage) -> None:
        super().__init__(root=primary.root)
        self.primary = primary
        self.secondary = secondary

    def exists(self, path: str | Path) -> bool:
        return self.primary.exists(path) or self.secondary.exists(path)

    def write(self, df: pl.DataFrame, path: str | Path) -> None:
        self.primary.write(df, path)
        self.secondary.write(df, path)

    def read(self, path: str | Path) -> pl.DataFrame:
        if self.primary.exists(path):
            return self.primary.read(path)
        return self.secondary.read(path)

    def scan(self, pattern: str | Path) -> pl.LazyFrame:
        try:
            return self.primary.scan(pattern)
        except Exception:
            return self.secondary.scan(pattern)
