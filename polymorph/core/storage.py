from abc import ABC, abstractmethod
from pathlib import Path

import polars as pl


class StorageBackend(ABC):
    @abstractmethod
    def write(self, data: pl.DataFrame, path: str | Path, **kwargs: object) -> None:
        pass

    @abstractmethod
    def read(self, path: str | Path, **kwargs: object) -> pl.DataFrame:
        pass

    @abstractmethod
    def scan(self, path: str | Path, **kwargs: object) -> pl.LazyFrame:
        pass

    @abstractmethod
    def exists(self, path: str | Path) -> bool:
        pass


class ParquetStorage(StorageBackend):
    def __init__(self, base_dir: str | Path = "data"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_path(self, path: str | Path) -> Path:
        p = Path(path)
        if p.is_absolute():
            return p
        return self.base_dir / p

    def write(self, data: pl.DataFrame, path: str | Path, **kwargs: object) -> None:
        resolved_path = self._resolve_path(path)
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        data.write_parquet(resolved_path, **kwargs)  # type: ignore[arg-type]

    def read(self, path: str | Path, **kwargs: object) -> pl.DataFrame:
        resolved_path = self._resolve_path(path)
        return pl.read_parquet(resolved_path, **kwargs)  # type: ignore[arg-type]

    def scan(self, path: str | Path, **kwargs: object) -> pl.LazyFrame:
        resolved_path = self._resolve_path(path)
        return pl.scan_parquet(resolved_path, **kwargs)  # type: ignore[arg-type]

    def exists(self, path: str | Path) -> bool:
        resolved_path = self._resolve_path(path)
        return resolved_path.exists()

    def __repr__(self) -> str:
        return f"ParquetStorage(base_dir={self.base_dir})"
