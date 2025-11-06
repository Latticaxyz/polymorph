"""Storage abstractions for pipeline data."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import polars as pl


class StorageBackend(ABC):
    """Abstract base class for storage backends."""

    @abstractmethod
    def write(self, data: pl.DataFrame, path: str | Path, **kwargs) -> None:
        """Write a DataFrame to storage.

        Args:
            data: DataFrame to write
            path: Path to write to
            **kwargs: Backend-specific options
        """
        pass

    @abstractmethod
    def read(self, path: str | Path, **kwargs) -> pl.DataFrame:
        """Read a DataFrame from storage.

        Args:
            path: Path to read from
            **kwargs: Backend-specific options

        Returns:
            DataFrame read from storage
        """
        pass

    @abstractmethod
    def scan(self, path: str | Path, **kwargs) -> pl.LazyFrame:
        """Lazily scan a DataFrame from storage.

        Args:
            path: Path to scan
            **kwargs: Backend-specific options

        Returns:
            LazyFrame for lazy evaluation
        """
        pass

    @abstractmethod
    def exists(self, path: str | Path) -> bool:
        """Check if a path exists in storage.

        Args:
            path: Path to check

        Returns:
            True if path exists, False otherwise
        """
        pass


class ParquetStorage(StorageBackend):
    """Parquet-based storage backend."""

    def __init__(self, base_dir: str | Path = "data"):
        """Initialize Parquet storage.

        Args:
            base_dir: Base directory for data storage
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_path(self, path: str | Path) -> Path:
        """Resolve a path relative to base_dir.

        Args:
            path: Path to resolve

        Returns:
            Resolved absolute path
        """
        p = Path(path)
        if p.is_absolute():
            return p
        return self.base_dir / p

    def write(self, data: pl.DataFrame, path: str | Path, **kwargs) -> None:
        """Write a DataFrame to Parquet.

        Args:
            data: DataFrame to write
            path: Path to write to (relative to base_dir or absolute)
            **kwargs: Additional options for write_parquet
        """
        resolved_path = self._resolve_path(path)
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        data.write_parquet(resolved_path, **kwargs)

    def read(self, path: str | Path, **kwargs) -> pl.DataFrame:
        """Read a DataFrame from Parquet.

        Args:
            path: Path to read from
            **kwargs: Additional options for read_parquet

        Returns:
            DataFrame read from Parquet
        """
        resolved_path = self._resolve_path(path)
        return pl.read_parquet(resolved_path, **kwargs)

    def scan(self, path: str | Path, **kwargs) -> pl.LazyFrame:
        """Lazily scan Parquet files.

        Args:
            path: Path or glob pattern to scan
            **kwargs: Additional options for scan_parquet

        Returns:
            LazyFrame for lazy evaluation
        """
        resolved_path = self._resolve_path(path)
        return pl.scan_parquet(resolved_path, **kwargs)

    def exists(self, path: str | Path) -> bool:
        """Check if a Parquet file exists.

        Args:
            path: Path to check

        Returns:
            True if file exists, False otherwise
        """
        resolved_path = self._resolve_path(path)
        return resolved_path.exists()

    def __repr__(self) -> str:
        return f"ParquetStorage(base_dir={self.base_dir})"
