import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import polars as pl


class DuckDBCatalog:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._con: duckdb.DuckDBPyConnection | None = None
        self._init()

    def _get_connection(self) -> duckdb.DuckDBPyConnection:
        if self._con is None:
            self._con = duckdb.connect(str(self.path))
        return self._con

    def _init(self) -> None:
        con = self._get_connection()
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS datasets (
                dataset TEXT,
                parquet_path TEXT,
                created_at TIMESTAMP,
                rows BIGINT,
                cols BIGINT,
                schema_json TEXT
            );
            """
        )

    def register(self, name: str, path: Path, df: pl.DataFrame) -> None:
        con = self._get_connection()
        con.execute(
            """
            INSERT INTO datasets VALUES (?, ?, ?, ?, ?, ?);
            """,
            [
                name,
                str(path),
                datetime.now(timezone.utc),
                df.height,
                df.width,
                json.dumps({k: str(v) for k, v in df.schema.items()}),
            ],
        )

    def close(self) -> None:
        if self._con is not None:
            self._con.close()
            self._con = None

    def __enter__(self) -> "DuckDBCatalog":
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: object,
    ) -> None:
        self.close()
