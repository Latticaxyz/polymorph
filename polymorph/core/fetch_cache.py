from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator


@dataclass
class CacheKey:
    token_id: str
    start_ts: int
    end_ts: int
    fidelity: int


class FetchCache:
    def __init__(self, cache_path: Path):
        self.cache_path = cache_path
        self._conn: sqlite3.Connection | None = None
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.cache_path))
            self._conn.execute("PRAGMA journal_mode=WAL")
        return self._conn

    def _init_db(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        conn = self._get_conn()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS completed_windows (
                token_id TEXT,
                start_ts INTEGER,
                end_ts INTEGER,
                fidelity INTEGER,
                completed_at TEXT,
                row_count INTEGER,
                PRIMARY KEY (token_id, start_ts, end_ts, fidelity)
            )
        """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_token_id ON completed_windows (token_id)
        """
        )
        conn.commit()

    def is_completed(self, key: CacheKey) -> bool:
        conn = self._get_conn()
        cursor = conn.execute(
            """
            SELECT 1 FROM completed_windows
            WHERE token_id = ? AND start_ts = ? AND end_ts = ? AND fidelity = ?
            """,
            (key.token_id, key.start_ts, key.end_ts, key.fidelity),
        )
        return cursor.fetchone() is not None

    def mark_completed(self, key: CacheKey, row_count: int) -> None:
        conn = self._get_conn()
        conn.execute(
            """
            INSERT OR REPLACE INTO completed_windows
            (token_id, start_ts, end_ts, fidelity, completed_at, row_count)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                key.token_id,
                key.start_ts,
                key.end_ts,
                key.fidelity,
                datetime.now(timezone.utc).isoformat(),
                row_count,
            ),
        )
        conn.commit()

    def get_pending_chunks(
        self,
        token_id: str,
        start_ts: int,
        end_ts: int,
        fidelity: int,
        chunk_size_ms: int,
    ) -> Iterator[tuple[int, int]]:
        current = start_ts
        while current < end_ts:
            chunk_end = min(current + chunk_size_ms, end_ts)
            key = CacheKey(token_id, current, chunk_end, fidelity)
            if not self.is_completed(key):
                yield (current, chunk_end)
            current = chunk_end + 1

    def get_completed_count(self, token_id: str) -> int:
        conn = self._get_conn()
        cursor = conn.execute(
            "SELECT COUNT(*) FROM completed_windows WHERE token_id = ?",
            (token_id,),
        )
        result = cursor.fetchone()
        return result[0] if result else 0

    def get_total_completed(self) -> int:
        conn = self._get_conn()
        cursor = conn.execute("SELECT COUNT(*) FROM completed_windows")
        result = cursor.fetchone()
        return result[0] if result else 0

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
