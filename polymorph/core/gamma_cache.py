from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from polymorph.models.api import Market


class GammaMarketCache:
    """Persistent cache for Gamma market data.

    Strategy:
    - resolved=True markets are immutable -> cache forever
    - closed=True, resolved=False markets may become resolved -> cache, update on fetch
    - closed=False markets are active -> always fetch fresh, update cache
    """

    def __init__(self, cache_path: Path):
        self.cache_path = cache_path
        self._conn: sqlite3.Connection | None = None
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.cache_path))
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _init_db(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        conn = self._get_conn()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS markets (
                id TEXT PRIMARY KEY,
                data TEXT NOT NULL,
                resolved INTEGER NOT NULL DEFAULT 0,
                closed INTEGER NOT NULL DEFAULT 0,
                cached_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_resolved ON markets (resolved)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_closed ON markets (closed)")
        conn.commit()

    def get_resolved_markets(self) -> list[Market]:
        """Get all permanently cached resolved markets."""
        conn = self._get_conn()
        cursor = conn.execute("SELECT data FROM markets WHERE resolved = 1")
        markets: list[Market] = []
        for row in cursor:
            markets.append(Market.model_validate_json(row["data"]))
        return markets

    def get_resolved_market_ids(self) -> set[str]:
        """Get IDs of all cached resolved markets (fast lookup)."""
        conn = self._get_conn()
        cursor = conn.execute("SELECT id FROM markets WHERE resolved = 1")
        return {row["id"] for row in cursor}

    def upsert_markets(self, markets: list[Market]) -> tuple[int, int]:
        """Insert or update markets. Returns (inserted, updated) counts.

        Resolved markets are immutable once cached - updates to resolved
        markets are ignored to preserve the original resolution data.
        """
        conn = self._get_conn()
        now = datetime.now(timezone.utc).isoformat()
        inserted = 0
        updated = 0

        for market in markets:
            data = market.model_dump_json(by_alias=True)
            resolved = 1 if market.resolved else 0
            closed = 1 if market.closed else 0

            cursor = conn.execute(
                "SELECT id, resolved FROM markets WHERE id = ?",
                (market.id,),
            )
            existing = cursor.fetchone()

            if existing is None:
                conn.execute(
                    """
                    INSERT INTO markets (id, data, resolved, closed, cached_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (market.id, data, resolved, closed, now, now),
                )
                inserted += 1
            elif existing["resolved"] == 0:
                conn.execute(
                    """
                    UPDATE markets
                    SET data = ?, resolved = ?, closed = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (data, resolved, closed, now, market.id),
                )
                updated += 1

        conn.commit()
        return inserted, updated

    def get_cache_stats(self) -> dict[str, int]:
        """Get cache statistics."""
        conn = self._get_conn()
        cursor = conn.execute(
            """
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN resolved = 1 THEN 1 ELSE 0 END) as resolved,
                SUM(CASE WHEN closed = 1 AND resolved = 0 THEN 1 ELSE 0 END) as closed_unresolved,
                SUM(CASE WHEN closed = 0 THEN 1 ELSE 0 END) as active
            FROM markets
        """
        )
        row = cursor.fetchone()
        return {
            "total": row["total"] or 0,
            "resolved": row["resolved"] or 0,
            "closed_unresolved": row["closed_unresolved"] or 0,
            "active": row["active"] or 0,
        }

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
