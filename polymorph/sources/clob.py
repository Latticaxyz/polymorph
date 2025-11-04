from __future__ import annotations
import httpx
import polars as pl
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

CLOB_BASE = "https://clob.polymarket.com"
DATA_API = "https://data-api.polymarket.com"


class RateLimitError(Exception): ...


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((httpx.HTTPError,)),
)
async def _get(
    client: httpx.AsyncClient, url: str, params: dict | None = None
) -> dict | list:
    r = await client.get(url, params=params, timeout=client.timeout)
    r.raise_for_status()
    return r.json()


async def fetch_prices_history(
    client: httpx.AsyncClient,
    token_id: str,
    start_ts: int,
    end_ts: int,
    fidelity: int = 60,
) -> pl.DataFrame:
    url = f"{CLOB_BASE}/prices-history"
    params = {
        "market": token_id,
        "startTs": start_ts,
        "endTs": end_ts,
        "fidelity": fidelity,
    }
    data = await _get(client, url, params=params)
    if not data:
        return pl.DataFrame()
    return pl.DataFrame(data).with_columns([pl.lit(token_id).alias("token_id")])


async def fetch_trades_paged(
    client: httpx.AsyncClient,
    *,
    limit: int = 1000,
    offset: int = 0,
    market_ids: list[str] | None = None,
) -> list[dict]:
    params: dict[str, str | int] = {"limit": limit, "offset": offset}
    if market_ids:
        params["market"] = ",".join(market_ids)
    url = f"{DATA_API}/trades"
    data = await _get(client, url, params=params)
    return data if isinstance(data, list) else data.get("data", [])


async def backfill_trades(
    client: httpx.AsyncClient, *, market_ids: list[str] | None, since_ts: int
) -> pl.DataFrame:
    rows: list[dict] = []
    offset = 0
    limit = 1000
    while True:
        batch = await fetch_trades_paged(
            client, limit=limit, offset=offset, market_ids=market_ids
        )
        if not batch:
            break
        rows.extend(batch)
        offset += limit
        if len(batch) < limit or offset > 200000:
            break
    if not rows:
        return pl.DataFrame()
    df = pl.DataFrame(rows)
    if "timestamp" not in df.columns and "created_at" in df.columns:
        df = df.with_columns(
            pl.col("created_at")
            .str.strptime(pl.Datetime, strict=False, format="%Y-%m-%dT%H:%M:%S%z")
            .cast(pl.Int64)
            .alias("timestamp")
        )
    if "timestamp" in df.columns:
        df = df.filter(pl.col("timestamp") >= since_ts)
    return df
