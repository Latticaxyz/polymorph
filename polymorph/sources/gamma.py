from __future__ import annotations
import polars as pl
import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

GAMMA_BASE = "https://gamma-api.polymarket.com"


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


async def fetch_markets(
    client: httpx.AsyncClient, active_only: bool = True
) -> pl.DataFrame:
    url = f"{GAMMA_BASE}/markets"
    page = 1
    out = []

    while True:
        data = await _get(client, url, params={"page": page, "limit": 100})
        if isinstance(data, dict):
            items = data.get("data") or data.get("markets") or []
        elif isinstance(data, list):
            items = data
        else:
            items = []

        if not items:
            break
        out.extend(items)
        if len(items) < 100:
            break
        page += 1
    df = pl.DataFrame(out)
    if active_only and "closed" in df.columns:
        df = df.filter(~pl.col("closed"))
    return df
