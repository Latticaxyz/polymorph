import json
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
    retry=retry_if_exception_type(httpx.RequestError),
)
async def _get(
    client: httpx.AsyncClient, url: str, params: dict | None = None
) -> dict | list:
    r = await client.get(url, params=params, timeout=client.timeout)
    r.raise_for_status()
    return r.json()


def _normalize_ids(v):
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x) for x in v if x is not None]
    if isinstance(v, str):
        s = v.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                arr = json.loads(s)
                if isinstance(arr, list):
                    return [str(x) for x in arr if x is not None]
            except Exception:
                return [s]
        if "," in s:
            return [t.strip() for t in s.split(",") if t.strip()]
        return [s]
    return [str(v)]


async def fetch_markets(
    client: httpx.AsyncClient,
    active_only: bool = True,
    page_size: int = 250,
    max_pages: int = 200,
) -> pl.DataFrame:
    url = f"{GAMMA_BASE}/markets"
    offset = 0
    out: list[dict] = []
    for _ in range(max_pages):
        params = {"limit": page_size, "offset": offset}
        if active_only:
            params["closed"] = False
        payload = await _get(client, url, params=params)
        items = (
            payload
            if isinstance(payload, list)
            else payload.get("data") or payload.get("markets") or []
        )
        if not items:
            break
        out.extend(items)
        if len(items) < page_size:
            break
        offset += page_size
    df = pl.DataFrame(out) if out else pl.DataFrame()
    if df.height:
        if "clobTokenIds" in df.columns:
            df = df.with_columns(
                pl.col("clobTokenIds")
                .map_elements(_normalize_ids, return_dtype=pl.List(pl.Utf8))
                .alias("token_ids")
            )
        else:
            df = df.with_columns(pl.lit([]).cast(pl.List(pl.Utf8)).alias("token_ids"))
    else:
        df = pl.DataFrame({"token_ids": pl.Series([], dtype=pl.List(pl.Utf8))})
    return df
