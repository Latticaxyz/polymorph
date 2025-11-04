from __future__ import annotations
import httpx

DEFAULT_URL = (
    "https://api.goldsky.com/api/public/project_clob/subgraphs/polymarket/gnosis/1/gn"
)


async def query_subgraph(
    client: httpx.AsyncClient,
    query: str,
    variables: dict | None = None,
    url: str | None = None,
) -> dict:
    endpoint = url or DEFAULT_URL
    r = await client.post(
        endpoint,
        json={"query": query, "variables": variables or {}},
        timeout=client.timeout,
    )
    r.raise_for_status()
    return r.json()
