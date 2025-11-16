import json

import httpx
import polars as pl

from polymorph.core.base import DataSource, PipelineContext
from polymorph.core.retry import with_retry
from polymorph.utils.logging import get_logger

logger = get_logger(__name__)

# JSON type aliases for strict typing
JsonValue = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
JsonDict = dict[str, JsonValue]
JsonList = list[JsonValue]

GAMMA_BASE = "https://gamma-api.polymarket.com"


class Gamma(DataSource[pl.DataFrame]):
    def __init__(
        self,
        context: PipelineContext,
        base_url: str = GAMMA_BASE,
        page_size: int = 250,
        max_pages: int = 200,
    ):
        super().__init__(context)
        self.base_url = base_url
        self.page_size = page_size
        self.max_pages = max_pages
        self._client: httpx.AsyncClient | None = None

    @property
    def name(self) -> str:
        return "gamma"

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self.settings.http_timeout,
                http2=True,
            )
        return self._client

    @with_retry(max_attempts=5, min_wait=1.0, max_wait=10.0)
    async def _get(self, url: str, params: dict[str, int | bool] | None = None) -> JsonDict | JsonList:
        client = await self._get_client()
        r = await client.get(url, params=params, timeout=client.timeout)
        r.raise_for_status()
        result: JsonDict | JsonList = r.json()
        return result

    @staticmethod
    def _normalize_ids(v: object) -> list[str]:
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

    async def fetch(self, active_only: bool = True) -> pl.DataFrame:
        logger.info(f"Fetching markets from Gamma API (active_only={active_only})")

        url = f"{self.base_url}/markets"
        offset = 0
        markets_data: list[JsonValue] = []

        for page in range(self.max_pages):
            params: dict[str, int | bool] = {
                "limit": self.page_size,
                "offset": offset,
            }
            if active_only:
                params["closed"] = False

            payload = await self._get(url, params=params)
            if isinstance(payload, list):
                items = payload
            else:
                data_value = payload.get("data")
                markets_value = payload.get("markets")
                items = (
                    data_value
                    if isinstance(data_value, list)
                    else markets_value if isinstance(markets_value, list) else []
                )

            if not items:
                logger.debug(f"No more items at page {page}")
                break

            markets_data.extend(items)
            logger.debug(f"Fetched page {page + 1}: {len(items)} markets " f"(total: {len(markets_data)})")

            if len(items) < self.page_size:
                break

            offset += self.page_size

        logger.info(f"Fetched {len(markets_data)} total markets")

        if not markets_data:
            return pl.DataFrame({"token_ids": pl.Series([], dtype=pl.List(pl.Utf8))})

        df = pl.DataFrame(markets_data)

        # Normalize token IDs
        if "clobTokenIds" in df.columns:
            df = df.with_columns(
                pl.col("clobTokenIds")
                .map_elements(self._normalize_ids, return_dtype=pl.List(pl.Utf8))
                .alias("token_ids")
            )
        else:
            df = df.with_columns(pl.lit([]).cast(pl.List(pl.Utf8)).alias("token_ids"))

        return df

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "Gamma":
        return self

    async def __aexit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: object,
    ) -> None:
        await self.close()
