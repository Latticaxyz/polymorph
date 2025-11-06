"""Subgraph data source for GraphQL queries."""

from typing import Any

import httpx

from polymorph.core.base import DataSource, PipelineContext
from polymorph.core.retry import with_retry
from polymorph.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_URL = (
    "https://api.goldsky.com/api/public/project_clob/subgraphs/polymarket/gnosis/1/gn"
)


class SubgraphSource(DataSource[dict]):
    """Data source for Polymarket GraphQL subgraph.

    Provides GraphQL query interface to the Goldsky API.
    """

    def __init__(
        self,
        context: PipelineContext,
        url: str | None = None,
    ):
        """Initialize Subgraph source.

        Args:
            context: Pipeline context
            url: Subgraph URL (defaults to Goldsky API)
        """
        super().__init__(context)
        self.url = url or self.settings.subgraph_url or DEFAULT_URL
        self._client: httpx.AsyncClient | None = None

    @property
    def name(self) -> str:
        return "subgraph"

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self.settings.http_timeout,
                http2=True,
            )
        return self._client

    @with_retry(max_attempts=3, min_wait=1.0, max_wait=5.0)
    async def query(
        self,
        query: str,
        variables: dict[str, Any] | None = None,
        url: str | None = None,
    ) -> dict:
        """Execute a GraphQL query.

        Args:
            query: GraphQL query string
            variables: Query variables
            url: Override URL for this query

        Returns:
            Query result dictionary
        """
        endpoint = url or self.url
        client = await self._get_client()

        r = await client.post(
            endpoint,
            json={"query": query, "variables": variables or {}},
            timeout=client.timeout,
        )
        r.raise_for_status()
        return r.json()

    async def fetch(self, query: str, variables: dict | None = None, **kwargs) -> dict:
        """Fetch data using a GraphQL query.

        Args:
            query: GraphQL query string
            variables: Query variables
            **kwargs: Additional parameters

        Returns:
            Query result dictionary
        """
        logger.info("Executing subgraph query")
        result = await self.query(query, variables)
        logger.info(f"Query completed successfully")
        return result

    async def close(self):
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self):
        """Context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        await self.close()
