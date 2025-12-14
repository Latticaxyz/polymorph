"""Shared fixtures for integration tests."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from pathlib import Path

import pytest

from polymorph.config import config as base_config
from polymorph.core.base import PipelineContext, RuntimeConfig
from polymorph.sources.clob import CLOB, CLOB_BASE, DATA_API
from polymorph.sources.gamma import GAMMA_BASE, Gamma
from polymorph.utils.time import utc


@pytest.fixture
def integration_context(tmp_path: Path) -> PipelineContext:
    """Create a pipeline context for integration tests."""
    runtime_cfg = RuntimeConfig(http_timeout=30, max_concurrency=8, data_dir=str(tmp_path))
    return PipelineContext(
        config=base_config,
        run_timestamp=utc(),
        data_dir=tmp_path,
        runtime_config=runtime_cfg,
    )


@pytest.fixture
async def clob_client(integration_context: PipelineContext) -> AsyncGenerator[CLOB, None]:
    """Create a CLOB API client for integration tests."""
    async with CLOB(integration_context, clob_base_url=CLOB_BASE, data_api_url=DATA_API) as client:
        yield client


@pytest.fixture
async def gamma_client(integration_context: PipelineContext) -> AsyncGenerator[Gamma, None]:
    """Create a Gamma API client for integration tests."""
    async with Gamma(integration_context, base_url=GAMMA_BASE, max_pages=1, page_size=10) as client:
        yield client
