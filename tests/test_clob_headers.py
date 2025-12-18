from __future__ import annotations

from pathlib import Path

import pytest

from polymorph.config import config as base_config
from polymorph.core.base import PipelineContext, RuntimeConfig
from polymorph.sources.clob import CLOB
from polymorph.sources.gamma import Gamma
from polymorph.utils.time import utc


def _make_context(tmp_path: Path) -> PipelineContext:
    runtime_cfg = RuntimeConfig(http_timeout=None, max_concurrency=None, data_dir=str(tmp_path))
    return PipelineContext(
        config=base_config,
        run_timestamp=utc(),
        data_dir=tmp_path,
        runtime_config=runtime_cfg,
    )


@pytest.mark.asyncio
async def test_clob_client_has_user_agent_header(tmp_path: Path) -> None:
    """Test that CLOB client includes User-Agent header."""
    context = _make_context(tmp_path)
    clob = CLOB(context)

    client = await clob._get_client()

    user_agent = client.headers.get("User-Agent")
    assert user_agent is not None, "User-Agent header should be set"
    assert "polymorph" in user_agent.lower(), "User-Agent should contain 'polymorph'"

    await clob.close()


@pytest.mark.asyncio
async def test_user_agent_format(tmp_path: Path) -> None:
    """Test that User-Agent header follows expected format."""
    context = _make_context(tmp_path)
    clob = CLOB(context)

    client = await clob._get_client()

    user_agent = client.headers.get("User-Agent")
    assert user_agent is not None

    assert "polymorph/0.2.1" in user_agent, "User-Agent should include version number"
    assert "httpx" in user_agent.lower(), "User-Agent should mention httpx"

    await clob.close()


@pytest.mark.asyncio
async def test_gamma_client_has_user_agent_header(tmp_path: Path) -> None:
    """Test that Gamma client includes User-Agent header for consistency."""
    context = _make_context(tmp_path)
    gamma = Gamma(context)

    client = await gamma._get_client()

    user_agent = client.headers.get("User-Agent")
    assert user_agent is not None, "User-Agent header should be set"
    assert "polymorph" in user_agent.lower(), "User-Agent should contain 'polymorph'"

    await gamma.__aexit__(None, None, None)
