"""Tests for core base classes and utilities."""

import pytest
from datetime import datetime, timezone
from pathlib import Path
import tempfile

import polars as pl

from polymorph.config import Settings
from polymorph.core.base import PipelineContext, DataSource, PipelineStage
from polymorph.core.storage import ParquetStorage


class TestPipelineContext:
    """Test PipelineContext."""

    def test_create_context(self):
        """Test creating a pipeline context."""
        settings = Settings()
        context = PipelineContext(
            settings=settings,
            run_timestamp=datetime.now(timezone.utc),
            data_dir=Path("data"),
        )

        assert context.settings == settings
        assert isinstance(context.run_timestamp, datetime)
        assert isinstance(context.data_dir, Path)
        assert context.metadata == {}

    def test_data_dir_conversion(self):
        """Test that string data_dir is converted to Path."""
        settings = Settings()
        context = PipelineContext(
            settings=settings,
            run_timestamp=datetime.now(timezone.utc),
            data_dir="data",
        )

        assert isinstance(context.data_dir, Path)
        assert context.data_dir == Path("data")


class TestParquetStorage:
    """Test ParquetStorage backend."""

    def test_write_and_read(self):
        """Test writing and reading parquet files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = ParquetStorage(tmpdir)

            # Create test data
            df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})

            # Write
            storage.write(df, "test.parquet")

            # Read
            result = storage.read("test.parquet")

            assert result.equals(df)

    def test_scan(self):
        """Test lazy scanning of parquet files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = ParquetStorage(tmpdir)

            # Create test data
            df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
            storage.write(df, "test.parquet")

            # Scan
            lazy = storage.scan("test.parquet")
            result = lazy.collect()

            assert result.equals(df)

    def test_exists(self):
        """Test checking if file exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = ParquetStorage(tmpdir)

            assert not storage.exists("test.parquet")

            df = pl.DataFrame({"a": [1, 2, 3]})
            storage.write(df, "test.parquet")

            assert storage.exists("test.parquet")

    def test_resolve_path(self):
        """Test path resolution."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = ParquetStorage(tmpdir)

            # Relative path
            resolved = storage._resolve_path("test.parquet")
            assert resolved == Path(tmpdir) / "test.parquet"

            # Absolute path
            abs_path = Path("/tmp/test.parquet")
            resolved = storage._resolve_path(abs_path)
            assert resolved == abs_path


class DummySource(DataSource[pl.DataFrame]):
    """Dummy data source for testing."""

    @property
    def name(self) -> str:
        return "dummy"

    async def fetch(self, **kwargs) -> pl.DataFrame:
        return pl.DataFrame({"id": [1, 2, 3]})


class DummyStage(PipelineStage[None, str]):
    """Dummy pipeline stage for testing."""

    @property
    def name(self) -> str:
        return "dummy"

    async def execute(self, input_data: None = None) -> str:
        return "executed"


class TestDataSource:
    """Test DataSource base class."""

    def test_data_source_creation(self):
        """Test creating a data source."""
        settings = Settings()
        context = PipelineContext(
            settings=settings,
            run_timestamp=datetime.now(timezone.utc),
            data_dir=Path("data"),
        )

        source = DummySource(context)
        assert source.name == "dummy"
        assert source.context == context
        assert source.settings == settings

    @pytest.mark.asyncio
    async def test_data_source_fetch(self):
        """Test fetching from a data source."""
        settings = Settings()
        context = PipelineContext(
            settings=settings,
            run_timestamp=datetime.now(timezone.utc),
            data_dir=Path("data"),
        )

        source = DummySource(context)
        result = await source.fetch()

        assert isinstance(result, pl.DataFrame)
        assert result.height == 3


class TestPipelineStage:
    """Test PipelineStage base class."""

    def test_pipeline_stage_creation(self):
        """Test creating a pipeline stage."""
        settings = Settings()
        context = PipelineContext(
            settings=settings,
            run_timestamp=datetime.now(timezone.utc),
            data_dir=Path("data"),
        )

        stage = DummyStage(context)
        assert stage.name == "dummy"
        assert stage.context == context
        assert stage.settings == settings

    @pytest.mark.asyncio
    async def test_pipeline_stage_execute(self):
        """Test executing a pipeline stage."""
        settings = Settings()
        context = PipelineContext(
            settings=settings,
            run_timestamp=datetime.now(timezone.utc),
            data_dir=Path("data"),
        )

        stage = DummyStage(context)
        result = await stage.execute()

        assert result == "executed"
