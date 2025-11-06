"""Base classes for pipeline components."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Generic, TypeVar

import polars as pl
from pydantic import BaseModel

from polymorph.config import Settings


T = TypeVar("T")
InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


@dataclass
class PipelineContext:
    """Context passed between pipeline stages."""

    settings: Settings
    run_timestamp: datetime
    data_dir: Path
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Ensure data_dir is a Path object."""
        if isinstance(self.data_dir, str):
            self.data_dir = Path(self.data_dir)


class DataSource(ABC, Generic[T]):
    """Abstract base class for data sources.

    Data sources are responsible for fetching data from external APIs
    or services and returning it in a structured format.
    """

    def __init__(self, context: PipelineContext):
        """Initialize the data source.

        Args:
            context: Pipeline context containing settings and metadata
        """
        self.context = context
        self.settings = context.settings

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this data source."""
        pass

    @abstractmethod
    async def fetch(self, **kwargs) -> T:
        """Fetch data from the source.

        Args:
            **kwargs: Source-specific parameters

        Returns:
            Fetched data in source-specific format
        """
        pass

    async def validate(self, data: T) -> bool:
        """Validate fetched data.

        Args:
            data: Data to validate

        Returns:
            True if data is valid, False otherwise
        """
        return data is not None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"


class PipelineStage(ABC, Generic[InputT, OutputT]):
    """Abstract base class for pipeline stages.

    Pipeline stages represent distinct phases in the data pipeline:
    - Fetch: Ingest data from sources
    - Process: Transform and enrich data
    - Analyze: Run simulations and analysis
    """

    def __init__(self, context: PipelineContext):
        """Initialize the pipeline stage.

        Args:
            context: Pipeline context containing settings and metadata
        """
        self.context = context
        self.settings = context.settings

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this pipeline stage."""
        pass

    @abstractmethod
    async def execute(self, input_data: InputT | None = None) -> OutputT:
        """Execute the pipeline stage.

        Args:
            input_data: Input data from previous stage (optional)

        Returns:
            Output data for next stage
        """
        pass

    async def validate_input(self, input_data: InputT | None) -> bool:
        """Validate input data.

        Args:
            input_data: Input data to validate

        Returns:
            True if input is valid, False otherwise
        """
        return True

    async def validate_output(self, output_data: OutputT) -> bool:
        """Validate output data.

        Args:
            output_data: Output data to validate

        Returns:
            True if output is valid, False otherwise
        """
        return output_data is not None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"


class DataModel(BaseModel):
    """Base class for all data models in the pipeline.

    Uses Pydantic for validation and serialization.
    """

    model_config = {"frozen": False, "extra": "forbid"}
