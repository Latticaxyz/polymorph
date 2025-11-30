from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Generic, TypeVar

from pydantic import BaseModel

from polymorph.config import Settings

T = TypeVar("T")
InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


@dataclass
class PipelineContext:
    settings: Settings
    run_timestamp: datetime
    data_dir: Path
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.data_dir, str):
            self.data_dir = Path(self.data_dir)


class DataSource(ABC, Generic[T]):
    def __init__(self, context: PipelineContext):
        self.context = context
        self.settings = context.settings

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    async def validate(self, data: T) -> bool:
        return data is not None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"


class PipelineStage(ABC, Generic[InputT, OutputT]):
    def __init__(self, context: PipelineContext):
        self.context = context
        self.settings = context.settings

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    async def execute(self, _input_data: InputT | None = None) -> OutputT:
        pass

    async def validate_input(self, _input_data: InputT | None) -> bool:
        return True

    async def validate_output(self, output_data: OutputT) -> bool:
        return output_data is not None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"


class DataModel(BaseModel):
    model_config = {"frozen": False, "extra": "forbid"}
