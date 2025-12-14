from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Generic, TypeVar

from polymorph.config import Config
from polymorph.core.storage import PathStorage
from polymorph.core.storage_factory import make_storage

T = TypeVar("T")
InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


@dataclass
class RuntimeConfig:
    http_timeout: int | None = None
    max_concurrency: int | None = None
    data_dir: str | None = None

    def merge_with(self, base: Config) -> "RuntimeConfig":
        return RuntimeConfig(
            http_timeout=self.http_timeout if self.http_timeout is not None else base.general.http_timeout,
            max_concurrency=self.max_concurrency if self.max_concurrency is not None else base.general.max_concurrency,
            data_dir=self.data_dir if self.data_dir is not None else base.general.data_dir,
        )


@dataclass
class PipelineContext:
    config: Config
    run_timestamp: datetime
    data_dir: Path
    runtime_config: RuntimeConfig = field(default_factory=RuntimeConfig)
    metadata: dict[str, object] = field(default_factory=dict)
    storage: PathStorage = field(init=False)

    def __post_init__(self) -> None:
        if isinstance(self.data_dir, str):
            self.data_dir = Path(self.data_dir)
        self.storage = make_storage(self.config, root=self.data_dir)

    @property
    def effective_config(self) -> RuntimeConfig:
        return self.runtime_config.merge_with(self.config)

    @property
    def http_timeout(self) -> int:
        return self.effective_config.http_timeout  # type: ignore

    @property
    def max_concurrency(self) -> int:
        return self.effective_config.max_concurrency  # type: ignore


class DataSource(ABC, Generic[T]):
    def __init__(self, context: PipelineContext):
        self.context = context
        self.config = context.config

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    async def validate(self, data: T) -> bool:
        return data is not None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"


class PipelineStage(ABC, Generic[InputT, OutputT]):
    def __init__(self, context: PipelineContext):
        self.context = context
        self.config = context.config

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    async def execute(self, _input_data: InputT | None = None) -> OutputT:
        raise NotImplementedError

    async def validate_input(self, _input_data: InputT | None) -> bool:
        return True

    async def validate_output(self, output_data: OutputT) -> bool:
        return output_data is not None
