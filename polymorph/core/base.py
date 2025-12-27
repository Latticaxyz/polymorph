from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from polymorph.config import Config
from polymorph.core.storage import PathStorage
from polymorph.core.storage_factory import make_storage
from polymorph.utils.run_names import generate_run_name


@dataclass
class ResolvedConfig:
    http_timeout: int
    max_concurrency: int
    data_dir: Path
    gamma_max_pages: int | None


@dataclass
class RuntimeConfig:
    http_timeout: int | None = None
    max_concurrency: int | None = None
    data_dir: str | None = None
    gamma_max_pages: int | None = None

    def resolve(self, base: Config) -> ResolvedConfig:
        data_dir_str = self.data_dir if self.data_dir is not None else base.general.data_dir
        return ResolvedConfig(
            http_timeout=self.http_timeout if self.http_timeout is not None else base.general.http_timeout,
            max_concurrency=self.max_concurrency if self.max_concurrency is not None else base.general.max_concurrency,
            data_dir=Path(data_dir_str),
            gamma_max_pages=self.gamma_max_pages if self.gamma_max_pages is not None else base.general.gamma_max_pages,
        )


@dataclass
class PipelineContext:
    config: Config
    run_timestamp: datetime
    data_dir: Path
    runtime_config: RuntimeConfig = field(default_factory=RuntimeConfig)
    storage: PathStorage = field(init=False)
    run_name: str = field(init=False)
    _resolved: ResolvedConfig | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if isinstance(self.data_dir, str):
            self.data_dir = Path(self.data_dir)
        self.storage = make_storage(self.config, root=self.data_dir)
        self.run_name = generate_run_name(self.run_timestamp)

    @property
    def resolved(self) -> ResolvedConfig:
        if self._resolved is None:
            self._resolved = self.runtime_config.resolve(self.config)
        return self._resolved

    @property
    def http_timeout(self) -> int:
        return self.resolved.http_timeout

    @property
    def max_concurrency(self) -> int:
        return self.resolved.max_concurrency
