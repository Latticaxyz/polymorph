from __future__ import annotations

from pathlib import Path
from typing import Literal, Tuple

from pydantic import BaseModel
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)


class GeneralConfig(BaseModel):
    http_timeout: int = 30
    max_concurrency: int = 8
    data_dir: str = "data"
    gamma_max_pages: int | None = None


class StorageConfig(BaseModel):
    backend: Literal["parquet", "parquet_duckdb", "sql"] = "parquet_duckdb"
    parquet_root: str = ""
    duckdb_path: str = ""
    sql_url: str = ""
    sql_schema: str = ""


class Config(BaseSettings):
    general: GeneralConfig = GeneralConfig()
    storage: StorageConfig = StorageConfig()

    model_config = SettingsConfigDict(
        toml_file="polymorph.toml",
        env_prefix="POLYMORPH_",
        env_nested_delimiter="__",
        extra="forbid",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        _ = file_secret_settings
        return (
            init_settings,
            TomlConfigSettingsSource(settings_cls),
            env_settings,
            dotenv_settings,
        )


def _ensure_config_exists() -> None:
    path = Path("polymorph.toml")
    if path.exists():
        return

    path.write_text(
        """[general]
http_timeout = 30
max_concurrency = 8
data_dir = "data"
# gamma_max_pages = 1000  # Uncomment to limit Gamma API fetches (1000 pages × 100 records = 100k max per call)

[storage]
# backend:
# - "parquet": parquet on disk
# - "parquet_duckdb": parquet on disk + a local duckdb catalog (datasets table)
# - "sql": shared DB only (paths -> tables)
#
# If sql_url is set while backend is "parquet" or "parquet_duckdb", storage becomes HYBRID:
# it writes to parquet + mirrors into the shared DB.
backend = "parquet_duckdb"
parquet_root = ""
duckdb_path = ""
sql_url = ""
sql_schema = ""
"""
    )


def get_config() -> Config:
    _ensure_config_exists()
    return Config()


config = get_config()
