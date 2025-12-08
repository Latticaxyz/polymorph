from pathlib import Path
from typing import Tuple

from pydantic import Field
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict, TomlConfigSettingsSource


class Config(BaseSettings):
    http_timeout: int = Field(default=30, ge=1, le=300)
    max_concurrency: int = Field(default=8, ge=1, le=64)
    data_dir: str = Field(default="data")

    model_config = SettingsConfigDict(
        toml_file="polymorph.toml",
        env_prefix="POLYMORPH_",
        case_sensitive=False,
        extra="ignore",
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
        """
        Priority order:
        1. init_settings (passed to constructor)
        2. env_settings (environment variables)
        3. TomlConfigSettingsSource (polymorph.toml file)
        """
        _ = (dotenv_settings, file_secret_settings)
        return (
            init_settings,
            env_settings,
            TomlConfigSettingsSource(settings_cls),
        )


def _ensure_config_exists() -> None:
    config_path = Path("polymorph.toml")
    if not config_path.exists():
        default_config = """[default]
http_timeout = 30
max_concurrency = 8
data_dir = "data"
"""
        config_path.write_text(default_config)


def get_config() -> Config:
    _ensure_config_exists()
    return Config()


_ensure_config_exists()
config = Config()
