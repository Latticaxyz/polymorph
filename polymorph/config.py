from pathlib import Path
from typing import Tuple

from pydantic import BaseModel
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict, TomlConfigSettingsSource


class DefaultSettings(BaseModel):
    http_timeout: int = 30
    max_concurrency: int = 8
    data_dir: str = "data"


class RemoteSettings(BaseModel):
    http_timeout: int = 120
    max_concurrency: int = 32
    data_dir: str = "/data"
    user: str | None = None
    host: str | None = None
    path: str | None = None
    ssh_key: str | None = None


class Settings(BaseSettings):
    default: DefaultSettings = DefaultSettings()
    remote: RemoteSettings = RemoteSettings()

    subgraph_url: str | None = None

    model_config = SettingsConfigDict(
        toml_file="polymorph.toml",
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
        _ = (env_settings, dotenv_settings, file_secret_settings)

        return (
            init_settings,
            TomlConfigSettingsSource(settings_cls),
        )

    def get_active_config(self, use_remote: bool = False) -> DefaultSettings | RemoteSettings:
        """Get the active configuration based on context."""
        return self.remote if use_remote else self.default


def _ensure_config_exists() -> None:
    config_path = Path("polymorph.toml")
    if not config_path.exists():
        default_config = """[default]
http_timeout = 30
max_concurrency = 8
data_dir = "data"
"""
        config_path.write_text(default_config)


_ensure_config_exists()

settings = Settings()
