from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    http_timeout: int = 30
    max_concurrency: int = 8
    data_dir: str = "data"
    subgraph_url: str | None = None

    model_config = SettingsConfigDict(
        env_prefix="POLYMORPH_",
        env_file=".env",
        extra="ignore",
    )


settings = Settings()
