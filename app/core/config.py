from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Database
    database_url: str = "postgresql://user:password@localhost:5432/alphalens"

    # Data APIs
    polygon_api_key: str = ""

    # LLM
    openai_api_key: str = ""
    anthropic_api_key: str = ""

    # App
    env: str = "development"
    log_level: str = "INFO"

    # Data config
    default_lookback_years: int = 5
    supported_intervals: list[str] = ["1d", "1wk", "1mo"]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
