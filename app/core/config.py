import os
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_NAME: str = "Enterprise RAG Knowledge Base"
    APP_ENV: str = "development"
    LOG_LEVEL: str = "INFO"
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    LLM_PROVIDER: str = "google"
    LLM_API_KEY: str = ""
    LLM_MODEL: str = "gemini-2.5-flash"

    EMBEDDING_PROVIDER: str = "sentence-transformers"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    VECTOR_DB_PATH: str = "data/vectorstore"
    DOCUMENTS_PATH: str = "data/documents"
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    TOP_K: int = 4

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
