from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Chatbot API"
    app_env: str = "development"
    host: str = "127.0.0.1"
    port: int = 8000
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    llama_base_url: str = "http://127.0.0.1:8080"
    # llama_model: str = "DeepSeek-R1-0528-Qwen3-8B-Q4_K_M"
    llama_model: str = "gemma-4-E2B-it-Q4_K_M"  
    # llama_model: str = "Mellum2.1-12B-A2.5B-Thinking-Q4_K_M"

    llama_timeout: float = 300.0

    chroma_dir: Path = BACKEND_DIR / "chroma_db"
    embedding_model_dir: Path = BACKEND_DIR / "Qwen3-Embedding-0.6B"
    rag_collection: str = "oas_documents"
    rag_top_k: int = 4

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
