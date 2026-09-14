"""Central configuration loaded from environment variables."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    confluence_base_url: str = os.environ["CONFLUENCE_BASE_URL"].rstrip("/")
    confluence_email: str = os.environ["CONFLUENCE_EMAIL"]
    confluence_api_token: str = os.environ["CONFLUENCE_API_TOKEN"]
    confluence_space_key: str = os.environ["CONFLUENCE_SPACE_KEY"]
    openai_api_key: str = os.environ["OPENAI_API_KEY"]
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    chat_model: str = os.getenv("CHAT_MODEL", "gpt-5")
    postgres_url: str = os.getenv(
        "POSTGRES_URL",
        "postgresql+psycopg://langchain:langchain@localhost:6024/langchain",
    )
    pgvector_collection: str = os.getenv("PGVECTOR_COLLECTION", "confluence_knowledge")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    chunking_version: str = os.getenv("CHUNKING_VERSION", "v1")
    top_k: int = int(os.getenv("TOP_K", "5"))


settings = Settings()
