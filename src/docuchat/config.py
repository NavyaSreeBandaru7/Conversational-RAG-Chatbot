from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import os

@dataclass(frozen=True)
class Settings:
    docs_dir: Path = Path("docs")
    index_dir: Path = Path("indexes/faiss_index")
    collection_name: str = "docuchat_faiss"

    # Chunking
    chunk_size: int = 900
    chunk_overlap: int = 150

    # Retrieval
    k: int = 8

    # Models
    embedding_model: str = "text-embedding-3-large"
    chat_model: str = "gpt-4.1-mini"
    temperature: float = 0.0

def get_settings() -> Settings:
    return Settings()

def require_openai_key() -> str:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("OPENAI_API_KEY is missing. Add it to .env or your environment variables.")
    return key
