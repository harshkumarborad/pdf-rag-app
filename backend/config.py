"""
config.py — Central configuration for RAG Document Chat
All settings loaded from environment variables with sensible defaults.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ── HuggingFace ────────────────────────────────────────────
HF_TOKEN: str = os.getenv("HF_TOKEN", "")

# ── Models ─────────────────────────────────────────────────
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
LLM_MODEL: str = os.getenv("LLM_MODEL", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
RERANKER_MODEL: str = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# ── Chunking ───────────────────────────────────────────────
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "64"))

# ── Retrieval ──────────────────────────────────────────────
TOP_K: int = int(os.getenv("TOP_K", "5"))
MMR_LAMBDA: float = float(os.getenv("MMR_LAMBDA", "0.6"))

# ── Generation ─────────────────────────────────────────────
MAX_NEW_TOKENS: int = int(os.getenv("MAX_NEW_TOKENS", "2048"))
TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.2"))
DO_SAMPLE: bool = True

# ── Bonus Features ─────────────────────────────────────────
ENABLE_RERANKING: bool = os.getenv("ENABLE_RERANKING", "false").lower() == "true"

# ── Storage ────────────────────────────────────────────────
CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "./data/uploads")
