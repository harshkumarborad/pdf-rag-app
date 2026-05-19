"""
pipeline/embeddings.py — HuggingFace BGE Embeddings
====================================================
Stage 2 of the RAG pipeline.

Model: BAAI/bge-small-en-v1.5 (MTEB top-tier, 4× smaller than bge-large)
- Asymmetric prefixing: queries get "Represent this sentence for searching:"
- Documents get no prefix (per BGE model card)
- Batching with retry for HF serverless rate limits
"""

import time
from typing import List
import numpy as np
from huggingface_hub import InferenceClient

import config

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
BATCH_SIZE = 32
MAX_RETRIES = 5

_client: InferenceClient | None = None

def get_client() -> InferenceClient:
    global _client
    if _client is None:
        _client = InferenceClient(token=config.HF_TOKEN, timeout=45)
    return _client

def _embed_batch(texts: List[str], retries: int = MAX_RETRIES) -> List[List[float]]:
    client = get_client()
    for attempt in range(retries):
        try:
            result = client.feature_extraction(
                text=texts,
                model=config.EMBEDDING_MODEL,
            )
            arr = np.array(result, dtype=float)
            if arr.ndim == 3:
                # mean pooling if shape is (batch, seq_len, dim)
                arr = arr.mean(axis=1)
            # normalize for cosine similarity
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            arr = arr / np.where(norms == 0, 1, norms)
            return arr.tolist()
        except Exception as e:
            if "503 Server Error" in str(e) or "currently loading" in str(e).lower():
                wait = 2 ** attempt
                print(f"[Embeddings] Model loading, waiting {wait}s...")
                time.sleep(wait)
                continue
            if attempt == retries - 1:
                raise RuntimeError(f"[Embeddings] Failed after {retries} retries: {e}")
            time.sleep(2 ** attempt)

def embed_documents(texts: List[str]) -> List[List[float]]:
    all_embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i: i + BATCH_SIZE]
        embeddings = _embed_batch(batch)
        all_embeddings.extend(embeddings)
    return all_embeddings

def embed_query(query: str) -> List[float]:
    prefixed = QUERY_PREFIX + query
    result = _embed_batch([prefixed])
    return result[0]

