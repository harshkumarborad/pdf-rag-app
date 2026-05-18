"""
pipeline/vector_store.py — Session-Isolated ChromaDB Collections
================================================================
Stage 3 of the RAG pipeline.

Each upload session gets its own named ChromaDB collection:
  collection name = "session_<session_id>"

This ensures:
- Multiple simultaneous users don't pollute each other's indices
- Documents from different upload batches stay isolated
- Easy cleanup (just delete the collection)

HNSW index is used automatically by ChromaDB for ANN search.
"""

import os
import sys
import types

# ── Monkey-patch chromadb before import ───────────────────────────────────────
# ChromaDB instantiates ONNXMiniLM_L6_V2 at class-definition time, which
# requires onnxruntime. Since we provide all embeddings via HF API we don't
# need this at all. We inject a stub module so the import succeeds.
_stub = types.ModuleType("onnxruntime")
_stub.InferenceSession = object
sys.modules.setdefault("onnxruntime", _stub)

import chromadb
from chromadb.config import Settings
from typing import List, Tuple, Dict, Any
from langchain_core.documents import Document

import config

# Singleton client
_client = None


def get_client():
    global _client
    if _client is None:
        os.makedirs(config.CHROMA_PERSIST_DIR, exist_ok=True)
        _client = chromadb.PersistentClient(
            path=config.CHROMA_PERSIST_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
    return _client


def _collection_name(session_id: str) -> str:
    # ChromaDB collection names must be alphanumeric + underscores
    return f"session_{session_id.replace('-', '_')}"


def add_documents(chunks: List[Dict[str, Any]], session_id: str) -> int:
    """
    Embed and store chunks in a session-specific ChromaDB collection.

    Args:
        chunks: Output of ingestion.ingest_files()
        session_id: Unique session identifier

    Returns:
        Number of chunks stored
    """
    from pipeline.embeddings import embed_documents

    collection = get_client().get_or_create_collection(
        name=_collection_name(session_id),
        metadata={"hnsw:space": "cosine"},
    )

    texts = [c["content"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    ids = [f"{session_id}_{i}" for i in range(len(chunks))]

    print(f"[VectorStore] Embedding {len(texts)} chunks...")
    embeddings = embed_documents(texts)

    collection.add(
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids,
    )
    print(f"[VectorStore] Stored {len(chunks)} chunks in collection '{_collection_name(session_id)}'")
    return len(chunks)


def similarity_search(
    query_embedding: List[float],
    top_k: int,
    session_id: str,
) -> List[Tuple[Document, float]]:
    """
    Search the session collection for chunks similar to the query embedding.

    Returns:
        List of (Document, cosine_similarity_score) tuples
    """
    collection = get_client().get_collection(_collection_name(session_id))
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    output = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        # ChromaDB cosine distance → similarity: sim = 1 - dist
        similarity = round(1.0 - dist, 4)
        lc_doc = Document(page_content=doc, metadata=meta)
        output.append((lc_doc, similarity))

    return output


def get_stats(session_id: str) -> Dict[str, Any]:
    """Return stats for a session's collection."""
    try:
        col = get_client().get_collection(_collection_name(session_id))
        count = col.count()
        # Get unique source files
        if count > 0:
            result = col.get(limit=count, include=["metadatas"])
            files = list({m.get("source_file", "unknown") for m in result["metadatas"]})
        else:
            files = []
        return {"chunk_count": count, "files": files, "collection": _collection_name(session_id)}
    except Exception:
        return {"chunk_count": 0, "files": [], "collection": None}


def delete_session(session_id: str) -> bool:
    """Delete a session's entire collection."""
    try:
        get_client().delete_collection(_collection_name(session_id))
        return True
    except Exception:
        return False
