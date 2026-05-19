"""
pipeline/retriever.py — Retrieval with MMR + Optional Cross-Encoder Re-ranking
===============================================================================
Stage 4 of the RAG pipeline.

Steps:
  1. Embed query with BGE asymmetric prefix
  2. Vector search → top candidate_k chunks (HNSW in ChromaDB)
  3. MMR reranking for diversity (λ=config.MMR_LAMBDA)
  4. [Bonus] Cross-encoder re-ranking via HF API (if ENABLE_RERANKING=true)

Why cross-encoder re-ranking?
  - Bi-encoders (BGE) compress both query and doc into single vectors independently
  - Cross-encoders see query+doc TOGETHER → much richer attention → ~5-8% precision gain
  - Tradeoff: slower (one forward pass per candidate), so only applied to top-K after MMR
"""

import time
import requests
import numpy as np
from typing import List, Dict, Any, Tuple
from langchain_core.documents import Document

import config
from pipeline.embeddings import embed_query
from pipeline.vector_store import similarity_search, similarity_search_filtered, get_stats


# ── MMR Reranking ──────────────────────────────────────────────────────────────

def _mmr_rerank(
    candidates: List[Tuple[Document, float]],
    top_k: int,
    lambda_: float,
    source_diversity_bonus: float = 0.15,
) -> List[Tuple[Document, float]]:
    """
    Maximal Marginal Relevance reranking with source-diversity bonus.

    MMR(i) = λ·rel(q,dᵢ) - (1-λ)·redundancy + bonus_if_new_source

    The source_diversity_bonus rewards picking a chunk from a document that
    hasn't been represented yet, ensuring multi-PDF sessions surface content
    from every uploaded file rather than clustering on the highest-scoring one.
    """
    if len(candidates) <= top_k:
        return candidates

    selected: List[Tuple[Document, float]] = []
    remaining = list(candidates)

    while len(selected) < top_k and remaining:
        if not selected:
            best = max(remaining, key=lambda x: x[1])
        else:
            sel_scores = [s for _, s in selected]
            sel_sources = {doc.metadata.get("source_file", "") for doc, _ in selected}
            best, best_score = None, -float("inf")
            for doc, rel in remaining:
                redundancy = max(1 - abs(rel - s) for s in sel_scores)
                new_source = doc.metadata.get("source_file", "") not in sel_sources
                mmr_score = (
                    lambda_ * rel
                    - (1 - lambda_) * redundancy
                    + (source_diversity_bonus if new_source else 0.0)
                )
                if mmr_score > best_score:
                    best_score, best = mmr_score, (doc, rel)
        if best:
            selected.append(best)
            remaining.remove(best)

    return selected


# ── Cross-Encoder Re-ranking (Bonus) ──────────────────────────────────────────

def _cross_encoder_rerank(
    query: str,
    candidates: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Re-rank chunks using a cross-encoder model via HF Inference API.
    Model: cross-encoder/ms-marco-MiniLM-L-6-v2
    Sends [query, passage] pairs → gets relevance logit → sort descending.
    """
    url = f"https://api-inference.huggingface.co/models/{config.RERANKER_MODEL}"
    headers = {"Authorization": f"Bearer {config.HF_TOKEN}"}

    pairs = [{"text": query, "text_pair": c["content"]} for c in candidates]
    try:
        resp = requests.post(
            url,
            headers=headers,
            json={"inputs": pairs, "options": {"wait_for_model": True}},
            timeout=30,
        )
        resp.raise_for_status()
        scores = resp.json()  # List of {"label":..., "score":...} or raw floats
        # Normalize: extract highest-class score
        if isinstance(scores[0], list):
            norm_scores = [max(s["score"] for s in item) for item in scores]
        elif isinstance(scores[0], dict):
            norm_scores = [item["score"] for item in scores]
        else:
            norm_scores = [float(s) for s in scores]

        for c, s in zip(candidates, norm_scores):
            c["rerank_score"] = round(s, 4)

        candidates.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
        print(f"[Retriever] Cross-encoder re-ranking applied.")
    except Exception as e:
        print(f"[Retriever] Cross-encoder re-ranking failed (using MMR order): {e}")

    return candidates


# ── Main Retrieve Function ─────────────────────────────────────────────────────

def retrieve(
    query: str,
    session_id: str,
    top_k: int = None,
    use_mmr: bool = True,
    use_reranking: bool = None,
) -> List[Dict[str, Any]]:
    """
    Full retrieval pipeline: embed → vector search → MMR → [cross-encoder].

    Args:
        query: User's question
        session_id: Which ChromaDB collection to search
        top_k: Number of chunks to return
        use_mmr: Apply MMR diversity reranking
        use_reranking: Apply cross-encoder re-ranking (overrides config if set)

    Returns:
        List of dicts: {content, score, metadata, [rerank_score]}
    """
    top_k = top_k or config.TOP_K
    apply_reranking = use_reranking if use_reranking is not None else config.ENABLE_RERANKING

    # Step 1: Embed query
    query_emb = embed_query(query)

    # Step 2: Fetch extra candidates for MMR
    candidate_k = min(top_k * 3, 30)
    candidates = similarity_search(query_emb, top_k=candidate_k, session_id=session_id)

    if not candidates:
        return []

    # Step 3: MMR reranking
    if use_mmr and len(candidates) > top_k:
        final_pairs = _mmr_rerank(candidates, top_k, config.MMR_LAMBDA)
    else:
        final_pairs = candidates[:top_k]

    # Step 4: Format results
    results = [
        {
            "content": doc.page_content,
            "score": round(score, 4),
            "metadata": doc.metadata,
        }
        for doc, score in final_pairs
    ]

    # Step 5: [Bonus] Cross-encoder re-ranking
    if apply_reranking and results:
        results = _cross_encoder_rerank(query, results)

    # Step 6: Multi-PDF coverage guarantee
    # If multiple files were uploaded, ensure every document contributes at
    # least one chunk so the LLM can synthesise across all sources.
    session_files = set(get_stats(session_id).get("files", []))
    if len(session_files) > 1:
        covered = {r["metadata"].get("source_file") for r in results}
        missing = session_files - covered
        for src in missing:
            extra = similarity_search_filtered(query_emb, session_id, src, top_k=1)
            if extra:
                doc, score = extra[0]
                results.append({
                    "content": doc.page_content,
                    "score": round(score, 4),
                    "metadata": doc.metadata,
                    "rerank_score": None,
                })
                print(f"[Retriever] Added coverage chunk from '{src}' (score={score:.4f})")

    print(f"[Retriever] Returning {len(results)} chunks "
          f"(MMR={'on' if use_mmr else 'off'}, "
          f"reranking={'on' if apply_reranking else 'off'}, "
          f"docs_covered={len({r['metadata'].get('source_file') for r in results})}/"
          f"{len(session_files)})")
    return results
