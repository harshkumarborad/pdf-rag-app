"""
pipeline/evaluator.py — RAG Evaluation Metrics
===============================================
Stage 5 of the RAG pipeline.

Metrics (lightweight, no extra API calls):
  1. Context Relevancy  — bigram Jaccard between query and retrieved chunks
  2. Faithfulness       — trigram overlap between answer and context
  3. Answer Relevancy   — keyword hit-rate of query terms in answer
  4. Retrieval Score    — average cosine similarity from vector store
"""

import re
import math
from typing import List, Dict, Any
from datetime import datetime


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())


def _ngrams(tokens: List[str], n: int) -> set:
    return set(zip(*[tokens[i:] for i in range(n)])) if len(tokens) >= n else set()


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def compute_context_relevancy(query: str, chunks: List[Dict[str, Any]]) -> float:
    """Bigram Jaccard between query and each retrieved chunk, weighted by score."""
    q_tokens = _tokenize(query)
    q_bg = _ngrams(q_tokens, 2) or _ngrams(q_tokens, 1)
    if not q_bg or not chunks:
        return 0.0
    scores = []
    for c in chunks:
        c_bg = _ngrams(_tokenize(c["content"]), 2) or _ngrams(_tokenize(c["content"]), 1)
        scores.append(_jaccard(q_bg, c_bg) * c.get("score", 1.0))
    total_w = sum(c.get("score", 1.0) for c in chunks) or 1.0
    return min(1.0, sum(scores) / total_w * 15)


def compute_faithfulness(answer: str, chunks: List[Dict[str, Any]]) -> float:
    """Trigram overlap: fraction of answer trigrams present in combined context."""
    if not answer or not chunks:
        return 0.0
    ctx = " ".join(c["content"] for c in chunks)
    a_tri = _ngrams(_tokenize(answer), 3) or _ngrams(_tokenize(answer), 2)
    c_tri = _ngrams(_tokenize(ctx), 3) or _ngrams(_tokenize(ctx), 2)
    if not a_tri:
        return 0.0
    raw = len(a_tri & c_tri) / len(a_tri)
    return round(min(1.0, math.log1p(raw * 10) / math.log1p(10)), 4)


def compute_answer_relevancy(query: str, answer: str) -> float:
    """Keyword hit-rate: fraction of non-stopword query terms in the answer."""
    STOP = {
        "what", "how", "why", "when", "where", "who", "is", "are", "the",
        "a", "an", "of", "in", "to", "and", "or", "for", "with", "that",
        "this", "it", "be", "do", "does", "did", "was", "were", "will",
    }
    q_kw = {t for t in _tokenize(query) if t not in STOP and len(t) > 2}
    a_tokens = set(_tokenize(answer))
    if not q_kw:
        return 0.5
    return round(min(1.0, len(q_kw & a_tokens) / len(q_kw)), 4)


def evaluate(
    query: str,
    answer: str,
    chunks: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Run all metrics and return a combined report."""
    ctx_rel = compute_context_relevancy(query, chunks)
    faith = compute_faithfulness(answer, chunks)
    ans_rel = compute_answer_relevancy(query, answer)
    avg_ret = round(sum(c.get("score", 0) for c in chunks) / len(chunks), 4) if chunks else 0.0

    overall = round(0.4 * faith + 0.35 * ctx_rel + 0.25 * ans_rel, 4)

    if overall >= 0.75:
        quality = "Excellent"
    elif overall >= 0.55:
        quality = "Good"
    elif overall >= 0.35:
        quality = "Fair"
    else:
        quality = "Poor"

    return {
        "context_relevancy": round(ctx_rel, 4),
        "faithfulness": round(faith, 4),
        "answer_relevancy": round(ans_rel, 4),
        "avg_retrieval_score": avg_ret,
        "overall_score": overall,
        "quality": quality,
        "num_chunks": len(chunks),
        "timestamp": datetime.utcnow().isoformat(),
    }
