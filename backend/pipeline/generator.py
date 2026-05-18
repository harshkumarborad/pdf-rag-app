"""
pipeline/generator.py — LLM Generation with DeepSeek-R1
=========================================================
Stage 5 of the RAG pipeline.

Two modes:
  1. Standard (batch): returns full answer string
  2. Streaming (bonus): yields token strings for SSE

Chain-of-thought cleanup: DeepSeek-R1 emits <think>...</think> blocks
before the final answer — we strip those for the user-facing response.
"""

import re
import time
import json
from typing import List, Dict, Any, Generator

from huggingface_hub import InferenceClient

import config

_client: InferenceClient | None = None

def get_client() -> InferenceClient:
    global _client
    if _client is None:
        _client = InferenceClient(token=config.HF_TOKEN, timeout=120)
    return _client

def _build_prompt(query: str, chunks: List[Dict[str, Any]]) -> str:
    """Build a RAG prompt with source-grounded instructions."""
    context_parts = []
    for i, c in enumerate(chunks, 1):
        meta = c["metadata"]
        source_tag = f"[Source {i}: {meta.get('source_file','unknown')}, p.{meta.get('page','?')}]"
        context_parts.append(f"{source_tag}\n{c['content']}")

    context = "\n\n---\n\n".join(context_parts)

    return (
        f"You are a precise document assistant. Answer the user's question using ONLY the "
        f"provided context. Always cite which source (file name and page) your answer comes from. "
        f"If the context doesn't contain enough information, say so clearly.\n\n"
        f"=== CONTEXT ===\n{context}\n\n"
        f"=== QUESTION ===\n{query}\n\n"
        f"=== ANSWER ===\n"
    )

def _strip_cot(text: str) -> str:
    """Remove DeepSeek-R1 chain-of-thought <think>...</think> blocks."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()

# ── Standard (non-streaming) generation ───────────────────────────────────────

def generate(query: str, chunks: List[Dict[str, Any]]) -> str:
    """
    Generate a grounded answer using the retrieved chunks.

    Returns:
        Clean answer string (chain-of-thought stripped).
    """
    prompt = _build_prompt(query, chunks)
    client = get_client()
    messages = [{"role": "user", "content": prompt}]
    
    for attempt in range(3):
        try:
            resp = client.chat_completion(
                messages=messages,
                model=config.LLM_MODEL,
                max_tokens=config.MAX_NEW_TOKENS,
                temperature=config.TEMPERATURE,
            )
            raw = resp.choices[0].message.content
            return _strip_cot(raw)
        except Exception as e:
            if "503 Server Error" in str(e) or "loading" in str(e).lower():
                time.sleep(2 ** attempt)
                continue
            if attempt == 2:
                raise RuntimeError(f"[Generator] Failed: {e}")
            time.sleep(2 ** attempt)

    return "Error: could not generate an answer."

# ── Streaming generation (Bonus) ───────────────────────────────────────────────

def generate_stream(query: str, chunks: List[Dict[str, Any]]) -> Generator[str, None, None]:
    """
    Stream tokens from the HF Inference API using Server-Sent Events.
    Yields token strings one by one.
    """
    prompt = _build_prompt(query, chunks)
    client = get_client()
    messages = [{"role": "user", "content": prompt}]
    buffer = ""

    try:
        resp = client.chat_completion(
            messages=messages,
            model=config.LLM_MODEL,
            max_tokens=config.MAX_NEW_TOKENS,
            temperature=config.TEMPERATURE,
            stream=True,
        )
        for chunk in resp:
            # HF Inference can emit a final terminator chunk with empty choices;
            # also defend against missing delta or null content.
            if not getattr(chunk, "choices", None):
                continue
            delta = getattr(chunk.choices[0], "delta", None)
            token = getattr(delta, "content", None) if delta else None
            if token is None:
                continue
            buffer += token
            if "</think>" in buffer:
                _, _, after = buffer.partition("</think>")
                buffer = after
                if after:
                    yield after
            elif "<think>" not in buffer:
                yield token
    except Exception as e:
        yield f"\n[Error during streaming: {e}]"
