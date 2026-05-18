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

    unique_docs = sorted({c["metadata"].get("source_file", "unknown") for c in chunks})
    multi_doc_note = (
        f" You have access to {len(unique_docs)} documents: {', '.join(unique_docs)}."
        f" If the question spans multiple documents, synthesise the information and"
        f" cite each relevant source."
        if len(unique_docs) > 1 else ""
    )

    return (
        f"You are a precise document assistant.{multi_doc_note} Answer the user's question "
        f"using ONLY the provided context. Always cite which source (file name and page) "
        f"your answer comes from. If the context doesn't contain enough information, say so clearly.\n\n"
        f"=== CONTEXT ===\n{context}\n\n"
        f"=== QUESTION ===\n{query}\n\n"
        f"=== ANSWER ===\n"
    )

def _strip_cot(text: str) -> str:
    """Remove DeepSeek-R1 chain-of-thought <think>...</think> blocks."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()

# ── Standard (non-streaming) generation ───────────────────────────────────────

def generate(query: str, chunks: List[Dict[str, Any]], model: str = None) -> str:
    """
    Generate a grounded answer using the retrieved chunks.

    Args:
        model: HF model ID override; falls back to config.LLM_MODEL if None.

    Returns:
        Clean answer string (chain-of-thought stripped).
    """
    prompt = _build_prompt(query, chunks)
    client = get_client()
    messages = [{"role": "user", "content": prompt}]
    active_model = model or config.LLM_MODEL

    for attempt in range(3):
        try:
            resp = client.chat_completion(
                messages=messages,
                model=active_model,
                max_tokens=config.MAX_NEW_TOKENS,
                temperature=config.TEMPERATURE,
            )
            raw = resp.choices[0].message.content
            return _strip_cot(raw)
        except Exception as e:
            err = str(e)
            if "404" in err or "Not Found" in err:
                # Model not available on the current HF inference provider.
                short = active_model.split("/")[-1]
                raise RuntimeError(
                    f"**{short}** is not available on the free HF inference tier right now. "
                    f"Please switch to **DeepSeek-R1 7B** in the sidebar — it is the confirmed default."
                )
            if "503 Server Error" in err or "loading" in err.lower():
                time.sleep(2 ** attempt)
                continue
            if attempt == 2:
                raise RuntimeError(f"[Generator] Failed: {e}")
            time.sleep(2 ** attempt)

    return "Error: could not generate an answer."

# ── Streaming generation (Bonus) ───────────────────────────────────────────────

def generate_stream(query: str, chunks: List[Dict[str, Any]], model: str = None) -> Generator[str, None, None]:
    """
    Stream tokens from the HF Inference API using Server-Sent Events.
    Yields token strings one by one.

    Args:
        model: HF model ID override; falls back to config.LLM_MODEL if None.
    """
    prompt = _build_prompt(query, chunks)
    client = get_client()
    messages = [{"role": "user", "content": prompt}]
    buffer = ""
    active_model = model or config.LLM_MODEL

    try:
        resp = client.chat_completion(
            messages=messages,
            model=active_model,
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
        err = str(e)
        if "404" in err or "Not Found" in err:
            short = active_model.split("/")[-1]
            yield (
                f"\n\n**{short}** is not available on the free HF inference tier right now. "
                f"Please switch to **DeepSeek-R1 7B** in the sidebar."
            )
        else:
            yield f"\n[Error during streaming: {e}]"
