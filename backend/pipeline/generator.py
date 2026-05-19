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
        f"your answer comes from. If the context doesn't contain enough information, say so clearly.\n"
        f"CRITICAL: Detect the language of the QUESTION and write your entire answer in that "
        f"same language. If the question is in German, answer fully in German. If in French, "
        f"answer in French. Never switch to English unless the question itself is in English.\n\n"
        f"=== CONTEXT ===\n{context}\n\n"
        f"=== QUESTION ===\n{query}\n\n"
        f"=== ANSWER ===\n"
    )


def _strip_cot(text: str) -> str:
    """Remove DeepSeek-R1 <think>...</think> blocks and deduplicate preamble/answer.

    DeepSeek-R1 sometimes writes a draft paragraph outside <think>, then adds
    a separator and '**Answer:**' with the polished version. We keep only the
    final answer section when that pattern is detected.
    """
    # 1. Remove complete <think>...</think> blocks
    stripped = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # Unclosed <think> block (token limit hit mid-reasoning) → signal retry
    if not stripped and "<think>" in text and "</think>" not in text:
        return ""

    result = stripped if stripped else text.strip()

    # 2. If model wrote "draft ... --- **Answer:** final", keep only the final part
    answer_match = re.search(
        r"(?:---+\s*)?\*\*Answer:\*\*\s*(.+)",
        result,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if answer_match and answer_match.group(1).strip():
        result = answer_match.group(1).strip()

    return result


def _token_limit(model: str, requested: int) -> int:
    """Ensure CoT models always have enough budget for the actual answer.

    Both DeepSeek-R1 and Qwen3 emit <think>...</think> blocks that consume
    300–600 tokens before the actual answer begins.
    """
    m = model.lower()
    is_cot = "deepseek" in m or "r1" in m or "qwen3" in m
    return max(requested, 768) if is_cot else requested


# ── Standard (non-streaming) generation ───────────────────────────────────────

def generate(query: str, chunks: List[Dict[str, Any]], model: str = None,
             max_tokens: int = None) -> str:
    prompt = _build_prompt(query, chunks)
    client = get_client()
    messages = [{"role": "user", "content": prompt}]
    active_model = model or config.LLM_MODEL
    token_limit = _token_limit(active_model, max_tokens or config.MAX_NEW_TOKENS)

    for attempt in range(3):
        try:
            resp = client.chat_completion(
                messages=messages,
                model=active_model,
                max_tokens=token_limit,
                temperature=config.TEMPERATURE,
            )
            raw = resp.choices[0].message.content

            # HF occasionally returns None — retry as transient failure
            if raw is None:
                if attempt < 2:
                    time.sleep(3 * (attempt + 1))
                    continue
                return "The model did not return a response. Please try again."

            result = _strip_cot(raw)

            # Empty after stripping = CoT used entire budget; retry once with a note
            if not result:
                if attempt < 2:
                    time.sleep(2)
                    continue
                return (
                    "The model used all available tokens for reasoning without producing "
                    "an answer. Please increase the token limit and try again, or switch "
                    "to a different model."
                )

            return result

        except Exception as e:
            err = str(e)
            if "404" in err or "Not Found" in err:
                short = active_model.split("/")[-1]
                raise RuntimeError(
                    f"**{short}** is not available on the free HF inference tier. "
                    f"Please switch to **DeepSeek-R1 7B** in the sidebar."
                )
            if attempt == 2:
                raise RuntimeError(f"[Generator] Failed after 3 attempts: {e}")
            wait = 2 ** attempt
            if "503" in err or "loading" in err.lower():
                print(f"[Generator] Model loading, retrying in {wait}s...")
            time.sleep(wait)

    return "Could not generate a response after multiple attempts."


# ── Streaming generation (Bonus) ───────────────────────────────────────────────

def generate_stream(query: str, chunks: List[Dict[str, Any]], model: str = None,
                    max_tokens: int = None) -> Generator[str, None, None]:
    """
    Stream tokens, filtering out <think>...</think> reasoning blocks.

    State machine:
      - Before </think>: buffer tokens, suppress if inside a think block
      - After </think>: yield every token directly (fast path)
    Handles: no-think models, single think block, token-limit mid-think.
    """
    prompt = _build_prompt(query, chunks)
    client = get_client()
    messages = [{"role": "user", "content": prompt}]
    active_model = model or config.LLM_MODEL
    token_limit = _token_limit(active_model, max_tokens or config.MAX_NEW_TOKENS)

    buffer = ""
    post_think = False   # True once we've passed </think>
    in_think = False     # True while we're inside a <think> block
    got_answer = False
    # Accumulated output after think block — used to detect draft+answer pattern
    output_so_far = ""

    try:
        resp = client.chat_completion(
            messages=messages,
            model=active_model,
            max_tokens=token_limit,
            temperature=config.TEMPERATURE,
            stream=True,
        )

        for chunk in resp:
            if not getattr(chunk, "choices", None):
                continue
            delta = getattr(chunk.choices[0], "delta", None)
            token = getattr(delta, "content", None) if delta else None
            if token is None:
                continue

            # Fast path: past the think block, stream every token directly
            if post_think:
                output_so_far += token

                # Detect "draft --- **Answer:** final" pattern mid-stream.
                # When we see **Answer:** reset everything already sent and
                # start fresh (the placeholder in the UI will overwrite anyway).
                if "**Answer:**" in output_so_far or "**answer:**" in output_so_far:
                    marker = "**Answer:**" if "**Answer:**" in output_so_far else "**answer:**"
                    _, _, clean = output_so_far.partition(marker)
                    # Yield a reset token so the frontend replaces its buffer
                    got_answer = True
                    yield "\x00RESET\x00" + clean.lstrip()
                    output_so_far = clean.lstrip()
                    continue

                got_answer = True
                yield token
                continue

            buffer += token

            # Check for end of think block
            if "</think>" in buffer:
                _, _, after = buffer.partition("</think>")
                post_think = True
                in_think = False
                buffer = ""
                if after:
                    output_so_far = after
                    got_answer = True
                    yield after
            elif "<think>" in buffer:
                in_think = True
            else:
                # No think block — yield immediately
                output_so_far += buffer
                got_answer = True
                yield buffer
                buffer = ""

        # Flush any remaining buffer (e.g. model ended without </think>)
        if buffer and not in_think:
            got_answer = True
            yield buffer

        # If we got nothing (entire output was inside an unclosed think block)
        if not got_answer:
            yield (
                "\n\n*The model used all available tokens for reasoning without "
                "producing an answer. Please increase the token limit and try again.*"
            )

    except Exception as e:
        err = str(e)
        if "404" in err or "Not Found" in err:
            short = active_model.split("/")[-1]
            yield (
                f"\n\n**{short}** is not available on the free HF inference tier. "
                f"Please switch to **DeepSeek-R1 7B** in the sidebar."
            )
        else:
            yield f"\n[Streaming error: {e}]"
