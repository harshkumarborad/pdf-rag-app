"""
main.py — FastAPI Backend for RAG Document Chat
================================================
Endpoints:
  POST /upload              Upload PDFs → chunk → embed → store
  POST /chat                Query → retrieve → generate → evaluate
  GET  /chat/stream         Streaming version (SSE)
  POST /test-suite          Run 5 hardcoded questions
  GET  /sessions/{id}/stats Session statistics
  DELETE /sessions/{id}     Clear a session's index
  GET  /health              Health check
"""

import os
import uuid
import shutil
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import json

import config
from pipeline.ingestion import ingest_files
from pipeline.vector_store import add_documents, get_stats, delete_session
from pipeline.retriever import retrieve
from pipeline.generator import generate, generate_stream
from pipeline.evaluator import evaluate
from test_suite import run_test_suite, QUESTION_SETS

# ── App Setup ──────────────────────────────────────────────────────────────────

app = FastAPI(
    title="RAG Document Chat API",
    description="Upload PDFs and chat with them using DeepSeek-R1 + BGE embeddings.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(config.UPLOAD_DIR, exist_ok=True)
os.makedirs(config.CHROMA_PERSIST_DIR, exist_ok=True)


# ── Request / Response Models ──────────────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id: str
    query: str
    top_k: int = Field(default=5, ge=1, le=20)
    use_mmr: bool = True
    use_reranking: bool = False
    llm_model: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    sources: List[dict]
    metrics: dict
    session_id: str
    query: str
    llm_model: str


class TestSuiteRequest(BaseModel):
    session_id: str
    question_set: str = "Generic"
    top_k: int = Field(default=5, ge=1, le=10)
    use_reranking: bool = False
    llm_model: Optional[str] = None


class UploadResponse(BaseModel):
    session_id: str
    files: List[str]
    total_chunks: int
    message: str


# ── Health Check ──────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "models": {
        "embedding": config.EMBEDDING_MODEL,
        "llm": config.LLM_MODEL,
    }}


@app.get("/models")
def get_models():
    """Return available LLM models and the current default."""
    return {"models": config.AVAILABLE_MODELS, "default": config.LLM_MODEL}


# ── Upload Endpoint ────────────────────────────────────────────────────────────

@app.post("/upload", response_model=UploadResponse)
async def upload_pdfs(files: List[UploadFile] = File(...)):
    """
    Upload one or more PDF files. Creates a new session and indexes all documents.
    Returns session_id to use in subsequent /chat and /test-suite requests.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    # Validate file types
    for f in files:
        if not f.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"Only PDF files accepted. Got: {f.filename}")

    # Create session directory
    session_id = uuid.uuid4().hex[:8]
    session_dir = Path(config.UPLOAD_DIR) / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for upload in files:
        dest = session_dir / upload.filename
        with open(dest, "wb") as out:
            content = await upload.read()
            out.write(content)
        saved_paths.append(str(dest))

    # Ingest + embed + store
    try:
        chunks = ingest_files(saved_paths)
        if not chunks:
            raise HTTPException(
                status_code=422,
                detail=(
                    "No text could be extracted. The PDF(s) appear to be scanned or "
                    "image-based. Please upload PDFs with selectable text, or run OCR "
                    "(e.g. ocrmypdf) on them first."
                ),
            )
        total = add_documents(chunks, session_id)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {e}")

    return UploadResponse(
        session_id=session_id,
        files=[f.filename for f in files],
        total_chunks=total,
        message=f"Successfully indexed {total} chunks from {len(files)} file(s).",
    )


# ── Chat Endpoint ──────────────────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    Answer a question using the documents indexed in the given session.
    Returns answer, source chunks (with file + page attribution), and evaluation metrics.
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    stats = get_stats(req.session_id)
    if stats["chunk_count"] == 0:
        raise HTTPException(status_code=404, detail=f"Session '{req.session_id}' not found or empty. Please upload documents first.")

    active_model = req.llm_model or config.LLM_MODEL
    try:
        chunks = retrieve(
            query=req.query,
            session_id=req.session_id,
            top_k=req.top_k,
            use_mmr=req.use_mmr,
            use_reranking=req.use_reranking,
        )
        answer = generate(req.query, chunks, model=active_model)
        metrics = evaluate(req.query, answer, chunks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    sources = [
        {
            "file": c["metadata"].get("source_file", "unknown"),
            "page": c["metadata"].get("page", "?"),
            "excerpt": c["content"][:300] + "..." if len(c["content"]) > 300 else c["content"],
            "score": c["score"],
            "rerank_score": c.get("rerank_score"),
        }
        for c in chunks
    ]

    return ChatResponse(
        answer=answer,
        sources=sources,
        metrics=metrics,
        session_id=req.session_id,
        query=req.query,
        llm_model=active_model,
    )


# ── Streaming Chat (Bonus) ────────────────────────────────────────────────────

@app.get("/chat/stream")
def chat_stream(
    session_id: str,
    query: str,
    top_k: int = 5,
    use_mmr: bool = True,
    use_reranking: bool = False,
    llm_model: Optional[str] = None,
):
    """
    [Bonus] Streaming chat via Server-Sent Events.
    First event: sources JSON. Then token events. Final event: metrics JSON.
    """
    stats = get_stats(session_id)
    if stats["chunk_count"] == 0:
        raise HTTPException(status_code=404, detail="Session not found or empty.")

    active_model = llm_model or config.LLM_MODEL

    def event_stream():
        try:
            chunks = retrieve(query, session_id, top_k=top_k, use_mmr=use_mmr, use_reranking=use_reranking)
            sources = [
                {
                    "file": c["metadata"].get("source_file", "unknown"),
                    "page": c["metadata"].get("page", "?"),
                    "excerpt": c["content"][:300],
                    "score": c["score"],
                }
                for c in chunks
            ]
            # Send sources first
            yield f"event: sources\ndata: {json.dumps(sources)}\n\n"

            # Stream tokens
            full_answer = ""
            for token in generate_stream(query, chunks, model=active_model):
                full_answer += token
                yield f"event: token\ndata: {json.dumps({'token': token})}\n\n"

            # Send metrics last
            metrics = evaluate(query, full_answer, chunks)
            yield f"event: metrics\ndata: {json.dumps(metrics)}\n\n"
            yield "event: done\ndata: [DONE]\n\n"

        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ── Test Suite Endpoint ────────────────────────────────────────────────────────

@app.post("/test-suite")
def run_tests(req: TestSuiteRequest):
    """
    Run 5 hardcoded test questions against indexed documents.
    Returns per-question scores and an aggregate summary.
    """
    if req.question_set not in QUESTION_SETS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown question set '{req.question_set}'. Available: {list(QUESTION_SETS.keys())}"
        )

    stats = get_stats(req.session_id)
    if stats["chunk_count"] == 0:
        raise HTTPException(status_code=404, detail="Session not found or empty.")

    try:
        result = run_test_suite(
            question_set=req.question_set,
            session_id=req.session_id,
            top_k=req.top_k,
            use_reranking=req.use_reranking,
            llm_model=req.llm_model,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return result


@app.get("/question-sets")
def get_question_sets():
    """Return all question sets with their questions (single source of truth)."""
    return {"sets": QUESTION_SETS}


# ── Session Management ─────────────────────────────────────────────────────────

@app.get("/sessions/{session_id}/stats")
def session_stats(session_id: str):
    """Return stats for a session (chunk count, file list)."""
    stats = get_stats(session_id)
    if stats["chunk_count"] == 0:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {
        **stats,
        "session_id": session_id,
        "embedding_model": config.EMBEDDING_MODEL,
        "llm_model": config.LLM_MODEL,
    }


@app.delete("/sessions/{session_id}")
def clear_session(session_id: str):
    """Delete a session's vector index and uploaded files."""
    deleted = delete_session(session_id)
    # Also clean up uploaded files
    session_dir = Path(config.UPLOAD_DIR) / session_id
    if session_dir.exists():
        shutil.rmtree(session_dir)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {"message": f"Session '{session_id}' deleted.", "session_id": session_id}
