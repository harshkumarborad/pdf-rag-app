"""
main.py — FastAPI Backend for RAG Document Chat
================================================
Endpoints:
  POST /upload              Upload PDFs → background index → returns immediately
  GET  /sessions/{id}/status  Poll indexing progress
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
from typing import List, Optional, Dict, Any

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


@app.on_event("startup")
async def warmup():
    """Ping the embedding model on boot so the first user request skips cold-start."""
    from pipeline.embeddings import embed_query
    try:
        embed_query("warmup")
        print("[Warmup] Embedding model ready.")
    except Exception as e:
        print(f"[Warmup] Embedding warmup failed (non-fatal): {e}")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(config.UPLOAD_DIR, exist_ok=True)
os.makedirs(config.CHROMA_PERSIST_DIR, exist_ok=True)

# ── In-memory session status (survives page reloads, lost on restart) ──────────
# Keys: session_id → {status, total_chunks, files, error}
_sessions: Dict[str, Dict[str, Any]] = {}


# ── Request / Response Models ──────────────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id: str
    query: str
    top_k: int = Field(default=5, ge=1, le=20)
    use_mmr: bool = True
    use_reranking: bool = False
    llm_model: Optional[str] = None
    max_tokens: Optional[int] = None


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
    status: str = "indexing"


# ── Health Check ──────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "models": {
        "embedding": config.EMBEDDING_MODEL,
        "llm": config.LLM_MODEL,
    }}


@app.get("/models")
def get_models():
    return {"models": config.AVAILABLE_MODELS, "default": config.LLM_MODEL}


# ── Background Indexing ────────────────────────────────────────────────────────

def _index_in_background(session_id: str, saved_paths: List[str]):
    """
    Runs in a background thread after /upload returns.
    Updates _sessions[session_id] with progress.
    """
    try:
        print(f"[Indexing] Starting background index for session {session_id}")
        chunks = ingest_files(saved_paths)
        if not chunks:
            _sessions[session_id].update({
                "status": "failed",
                "error": (
                    "No text could be extracted. The PDF(s) appear to be scanned or "
                    "image-based. Please upload PDFs with selectable text, or run OCR "
                    "(e.g. ocrmypdf) on them first."
                ),
            })
            return
        total = add_documents(chunks, session_id)
        _sessions[session_id].update({"status": "ready", "total_chunks": total})
        print(f"[Indexing] Session {session_id} ready — {total} chunks")
    except Exception as e:
        print(f"[Indexing] Session {session_id} failed: {e}")
        _sessions[session_id].update({"status": "failed", "error": str(e)})


# ── Upload Endpoint ────────────────────────────────────────────────────────────

@app.post("/upload", response_model=UploadResponse)
async def upload_pdfs(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
):
    """
    Save uploaded PDFs and start indexing in the background.
    Returns immediately with session_id. Poll /sessions/{id}/status for progress.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    for f in files:
        if not f.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=400,
                detail=f"Only PDF files accepted. Got: {f.filename}"
            )

    session_id = uuid.uuid4().hex[:8]
    session_dir = Path(config.UPLOAD_DIR) / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    filenames = []
    for upload in files:
        dest = session_dir / upload.filename
        with open(dest, "wb") as out:
            out.write(await upload.read())
        saved_paths.append(str(dest))
        filenames.append(upload.filename)

    # Register session as "indexing" before kicking off background task
    _sessions[session_id] = {
        "status": "indexing",
        "total_chunks": 0,
        "files": filenames,
        "error": None,
    }
    background_tasks.add_task(_index_in_background, session_id, saved_paths)

    return UploadResponse(
        session_id=session_id,
        files=filenames,
        total_chunks=0,
        message=(
            f"Received {len(files)} file(s). Indexing started in the background. "
            f"Poll GET /sessions/{session_id}/status to track progress."
        ),
        status="indexing",
    )


# ── Session Status (for polling) ───────────────────────────────────────────────

@app.get("/sessions/{session_id}/status")
def session_status(session_id: str):
    """
    Returns indexing progress. Frontend polls this after /upload.
    Possible statuses: indexing | ready | failed | unknown
    """
    if session_id in _sessions:
        return _sessions[session_id]
    # Fallback: session pre-dates this server restart but ChromaDB may have data
    stats = get_stats(session_id)
    if stats["chunk_count"] > 0:
        return {"status": "ready", "total_chunks": stats["chunk_count"],
                "files": stats["files"], "error": None}
    return {"status": "unknown", "total_chunks": 0, "files": [], "error": None}


# ── Chat Endpoint ──────────────────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    # Check indexing state first
    sess = _sessions.get(req.session_id, {})
    if sess.get("status") == "indexing":
        raise HTTPException(
            status_code=202,
            detail="Documents are still being indexed. Please wait and try again in a few seconds."
        )
    if sess.get("status") == "failed":
        raise HTTPException(
            status_code=422,
            detail=f"Indexing failed: {sess.get('error', 'unknown error')}"
        )

    stats = get_stats(req.session_id)
    if stats["chunk_count"] == 0:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{req.session_id}' not found or empty. Please upload documents first."
        )

    active_model = req.llm_model or config.LLM_MODEL
    try:
        chunks = retrieve(
            query=req.query,
            session_id=req.session_id,
            top_k=req.top_k,
            use_mmr=req.use_mmr,
            use_reranking=req.use_reranking,
        )
        answer = generate(req.query, chunks, model=active_model, max_tokens=req.max_tokens)
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
    max_tokens: Optional[int] = None,
):
    sess = _sessions.get(session_id, {})
    if sess.get("status") == "indexing":
        raise HTTPException(status_code=202,
                            detail="Documents are still being indexed.")
    if sess.get("status") == "failed":
        raise HTTPException(status_code=422,
                            detail=f"Indexing failed: {sess.get('error')}")

    stats = get_stats(session_id)
    if stats["chunk_count"] == 0:
        raise HTTPException(status_code=404, detail="Session not found or empty.")

    active_model = llm_model or config.LLM_MODEL

    def event_stream():
        try:
            chunks = retrieve(query, session_id, top_k=top_k,
                              use_mmr=use_mmr, use_reranking=use_reranking)
            sources = [
                {
                    "file": c["metadata"].get("source_file", "unknown"),
                    "page": c["metadata"].get("page", "?"),
                    "excerpt": c["content"][:300],
                    "score": c["score"],
                }
                for c in chunks
            ]
            yield f"event: sources\ndata: {json.dumps(sources)}\n\n"

            full_answer = ""
            for token in generate_stream(query, chunks, model=active_model,
                                         max_tokens=max_tokens):
                full_answer += token
                yield f"event: token\ndata: {json.dumps({'token': token})}\n\n"

            metrics = evaluate(query, full_answer, chunks)
            yield f"event: metrics\ndata: {json.dumps(metrics)}\n\n"
            yield "event: done\ndata: [DONE]\n\n"

        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ── Test Suite Endpoint ────────────────────────────────────────────────────────

@app.post("/test-suite")
def run_tests(req: TestSuiteRequest):
    if req.question_set not in QUESTION_SETS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown question set '{req.question_set}'. Available: {list(QUESTION_SETS.keys())}"
        )

    sess = _sessions.get(req.session_id, {})
    if sess.get("status") == "indexing":
        raise HTTPException(status_code=202, detail="Documents are still being indexed.")

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
    return {"sets": QUESTION_SETS}


# ── Session Management ─────────────────────────────────────────────────────────

@app.get("/sessions/{session_id}/stats")
def session_stats(session_id: str):
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
    deleted = delete_session(session_id)
    session_dir = Path(config.UPLOAD_DIR) / session_id
    if session_dir.exists():
        shutil.rmtree(session_dir)
    _sessions.pop(session_id, None)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {"message": f"Session '{session_id}' deleted.", "session_id": session_id}
