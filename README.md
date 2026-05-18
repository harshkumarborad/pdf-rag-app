# RAG Document Chat

**Submission for Agentur Philipp GmbH — Technical Assessment**

> Upload PDFs → Chat with them → Automatic retrieval quality scoring

---

## Live Demo

| Service | URL |
|---------|-----|
| **Frontend (Streamlit)** | https://pdf-rag-app-production-0b80.up.railway.app |
| **Backend API (FastAPI)** | https://pdf-rag-app-production-c25d.up.railway.app/docs |

## Quick Start

```bash
# 1. Clone / copy this directory
cd pdf-rag-app

# 2. Create your .env file
cp .env.example .env
# Edit .env and set HF_TOKEN=hf_your_token_here

# 3. Start everything
docker-compose up --build

# 4. Open http://localhost:8501 in your browser
```

---

## Architecture

```
pdf-rag-app/
├── backend/          FastAPI (port 8000)
│   ├── main.py       7 REST endpoints
│   ├── pipeline/     Ingestion → Embed → Store → Retrieve → Generate → Evaluate
│   └── test_suite.py 5 hardcoded Q&A × 5 domain sets
├── frontend/         Streamlit (port 8501)
│   └── app.py        3 tabs: Chat · Test Suite · About
├── docker-compose.yml
└── .env.example
```

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/upload` | Upload PDFs → returns `session_id` + chunk count |
| `POST` | `/chat` | Query → answer + sources + metrics |
| `GET`  | `/chat/stream` | ✨ SSE streaming version |
| `POST` | `/test-suite` | Run 5 hardcoded questions |
| `GET`  | `/sessions/{id}/stats` | Session stats |
| `DELETE` | `/sessions/{id}` | Clear session |
| `GET`  | `/health` | Health check |

---

## Chunking Strategy

**Algorithm:** `RecursiveCharacterTextSplitter`  
**Parameters:** `chunk_size=512`, `chunk_overlap=64`

### Why these choices?

| Decision | Rationale |
|----------|-----------|
| **512 chars** | Fits comfortably within BGE-large's 512-token context window; larger chunks increase noise in the retrieved context |
| **64 char overlap** | ~12% overlap prevents facts that span chunk boundaries from being lost during retrieval |
| **Recursive splitting** | Tries `\n\n → \n → ". " → " "` in order — preserves paragraph and sentence boundaries before resorting to word splits |

### Impact on retrieval quality

- **Too small (< 128):** High precision but loses sentence context → incomplete answers
- **Too large (> 1024):** Irrelevant text dilutes embedding signal → lower similarity scores
- **512 / 64:** Sweet spot for document Q&A — confirmed by BEIR benchmark results

---

## Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| Embeddings | `BAAI/bge-large-en-v1.5` | MTEB leaderboard, asymmetric retrieval support |
| LLM | `DeepSeek-R1-Distill-Qwen-7B` | Strong reasoning, free HF serverless tier |
| Vector Store | ChromaDB (HNSW) | Zero-config, persistent, cosine similarity |
| Reranking ✨ | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Sees query+doc together → higher precision |
| Retrieval | MMR (λ=0.6) | Balances relevance and diversity |
| Backend | FastAPI + Uvicorn | Async, auto OpenAPI docs |
| Frontend | Streamlit | Rapid UI, no JS required |
| Deploy | Docker Compose | Single command startup |

---

## Evaluation Metrics

Each chat response shows 4 metrics computed without extra API calls:

| Metric | Method | Target |
|--------|--------|--------|
| **Faithfulness** | Trigram overlap: answer ∩ context | > 0.70 |
| **Context Relevancy** | Bigram Jaccard: query ↔ chunks | > 0.50 |
| **Answer Relevancy** | Keyword hit-rate: query terms in answer | > 0.60 |
| **Keyword Hit Rate** | Custom: expected keywords found in answer | > 0.50 |

> **Reading the metrics:** these are lightweight lexical signals, not semantic
> judgements. They can under-score a correct answer when the model paraphrases
> heavily, when the query uses different vocabulary than the chunks (English
> question over a German document, for example), or when the document is short.
> `avg_retrieval_score` (the cosine similarity of retrieved chunks against the
> query embedding, computed independently in the vector store) is the most
> reliable headline signal for retrieval quality. The lexical metrics are a
> useful triangulation, not a verdict.

---

## Bonus Features

- ✨ **Streaming responses** — `GET /chat/stream` exposes Server-Sent Events
  (`sources` → `token` × N → `metrics` → `done`). The Streamlit chat tab
  consumes the SSE stream directly: tokens land in a styled bubble live as
  they are generated. Toggle in the sidebar.
- ✨ **Cross-encoder re-ranking** — toggle in sidebar; re-ranks MMR results with
  a cross-encoder before LLM call.
- ✨ **Domain-aware test suite** — 5 question sets (Generic, Legal, Technical,
  Business, HR) with user-selectable domain. Question definitions live in the
  backend (`backend/test_suite.py`) and are fetched by the frontend via
  `GET /question-sets` — single source of truth.

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | *(required)* | HuggingFace API token |
| `EMBEDDING_MODEL` | `BAAI/bge-large-en-v1.5` | Embedding model |
| `LLM_MODEL` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | Generation model |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder (bonus) |
| `CHUNK_SIZE` | `512` | Chunk size in characters |
| `CHUNK_OVERLAP` | `64` | Overlap between chunks |
| `TOP_K` | `5` | Chunks to retrieve per query |
| `MMR_LAMBDA` | `0.6` | MMR relevance/diversity trade-off (1.0 = pure relevance) |
| `MAX_NEW_TOKENS` | `2048` | Max answer length |
| `TEMPERATURE` | `0.2` | LLM sampling temperature |
| `ENABLE_RERANKING` | `false` | Enable cross-encoder re-ranking |

---

## Limitations

- **Scanned / image-only PDFs are not supported.** The ingestion pipeline uses
  `pypdf` for text extraction; PDFs that contain only scanned images will be
  rejected with a clear 422 message asking the user to OCR them first
  (`ocrmypdf input.pdf output.pdf` is a one-liner). Adding an OCR fallback is
  intentionally out of scope to keep the container small and the cold start fast.
- **HF Inference free tier rate-limits** can cause the first request after a
  model goes cold to take 10–30 s while the model loads. Retries with backoff
  are built into `pipeline/embeddings.py` and `pipeline/generator.py`.
