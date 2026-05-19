# RAG Document Chat

**Submission for Agentur Philipp GmbH — Technical Assessment**

> Upload one or more PDFs → chat with them in any language → automatic retrieval quality scoring

---

## Live Demo

| Service | URL |
|---------|-----|
| **Frontend (Streamlit)** | https://pdf-rag-app-production-0b80.up.railway.app |
| **Backend API + Docs** | https://pdf-rag-app-production-c25d.up.railway.app/docs |

---

## Quick Start (Local)

```bash
# 1. Clone the repository
git clone https://github.com/HarshkumarBorad/pdf-rag-app.git
cd pdf-rag-app

# 2. Create your .env file
cp .env.example .env
# Edit .env and set HF_TOKEN=hf_your_token_here

# 3. Start everything
docker compose up --build

# 4. Open http://localhost:8501 in your browser
```

> **First run:** the HF embedding model is downloaded on startup — expect ~30 s before the first upload is accepted.

---

## How It Works

```
User uploads PDF(s)
        │
        ▼
  ┌─────────────┐     ┌──────────────┐     ┌─────────────┐
  │  Ingestion  │────▶│  Embeddings  │────▶│  ChromaDB   │
  │ pypdf +     │     │ bge-small    │     │ HNSW index  │
  │ Recursive   │     │ (HF API)     │     │ per session │
  │ Chunking    │     └──────────────┘     └─────────────┘
  └─────────────┘                                 │
                                                  ▼
  User asks a question ──────────────▶   ┌─────────────┐
                                         │  Retriever  │
                                         │ MMR + cover │
                                         │ guarantee   │
                                         └──────┬──────┘
                                                │
                                         ┌──────▼──────┐
                                         │  Generator  │
                                         │ DeepSeek-R1 │
                                         │ (HF API)    │
                                         └──────┬──────┘
                                                │
                                         ┌──────▼──────┐
                                         │  Evaluator  │
                                         │ Faithfulness│
                                         │ Relevancy   │
                                         └─────────────┘
```

---

## Architecture

```
pdf-rag-app/
├── backend/
│   ├── main.py              9 REST endpoints (FastAPI)
│   ├── config.py            All settings + model catalogue
│   ├── test_suite.py        5 questions × 5 domain sets
│   └── pipeline/
│       ├── ingestion.py     PDF loading + chunking
│       ├── embeddings.py    BGE embeddings via HF API
│       ├── vector_store.py  ChromaDB HNSW operations
│       ├── retriever.py     MMR + cross-encoder re-ranking
│       ├── generator.py     LLM generation + CoT stripping
│       └── evaluator.py     Lexical quality metrics
├── frontend/
│   └── app.py               Streamlit UI (3 tabs)
├── docker-compose.yml
└── .env.example
```

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/upload` | Save PDFs, start background indexing, return `session_id` immediately |
| `GET`  | `/sessions/{id}/status` | Poll indexing progress (`indexing` → `ready` → `failed`) |
| `POST` | `/chat` | Query → answer + sources + evaluation metrics |
| `GET`  | `/chat/stream` | ✨ Same as `/chat` but streams tokens via Server-Sent Events |
| `POST` | `/test-suite` | Run 5 hardcoded domain questions, return per-question scores |
| `GET`  | `/question-sets` | Fetch question catalogue (single source of truth) |
| `GET`  | `/models` | Fetch available LLM model list |
| `GET`  | `/sessions/{id}/stats` | Chunk count + file list for a session |
| `DELETE` | `/sessions/{id}` | Delete session index and uploaded files |
| `GET`  | `/health` | Health check + active model names |

---

## Chunking Strategy

**Algorithm:** `RecursiveCharacterTextSplitter`
**Parameters:** `chunk_size=512`, `chunk_overlap=64`

| Decision | Rationale |
|----------|-----------|
| **512 chars** | Fits within BGE-small's context window; larger chunks add noise to embeddings |
| **64 char overlap** | ~12% overlap prevents facts split across boundaries from being lost |
| **Recursive splitting** | Tries `\n\n → \n → ". " → " "` in order — preserves paragraph and sentence boundaries before word splits |

---

## Tech Stack

| Layer | Technology | Notes |
|-------|-----------|-------|
| **Embeddings** | `BAAI/bge-small-en-v1.5` | MTEB top-tier, asymmetric retrieval, 4× smaller than bge-large |
| **LLM (default)** | `DeepSeek-R1-Distill-Qwen-7B` | Chain-of-thought reasoning, free HF serverless tier |
| **Vector Store** | ChromaDB (HNSW, cosine) | Zero-config, session-isolated collections |
| **Retrieval** | MMR (λ=0.6) + coverage guarantee | Diversity + every uploaded document is represented |
| **Re-ranking ✨** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Optional; sees query+doc together → higher precision |
| **Backend** | FastAPI + Uvicorn | Async, auto OpenAPI docs at `/docs` |
| **Frontend** | Streamlit | Chat · Test Suite · About tabs |
| **Deployment** | Docker Compose / Railway | Single-command local start |

---

## Available LLM Models

Select the model from the sidebar dropdown. All are served via the HuggingFace Inference API (free tier via Novita provider).

| Model | Speed | Best for |
|-------|-------|---------|
| **DeepSeek-R1 7B** *(default)* | Medium | Thorough, cited answers; confirmed on free tier |
| **Qwen 3 8B** ⭐ | Medium | Latest (Apr 2025); outscores DeepSeek-R1 7B; best multilingual |
| **Qwen 2.5 72B** | Slow | Richest answers; best quality; 2–3× longer wait |
| **Mistral Nemo 12B** | Medium | European language support (German, French, etc.) |
| **Llama 3.1 8B** | Fast | Meta flagship; well-tested; good English answers |
| **DeepSeek-R1 Llama 8B** | Medium | Llama-based R1 distill; concise |
| **Qwen 2.5 7B** | Medium | Solid multilingual fallback |
| **Qwen 2.5 3B** | Fast | Smallest; best for quick lookups |

> Models marked with (default) are confirmed working on the free tier. Others may return a "not available" message if the Novita provider is not hosting them — the app handles this with a clear error and prompts you to switch back.

---

## Features

### Core
- **Multi-PDF support** — upload several PDFs in one session; the retriever guarantees at least one chunk from every document regardless of query phrasing
- **Language-aware answers** — the LLM detects the question language and responds in the same language (ask in German → get a German answer)
- **Session isolation** — each upload gets its own ChromaDB collection; multiple users don't interfere
- **Source citations** — every answer shows file name, page number, excerpt, and similarity score for each retrieved chunk

### Bonus
- ✨ **Real streaming** — `GET /chat/stream` produces Server-Sent Events; the Streamlit chat tab renders tokens live with a typing cursor
- ✨ **Cross-encoder re-ranking** — optional toggle in sidebar; re-ranks MMR results with a cross-encoder before the LLM call (~5-8% precision gain)
- ✨ **Domain-aware test suite** — 5 question sets (Generic, Legal/Regulatory, Technical, Business, HR/Recruitment); runs all 5 questions and returns per-question scores plus aggregate summary
- ✨ **Model selector** — choose from 8 models without restarting; model shown on each chat bubble

### Production hardening
- **Async background indexing** — `/upload` returns in < 1 s; embedding runs in a background thread; frontend polls `/sessions/{id}/status` with a live counter
- **CoT token guard** — DeepSeek-R1 and Qwen3 get a minimum 768-token budget so reasoning never consumes the entire limit before the answer is written
- **Duplicate-answer fix** — strips draft preambles when the model writes reasoning outside `<think>` tags followed by a `**Answer:**` section
- **Streaming edge-case handling** — correctly handles models without think blocks, token-limit hit mid-think, and mid-stream `**Answer:**` resets

---

## Evaluation Metrics

Each chat response shows 4 metrics computed locally (no extra API calls):

| Metric | Method | What it signals |
|--------|--------|----------------|
| **Faithfulness** | Trigram overlap: answer ∩ retrieved context | Is the answer grounded in the source? |
| **Context Relevancy** | Bigram Jaccard: query ↔ chunks | Did retrieval find relevant chunks? |
| **Answer Relevancy** | Keyword hit-rate: query terms in answer | Does the answer address the question? |
| **Keyword Hit Rate** | Custom: expected keywords in answer | Test suite only — domain-specific accuracy |

> **Note:** These are lightweight *lexical* signals, not semantic judges. A correct answer that paraphrases the source may score lower than expected. `avg_retrieval_score` (cosine similarity from the vector store) is the most reliable retrieval signal. For a production system, RAGAS with an LLM-as-judge would provide more accurate evaluation.

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | *(required)* | HuggingFace API token (free at huggingface.co/settings/tokens) |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Embedding model |
| `LLM_MODEL` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | Default generation model |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder for optional re-ranking |
| `CHUNK_SIZE` | `512` | Chunk size in characters |
| `CHUNK_OVERLAP` | `64` | Overlap between consecutive chunks |
| `TOP_K` | `5` | Chunks to retrieve per query |
| `MMR_LAMBDA` | `0.6` | MMR relevance/diversity balance (1.0 = pure relevance) |
| `MAX_NEW_TOKENS` | `1024` | Max tokens generated per response |
| `TEMPERATURE` | `0.2` | LLM sampling temperature |
| `ENABLE_RERANKING` | `false` | Enable cross-encoder re-ranking by default |
| `CHROMA_PERSIST_DIR` | `./data/chroma_db` | ChromaDB storage path |
| `UPLOAD_DIR` | `./data/uploads` | Uploaded PDF storage path |

---

## Limitations

- **Scanned / image-only PDFs** are not supported. The pipeline uses `pypdf` for text extraction. Scanned PDFs are rejected with a 422 error that explains how to OCR them first (`ocrmypdf input.pdf output.pdf`).
- **HF free-tier cold starts** — the first request after a model is idle can take 10–30 s while HF loads the model. The warmup ping on startup reduces this for the embedding model; the LLM cold start is unavoidable on the free tier.
- **Railway ephemeral storage** — the Railway deployment uses ephemeral disk. Indexed sessions are lost when the backend redeploys. For persistent storage, attach a Railway volume at `/app/data`.
- **Large PDF indexing time** — embedding 1 000+ chunks via the HF Inference API takes several minutes. The async upload flow keeps the UI responsive; the sidebar shows a live elapsed-time counter during indexing.
