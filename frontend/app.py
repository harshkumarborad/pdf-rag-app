"""
frontend/app.py — Streamlit UI for RAG Document Chat
Branding: Agentur Philipp GmbH
"""
import os, json, time, requests
import streamlit as st

BACKEND = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")


@st.cache_data(ttl=300)
def fetch_models() -> list:
    """Fetch available LLM models from the backend. Raises on failure so cache stays empty."""
    r = requests.get(f"{BACKEND}/models", timeout=10)
    r.raise_for_status()
    return r.json().get("models", [])


@st.cache_data(ttl=300)
def fetch_question_sets() -> dict:
    """Fetch the full question-set catalog from the backend. Raises on failure so cache stays empty."""
    r = requests.get(f"{BACKEND}/question-sets", timeout=10)
    r.raise_for_status()
    sets = r.json().get("sets", {})
    return sets if isinstance(sets, dict) else {}

st.set_page_config(page_title="RAG Document Chat | Agentur Philipp GmbH",
                   page_icon="📄", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
:root{--navy:#1d3557;--blue:#457b9d;--light:#a8dadc;--cream:#f1faee;--white:#ffffff;--border:#e2e8f0;--muted:#64748b;}
*{font-family:'Inter',sans-serif;box-sizing:border-box;}
body,.stApp,[data-testid="stAppViewContainer"],[data-testid="block-container"],.main .block-container{
  background:#f8fafc!important;color:#1d3557!important;}
[data-testid="stSidebar"],[data-testid="stSidebar"]>div{background:#1d3557!important;border-right:none!important;}
[data-testid="stSidebar"] *{color:#e8eef4!important;}
[data-testid="stSidebar"] input,[data-testid="stSidebar"] textarea{
  background:rgba(255,255,255,0.08)!important;border:1px solid rgba(168,218,220,0.3)!important;
  color:#e8eef4!important;border-radius:8px!important;}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"],
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] *,
[data-testid="stSidebar"] [data-testid="stFileUploader"] small,
[data-testid="stSidebar"] [data-testid="stFileUploader"] span,
[data-testid="stSidebar"] [data-testid="stFileUploader"] p{
  color:#1d3557!important;}
[data-testid="stSidebar"] [data-testid="stSelectbox"] div,
[data-testid="stSidebar"] [data-testid="stSelectbox"] span,
[data-testid="stSidebar"] [data-testid="stSelectbox"] p,
[data-testid="stSidebar"] [data-baseweb="select"] div,
[data-testid="stSidebar"] [data-baseweb="select"] span,
[data-testid="stSidebar"] [data-baseweb="select"] input{
  color:#1d3557!important;background-color:#fff!important;}
[data-baseweb="popover"] *,[data-baseweb="menu"] *{color:#1d3557!important;}
.stButton>button{background:#457b9d;color:#fff;border:none;border-radius:8px;font-weight:600;transition:all .2s;}
.stButton>button:hover{background:#1d3557;transform:translateY(-1px);}
.hero{background:#1d3557;border-radius:16px;padding:28px 36px;margin-bottom:20px;position:relative;overflow:hidden;}
.hero::after{content:'';position:absolute;top:-40px;right:-40px;width:200px;height:200px;
  background:radial-gradient(circle,rgba(168,218,220,0.15) 0%,transparent 70%);}
.hero h1{color:#fff;font-size:1.8rem;font-weight:800;margin:0 0 6px;letter-spacing:-.02em;}
.hero h1 span{color:#a8dadc;}
.hero p{color:#7fa8bf;margin:0;font-size:.85rem;}
.pill{display:inline-block;padding:3px 11px;border-radius:999px;font-size:.7rem;font-weight:600;margin:2px;}
.pill-blue{background:rgba(69,123,157,.15);color:#457b9d;border:1px solid rgba(69,123,157,.3);}
.pill-green{background:rgba(34,197,94,.1);color:#15803d;border:1px solid rgba(34,197,94,.25);}
.status-bar{display:flex;align-items:center;gap:10px;background:#fff;border:1px solid #e2e8f0;
  border-radius:999px;padding:8px 18px;font-size:.8rem;color:#64748b;margin-bottom:16px;}
.dot{width:8px;height:8px;border-radius:50%;}
.dot-green{background:#22c55e;box-shadow:0 0 6px rgba(34,197,94,.6);}
.dot-yellow{background:#eab308;}
.chat-user{background:#1d3557;border-radius:16px 16px 4px 16px;padding:14px 20px;
  margin:10px 0;margin-left:12%;color:#fff;}
.chat-ai{background:#fff;border:1px solid #e2e8f0;border-radius:16px 16px 16px 4px;
  padding:14px 20px;margin:10px 0;margin-right:12%;color:#1d3557;}
.label{font-size:.62rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;margin-bottom:6px;}
.label-u{color:#a8dadc;}.label-ai{color:#64748b;}
.src-card{background:#fff;border:1px solid #e2e8f0;border-left:3px solid #457b9d;
  border-radius:10px;padding:10px 14px;margin:6px 0;font-size:.8rem;color:#475569;}
.score-badge{display:inline-block;padding:2px 9px;border-radius:999px;font-size:.68rem;
  font-weight:700;margin-left:6px;}
.badge-good{background:#dcfce7;color:#15803d;}
.badge-fair{background:#fef9c3;color:#92400e;}
.badge-poor{background:#fee2e2;color:#991b1b;}
.metric-box{background:#fff;border:1px solid #e2e8f0;border-radius:12px;padding:16px 10px;text-align:center;}
.metric-val{font-size:1.7rem;font-weight:800;color:#1d3557;}
.metric-lbl{font-size:.62rem;color:#64748b;text-transform:uppercase;letter-spacing:.08em;margin-top:4px;font-weight:600;}
hr{border-color:#e2e8f0!important;}
.stTabs [data-baseweb="tab-list"]{background:#f1f5f9;border-radius:999px;padding:3px;}
.stTabs [aria-selected="true"]{background:#1d3557!important;color:#fff!important;border-radius:999px!important;font-weight:700;}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────
for k, v in {"session_id": None, "chat": [], "stats": {}, "streaming": True}.items():
    if k not in st.session_state:
        st.session_state[k] = v

try:
    QUESTION_SETS = fetch_question_sets()
except Exception:
    QUESTION_SETS = {}

try:
    AVAILABLE_MODELS = fetch_models()
except Exception:
    AVAILABLE_MODELS = []

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:16px 0 20px'>
      <div style='font-size:2rem'>📄</div>
      <div style='font-weight:800;font-size:1.05rem;color:#a8dadc;margin-top:6px'>Agentur Philipp</div>
      <div style='font-size:.62rem;color:#5a7d9a;letter-spacing:.1em;text-transform:uppercase;margin-top:2px'>RAG Document Chat</div>
      <div style='width:32px;height:2px;background:#457b9d;border-radius:2px;margin:10px auto 0'></div>
    </div>""", unsafe_allow_html=True)

    # ── Backend status ────────────────────────────────────
    backend_ok = bool(AVAILABLE_MODELS or QUESTION_SETS)
    if backend_ok:
        st.markdown("<div style='font-size:.72rem;color:#4ade80;margin-bottom:8px'>● Backend connected</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='font-size:.72rem;color:#f87171;margin-bottom:4px'>● Backend unreachable</div>", unsafe_allow_html=True)
        if st.button("🔄 Retry connection", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    st.markdown("#### 📤 Upload Documents")
    uploaded = st.file_uploader("Select PDF files", type=["pdf"],
                                 accept_multiple_files=True, label_visibility="collapsed")

    if uploaded and st.button("🚀 Index Documents", use_container_width=True):
        try:
            files_data = [("files", (f.name, f.read(), "application/pdf")) for f in uploaded]
            with st.spinner("Sending files to backend…"):
                r = requests.post(f"{BACKEND}/upload", files=files_data, timeout=60)
                r.raise_for_status()
                data = r.json()
                sid = data["session_id"]

            # Poll /sessions/{id}/status until indexing finishes
            progress_box = st.empty()
            start = time.time()
            while True:
                try:
                    sr = requests.get(f"{BACKEND}/sessions/{sid}/status", timeout=10)
                    info = sr.json()
                except Exception:
                    info = {"status": "indexing"}

                status = info.get("status", "indexing")
                elapsed = int(time.time() - start)

                if status == "ready":
                    chunks = info.get("total_chunks", "?")
                    st.session_state.session_id = sid
                    st.session_state.stats = {
                        "session_id": sid,
                        "total_chunks": chunks,
                        "files": info.get("files", [f[1][0] for f in files_data]),
                    }
                    st.session_state.chat = []
                    progress_box.success(f"✅ {chunks} chunks indexed from {len(uploaded)} file(s)!")
                    time.sleep(1)
                    st.rerun()
                    break
                elif status == "failed":
                    progress_box.error(f"❌ Indexing failed: {info.get('error', 'Unknown error')}")
                    break
                else:
                    progress_box.info(f"⏳ Indexing… {elapsed}s elapsed — large PDFs may take a few minutes.")
                    time.sleep(4)
        except Exception as e:
            st.error(f"Upload failed: {e}")

    if st.session_state.session_id:
        st.markdown("---")
        st.markdown("#### ⚙️ Settings")

        # ── LLM model selector ────────────────────────────────
        if AVAILABLE_MODELS:
            model_labels = [m["label"] for m in AVAILABLE_MODELS]
            model_ids    = [m["id"]    for m in AVAILABLE_MODELS]
            model_descs  = [m.get("description", "") for m in AVAILABLE_MODELS]
            sel_idx = st.selectbox(
                "🤖 LLM Model",
                range(len(model_labels)),
                format_func=lambda i: model_labels[i],
                help="\n".join(f"**{model_labels[i]}** — {model_descs[i]}" for i in range(len(model_labels))),
            )
            selected_model = model_ids[sel_idx]
            st.caption(f"_{model_descs[sel_idx]}_")
        else:
            selected_model = None

        top_k = st.slider("Chunks to retrieve (Top-K)", 3, 10, 5)
        max_tokens = st.slider("Max answer length (tokens)", 256, 1536, 768, step=128,
                               help="DeepSeek-R1 uses 300–600 tokens for reasoning before the answer. 768 is the safe minimum.")
        use_mmr = st.toggle("MMR Diversity", value=True)
        use_rerank = st.toggle("Cross-Encoder Re-ranking ✨", value=False,
                               help="Bonus: Re-ranks chunks with a cross-encoder before generation")
        st.session_state.streaming = st.toggle("Streaming responses ✨", value=True,
                                               help="Bonus: Stream tokens as they are generated")
        st.markdown("---")
        s = st.session_state.stats
        st.markdown(f"""
        <div style='font-size:.75rem;color:#7fa8bf'>
        <b style='color:#a8dadc'>Session:</b> <code style='background:rgba(255,255,255,.08);
        padding:1px 6px;border-radius:4px'>{st.session_state.session_id}</code><br>
        <b style='color:#a8dadc'>Files:</b> {', '.join(s.get('files', []))}<br>
        <b style='color:#a8dadc'>Chunks:</b> {s.get('total_chunks', '—')}
        </div>""", unsafe_allow_html=True)

        if st.button("🗑️ Clear Session", use_container_width=True):
            requests.delete(f"{BACKEND}/sessions/{st.session_state.session_id}", timeout=10)
            st.session_state.session_id = None
            st.session_state.chat = []
            st.session_state.stats = {}
            st.rerun()
    else:
        top_k, max_tokens, use_mmr, use_rerank, selected_model = 5, 768, True, False, None

# ── Hero ──────────────────────────────────────────────────────
st.markdown("""
<div class='hero'>
  <h1>RAG Document Chat &mdash; <span>Agentur Philipp GmbH</span></h1>
  <p>Upload PDFs · Chat with sources · DeepSeek-R1 &nbsp;·&nbsp; BAAI/bge-small-en-v1.5 &nbsp;·&nbsp; ChromaDB HNSW</p>
</div>""", unsafe_allow_html=True)

if st.session_state.session_id:
    st.markdown(
        f"<div class='status-bar'><div class='dot dot-green'></div>"
        f"<span><b style='color:#1d3557'>{st.session_state.stats.get('total_chunks','?')}</b>"
        f" chunks indexed &nbsp;·&nbsp; Session "
        f"<code style='background:#f1f5f9;padding:1px 6px;border-radius:5px'>"
        f"{st.session_state.session_id}</code></span></div>",
        unsafe_allow_html=True)
else:
    st.markdown("<div class='status-bar'><div class='dot dot-yellow'></div>"
                "<span>No documents loaded — upload PDFs in the sidebar to begin</span></div>",
                unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────
tab_chat, tab_tests, tab_info = st.tabs(["💬 Chat", "🧪 Test Suite", "ℹ️ About"])

# ─── CHAT TAB ─────────────────────────────────────────────────
with tab_chat:
    for msg in st.session_state.chat:
        if msg["role"] == "user":
            st.markdown(f"<div class='chat-user'><div class='label label-u'>You →</div>{msg['content']}</div>",
                        unsafe_allow_html=True)
        else:
            used_model = msg.get("llm_model", "")
            model_label = next((m["label"] for m in AVAILABLE_MODELS if m["id"] == used_model), used_model.split("/")[-1] if used_model else "DeepSeek-R1")
            st.markdown(f"<div class='chat-ai'><div class='label label-ai'>Agentur Philipp RAG · {model_label}</div>"
                        f"{msg['content'].replace(chr(10),'<br>')}</div>", unsafe_allow_html=True)
            if msg.get("sources"):
                with st.expander(f"📎 {len(msg['sources'])} source chunks", expanded=False):
                    for s in msg["sources"]:
                        badge_color = "#457b9d"
                        st.markdown(
                            f"<div class='src-card'>"
                            f"<b style='color:#1d3557'>{s['file']}</b> &nbsp;·&nbsp; Page {s['page']}"
                            f"<span class='score-badge' style='background:rgba(69,123,157,.12);color:#457b9d'>"
                            f"score {s['score']:.2f}</span><br>"
                            f"<span style='color:#64748b'>{s['excerpt']}</span></div>",
                            unsafe_allow_html=True)
            if msg.get("metrics"):
                m = msg["metrics"]
                c1, c2, c3, c4 = st.columns(4)
                for col, lbl, key in [
                    (c1, "Faithfulness", "faithfulness"),
                    (c2, "Ctx Relevancy", "context_relevancy"),
                    (c3, "Ans Relevancy", "answer_relevancy"),
                    (c4, "Overall", "overall_score"),
                ]:
                    v = m.get(key, 0)
                    col.markdown(
                        f"<div class='metric-box'><div class='metric-val'>{v:.0%}</div>"
                        f"<div class='metric-lbl'>{lbl}</div></div>",
                        unsafe_allow_html=True)

    st.markdown("")
    with st.form("chat_form", clear_on_submit=True):
        query = st.text_input(
            "Ask a question about your documents…",
            placeholder="e.g. What are the main obligations for providers?",
            label_visibility="collapsed",
        )
        col_ask, col_clr = st.columns([4, 1])
        ask = col_ask.form_submit_button(
            "🔍 Ask", use_container_width=True,
            disabled=not st.session_state.session_id,
        )
        clear_chat = col_clr.form_submit_button("🗑️", use_container_width=True)

    if clear_chat:
        st.session_state.chat = []
        st.rerun()

    if ask and query.strip():
        st.session_state.chat.append({"role": "user", "content": query})

        if st.session_state.streaming:
            # Render the just-added user bubble so the stream appears below it
            st.markdown(
                f"<div class='chat-user'><div class='label label-u'>You →</div>{query}</div>",
                unsafe_allow_html=True,
            )

            params = {
                "session_id": st.session_state.session_id,
                "query": query,
                "top_k": top_k,
                "use_mmr": use_mmr,
                "use_reranking": use_rerank,
                "max_tokens": max_tokens,
                **({"llm_model": selected_model} if selected_model else {}),
            }

            stream_label = next(
                (m["label"] for m in AVAILABLE_MODELS if m["id"] == selected_model),
                selected_model.split("/")[-1] if selected_model else "DeepSeek-R1",
            )
            placeholder = st.empty()
            sources: list = []
            metrics: dict = {}
            full_answer = ""

            try:
                with requests.get(
                    f"{BACKEND}/chat/stream",
                    params=params,
                    stream=True,
                    timeout=300,
                    headers={"Accept": "text/event-stream"},
                ) as r:
                    r.raise_for_status()
                    current_event = None
                    for raw in r.iter_lines(decode_unicode=True):
                        if not raw:
                            current_event = None
                            continue
                        if raw.startswith("event:"):
                            current_event = raw[6:].strip()
                        elif raw.startswith("data:"):
                            data = raw[5:].strip()
                            if current_event == "sources":
                                sources = json.loads(data)
                            elif current_event == "token":
                                tok = json.loads(data).get("token", "")
                                if not tok:
                                    continue
                                # RESET signal: model wrote a draft then **Answer:**
                                # — discard the draft and start fresh
                                if tok.startswith("\x00RESET\x00"):
                                    full_answer = tok[len("\x00RESET\x00"):]
                                else:
                                    full_answer += tok
                                placeholder.markdown(
                                    f"<div class='chat-ai'>"
                                    f"<div class='label label-ai'>Agentur Philipp RAG · {stream_label} · streaming</div>"
                                    f"{full_answer.replace(chr(10), '<br>')}<span style='opacity:.5'>▍</span>"
                                    f"</div>",
                                    unsafe_allow_html=True,
                                )
                            elif current_event == "metrics":
                                metrics = json.loads(data)
                            elif current_event == "error":
                                err = json.loads(data).get("error", "?")
                                full_answer += f"\n\n[Stream error: {err}]"
                            elif current_event == "done":
                                break
            except Exception as e:
                full_answer = full_answer or f"Streaming error: {e}"

            st.session_state.chat.append({
                "role": "assistant",
                "content": full_answer,
                "sources": sources,
                "metrics": metrics,
                "llm_model": selected_model or "",
            })
            st.rerun()
        else:
            payload = {"session_id": st.session_state.session_id, "query": query,
                       "top_k": top_k, "use_mmr": use_mmr, "use_reranking": use_rerank,
                       "max_tokens": max_tokens,
                       **({"llm_model": selected_model} if selected_model else {})}
            with st.spinner("Generating answer…"):
                try:
                    r = requests.post(f"{BACKEND}/chat", json=payload, timeout=180)
                    r.raise_for_status()
                    data = r.json()
                    st.session_state.chat.append({
                        "role": "assistant",
                        "content": data["answer"],
                        "sources": data["sources"],
                        "metrics": data["metrics"],
                        "llm_model": data.get("llm_model", selected_model or ""),
                    })
                except Exception as e:
                    st.session_state.chat.append({"role": "assistant", "content": f"Error: {e}", "sources": [], "metrics": {}})
            st.rerun()

# ─── TEST SUITE TAB ───────────────────────────────────────────
with tab_tests:
    st.markdown("### 🧪 Automated Test Suite")
    st.markdown("Run 5 hardcoded questions against your uploaded documents. "
                "Select a question set that matches your document type.")

    if not QUESTION_SETS:
        st.warning("Could not reach the backend. Click **Retry connection** in the sidebar, then wait ~30 s for the backend to warm up.")
        if st.button("🔄 Retry now", key="retry_qs"):
            st.cache_data.clear()
            st.rerun()
        q_set = None
        run_tests = False
    else:
        col_sel, col_run = st.columns([2, 1])
        q_set = col_sel.selectbox("Question Set", list(QUESTION_SETS.keys()),
                                   help="Generic works with any PDF. Domain sets give more targeted questions.")
        run_tests = col_run.button("▶ Run All 5 Tests", use_container_width=True,
                                   disabled=not st.session_state.session_id)

        # Preview questions
        with st.expander("👁️ Preview questions in this set", expanded=False):
            for q in QUESTION_SETS[q_set]:
                st.markdown(f"**Q{q['id']} ({q['category']}):** {q['question']}")
                st.caption(f"Expected keywords: `{', '.join(q['expected_keywords'][:5])}`")

    if run_tests:
        with st.spinner(f"Running {q_set} test suite (this may take 1-2 minutes)…"):
            try:
                payload = {"session_id": st.session_state.session_id, "question_set": q_set,
                           "top_k": top_k, "use_reranking": use_rerank,
                           **({"llm_model": selected_model} if selected_model else {})}
                r = requests.post(f"{BACKEND}/test-suite", json=payload, timeout=600)
                r.raise_for_status()
                result = r.json()

                summary = result.get("summary", {})
                st.markdown("#### 📊 Summary")
                cs = st.columns(5)
                for col, (lbl, key) in zip(cs, [
                    ("Faithfulness", "faithfulness"),
                    ("Ctx Relevancy", "context_relevancy"),
                    ("Ans Relevancy", "answer_relevancy"),
                    ("Overall", "overall_score"),
                    ("Keyword Hit", "avg_keyword_hit_rate"),
                ]):
                    v = summary.get(key, 0)
                    col.markdown(
                        f"<div class='metric-box'><div class='metric-val'>{v:.0%}</div>"
                        f"<div class='metric-lbl'>{lbl}</div></div>",
                        unsafe_allow_html=True)

                st.markdown("#### 📋 Per-Question Results")
                for res in result["results"]:
                    overall = res.get("metrics", {}).get("overall_score", 0)
                    khr = res.get("keyword_hit_rate", 0)
                    badge_cls = "badge-good" if overall >= 0.6 else "badge-fair" if overall >= 0.35 else "badge-poor"
                    with st.expander(
                        f"Q{res['id']} · {res['category']} · "
                        f"Overall: {overall:.0%} · Keyword Hit: {khr:.0%}", expanded=False
                    ):
                        st.markdown(f"**Question:** {res['question']}")
                        st.markdown(f"**Answer:** {res['answer']}")
                        m = res.get("metrics", {})
                        if m:
                            c1, c2, c3, c4, c5 = st.columns(5)
                            for col, lbl, key in [
                                (c1, "Faithfulness", "faithfulness"),
                                (c2, "Ctx Rel.", "context_relevancy"),
                                (c3, "Ans Rel.", "answer_relevancy"),
                                (c4, "Avg Ret.", "avg_retrieval_score"),
                                (c5, "Overall", "overall_score"),
                            ]:
                                col.metric(lbl, f"{m.get(key, 0):.0%}")
                        st.metric("🔑 Keyword Hit Rate", f"{khr:.0%}")
                        if res.get("sources"):
                            st.markdown("**Sources used:**")
                            for s in res["sources"]:
                                st.markdown(
                                    f"<div class='src-card'><b>{s['file']}</b> · Page {s['page']} "
                                    f"<span class='score-badge' style='background:rgba(69,123,157,.12);color:#457b9d'>"
                                    f"{s['score']:.2f}</span><br><small>{s['excerpt'][:200]}</small></div>",
                                    unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Test suite failed: {e}")

# ─── ABOUT TAB ────────────────────────────────────────────────
with tab_info:
    st.markdown("### ℹ️ About This Application")
    st.markdown("""
**RAG Document Chat** — built for Agentur Philipp GmbH technical assessment.

#### 🔧 Tech Stack
| Layer | Technology |
|---|---|
| **Frontend** | Streamlit |
| **Backend** | FastAPI + Uvicorn |
| **Embeddings** | BAAI/bge-small-en-v1.5 (HF Serverless API) |
| **LLM** | DeepSeek-R1-Distill-Qwen-7B (HF Serverless API) |
| **Vector Store** | ChromaDB (HNSW index, cosine similarity) |
| **Chunking** | RecursiveCharacterTextSplitter (512 chars, 64 overlap) |
| **Reranking ✨** | cross-encoder/ms-marco-MiniLM-L-6-v2 (bonus) |
| **Streaming ✨** | FastAPI SSE + Streamlit (bonus) |
| **Deployment** | Docker Compose |

#### 📐 Chunking Strategy
- **size=512, overlap=64** balances retrieval precision with context coverage
- `RecursiveCharacterTextSplitter` respects paragraph → sentence → word boundaries
- Overlap prevents facts split across chunk boundaries from being lost

#### 📊 Evaluation Metrics
- **Faithfulness**: trigram overlap between answer and retrieved context
- **Context Relevancy**: bigram Jaccard between query and retrieved chunks
- **Answer Relevancy**: keyword hit-rate of query terms in answer
- **Keyword Hit Rate**: custom metric — fraction of expected keywords found in answer
""")

# Footer
st.markdown("""
<div style='margin-top:28px;padding:16px 0 8px;border-top:1px solid #e2e8f0;text-align:center'>
  <span style='background:#1d3557;color:#a8dadc;font-size:.68rem;font-weight:700;
    letter-spacing:.08em;text-transform:uppercase;padding:5px 14px;border-radius:999px'>
    📄 Agentur Philipp GmbH · RAG Document Chat
  </span>
  <div style='color:#94a3b8;font-size:.68rem;margin-top:6px'>
    DeepSeek-R1 · BAAI/bge-small-en-v1.5 · ChromaDB HNSW · FastAPI · Streamlit
  </div>
</div>""", unsafe_allow_html=True)
