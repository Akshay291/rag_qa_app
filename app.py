"""
app.py — DocMind RAG Document Q&A System
Multi-model free stack: Groq LLM + local embeddings + ChromaDB
Resume Project #4 (STAR) | Akshay Kiran Rajput
Run: streamlit run app.py
"""

import streamlit as st
import os
from rag_engine import (
    load_pdf,
    chunk_documents,
    build_vectorstore,
    add_to_vectorstore,
    build_rag_chain,
    get_source_docs,
    stream_answer,
    clear_vectorstore,
    get_collection_count,
)
from config import (
    GREETING,
    EXAMPLE_QUESTIONS,
    FREE_MODELS,
    MODEL_INFO,
    DEFAULT_MODEL,
)

st.set_page_config(
    page_title="DocMind — RAG Q&A",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROVIDER_COLOR = {
    "OpenAI": "#06D6A0",
    "Alibaba": "#C77DFF",
}

# ─────────────────────────────────────────────────────────
# GLOBAL CSS  (only structural things that don't need
# per-element dynamic values go here; everything dynamic
# is inlined in the HTML strings below)
# ─────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }

[data-testid="stAppViewContainer"] {
    background: #0B0F1A !important;
    font-family: 'Inter', sans-serif;
    color: #C9C4E8;
}
[data-testid="stSidebar"] {
    background: #0D1120 !important;
    border-right: 1px solid #1C2235 !important;
}
[data-testid="stSidebar"] > div:first-child { padding-top: 1.2rem; }

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.6rem !important; padding-bottom: 4rem !important; max-width: 1100px; }

h1, h2, h3, h4 { font-family: 'Space Grotesk', sans-serif !important; color: #F0EDFF !important; letter-spacing: -.02em; }

/* Streamlit widget overrides */
.stTextInput > label { color: #9BA8C0 !important; font-size: 12px !important; }
.stTextInput > div > div > input {
    background: #131929 !important; border: 1px solid #1C2235 !important;
    color: #F0EDFF !important; border-radius: 8px !important; font-family: 'JetBrains Mono', monospace !important;
}
.stTextInput > div > div > input:focus { border-color: #FF5E5B !important; box-shadow: 0 0 0 2px rgba(255,94,91,.15) !important; }

.stButton > button {
    font-family: 'Space Grotesk', sans-serif !important; font-weight: 600 !important;
    border-radius: 8px !important; transition: all .15s !important;
}
.stButton > button[kind="primary"] {
    background: #FF5E5B !important; border: none !important; color: #fff !important;
}
.stButton > button[kind="primary"]:hover {
    background: #ff3f3c !important; transform: translateY(-1px); box-shadow: 0 4px 18px rgba(255,94,91,.4) !important;
}
.stButton > button:not([kind="primary"]) {
    background: #131929 !important; border: 1px solid #1C2235 !important; color: #C9C4E8 !important;
}
.stButton > button:not([kind="primary"]):hover { border-color: #FF5E5B !important; color: #F0EDFF !important; }

.stProgress > div > div > div { background: linear-gradient(90deg, #FF5E5B, #FFD166) !important; }

[data-testid="stChatMessage"] {
    background: #0D1120 !important; border: 1px solid #1C2235 !important;
    border-radius: 12px !important; margin-bottom: 10px !important;
}
div[data-testid="stExpander"] {
    background: #131929 !important; border: 1px solid #1C2235 !important; border-radius: 8px !important;
}
div[data-testid="stExpander"] summary { color: #A8AECB !important; font-size: 13px !important; }

.stChatInput > div {
    background: #0D1120 !important; border: 1px solid #1C2235 !important; border-radius: 12px !important;
}

[data-testid="stFileUploader"] {
    background: #131929 !important; border: 1px dashed #4A5580 !important; border-radius: 10px !important;
}

hr { border-color: #1C2235 !important; margin: 20px 0 !important; }
</style>
""",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────
# HELPERS — return HTML strings (all styles inlined)
# ─────────────────────────────────────────────────────────


def _hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))


def html_tag(
    text,
    color,
    size="10px",
    weight="600",
    spacing=".12em",
    upper=True,
    family="'JetBrains Mono',monospace",
):
    transform = "text-transform:uppercase;" if upper else ""
    return (
        f'<div style="font-family:{family};font-size:{size};font-weight:{weight};'
        f'letter-spacing:{spacing};{transform}color:{color};margin-bottom:5px">{text}</div>'
    )


def html_chip(text):
    return (
        f"<span style=\"display:inline-block;font-family:'JetBrains Mono',monospace;"
        f"font-size:10px;padding:3px 9px;border-radius:4px;"
        f'background:#131929;color:#A8AECB;border:1px solid #1C2235;margin:2px 3px 2px 0">'
        f"{text}</span>"
    )


def html_model_card(name, info, is_active, pcolor):
    r, g, b = _hex_to_rgb(pcolor)
    bg = f"rgba({r},{g},{b},.07)" if is_active else "#0D1120"
    border = pcolor if is_active else "#1C2235"
    clean = name.split("—")[0].strip().lstrip("⚡🧠🔭🔬💎🌐 ").strip()

    badge = (
        (
            f'<div style="position:absolute;top:12px;right:12px;background:{pcolor};'
            f"color:#0B0F1A;font-family:monospace;font-size:9px;font-weight:700;"
            f'padding:2px 9px;border-radius:3px;letter-spacing:.06em">SELECTED</div>'
        )
        if is_active
        else ""
    )

    chips = "".join(
        html_chip(c) for c in [info["params"], info["speed"], f"{info['ctx']} ctx"]
    )

    return (
        f'<div style="position:relative;background:{bg};border:1px solid {border};'
        f"border-left:3px solid {pcolor};border-radius:12px;padding:16px 16px 14px;"
        f'margin-bottom:2px;min-height:148px">'
        f"{badge}"
        f"{html_tag(info['provider'], pcolor)}"
        f"<div style=\"font-family:'Space Grotesk',sans-serif;font-size:15px;font-weight:700;"
        f'color:#F0EDFF;margin-bottom:10px;line-height:1.25">{clean}</div>'
        f'<div style="margin-bottom:9px">{chips}</div>'
        f'<div style="font-size:11px;color:#9BA8C0;line-height:1.5">{info["best_for"]}</div>'
        f"</div>"
    )


def html_doc_pill(doc):
    name = doc["filename"]
    short = name[:24] + ("…" if len(name) > 24 else "")
    return (
        f'<div style="display:flex;align-items:center;gap:8px;background:#131929;'
        f'border:1px solid #1C2235;border-radius:8px;padding:8px 10px;margin-bottom:6px">'
        f'<div style="font-size:14px;flex-shrink:0">📄</div>'
        f'<div><div style="font-size:12px;font-weight:600;color:#C9C4E8;'
        f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:150px">{short}</div>'
        f'<div style="font-size:10px;color:#8892B0;font-family:monospace">'
        f"{doc['pages']}p · {doc['file_size']}KB</div></div></div>"
    )


def html_stat_grid(docs, chunks, qa_count):
    pages = sum(d["pages"] for d in docs)

    def box(val, lbl):
        return (
            f'<div style="background:#131929;border:1px solid #1C2235;border-radius:8px;'
            f'padding:10px 8px;text-align:center">'
            f"<div style=\"font-family:'Space Grotesk',sans-serif;font-size:20px;"
            f'font-weight:700;color:#F0EDFF;line-height:1">{val}</div>'
            f'<div style="font-family:monospace;font-size:9px;color:#8892B0;'
            f'letter-spacing:.08em;margin-top:3px">{lbl}</div></div>'
        )

    return (
        f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;margin:10px 0">'
        f"{box(len(docs), 'DOCS')}{box(pages, 'PAGES')}"
        f"{box(chunks, 'CHUNKS')}{box(qa_count, 'Q&As')}"
        f"</div>"
    )


def html_source_box(j, fname, page, text):
    preview = text[:420] + ("…" if len(text) > 420 else "")
    return (
        f'<div style="background:#131929;border:1px solid #1C2235;border-left:3px solid #FF5E5B;'
        f'border-radius:8px;padding:12px 14px;margin:6px 0">'
        f'<div style="font-family:monospace;font-size:10px;color:#FF5E5B;'
        f'letter-spacing:.08em;text-transform:uppercase;margin-bottom:6px;font-weight:500">'
        f"↳ {fname} — p.{page}</div>"
        f'<div style="font-size:12px;color:#A8AECB;line-height:1.65">{preview}</div>'
        f"</div>"
    )


# ─────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────
defaults = {
    "messages": [],
    "vectorstore": None,
    "chain": None,
    "retriever": None,
    "uploaded_docs": [],
    "api_key_valid": False,
    "total_chunks": 0,
    "qa_count": 0,
    "selected_model": DEFAULT_MODEL,
    "active_model_id": FREE_MODELS[DEFAULT_MODEL],
    "chat_started": False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ═════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════
with st.sidebar:
    # Brand
    st.markdown(
        '<div style="display:flex;align-items:center;gap:10px;margin-bottom:6px">'
        '<div style="width:34px;height:34px;background:linear-gradient(135deg,#FF5E5B,#FFD166);'
        "border-radius:8px;display:flex;align-items:center;justify-content:center;"
        'font-size:16px;flex-shrink:0">🧠</div>'
        "<div><div style=\"font-family:'Space Grotesk',sans-serif;font-size:18px;"
        'font-weight:700;color:#F0EDFF;line-height:1.1">DocMind</div>'
        '<div style="font-size:11px;color:#9BA8C0;font-family:monospace">RAG Document Q&amp;A</div>'
        "</div></div>"
        '<div style="display:inline-flex;align-items:center;gap:5px;'
        "background:rgba(6,214,160,.12);border:1px solid rgba(6,214,160,.3);"
        "border-radius:20px;padding:3px 12px;font-size:11px;font-weight:600;"
        'color:#06D6A0;font-family:monospace;letter-spacing:.03em;margin:8px 0 16px">'
        "⚡ 100% FREE · NO CREDIT CARD</div>",
        unsafe_allow_html=True,
    )

    # API Key
    st.markdown(
        '<div style="font-family:monospace;font-size:10px;font-weight:500;'
        'letter-spacing:.12em;color:#8892B0;text-transform:uppercase;margin:0 0 8px">Groq API Key</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="font-size:11px;color:#8892B0;margin-bottom:8px">'
        'Free at <a href="https://console.groq.com/keys" target="_blank" '
        'style="color:#FFD166;text-decoration:none">console.groq.com/keys</a> — no card needed</div>',
        unsafe_allow_html=True,
    )
    api_key = st.text_input(
        "API Key", type="password", placeholder="gsk_...", label_visibility="collapsed"
    )
    if api_key:
        st.session_state.api_key_valid = True
        os.environ["GROQ_API_KEY"] = api_key

    if st.session_state.api_key_valid:
        st.markdown(
            '<div style="display:flex;align-items:center;gap:6px;font-size:12px;'
            'color:#06D6A0;font-family:monospace;margin-top:6px">'
            '<div style="width:7px;height:7px;background:#06D6A0;border-radius:50%;'
            'box-shadow:0 0 6px #06D6A0"></div>Connected to Groq</div>',
            unsafe_allow_html=True,
        )

    # Upload
    st.markdown(
        '<div style="font-family:monospace;font-size:10px;font-weight:500;'
        'letter-spacing:.12em;color:#8892B0;text-transform:uppercase;margin:18px 0 8px">Documents</div>',
        unsafe_allow_html=True,
    )
    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    process_btn = st.button(
        "Process Documents →",
        use_container_width=True,
        type="primary",
        disabled=not (uploaded_files and st.session_state.api_key_valid),
    )

    # Display toggle
    st.markdown(
        '<div style="font-family:monospace;font-size:10px;font-weight:500;'
        'letter-spacing:.12em;color:#8892B0;text-transform:uppercase;margin:18px 0 8px">Display</div>',
        unsafe_allow_html=True,
    )
    show_sources = st.toggle("Show source passages", value=True)

    # Loaded docs
    if st.session_state.uploaded_docs:
        st.markdown(
            '<div style="font-family:monospace;font-size:10px;font-weight:500;'
            'letter-spacing:.12em;color:#8892B0;text-transform:uppercase;margin:18px 0 8px">Loaded</div>',
            unsafe_allow_html=True,
        )
        for doc in st.session_state.uploaded_docs:
            st.markdown(html_doc_pill(doc), unsafe_allow_html=True)

        st.markdown(
            html_stat_grid(
                st.session_state.uploaded_docs,
                st.session_state.total_chunks,
                st.session_state.qa_count,
            ),
            unsafe_allow_html=True,
        )
        st.markdown('<div style="height:4px"></div>', unsafe_allow_html=True)
        if st.button("Clear all documents", use_container_width=True):
            clear_vectorstore()
            for k, v in defaults.items():
                st.session_state[k] = v
            st.rerun()

    st.markdown(
        '<div style="margin-top:24px;font-family:monospace;font-size:10px;'
        'color:#4A5580;line-height:1.8">'
        "all-MiniLM-L6-v2 · ChromaDB<br>PyPDF · LangChain · Groq LPU</div>",
        unsafe_allow_html=True,
    )


# ═════════════════════════════════════════════════════════
# MAIN — HERO
# ═════════════════════════════════════════════════════════
st.markdown(
    '<div style="background:#0D1120;border:1px solid #1C2235;border-radius:16px;'
    'padding:28px 32px 22px;margin-bottom:24px">'
    '<div style="font-family:monospace;font-size:11px;color:#FF5E5B;'
    'letter-spacing:.1em;text-transform:uppercase;margin-bottom:6px">'
    "Resume Project #4 · Akshay Kiran Rajput</div>"
    "<div style=\"font-family:'Space Grotesk',sans-serif;font-size:32px;font-weight:700;"
    'color:#F0EDFF;line-height:1.15;letter-spacing:-.03em;margin-bottom:8px">'
    'Ask anything about your <span style="color:#FF5E5B">documents.</span></div>'
    '<div style="font-size:14px;color:#A8AECB;line-height:1.6;max-width:540px;margin-bottom:14px">'
    "Upload PDFs, pick a free AI model, and get instant answers with exact page citations — "
    "no subscriptions, no credit cards, runs entirely on Groq's free tier.</div>"
    '<div style="display:flex;flex-wrap:wrap;gap:6px">'
    '<span style="font-family:monospace;font-size:10px;padding:4px 10px;border-radius:4px;'
    'color:#FF5E5B;border:1px solid rgba(255,94,91,.3);background:rgba(255,94,91,.08)">LangChain</span>'
    '<span style="font-family:monospace;font-size:10px;padding:4px 10px;border-radius:4px;'
    'color:#06D6A0;border:1px solid rgba(6,214,160,.3);background:rgba(6,214,160,.08)">ChromaDB</span>'
    '<span style="font-family:monospace;font-size:10px;padding:4px 10px;border-radius:4px;'
    'color:#FFD166;border:1px solid rgba(255,209,102,.3);background:rgba(255,209,102,.08)">Groq Free Tier</span>'
    '<span style="font-family:monospace;font-size:10px;padding:4px 10px;border-radius:4px;'
    'color:#C77DFF;border:1px solid rgba(199,125,255,.3);background:rgba(199,125,255,.08)">Local Embeddings</span>'
    '<span style="font-family:monospace;font-size:10px;padding:4px 10px;border-radius:4px;'
    'color:#FF5E5B;border:1px solid rgba(255,94,91,.3);background:rgba(255,94,91,.08)">'
    "GPT-OSS · Qwen 3.6</span>"
    "</div></div>",
    unsafe_allow_html=True,
)

# Pipeline strip
steps = [
    ("📄", "PDF Upload"),
    ("✂️", "Chunking"),
    ("🔢", "Local Embed"),
    ("🗄", "ChromaDB"),
    ("🔍", "Retrieval"),
    ("🤖", "Groq LLM"),
    ("💬", "Answer"),
]
step_html = ""
for i, (icon, label) in enumerate(steps):
    step_html += (
        f'<div style="display:flex;align-items:center;gap:5px;font-size:12px;'
        f'color:#A8AECB;padding:4px 10px">'
        f"<span>{icon}</span><span>{label}</span></div>"
    )
    if i < len(steps) - 1:
        step_html += '<span style="color:#4A5580;font-size:16px;margin:0 2px">›</span>'

st.markdown(
    f'<div style="display:flex;align-items:center;flex-wrap:wrap;gap:0;'
    f"background:#131929;border:1px solid #1C2235;border-radius:10px;"
    f'padding:12px 16px;margin-bottom:24px">{step_html}</div>',
    unsafe_allow_html=True,
)


# ═════════════════════════════════════════════════════════
# PROCESS PDFs
# ═════════════════════════════════════════════════════════
if process_btn and uploaded_files:
    if not st.session_state.api_key_valid:
        st.error("Enter your Groq API key first.")
    else:
        progress = st.progress(0, text="Starting…")
        all_chunks = []
        new_doc_info = []
        total = len(uploaded_files)

        for i, f in enumerate(uploaded_files):
            already = [d["filename"] for d in st.session_state.uploaded_docs]
            if f.name in already:
                st.info(f"'{f.name}' is already loaded — skipping.")
                continue
            progress.progress(i / total, text=f"Reading {f.name}…")
            docs, info = load_pdf(f)
            progress.progress(
                (i + 0.4) / total, text=f"Splitting {f.name} into chunks…"
            )
            chunks = chunk_documents(docs)
            all_chunks.extend(chunks)
            new_doc_info.append({**info, "chunks": len(chunks)})

        if all_chunks:
            progress.progress(
                0.75, text="Embedding chunks locally — this takes a moment…"
            )
            if st.session_state.vectorstore is None:
                vs = build_vectorstore(all_chunks, api_key)
            else:
                vs = add_to_vectorstore(st.session_state.vectorstore, all_chunks)

            progress.progress(0.92, text="Wiring up the RAG chain…")
            chain, retriever = build_rag_chain(
                vs, api_key, st.session_state.active_model_id
            )

            st.session_state.vectorstore = vs
            st.session_state.chain = chain
            st.session_state.retriever = retriever
            st.session_state.uploaded_docs += new_doc_info
            st.session_state.total_chunks = get_collection_count(vs)

            progress.progress(1.0, text="Done!")
            st.success(
                f"Processed {len(new_doc_info)} file(s) — "
                f"{sum(d['chunks'] for d in new_doc_info)} chunks stored in ChromaDB."
            )
            if not st.session_state.messages:
                st.session_state.messages.append(
                    {"role": "assistant", "content": GREETING}
                )
            st.rerun()
        else:
            progress.empty()
            st.warning("All those files are already loaded.")


# ═════════════════════════════════════════════════════════
# NO KEY NOTICE
# ═════════════════════════════════════════════════════════
if not st.session_state.api_key_valid:
    st.markdown(
        '<div style="background:rgba(255,209,102,.06);border:1px solid rgba(255,209,102,.2);'
        "border-left:3px solid #FFD166;border-radius:8px;padding:12px 16px;"
        'font-size:13px;color:#C9C4E8;margin:12px 0">'
        "🔑 <strong>Paste your Groq API key in the sidebar to get started.</strong><br>"
        'Completely free at <a href="https://console.groq.com/keys" target="_blank" '
        'style="color:#FFD166">console.groq.com/keys</a> — just an email, no credit card.</div>',
        unsafe_allow_html=True,
    )


# ═════════════════════════════════════════════════════════
# MODEL SELECTOR
# ═════════════════════════════════════════════════════════
if st.session_state.api_key_valid:
    if st.session_state.chat_started:
        # Compact active model banner
        info = MODEL_INFO[st.session_state.selected_model]
        pcolor = PROVIDER_COLOR.get(info["provider"], "#FF5E5B")
        r, g, b = _hex_to_rgb(pcolor)
        short = (
            st.session_state.selected_model.split("—")[0]
            .strip()
            .lstrip("⚡🧠🔭🔬💎🌐 ")
            .strip()
        )

        col_banner, col_btn = st.columns([5, 1])
        with col_banner:
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:10px;'
                f"background:#0D1120;border:1px solid #1C2235;border-radius:10px;"
                f'padding:12px 16px;margin-bottom:20px">'
                f'<div style="width:8px;height:8px;border-radius:50%;background:{pcolor};'
                f'box-shadow:0 0 8px rgba({r},{g},{b},.6);flex-shrink:0"></div>'
                f'<div><div style="font-family:monospace;font-size:9px;color:#8892B0;'
                f'letter-spacing:.1em;text-transform:uppercase">Active Model</div>'
                f"<div style=\"font-family:'Space Grotesk',sans-serif;font-size:14px;"
                f'font-weight:700;color:#F0EDFF">{short}</div></div>'
                f'<div style="margin-left:auto;font-family:monospace;font-size:10px;color:#9BA8C0">'
                f"{info['params']} · {info['speed']} · {info['ctx']} ctx · {info['provider']}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
        with col_btn:
            if st.button("Switch model", help="Clears chat, keeps documents"):
                st.session_state.chat_started = False
                st.session_state.messages = []
                st.session_state.chain = None
                st.session_state.retriever = None
                st.rerun()

    else:
        # ── Full model picker ──────────────────────────────────────
        st.markdown(
            '<div style="display:flex;align-items:center;gap:12px;margin-bottom:20px">'
            '<div style="width:32px;height:32px;flex-shrink:0;'
            "background:linear-gradient(135deg,#FF5E5B,#FFD166);border-radius:50%;"
            "display:flex;align-items:center;justify-content:center;"
            "font-family:'Space Grotesk',sans-serif;font-size:14px;font-weight:700;color:#0B0F1A\">1</div>"
            "<div><div style=\"font-family:'Space Grotesk',sans-serif;font-size:20px;"
            'font-weight:700;color:#F0EDFF">Pick your AI model</div>'
            '<div style="font-size:13px;color:#9BA8C0;margin-top:1px">'
            "All three run on Groq's free tier — no payment needed</div></div></div>",
            unsafe_allow_html=True,
        )

        model_names = list(FREE_MODELS.keys())
        cols = st.columns(2)

        for idx, name in enumerate(model_names):
            info = MODEL_INFO[name]
            is_active = name == st.session_state.selected_model
            pcolor = PROVIDER_COLOR.get(info["provider"], "#FF5E5B")

            with cols[idx % 2]:
                st.markdown(
                    html_model_card(name, info, is_active, pcolor),
                    unsafe_allow_html=True,
                )
                label = "✓ Selected" if is_active else "Select"
                btn_type = "primary" if is_active else "secondary"
                if st.button(
                    label, key=f"sel_{idx}", use_container_width=True, type=btn_type
                ):
                    st.session_state.selected_model = name
                    st.session_state.active_model_id = FREE_MODELS[name]
                    if st.session_state.vectorstore is not None:
                        chain, retriever = build_rag_chain(
                            st.session_state.vectorstore, api_key, FREE_MODELS[name]
                        )
                        st.session_state.chain = chain
                        st.session_state.retriever = retriever
                    st.rerun()

        st.markdown("<hr>", unsafe_allow_html=True)

        # ── Upload / Start section ─────────────────────────────────
        if not st.session_state.uploaded_docs:
            st.markdown(
                '<div style="display:flex;align-items:center;gap:12px;margin-bottom:16px">'
                '<div style="width:32px;height:32px;flex-shrink:0;'
                "background:linear-gradient(135deg,#FF5E5B,#FFD166);border-radius:50%;"
                "display:flex;align-items:center;justify-content:center;"
                "font-family:'Space Grotesk',sans-serif;font-size:14px;font-weight:700;color:#0B0F1A\">2</div>"
                "<div><div style=\"font-family:'Space Grotesk',sans-serif;font-size:20px;"
                'font-weight:700;color:#F0EDFF">Upload your PDFs</div>'
                '<div style="font-size:13px;color:#9BA8C0;margin-top:1px">'
                "Use the sidebar — research papers, reports, textbooks, contracts</div></div></div>"
                '<div style="background:#0D1120;border:1px dashed #4A5580;border-radius:12px;'
                'padding:36px 24px;text-align:center;margin:0 0 20px">'
                '<div style="font-size:36px;margin-bottom:10px">📂</div>'
                "<div style=\"font-family:'Space Grotesk',sans-serif;font-size:16px;"
                'font-weight:600;color:#F0EDFF;margin-bottom:4px">Drop your PDFs in the sidebar</div>'
                '<div style="font-size:12px;color:#9BA8C0">'
                "Supports any readable PDF · multiple files at once</div></div>",
                unsafe_allow_html=True,
            )
            st.markdown("**Things you could ask once uploaded:**")
            ecols = st.columns(2)
            for i, q in enumerate(EXAMPLE_QUESTIONS):
                with ecols[i % 2]:
                    st.markdown(f"- {q}")

        else:
            n = len(st.session_state.uploaded_docs)
            ch = st.session_state.total_chunks
            sel_short = (
                st.session_state.selected_model.split("—")[0]
                .strip()
                .lstrip("⚡🧠🔭🔬💎🌐 ")
                .strip()
            )
            st.markdown(
                '<div style="display:flex;align-items:center;gap:14px;'
                "background:rgba(6,214,160,.06);border:1px solid rgba(6,214,160,.2);"
                'border-radius:12px;padding:18px 20px;margin-bottom:16px">'
                '<div style="font-size:28px;flex-shrink:0">✅</div>'
                f"<div><div style=\"font-family:'Space Grotesk',sans-serif;font-size:16px;"
                f'font-weight:700;color:#F0EDFF">'
                f"{n} document{'s' if n > 1 else ''} ready · {ch} chunks indexed</div>"
                f'<div style="font-size:12px;color:#9BA8C0;margin-top:2px">'
                f'You\'ve selected <strong style="color:#F0EDFF">{sel_short}</strong>. '
                f"Hit the button below to start chatting.</div></div></div>",
                unsafe_allow_html=True,
            )
            if st.button("Start chatting →", use_container_width=True, type="primary"):
                st.session_state.chat_started = True
                if not st.session_state.messages:
                    st.session_state.messages.append(
                        {"role": "assistant", "content": GREETING}
                    )
                st.rerun()


# ═════════════════════════════════════════════════════════
# CHAT INTERFACE
# ═════════════════════════════════════════════════════════
if st.session_state.chat_started and st.session_state.chain:
    # Quick question buttons
    qcols = st.columns(3)
    for i, q in enumerate(EXAMPLE_QUESTIONS):
        with qcols[i % 3]:
            if st.button(
                q[:38] + ("…" if len(q) > 38 else ""),
                key=f"qbtn_{i}",
                use_container_width=True,
            ):
                st.session_state._pending_question = q

    st.markdown("<hr>", unsafe_allow_html=True)

    # Chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("model"):
                st.markdown(
                    f'<div style="font-family:monospace;font-size:10px;color:#8892B0;'
                    f'margin-top:8px;padding-top:8px;border-top:1px solid #1C2235">'
                    f"via {msg['model']}</div>",
                    unsafe_allow_html=True,
                )
            if msg.get("sources") and show_sources:
                with st.expander(f"View {len(msg['sources'])} source passages"):
                    for j, src in enumerate(msg["sources"], 1):
                        meta = src.metadata
                        fname = meta.get("source_file", "Document")
                        page = meta.get("page", "?")
                        st.markdown(
                            html_source_box(j, fname, page, src.page_content),
                            unsafe_allow_html=True,
                        )

    # Input
    pending = getattr(st.session_state, "_pending_question", None)
    user_question = st.chat_input("Ask something about your documents…")
    question = pending or user_question
    if pending:
        st.session_state._pending_question = None

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        sources = get_source_docs(st.session_state.retriever, question)

        with st.chat_message("assistant"):
            response = ""
            container = st.empty()
            for chunk in stream_answer(st.session_state.chain, question):
                response += chunk
                container.markdown(response + "▌")
            container.markdown(response)

            st.markdown(
                f'<div style="font-family:monospace;font-size:10px;color:#8892B0;'
                f'margin-top:8px;padding-top:8px;border-top:1px solid #1C2235">'
                f"via {st.session_state.active_model_id}</div>",
                unsafe_allow_html=True,
            )

            if show_sources and sources:
                with st.expander(f"View {len(sources)} source passages"):
                    for j, src in enumerate(sources, 1):
                        meta = src.metadata
                        fname = meta.get("source_file", "Document")
                        page = meta.get("page", "?")
                        st.markdown(
                            html_source_box(j, fname, page, src.page_content),
                            unsafe_allow_html=True,
                        )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response,
                "sources": sources,
                "model": st.session_state.active_model_id,
            }
        )
        st.session_state.qa_count += 1
        st.rerun()


# ═════════════════════════════════════════════════════════
# FOOTER
# ═════════════════════════════════════════════════════════
st.markdown(
    '<div style="text-align:center;font-family:monospace;font-size:10px;'
    'color:#4A5580;padding:16px 0 4px;letter-spacing:.05em">'
    "DOCMIND · RESUME PROJECT #4 ⭐ · AKSHAY KIRAN RAJPUT · "
    "LANGCHAIN · CHROMADB · GROQ · STREAMLIT"
    "</div>",
    unsafe_allow_html=True,
)
