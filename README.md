# 🧠 DocMind — RAG-Powered Document Q&A System

> **Resume Project #4 ⭐ STAR PROJECT** | Akshay Kiran Rajput | GenAI Developer Portfolio

Upload any PDF document and ask questions in plain English. DocMind uses **Retrieval-Augmented Generation (RAG)** to semantically search your documents and generate accurate, cited answers with exact page references — powered by **LangChain**, **ChromaDB**, **Groq LLM** (free tier), and **local HuggingFace embeddings** (zero API cost).

---

## 🔴 Live Demo
> 🚀 **[Open Streamlit App →](YOUR_STREAMLIT_URL_HERE)**

---

## 🏗 Architecture

```
📄 PDF Upload
     │
     ▼
 PyPDFLoader  →  RecursiveCharacterTextSplitter (1000 chars, 200 overlap)
     │
     ▼
 HuggingFace Embeddings (all-MiniLM-L6-v2, runs 100% locally, no API)
     │
     ▼
 ChromaDB Vector Store (persistent on disk)
     │
     ▼
 User Question  →  Cosine Similarity Search  →  Top-4 Chunks Retrieved
     │
     ▼
 RAG Prompt (context + question)  →  Groq LLM (free tier)  →  Cited Answer
     │
     ▼
 Streamlit Chat UI (streamed token-by-token + source passage expander)
```

**Key design decision:** Embeddings use `sentence-transformers` locally — so the embedding step is completely free with no API quota. Only the final generation step calls Groq (1,000 free requests/day).

---

## ✨ Features

| Feature | Details |
|---|---|
| **Multi-PDF Support** | Upload multiple PDFs and query across all of them simultaneously |
| **Persistent ChromaDB** | Vector store persists between sessions (`./chroma_db`) |
| **Cited Answers** | Every answer shows exact source file + page number |
| **Streaming Responses** | Token-by-token streaming via Groq for real-time feel |
| **Source Viewer** | Expandable source chunk passages per answer |
| **3-Model Selector** | Switch between GPT-OSS 20B, GPT-OSS 120B, and Qwen 3.6 27B |
| **Session Stats** | Live counters for docs loaded, pages, chunks indexed, Q&As answered |
| **Quick Questions** | One-click example questions to get started |
| **100% Free Stack** | Local embeddings + Groq free tier — no credit card required |

---

## 🤖 Available Models (all free on Groq)

| Model | Params | Speed | Context | Best For |
|---|---|---|---|---|
| GPT-OSS 20B | 20B | 1,000 tok/s | 128K | Real-time responses, fastest |
| GPT-OSS 120B | 120B | 500 tok/s | 128K | Complex reasoning, flagship quality |
| Qwen 3.6 27B | 27B | 500 tok/s | 131K | Agentic coding, multilingual docs |

---

## 📁 Project Structure

```
docmind/
├── app.py              ← Streamlit UI (run this)
├── rag_engine.py       ← Full RAG pipeline: load → chunk → embed → store → retrieve → generate
├── config.py           ← Model settings, RAG prompt, chunking params, example questions
├── requirements.txt
├── .env.example
├── README.md
└── chroma_db/          ← Auto-created persistent vector store (gitignored)
```

---

## ⚡ Run Locally

```bash
git clone https://github.com/Akshay291/docmind.git
cd docmind

pip install -r requirements.txt

# Option 1: Set key in .env
cp .env.example .env
# Paste your Groq key into .env

# Option 2: Enter key in sidebar at runtime

streamlit run app.py
```

Get a **free Groq API key** (no credit card) at [console.groq.com/keys](https://console.groq.com/keys)

---

## 🌐 Deploy Free on Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → Connect repo → Select `app.py`
3. Add `GROQ_API_KEY` in **Settings → Secrets**
4. Deploy — paste the live URL on your resume and LinkedIn

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| **LLM** | Groq (GPT-OSS 20B / 120B, Qwen 3.6 27B) — free tier |
| **Embeddings** | `sentence-transformers` · `all-MiniLM-L6-v2` — 100% local, no API |
| **Vector DB** | ChromaDB — persistent cosine similarity search |
| **RAG Framework** | LangChain (`langchain`, `langchain-community`, `langchain-groq`) |
| **PDF Loading** | PyPDFLoader (pypdf) |
| **Frontend** | Streamlit — dark theme with custom CSS |

---

## ✍ Resume Bullets

```
• Built DocMind, a production RAG system using LangChain + ChromaDB + Groq API —
  users upload PDFs and receive semantically-retrieved answers with exact page
  citations; deployed live on Streamlit Cloud

• Implemented full pipeline: PDF extraction (PyPDFLoader) → recursive chunking
  (1,000 chars / 200 overlap) → local HuggingFace embeddings (all-MiniLM-L6-v2,
  zero API cost) → ChromaDB persistent vector store → Top-K cosine similarity
  retrieval → streamed RAG generation with source attribution

• Integrated 3-model selector (GPT-OSS 20B, 120B, Qwen 3.6 27B) on Groq free
  tier; supports multi-PDF querying, persistent ChromaDB across sessions, and
  real-time token streaming with collapsible source chunk viewer per answer
```

---

## 💬 Interview Talking Points

**Q: What is RAG and why did you use it?**
> RAG combines retrieval with generation. Instead of relying on the LLM's training data, we retrieve relevant chunks from the user's own documents and inject them into the prompt context. This gives accurate, grounded answers on private or custom data the model was never trained on — and every answer is traceable to a specific page.

**Q: Why use local embeddings instead of an API?**
> `sentence-transformers` runs entirely on CPU — no API key, no quota, no cost. Only the final generation step uses Groq, keeping the system completely free. This also means embedding speed isn't throttled by network latency.

**Q: How does ChromaDB work?**
> ChromaDB is a vector database. We convert each text chunk into a high-dimensional embedding vector. When a user asks a question, we embed it the same way and find the most similar chunks using cosine similarity search. The top-4 results are injected into the LLM prompt as context.

**Q: What is chunking and why does overlap matter?**
> We split large documents into smaller pieces so they fit in the LLM's context window. Overlap of 200 characters ensures information at chunk boundaries isn't lost — if a key sentence spans two chunks, both chunks still contain it, improving retrieval recall.

**Q: How do you handle multiple PDFs?**
> Each PDF is loaded, chunked, and added to the same ChromaDB collection with `source_file` and `page` metadata. At retrieval time, the top-K search spans the entire collection — the answer can reference chunks from multiple documents simultaneously.

---

## 👤 Author

**Akshay Kiran Rajput** — MCA Student, Jain Online University
[LinkedIn](https://linkedin.com/in/akshay-rajput-0925b8264) · [GitHub](https://github.com/Akshay291) · Surat, Gujarat, India
