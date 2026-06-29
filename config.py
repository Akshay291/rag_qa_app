"""
config.py — RAG Document Q&A System
Multi-model free stack: Groq (LLM) + sentence-transformers (local embeddings)
"""

# ── Embeddings: 100% local, no API, no cost ──────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # ~90 MB, downloaded once, cached

# ── RAG settings ─────────────────────────────────────────────────────────────
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K_RESULTS = 4
CHROMA_PERSIST = "./chroma_db"
COLLECTION_NAME = "rag_documents"

# Key: display name shown in UI   Value: Groq model ID
FREE_MODELS = {
    "🚀 GPT-OSS 20B  — Fastest 1 000 tok/s (1 000 req/day)": "openai/gpt-oss-20b",
    "🧠 GPT-OSS 120B — Best Quality 500 tok/s (1 000 req/day)": "openai/gpt-oss-120b",
    "🌟 Qwen 3.6 27B — Coding & Vision 500 tok/s (1 000 req/day)": "qwen/qwen3.6-27b",
}

# Default model shown when page loads
DEFAULT_MODEL = "🧠 GPT-OSS 120B — Best Quality 500 tok/s (1 000 req/day)"

# Model cards info (shown in the selector UI)
MODEL_INFO = {
    "🚀 GPT-OSS 20B  — Fastest 1 000 tok/s (1 000 req/day)": {
        "params": "20B",
        "speed": "1 000 tok/s",
        "ctx": "128K",
        "best_for": "Real-time responses, fastest model on Groq",
        "provider": "OpenAI",
    },
    "🧠 GPT-OSS 120B — Best Quality 500 tok/s (1 000 req/day)": {
        "params": "120B",
        "speed": "500 tok/s",
        "ctx": "128K",
        "best_for": "Complex reasoning, detailed answers, flagship quality",
        "provider": "OpenAI",
    },
    "🌟 Qwen 3.6 27B — Coding & Vision 500 tok/s (1 000 req/day)": {
        "params": "27B",
        "speed": "500 tok/s",
        "ctx": "131K",
        "best_for": "Agentic coding, multilingual docs, complex analysis",
        "provider": "Alibaba",
    },
}

TEMPERATURE = 0.2
MAX_OUTPUT_TOKENS = 2048

# ── Prompts ──────────────────────────────────────────────────────────────────
RAG_SYSTEM_PROMPT = """You are an expert document analyst and question-answering assistant. You have been given relevant excerpts from one or more documents as context.

Your job:
1. Answer the user's question ONLY using the provided context
2. Be accurate, concise, and structured
3. Always cite which document/page the information comes from
4. If the answer is not in the context, say: "I couldn't find this information in the uploaded documents."
5. Use markdown formatting for clarity (headers, bullet points, bold)
6. Never make up information not present in the context

Format your answers clearly:
- Start with a direct answer
- Provide supporting details from the context
- End with the source citation

Context from documents:
{context}

Question: {question}

Answer:"""

GREETING = """👋 **Welcome to the RAG Document Q&A System!**

Upload one or more PDF documents using the sidebar, then ask me anything about them.

**I can help you:**
- 📋 Summaries key sections
- 🔍 Find specific information quickly
- 📊 Compare information across documents
- 💡 Answer detailed questions with source citations
- 🧠 Explain complex concepts from your documents

**Select a model, upload your PDFs, and start chatting →**"""

EXAMPLE_QUESTIONS = [
    "What is the main topic of this document?",
    "Summaries the key points in bullet points",
    "What are the most important findings?",
    "What does the document say about [topic]?",
    "List all recommendations mentioned",
    "What conclusions are drawn?",
]
