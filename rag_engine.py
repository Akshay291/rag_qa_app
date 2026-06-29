"""
rag_engine.py — RAG Document Q&A System
Multi-model free stack: Groq LLM + local sentence-transformers + ChromaDB
"""

import os
import hashlib
import tempfile
from typing import List, Tuple

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters             import RecursiveCharacterTextSplitter
from langchain_huggingface                import HuggingFaceEmbeddings # type: ignore
from langchain_groq                       import ChatGroq # type: ignore
from langchain_community.vectorstores     import Chroma
from langchain_core.documents             import Document
from langchain_core.prompts               import PromptTemplate
from langchain_core.output_parsers        import StrOutputParser
from langchain_core.runnables             import RunnablePassthrough

from config import (
    EMBEDDING_MODEL, TEMPERATURE,
    CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_RESULTS,
    CHROMA_PERSIST, COLLECTION_NAME, RAG_SYSTEM_PROMPT
)

# ── Shared embedding instance (loaded once, reused) ───────────────────────────
_embeddings = None

def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name    = EMBEDDING_MODEL,
            model_kwargs  = {"device": "cpu"},
            encode_kwargs = {"normalize_embeddings": True},
        )
    return _embeddings


# ── 1. Load PDF ───────────────────────────────────────────────────────────────
def load_pdf(uploaded_file) -> Tuple[List[Document], dict]:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    docs   = loader.load()
    os.unlink(tmp_path)

    for doc in docs:
        doc.metadata["source_file"] = uploaded_file.name
        doc.metadata["page"]        = doc.metadata.get("page", 0) + 1

    info = {
        "filename" : uploaded_file.name,
        "pages"    : len(docs),
        "file_size": round(uploaded_file.size / 1024, 1),
        "doc_hash" : hashlib.md5(uploaded_file.name.encode()).hexdigest()[:8],
    }
    return docs, info


# ── 2. Chunk documents ────────────────────────────────────────────────────────
def chunk_documents(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size      = CHUNK_SIZE,
        chunk_overlap   = CHUNK_OVERLAP,
        length_function = len,
        separators      = ["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
    return chunks


# ── 3. Vector store ───────────────────────────────────────────────────────────
def build_vectorstore(chunks: List[Document], api_key: str = None) -> Chroma: # type: ignore
    return Chroma.from_documents(
        documents        = chunks,
        embedding        = get_embeddings(),
        persist_directory= CHROMA_PERSIST,
        collection_name  = COLLECTION_NAME,
    )

def load_existing_vectorstore(api_key: str = None) -> Chroma: # type: ignore
    return Chroma(
        persist_directory  = CHROMA_PERSIST,
        collection_name    = COLLECTION_NAME,
        embedding_function = get_embeddings(),
    )

def add_to_vectorstore(vectorstore: Chroma, chunks: List[Document]) -> Chroma:
    vectorstore.add_documents(chunks)
    return vectorstore

def clear_vectorstore():
    import shutil
    if os.path.exists(CHROMA_PERSIST):
        shutil.rmtree(CHROMA_PERSIST)


# ── 4. Build RAG chain (accepts model_id so user can switch) ──────────────────
def build_rag_chain(vectorstore: Chroma, api_key: str, model_id: str):
    retriever = vectorstore.as_retriever(
        search_type   = "similarity",
        search_kwargs = {"k": TOP_K_RESULTS},
    )
    llm = ChatGroq(
        model        = model_id,
        groq_api_key = api_key,
        temperature  = TEMPERATURE,
        streaming    = True,
    )
    prompt = PromptTemplate(
        input_variables = ["context", "question"],
        template        = RAG_SYSTEM_PROMPT,
    )

    def format_docs(docs: List[Document]) -> str:
        parts = []
        for doc in docs:
            src  = doc.metadata.get("source_file", "Unknown")
            page = doc.metadata.get("page", "?")
            parts.append(f"--- Source: {src} | Page {page} ---\n{doc.page_content}")
        return "\n\n".join(parts)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever


# ── 5. Retrieve + stream ──────────────────────────────────────────────────────
def get_source_docs(retriever, question: str) -> List[Document]:
    return retriever.invoke(question)

def stream_answer(chain, question: str):
    for chunk in chain.stream(question):
        yield chunk


# ── 6. Utility ────────────────────────────────────────────────────────────────
def get_collection_count(vectorstore: Chroma) -> int:
    try:
        return vectorstore._collection.count()
    except Exception:
        return 0
