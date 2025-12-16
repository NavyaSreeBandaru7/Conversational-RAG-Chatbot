from __future__ import annotations

import shutil
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from .config import get_settings, require_openai_key

def load_documents(docs_dir: Path):
    docs = []
    for p in docs_dir.rglob("*"):
        if p.is_dir():
            continue

        suffix = p.suffix.lower()
        if suffix == ".pdf":
            loader = PyPDFLoader(str(p))
            file_docs = loader.load()  # one doc per page
        elif suffix in [".txt", ".md"]:
            loader = TextLoader(str(p), encoding="utf-8")
            file_docs = loader.load()
        else:
            continue

        for d in file_docs:
            d.metadata = dict(d.metadata or {})
            d.metadata["source"] = str(p)
        docs.extend(file_docs)

    return docs

def main():
    load_dotenv()
    settings = get_settings()
    require_openai_key()

    if not settings.docs_dir.exists():
        raise SystemExit("Create ./docs and add PDFs/TXT/MD first.")

    raw_docs = load_documents(settings.docs_dir)
    if not raw_docs:
        raise SystemExit("No supported files found in ./docs (pdf/txt/md).")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = splitter.split_documents(raw_docs)

    embeddings = OpenAIEmbeddings(model=settings.embedding_model)

    # Rebuild index from scratch (simple + reliable for learning)
    if settings.index_dir.exists():
        shutil.rmtree(settings.index_dir)
    settings.index_dir.mkdir(parents=True, exist_ok=True)

    vs = FAISS.from_documents(chunks, embeddings)
    vs.save_local(str(settings.index_dir))

    print(f"Loaded docs: {len(raw_docs)}")
    print(f"Chunks: {len(chunks)}")
    print(f"✅ FAISS index saved to: {settings.index_dir}")

if __name__ == "__main__":
    main()
