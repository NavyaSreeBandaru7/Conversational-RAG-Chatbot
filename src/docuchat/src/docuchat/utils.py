from __future__ import annotations
from pathlib import Path
from typing import Iterable
from langchain_core.documents import Document

def safe_source_name(doc: Document) -> str:
    src = doc.metadata.get("source", "unknown")
    return Path(src).name

def safe_page(doc: Document):
    # PyPDFLoader stores 0-indexed page in metadata["page"]
    p = doc.metadata.get("page", None)
    if p is None:
        return None
    try:
        return int(p) + 1
    except Exception:
        return None

def format_docs_for_context(docs: list[Document], max_chars: int = 12_000) -> str:
    """
    Build a context string with lightweight citations. Hard cap size to avoid huge prompts.
    """
    chunks = []
    total = 0
    for d in docs:
        src = safe_source_name(d)
        page = safe_page(d)
        tag = f"[{src}:p{page}]" if page else f"[{src}]"
        block = f"{tag}\n{d.page_content}".strip()
        if total + len(block) > max_chars:
            break
        chunks.append(block)
        total += len(block)
    return "\n\n".join(chunks)

def format_sources(docs: Iterable[Document]) -> list[str]:
    seen = set()
    out = []
    for d in docs:
        src = safe_source_name(d)
        page = safe_page(d)
        tag = f"{src}:p{page}" if page else src
        if tag not in seen:
            seen.add(tag)
            out.append(tag)
    return out
