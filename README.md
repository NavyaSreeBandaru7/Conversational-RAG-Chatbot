# DocuChat — Conversational RAG Chatbot (LangChain + FAISS)

This project lets you "chat with your PDFs" using a production-style RAG pipeline:
- Load PDFs/TXT/MD from `docs/`
- Chunk + embed text
- Store vectors in FAISS index (`indexes/`)
- Conversational Q&A with sources (file + page)

## Setup
1) Create a virtual environment (recommended)
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
