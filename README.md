# Enterprise AI Operations Copilot — Module 8 RAG Baseline

Educational project built incrementally: RAG today; HR, Accounting and Legal agents later.
All included policies are **fictional training samples**, not real company policies.

## Setup (Python >=3.11 and uv)

```bash
uv sync --extra rag --extra dev
cp .env.example .env  # Windows PowerShell: Copy-Item .env.example .env
```

The embedding model is downloaded once on first run: internet access is required unless pre-cached.

## Index synthetic documents

```bash
uv run --extra rag python -m scripts.ingest_documents
```

This creates a persistent local Chroma index under `data/processed/chroma`.

## Run API

```bash
uv run uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000/docs. POST to `/api/v1/rag/ask`:

```json
{"question":"How many days of annual leave after one year?","department":"hr","top_k":3}
```

Without `OPENAI_API_KEY`, the endpoint returns the **retrieved evidence only**, explicitly labeled `retrieval_only`. This is real dense retrieval, not LLM answer generation. To enable optional generation, set `OPENAI_API_KEY` (in an untracked `.env`) and `OPENAI_MODEL` to an available model. Do not send private documents to external APIs without authorization.

## Tests

```bash
uv run --extra dev pytest -q
```

## Known boundaries

- No user auth/RBAC yet: ALL DATA IS SYNTHETIC. Domain is selected by the caller; this is **not** adequate isolation for production.
- Chroma local persistence is a classroom setup, not a hardened production database.
- PDF parsing handles text-based PDFs. Scanned PDFs require an OCR pipeline.
- First build: fixed-size chunking and dense retrieval; hybrid retrieval, reranking, evaluation, permission enforcement, and agent orchestration come later.
- Avoid committing `.env`, model downloads or Chroma index; see `.gitignore`.
