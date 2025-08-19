# AI Tax Agent — ChromaDB + OpenAI Embeddings (No Local Transformers)

This starter shows how to build a local vector store with ChromaDB while using **OpenAI embeddings via API** (so no heavy local models).

## Structure
```
ai-tax-agent/
├─ app/
│  ├─ ingest.py       # Index documents into Chroma
│  ├─ query.py        # Ask questions against the index
│  └─ common.py       # Shared helpers (embedding, chunking, file reading)
├─ data/
│  └─ raw/            # Put your .txt/.md/.pdf/.html here
├─ chroma_db/         # On-disk Chroma store
├─ requirements.txt
└─ .env.example
```

## Quickstart

1) Create venv and install:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

2) Create `.env` from example and set your OpenAI key:
```bash
cp .env.example .env
# Edit .env and set OPENAI_API_KEY
```

3) Put your tax documents into `data/raw/` (txt, md, pdf, html).

4) Ingest (build index):
```bash
python app/ingest.py
```

5) Query:
```bash
python app/query.py "Як ФОП 2-ї групи сплачує єдиний податок у 2025 році?"
```

See also inline help:
```bash
python app/ingest.py --help
python app/query.py --help
```

## Notes
- No local transformers are downloaded; embeddings are computed via OpenAI API.
- Distance metric is set to **cosine** for better semantic search.
- Chunking uses token-based splits (tiktoken) with overlap to retain context.
