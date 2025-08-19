#!/usr/bin/env python3
import os
import argparse
import requests
from dotenv import load_dotenv
from common import (
    file_to_text, chunk_by_tokens, embed_texts, walk_raw_files,
    make_chunk_id, EMBED_MODEL
)

load_dotenv()

CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", 5000))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "tax_docs")
N_RESULTS = int(os.getenv("N_RESULTS", "5"))

BASE_URL = f"http://{CHROMA_HOST}:{CHROMA_PORT}/api/v2/tenants/default_tenant/databases/default_database"


def ensure_collection():
    """Check if the collection exists, create if missing."""
    r = requests.get(f"{BASE_URL}/collections")
    r.raise_for_status()
    collections = [c["name"] for c in r.json()]

    if COLLECTION_NAME not in collections:
        print(f"[INFO] Creating collection: {COLLECTION_NAME}")
        r = requests.post(f"{BASE_URL}/collections", json={"name": COLLECTION_NAME})
        r.raise_for_status()


def ingest(root: str = "data/raw"):
    """Ingest documents from a folder into Chroma via REST API."""
    ensure_collection()
    total_chunks = 0

    for path in walk_raw_files(root):
        try:
            text = file_to_text(path)
            chunks = chunk_by_tokens(text)
        except Exception as e:
            print(f"[WARN] Skipping {path}: {e}")
            continue

        if not chunks:
            print(f"[INFO] No text in {path}, skipping.")
            continue

        ids, docs, metas = [], [], []
        for i, ch in enumerate(chunks):
            ids.append(make_chunk_id(path, i, ch))
            docs.append(ch)
            metas.append({
                "source": os.path.abspath(path),
                "chunk_index": i,
                "embed_model": EMBED_MODEL
            })

        embeddings = embed_texts(docs)

        payload = {
            "collection_name": COLLECTION_NAME,
            "ids": ids,
            "documents": docs,
            "metadatas": metas,
            "embeddings": embeddings
        }
        r = requests.post(f"{BASE_URL}/collections/{COLLECTION_NAME}/upsert", json=payload)
        r.raise_for_status()

        total_chunks += len(chunks)
        print(f"[OK] {path} → {len(chunks)} chunks")

    print(f"Done. Total chunks ingested: {total_chunks}")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Chroma REST API: {BASE_URL}")


def search(query: str, n_results: int = N_RESULTS):
    """Query the Chroma collection via REST API."""
    q_emb = embed_texts([query])[0]

    payload = {
        "collection_name": COLLECTION_NAME,
        "query_embeddings": [q_emb],
        "n_results": n_results,
        "include": ["documents", "metadatas", "distances"]
    }

    r = requests.post(f"{BASE_URL}/collections/{COLLECTION_NAME}/query", json=payload)
    r.raise_for_status()
    return r.json()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chroma REST: ingest or query documents")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents")
    ingest_parser.add_argument("--root", default="data/raw", help="Folder with source documents")

    query_parser = subparsers.add_parser("query", help="Query the collection")
    query_parser.add_argument("query", help="Your question")
    query_parser.add_argument("--n", type=int, default=N_RESULTS, help="Number of results")

    args = parser.parse_args()

    if args.command == "ingest":
        ingest(args.root)
    elif args.command == "query":
        result = search(args.query, args.n)
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        dists = result.get("distances", [[]])[0]

        print(f"Top {len(docs)} results:")
        for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists), start=1):
            print("-"*80)
            print(f"#{i}  distance={dist:.4f}")
            print(f"source: {meta.get('source')}  chunk: {meta.get('chunk_index')}")
            print(doc[:600].strip().replace("\n", " "))
            if len(doc) > 600:
                print("...")

