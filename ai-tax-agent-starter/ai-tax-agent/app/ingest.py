#!/usr/bin/env python3
import os
import argparse
from dotenv import load_dotenv
import requests
from common import (
    file_to_text, chunk_by_tokens, embed_texts, walk_raw_files,
    make_chunk_id, EMBED_MODEL
)

load_dotenv()

CHROMA_HOST = os.getenv("CHROMA_HOST", "host.docker.internal")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", 5000))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "tax_docs")

BASE_URL = f"http://{CHROMA_HOST}:{CHROMA_PORT}"

def get_collection():
    # Check if collection exists
    r = requests.get(f"{BASE_URL}/collections")
    r.raise_for_status()
    collections = [c["name"] for c in r.json().get("collections", [])]

    if COLLECTION_NAME not in collections:
        print(f"[INFO] Creating collection: {COLLECTION_NAME}")
        r = requests.post(f"{BASE_URL}/collections", json={"name": COLLECTION_NAME})
        r.raise_for_status()

def ingest(root: str = "data/raw"):
    get_collection()

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

        ids, docs, metas, embeddings = [], [], [], []
        for i, ch in enumerate(chunks):
            ids.append(make_chunk_id(path, i, ch))
            docs.append(ch)
            metas.append({
                "source": os.path.abspath(path),
                "chunk_index": i,
                "embed_model": EMBED_MODEL
            })

        embeddings = embed_texts(docs)

        # Upsert via REST
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents into Chroma via REST API.")
    parser.add_argument("--root", default="data/raw", help="Folder with source documents")
    args = parser.parse_args()
    ingest(args.root)
