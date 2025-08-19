#!/usr/bin/env python3
import os
import argparse
from dotenv import load_dotenv

from chromadb import Client
from chromadb.config import Settings

from common import embed_texts

load_dotenv()

CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "tax_docs")
HNSW_SPACE = os.getenv("HNSW_SPACE", "cosine")
N_RESULTS = int(os.getenv("N_RESULTS", "5"))

def get_collection():
    client = Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=CHROMA_DIR))
    return client.get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": HNSW_SPACE})

def search(query: str, n_results: int = N_RESULTS):
    col = get_collection()
    q_emb = embed_texts([query])[0]
    res = col.query(query_embeddings=[q_emb], n_results=n_results, include=["documents", "metadatas", "distances"])
    return res

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query Chroma index built with OpenAI embeddings.")
    parser.add_argument("query", help="Your question")
    parser.add_argument("--n", type=int, default=N_RESULTS, help="Number of results")
    args = parser.parse_args()

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
