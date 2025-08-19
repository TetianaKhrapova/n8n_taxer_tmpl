import os
import re
import hashlib
from typing import List, Iterable
from dotenv import load_dotenv

# OpenAI v1 SDK
from openai import OpenAI

import tiktoken

# Optional parsers
from bs4 import BeautifulSoup
from pypdf import PdfReader

# Load .env once
load_dotenv()

# --- Config ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "800"))
OVERLAP = int(os.getenv("OVERLAP", "200"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "64"))

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set. Create a .env based on .env.example.")

client = OpenAI(api_key=OPENAI_API_KEY)

# For OpenAI embeddings, 'cl100k_base' is appropriate
ENC = tiktoken.get_encoding("cl100k_base")

def sanitize_id(s: str) -> str:
    """Sanitize a string to be used as a Chroma id."""
    s = re.sub(r"[^a-zA-Z0-9_-]+", "_", s)
    return s[:64]  # keep ids short but stable

def file_to_text(path: str) -> str:
    """Read a file and extract text (txt/md/pdf/html supported)."""
    ext = os.path.splitext(path)[1].lower()
    if ext in [".txt", ".md"]:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    if ext in [".html", ".htm"]:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            soup = BeautifulSoup(f.read(), "html.parser")
            return soup.get_text(" ", strip=True)
    if ext == ".pdf":
        reader = PdfReader(path)
        text = []
        for page in reader.pages:
            try:
                text.append(page.extract_text() or "")
            except Exception:
                text.append("")
        return "\n".join(text)
    raise ValueError(f"Unsupported file type: {ext}")

def chunk_by_tokens(text: str, max_tokens: int = MAX_TOKENS, overlap: int = OVERLAP) -> List[str]:
    """Split text into token chunks with overlap using tiktoken."""
    tokens = ENC.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunks.append(ENC.decode(chunk_tokens))
        if end == len(tokens):
            break
        start = max(0, end - overlap)
    return chunks

def embed_texts(texts: List[str], model: str = EMBED_MODEL) -> List[List[float]]:
    """Call OpenAI embedding API in batches for a list of texts."""
    out = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]
        resp = client.embeddings.create(model=model, input=batch)
        out.extend([d.embedding for d in resp.data])
    return out

def walk_raw_files(root: str) -> Iterable[str]:
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if os.path.splitext(fn)[1].lower() in [".txt", ".md", ".pdf", ".html", ".htm"]:
                yield os.path.join(dirpath, fn)

def make_chunk_id(file_path: str, idx: int, content: str) -> str:
    base = f"{file_path}:{idx}"
    h = hashlib.sha1((base + content).encode("utf-8")).hexdigest()[:10]
    return sanitize_id(f"{os.path.basename(file_path)}_{idx}_{h}")
