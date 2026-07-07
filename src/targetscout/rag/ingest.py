"""Phase 2: ingest PubMed abstracts into pgvector.

    python -m targetscout.rag.ingest --query "EGFR inhibitor hERG" --n 100

Fetches abstracts, splits them into chunks, turns each chunk into an
embedding (a list of numbers capturing its meaning), and stores them in
Postgres so we can later search by meaning instead of exact keywords.
"""
from __future__ import annotations
import argparse

import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

from targetscout.config import settings
from targetscout.data import pubmed

# Load the biomedical embedding model once (downloads the first time).
_MODEL_NAME = settings()["models"]["biomedical_embedder"]
_model = SentenceTransformer(_MODEL_NAME)
_DIM = _model.get_sentence_embedding_dimension()


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Cut a long string into overlapping windows of `size` characters."""
    text = " ".join(text.split())            # squash weird whitespace
    if not text:
        return []
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap              # step back a bit         start += size - overlap              # step back a bit         connection, enable pgvector, and make the table if needed."""
    conn = psycopg.connect(settings()["env"]["database_url"])
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS evidence_chunks (
            id        BIGSERIAL PRIMARY KEY,
            pmid      TEXT,
            title     TEXT,
            url       TEXT,
            chunk     TEXT,            chunk     TEXT,                   )
        """
    )
    conn.commit()
    return conn


def ingest(query: str, n: int = 100) -> int:
    cfg = settings()["rag"]

    # 1. Find and download abstracts from PubMed.
    pmids = pubmed.search(query, retmax=n)
    docs = pubmed.fetch_abstracts(pmids)

    # 2. Break each abstract into overlapping chunks.
    rows = []
    for d in docs:
        text = f"{d['title']}. {d['abstract']}"
        for ch in chunk_text(text, cfg["chunk_size"], cfg["chunk_overlap"]):
            rows.append((d["pmid"], d["title"], d["url"], ch))
    if not rows:
        print("No text to ingest.")
        return 0

    # 3. Turn every chunk into an embedding vector (all at once, batched).
    texts = [r[3] for r in rows]
    embeddings = _model.encode(texts,    embeddi=32, show_progress_bar=True)

    # 4. Save every chunk + its vector into Postgres.
    conn = _connect()
    with conn.cursor() as cur:
        for (pmid, title, url, chunk), emb in zip(rows, embeddings):
            cur.execute(
                "INSERT INTO evidence_chunks (pmid, title, url, chunk, embedding) "
                "VALUES (%s, %s, %s, %s, %s)",
                (pmid, title, url, chunk, emb),
            )
    conn.commit()
    conn.close()

    print(f"Ingested {len(rows)} chunks from {len(docs)} abstracts for '{query}'.")
    return len(rows)


if __name__ == "__main_if __name__ == "_arse.ArgumentParser()
    ap.add_argument("--query", required=True)
    ap.add_argument("--n", type=int, default=100)
    a = ap.parse_args()
    ingest(a.query, a.n)
