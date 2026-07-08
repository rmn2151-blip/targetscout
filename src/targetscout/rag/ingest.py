"""Phase 2: ingest PubMed abstracts into pgvector."""
from __future__ import annotations
import argparse

import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

from targetscout.config import settings
from targetscout.data import pubmed

_MODEL_NAME = settings()["models"]["biomedical_embedder"]
_model = SentenceTransformer(_MODEL_NAME)
_DIM = _model.get_sentence_embedding_dimension()


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    text = " ".join(text.split())
    if not text:
        return []
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks


def _connect():
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
            chunk     TEXT,
            embedding VECTOR({_DIM})
        )
        """
    )
    conn.commit()
    return conn


def ingest(query: str, n: int = 100) -> int:
    cfg = settings()["rag"]
    pmids = pubmed.search(query, retmax=n)
    docs = pubmed.fetch_abstracts(pmids)
    rows = []
    for d in docs:
        text = f"{d['title']}. {d['abstract']}"
        for ch in chunk_text(text, cfg["chunk_size"], cfg["chunk_overlap"]):
            rows.append((d["pmid"], d["title"], d["url"], ch))
    if not rows:
        print(f"No text to ingest for '{query}'.")
        return 0
    texts = [r[3] for r in rows]
    embeddings = _model.encode(texts, batch_size=32, show_progress_bar=False)
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True)
    ap.add_argument("--n", type=int, default=100)
    a = ap.parse_args()
    ingest(a.query, a.n)
