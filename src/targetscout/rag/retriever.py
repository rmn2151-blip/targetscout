"""Phase 2: two-stage retrieve (vector search + cross-encoder rerank).

    python -m targetscout.rag.retriever "Which EGFR inhibitors cause hERG toxicity?"
"""
from __future__ import annotations
from dataclasses import dataclass

import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import CrossEncoder, SentenceTransformer

from targetscout.config import settings

_cfg = settings()
_embedder = SentenceTransformer(_cfg["models"]["biomedical_embedder"])
_reranker = CrossEncoder(_cfg["models"]["reranker"])


@dataclass
class Evidence:
    pmid: str
    title: str
    text: str
    url: str
    score: float


def retrieve(query: str, top_k: int | None = None, rerank_k: int | None = None) -> list[Evidence]:
    top_k = top_k or _cfg["rag"]["top_k"]
    rerank_k = rerank_k or _cfg["rag"]["rerank_k"]

    q_emb = _embedder.encode(query)

    conn = psycopg.connect(_cfg["env"]["database_url"])
    register_vector(conn)
    rows = conn.execute(
        """
        SELECT pmid, title, url, chunk
        FROM evidence_chunks
        ORDER BY embedding <=> %s
        LIMIT %s
        """,
        (q_emb, top_k),
    ).fetchall()
    conn.close()

    if not rows:
        return []

    pairs = [(query, r[3]) for r in rows]
    scores = _reranker.predict(pairs)

    ranked = sorted(zip(rows, scores), key=lambda pair: pair[1], reverse=True)[:rerank_k]
    return [
        Evidence(pmid=r[0], title=r[1], text=r[3], url=r[2], score=float(s))
        for r, s in ranked
    ]


if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "Which EGFR inhibitors cause hERG cardiotoxicity?"
    for ev in retrieve(q):
        print(f"[{ev.score:.2f}] {ev.pmid}  {ev.title[:70]}")
        print(f"     {ev.text[:160]}...")
        print(f"     {ev.url}\n")
