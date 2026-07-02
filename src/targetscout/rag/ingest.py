"""Phase 2: ingest PubMed abstracts into pgvector.

    python -m targetscout.rag.ingest --query "EGFR inhibitor hERG" --n 100

Skeleton — implement the marked steps in Phase 2 of the roadmap.
"""
from __future__ import annotations
import argparse

from targetscout.data import pubmed


def ingest(query: str, n: int = 100) -> int:
    ids = pubmed.search(query, retmax=n)
    docs = pubmed.fetch_abstracts(ids)
    # TODO(Phase 2):
    #   1. chunk each abstract (config.rag.chunk_size / overlap)
    #   2. embed chunks with the biomedical embedder (config.models.biomedical_embedder)
    #   3. upsert into pgvector (table: evidence_chunks) with pmid + url metadata
    print(f"Fetched {len(docs)} abstracts for '{query}'. TODO: chunk + embed + upsert.")
    return len(docs)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True)
    ap.add_argument("--n", type=int, default=100)
    a = ap.parse_args()
    ingest(a.query, a.n)
