"""Phase 2: hybrid retrieve + cross-encoder rerank over pgvector.

HF tasks: Sentence Similarity (dense retrieval) + Text Ranking (reranker).
Skeleton — implement in Phase 2.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class Evidence:
    pmid: str
    text: str
    url: str
    score: float


def retrieve(query: str, top_k: int = 20, rerank_k: int = 5) -> list[Evidence]:
    # TODO(Phase 2):
    #   1. embed query with biomedical embedder
    #   2. pgvector ANN search (top_k) + optional BM25 (hybrid)
    #   3. cross-encoder rerank -> keep rerank_k
    raise NotImplementedError("Implement in Phase 2 (see roadmap).")
