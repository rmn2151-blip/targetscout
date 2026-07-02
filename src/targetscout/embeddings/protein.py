"""ESM-2 protein embeddings (HF task: Feature Extraction).

Mean-pooled sequence embedding for a target protein. Used to condition
target-aware models and for target similarity search.
"""
from __future__ import annotations
import functools

from targetscout.config import settings


@functools.lru_cache(maxsize=1)
def _load():
    import torch  # noqa
    from transformers import AutoModel, AutoTokenizer

    name = settings()["models"]["protein_embedder"]
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name)
    model.eval()
    return tok, model


def embed_sequence(sequence: str):
    """Return a 1-D mean-pooled ESM-2 embedding for an amino-acid sequence."""
    import torch

    tok, model = _load()
    seq = sequence[:1022]  # ESM-2 max length
    inputs = tok(seq, return_tensors="pt", truncation=True, max_length=1024)
    with torch.no_grad():
        out = model(**inputs).last_hidden_state
    return out.mean(dim=1).squeeze().numpy()


if __name__ == "__main__":
    v = embed_sequence("MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGP")
    print("embedding dim:", v.shape)
