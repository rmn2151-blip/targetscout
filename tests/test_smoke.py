"""Lightweight unit tests that don't require network or heavy deps."""
from targetscout.embeddings.molecule import rdkit_features


def test_rdkit_features_shape():
    # aspirin
    v = rdkit_features("CC(=O)OC1=CC=CC=C1C(=O)O")
    assert v is not None
    assert v.shape[0] == 8 + 1024  # descriptors + Morgan bits


def test_rdkit_features_invalid_smiles():
    assert rdkit_features("not_a_molecule") is None
