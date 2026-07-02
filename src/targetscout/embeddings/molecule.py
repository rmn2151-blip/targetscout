"""Molecule featurizers (HF task: Feature Extraction).

Default: RDKit physicochemical descriptors + Morgan fingerprint — fast, no GPU,
strong ADMET baseline. Optionally concatenate ChemBERTa embeddings.
"""
from __future__ import annotations
import numpy as np


def rdkit_features(smiles: str) -> np.ndarray | None:
    """Return descriptor + Morgan-fingerprint vector for a SMILES string."""
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    desc = np.array([
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.RingCount(mol),
        Descriptors.FractionCSP3(mol),
    ], dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
    fp_arr = np.zeros((1024,), dtype=np.float32)
    AllChem.DataStructs.ConvertToNumpyArray(fp, fp_arr)
    return np.concatenate([desc, fp_arr])


def featurize_many(smiles_list: list[str]):
    feats, keep = [], []
    for i, smi in enumerate(smiles_list):
        f = rdkit_features(smi)
        if f is not None:
            feats.append(f)
            keep.append(i)
    return np.vstack(feats), keep
