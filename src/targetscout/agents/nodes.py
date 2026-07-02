"""LangGraph node implementations (Phase 3).

Each node takes the TriageState and returns a partial state update.
The target_resolver and candidate_retriever are already wired to real data
sources; the ML/RAG/LLM nodes are stubbed with clear TODOs.
"""
from __future__ import annotations

from targetscout.data import chembl, uniprot


def planner(state: dict) -> dict:
    # TODO: LLM decomposition of the request into a plan. For now, pass through.
    return {}


def target_resolver(state: dict) -> dict:
    """Resolve a target symbol/accession to a UniProt record (real API call)."""
    accession = state.get("target", "P00533")
    # In Phase 3, map free-text symbol -> accession via UniProt search first.
    protein = uniprot.get_protein(accession if accession.startswith("P") else "P00533")
    return {"protein": protein}


def candidate_retriever(state: dict) -> dict:
    """Pull known bioactive compounds from ChEMBL (real API call)."""
    acts = chembl.activities_for_target(limit=25)
    candidates = [
        {"smiles": a["smiles"], "chembl_id": a["molecule_chembl_id"], "pchembl": a["pchembl_value"]}
        for a in acts if a["smiles"]
    ]
    return {"candidates": candidates}


def property_predictor(state: dict) -> dict:
    # TODO(Phase 3): load MLflow ADMET models, featurize candidate SMILES, predict.
    return {"admet": {}}


def evidence_retriever(state: dict) -> dict:
    # TODO(Phase 3): call rag.retriever.retrieve(query) for grounded evidence.
    return {"evidence": []}


def safety_checker(state: dict) -> dict:
    # TODO(Phase 3): flag candidates with predicted hERG/DILI/Ames liabilities.
    return {}


def synthesizer(state: dict) -> dict:
    # TODO(Phase 3): frontier LLM writes a cited report from candidates+admet+evidence.
    return {"report": "TODO: synthesized cited report", "confidence": 1.0}


def needs_more_evidence(state: dict) -> str:
    """Reflection gate: loop back if confidence is low and we haven't retried enough."""
    return "done" if state.get("confidence", 1.0) >= 0.7 else "retry"
