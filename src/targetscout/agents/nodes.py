"""LangGraph node implementations (Phase 3)."""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path

from targetscout.data import chembl, uniprot

MODELS_DIR = Path("artifacts/admet")
CLASSIFICATION = {"hERG", "DILI", "BBB_Martins"}


@lru_cache(maxsize=1)
def _load_admet_models() -> dict:
    import joblib
    return {p.stem: joblib.load(p) for p in MODELS_DIR.glob("*.joblib")}


def planner(state: dict) -> dict:
    return {}


def target_resolver(state: dict) -> dict:
    accession = state.get("target", "P00533")
    protein = uniprot.get_protein(accession if accession.startswith("P") else "P00533")
    return {"protein": protein}


def candidate_retriever(state: dict) -> dict:
    acts = chembl.activities_for_target(limit=25)
    candidates = [
        {"smiles": a["smiles"], "chembl_id": a["molecule_chembl_id"], "pchembl": a["pchembl_value"]}
        for a in acts if a["smiles"]
    ]
    return {"candidates": candidates}


def property_predictor(state: dict) -> dict:
    from targetscout.embeddings.molecule import rdkit_features

    models = _load_admet_models()
    admet = {}
    for cand in state.get("candidates", []):
        smi = cand["smiles"]
        feats = rdkit_features(smi)
        if feats is None:
            continue
        X = feats.reshape(1, -1)
        preds = {}
        for name, model in models.items():
            if name in CLASSIFICATION:
                preds[name] = round(float(model.predict_proba(X)[0, 1]), 3)
            else:
                preds[name] = round(float(model.predict(X)[0]), 3)
        admet[smi] = preds
    return {"admet": admet}


def evidence_retriever(state: dict) -> dict:
    from targetscout.rag.retriever import retrieve

    protein = state.get("protein", {})
    genes = protein.get("genes") or []
    name = (genes[0] if genes else protein.get("name")) or "EGFR"
    query = f"{name} inhibitor hERG cardiotoxicity safety liability"
    hits = retrieve(query)
    evidence = [
        {"pmid": e.pmid, "title": e.title, "url": e.url,
         "text": e.text, "score": round(e.score, 3)}
        for e in hits
    ]
    return {"evidence": evidence}


def safety_checker(state: dict) -> dict:
    """Flag candidates over risk thresholds and rank them safest-first."""
    admet = state.get("admet", {})
    flagged = {}
    for smi, preds in admet.items():
        flags = []
        if preds.get("hERG", 0) >= 0.5:
            flags.append("hERG cardiotoxicity risk")
        if preds.get("DILI", 0) >= 0.5:
            flags.append("liver toxicity (DILI) risk")
        if preds.get("Solubility_AqSolDB", 0) < -4:
            flags.append("poor aqueous solubility")
        flagged[smi] = flags
    ranked = sorted(admet.keys(),
                    key=lambda s: (len(flagged[s]), admet[s].get("hERG", 1.0)))
    return {"safety": flagged, "ranked": ranked}


def synthesizer(state: dict) -> dict:
    return {"report": "TODO: synthesized cited report", "confidence": 1.0}


def needs_more_evidence(state: dict) -> str:
    return "done" if state.get("confidence", 1.0) >= 0.7 else "retry"
