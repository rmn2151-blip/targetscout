"""FastAPI service exposing the triage agent."""
from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="TargetScout", version="0.1.0")


class TriageRequest(BaseModel):
    target: str = "P00533"      # UniProt accession or symbol
    disease: str | None = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/triage")
def triage(req: TriageRequest) -> dict:
    from targetscout.agents.graph import build_graph

    app_graph = build_graph()
    result = app_graph.invoke({"target": req.target, "disease": req.disease})
    return {
        "target": result.get("protein", {}).get("name"),
        "n_candidates": len(result.get("candidates", [])),
        "report": result.get("report"),
        "evidence": result.get("evidence", []),
    }
