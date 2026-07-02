"""Phase 3: the TargetScout LangGraph state machine.

Nodes: planner -> target_resolver -> candidate_retriever -> property_predictor
       -> evidence_retriever -> safety_checker -> synthesizer
with a reflection edge (synthesizer -> evidence_retriever) when confidence is low.

This is a runnable skeleton: it defines the state + graph shape with stubbed nodes
so you can `python -m targetscout.agents.run --target EGFR` and see the flow, then
fill each node in during Phase 3.
"""
from __future__ import annotations
from typing import Annotated, TypedDict


class TriageState(TypedDict, total=False):
    target: str            # symbol or UniProt/ChEMBL id
    disease: str           # optional entry point
    protein: dict          # UniProt record
    candidates: list       # [{smiles, chembl_id, pchembl}]
    admet: dict            # {smiles: {hERG: .., solubility: ..}}
    evidence: list         # [{pmid, text, url}]
    confidence: float
    report: str


def build_graph():
    """Return a compiled LangGraph app. Requires `langgraph` installed."""
    from langgraph.graph import END, StateGraph

    from targetscout.agents import nodes

    g = StateGraph(TriageState)
    g.add_node("planner", nodes.planner)
    g.add_node("target_resolver", nodes.target_resolver)
    g.add_node("candidate_retriever", nodes.candidate_retriever)
    g.add_node("property_predictor", nodes.property_predictor)
    g.add_node("evidence_retriever", nodes.evidence_retriever)
    g.add_node("safety_checker", nodes.safety_checker)
    g.add_node("synthesizer", nodes.synthesizer)

    g.set_entry_point("planner")
    g.add_edge("planner", "target_resolver")
    g.add_edge("target_resolver", "candidate_retriever")
    g.add_edge("candidate_retriever", "property_predictor")
    g.add_edge("property_predictor", "evidence_retriever")
    g.add_edge("evidence_retriever", "safety_checker")
    g.add_edge("safety_checker", "synthesizer")

    # reflection loop: re-retrieve evidence if the synthesizer is unsure
    g.add_conditional_edges(
        "synthesizer",
        nodes.needs_more_evidence,
        {"retry": "evidence_retriever", "done": END},
    )
    return g.compile()
