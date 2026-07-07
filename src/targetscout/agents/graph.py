"""Phase 3: the TargetScout LangGraph state machine."""
from __future__ import annotations
from typing import TypedDict


class TriageState(TypedDict, total=False):
    target: str
    disease: str
    protein: dict
    candidates: list
    admet: dict
    safety: dict
    ranked: list
    evidence: list
    confidence: float
    report: str


def build_graph():
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
    g.add_conditional_edges(
        "synthesizer",
        nodes.needs_more_evidence,
        {"retry": "evidence_retriever", "done": END},
    )
    return g.compile()
