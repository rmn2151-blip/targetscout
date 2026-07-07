"""Run the triage agent end-to-end.  python -m targetscout.agents.run --target P00533"""
from __future__ import annotations
import argparse


def main(target: str) -> None:
    from targetscout.agents.graph import build_graph

    result = build_graph().invoke({"target": target})
    print("Target:     ", result.get("protein", {}).get("name"))
    print("Candidates: ", len(result.get("candidates", [])))
    print("Scored mols:", len(result.get("admet", {})))
    print("Evidence:   ", len(result.get("evidence", [])))
    safety = result.get("safety", {})
    ranked = result.get("ranked", [])
    n_flagged = sum(1 for f in safety.values() if f)
    print(f"Flagged:     {n_flagged} of {len(safety)} molecules have liabilities")
    if ranked:
        top = ranked[0]
        print("Safest pick:", top[:45], "| flags:", safety.get(top) or "none")
    print("Report:     ", result.get("report"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="P00533")
    main(ap.parse_args().target)
