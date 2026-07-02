"""Run the triage agent end-to-end.

    python -m targetscout.agents.run --target P00533
"""
from __future__ import annotations
import argparse
import json


def main(target: str) -> None:
    from targetscout.agents.graph import build_graph

    app = build_graph()
    result = app.invoke({"target": target})
    print(json.dumps({
        "target": result.get("protein", {}).get("name"),
        "n_candidates": len(result.get("candidates", [])),
        "report": result.get("report"),
    }, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="P00533")
    main(ap.parse_args().target)
