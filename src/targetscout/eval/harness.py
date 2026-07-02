"""Eval harness + CI gate.

    python -m targetscout.eval.harness         # run + print metrics
    python -m targetscout.eval.harness --ci     # exit non-zero if below thresholds

Metrics:
  - RAGAS faithfulness / context precision / answer relevancy   (RAG quality)
  - LLM-as-judge citation accuracy                              (are citations real?)
  - Retrospective ranking ROC-AUC / enrichment                 (domain validity)

Phase 2 wires RAGAS; Phase 4 adds the judge + retrospective ranking.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

from targetscout.config import settings

GOLDEN = Path(__file__).parent / "golden" / "qa_seed.jsonl"


def load_golden() -> list[dict]:
    if not GOLDEN.exists():
        return []
    return [json.loads(line) for line in GOLDEN.read_text().splitlines() if line.strip()]


def run_eval() -> dict:
    golden = load_golden()
    # TODO(Phase 2/4): run each golden question through the agent, score with RAGAS,
    # LLM-as-judge, and retrospective ranking. Placeholder metrics for now:
    metrics = {
        "n_golden": len(golden),
        "faithfulness": None,
        "citation_accuracy": None,
        "retrospective_roc_auc": None,
    }
    return metrics


def main(ci: bool) -> None:
    metrics = run_eval()
    print(json.dumps(metrics, indent=2))
    if ci:
        thr = settings()["eval"]["thresholds"]
        failures = [
            f"{k}={metrics.get(k)} < {v}"
            for k, v in thr.items()
            if metrics.get(k) is not None and metrics[k] < v
        ]
        if failures:
            print("EVAL GATE FAILED:", "; ".join(failures))
            sys.exit(1)
        print("Eval gate passed (or metrics not yet implemented).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ci", action="store_true")
    main(ap.parse_args().ci)
