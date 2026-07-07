"""Eval harness: LLM-as-judge faithfulness + relevancy over the golden set.

    python -m targetscout.eval.harness          # run + print metrics
    python -m targetscout.eval.harness --ci      # exit non-zero if below thresholds

For each golden question we:
  1. retrieve evidence chunks (your Phase 2 retriever),
  2. have an LLM answer using ONLY that evidence (a grounded RAG answer),
  3. have an LLM *judge* score how faithful + relevant that answer is.

Needs an LLM key in .env (ANTHROPIC_API_KEY or OPENAI_API_KEY).
"""
from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path

from targetscout.config import settings
from targetscout.rag.retriever import retrieve

GOLDEN = Path(__file__).parent / "golden" / "qa_seed.jsonl"


def _llm(prompt: str, max_tokens: int = 700) -> str:
    """Send a prompt to whichever LLM key is configured; return the text reply."""
    env = settings()["env"]
    if env.get("anthropic_api_key"):
        import anthropic
        client = anthropic.Anthropic(api_key=env["anthropic_api_key"])
        msg = client.messages.create(
            model="claude-3-5-haiku-latest",           # cheap; change if it errors
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text
    if env.get("openai_api_key"):
        from openai import OpenAI
        client = OpenAI(api_key=env["openai_api_key"])
        r = client.chat.completions.create(
            model="gpt-4o-mini",                        # cheap; change if it errors
            messages=[{"role": "user", "content": prompt}],
        )
        return r.choices[0].message.content
    raise RuntimeError("No LLM key found. Add ANTHROPIC_API_KEY or OPENAI_API_KEY to .env")


def _score(text: str) -> float:
    """Pull the first 0..1 number out of the judge's reply."""
    m = re.search(r"(0(?:\.\d+)?|1(?:\.0+)?)", text)
    return float(m.group(1)) if m else 0.0


def rag_answer(question: str):
    """Retrieve evidence, then answer using ONLY that evidence."""
    evidence = retrieve(question)
    context = "\n\n".join(f"[{e.pmid}] {e.text}" for e in evidence)
    prompt = (
        "Answer the question using ONLY the context below. "
        "Cite sources inline as [PMID]. If the context does not contain the answer, "
        "say 'The evidence does not cover this.'\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    )
    return _llm(prompt), evidence, context


def judge_faithfulness(answer: str, context: str) -> float:
    p = (
        "Grade whether an ANSWER is supported by CONTEXT. Reply with ONE number from "
        "0 to 1: the fraction of the answer's claims directly supported by the context. "
        "Output only the number.\n\n"
        f"CONTEXT:\n{context}\n\nANSWER:\n{answer}"
    )
    return _score(_llm(p, max_tokens=10))


def judge_relevancy(answer: str, question: str) -> float:
    p = (
        "Reply with ONE number from 0 to 1 for how well the ANSWER addresses the "
        "QUESTION (1 = fully answers, 0 = irrelevant). Output only the number.\n\n"
        f"QUESTION:\n{question}\n\nANSWER:\n{answer}"
    )
    return _score(_llm(p, max_tokens=10))


def load_golden() -> list[dict]:
    return [json.loads(line) for line in GOLDEN.read_text().splitlines() if line.strip()]


def run_eval() -> dict:
    golden = load_golden()
    items = [g for g in golden if g.get("type") in {"evidence_qa", "data_qa"}]
    faiths, rels = [], []
    for g in items:
        answer, evidence, context = rag_answer(g["question"])
        f = judge_faithfulness(answer, context)
        r = judge_relevancy(answer, g["question"])
        faiths.append(f)
        rels.append(r)
        print(f"- {g['id']}: faithfulness={f:.2f} relevancy={r:.2f} ({len(evidence)} chunks)")
    avg = lambda xs: round(sum(xs) / len(xs), 3) if xs else None
    return {"n_evaluated": len(items),
            "faithfulness": avg(faiths),
            "answer_relevancy": avg(rels)}


def main(ci: bool) -> None:
    metrics = run_eval()
    print("\n== EVAL RESULTS ==")
    print(json.dumps(metrics, indent=2))
    if ci and metrics["faithfulness"] is not None:
        thr = settings()["eval"]["thresholds"]["faithfulness"]
        if metrics["faithfulness"] < thr:
            print(f"EVAL GATE FAILED: faithfulness {metrics['faithfulness']} < {thr}")
            sys.exit(1)
        print("Eval gate passed.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ci", action="store_true")
    main(ap.parse_args().ci)
