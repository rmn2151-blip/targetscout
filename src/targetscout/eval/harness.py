"""Eval harness: LLM-as-judge faithfulness + relevancy over the golden set.

    python -m targetscout.eval.harness          # run + print metrics
    python -m targetscout.eval.harness --ci      # exit non-zero if below thresholds
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys
from pathlib import Path

from targetscout.config import settings
from targetscout.rag.retriever import retrieve

GOLDEN = Path(__file__).parent / "golden" / "qa_seed.jsonl"


def _llm(prompt: str, max_tokens: int = 700) -> str:
    """Send a prompt to whichever LLM is configured; return the text reply.

    Priority: 1) LLM_BASE_URL (free: Ollama/Groq)  2) Anthropic  3) OpenAI.
    """
    base_url = os.getenv("LLM_BASE_URL")
    if base_url:
        from openai import OpenAI
        client = OpenAI(base_url=base_url, api_key=os.getenv("LLM_API_KEY", "not-needed"))
        r = client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "llama3.1"),
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )
        return r.choices[0].message.content

    env = settings()["env"]
    if env.get("anthropic_api_key"):
        import anthropic
        client = anthropic.Anthropic(api_key=env["anthropic_api_key"])
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text
    if env.get("openai_api_key"):
        from openai import OpenAI
        client = OpenAI(api_key=env["openai_api_key"])
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )
        return r.choices[0].message.content
    raise RuntimeError("No LLM configured. Set LLM_BASE_URL (free) or an API key in .env")


def _score(text: str) -> float:
    m = re.search(r"(0(?:\.\d+)?|1(?:\.0+)?)", text or "")
    return float(m.group(1)) if m else 0.0


def rag_answer(question: str):
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
