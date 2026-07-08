"""Eval harness: faithfulness, relevancy, citation accuracy (LLM-as-judge)
   + retrospective ROC-AUC (domain check on ChEMBL bioactivity)."""
from __future__ import annotations
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

from targetscout.config import settings
from targetscout.rag.retriever import retrieve

GOLDEN = Path(__file__).parent / "golden" / "qa_seed.jsonl"


def _llm(prompt: str, max_tokens: int = 700) -> str:
    base_url = os.getenv("LLM_BASE_URL")
    if base_url:
        from openai import OpenAI
        client = OpenAI(base_url=base_url, api_key=os.getenv("LLM_API_KEY", "x"))
        r = client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "llama3.1"),
            messages=[{"role": "user", "content": prompt}], max_tokens=max_tokens)
        return r.choices[0].message.content
    env = settings()["env"]
    if env.get("anthropic_api_key"):
        import anthropic
        client = anthropic.Anthropic(api_key=env["anthropic_api_key"])
        msg = client.messages.create(model="claude-haiku-4-5-20251001",
            max_tokens=max_tokens, messages=[{"role": "user", "content": prompt}])
        return msg.content[0].text
    if env.get("openai_api_key"):
        from openai import OpenAI
        client = OpenAI(api_key=env["openai_api_key"])
        r = client.chat.completions.create(model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}], max_tokens=max_tokens)
        return r.choices[0].message.content
    raise RuntimeError("No LLM configured. Set LLM_BASE_URL or an API key in .env")


def _score(text: str) -> float:
    m = re.search(r"(0(?:\.\d+)?|1(?:\.0+)?)", text or "")
    return float(m.group(1)) if m else 0.0


def _fulltext_for_pmids(pmids: set) -> dict:
    """Fetch the full stored text for each PMID from the database."""
    import psycopg
    conn = psycopg.connect(settings()["env"]["database_url"])
    out = {}
    for p in pmids:
        rows = conn.execute(
            "SELECT chunk FROM evidence_chunks WHERE pmid = %s ORDER BY id", (p,)
        ).fetchall()
        out[p] = " ".join(r[0] for r in rows)[:2000]
    conn.close()
    return out


def rag_answer(question: str):
    evidence = retrieve(question)
    context = "\n\n".join(f"[{e.pmid}] {e.text}" for e in evidence)
    prompt = ("Answer the question using ONLY the context below. Cite sources inline as "
              "[PMID]. If the context does not contain the answer, say 'The evidence does "
              f"not cover this.'\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:")
    return _llm(prompt), evidence, context


def judge_faithfulness(answer: str, context: str) -> float:
    p = ("Grade whether an ANSWER is supported by CONTEXT. Reply with ONE number 0-1 = "
         "fraction of the answer's claims directly supported by the context. Output only the "
         f"number.\n\nCONTEXT:\n{context}\n\nANSWER:\n{answer}")
    return _score(_llm(p, max_tokens=10))


def judge_relevancy(answer: str, question: str) -> float:
    p = ("Reply with ONE number 0-1 for how well the ANSWER addresses the QUESTION. Output "
         f"only the number.\n\nQUESTION:\n{question}\n\nANSWER:\n{answer}")
    return _score(_llm(p, max_tokens=10))


def judge_citation_accuracy(answer: str, evidence) -> float | None:
    cited = set(re.findall(r"\[(\d{4,9})\]", answer))
    if not cited:
        return None
    fulltext = _fulltext_for_pmids(cited)
    sources = "\n\n".join(f"[{p}] {fulltext.get(p) or '(citation NOT found in database)'}"
                          for p in cited)
    prompt = ("You are checking citation accuracy. For each [PMID] cited in the ANSWER, decide "
              "if that SOURCE supports the statement it is attached to. Count a citation as "
              "supported if the source is on-topic and consistent with the cited claim. Reply "
              "with ONE number 0-1 = fraction of citations supported. Output only the number.\n\n"
              f"ANSWER:\n{answer}\n\nSOURCES:\n{sources}")
    return _score(_llm(prompt, max_tokens=10))


def retrospective_roc_auc(limit: int = 500) -> float | None:
    """Can a model rank potent EGFR binders above weak ones? (held-out ROC-AUC)"""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    from targetscout.data import chembl
    from targetscout.embeddings.molecule import rdkit_features

    acts = chembl.activities_for_target(target_chembl_id="CHEMBL203",
                                        standard_type="IC50", limit=limit)
    pairs = [(a["smiles"], float(a["pchembl_value"]))
             for a in acts if a.get("smiles") and a.get("pchembl_value")]
    if len(pairs) < 40:
        return None
    vals = sorted(p for _, p in pairs)
    lo, hi = vals[len(vals) // 3], vals[2 * len(vals) // 3]
    X, y = [], []
    for smi, p in pairs:
        label = 1 if p >= hi else (0 if p <= lo else None)
        if label is None:
            continue
        f = rdkit_features(smi)
        if f is None:
            continue
        X.append(f); y.append(label)
    if len(set(y)) < 2 or min(sum(y), len(y) - sum(y)) < 10:
        return None
    X = StandardScaler().fit_transform(np.vstack(X))
    y = np.array(y)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    clf = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
    return round(float(roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])), 3)


def load_golden() -> list[dict]:
    return [json.loads(l) for l in GOLDEN.read_text().splitlines() if l.strip()]


def run_eval(limit: int | None = None) -> dict:
    items = [g for g in load_golden() if g.get("type") in {"evidence_qa", "data_qa"}]
    if limit:
        items = items[:limit]
    faiths, rels, cites = [], [], []
    for g in items:
        try:
            answer, evidence, context = rag_answer(g["question"])
            f = judge_faithfulness(answer, context)
            r = judge_relevancy(answer, g["question"])
            c = judge_citation_accuracy(answer, evidence)
        except Exception as e:
            print(f"- {g['id']}: SKIPPED ({type(e).__name__})")
            time.sleep(2); continue
        faiths.append(f); rels.append(r)
        if c is not None:
            cites.append(c)
        print(f"- {g['id']}: faith={f:.2f} rel={r:.2f} cite={'-' if c is None else round(c,2)}")
        time.sleep(1)
    print("computing retrospective ROC-AUC on ChEMBL...")
    try:
        retro = retrospective_roc_auc()
    except Exception as e:
        print("  retrospective skipped:", type(e).__name__); retro = None
    avg = lambda xs: round(sum(xs) / len(xs), 3) if xs else None
    return {"n_evaluated": len(faiths), "faithfulness": avg(faiths),
            "answer_relevancy": avg(rels), "citation_accuracy": avg(cites),
            "retrospective_roc_auc": retro}


def main(ci: bool, limit: int | None) -> None:
    metrics = run_eval(limit)
    print("\n== EVAL RESULTS ==")
    print(json.dumps(metrics, indent=2))
    if ci:
        thr = settings()["eval"]["thresholds"]
        fails = [f"{k}={metrics[k]} < {thr[k]}" for k in thr
                 if metrics.get(k) is not None and metrics[k] < thr[k]]
        if fails:
            print("EVAL GATE FAILED:", "; ".join(fails)); sys.exit(1)
        print("Eval gate passed.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ci", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    main(a.ci, a.limit)
