# 🧬 TargetScout

![CI](https://github.com/rmn2151-blip/targetscout/actions/workflows/eval.yml/badge.svg)

**An AI hit-triage copilot for early-stage drug discovery.**

Give TargetScout a protein target (e.g. `EGFR` / UniProt `P00533`) and it returns a ranked, **evidence-backed** shortlist of candidate molecules — with predicted ADMET/safety liabilities and **cited** literature rationale — using a multi-agent RAG pipeline over real public bioactivity and biomedical-literature data.

> A portfolio project demonstrating production AI-engineering patterns end to end: trained ML models, retrieval-augmented generation, multi-agent orchestration, rigorous evaluation, and LLMOps.

---

## Demo

> _Add a screenshot or a 2-minute Loom link here._ Run `make ui`, enter `P00533`, and record the cited report it generates.

## What it does

For a given target, a **LangGraph** agent runs a 6-step pipeline:

1. **Resolve** the target (UniProt).
2. **Retrieve** known bioactive compounds (ChEMBL).
3. **Predict** ADMET properties for each candidate with trained ML models.
4. **Retrieve** supporting literature (PubMed → pgvector semantic search + reranking).
5. **Flag** safety liabilities (hERG, DILI, solubility) and rank candidates safest-first.
6. **Synthesize** a concise, `[PMID]`-cited triage report with an LLM.

## Key results

| Component | Metric | Score |
|---|---|---|
| ADMET — hERG cardiotoxicity | ROC-AUC | **0.84** |
| ADMET — drug-induced liver injury (DILI) | ROC-AUC | **0.93** |
| ADMET — blood-brain-barrier penetration | ROC-AUC | **0.88** |
| ADMET — lipophilicity | MAE | **0.57** |
| RAG answer faithfulness (literature QA) | RAGAS-style judge | **0.79** |
| RAG answer relevancy | LLM judge | **0.80** |
| Citation accuracy | LLM judge | **0.63** |
| Retrospective potency ranking (held-out ChEMBL) | ROC-AUC | **0.93** |

Using the eval harness, I diagnosed a retrieval-coverage gap and expanded the literature corpus, **raising faithfulness from 0.19 → 0.79**.

## Architecturetarget ──▶ Planner ──▶ Target Resolver ──▶ Candidate Retriever
│
Evidence Retriever ◀───────┤
(PubMed → pgvector)         ▼
│          Property Predictor
▼          (ADMET ML models)
Safety Checker ──▶ Synthesizer ──▶ cited report
▲   │
└───┘ reflection loop## Tech stack

- **Agents:** LangGraph · LangChain / LlamaIndex
- **Retrieval:** pgvector (Postgres) · PubMedBERT embeddings · cross-encoder reranker
- **ML:** scikit-learn / LightGBM ADMET models · RDKit features · trained on Therapeutics Data Commons
- **LLM:** pluggable (Anthropic / OpenAI / local via OpenAI-compatible endpoint)
- **Eval:** LLM-as-judge (faithfulness, relevancy, citation accuracy) + retrospective ROC-AUC
- **LLMOps:** MLflow (experiment tracking) · FastAPI + Docker · Streamlit UI · GitHub Actions CI
- **Data (all free/public):** ChEMBL · UniProt · PubMed · Open Targets · Therapeutics Data Commons

## Quickstart

```bash
# 1. Environment (conda recommended for rdkit)
conda create -n targetscout python=3.12 -y && conda activate targetscout
pip install -r requirements.txt && pip install -e .
cp .env.example .env            # add an LLM key (or a free LLM_BASE_URL)

# 2. Database (for the RAG layer)
docker compose up -d db

# 3. Train the ADMET models
python -m targetscout.models.admet_train --endpoint hERG   # + DILI, Solubility_AqSolDB, ...

# 4. Ingest literature, then run the agent
python scripts/ingest_corpus.py
make agent TARGET=P00533

# 5. Web app
make api    # FastAPI docs at http://localhost:8000/docs
make ui     # Streamlit UI
```

## Evaluation

`make eval` scores the pipeline on a 75-question golden set with an LLM-as-judge (faithfulness, relevancy, citation accuracy) plus a retrospective ChEMBL potency-ranking ROC-AUC. Metrics and thresholds live in `config/config.yaml`; GitHub Actions runs tests + lint on every push.

## Repo layout src/targetscout/
data/        # API clients: ChEMBL, UniProt, Open Targets, PubMed, TDC
embeddings/  # ESM-2 protein + RDKit molecule featurizers
models/      # ADMET tabular training + inference
rag/         # pgvector ingest + retriever + reranker
agents/      # LangGraph state graph + nodes
eval/        # eval harness, metrics, golden set
api/  app/   # FastAPI service + Streamlit UI ## Limitations & future work

- Citation precision (0.63) is the weakest metric — a known RAG attribution challenge; would improve with a stronger judge and tighter source-grounding.
- Data-lookup questions (structured ChEMBL facts) need a dedicated tool, not literature RAG.
- ADMET models use fingerprint features; graph neural networks would likely improve accuracy.

## Disclaimer

Research/educational tool only. Predictions are **not** validated for clinical, safety, or regulatory use.
