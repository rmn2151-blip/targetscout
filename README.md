# 🧬 TargetScout
![CI](https://github.com/rmn2151-blip/targetscout/actions/workflows/eval.yml/badge.svg)
**An AI hit-triage copilot for early-stage drug discovery.**

Given a protein target (or a disease that maps to targets), TargetScout produces a ranked, **evidence-backed** shortlist of candidate small molecules — with predicted ADMET / safety flags and **cited** literature rationale — using a multi-agent RAG pipeline over real public bioactivity and biomedical-literature data.

> Portfolio project demonstrating production AI-engineering patterns: RAG, multi-agent orchestration (LangGraph), trained ML models (ADMET), rigorous evaluation, and LLMOps.

---

## Architecture

```
                         ┌──────────────┐
   user query  ───────▶  │   Planner    │
   (target/disease)      └──────┬───────┘
                                │
        ┌───────────────────────┼───────────────────────────┐
        ▼                       ▼                             ▼
 ┌─────────────┐        ┌───────────────┐            ┌────────────────┐
 │Target Resolver│      │Candidate Retriever│         │Evidence Retriever│
 │(UniProt/OpenT)│      │  (ChEMBL/PubChem) │         │ (PubMed → pgvector)│
 └──────┬───────┘       └───────┬────────┘            └────────┬───────┘
        │                       ▼                              │
        │              ┌────────────────┐                      │
        └────────────▶ │Property Predictor│◀────────────────────┘
                       │ (ADMET tabular ML)│
                       └───────┬─────────┘
                               ▼
                       ┌───────────────┐      reflection / re-retrieval loop
                       │ Safety Checker │◀───────────────┐
                       └───────┬────────┘                │
                               ▼                         │
                       ┌───────────────┐                 │
                       │  Synthesizer   │─── low conf ────┘
                       │ (cited report) │
                       └───────────────┘
```

## Tech stack

- **Agents:** LangGraph · LangChain / LlamaIndex
- **Vector store:** pgvector (Postgres) — Pinecone backend behind an interface
- **Models:** ESM-2 (protein embeddings) · ChemBERTa / RDKit (molecules) · XGBoost/LightGBM ADMET heads · biomedical embedder + cross-encoder reranker · frontier LLM for synthesis
- **Benchmarks/data:** Therapeutics Data Commons (ADMET) · ChEMBL · BindingDB · PubChem · UniProt · Open Targets · PubMed
- **Eval:** RAGAS · LLM-as-judge · retrospective ranking (ROC-AUC / enrichment)
- **LLMOps:** MLflow (model registry) · Langfuse (tracing) · FastAPI + Docker · GitHub Actions CI eval gate

## Quickstart

```bash
# 1. Environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # add your keys / email

# 2. Postgres + pgvector (for the RAG layer, needed from Phase 2)
docker compose up -d db

# 3. Smoke-test the data APIs (no ML needed)
make smoke                      # hits ChEMBL, UniProt, Open Targets, PubMed

# 4. Train a first ADMET model (Phase 1)
python -m targetscout.models.admet_train --endpoint hERG

# 5. Run the API / UI (later phases)
make api                        # FastAPI on :8000
make ui                         # Streamlit
```

## Repo layout

```
src/targetscout/
  data/        # API clients: ChEMBL, UniProt, Open Targets, PubChem, PubMed, TDC loader
  embeddings/  # ESM-2 protein + molecule featurizers
  models/      # ADMET tabular training + inference
  rag/         # pgvector index + retriever + reranker
  agents/      # LangGraph state graph + nodes
  eval/        # eval harness, metrics, golden set
  api/         # FastAPI service
  app/         # Streamlit UI
```

## Build roadmap
See **`../TargetScout_Project_Spec_and_Roadmap.md`** for the phased 1–3 month plan and resume metrics, and **`../TargetScout_Data_Sources.md`** for API details.

## Status
🟡 Scaffold — runnable data-API stubs + ADMET trainer skeleton in place. Follow the roadmap phase by phase.

## Disclaimer
Research/educational tool. Predictions are not validated for clinical or safety decisions.
