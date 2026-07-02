.PHONY: install smoke train rag agent api ui test lint eval

install:
	pip install -r requirements.txt

smoke:            ## Hit each data API once and print real data
	python -m targetscout.data.smoke_test

train:            ## Train an ADMET model (make train ENDPOINT=hERG)
	python -m targetscout.models.admet_train --endpoint $(or $(ENDPOINT),hERG)

rag:              ## Ingest PubMed abstracts into pgvector (make rag QUERY="EGFR inhibitor")
	python -m targetscout.rag.ingest --query "$(or $(QUERY),EGFR inhibitor)"

agent:            ## Run the triage agent (make agent TARGET=EGFR)
	python -m targetscout.agents.run --target $(or $(TARGET),EGFR)

api:
	uvicorn targetscout.api.main:app --reload --port 8000

ui:
	streamlit run src/targetscout/app/streamlit_app.py

test:
	pytest -q

lint:
	ruff check src tests

eval:             ## Run the eval harness against the golden set
	python -m targetscout.eval.harness
