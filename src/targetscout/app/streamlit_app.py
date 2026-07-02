"""Streamlit UI — enter a target, get a cited triage report (Phase 4)."""
from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="TargetScout", page_icon="🧬", layout="wide")
st.title("🧬 TargetScout — hit-triage copilot")

target = st.text_input("Target (UniProt accession or gene symbol)", "P00533")
if st.button("Run triage"):
    with st.spinner("Resolving target, retrieving candidates & evidence..."):
        from targetscout.agents.graph import build_graph

        result = build_graph().invoke({"target": target})
    st.subheader(result.get("protein", {}).get("name", target))
    st.metric("Candidates retrieved", len(result.get("candidates", [])))
    st.markdown(result.get("report", "_report pending — implement synthesizer (Phase 3)_"))
    with st.expander("Cited evidence"):
        for e in result.get("evidence", []):
            st.markdown(f"- [{e.get('pmid')}]({e.get('url')}): {e.get('text', '')[:200]}")
