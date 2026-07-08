"""Generate a golden evaluation set for TargetScout (EGFR focus)."""
import json
from pathlib import Path

OUT = Path("src/targetscout/eval/golden/qa_seed.jsonl")
rows = []
def add(qid, question, expected, qtype, target="EGFR"):
    rows.append({"id": qid, "target": target, "question": question,
                 "expected_points": expected, "type": qtype})

hand = [
 ("egfr-herg-1","Which EGFR tyrosine kinase inhibitors have documented hERG or cardiotoxicity signals, and what is the evidence?",["names >=1 EGFR TKI with a cardiac signal","cites a PMID"]),
 ("egfr-t790m-1","How does osimertinib overcome the EGFR T790M resistance mutation?",["mentions T790M","covalent/irreversible binding"]),
 ("egfr-c797s-1","What is the significance of the EGFR C797S mutation for third-generation inhibitors?",["C797S disrupts covalent binding","confers resistance"]),
 ("egfr-exon20-1","Which agents target EGFR exon 20 insertion mutations?",["names an exon 20 agent","grounded in literature"]),
 ("egfr-cns-1","Which EGFR inhibitors show meaningful CNS / brain penetration?",["names a CNS-penetrant TKI","evidence of BBB penetration"]),
 ("egfr-covalent-1","What distinguishes covalent (irreversible) EGFR inhibitors from reversible ones?",["covalent binds Cys797","reversible vs irreversible"]),
 ("egfr-scaffold-1","What chemical scaffold is common among first-generation EGFR inhibitors like gefitinib and erlotinib?",["quinazoline / anilinoquinazoline","grounded in evidence"]),
 ("egfr-dili-1","Is there evidence of hepatotoxicity (liver injury) associated with EGFR inhibitors?",["addresses liver toxicity","cites evidence"]),
 ("egfr-selectivity-1","Why is kinase selectivity important for reducing EGFR inhibitor toxicity?",["off-target kinases cause toxicity","selectivity reduces side effects"]),
 ("egfr-resistance-1","What are common mechanisms of acquired resistance to EGFR inhibitors?",["mentions >=1 resistance mechanism","grounded"]),
 ("egfr-l858r-1","What is the EGFR L858R mutation and why is it clinically important?",["activating mutation in exon 21","sensitizes to TKIs"]),
 ("egfr-1stgen-1","What limits the durability of first-generation EGFR inhibitors in patients?",["acquired resistance","T790M emergence"]),
 ("egfr-herg-mech-1","By what molecular mechanism can kinase inhibitors block the hERG channel?",["direct hERG channel block","links to QT prolongation"]),
 ("egfr-offtarget-1","How can off-target profiling (docking to hERG/CYP) improve EGFR drug safety?",["off-target screening","reduces late-stage failure"]),
 ("egfr-pk-1","What physicochemical properties influence oral bioavailability of EGFR inhibitors?",["lipophilicity/solubility mentioned","links to absorption"]),
 ("egfr-skin-1","What common on-target side effects (skin rash, diarrhea) are seen with EGFR inhibitors?",["names an on-target toxicity","grounded"]),
 ("egfr-4thgen-1","Why are fourth-generation EGFR inhibitors being developed?",["to address C797S/triple-mutant resistance","grounded"]),
 ("egfr-combo-1","Why are combination therapies explored to delay EGFR inhibitor resistance?",["rationale for combinations","grounded"]),
 ("egfr-mutsel-1","What does 'mutant-selective' mean for an EGFR inhibitor and why is it desirable?",["spares wild-type EGFR","reduces toxicity"]),
 ("egfr-biomarker-1","What biomarker testing guides EGFR inhibitor treatment selection?",["EGFR mutation testing","grounded"]),
 ("egfr-qt-1","How is QT-prolongation risk assessed for kinase inhibitor candidates?",["hERG/QT assays","grounded"]),
 ("egfr-struct-1","How do structural studies of the EGFR kinase domain guide inhibitor design?",["ATP-binding site","structure-guided design"]),
 ("egfr-tox-pred-1","How can in-silico ADMET prediction reduce attrition in EGFR drug discovery?",["early toxicity prediction","reduces failures"]),
]
for qid,q,e in hand: add(qid,q,e,"evidence_qa")

drugs = ["gefitinib","erlotinib","afatinib","osimertinib","dacomitinib",
         "lazertinib","neratinib","brigatinib","icotinib","mobocertinib"]
for i,d in enumerate(drugs):
    add(f"drug-card-{i}", f"Does the literature report any cardiac or hERG-related safety concern for {d}?",
        ["gives a directional answer","cites evidence or states none found"], "evidence_qa")
    add(f"drug-mut-{i}", f"Which EGFR mutations or patient populations is {d} used or studied for?",
        ["names a relevant mutation/indication","grounded in evidence"], "evidence_qa")
    add(f"drug-gen-{i}", f"What generation of EGFR inhibitor is {d}, and is its binding reversible or covalent?",
        ["states generation","reversible vs covalent binding"], "evidence_qa")

data = [
 ("data-potency-1","Among EGFR inhibitors in ChEMBL, what pIC50 range is considered high potency?",["pIC50 >= 8 ~ high potency","grounded"]),
 ("data-scaffold-2","What recurring scaffold appears among high-potency EGFR binders in the bioactivity data?",["quinazoline/pyrimidine","data-grounded"]),
 ("data-activitytype-1","Which standard activity types (IC50, Ki) are most common for EGFR in ChEMBL?",["IC50 commonly reported","grounded"]),
 ("data-count-1","Roughly how many bioactivity records exist for the EGFR target in ChEMBL?",["acknowledges thousands of records","reasonable estimate"]),
 ("data-unit-1","What units are EGFR IC50 values typically reported in?",["nM (nanomolar)","grounded"]),
 ("data-mw-1","What molecular-weight range is typical for oral EGFR inhibitors?",["~300-600 Da typical","links to drug-likeness"]),
 ("data-lipinski-1","Do most approved EGFR inhibitors obey Lipinski's rule of five?",["mostly yes with exceptions","grounded"]),
 ("data-pchembl-1","What does the pChEMBL value represent for an EGFR bioactivity record?",["normalized -log potency","grounded"]),
 ("data-target-1","What ChEMBL target ID corresponds to human EGFR?",["CHEMBL203","grounded"]),
 ("data-subs-1","What substructure motif is shared by anilinoquinazoline EGFR inhibitors?",["aniline + quinazoline","data-grounded"]),
 ("data-active-1","What IC50 threshold is often used to label an EGFR compound 'active'?",["e.g., <= 1 uM","reasonable"]),
 ("data-halogen-1","Are halogenated aniline substituents common in potent EGFR inhibitors?",["yes, chloro/fluoro anilines","data-grounded"]),
]
for qid,q,e in data: add(qid,q,e,"data_qa")

props = ["hERG cardiotoxicity","DILI liver toxicity","aqueous solubility","lipophilicity","blood-brain-barrier penetration"]
for i,p in enumerate(props):
    add(f"admet-{i}", f"For a candidate EGFR inhibitor, how should a high predicted {p} score influence triage?",
        ["explains the implication","actionable"], "admet_qa")
add("admet-cmp-1","Between two EGFR inhibitors, how would you use predicted hERG and solubility to pick the safer lead?",["balances toxicity vs developability"],"admet_qa")

for i in range(4):
    add(f"retro-{i}","RETROSPECTIVE: on held-out ChEMBL actives/inactives for EGFR, does the system rank actives above inactives?",["ROC-AUC computed on held-out set"],"retrospective")

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
counts={}
for r in rows: counts[r["type"]]=counts.get(r["type"],0)+1
print(f"Wrote {len(rows)} questions. By type: {counts}")
