"""Ingest a broad EGFR literature corpus for the eval questions."""
from targetscout.rag.ingest import ingest

QUERIES = [
    "EGFR T790M osimertinib acquired resistance",
    "EGFR C797S fourth generation inhibitor resistance",
    "EGFR exon 20 insertion mutation inhibitor",
    "EGFR inhibitor CNS brain blood-brain-barrier penetration",
    "covalent irreversible EGFR inhibitor cysteine 797",
    "quinazoline EGFR inhibitor gefitinib erlotinib structure",
    "EGFR inhibitor hepatotoxicity liver injury DILI",
    "EGFR inhibitor cardiotoxicity hERG QT prolongation",
    "afatinib dacomitinib neratinib EGFR selectivity",
    "osimertinib lazertinib mobocertinib mutant selective EGFR",
    "EGFR inhibitor mechanism of action kinase",
    "EGFR L858R exon 19 deletion non-small cell lung cancer",
]

total = 0
for q in QUERIES:
    print("===", q)
    total += ingest(q, n=60)
print("TOTAL new chunks:", total)
