"""
Step 5 - Assign Metabolic State Labels
Project: Predict Tumor Metabolic State from Multi-Omic Biological Data

This script assigns each BRCA tumor sample one of three metabolic labels:
    - Glycolytic  : tumor primarily uses glycolysis for energy
    - Oxidative   : tumor primarily uses oxidative phosphorylation
    - Mixed       : tumor uses both pathways

Method: ssGSEA-style scoring (simplified)
    For each patient we compute:
        1. Glycolysis score    = mean z-score of glycolysis genes
        2. OXPHOS score        = mean z-score of OXPHOS genes
        3. Metabolic ratio     = Glycolysis score - OXPHOS score

    Labeling rules (percentile-based):
        - Top 33% ratio    -> Glycolytic
        - Bottom 33% ratio -> Oxidative
        - Middle 33%       -> Mixed

    Using percentile-based cutoffs (rather than fixed thresholds)
    ensures balanced class distribution regardless of data scale,
    which is important for training a fair classifier.

Input  (from step4_output/):
    - brca_rnaseq_normalized.csv

Output (to step5_output/):
    - brca_labels.csv               <- patient ID + metabolic label
    - brca_metabolic_scores.csv     <- raw scores per patient
    - label_distribution.txt        <- class counts for report
"""

import pandas as pd
import numpy as np
import os

INPUT_DIR  = "step4_output"
OUTPUT_DIR = "step5_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# SECTION 1: Load Normalized RNA-seq
# We only need RNA-seq for labeling — methylation is not used
# here because gene expression directly reflects metabolic
# activity, while methylation reflects regulatory state.
# ─────────────────────────────────────────────────────────────

print("=" * 60)
print("SECTION 1: Loading Normalized RNA-seq")
print("=" * 60)

rna_df = pd.read_csv(os.path.join(INPUT_DIR, "brca_rnaseq_normalized.csv"), index_col=0)

# Strip RNA_ prefix to get back gene names for matching
rna_df.columns = [c.replace("RNA_", "") for c in rna_df.columns]

print(f"  Loaded: {rna_df.shape[0]} patients x {rna_df.shape[1]} genes")
print(f"  Genes available: {list(rna_df.columns)}")

# ─────────────────────────────────────────────────────────────
# SECTION 2: Define Metabolic Gene Sets
#
# These are the core glycolysis and OXPHOS genes from the
# MSigDB Hallmark gene sets. We use only genes that are
# actually present in our filtered RNA-seq dataset.
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("SECTION 2: Defining Metabolic Gene Sets")
print("=" * 60)

GLYCOLYSIS_GENES = [
    "HK1","HK2","HK3","GPI","PFKL","PFKM","PFKP","ALDOA","ALDOB","ALDOC",
    "TPI1","GAPDH","PGK1","PGK2","PGAM1","PGAM2","ENO1","ENO2","ENO3",
    "PKM","PKLR","LDHA","LDHB","SLC16A1","SLC16A3","PDK1","PDK2",
    "PDK3","PDK4","PDHA1","PDHA2","PDHB","G6PD","PGD","TALDO1","TKT"
]

OXPHOS_GENES = [
    "NDUFA1","NDUFA2","NDUFA3","NDUFA4","NDUFA5","NDUFA6","NDUFA7",
    "NDUFB1","NDUFB2","NDUFB3","NDUFB4","NDUFB5","NDUFB6","NDUFB7",
    "SDHB","SDHC","SDHD","UQCRC1","UQCRC2","UQCRB","CYC1",
    "COX4I1","COX5A","COX5B","COX6A1","COX6B1","COX7A1","COX7A2",
    "ATP5F1A","ATP5F1B","ATP5F1C","ATP5F1D","ATP5MC1","ATP5PB",
    "CS","ACO2","IDH2","IDH3A","OGDH","SUCLA2","SDHA","FH","MDH2"
]

# Find which genes are in our dataset
glyc_found  = [g for g in GLYCOLYSIS_GENES if g in rna_df.columns]
oxphos_found = [g for g in OXPHOS_GENES    if g in rna_df.columns]

print(f"  Glycolysis genes in dataset:  {len(glyc_found)} / {len(GLYCOLYSIS_GENES)}")
print(f"  OXPHOS genes in dataset:      {len(oxphos_found)} / {len(OXPHOS_GENES)}")
print(f"  Glycolysis: {glyc_found}")
print(f"  OXPHOS:     {oxphos_found}")

if len(glyc_found) < 3 or len(oxphos_found) < 3:
    print("\n  WARNING: Very few metabolic genes found.")
    print("  Using ALL available genes for scoring.")
    # Fallback: use all genes present
    glyc_found   = [g for g in rna_df.columns if any(
        k in g for k in ["HK","PK","LDH","ENO","PFK","GAPDH","PDK","LDHA"])]
    oxphos_found = [g for g in rna_df.columns if any(
        k in g for k in ["NDUF","SDH","COX","ATP5","UQCR","CYC"])]
    print(f"  Fallback glycolysis: {glyc_found}")
    print(f"  Fallback OXPHOS:     {oxphos_found}")

# ─────────────────────────────────────────────────────────────
# SECTION 3: Compute Metabolic Scores Per Patient
#
# Glycolysis score  = mean expression of glycolysis genes
# OXPHOS score      = mean expression of OXPHOS genes
# Metabolic ratio   = Glycolysis score - OXPHOS score
#
# A high ratio = more glycolytic
# A low ratio  = more oxidative
# Near zero    = mixed
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("SECTION 3: Computing Metabolic Scores")
print("=" * 60)

scores = pd.DataFrame(index=rna_df.index)

if glyc_found:
    scores["glycolysis_score"] = rna_df[glyc_found].mean(axis=1)
else:
    scores["glycolysis_score"] = 0.0
    print("  WARNING: No glycolysis genes found — scores set to 0")

if oxphos_found:
    scores["oxphos_score"] = rna_df[oxphos_found].mean(axis=1)
else:
    scores["oxphos_score"] = 0.0
    print("  WARNING: No OXPHOS genes found — scores set to 0")

scores["metabolic_ratio"] = scores["glycolysis_score"] - scores["oxphos_score"]

print(f"  Glycolysis score — mean: {scores['glycolysis_score'].mean():.3f} "
      f"std: {scores['glycolysis_score'].std():.3f}")
print(f"  OXPHOS score     — mean: {scores['oxphos_score'].mean():.3f} "
      f"std: {scores['oxphos_score'].std():.3f}")
print(f"  Metabolic ratio  — mean: {scores['metabolic_ratio'].mean():.3f} "
      f"std: {scores['metabolic_ratio'].std():.3f}")
print(f"  Ratio range: {scores['metabolic_ratio'].min():.3f} "
      f"to {scores['metabolic_ratio'].max():.3f}")

# ─────────────────────────────────────────────────────────────
# SECTION 4: Assign Labels Using Percentile Cutoffs
#
# We split the metabolic ratio into three equal thirds:
#   Top 33%    -> Glycolytic  (high glycolysis relative to OXPHOS)
#   Bottom 33% -> Oxidative   (low glycolysis relative to OXPHOS)
#   Middle 33% -> Mixed       (balanced)
#
# This gives roughly equal class sizes which is important for
# avoiding class imbalance during model training.
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("SECTION 4: Assigning Metabolic Labels")
print("=" * 60)

p33 = scores["metabolic_ratio"].quantile(0.333)
p67 = scores["metabolic_ratio"].quantile(0.667)

print(f"  33rd percentile cutoff: {p33:.3f}")
print(f"  67th percentile cutoff: {p67:.3f}")

def assign_label(ratio):
    if ratio >= p67:
        return "Glycolytic"
    elif ratio <= p33:
        return "Oxidative"
    else:
        return "Mixed"

scores["metabolic_label"] = scores["metabolic_ratio"].apply(assign_label)

# Count distribution
label_counts = scores["metabolic_label"].value_counts()
print(f"\n  Label distribution:")
for label, count in label_counts.items():
    pct = count / len(scores) * 100
    print(f"    {label:12s}: {count} patients ({pct:.1f}%)")

# ─────────────────────────────────────────────────────────────
# SECTION 5: Save Outputs
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("SECTION 5: Saving Outputs")
print("=" * 60)

# Labels file — just patient ID and label
labels_df = scores[["metabolic_label"]].copy()
labels_df.to_csv(os.path.join(OUTPUT_DIR, "brca_labels.csv"))

# Full scores file — for report and analysis
scores.to_csv(os.path.join(OUTPUT_DIR, "brca_metabolic_scores.csv"))

# Distribution report
report_lines = [
    "METABOLIC LABEL DISTRIBUTION",
    "=" * 40,
    f"Total patients: {len(scores)}",
    f"Labeling method: percentile-based (33/33/33 split)",
    f"33rd percentile cutoff: {p33:.4f}",
    f"67th percentile cutoff: {p67:.4f}",
    "",
    "Class counts:",
]
for label, count in label_counts.items():
    pct = count / len(scores) * 100
    report_lines.append(f"  {label}: {count} ({pct:.1f}%)")

report_lines += [
    "",
    f"Glycolysis genes used ({len(glyc_found)}): {glyc_found}",
    f"OXPHOS genes used ({len(oxphos_found)}):   {oxphos_found}",
]

with open(os.path.join(OUTPUT_DIR, "label_distribution.txt"), "w") as f:
    f.write("\n".join(report_lines))

print(f"  Saved: brca_labels.csv")
print(f"  Saved: brca_metabolic_scores.csv")
print(f"  Saved: label_distribution.txt")

print("\n" + "=" * 60)
print("STEP 5 COMPLETE - LABEL SUMMARY")
print("=" * 60)
print(f"  Total patients labeled: {len(scores)}")
for label, count in label_counts.items():
    print(f"  {label:12s}: {count} ({count/len(scores)*100:.1f}%)")
print(f"\nOutputs saved to: {OUTPUT_DIR}/")
print("Next step: Run step6_feature_selection.py")