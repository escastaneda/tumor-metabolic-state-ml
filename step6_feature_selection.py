"""
Step 6 - Feature Selection
Project: Predict Tumor Metabolic State from Multi-Omic Biological Data

Memory-efficient version:
    - Loads RNA-seq and methylation SEPARATELY (never combined)
    - Runs ANOVA on methylation in column chunks (same as Step 3)
    - Runs Random Forest only on the small filtered set
    - Saves final selected feature matrix at the end

Input:
    - step4_output/brca_rnaseq_normalized.csv
    - step4_output/brca_methylation_normalized.csv
    - step5_output/brca_labels.csv

Output (to step6_output/):
    - brca_features_selected.csv
    - feature_importance.csv
    - feature_selection_report.txt
"""

import pandas as pd
import numpy as np
import os
from scipy import stats
from sklearn.ensemble import RandomForestClassifier

INPUT_DIR_4  = "step4_output"
INPUT_DIR_5  = "step5_output"
OUTPUT_DIR   = "step6_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TOP_METH_ANOVA = 500    # keep top 500 methylation sites after ANOVA
TOP_RF         = 500    # keep top 500 after Random Forest
CHUNK_SIZE     = 2000   # methylation columns per chunk

# ─────────────────────────────────────────────────────────────
# SECTION 1: Load Labels and RNA-seq (both small)
# ─────────────────────────────────────────────────────────────

print("=" * 60)
print("SECTION 1: Loading Labels and RNA-seq")
print("=" * 60)

y = pd.read_csv(os.path.join(INPUT_DIR_5, "brca_labels.csv"), index_col=0).squeeze()
label_map = {"Glycolytic": 0, "Mixed": 1, "Oxidative": 2}
y_enc = y.map(label_map)
print(f"  Labels loaded: {len(y)} patients")
print(f"  Classes: {y.value_counts().to_dict()}")

rna_df = pd.read_csv(os.path.join(INPUT_DIR_4, "brca_rnaseq_normalized.csv"), index_col=0)
common_patients = sorted(set(rna_df.index) & set(y.index))
rna_df = rna_df.loc[common_patients]
y_enc  = y_enc.loc[common_patients]
y      = y.loc[common_patients]
print(f"  RNA-seq loaded: {rna_df.shape[0]} patients x {rna_df.shape[1]} genes")

# ─────────────────────────────────────────────────────────────
# SECTION 2: ANOVA on RNA-seq Features (fast — only 72 genes)
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("SECTION 2: ANOVA on RNA-seq Features")
print("=" * 60)

groups_rna = [rna_df[y == lbl].values for lbl in y.unique()]
rna_f, rna_p = [], []

for i in range(rna_df.shape[1]):
    col_groups = [g[:, i] for g in groups_rna]
    try:
        f, p = stats.f_oneway(*col_groups)
    except:
        f, p = 0.0, 1.0
    rna_f.append(f)
    rna_p.append(p)

rna_anova = pd.DataFrame({
    "feature": rna_df.columns,
    "f_stat":  rna_f,
    "p_value": rna_p
}).sort_values("f_stat", ascending=False)

print(f"  RNA-seq ANOVA results (all {len(rna_anova)} genes):")
print(rna_anova.to_string(index=False))

# Keep all RNA features (only 72, all informative)
rna_selected = list(rna_df.columns)

# ─────────────────────────────────────────────────────────────
# SECTION 3: ANOVA on Methylation in Column Chunks
#
# Load 2,000 CpG columns at a time, run ANOVA, keep track
# of the top scoring sites without loading everything at once.
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("SECTION 3: ANOVA on Methylation (chunked)")
print("=" * 60)

meth_path = os.path.join(INPUT_DIR_4, "brca_methylation_normalized.csv")

# Read column names
with open(meth_path, "r") as f:
    all_cols = f.readline().strip().split(",")
cpg_cols  = [c for c in all_cols[1:] if c.strip() != ""]
n_cpg     = len(cpg_cols)
n_chunks  = (n_cpg + CHUNK_SIZE - 1) // CHUNK_SIZE

print(f"  Total CpG sites: {n_cpg:,}")
print(f"  Processing in {n_chunks} chunks of {CHUNK_SIZE}...")
print(f"  Keeping top {TOP_METH_ANOVA} by F-statistic")

# We maintain a rolling top-N list
all_meth_results = []

for chunk_idx in range(n_chunks):
    col_start    = chunk_idx * CHUNK_SIZE
    col_end      = min(col_start + CHUNK_SIZE, n_cpg)
    col_positions = [0] + list(range(col_start + 1, col_end + 1))

    chunk_df = pd.read_csv(
        meth_path,
        index_col=0,
        usecols=col_positions,
        low_memory=False
    )
    chunk_df = chunk_df.loc[common_patients]

    # Run ANOVA per CpG site in this chunk
    groups_meth = [chunk_df[y == lbl].values for lbl in y.unique()]

    for i, cpg in enumerate(chunk_df.columns):
        col_groups = [g[:, i] for g in groups_meth]
        try:
            f, p = stats.f_oneway(*col_groups)
        except:
            f, p = 0.0, 1.0
        all_meth_results.append((cpg, f, p))

    if (chunk_idx + 1) % 20 == 0 or chunk_idx == n_chunks - 1:
        print(f"    Chunk {chunk_idx+1}/{n_chunks} done | "
              f"{len(all_meth_results):,} CpG sites tested")

# Sort and keep top N
meth_anova = pd.DataFrame(all_meth_results, columns=["feature", "f_stat", "p_value"])
meth_anova = meth_anova.sort_values("f_stat", ascending=False)
meth_selected = list(meth_anova.head(TOP_METH_ANOVA)["feature"])

print(f"\n  Top 10 methylation sites by F-stat:")
print(meth_anova.head(10).to_string(index=False))
print(f"\n  Methylation sites selected: {len(meth_selected)}")

# ─────────────────────────────────────────────────────────────
# SECTION 4: Build Filtered Feature Matrix
# Load only the selected methylation columns + RNA
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("SECTION 4: Building Filtered Feature Matrix")
print("=" * 60)

print(f"  Loading top {len(meth_selected)} methylation features...")
meth_filtered = pd.read_csv(
    meth_path,
    index_col=0,
    usecols=[0] + [all_cols.index(c) for c in meth_selected],
    low_memory=False
)
meth_filtered = meth_filtered.loc[common_patients]
meth_filtered.columns = ["METH_" + c for c in meth_filtered.columns]

# RNA already loaded — add prefix
rna_prefixed = rna_df.copy()
rna_prefixed.columns = ["RNA_" + c for c in rna_df.columns]

# Combine
X_filtered = pd.concat([rna_prefixed, meth_filtered], axis=1)
print(f"  Combined shape: {X_filtered.shape[0]} patients x {X_filtered.shape[1]} features")

# ─────────────────────────────────────────────────────────────
# SECTION 5: Random Forest Feature Importance
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("SECTION 5: Random Forest Feature Importance")
print("=" * 60)

print(f"  Training Random Forest on {X_filtered.shape[1]} features...")
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_filtered.values, y_enc.values)

importances = pd.DataFrame({
    "feature":    X_filtered.columns,
    "importance": rf.feature_importances_
}).sort_values("importance", ascending=False)

imp_rna  = importances[importances["feature"].str.startswith("RNA_")]
imp_meth = importances[importances["feature"].str.startswith("METH_")]

print(f"\n  Top RNA features:")
print(imp_rna.head(15).to_string(index=False))
print(f"\n  Top Methylation features:")
print(imp_meth.head(10).to_string(index=False))

# Keep top features — all RNA + top methylation up to TOP_RF total
top_features = list(imp_rna["feature"]) + \
               list(imp_meth.head(TOP_RF - len(imp_rna))["feature"])
X_final = X_filtered[top_features]

print(f"\n  Final selected features: {X_final.shape[1]}")
print(f"    RNA:         {len(imp_rna)}")
print(f"    Methylation: {len(top_features) - len(imp_rna)}")

# ─────────────────────────────────────────────────────────────
# SECTION 6: Save Outputs
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("SECTION 6: Saving Outputs")
print("=" * 60)

X_final.to_csv(os.path.join(OUTPUT_DIR, "brca_features_selected.csv"))
importances.to_csv(os.path.join(OUTPUT_DIR, "feature_importance.csv"), index=False)
meth_anova.to_csv(os.path.join(OUTPUT_DIR, "anova_methylation.csv"), index=False)
rna_anova.to_csv(os.path.join(OUTPUT_DIR, "anova_rna.csv"), index=False)

report = [
    "STEP 6 - FEATURE SELECTION REPORT",
    "=" * 60,
    f"Original RNA-seq genes:       {rna_df.shape[1]}",
    f"Original methylation sites:   {n_cpg:,}",
    f"Total original features:      {rna_df.shape[1] + n_cpg:,}",
    "",
    f"After ANOVA filter:",
    f"  RNA kept (all):             {len(rna_selected)}",
    f"  Methylation kept (top):     {len(meth_selected)}",
    "",
    f"After Random Forest filter:",
    f"  RNA features:               {len(imp_rna)}",
    f"  Methylation features:       {len(top_features) - len(imp_rna)}",
    f"  Total final features:       {X_final.shape[1]}",
    "",
    f"Dimensionality reduction:     "
    f"{100*(1 - X_final.shape[1]/(rna_df.shape[1]+n_cpg)):.1f}% removed",
    "",
    "Top 15 features by RF importance:",
]
for _, row in importances.head(15).iterrows():
    report.append(f"  {row['feature']:45s} {row['importance']:.5f}")

with open(os.path.join(OUTPUT_DIR, "feature_selection_report.txt"), "w") as f:
    f.write("\n".join(report))

print(f"  Saved: brca_features_selected.csv")
print(f"  Saved: feature_importance.csv")
print(f"  Saved: anova_methylation.csv")
print(f"  Saved: anova_rna.csv")
print(f"  Saved: feature_selection_report.txt")

print("\n" + "=" * 60)
print("STEP 6 COMPLETE")
print("=" * 60)
print(f"  Patients:          {X_final.shape[0]}")
print(f"  Final features:    {X_final.shape[1]}")
print(f"  Feature breakdown: {len(imp_rna)} RNA + {len(top_features)-len(imp_rna)} methylation")
print(f"\nOutputs saved to: {OUTPUT_DIR}/")
print("Next step: Run step7_split_and_train.py")