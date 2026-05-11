"""
Step 3 - Data Cleaning & Quality Control
Project: Predict Tumor Metabolic State from Multi-Omic Biological Data

The methylation CSV from Step 2 has:
    rows    = 770 patients
    columns = 396,065 CpG sites

We process CpG sites in chunks of columns to avoid memory issues.
"""

import pandas as pd
import numpy as np
import os

INPUT_DIR  = "step2_output"
OUTPUT_DIR = "step3_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

qc_log = []
qc_log.append("STEP 3 - DATA CLEANING & QC REPORT")
qc_log.append("=" * 60)

# ─────────────────────────────────────────────────────────────
# SECTION 1: Load RNA-seq
# ─────────────────────────────────────────────────────────────

print("=" * 60)
print("SECTION 1: Loading RNA-seq Data")
print("=" * 60)

rna_df = pd.read_csv(os.path.join(INPUT_DIR, "brca_rnaseq_metabolic.csv"), index_col=0)
print(f"  RNA-seq: {rna_df.shape[0]} patients x {rna_df.shape[1]} genes")

# ─────────────────────────────────────────────────────────────
# SECTION 2: Clean RNA-seq
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("SECTION 2: Cleaning RNA-seq")
print("=" * 60)

n_orig = rna_df.shape[1]

# Remove low variance genes
rna_df      = rna_df.loc[:, rna_df.std(axis=0) >= 0.1]
removed_var = n_orig - rna_df.shape[1]
print(f"  Removed (low variance <0.1):    {removed_var} genes")

# Remove genes expressed in < 10% of patients
rna_df       = rna_df.loc[:, (rna_df > 0).mean(axis=0) >= 0.10]
removed_expr = n_orig - removed_var - rna_df.shape[1]
print(f"  Removed (expressed <10%):       {removed_expr} genes")
print(f"  Final genes:                    {rna_df.shape[1]}")
print(f"  Genes: {list(rna_df.columns)}")

qc_log.append(f"RNA-seq: {n_orig} -> {rna_df.shape[1]} genes")
qc_log.append(f"  Genes kept: {list(rna_df.columns)}")
qc_log.append("")

# ─────────────────────────────────────────────────────────────
# SECTION 3: Clean Methylation in Column Chunks
#
# The methylation file has 770 rows (patients) x 396,065 cols
# (CpG sites). We read 2,000 columns at a time, apply filters,
# and collect only the passing CpG sites.
#
# Filters applied per CpG site (column):
#   1. Remove if missing in > 20% of patients
#   2. Remove if std < 0.02 (near-constant across patients)
#   3. Impute remaining NaN with column median
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("SECTION 3: Cleaning Methylation (column chunks)")
print("=" * 60)

meth_path      = os.path.join(INPUT_DIR, "brca_methylation.csv")
meth_clean_out = os.path.join(OUTPUT_DIR, "brca_methylation_clean.csv")

MISSING_THRESH = 0.20
VAR_THRESH     = 0.02
CHUNK_SIZE     = 2000   # columns per chunk

# Get all column names first
print("  Reading column names...")
with open(meth_path, "r") as f:
    all_cols = f.readline().strip().split(",")

# The first element may be empty string (unnamed index column)
# pandas uses positional index 0 for this
cpg_cols   = [c for c in all_cols[1:] if c.strip() != ""]
n_cpg      = len(cpg_cols)
n_chunks   = (n_cpg + CHUNK_SIZE - 1) // CHUNK_SIZE

print(f"  Total CpG sites: {n_cpg:,}")
print(f"  Processing in {n_chunks} chunks of {CHUNK_SIZE} columns each...")
print(f"  Filters: missing>{int(MISSING_THRESH*100)}% | std<{VAR_THRESH}")

kept_frames   = []
total_kept    = 0
total_removed_miss = 0
total_removed_var  = 0

for chunk_idx in range(n_chunks):
    col_start = chunk_idx * CHUNK_SIZE
    col_end   = min(col_start + CHUNK_SIZE, n_cpg)
    cols_to_load = cpg_cols[col_start:col_end]

    # Load this chunk of CpG columns using integer positions
    col_positions = [0] + list(range(col_start + 1, col_end + 1))
    chunk_df = pd.read_csv(
        meth_path,
        index_col=0,
        usecols=col_positions,
        low_memory=False
    )

    # Filter 1: missing fraction per CpG site
    missing_frac = chunk_df.isna().mean(axis=0)
    pass_missing = missing_frac[missing_frac <= MISSING_THRESH].index
    removed_miss = len(chunk_df.columns) - len(pass_missing)
    chunk_df     = chunk_df[pass_missing]

    # Impute with median before variance check
    chunk_df = chunk_df.fillna(chunk_df.median(axis=0))

    # Filter 2: variance per CpG site
    cpg_std    = chunk_df.std(axis=0)
    pass_var   = cpg_std[cpg_std >= VAR_THRESH].index
    removed_v  = len(chunk_df.columns) - len(pass_var)
    chunk_df   = chunk_df[pass_var]

    total_removed_miss += removed_miss
    total_removed_var  += removed_v
    total_kept         += len(chunk_df.columns)
    kept_frames.append(chunk_df)

    if (chunk_idx + 1) % 10 == 0 or chunk_idx == n_chunks - 1:
        print(f"    Chunk {chunk_idx+1}/{n_chunks} | "
              f"kept so far: {total_kept:,} | "
              f"removed: {total_removed_miss+total_removed_var:,}")

print(f"\n  Combining {len(kept_frames)} chunks...")
meth_df = pd.concat(kept_frames, axis=1)
del kept_frames

print(f"\n  Methylation cleaning complete:")
print(f"    Original CpG sites:         {n_cpg:,}")
print(f"    Removed (>20% missing):     {total_removed_miss:,}")
print(f"    Removed (low variance):     {total_removed_var:,}")
print(f"    CpG sites kept:             {total_kept:,}")
print(f"    Final shape: {meth_df.shape[0]} patients x {meth_df.shape[1]} CpG sites")

qc_log.append(f"Methylation: {n_cpg:,} -> {total_kept:,} CpG sites")
qc_log.append(f"  Removed >20% missing: {total_removed_miss:,}")
qc_log.append(f"  Removed low variance: {total_removed_var:,}")
qc_log.append("")

# ─────────────────────────────────────────────────────────────
# SECTION 4: Outlier Sample Detection
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("SECTION 4: Outlier Sample Detection")
print("=" * 60)

def find_outliers(series, label, k=3):
    Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
    IQR    = Q3 - Q1
    outs   = series[(series < Q1 - k*IQR) | (series > Q3 + k*IQR)].index.tolist()
    print(f"  {label}")
    print(f"    Range: {series.min():.3f} to {series.max():.3f}")
    print(f"    Outliers (3x IQR): {len(outs)}")
    return outs

rna_outliers  = find_outliers(rna_df.mean(axis=1),  "RNA-seq mean expression")
meth_outliers = find_outliers(meth_df.mean(axis=1), "Methylation mean beta")

all_outliers = list(set(rna_outliers + meth_outliers))
print(f"\n  Total outliers to remove: {len(all_outliers)}")
if all_outliers:
    print(f"  Samples: {all_outliers}")
    rna_df  = rna_df.drop(index=[s for s in all_outliers if s in rna_df.index])
    meth_df = meth_df.drop(index=[s for s in all_outliers if s in meth_df.index])

qc_log.append(f"Outliers removed: {len(all_outliers)}")
qc_log.append("")

# ─────────────────────────────────────────────────────────────
# SECTION 5: Re-match & Save
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("SECTION 5: Final Match & Save")
print("=" * 60)

matched = sorted(set(rna_df.index) & set(meth_df.index))
rna_df  = rna_df.loc[matched]
meth_df = meth_df.loc[matched]

print(f"  Final matched patients: {len(matched)}")

rna_df.to_csv(os.path.join(OUTPUT_DIR, "brca_rnaseq_clean.csv"))
meth_df.to_csv(meth_clean_out)

qc_log.append(f"Final: {len(matched)} patients")
qc_log.append(f"  RNA-seq genes:         {rna_df.shape[1]}")
qc_log.append(f"  Methylation CpG sites: {meth_df.shape[1]:,}")

with open(os.path.join(OUTPUT_DIR, "qc_report.txt"), "w") as f:
    f.write("\n".join(qc_log))

print(f"  Saved: brca_rnaseq_clean.csv")
print(f"  Saved: brca_methylation_clean.csv")
print(f"  Saved: qc_report.txt")

print("\n" + "=" * 60)
print("STEP 3 COMPLETE")
print("=" * 60)
print(f"  Patients retained:     {len(matched)}")
print(f"  RNA-seq genes:         {rna_df.shape[1]}")
print(f"  Methylation CpG sites: {meth_df.shape[1]:,}")
print(f"  RNA-seq missing:       {rna_df.isna().sum().sum()}")
print(f"  Methylation missing:   {meth_df.isna().sum().sum()}")
print(f"\nOutputs saved to: {OUTPUT_DIR}/")
print("Next step: Run step4_normalize.py")