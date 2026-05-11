"""
Step 4 - Normalize Molecular Data
Project: Predict Tumor Metabolic State from Multi-Omic Biological Data

Normalization strategy:
    RNA-seq : Already log2 normalized by UCSC Xena Pan-Cancer pipeline.
              We apply z-score standardization (zero mean, unit variance)
              so that all genes are on the same scale for ML.

    Methylation: Beta values are already on a 0-1 scale (bounded).
              We apply z-score standardization per CpG site so that
              methylation and expression features are comparable when
              combined into one feature matrix.

Why z-score?
    - Makes features comparable across different scales
    - Required by distance-based models (SVM, KNN)
    - Prevents high-variance features from dominating
    - Standard practice for multi-omic ML pipelines

Input  (from step3_output/):
    - brca_rnaseq_clean.csv
    - brca_methylation_clean.csv

Output (to step4_output/):
    - brca_rnaseq_normalized.csv
    - brca_methylation_normalized.csv
    - brca_combined_features.csv      <- RNA + methylation merged
    - normalization_stats.csv         <- mean/std per feature (for report)
"""

import pandas as pd
import numpy as np
import os

INPUT_DIR  = "step3_output"
OUTPUT_DIR = "step4_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# SECTION 1: Load Cleaned Data
# ─────────────────────────────────────────────────────────────

print("=" * 60)
print("SECTION 1: Loading Cleaned Data")
print("=" * 60)

rna_df  = pd.read_csv(os.path.join(INPUT_DIR, "brca_rnaseq_clean.csv"),      index_col=0)
meth_df = pd.read_csv(os.path.join(INPUT_DIR, "brca_methylation_clean.csv"), index_col=0)

print(f"  RNA-seq:      {rna_df.shape[0]} patients x {rna_df.shape[1]} genes")
print(f"  Methylation:  {meth_df.shape[0]} patients x {meth_df.shape[1]} CpG sites")

# Confirm patients match
assert list(rna_df.index) == list(meth_df.index), \
    "Patient order mismatch between RNA-seq and methylation!"
print(f"  Patient order verified: OK")

# ─────────────────────────────────────────────────────────────
# SECTION 2: Normalize RNA-seq (z-score per gene)
#
# Formula: z = (x - mean) / std
# Applied column-wise (per gene across all patients)
#
# Before: log2 TPM values, range roughly -5 to 20
# After:  z-scores, mean=0, std=1 per gene
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("SECTION 2: Normalizing RNA-seq (z-score per gene)")
print("=" * 60)

print(f"  Before normalization:")
print(f"    Mean expression range: {rna_df.mean().min():.2f} to {rna_df.mean().max():.2f}")
print(f"    Std range:             {rna_df.std().min():.2f} to {rna_df.std().max():.2f}")

# Save stats for report
rna_stats = pd.DataFrame({
    "mean_before": rna_df.mean(),
    "std_before":  rna_df.std()
})

# Z-score normalization
rna_mean   = rna_df.mean(axis=0)
rna_std    = rna_df.std(axis=0)
rna_norm   = (rna_df - rna_mean) / rna_std

# Handle any zero-std columns (shouldn't exist after Step 3 but just in case)
rna_norm   = rna_norm.fillna(0)

print(f"\n  After normalization:")
print(f"    Mean range: {rna_norm.mean().min():.4f} to {rna_norm.mean().max():.4f}  (should be ~0)")
print(f"    Std range:  {rna_norm.std().min():.4f} to {rna_norm.std().max():.4f}   (should be ~1)")
print(f"    Any NaN:    {rna_norm.isna().sum().sum()}")

rna_stats["mean_after"] = rna_norm.mean()
rna_stats["std_after"]  = rna_norm.std()

# ─────────────────────────────────────────────────────────────
# SECTION 3: Normalize Methylation (z-score per CpG site)
#
# Beta values are bounded 0-1 but z-scoring still helps
# because different CpG sites have very different baseline
# methylation levels (some always ~0.8, some always ~0.1)
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("SECTION 3: Normalizing Methylation (z-score per CpG site)")
print("=" * 60)

print(f"  Before normalization:")
print(f"    Mean beta range: {meth_df.mean().min():.3f} to {meth_df.mean().max():.3f}")
print(f"    Std range:       {meth_df.std().min():.4f} to {meth_df.std().max():.4f}")
print(f"  Normalizing {meth_df.shape[1]:,} CpG sites...")

meth_mean  = meth_df.mean(axis=0)
meth_std   = meth_df.std(axis=0)
meth_norm  = (meth_df - meth_mean) / meth_std
meth_norm  = meth_norm.fillna(0)

print(f"\n  After normalization:")
print(f"    Mean range: {meth_norm.mean().min():.4f} to {meth_norm.mean().max():.4f}  (should be ~0)")
print(f"    Std range:  {meth_norm.std().min():.4f} to {meth_norm.std().max():.4f}   (should be ~1)")
print(f"    Any NaN:    {meth_norm.isna().sum().sum()}")

# ─────────────────────────────────────────────────────────────
# SECTION 4: Build Combined Feature Matrix
#
# Merge RNA-seq and methylation into one matrix per patient.
# Add prefixes to column names so we can tell them apart later.
#
# Final shape: 768 patients x (n_genes + n_cpg) features
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("SECTION 4: Building Combined Feature Matrix")
print("=" * 60)

# Add prefixes to distinguish feature types
rna_norm.columns  = ["RNA_" + c for c in rna_norm.columns]
meth_norm.columns = ["METH_" + c for c in meth_norm.columns]

# Combine side by side (patients as rows)
combined = pd.concat([rna_norm, meth_norm], axis=1)

print(f"  RNA-seq features:      {rna_norm.shape[1]}")
print(f"  Methylation features:  {meth_norm.shape[1]:,}")
print(f"  Combined shape:        {combined.shape[0]} patients x {combined.shape[1]:,} features")
print(f"  Any missing values:    {combined.isna().sum().sum()}")

# ─────────────────────────────────────────────────────────────
# SECTION 5: Save All Outputs
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("SECTION 5: Saving Outputs")
print("=" * 60)

rna_out      = os.path.join(OUTPUT_DIR, "brca_rnaseq_normalized.csv")
meth_out     = os.path.join(OUTPUT_DIR, "brca_methylation_normalized.csv")
combined_out = os.path.join(OUTPUT_DIR, "brca_combined_features.csv")
stats_out    = os.path.join(OUTPUT_DIR, "normalization_stats.csv")

rna_norm.to_csv(rna_out)
meth_norm.to_csv(meth_out)
combined.to_csv(combined_out)
rna_stats.to_csv(stats_out)

print(f"  Saved: brca_rnaseq_normalized.csv")
print(f"  Saved: brca_methylation_normalized.csv")
print(f"  Saved: brca_combined_features.csv")
print(f"  Saved: normalization_stats.csv")

# ─────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 4 COMPLETE - NORMALIZATION SUMMARY")
print("=" * 60)
print(f"  Patients:                  {combined.shape[0]}")
print(f"  RNA-seq genes normalized:  {rna_norm.shape[1]}")
print(f"  CpG sites normalized:      {meth_norm.shape[1]:,}")
print(f"  Total features combined:   {combined.shape[1]:,}")
print(f"  Missing values:            {combined.isna().sum().sum()}")
print(f"\nOutputs saved to: {OUTPUT_DIR}/")
print("Next step: Run step5_assign_labels.py")
print("  (assigns Glycolytic / Oxidative / Mixed labels to each patient)")