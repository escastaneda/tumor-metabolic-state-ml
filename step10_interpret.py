"""
Step 10 - Model Interpretation & Feature Explanation
Project: Predict Tumor Metabolic State from Multi-Omic Biological Data

Since SHAP KernelExplainer fails with 500 features (too high-dimensional),
we use two robust alternatives:

1. Permutation Importance (model-agnostic):
   Shuffle one feature at a time and measure accuracy drop.
   Large drop = feature is important. Works with any model.

2. SVM Decision Function Weights (for linear kernel):
   If best SVM used linear kernel, coefficients directly show
   feature contributions per class.

3. Class-conditional feature analysis:
   Compare mean feature values per class to show what drives
   each metabolic state biologically.

These methods are actually MORE interpretable than SHAP for a report
because they are easier to explain and visualize clearly.
"""

import pandas as pd
import numpy as np
import os
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score

INPUT_DIR_9 = "step9_output"
INPUT_DIR_7 = "step7_output"
OUTPUT_DIR  = "step10_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASS_NAMES  = ["Glycolytic", "Mixed", "Oxidative"]
CLASS_COLORS = ["#E53935", "#FB8C00", "#43A047"]

# ─────────────────────────────────────────────────────────────
# SECTION 1: Load Model and Data
# ─────────────────────────────────────────────────────────────

print("=" * 60)
print("SECTION 1: Loading Model and Data")
print("=" * 60)

with open(os.path.join(INPUT_DIR_9, "best_model.pkl"), "rb") as f:
    model = pickle.load(f)
with open(os.path.join(INPUT_DIR_7, "label_encoder.pkl"), "rb") as f:
    le = pickle.load(f)

X_train = pd.read_csv(os.path.join(INPUT_DIR_7, "X_train.csv"), index_col=0)
X_test  = pd.read_csv(os.path.join(INPUT_DIR_7, "X_test.csv"),  index_col=0)
y_test  = pd.read_csv(os.path.join(INPUT_DIR_7, "y_test.csv"),  index_col=0).squeeze()
y_train = pd.read_csv(os.path.join(INPUT_DIR_7, "y_train.csv"), index_col=0).squeeze()
y_test_enc  = le.transform(y_test)
y_train_enc = le.transform(y_train)

feature_names = list(X_test.columns)
short_names   = [f.replace("RNA_", "").replace("METH_", "meth_") for f in feature_names]
rna_mask      = np.array([f.startswith("RNA_") for f in feature_names])

print(f"  Model:  {type(model).__name__} | kernel: {model.kernel}")
print(f"  Test:   {X_test.shape[0]} patients x {X_test.shape[1]} features")
print(f"  RNA:    {rna_mask.sum()} | Methylation: {(~rna_mask).sum()}")

# ─────────────────────────────────────────────────────────────
# SECTION 2: Permutation Feature Importance
#
# For each feature:
#   1. Shuffle its values across all test patients (break signal)
#   2. Measure accuracy drop
#   3. Restore original values
#   4. Repeat 10 times and average
#
# Features that cause the biggest accuracy drop when shuffled
# are the most important for the model's predictions.
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("SECTION 2: Permutation Feature Importance")
print("=" * 60)

print("  Computing permutation importance (10 repeats)...")
print("  This takes 3-5 minutes...")

perm = permutation_importance(
    model,
    X_test.values, y_test_enc,
    n_repeats    = 10,
    random_state = 42,
    n_jobs       = -1,
    scoring      = "f1_macro"
)

perm_df = pd.DataFrame({
    "feature":      feature_names,
    "short_name":   short_names,
    "importance":   perm.importances_mean,
    "std":          perm.importances_std,
    "feature_type": ["RNA" if f.startswith("RNA_") else "Methylation"
                     for f in feature_names]
}).sort_values("importance", ascending=False)

print(f"\n  Top 15 features by permutation importance:")
print(perm_df.head(15)[["short_name", "importance", "std", "feature_type"]].to_string(index=False))

perm_df.to_csv(os.path.join(OUTPUT_DIR, "permutation_importance.csv"), index=False)

# ─────────────────────────────────────────────────────────────
# SECTION 3: SVM Coefficients (if linear kernel)
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("SECTION 3: SVM Decision Boundary Analysis")
print("=" * 60)

coef_df = None
if model.kernel == "linear":
    print("  Linear kernel detected — extracting coefficients...")
    # coef_ shape: (n_classes*(n_classes-1)/2, n_features) for OvO
    # For 3 classes: 3 pairs (0v1, 0v2, 1v2)
    coefs = model.coef_
    print(f"  Coefficient matrix shape: {coefs.shape}")

    # Average absolute coefficient across all class pairs
    mean_coef = np.abs(coefs).mean(axis=0)

    coef_df = pd.DataFrame({
        "feature":      feature_names,
        "short_name":   short_names,
        "mean_abs_coef": mean_coef,
        "feature_type": ["RNA" if f.startswith("RNA_") else "Methylation"
                         for f in feature_names]
    }).sort_values("mean_abs_coef", ascending=False)

    print(f"\n  Top 15 features by SVM coefficient:")
    print(coef_df.head(15)[["short_name", "mean_abs_coef", "feature_type"]].to_string(index=False))
    coef_df.to_csv(os.path.join(OUTPUT_DIR, "svm_coefficients.csv"), index=False)
else:
    print(f"  Kernel: {model.kernel} — coefficients not directly available for RBF")
    print("  Using permutation importance as primary interpretation method.")

# ─────────────────────────────────────────────────────────────
# SECTION 4: Class-Conditional Feature Profiles
#
# For each metabolic class, compute the mean feature value
# and compare across classes. This shows the biological
# signature of each metabolic state directly from the data.
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("SECTION 4: Class-Conditional Feature Profiles")
print("=" * 60)

# Combine train and test for better statistics
X_all = pd.concat([X_train, X_test])
y_all = pd.concat([y_train, y_test])

class_profiles = {}
for cls in CLASS_NAMES:
    mask = y_all == cls
    class_profiles[cls] = X_all[mask].mean()
    print(f"  {cls}: {mask.sum()} patients")

profile_df = pd.DataFrame(class_profiles)
profile_df["feature_type"] = ["RNA" if f.startswith("RNA_") else "Methylation"
                               for f in profile_df.index]

# Focus on RNA genes — most biologically interpretable
rna_profile = profile_df[profile_df["feature_type"] == "RNA"].copy()
rna_profile.index = [i.replace("RNA_", "") for i in rna_profile.index]
rna_profile = rna_profile.drop("feature_type", axis=1)

print(f"\n  RNA gene mean expression by class (z-scores):")
print(rna_profile.to_string())

# ─────────────────────────────────────────────────────────────
# SECTION 5: Generate Figures
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("SECTION 5: Generating Figures")
print("=" * 60)

# ── FIGURE 1: Permutation Importance — Top 20 ────────────────

top20 = perm_df.head(20).copy()
colors_bar = ["#1565C0" if t == "RNA" else "#2E7D32"
              for t in top20["feature_type"]]

fig, ax = plt.subplots(figsize=(10, 8))
y_pos = range(len(top20))
ax.barh(y_pos, top20["importance"].values[::-1],
        xerr=top20["std"].values[::-1],
        color=colors_bar[::-1], edgecolor="black", alpha=0.85,
        capsize=3)
ax.set_yticks(y_pos)
ax.set_yticklabels(top20["short_name"].values[::-1], fontsize=9)
ax.set_xlabel("Mean decrease in F1 Macro when feature is shuffled", fontsize=10)
ax.set_title("Top 20 Features — Permutation Importance\n"
             "(Higher = more critical for model performance)",
             fontsize=12, fontweight="bold")
ax.axvline(0, color="black", lw=0.8)

rna_p  = mpatches.Patch(color="#1565C0", label="RNA-seq gene")
meth_p = mpatches.Patch(color="#2E7D32", label="Methylation CpG")
ax.legend(handles=[rna_p, meth_p], fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "permutation_importance.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: permutation_importance.png")

# ── FIGURE 2: RNA Gene Heatmap by Metabolic Class ────────────

fig, ax = plt.subplots(figsize=(10, 10))
data    = rna_profile[CLASS_NAMES].values
im      = ax.imshow(data, cmap="RdBu_r", aspect="auto")

ax.set_xticks(range(3))
ax.set_xticklabels(CLASS_NAMES, fontsize=12, fontweight="bold")
ax.set_yticks(range(len(rna_profile)))
ax.set_yticklabels(rna_profile.index, fontsize=9)
ax.set_title("Mean Normalized Expression of Metabolic Genes\nby Tumor Metabolic Class",
             fontsize=12, fontweight="bold")

plt.colorbar(im, ax=ax, label="Mean z-score expression")

# Add text values
for i in range(len(rna_profile)):
    for j in range(3):
        val = data[i, j]
        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                fontsize=7, color="white" if abs(val) > 0.5 else "black")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "rna_class_heatmap.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: rna_class_heatmap.png")

# ── FIGURE 3: Top RNA Gene Distributions by Class ────────────

# Pick top 6 RNA genes by permutation importance
top_rna = perm_df[perm_df["feature_type"] == "RNA"].head(6)

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle("Expression Distribution of Top RNA Features by Metabolic Class",
             fontsize=12, fontweight="bold")

for ax, (_, row) in zip(axes.flat, top_rna.iterrows()):
    gene     = row["feature"]
    genename = row["short_name"]
    for cls, color in zip(CLASS_NAMES, CLASS_COLORS):
        mask   = y_all == cls
        values = X_all[mask][gene].values
        ax.hist(values, bins=25, alpha=0.6, color=color,
                label=cls, edgecolor="white", density=True)
    ax.set_title(genename, fontweight="bold", fontsize=11)
    ax.set_xlabel("Normalized expression (z-score)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "top_gene_distributions.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: top_gene_distributions.png")

# ── FIGURE 4: Glycolysis vs OXPHOS Score Scatter ─────────────

scores = pd.read_csv("step5_output/brca_metabolic_scores.csv", index_col=0)

fig, ax = plt.subplots(figsize=(9, 7))
for cls, color in zip(CLASS_NAMES, CLASS_COLORS):
    mask = scores["metabolic_label"] == cls
    ax.scatter(scores[mask]["glycolysis_score"],
               scores[mask]["oxphos_score"],
               c=color, label=cls, alpha=0.6, s=30, edgecolors="none")

ax.set_xlabel("Glycolysis Score (mean z-score)", fontsize=12)
ax.set_ylabel("OXPHOS Score (mean z-score)", fontsize=12)
ax.set_title("Tumor Metabolic Phenotype Space\n"
             "Glycolysis vs Oxidative Phosphorylation Scores",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=11, title="Metabolic Class", title_fontsize=10)
ax.axhline(0, color="gray", lw=0.5, linestyle="--")
ax.axvline(0, color="gray", lw=0.5, linestyle="--")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "metabolic_phenotype_space.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: metabolic_phenotype_space.png")

# ── FIGURE 5: SVM Coefficients if linear kernel ───────────────

if coef_df is not None:
    top20c  = coef_df.head(20)
    colorsC = ["#1565C0" if t == "RNA" else "#2E7D32"
               for t in top20c["feature_type"]]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(top20c)), top20c["mean_abs_coef"].values[::-1],
            color=colorsC[::-1], edgecolor="black", alpha=0.85)
    ax.set_yticks(range(len(top20c)))
    ax.set_yticklabels(top20c["short_name"].values[::-1], fontsize=9)
    ax.set_xlabel("Mean |SVM Coefficient|")
    ax.set_title("Top 20 Features by SVM Coefficient\n(Linear kernel)",
                 fontweight="bold")
    ax.legend(handles=[rna_p, meth_p])
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "svm_coefficients.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: svm_coefficients.png")

# ─────────────────────────────────────────────────────────────
# SECTION 6: Interpretation Report
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("SECTION 6: Saving Interpretation Report")
print("=" * 60)

report = [
    "STEP 10 - INTERPRETATION REPORT",
    "=" * 60,
    f"Model: {type(model).__name__} (kernel={model.kernel})",
    f"Method: Permutation Importance + Class-Conditional Profiles",
    "",
    "TOP 20 FEATURES BY PERMUTATION IMPORTANCE:",
    "=" * 40,
]
for _, row in perm_df.head(20).iterrows():
    report.append(f"  {row['short_name']:35s} {row['importance']:+.5f} "
                  f"+/- {row['std']:.5f}  [{row['feature_type']}]")

report += [
    "",
    "RNA GENE CLASS PROFILES (mean z-score):",
    "=" * 40,
    f"{'Gene':20s} {'Glycolytic':>12} {'Mixed':>10} {'Oxidative':>12}",
    "-" * 55,
]
for gene in rna_profile.index:
    g = rna_profile.loc[gene, "Glycolytic"]
    m = rna_profile.loc[gene, "Mixed"]
    o = rna_profile.loc[gene, "Oxidative"]
    report.append(f"  {gene:20s} {g:>12.3f} {m:>10.3f} {o:>12.3f}")

report += [
    "",
    "BIOLOGICAL INTERPRETATION:",
    "=" * 40,
    "Glycolytic tumors: upregulate glycolysis enzymes (HK2, LDHA, PKM, ENO)",
    "  -> Warburg effect: aerobic glycolysis for rapid ATP and biosynthesis",
    "Oxidative tumors: upregulate OXPHOS complexes (NDUF*, COX*, ATP5*)",
    "  -> Efficient ATP generation via mitochondrial respiration",
    "Mixed tumors: intermediate expression of both pathways",
    "  -> Metabolic plasticity, may switch under stress conditions",
    "",
    "Methylation features contribute regulatory information:",
    "  -> Promoter methylation can silence metabolic genes",
    "  -> Adds epigenetic layer beyond expression alone",
]

with open(os.path.join(OUTPUT_DIR, "interpretation_report.txt"), "w") as f:
    f.write("\n".join(report))

print("  Saved: interpretation_report.txt")

print("\n" + "=" * 60)
print("STEP 10 COMPLETE")
print("=" * 60)
print(f"  Figures saved to: {OUTPUT_DIR}/")
print(f"    permutation_importance.png")
print(f"    rna_class_heatmap.png")
print(f"    top_gene_distributions.png")
print(f"    metabolic_phenotype_space.png")
if coef_df is not None:
    print(f"    svm_coefficients.png")
print(f"  interpretation_report.txt")
print(f"\nNext step: Run step11_final_report.py")