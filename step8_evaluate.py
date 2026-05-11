"""
Step 8 - Model Evaluation & Visualization
Project: Predict Tumor Metabolic State from Multi-Omic Biological Data

This script:
1. Loads trained models and test set
2. Evaluates each model on the held-out test set
3. Generates publication-quality figures for the final report

Figures generated:
    1. model_comparison.png       - CV vs Test accuracy bar chart
    2. confusion_matrix_LR.png    - Logistic Regression confusion matrix
    3. confusion_matrix_RF.png    - Random Forest confusion matrix
    4. confusion_matrix_SVM.png   - SVM confusion matrix
    5. roc_curves.png             - ROC curves for all models
    6. feature_importance.png     - Top 20 features by RF importance
    7. metabolic_scores.png       - Patient score distributions by label
    8. f1_per_class.png           - F1 score per class per model

Input:
    - step7_output/ (models, train/test splits)
    - step6_output/ (feature importance)
    - step5_output/ (metabolic scores)

Output (to step8_output/):
    - evaluation_report.txt
    - all .png figures
"""

import pandas as pd
import numpy as np
import os
import pickle
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — saves files without showing window
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, roc_curve, auc
)
from sklearn.preprocessing import label_binarize

INPUT_DIR_7  = "step7_output"
INPUT_DIR_6  = "step6_output"
INPUT_DIR_5  = "step5_output"
OUTPUT_DIR   = "step8_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASS_NAMES  = ["Glycolytic", "Mixed", "Oxidative"]
COLORS       = {"Logistic Regression": "#2196F3",
                "Random Forest":       "#4CAF50",
                "SVM":                 "#FF5722"}
CLASS_COLORS = ["#E53935", "#FB8C00", "#43A047"]

# ─────────────────────────────────────────────────────────────
# SECTION 1: Load Everything
# ─────────────────────────────────────────────────────────────

print("=" * 60)
print("SECTION 1: Loading Models and Test Data")
print("=" * 60)

X_test  = pd.read_csv(os.path.join(INPUT_DIR_7, "X_test.csv"),  index_col=0)
y_test  = pd.read_csv(os.path.join(INPUT_DIR_7, "y_test.csv"),  index_col=0).squeeze()
X_train = pd.read_csv(os.path.join(INPUT_DIR_7, "X_train.csv"), index_col=0)
y_train = pd.read_csv(os.path.join(INPUT_DIR_7, "y_train.csv"), index_col=0).squeeze()

with open(os.path.join(INPUT_DIR_7, "label_encoder.pkl"), "rb") as f:
    le = pickle.load(f)

y_test_enc  = le.transform(y_test)
y_train_enc = le.transform(y_train)

model_names = ["Logistic Regression", "Random Forest", "SVM"]
models = {}
for name in model_names:
    path = os.path.join(INPUT_DIR_7, f"{name.replace(' ', '_')}.pkl")
    with open(path, "rb") as f:
        models[name] = pickle.load(f)
    print(f"  Loaded: {name}")

cv_df = pd.read_csv(os.path.join(INPUT_DIR_6, "feature_importance.csv"))
scores_df = pd.read_csv(os.path.join(INPUT_DIR_5, "brca_metabolic_scores.csv"), index_col=0)

print(f"  Test set:  {X_test.shape[0]} patients x {X_test.shape[1]} features")
print(f"  Train set: {X_train.shape[0]} patients x {X_train.shape[1]} features")

# ─────────────────────────────────────────────────────────────
# SECTION 2: Evaluate All Models on Test Set
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("SECTION 2: Test Set Evaluation")
print("=" * 60)

results = {}
report_lines = [
    "STEP 8 - MODEL EVALUATION REPORT",
    "=" * 60,
    f"Test set: {X_test.shape[0]} patients",
    f"Classes: {CLASS_NAMES}",
    ""
]

for name, model in models.items():
    y_pred = model.predict(X_test.values)
    acc    = accuracy_score(y_test_enc, y_pred)
    f1     = f1_score(y_test_enc, y_pred, average="macro")
    f1_per = f1_score(y_test_enc, y_pred, average=None)
    cm     = confusion_matrix(y_test_enc, y_pred)
    report = classification_report(y_test_enc, y_pred,
                                   target_names=CLASS_NAMES)

    # Probabilities for ROC
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test.values)
    else:
        from sklearn.calibration import CalibratedClassifierCV
        y_prob = None

    results[name] = {
        "y_pred":   y_pred,
        "y_prob":   y_prob,
        "accuracy": acc,
        "f1_macro": f1,
        "f1_per":   f1_per,
        "cm":       cm
    }

    print(f"\n  {name}")
    print(f"    Test Accuracy: {acc:.4f}")
    print(f"    Test F1 Macro: {f1:.4f}")
    print(f"    Per-class F1:  Glycolytic={f1_per[0]:.3f}  "
          f"Mixed={f1_per[1]:.3f}  Oxidative={f1_per[2]:.3f}")

    report_lines += [
        f"{'='*40}",
        f"Model: {name}",
        f"  Test Accuracy: {acc:.4f}",
        f"  Test F1 Macro: {f1:.4f}",
        f"  Per-class F1:",
        f"    Glycolytic: {f1_per[0]:.4f}",
        f"    Mixed:      {f1_per[1]:.4f}",
        f"    Oxidative:  {f1_per[2]:.4f}",
        "",
        "  Classification Report:",
        report,
        ""
    ]

# ─────────────────────────────────────────────────────────────
# FIGURE 1: Model Comparison Bar Chart
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("SECTION 3: Generating Figures")
print("=" * 60)

cv_results = pd.read_csv(os.path.join(INPUT_DIR_7, "model_cv_results.csv"))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Model Performance Comparison", fontsize=14, fontweight="bold")

x     = np.arange(len(model_names))
width = 0.35

# Accuracy comparison
cv_accs   = [cv_results[cv_results["model"]==n]["cv_accuracy"].values[0]  for n in model_names]
test_accs = [results[n]["accuracy"] for n in model_names]
ax = axes[0]
bars1 = ax.bar(x - width/2, cv_accs,   width, label="CV Accuracy",   color="#90CAF9", edgecolor="black")
bars2 = ax.bar(x + width/2, test_accs, width, label="Test Accuracy",  color="#1565C0", edgecolor="black")
ax.set_xticks(x)
ax.set_xticklabels(["LR", "RF", "SVM"], fontsize=11)
ax.set_ylabel("Accuracy")
ax.set_title("Accuracy: CV vs Test")
ax.set_ylim(0, 1.05)
ax.legend()
ax.axhline(0.333, color="red", linestyle="--", alpha=0.5, label="Random chance")
for bar in bars1: ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                           f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
for bar in bars2: ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                           f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

# F1 comparison
cv_f1s   = [cv_results[cv_results["model"]==n]["cv_f1_macro"].values[0] for n in model_names]
test_f1s = [results[n]["f1_macro"] for n in model_names]
ax = axes[1]
bars3 = ax.bar(x - width/2, cv_f1s,   width, label="CV F1 Macro",   color="#A5D6A7", edgecolor="black")
bars4 = ax.bar(x + width/2, test_f1s, width, label="Test F1 Macro", color="#2E7D32", edgecolor="black")
ax.set_xticks(x)
ax.set_xticklabels(["LR", "RF", "SVM"], fontsize=11)
ax.set_ylabel("F1 Score (Macro)")
ax.set_title("F1 Score: CV vs Test")
ax.set_ylim(0, 1.05)
ax.legend()
for bar in bars3: ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                           f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
for bar in bars4: ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                           f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "model_comparison.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: model_comparison.png")

# ─────────────────────────────────────────────────────────────
# FIGURE 2-4: Confusion Matrices
# ─────────────────────────────────────────────────────────────

for name in model_names:
    cm = results[name]["cm"]
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Confusion Matrix — {name}", fontsize=13, fontweight="bold")

    for ax, data, title in zip(axes,
                                [cm, cm_norm],
                                ["Raw Counts", "Normalized (row %)"]):
        im = ax.imshow(data, interpolation="nearest", cmap="Blues")
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right")
        ax.set_yticklabels(CLASS_NAMES)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title(title)
        plt.colorbar(im, ax=ax)
        thresh = data.max() / 2
        for i in range(3):
            for j in range(3):
                val = f"{data[i,j]:.2f}" if title != "Raw Counts" else str(data[i,j])
                ax.text(j, i, val, ha="center", va="center",
                        color="white" if data[i,j] > thresh else "black", fontsize=12)

    plt.tight_layout()
    fname = f"confusion_matrix_{name.replace(' ', '_')}.png"
    plt.savefig(os.path.join(OUTPUT_DIR, fname), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")

# ─────────────────────────────────────────────────────────────
# FIGURE 5: ROC Curves
# ─────────────────────────────────────────────────────────────

y_bin = label_binarize(y_test_enc, classes=[0, 1, 2])
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("ROC Curves — One vs Rest (per class)", fontsize=13, fontweight="bold")

for ax, name in zip(axes, model_names):
    y_prob = results[name]["y_prob"]
    if y_prob is None:
        ax.text(0.5, 0.5, "Probabilities\nnot available", ha="center", va="center")
        ax.set_title(name)
        continue
    for i, cls in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=CLASS_COLORS[i], lw=2,
                label=f"{cls} (AUC={roc_auc:.3f})")
    ax.plot([0,1],[0,1], "k--", lw=1, alpha=0.5)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(name)
    ax.legend(loc="lower right", fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "roc_curves.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: roc_curves.png")

# ─────────────────────────────────────────────────────────────
# FIGURE 6: Feature Importance (Top 20)
# ─────────────────────────────────────────────────────────────

top20 = cv_df.head(20).copy()
top20["color"] = top20["feature"].apply(
    lambda x: "#1565C0" if x.startswith("RNA_") else "#2E7D32")
top20["short"] = top20["feature"].str.replace("RNA_", "").str.replace("METH_", "")

fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.barh(range(len(top20)), top20["importance"].values,
               color=top20["color"].values, edgecolor="black", alpha=0.85)
ax.set_yticks(range(len(top20)))
ax.set_yticklabels(top20["short"].values, fontsize=9)
ax.invert_yaxis()
ax.set_xlabel("Random Forest Feature Importance")
ax.set_title("Top 20 Features by Importance\n(Blue = RNA-seq | Green = Methylation)",
             fontweight="bold")

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor="#1565C0", label="RNA-seq gene"),
                   Patch(facecolor="#2E7D32", label="Methylation CpG")]
ax.legend(handles=legend_elements, loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: feature_importance.png")

# ─────────────────────────────────────────────────────────────
# FIGURE 7: Metabolic Score Distributions
# ─────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Metabolic Score Distributions by Label", fontsize=13, fontweight="bold")

score_cols = ["glycolysis_score", "oxphos_score", "metabolic_ratio"]
titles     = ["Glycolysis Score", "OXPHOS Score", "Metabolic Ratio\n(Glycolysis - OXPHOS)"]
label_col  = "metabolic_label"

for ax, col, title in zip(axes, score_cols, titles):
    for label, color in zip(CLASS_NAMES, CLASS_COLORS):
        subset = scores_df[scores_df[label_col] == label][col]
        ax.hist(subset, bins=30, alpha=0.6, color=color, label=label, edgecolor="white")
    ax.set_xlabel("Score")
    ax.set_ylabel("Number of Patients")
    ax.set_title(title)
    ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "metabolic_scores.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: metabolic_scores.png")

# ─────────────────────────────────────────────────────────────
# FIGURE 8: Per-Class F1 Scores
# ─────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 5))
x     = np.arange(len(CLASS_NAMES))
width = 0.25

for i, name in enumerate(model_names):
    f1_per = results[name]["f1_per"]
    bars   = ax.bar(x + i*width, f1_per, width,
                    label=name, color=list(COLORS.values())[i],
                    edgecolor="black", alpha=0.85)
    for bar in bars:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)

ax.set_xticks(x + width)
ax.set_xticklabels(CLASS_NAMES, fontsize=11)
ax.set_ylabel("F1 Score")
ax.set_title("Per-Class F1 Score by Model", fontweight="bold")
ax.set_ylim(0, 1.1)
ax.legend()
ax.axhline(0.333, color="red", linestyle="--", alpha=0.4, label="Random chance")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "f1_per_class.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: f1_per_class.png")

# ─────────────────────────────────────────────────────────────
# SECTION 4: Save Evaluation Report
# ─────────────────────────────────────────────────────────────

with open(os.path.join(OUTPUT_DIR, "evaluation_report.txt"), "w") as f:
    f.write("\n".join(report_lines))
print("  Saved: evaluation_report.txt")

# ─────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 8 COMPLETE - EVALUATION SUMMARY")
print("=" * 60)
print(f"\n  {'Model':<25} {'Test Acc':>10} {'Test F1':>10}")
print(f"  {'-'*25} {'-'*10} {'-'*10}")
for name in model_names:
    print(f"  {name:<25} {results[name]['accuracy']:>10.4f} {results[name]['f1_macro']:>10.4f}")

best_name = max(results, key=lambda n: results[n]["f1_macro"])
print(f"\n  Best model on test set: {best_name}")
print(f"  Test Accuracy: {results[best_name]['accuracy']:.4f}")
print(f"  Test F1 Macro: {results[best_name]['f1_macro']:.4f}")
print(f"\n  Figures saved to: {OUTPUT_DIR}/")
print("  model_comparison.png")
print("  confusion_matrix_*.png  (3 files)")
print("  roc_curves.png")
print("  feature_importance.png")
print("  metabolic_scores.png")
print("  f1_per_class.png")
print("\nNext step: Run step9_hyperparameter_tuning.py")