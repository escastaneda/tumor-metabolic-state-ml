"""
Step 9 - Hyperparameter Tuning
Project: Predict Tumor Metabolic State from Multi-Omic Biological Data

We tune the two best models from Step 8:
    1. Logistic Regression (best test accuracy: 0.7706)
    2. SVM                 (best generalization, test F1: 0.7509)

Method: GridSearchCV with 5-fold stratified cross-validation
    - Exhaustively tests all hyperparameter combinations
    - Selects best combination by CV F1 macro score
    - Refits best model on full training set

Hyperparameters tuned:
    Logistic Regression:
        C         : regularization strength (smaller = more regularization)
        solver    : optimization algorithm
        penalty   : regularization type (l1, l2)

    SVM:
        C         : regularization strength
        kernel    : rbf, linear, poly
        gamma     : rbf kernel coefficient

Input  (from step7_output/):
    - X_train.csv, y_train.csv
    - X_test.csv,  y_test.csv
    - label_encoder.pkl

Output (to step9_output/):
    - tuning_results_LR.csv
    - tuning_results_SVM.csv
    - best_model.pkl
    - tuning_report.txt
    - tuning_comparison.png
"""

import pandas as pd
import numpy as np
import os
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model    import LogisticRegression
from sklearn.svm             import SVC
from sklearn.metrics         import (accuracy_score, f1_score,
                                     classification_report, confusion_matrix)
from sklearn.preprocessing   import label_binarize
from sklearn.metrics         import roc_curve, auc

INPUT_DIR  = "step7_output"
OUTPUT_DIR = "step9_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASS_NAMES = ["Glycolytic", "Mixed", "Oxidative"]

# ─────────────────────────────────────────────────────────────
# SECTION 1: Load Data
# ─────────────────────────────────────────────────────────────

print("=" * 60)
print("SECTION 1: Loading Train/Test Data")
print("=" * 60)

X_train = pd.read_csv(os.path.join(INPUT_DIR, "X_train.csv"), index_col=0)
X_test  = pd.read_csv(os.path.join(INPUT_DIR, "X_test.csv"),  index_col=0)
y_train = pd.read_csv(os.path.join(INPUT_DIR, "y_train.csv"), index_col=0).squeeze()
y_test  = pd.read_csv(os.path.join(INPUT_DIR, "y_test.csv"),  index_col=0).squeeze()

with open(os.path.join(INPUT_DIR, "label_encoder.pkl"), "rb") as f:
    le = pickle.load(f)

y_train_enc = le.transform(y_train)
y_test_enc  = le.transform(y_test)

print(f"  Train: {X_train.shape[0]} patients x {X_train.shape[1]} features")
print(f"  Test:  {X_test.shape[0]} patients x {X_test.shape[1]} features")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

report_lines = [
    "STEP 9 - HYPERPARAMETER TUNING REPORT",
    "=" * 60,
    ""
]

tuning_summary = {}

# ─────────────────────────────────────────────────────────────
# SECTION 2: Tune Logistic Regression
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("SECTION 2: Tuning Logistic Regression")
print("=" * 60)

lr_param_grid = {
    "C":        [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    "penalty":  ["l2"],
    "solver":   ["lbfgs", "saga"],
    "max_iter": [2000]
}

print(f"  Grid size: {6 * 1 * 2} combinations x 5 folds = {6*2*5} fits")
print(f"  Searching...")

lr_grid = GridSearchCV(
    LogisticRegression(random_state=42, multi_class="multinomial"),
    lr_param_grid,
    cv      = cv,
    scoring = "f1_macro",
    n_jobs  = -1,
    verbose = 1
)
lr_grid.fit(X_train.values, y_train_enc)

print(f"\n  Best parameters: {lr_grid.best_params_}")
print(f"  Best CV F1:      {lr_grid.best_score_:.4f}")

# Evaluate best LR on test set
lr_best   = lr_grid.best_estimator_
lr_pred   = lr_best.predict(X_test.values)
lr_prob   = lr_best.predict_proba(X_test.values)
lr_acc    = accuracy_score(y_test_enc, lr_pred)
lr_f1     = f1_score(y_test_enc, lr_pred, average="macro")
lr_f1_per = f1_score(y_test_enc, lr_pred, average=None)
lr_cm     = confusion_matrix(y_test_enc, lr_pred)

print(f"  Test Accuracy: {lr_acc:.4f}")
print(f"  Test F1 Macro: {lr_f1:.4f}")
print(f"  Per-class F1:  Glycolytic={lr_f1_per[0]:.3f}  "
      f"Mixed={lr_f1_per[1]:.3f}  Oxidative={lr_f1_per[2]:.3f}")

# Save CV results
lr_cv_results = pd.DataFrame(lr_grid.cv_results_)
lr_cv_results.to_csv(os.path.join(OUTPUT_DIR, "tuning_results_LR.csv"), index=False)

tuning_summary["Logistic Regression (tuned)"] = {
    "best_params": lr_grid.best_params_,
    "cv_f1":       lr_grid.best_score_,
    "test_acc":    lr_acc,
    "test_f1":     lr_f1,
    "f1_per":      lr_f1_per,
    "cm":          lr_cm,
    "prob":        lr_prob
}

report_lines += [
    "LOGISTIC REGRESSION TUNING",
    f"  Best params:   {lr_grid.best_params_}",
    f"  Best CV F1:    {lr_grid.best_score_:.4f}",
    f"  Test Accuracy: {lr_acc:.4f}",
    f"  Test F1 Macro: {lr_f1:.4f}",
    f"  Per-class F1: Glycolytic={lr_f1_per[0]:.4f} "
    f"Mixed={lr_f1_per[1]:.4f} Oxidative={lr_f1_per[2]:.4f}",
    "",
    classification_report(y_test_enc, lr_pred, target_names=CLASS_NAMES),
    ""
]

# ─────────────────────────────────────────────────────────────
# SECTION 3: Tune SVM
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("SECTION 3: Tuning SVM")
print("=" * 60)

svm_param_grid = {
    "C":      [0.1, 1.0, 10.0, 100.0],
    "kernel": ["rbf", "linear"],
    "gamma":  ["scale", "auto"]
}

n_svm_combos = 4 * 2 * 2
print(f"  Grid size: {n_svm_combos} combinations x 5 folds = {n_svm_combos*5} fits")
print(f"  Searching (may take 5-10 minutes)...")

svm_grid = GridSearchCV(
    SVC(probability=True, random_state=42),
    svm_param_grid,
    cv      = cv,
    scoring = "f1_macro",
    n_jobs  = -1,
    verbose = 1
)
svm_grid.fit(X_train.values, y_train_enc)

print(f"\n  Best parameters: {svm_grid.best_params_}")
print(f"  Best CV F1:      {svm_grid.best_score_:.4f}")

svm_best   = svm_grid.best_estimator_
svm_pred   = svm_best.predict(X_test.values)
svm_prob   = svm_best.predict_proba(X_test.values)
svm_acc    = accuracy_score(y_test_enc, svm_pred)
svm_f1     = f1_score(y_test_enc, svm_pred, average="macro")
svm_f1_per = f1_score(y_test_enc, svm_pred, average=None)
svm_cm     = confusion_matrix(y_test_enc, svm_pred)

print(f"  Test Accuracy: {svm_acc:.4f}")
print(f"  Test F1 Macro: {svm_f1:.4f}")
print(f"  Per-class F1:  Glycolytic={svm_f1_per[0]:.3f}  "
      f"Mixed={svm_f1_per[1]:.3f}  Oxidative={svm_f1_per[2]:.3f}")

svm_cv_results = pd.DataFrame(svm_grid.cv_results_)
svm_cv_results.to_csv(os.path.join(OUTPUT_DIR, "tuning_results_SVM.csv"), index=False)

tuning_summary["SVM (tuned)"] = {
    "best_params": svm_grid.best_params_,
    "cv_f1":       svm_grid.best_score_,
    "test_acc":    svm_acc,
    "test_f1":     svm_f1,
    "f1_per":      svm_f1_per,
    "cm":          svm_cm,
    "prob":        svm_prob
}

report_lines += [
    "SVM TUNING",
    f"  Best params:   {svm_grid.best_params_}",
    f"  Best CV F1:    {svm_grid.best_score_:.4f}",
    f"  Test Accuracy: {svm_acc:.4f}",
    f"  Test F1 Macro: {svm_f1:.4f}",
    f"  Per-class F1: Glycolytic={svm_f1_per[0]:.4f} "
    f"Mixed={svm_f1_per[1]:.4f} Oxidative={svm_f1_per[2]:.4f}",
    "",
    classification_report(y_test_enc, svm_pred, target_names=CLASS_NAMES),
    ""
]

# ─────────────────────────────────────────────────────────────
# SECTION 4: Select & Save Best Overall Model
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("SECTION 4: Selecting Best Model")
print("=" * 60)

best_name  = max(tuning_summary, key=lambda n: tuning_summary[n]["test_f1"])
best_model = lr_best if "Logistic" in best_name else svm_best

print(f"  Best model: {best_name}")
print(f"  Test F1:    {tuning_summary[best_name]['test_f1']:.4f}")
print(f"  Test Acc:   {tuning_summary[best_name]['test_acc']:.4f}")

with open(os.path.join(OUTPUT_DIR, "best_model.pkl"), "wb") as f:
    pickle.dump(best_model, f)
print(f"  Saved: best_model.pkl")

# ─────────────────────────────────────────────────────────────
# SECTION 5: Visualizations
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("SECTION 5: Generating Figures")
print("=" * 60)

# Before vs After comparison
untuned = {
    "Logistic Regression": {"acc": 0.7706, "f1": 0.7705},
    "SVM":                 {"acc": 0.7489, "f1": 0.7509}
}
tuned = {
    "Logistic Regression": {"acc": lr_acc, "f1": lr_f1},
    "SVM":                 {"acc": svm_acc, "f1": svm_f1}
}

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Hyperparameter Tuning: Before vs After", fontsize=13, fontweight="bold")

model_labels = ["Logistic\nRegression", "SVM"]
x = np.arange(len(model_labels))
w = 0.35

for ax, metric, title in zip(axes, ["acc", "f1"], ["Test Accuracy", "Test F1 Macro"]):
    before = [untuned[m][metric] for m in ["Logistic Regression", "SVM"]]
    after  = [tuned[m][metric]   for m in ["Logistic Regression", "SVM"]]
    b1 = ax.bar(x - w/2, before, w, label="Before tuning",
                color="#90CAF9", edgecolor="black")
    b2 = ax.bar(x + w/2, after,  w, label="After tuning",
                color="#1565C0", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels)
    ax.set_ylabel(title)
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.legend()
    for bar in b1 + b2:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "tuning_comparison.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: tuning_comparison.png")

# Confusion matrix for best tuned model
cm   = tuning_summary[best_name]["cm"]
cm_n = cm.astype(float) / cm.sum(axis=1, keepdims=True)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(f"Best Model Confusion Matrix — {best_name}", fontsize=13, fontweight="bold")

for ax, data, title in zip(axes, [cm, cm_n], ["Raw Counts", "Normalized"]):
    im = ax.imshow(data, cmap="Blues")
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right")
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.colorbar(im, ax=ax)
    thresh = data.max() / 2
    for i in range(3):
        for j in range(3):
            val = f"{data[i,j]:.2f}" if title != "Raw Counts" else str(data[i,j])
            ax.text(j, i, val, ha="center", va="center",
                    color="white" if data[i,j] > thresh else "black", fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "best_model_confusion_matrix.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: best_model_confusion_matrix.png")

# ROC curves for best model
y_bin = label_binarize(y_test_enc, classes=[0, 1, 2])
y_prob = tuning_summary[best_name]["prob"]
CLASS_COLORS = ["#E53935", "#FB8C00", "#43A047"]

fig, ax = plt.subplots(figsize=(7, 6))
for i, cls in enumerate(CLASS_NAMES):
    fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
    roc_auc     = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=CLASS_COLORS[i], lw=2,
            label=f"{cls} (AUC = {roc_auc:.3f})")
ax.plot([0,1],[0,1], "k--", lw=1, alpha=0.5, label="Random (AUC=0.5)")
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title(f"ROC Curves — {best_name}\n(Best Tuned Model)", fontweight="bold")
ax.legend(loc="lower right")
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.05])
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "best_model_roc.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: best_model_roc.png")

# ─────────────────────────────────────────────────────────────
# Save Report
# ─────────────────────────────────────────────────────────────

with open(os.path.join(OUTPUT_DIR, "tuning_report.txt"), "w") as f:
    f.write("\n".join(report_lines))
print("  Saved: tuning_report.txt")

print("\n" + "=" * 60)
print("STEP 9 COMPLETE - TUNING SUMMARY")
print("=" * 60)
print(f"\n  {'Model':<30} {'CV F1':>8} {'Test Acc':>10} {'Test F1':>10}")
print(f"  {'-'*30} {'-'*8} {'-'*10} {'-'*10}")
for name, res in tuning_summary.items():
    print(f"  {name:<30} {res['cv_f1']:>8.4f} {res['test_acc']:>10.4f} {res['test_f1']:>10.4f}")

print(f"\n  Best model: {best_name}")
print(f"  Saved as:   step9_output/best_model.pkl")
print(f"\nOutputs saved to: {OUTPUT_DIR}/")
print("Next step: Run step10_interpret.py")