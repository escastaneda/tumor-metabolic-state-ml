"""
Step 7 - Split Dataset and Train Models
Project: Predict Tumor Metabolic State from Multi-Omic Biological Data

This script:
1. Splits data into 70% train / 30% test (stratified by class)
2. Trains 3 classifiers with 5-fold cross-validation:
      - Logistic Regression
      - Random Forest
      - Support Vector Machine (SVM)
3. Saves trained models and cross-validation results

Why these 3 models?
    Logistic Regression : Simple linear baseline. Fast, interpretable.
                          Good for checking if the problem is linearly separable.
    Random Forest       : Ensemble of decision trees. Handles non-linear
                          relationships, robust to outliers, gives feature
                          importance scores.
    SVM                 : Finds optimal decision boundary between classes.
                          Works well in high-dimensional spaces like genomics.

Why 5-fold cross-validation?
    With 768 samples, we can't afford to waste data on a large validation set.
    5-fold CV uses all training data for both training and validation by
    rotating which fold is held out. This gives a reliable estimate of
    generalization performance before touching the test set.

Input  (from step6_output/ and step5_output/):
    - brca_features_selected.csv
    - brca_labels.csv

Output (to step7_output/):
    - model_cv_results.csv       <- cross-validation scores per model
    - train_test_split.csv       <- which patients are train vs test
    - X_train.csv, X_test.csv
    - y_train.csv, y_test.csv
    - trained models saved as .pkl files
"""

import pandas as pd
import numpy as np
import os
import pickle

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.linear_model  import LogisticRegression
from sklearn.ensemble      import RandomForestClassifier
from sklearn.svm           import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics       import make_scorer, accuracy_score, f1_score

INPUT_FEATURES = "step6_output/brca_features_selected.csv"
INPUT_LABELS   = "step5_output/brca_labels.csv"
OUTPUT_DIR     = "step7_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# SECTION 1: Load Data
# ─────────────────────────────────────────────────────────────

print("=" * 60)
print("SECTION 1: Loading Data")
print("=" * 60)

X = pd.read_csv(INPUT_FEATURES, index_col=0)
y = pd.read_csv(INPUT_LABELS,   index_col=0).squeeze()

common = sorted(set(X.index) & set(y.index))
X = X.loc[common]
y = y.loc[common]

le = LabelEncoder()
y_enc = le.fit_transform(y)

print(f"  Features: {X.shape[0]} patients x {X.shape[1]} features")
print(f"  Classes:  {dict(zip(le.classes_, np.bincount(y_enc)))}")

# ─────────────────────────────────────────────────────────────
# SECTION 2: Train / Test Split
#
# 70% train, 30% test
# stratified = same class proportions in both splits
# random_state = 42 for reproducibility (same split every run)
#
# IMPORTANT: The test set is locked away after this point.
# We never look at test set performance until Step 11.
# All model selection happens on training data only.
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("SECTION 2: Train / Test Split (70/30 stratified)")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc,
    test_size    = 0.30,
    stratify     = y_enc,
    random_state = 42
)

print(f"  Training set: {X_train.shape[0]} patients")
print(f"  Test set:     {X_test.shape[0]} patients")
print(f"\n  Training class distribution:")
for i, cls in enumerate(le.classes_):
    print(f"    {cls:12s}: {np.sum(y_train == i)}")
print(f"\n  Test class distribution:")
for i, cls in enumerate(le.classes_):
    print(f"    {cls:12s}: {np.sum(y_test == i)}")

# Save split info
split_df = pd.DataFrame({
    "patient_id": common,
    "split": ["train" if p in X_train.index else "test" for p in common]
})
split_df.to_csv(os.path.join(OUTPUT_DIR, "train_test_split.csv"), index=False)

# Save train/test matrices
X_train_df = pd.DataFrame(X_train, index=X_train.index, columns=X.columns)
X_test_df  = pd.DataFrame(X_test,  index=X_test.index,  columns=X.columns)
y_train_df = pd.Series(le.inverse_transform(y_train), index=X_train.index, name="label")
y_test_df  = pd.Series(le.inverse_transform(y_test),  index=X_test.index,  name="label")

X_train_df.to_csv(os.path.join(OUTPUT_DIR, "X_train.csv"))
X_test_df.to_csv(os.path.join(OUTPUT_DIR,  "X_test.csv"))
y_train_df.to_csv(os.path.join(OUTPUT_DIR, "y_train.csv"))
y_test_df.to_csv(os.path.join(OUTPUT_DIR,  "y_test.csv"))

# ─────────────────────────────────────────────────────────────
# SECTION 3: Define Models
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("SECTION 3: Defining Models")
print("=" * 60)

models = {
    "Logistic Regression": LogisticRegression(
        max_iter     = 1000,
        C            = 1.0,
        solver       = "lbfgs",
        multi_class  = "multinomial",
        random_state = 42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators    = 200,
        max_depth       = 10,
        min_samples_leaf= 5,
        random_state    = 42,
        n_jobs          = -1
    ),
    "SVM": SVC(
        C            = 1.0,
        kernel       = "rbf",
        probability  = True,
        random_state = 42
    )
}

for name in models:
    print(f"  {name}: defined")

# ─────────────────────────────────────────────────────────────
# SECTION 4: 5-Fold Cross-Validation on Training Set
#
# We evaluate each model using stratified 5-fold CV.
# Metrics computed per fold:
#   - Accuracy    : fraction of correct predictions
#   - F1 (macro)  : average F1 across all 3 classes
#                   (macro = equal weight per class, good for balanced data)
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("SECTION 4: 5-Fold Cross-Validation")
print("=" * 60)

cv       = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scorers  = {
    "accuracy": make_scorer(accuracy_score),
    "f1_macro": make_scorer(f1_score, average="macro")
}

cv_results = []

for name, model in models.items():
    print(f"\n  Training: {name}...")
    scores = cross_validate(
        model,
        X_train.values, y_train,
        cv            = cv,
        scoring       = scorers,
        return_train_score = True,
        n_jobs        = -1
    )

    mean_acc   = scores["test_accuracy"].mean()
    std_acc    = scores["test_accuracy"].std()
    mean_f1    = scores["test_f1_macro"].mean()
    std_f1     = scores["test_f1_macro"].std()
    train_acc  = scores["train_accuracy"].mean()

    print(f"    CV Accuracy: {mean_acc:.4f} (+/- {std_acc:.4f})")
    print(f"    CV F1 Macro: {mean_f1:.4f} (+/- {std_f1:.4f})")
    print(f"    Train Acc:   {train_acc:.4f}  "
          f"({'overfit' if train_acc - mean_acc > 0.15 else 'ok'})")

    cv_results.append({
        "model":         name,
        "cv_accuracy":   round(mean_acc, 4),
        "cv_accuracy_std": round(std_acc, 4),
        "cv_f1_macro":   round(mean_f1, 4),
        "cv_f1_std":     round(std_f1, 4),
        "train_accuracy": round(train_acc, 4),
    })

# ─────────────────────────────────────────────────────────────
# SECTION 5: Refit Best Model on Full Training Set
#
# After CV we pick the best model and refit it on ALL
# training data (not just 4/5 of it) before saving.
# This gives the strongest possible model for evaluation.
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("SECTION 5: Refitting All Models on Full Training Set")
print("=" * 60)

trained_models = {}
for name, model in models.items():
    print(f"  Fitting {name} on full training set...")
    model.fit(X_train.values, y_train)
    trained_models[name] = model

    # Save model
    model_path = os.path.join(OUTPUT_DIR, f"{name.replace(' ', '_')}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"    Saved: {model_path}")

# Save label encoder too (needed for evaluation)
with open(os.path.join(OUTPUT_DIR, "label_encoder.pkl"), "wb") as f:
    pickle.dump(le, f)

# ─────────────────────────────────────────────────────────────
# SECTION 6: Save CV Results & Summary
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("SECTION 6: Saving Results")
print("=" * 60)

cv_df = pd.DataFrame(cv_results).sort_values("cv_f1_macro", ascending=False)
cv_df.to_csv(os.path.join(OUTPUT_DIR, "model_cv_results.csv"), index=False)

print(f"  Saved: model_cv_results.csv")
print(f"  Saved: train_test_split.csv")
print(f"  Saved: X_train.csv, X_test.csv, y_train.csv, y_test.csv")
print(f"  Saved: model .pkl files")

print("\n" + "=" * 60)
print("STEP 7 COMPLETE - CROSS-VALIDATION SUMMARY")
print("=" * 60)
print(f"\n  {'Model':<25} {'CV Accuracy':>12} {'CV F1 Macro':>12} {'Train Acc':>10}")
print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*10}")
for _, row in cv_df.iterrows():
    print(f"  {row['model']:<25} "
          f"{row['cv_accuracy']:>10.4f}   "
          f"{row['cv_f1_macro']:>10.4f}   "
          f"{row['train_accuracy']:>10.4f}")

best = cv_df.iloc[0]
print(f"\n  Best model by F1: {best['model']}")
print(f"  CV Accuracy:      {best['cv_accuracy']:.4f} (+/- {best['cv_accuracy_std']:.4f})")
print(f"  CV F1 Macro:      {best['cv_f1_macro']:.4f} (+/- {best['cv_f1_std']:.4f})")
print(f"\nOutputs saved to: {OUTPUT_DIR}/")
print("Next step: Run step8_evaluate.py")