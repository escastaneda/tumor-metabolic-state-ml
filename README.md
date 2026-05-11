# Predicting Tumor Metabolic State from Multi-Omic Biological Data

**Authors:** Emmily Castaneda & Joshua Streat  
**Course:** ECE / ML Final Project — Virginia Tech  
**Dataset:** TCGA Pan-Cancer Atlas (2018) via UCSC Xena Browser  

---

## Overview

This project develops a supervised machine learning pipeline to classify breast cancer (BRCA) tumor samples into three metabolic states — **Glycolytic**, **Oxidative**, and **Mixed** — using multi-omic molecular data (RNA-seq gene expression + DNA methylation).

**Best model:** Tuned SVM — 82.25% test accuracy, 0.821 F1 macro  
**External validation:** GSE45827 (GEO, n=155) — 80.0% accuracy, 0.806 F1 macro  
**Biological validation:** 56.9% of selected RNA features match known metabolic biomarkers  

---

## Repository Structure

```
├── step2_load_data.py             # Load & match RNA-seq + methylation from TCGA
├── step3_clean_data.py            # Quality control & missing value imputation
├── step4_normalize.py             # Z-score normalization + combined feature matrix
├── step5_assign_labels.py         # Derive metabolic labels via ssGSEA-style scoring
├── step6_feature_selection.py     # Two-stage feature selection (ANOVA + Random Forest)
├── step7_split_and_train.py       # Train/test split + 3-model cross-validation
├── step8_evaluate.py              # Test set evaluation + all figures
├── step9_hyperparameter_tuning.py # GridSearchCV tuning for LR and SVM
├── step10_interpret.py            # Permutation importance + biological interpretation
├── step12_biological_validation.py# Literature cross-reference of model features
├── step13_external_validation.py  # External validation on GSE45827 (GEO)
│
├── step2_output/                  # Matched RNA-seq + methylation matrices
├── step3_output/                  # Cleaned data + QC report
├── step4_output/                  # Normalized features
├── step5_output/                  # Metabolic labels + scores
├── step6_output/                  # Selected 500 features + importance scores
├── step7_output/                  # Train/test splits + trained models
├── step8_output/                  # Evaluation figures (confusion matrices, ROC, etc.)
├── step9_output/                  # Tuned models + tuning figures
├── step10_output/                 # Permutation importance + interpretation figures
├── step12_output/                 # Biological validation figures + report
└── step13_output/                 # External validation figures + report
```

---

## Requirements

```
Python 3.8 (64-bit)
pandas==2.0.3
numpy==1.24.4
scikit-learn
scipy
matplotlib
GEOparse
requests
```

Install all dependencies:
```bash
pip install pandas numpy scikit-learn scipy matplotlib GEOparse requests
```

---

## Data

Data is **not included** in this repository due to file size. Download the following files from [UCSC Xena Browser](https://xenabrowser.net/datapages/?cohort=TCGA%20Pan-Cancer%20(PANCAN)):

| File | Description |
|---|---|
| `EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena.gz` | Batch-normalized RNA-seq (Pan-Cancer) |
| `jhu-usc.edu_PANCAN_HumanMethylation450.betaValue_whitelisted.tsv.synapse_download_5096262.xena.gz` | 450K methylation beta values (Pan-Cancer) |

Place both files in the root project directory before running.

---

## How to Reproduce Results

Run each script in order from your project directory:

```bash
# Step 2: Load and match data
py -3.8-64 step2_load_data.py

# Step 3: Clean and QC
py -3.8-64 step3_clean_data.py

# Step 4: Normalize
py -3.8-64 step4_normalize.py

# Step 5: Assign metabolic labels
py -3.8-64 step5_assign_labels.py

# Step 6: Feature selection
py -3.8-64 step6_feature_selection.py

# Step 7: Train models (5-fold CV)
py -3.8-64 step7_split_and_train.py

# Step 8: Evaluate and generate figures
py -3.8-64 step8_evaluate.py

# Step 9: Hyperparameter tuning
py -3.8-64 step9_hyperparameter_tuning.py

# Step 10: Model interpretation
py -3.8-64 step10_interpret.py

# Step 12: Biological validation
py -3.8-64 step12_biological_validation.py

# Step 13: External validation (downloads GSE45827 from GEO automatically)
py -3.8-64 step13_external_validation.py
```

> **Note:** Steps 3 and 6 are the most time-intensive (~30–60 min each) due to the large methylation file (396,065 CpG sites). All other steps complete in under 10 minutes.

---

## Pipeline Overview

```
TCGA Pan-Cancer Atlas (RNA-seq + Methylation)
                ↓
        Filter to BRCA primary tumors
                ↓
    Quality Control & Normalization
                ↓
    Label Assignment (Glycolytic / Mixed / Oxidative)
    via ssGSEA-style metabolic gene set scoring
                ↓
    Two-Stage Feature Selection
    ANOVA F-test → Random Forest Importance
    335,121 features → 500 features
                ↓
    Train 3 Classifiers (LR, RF, SVM)
    5-fold Stratified Cross-Validation
                ↓
    Hyperparameter Tuning (GridSearchCV)
                ↓
    Best Model: Tuned SVM (linear kernel)
    Test Accuracy: 82.25% | F1 Macro: 0.821
                ↓
    Biological + External Validation
```

---

## Key Results

| Model | CV F1 | Test Accuracy | Test F1 Macro |
|---|---|---|---|
| Logistic Regression (tuned) | 0.803 | 0.779 | 0.781 |
| **SVM (tuned)** | **0.813** | **0.823** | **0.821** |
| Random Forest | 0.662 | 0.693 | 0.688 |
| Random baseline | 0.333 | 0.333 | 0.333 |

**External Validation (GSE45827, GEO):** 80.0% accuracy, 0.806 F1 macro (RNA features only)  
**Biological Validation:** 56.9% of selected RNA features match established metabolic biomarkers  

---

## Output Figures

All figures are saved automatically to their respective `stepN_output/` folders:

| Figure | Location |
|---|---|
| Model comparison (CV vs Test) | `step8_output/model_comparison.png` |
| Confusion matrices (all models) | `step8_output/confusion_matrix_*.png` |
| ROC curves | `step8_output/roc_curves.png` |
| Hyperparameter tuning comparison | `step9_output/tuning_comparison.png` |
| Best model confusion matrix | `step9_output/best_model_confusion_matrix.png` |
| Permutation importance | `step10_output/permutation_importance.png` |
| Gene expression heatmap | `step10_output/rna_class_heatmap.png` |
| Metabolic phenotype space | `step10_output/metabolic_phenotype_space.png` |
| Biomarker validation | `step12_output/biomarker_validation.png` |
| External validation | `step13_output/external_validation.png` |

---

## Citation

If you use this code or pipeline, please cite:

> Castaneda, E. & Streat, J. (2026). Predicting Tumor Metabolic State from Multi-Omic Biological Data Using Machine Learning. Virginia Tech ECE Final Project.

---

## License

This project is for academic purposes. Data sourced from TCGA (open access) via UCSC Xena Browser and GEO (GSE45827, public domain).
