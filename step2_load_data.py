"""
Step 2 — Data Collection & Loading
Project: Predict Tumor Metabolic State from Multi-Omic Biological Data
Dataset: TCGA Pan-Cancer Atlas (PANCAN)

This script:
1. Fetches metabolic gene sets from MSigDB API (no browser needed)
2. Loads RNA-seq expression data and filters to BRCA + metabolic genes
3. Loads DNA methylation data and filters to matched BRCA samples
4. Saves clean, matched datasets ready for Step 3 (cleaning & QC)

Files needed in same folder as this script:
    - EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena
    - jhu-usc.edu_PANCAN_HumanMethylation450.betaValue_whitelisted.tsv.synapse_download_5096262.xena
"""

import pandas as pd
import requests
import os
import gzip

# ─────────────────────────────────────────────────────────────
# FILE PATHS — update these if your files are in a different folder
# ─────────────────────────────────────────────────────────────

RNASEQ_FILE      = "EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena.gz"
METHYL_FILE      = "jhu-usc.edu_PANCAN_HumanMethylation450.betaValue_whitelisted.tsv.synapse_download_5096262.xena.gz"
OUTPUT_DIR       = "step2_output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# SECTION 1: Fetch Metabolic Gene Sets from MSigDB API
#
# We pull two Hallmark gene sets:
#   - HALLMARK_GLYCOLYSIS
#   - HALLMARK_OXIDATIVE_PHOSPHORYLATION
#
# These define which genes are metabolically relevant
# and will be used to:
#   (a) filter expression features
#   (b) score each sample to assign metabolic labels
# ─────────────────────────────────────────────────────────────

print("=" * 60)
print("SECTION 1: Fetching MSigDB Metabolic Gene Sets")
print("=" * 60)

MSIGDB_URL = "https://www.gsea-msigdb.org/gsea/msigdb/human/download_geneset.jsp"

def fetch_msigdb_geneset(gene_set_name):
    """Fetch gene set from MSigDB and return list of gene symbols."""
    params = {
        "geneSetName": gene_set_name,
        "fileType": "txt"
    }
    try:
        response = requests.get(MSIGDB_URL, params=params, timeout=30)
        if response.status_code == 200:
            lines = response.text.strip().split("\n")
            # Format: name \t description \t gene1 \t gene2 ...
            genes = lines[0].split("\t")[2:]
            print(f"  {gene_set_name}: {len(genes)} genes fetched")
            return genes
        else:
            print(f"  WARNING: Could not fetch {gene_set_name} (status {response.status_code})")
            return []
    except Exception as e:
        print(f"  WARNING: Request failed for {gene_set_name}: {e}")
        return []

glycolysis_genes   = fetch_msigdb_geneset("HALLMARK_GLYCOLYSIS")
oxphos_genes       = fetch_msigdb_geneset("HALLMARK_OXIDATIVE_PHOSPHORYLATION")

# Fallback: hardcoded core metabolic genes if API is unavailable
# These are the most well-established glycolysis and OXPHOS genes
if not glycolysis_genes:
    print("  Using fallback glycolysis gene list...")
    glycolysis_genes = [
        "HK1","HK2","HK3","GPI","PFKL","PFKM","PFKP","ALDOA","ALDOB","ALDOC",
        "TPI1","GAPDH","PGK1","PGK2","PGAM1","PGAM2","ENO1","ENO2","ENO3",
        "PKM","PKLR","LDHA","LDHB","LDHC","SLC16A1","SLC16A3","PDK1","PDK2",
        "PDK3","PDK4","PDHA1","PDHA2","PDHB","G6PD","PGD","TALDO1","TKT"
    ]

if not oxphos_genes:
    print("  Using fallback OXPHOS gene list...")
    oxphos_genes = [
        "NDUFA1","NDUFA2","NDUFA3","NDUFA4","NDUFA5","NDUFA6","NDUFA7",
        "NDUFB1","NDUFB2","NDUFB3","NDUFB4","NDUFB5","NDUFB6","NDUFB7",
        "SDHB","SDHC","SDHD","UQCRC1","UQCRC2","UQCRB","CYC1",
        "COX4I1","COX5A","COX5B","COX6A1","COX6B1","COX7A1","COX7A2",
        "ATP5F1A","ATP5F1B","ATP5F1C","ATP5F1D","ATP5MC1","ATP5PB",
        "CS","ACO2","IDH2","IDH3A","OGDH","SUCLA2","SDHA","FH","MDH2"
    ]

all_metabolic_genes = list(set(glycolysis_genes + oxphos_genes))
print(f"\n  Total unique metabolic genes: {len(all_metabolic_genes)}")
print(f"  Glycolysis genes:  {len(glycolysis_genes)}")
print(f"  OXPHOS genes:      {len(oxphos_genes)}")

# Save gene lists for reference
pd.DataFrame({"gene": glycolysis_genes, "pathway": "GLYCOLYSIS"}).to_csv(
    os.path.join(OUTPUT_DIR, "glycolysis_genes.csv"), index=False)
pd.DataFrame({"gene": oxphos_genes, "pathway": "OXPHOS"}).to_csv(
    os.path.join(OUTPUT_DIR, "oxphos_genes.csv"), index=False)

# ─────────────────────────────────────────────────────────────
# SECTION 2: Load RNA-seq Data & Filter to BRCA + Metabolic Genes
#
# The PANCAN file contains all 33 cancer types (~11,000 samples)
# BRCA sample barcodes follow format: TCGA-XX-XXXX-01
# where the 2-letter project code for breast cancer is NOT fixed
# — we identify BRCA by the TCGA project code list
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("SECTION 2: Loading RNA-seq Data")
print("=" * 60)

# BRCA patient barcodes start with these known BRCA project prefixes
# TCGA-BRCA uses patient IDs like TCGA-A2, TCGA-BH, TCGA-E2, etc.
# The 4th segment (sample type): 01 = primary tumor, 11 = normal
BRCA_PREFIXES = [
    "TCGA-A1","TCGA-A2","TCGA-A7","TCGA-A8","TCGA-AC","TCGA-AN",
    "TCGA-AO","TCGA-AQ","TCGA-AR","TCGA-B6","TCGA-BH","TCGA-C8",
    "TCGA-D8","TCGA-E2","TCGA-E9","TCGA-EW","TCGA-GI","TCGA-GM",
    "TCGA-HN","TCGA-LD","TCGA-LL","TCGA-LQ","TCGA-MS","TCGA-OL",
    "TCGA-OK","TCGA-PE","TCGA-PL","TCGA-S3","TCGA-UL","TCGA-WT",
    "TCGA-Z7","TCGA-3C","TCGA-4H","TCGA-5L","TCGA-5T"
]

print(f"\nLoading: {RNASEQ_FILE}")
print("(This may take 1-2 minutes for the large Pan-Cancer file...)")

# Detect if file is gzipped
def open_file(filepath):
    if filepath.endswith(".gz"):
        return gzip.open(filepath, "rt")
    return open(filepath, "r")

# Read just the header first to identify BRCA columns
with open_file(RNASEQ_FILE) as f:
    header = f.readline().strip().split("\t")

print(f"  Total samples in Pan-Cancer file: {len(header) - 1}")

# Find BRCA tumor sample columns (ending in -01 = primary tumor)
brca_cols = [
    col for col in header[1:]
    if any(col.startswith(prefix) for prefix in BRCA_PREFIXES)
    and col.endswith("-01")   # primary tumor only
]
print(f"  BRCA primary tumor samples found: {len(brca_cols)}")

# Load full file but keep only BRCA columns + gene column
rna_df = pd.read_csv(
    RNASEQ_FILE,
    sep="\t",
    index_col=0,
    usecols=["sample"] + brca_cols
)

print(f"  Loaded matrix shape: {rna_df.shape[0]} genes × {rna_df.shape[1]} samples")

# Filter to metabolic genes only
# RNA-seq uses gene symbols as index in this file
metabolic_mask = rna_df.index.isin(all_metabolic_genes)
rna_metabolic  = rna_df[metabolic_mask].copy()

print(f"  After filtering to metabolic genes: {rna_metabolic.shape[0]} genes")
print(f"  Genes found: {list(rna_metabolic.index[:10])} ...")

# Standardise sample IDs to 12-char patient barcode TCGA-XX-XXXX
# Full barcode example: TCGA-A2-A0T2-01 → patient: TCGA-A2-A0T2
rna_metabolic.columns = ["-".join(c.split("-")[:3]) for c in rna_metabolic.columns]

# Handle any duplicate patient IDs (keep first)
rna_metabolic = rna_metabolic.T  # now: samples × genes
rna_metabolic = rna_metabolic[~rna_metabolic.index.duplicated(keep="first")]
print(f"  Final RNA-seq: {rna_metabolic.shape[0]} patients × {rna_metabolic.shape[1]} genes")

# ─────────────────────────────────────────────────────────────
# SECTION 3: Load DNA Methylation Data & Filter to BRCA
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("SECTION 3: Loading DNA Methylation Data")
print("=" * 60)

print(f"\nLoading: {METHYL_FILE}")
print("(This file is large — may take 2-4 minutes...)")

# Read header to find BRCA columns
with open_file(METHYL_FILE) as f:
    meth_header = f.readline().strip().split("\t")

print(f"  Total samples in methylation file: {len(meth_header) - 1}")

meth_brca_cols = [
    col for col in meth_header[1:]
    if any(col.startswith(prefix) for prefix in BRCA_PREFIXES)
    and col.endswith("-01")
]
print(f"  BRCA primary tumor samples found: {len(meth_brca_cols)}")

cols_to_keep = [meth_header[0]] + meth_brca_cols
meth_out_temp = os.path.join(OUTPUT_DIR, "brca_methylation_raw.csv")
print(f"  Writing chunks directly to disk (avoids memory limit)...")
first_chunk = True
for i, chunk in enumerate(pd.read_csv(
    METHYL_FILE, sep="	", index_col=0,
    usecols=cols_to_keep, chunksize=2000, engine="python"
)):
    if first_chunk:
        chunk.to_csv(meth_out_temp, mode="w")
        first_chunk = False
    else:
        chunk.to_csv(meth_out_temp, mode="a", header=False)
    if (i + 1) % 25 == 0:
        print(f"    ...processed {(i+1) * 2000:,} CpG sites")

print(f"  All chunks written to disk. Reloading...")
meth_df = pd.read_csv(meth_out_temp, index_col=0)
print(f"  Loaded: {meth_df.shape[0]} CpG sites x {meth_df.shape[1]} samples")

# Standardise to patient barcodes
meth_df.columns = ["-".join(c.split("-")[:3]) for c in meth_df.columns]
meth_df = meth_df.T
meth_df = meth_df[~meth_df.index.duplicated(keep="first")]
print(f"  After dedup: {meth_df.shape[0]} patients × {meth_df.shape[1]} CpG sites")

# ─────────────────────────────────────────────────────────────
# SECTION 4: Match Patients Across Both Datasets
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("SECTION 4: Matching Patients Across Datasets")
print("=" * 60)

rna_patients  = set(rna_metabolic.index)
meth_patients = set(meth_df.index)
matched       = sorted(rna_patients & meth_patients)

print(f"  RNA-seq patients:      {len(rna_patients)}")
print(f"  Methylation patients:  {len(meth_patients)}")
print(f"  Matched patients:      {len(matched)}")
print(f"  RNA-seq only:          {len(rna_patients - meth_patients)}")
print(f"  Methylation only:      {len(meth_patients - rna_patients)}")

# Filter to matched patients
rna_final  = rna_metabolic.loc[matched]
meth_final = meth_df.loc[matched]

print(f"\n  Final RNA-seq matrix:       {rna_final.shape}")
print(f"  Final Methylation matrix:   {meth_final.shape}")

# ─────────────────────────────────────────────────────────────
# SECTION 5: Save Outputs
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("SECTION 5: Saving Outputs")
print("=" * 60)

rna_out   = os.path.join(OUTPUT_DIR, "brca_rnaseq_metabolic.csv")
meth_out  = os.path.join(OUTPUT_DIR, "brca_methylation.csv")
genes_out = os.path.join(OUTPUT_DIR, "metabolic_genes_used.csv")

rna_final.to_csv(rna_out)
meth_final.to_csv(meth_out)
pd.DataFrame({"gene": all_metabolic_genes}).to_csv(genes_out, index=False)

print(f"  Saved: {rna_out}")
print(f"  Saved: {meth_out}")
print(f"  Saved: {genes_out}")

# ─────────────────────────────────────────────────────────────
# SECTION 6: Data Summary Report
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 2 COMPLETE — DATA SUMMARY")
print("=" * 60)
print(f"  Matched BRCA patients:        {len(matched)}")
print(f"  RNA-seq metabolic genes:      {rna_final.shape[1]}")
print(f"  Methylation CpG sites:        {meth_final.shape[1]}")
print(f"  RNA-seq missing values:       {rna_final.isna().sum().sum()}")
print(f"  Methylation missing values:   {meth_final.isna().sum().sum()}")
print(f"\n  Sample patient IDs (first 5):")
for pid in matched[:5]:
    print(f"    {pid}")
print(f"\nOutputs saved to: {OUTPUT_DIR}/")
print("\nNext step: Run step3_clean_data.py")