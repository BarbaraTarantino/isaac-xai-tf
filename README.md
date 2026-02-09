# ISAAC: Intervention-Based Structural Auditing

**Code repository for:** "ISAAC: Intervention-Based Structural Auditing of Deep Models for Transcription Factor Binding"

This repository contains the code and notebooks implementing the ISAAC framework for structurally auditing deep learning models in regulatory genomics. The experiments use transcription factor (TF) binding prediction as a biological case study.

## Data Availability

Raw transcription factor binding datasets used in this study are publicly
available on the Open Science Framework (OSF):

https://osf.io/k73td/

All processed data, deterministic train/validation/test splits, and
subsampled training sets used in the paper are fully reproducible by the code provided in this repository (`0-isaac-load.ipynb`).

## Quick Start

1. Download the raw datasets from the OSF link above and place them in
   `data/raw/`

2. Run the notebooks in numerical order:
   - `0-isaac-load.ipynb` – Data preparation and structural prior construction  
   - `1-isaac-train.ipynb` – Training procedure  
   - `2-isaac-audit.ipynb` – ISAAC audit and result generation

## Contents

- **Data preparation and PWM construction** - Loading ENCODE datasets and constructing position weight matrix (PWM) priors
- **Model training procedure** - Reproducible training scripts for DeepBind, DeepSEA, and BPNet
- **ISAAC auditing framework** - Robust intervention-based auditing with balanced subsampling
- **Result generation** - Scripts to produce paper-ready tables and figures

## Reproducibility

All experiments are fully reproducible:
- Fixed random seeds throughout all experiments
- Audits performed on frozen model checkpoints
- Raw intervention effects are saved to disk
- All reported results, tables, and figures are generated directly from the notebooks

## Models & Datasets

- **Models:** DeepBind, DeepSEA, BPNet
- **Datasets:** ENCODE TF ChIP-seq (A549, GM12878, HepG2 cell lines)
- **Metrics:** AUROC for predictive performance; Entanglement, Collapse, Instability for structural auditing
