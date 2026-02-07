"""
ISAAC-SAFE data pipeline.

This module implements all deterministic data preparation steps used in the paper:
- dataset loading and cleaning
- stratified train/val/test splits
- stratified training subsampling
- construction and caching of regulatory PWM priors (CTCF, JASPAR)

All functions are fully parameterized and reproducible.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

def log(msg: str):
    print(f"[DATA] {msg}")


# ---------------------------------------------------------------------
# Load and clean data
# ---------------------------------------------------------------------

def load_raw_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if not {"sequence", "label"}.issubset(df.columns):
        raise ValueError("CSV must contain columns: ['sequence', 'label']")
    return df


def clean_sequences(df: pd.DataFrame, seq_col: str = "sequence") -> pd.DataFrame:
    """
    Clean and validate DNA sequences.
    - Remove whitespace
    - Uppercase
    - Drop sequences with non-ACGT characters
    - Enforce fixed length
    """
    df = df.copy()

    df[seq_col] = (
        df[seq_col]
        .astype(str)
        .str.replace(" ", "", regex=False)
        .str.replace("\t", "", regex=False)
        .str.upper()
    )

    mask_valid = df[seq_col].str.match("^[ACGT]+$")
    n_invalid = (~mask_valid).sum()

    if n_invalid > 0:
        log(
            f"[WARN] Dropping {n_invalid} invalid sequences "
            f"({n_invalid / len(df):.2e})"
        )
        df = df[mask_valid].reset_index(drop=True)

    lengths = df[seq_col].str.len()
    if lengths.nunique() != 1:
        raise ValueError("Sequences do not have fixed length")

    return df


# ---------------------------------------------------------------------
# Stratified splits
# ---------------------------------------------------------------------

def stratified_split(
    df: pd.DataFrame,
    *,
    test_size: float,
    val_size: float,
    seed: int,
):
    """
    Stratified train / val / test split.
    """
    X = df.index.values
    y = df["label"].values

    idx_trainval, idx_test = train_test_split(
        X,
        test_size=test_size,
        stratify=y,
        random_state=seed,
    )

    y_trainval = y[idx_trainval]
    idx_train, idx_val = train_test_split(
        idx_trainval,
        test_size=val_size,
        stratify=y_trainval,
        random_state=seed,
    )

    return idx_train, idx_val, idx_test


# ---------------------------------------------------------------------
# Save processed data and splits
# ---------------------------------------------------------------------

def save_processed_and_splits(
    df: pd.DataFrame,
    name: str,
    processed_dir: Path,
    splits_dir: Path,
    idx_train,
    idx_val,
    idx_test,
    *,
    seed: int,
    test_size: float,
    val_size: float,
):
    """
    Save full processed dataset and stratified splits.
    """
    processed_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    proc_path = processed_dir / f"{name}_full_sequences.csv"
    df.to_csv(proc_path, index=False)

    split_path = splits_dir / name
    split_path.mkdir(exist_ok=True)

    df.loc[idx_train].to_csv(split_path / "train.csv", index=False)
    df.loc[idx_val].to_csv(split_path / "val.csv", index=False)
    df.loc[idx_test].to_csv(split_path / "test.csv", index=False)

    # ---- metadata for reproducibility ----
    with open(split_path / "split_info.txt", "w") as f:
        f.write(
            f"seed={seed}\n"
            f"test_size={test_size}\n"
            f"val_size={val_size}\n"
            f"n_total={len(df)}\n"
            f"n_train={len(idx_train)}\n"
            f"n_val={len(idx_val)}\n"
            f"n_test={len(idx_test)}\n"
        )

    log(f"Saved processed file: {proc_path}")
    log(f"Saved splits in: {split_path}")


# ---------------------------------------------------------------------
# Stratified training subsampling
# ---------------------------------------------------------------------

def stratified_subsample(
    train_csv: Path,
    out_csv: Path,
    train_n: int,
    seed: int,
):
    """
    Stratified subsampling of training set.
    Preserves label proportions exactly up to rounding.
    """
    df = pd.read_csv(train_csv)

    if len(df) <= train_n:
        df.to_csv(out_csv, index=False)
        return df["label"].value_counts(normalize=True).to_dict()

    label_props = df["label"].value_counts(normalize=True)

    n0 = int(train_n * label_props[0])
    n1 = train_n - n0  # exact total

    df0 = df[df["label"] == 0].sample(n=n0, random_state=seed)
    df1 = df[df["label"] == 1].sample(n=n1, random_state=seed)

    df_sub = (
        pd.concat([df0, df1])
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )

    df_sub.to_csv(out_csv, index=False)

    return label_props.to_dict()


# ---------------------------------------------------------------------
# Full preprocessing pipeline
# ---------------------------------------------------------------------

def process_dataset(
    csv_path: Path,
    processed_dir: Path,
    splits_dir: Path,
    *,
    seed: int,
    test_size: float,
    val_size: float,
):
    """
    Full preprocessing pipeline for one dataset.
    """
    name = csv_path.stem.replace("_random", "")
    log(f"Processing dataset: {name}")

    df = load_raw_dataset(csv_path)
    log(f"N samples (raw): {len(df)}")

    df = clean_sequences(df)
    df = df.rename(columns={"sequence": "sequence_full"})

    log(f"Sequence length: {df['sequence_full'].str.len().iloc[0]}")
    log("Label distribution:")
    log(df["label"].value_counts(normalize=True).to_string())

    idx_train, idx_val, idx_test = stratified_split(
        df,
        test_size=test_size,
        val_size=val_size,
        seed=seed,
    )

    save_processed_and_splits(
        df,
        name,
        processed_dir,
        splits_dir,
        idx_train,
        idx_val,
        idx_test,
        seed=seed,
        test_size=test_size,
        val_size=val_size,
    )

    return df


# ---------------------------------------------------------------------
# Regulatory prior: CTCF PWM (JASPAR)
# ---------------------------------------------------------------------

def load_or_create_ctcf_pwm(
    out_path: Path,
    jaspar_id: str = "MA0139.1",
    pseudocount: float = 0.5,
):
    """
    Load a cached PWM if available, otherwise fetch CTCF PFM from JASPAR,
    apply pseudocount smoothing, convert to PWM, and save.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        return np.load(out_path)

    url = f"https://jaspar.genereg.net/api/v1/matrix/{jaspar_id}/"
    resp = requests.get(url, timeout=30)

    if resp.status_code != 200:
        raise RuntimeError(
            f"Failed to fetch JASPAR matrix {jaspar_id} "
            f"(status {resp.status_code})"
        )

    data = resp.json()
    pfm_dict = data["pfm"]

    pfm = np.array(
        [
            pfm_dict["A"],
            pfm_dict["C"],
            pfm_dict["G"],
            pfm_dict["T"],
        ],
        dtype=float,
    )

    pfm = pfm + pseudocount

    col_sums = pfm.sum(axis=0, keepdims=True)
    if np.any(col_sums == 0):
        raise ValueError("Zero column detected in PFM after smoothing")

    pwm = pfm / col_sums

    if not np.allclose(pwm.sum(axis=0), 1.0):
        raise AssertionError("PWM columns do not sum to 1")

    if not np.all(pwm > 0):
        raise AssertionError("PWM contains non-positive entries")

    np.save(out_path, pwm)
    return pwm
