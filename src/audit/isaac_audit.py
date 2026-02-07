"""
ISAAC-SAFE auditing module.

Implements robust, label-balanced ISAAC audits with:
- deterministic mechanistic vs spurious interventions
- site-consistent ΔS measurement
- balanced bootstrap subsampling
- reproducible confidence intervals

This module contains NO model training logic.
"""

import warnings
import random
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from .isaac_metrics import isaac_metrics


# ==================================================
# Constants
# ==================================================

BASE2IDX = {"A": 0, "C": 1, "G": 2, "T": 3}


# ==================================================
# Balanced sampling utilities
# ==================================================

def select_balanced_audit_set(
    sequences: List[str],
    labels: np.ndarray,
    n_samples: int = 20_000,
    seed: int = 42,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """
    Select a class-balanced audit set (50/50 if possible).

    Returns:
        audit_sequences
        audit_labels
        audit_indices (indices into original dataset)
    """
    rng = np.random.RandomState(seed)

    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]

    half = n_samples // 2
    rem = n_samples % 2

    n_pos = min(len(pos_idx), half + rem)
    n_neg = min(len(neg_idx), half)

    if n_pos < half:
        n_neg = min(len(neg_idx), n_samples - n_pos)
    elif n_neg < half:
        n_pos = min(len(pos_idx), n_samples - n_neg)

    pos_sample = rng.choice(pos_idx, size=n_pos, replace=False)
    neg_sample = rng.choice(neg_idx, size=n_neg, replace=False)

    indices = np.concatenate([pos_sample, neg_sample])
    rng.shuffle(indices)

    return (
        [sequences[i] for i in indices],
        labels[indices],
        indices,
    )


def balanced_subsample(
    sequences: List[str],
    labels: np.ndarray,
    subsample_size: int,
    rng: np.random.RandomState,
) -> Tuple[List[str], np.ndarray]:
    """
    Draw a balanced subsample preserving class proportions.
    """
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]

    half = subsample_size // 2
    rem = subsample_size % 2

    n_pos = min(len(pos_idx), half + rem)
    n_neg = min(len(neg_idx), half)

    if n_pos < half:
        n_neg = min(len(neg_idx), subsample_size - n_pos)
    elif n_neg < half:
        n_pos = min(len(pos_idx), subsample_size - n_neg)

    pos_sample = rng.choice(pos_idx, size=n_pos, replace=False)
    neg_sample = rng.choice(neg_idx, size=n_neg, replace=False)

    indices = np.concatenate([pos_sample, neg_sample])
    rng.shuffle(indices)

    return [sequences[i] for i in indices], indices


# ==================================================
# PWM targeting
# ==================================================

def find_best_pwm_site(seq: str, pwm: np.ndarray) -> int:
    """
    Return position with maximal PWM affinity.
    Returns -1 if no valid site exists.
    """
    Lm = pwm.shape[1]
    best_site = -1
    best_aff = -np.inf

    for i in range(len(seq) - Lm + 1):
        aff = 0.0
        for j in range(Lm):
            b = seq[i + j]
            if b in BASE2IDX:
                aff += pwm[BASE2IDX[b], j]
        if aff > best_aff:
            best_aff = aff
            best_site = i

    return best_site


# ==================================================
# Core ISAAC audit (site-consistent ΔS)
# ==================================================

def run_isaac_audit_correct(
    sequences: List[str],
    model_logits_batch,
    mech_interventions,
    spur_interventions,
    pwm: np.ndarray,
    batch_size: int,
    bootstrap_B: int = 200,
    seed: int = 0,
    return_bootstrap: bool = False,
    verbose: bool = False,
) -> Dict:
    """
    Run ISAAC audit measuring ΔS at the SAME targeted PWM site.
    Canonical, correctness-preserving implementation.
    """
    rng = np.random.RandomState(seed)
    random.seed(seed)

    dS_struct = {k: [] for k in ["M1", "M2", "M3"]}
    dS_spur   = {k: [] for k in ["M1", "M2", "M3"]}

    # --------------------------------------------------
    # Precompute target sites
    # --------------------------------------------------
    if verbose:
        print("[ISAAC] Locating PWM target sites...")

    seq_iter = sequences
    if verbose:
        seq_iter = tqdm(sequences, desc="PWM scan", leave=False)

    target_sites = [
        find_best_pwm_site(seq, pwm)
        for seq in seq_iter
    ]

    if verbose:
        n_valid = sum(s != -1 for s in target_sites)
        print(f"[ISAAC] Valid sites for {n_valid}/{len(sequences)} sequences")

    # --------------------------------------------------
    # Batch-wise audit
    # --------------------------------------------------
    batch_range = range(0, len(sequences), batch_size)
    if verbose:
        batch_range = tqdm(batch_range, desc="ISAAC audit", leave=False)

    for i in batch_range:
        batch = sequences[i:i + batch_size]
        sites = target_sites[i:i + batch_size]

        valid = [(s, site) for s, site in zip(batch, sites) if site != -1]
        if not valid:
            continue

        valid_seqs, valid_sites = zip(*valid)

        base_logits = model_logits_batch(list(valid_seqs))

        for k in ["M1", "M2", "M3"]:
            mech_mut = [mech_interventions(seq)[k] for seq in valid_seqs]
            spur_mut = [spur_interventions(seq)[k] for seq in valid_seqs]

            mech_logits = model_logits_batch(mech_mut)
            spur_logits = model_logits_batch(spur_mut)

            dS_struct[k].extend(mech_logits - base_logits)
            dS_spur[k].extend(spur_logits - base_logits)

    # --------------------------------------------------
    # Metric computation + bootstrap
    # --------------------------------------------------
    out = {}

    for k in ["M1", "M2", "M3"]:
        dS_m = np.asarray(dS_struct[k])
        dS_s = np.asarray(dS_spur[k])

        if len(dS_m) < 10:
            warnings.warn(f"[ISAAC] Too few samples for {k} ({len(dS_m)})")
            empty = {
                "entanglement": np.nan,
                "collapse": np.nan,
                "instability": np.nan,
                "entanglement_low": np.nan,
                "entanglement_high": np.nan,
                "collapse_low": np.nan,
                "collapse_high": np.nan,
                "instability_low": np.nan,
                "instability_high": np.nan,
                "ΔS_mech": [],
                "ΔS_spur": [],
                "n_samples": 0,
            }
            out[(k, "mech")] = empty.copy()
            out[(k, "spur")] = empty.copy()
            continue

        metrics = isaac_metrics(dS_m, dS_s)

        res = {
            **metrics,
            "ΔS_mech": dS_m.tolist(),
            "ΔS_spur": dS_s.tolist(),
            "n_samples": len(dS_m),
        }

        if bootstrap_B > 1:
            boot = {m: [] for m in metrics}
            n = len(dS_m)

            for _ in range(bootstrap_B):
                idx = rng.choice(n, n, replace=True)
                bm = isaac_metrics(dS_m[idx], dS_s[idx])
                for m in metrics:
                    boot[m].append(bm[m])

            for m, v in boot.items():
                res[f"{m}_low"]  = float(np.percentile(v, 2.5))
                res[f"{m}_high"] = float(np.percentile(v, 97.5))
                if return_bootstrap:
                    res[f"{m}_bootstrap"] = np.asarray(v)
        else:
            for m in metrics:
                res[f"{m}_low"]  = metrics[m]
                res[f"{m}_high"] = metrics[m]

        out[(k, "mech")] = res.copy()
        out[(k, "spur")] = res.copy()

    return out


# ==================================================
# Robust balanced bootstrap audit
# ==================================================

def isaac_audit(
    sequences: List[str],
    labels: np.ndarray,
    model_logits_batch,
    mech_interventions,
    spur_interventions,
    pwm: np.ndarray,
    total_samples: int = 20_000,
    subsample_size: int = 5_000,
    n_iterations: int = 30,
    batch_size: int = 128,
    seed: int = 42,
    return_distributions: bool = False,
    verbose: bool = False,
) -> Dict:
    """
    Robust ISAAC audit with label-balanced subsampling bootstrap.
    """
    rng = np.random.RandomState(seed)

    if verbose:
        print(
            f"[ROBUST ISAAC] {total_samples} → "
            f"{subsample_size} × {n_iterations} (balanced)"
        )

    if len(sequences) < total_samples:
        warnings.warn("Dataset smaller than requested audit size")
        audit_seqs, audit_labels = sequences, labels
    else:
        audit_seqs, audit_labels, _ = select_balanced_audit_set(
            sequences, labels, total_samples, seed
        )

    results = {
        "entanglement": [],
        "collapse": [],
        "instability": [],
        "entanglement_by_intervention": {k: [] for k in ["M1", "M2", "M3"]},
        "collapse_by_intervention":     {k: [] for k in ["M1", "M2", "M3"]},
        "instability_by_intervention":  {k: [] for k in ["M1", "M2", "M3"]},
    }

    it_range = range(n_iterations)
    if verbose:
        it_range = tqdm(it_range, desc="Balanced bootstrap", leave=False)

    for i in it_range:
        subsample, _ = balanced_subsample(
            audit_seqs, audit_labels, subsample_size, rng
        )

        res = run_isaac_audit_correct(
            subsample,
            model_logits_batch,
            mech_interventions,
            spur_interventions,
            pwm,
            batch_size,
            bootstrap_B=1,
            seed=seed + i,
            verbose=verbose,
        )

        for (k, _), stats in res.items():
            results["entanglement_by_intervention"][k].append(stats["entanglement"])
            results["collapse_by_intervention"][k].append(stats["collapse"])
            results["instability_by_intervention"][k].append(stats["instability"])

            if k == "M2":  # canonical representative
                results["entanglement"].append(stats["entanglement"])
                results["collapse"].append(stats["collapse"])
                results["instability"].append(stats["instability"])

    def summarize(x):
        x = np.asarray(x)
        return {
            "mean": float(np.mean(x)),
            "std": float(np.std(x)),
            "ci_low": float(np.percentile(x, 2.5)),
            "ci_high": float(np.percentile(x, 97.5)),
            "n": len(x),
        }

    final = {
        "entanglement": summarize(results["entanglement"]),
        "collapse":     summarize(results["collapse"]),
        "instability":  summarize(results["instability"]),
        "config": {
            "total_samples": len(audit_seqs),
            "subsample_size": subsample_size,
            "n_iterations": n_iterations,
            "batch_size": batch_size,
        },
    }

    for k in ["M1", "M2", "M3"]:
        final[f"entanglement_{k}"] = summarize(results["entanglement_by_intervention"][k])
        final[f"collapse_{k}"]     = summarize(results["collapse_by_intervention"][k])
        final[f"instability_{k}"]  = summarize(results["instability_by_intervention"][k])

    if return_distributions:
        final["distributions"] = results

    return final


# ==================================================
# Logit wrappers (unchanged semantics)
# ==================================================

def make_deepbind_logits(model, device):
    def encode(seqs):
        L = len(seqs[0])
        X = np.zeros((len(seqs), 4, L), dtype=np.float32)
        for i, s in enumerate(seqs):
            for j, b in enumerate(s):
                X[i, BASE2IDX[b], j] = 1.0
        return torch.tensor(X).to(device)

    def logits_batch(seqs):
        with torch.no_grad():
            return model(encode(seqs)).squeeze(-1).cpu().numpy()

    return logits_batch


make_deepsea_logits = make_bpnet_logits = make_deepbind_logits


def make_dnabert_logits(model, tokenizer_name="zhihan1996/DNA_bert_6", device="cuda"):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, do_lower_case=False)
    model.to(device).eval()

    def logits_batch(seqs):
        enc = tokenizer(
            seqs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            return model(**enc).logits.squeeze(-1).cpu().numpy()

    return logits_batch


# ==================================================
# Raw ΔS persistence
# ==================================================

def save_raw_deltas(dataset, model_name, seed, delta_dict, audit_dir):
    """
    Save raw ΔS values ONLY (re-metricable).
    """
    raw_dir = audit_dir / "raw_deltas"
    raw_dir.mkdir(exist_ok=True)

    path = raw_dir / f"{dataset}_{model_name}_seed{seed}_deltas.npz"

    np.savez_compressed(
        path,
        **delta_dict,
        dataset=dataset,
        model=model_name,
        seed=seed,
        timestamp=pd.Timestamp.now().isoformat(),
    )

    print(f"[ISAAC] Saved raw ΔS → {path.name}")
    return path
