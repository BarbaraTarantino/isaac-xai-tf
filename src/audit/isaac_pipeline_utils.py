"""
ISAAC audit pipeline utilities.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import torch

from src.models.deepbind import DeepBind
from src.models.deepsea import DeepSEA
from src.models.bpnet import BPNetClassifier

from src.audit.isaac_audit import (
    select_balanced_audit_set,
    isaac_audit,
    make_deepbind_logits,
    make_deepsea_logits,
    make_bpnet_logits,
)

# ==================================================
# DEVICE
# ==================================================

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==================================================
# MODEL LOADING
# ==================================================

def load_model_and_logits(
    model_name: str,
    model_path: Path,
    seq_length: int,
    device,
):
    """
    Load trained model and return logits function.
    """
    if model_name == "DeepBind":
        model = DeepBind(use_hidden=True).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return make_deepbind_logits(model, device)

    if model_name == "DeepSEA":
        model = DeepSEA(input_length=seq_length, num_tasks=1).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return make_deepsea_logits(model, device)

    if model_name == "BPNet":
        model = BPNetClassifier().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return make_bpnet_logits(model, device)

    raise ValueError(f"Unknown model: {model_name}")


# ==================================================
# AUROC COMPUTATION
# ==================================================

def compute_auroc(
    sequences,
    labels,
    logits_fn,
    batch_size: int,
):
    """
    Compute AUROC on a fixed sequence set.
    """
    preds = []
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i : i + batch_size]
        preds.extend(logits_fn(batch))

    preds = np.asarray(preds)
    return roc_auc_score(labels, preds)


# ==================================================
# AUROC FILTERING
# ==================================================

def filter_models_by_auroc(
    model_aurocs: dict,
    threshold: float,
    mode: str = "mean",
):
    """
    Decide which models pass AUROC filtering.
    """
    valid = []

    for model, seed_dict in model_aurocs.items():
        if not seed_dict:
            continue

        values = np.array(list(seed_dict.values()))

        if mode == "mean":
            if values.mean() >= threshold:
                valid.append(model)

        elif mode == "per-seed":
            if np.all(values >= threshold):
                valid.append(model)

        else:
            raise ValueError(f"Unknown filter mode: {mode}")

    return valid


# ==================================================
# BALANCED AUDIT SET CREATION
# ==================================================

def create_and_save_audit_set(
    sequences,
    labels,
    n_samples,
    seed,
    out_path: Path,
):
    """
    Create balanced audit set and save metadata.
    """
    audit_seqs, audit_labels, audit_idx = select_balanced_audit_set(
        sequences,
        labels,
        n_samples=n_samples,
        seed=seed,
    )

    meta = pd.DataFrame({
        "original_index": audit_idx,
        "sequence": audit_seqs,
        "label": audit_labels,
    })
    meta.to_csv(out_path, index=False)

    return audit_seqs, audit_labels
