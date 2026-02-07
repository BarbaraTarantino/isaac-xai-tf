"""
ISAAC-SAFE training runner.

This module implements the deterministic training procedure used in the paper.

Responsibilities:
- load training data
- instantiate model via factory
- run fixed-epoch training
- freeze and save final model
- log minimal training metadata

The output artifacts are:
- model.pt
- metrics.json
"""

from pathlib import Path
import json
import torch
import torch.nn as nn
import pandas as pd

from src.training.datasets import DNACNNDataset
from src.training.utils import set_seed, make_loader
from src.training.loops import train_epoch_cnn, freeze_model
from src.training.model_factory import build_model


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

def log(msg: str):
    print(f"[TRAIN] {msg}")


# ---------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------

def train_model(
    *,
    dataset: str,
    model_name: str,
    train_csv: Path,
    out_dir: Path,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
):
    """
    Train a single model on a fixed dataset split and seed.

    This function is fully deterministic given its inputs.
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "model.pt"
    metrics_path = out_dir / "metrics.json"

    # --------------------------------------------------
    # Guard: do not retrain if artifact exists
    # --------------------------------------------------
    if model_path.exists():
        log(
            f"{dataset} | {model_name} | seed={seed} "
            f"[SKIPPED: model already trained]"
        )
        return

    # --------------------------------------------------
    # Reproducibility
    # --------------------------------------------------
    set_seed(seed)

    device = torch.device(device)
    log(f"Device: {device}")

    # --------------------------------------------------
    # Load data
    # --------------------------------------------------
    train_df = pd.read_csv(train_csv)

    if "sequence_full" not in train_df.columns:
        raise ValueError("Training CSV must contain 'sequence_full' column")

    log(f"Dataset: {dataset}")
    log(f"Training samples: {len(train_df)}")

    # --------------------------------------------------
    # Dataset & loader
    # --------------------------------------------------
    ds = DNACNNDataset(train_df, view="sequence_full")
    loader = make_loader(ds, batch_size)

    # --------------------------------------------------
    # Model construction
    # --------------------------------------------------
    input_len = len(train_df["sequence_full"].iloc[0])

    model = build_model(
        model_name=model_name,
        input_length=input_len,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    # --------------------------------------------------
    # Training loop
    # --------------------------------------------------
    log(
        f"Training {model_name} | epochs={epochs} | "
        f"batch={batch_size} | lr={lr}"
    )

    losses = []

    for ep in range(epochs):
        loss = train_epoch_cnn(
            model=model,
            loader=loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )
        losses.append(float(loss))

        log(
            f"{model_name} | epoch {ep+1}/{epochs} "
            f"| train loss={loss:.4f}"
        )

    # --------------------------------------------------
    # Freeze & save model
    # --------------------------------------------------
    freeze_model(model)
    torch.save(model.state_dict(), model_path)

    # --------------------------------------------------
    # Save metrics
    # --------------------------------------------------
    metrics = {
        "dataset": dataset,
        "model": model_name,
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "train_file": Path(train_csv).name,
        "train_size": len(train_df),
        "final_train_loss": losses[-1],
        "train_loss_curve": losses,
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    log(f"Saved model to: {model_path}")
    log(f"Saved metrics to: {metrics_path}")
