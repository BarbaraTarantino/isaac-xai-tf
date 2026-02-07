"""
ISAAC-SAFE model factory.

This module provides a single, explicit entry point for constructing
all models used in the paper. No training logic, no side effects.
"""

from typing import Literal
import torch.nn as nn

from src.models.deepbind import DeepBind
from src.models.deepsea import DeepSEA
from src.models.bpnet import BPNetClassifier


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def build_model(
    *,
    model_name: Literal["DeepBind", "DeepSEA", "BPNet"],
    input_length: int,
    num_tasks: int = 1,
    bpnet_filters: int = 64,
    bpnet_dilations: int = 8,
) -> nn.Module:
    """
    Construct a model given its name and required input dimensions.

    Parameters
    ----------
    model_name : {"DeepBind", "DeepSEA", "BPNet"}
        Model architecture identifier.
    input_length : int
        Length of the DNA sequence.
        Ignored for DeepBind (fixed architecture).
    num_tasks : int, default=1
        Number of prediction tasks (binary TF binding = 1).
    bpnet_filters : int, default=64
        Number of convolutional filters for BPNet.
    bpnet_dilations : int, default=8
        Number of dilated convolution blocks for BPNet.

    Returns
    -------
    torch.nn.Module
        Instantiated model, untrained.
    """

    if model_name == "DeepBind":
        _ = input_length  
        return DeepBind(use_hidden=True)

    if model_name == "DeepSEA":
        return DeepSEA(
            input_length=input_length,
            num_tasks=num_tasks,
        )

    if model_name == "BPNet":
        return BPNetClassifier(
            n_filters=bpnet_filters,
            n_dilations=bpnet_dilations,
        )

    raise ValueError(
        f"Unknown model_name='{model_name}'. "
        "Valid options are: DeepBind, DeepSEA, BPNet."
    )
