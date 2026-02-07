import numpy as np
from typing import Dict

EPS = 1e-9

# ============================================================
# ISAAC core metrics
# ============================================================

def entanglement(dS_struct: np.ndarray, dS_spur: np.ndarray) -> float:
    """
    ENTANGLEMENT: Empirical distributional overlap (OVL).

    Measures the degree of overlap between the distributions of
    structural and spurious intervention effects.

    """
    if len(dS_struct) < 5 or len(dS_spur) < 5:
        return 0.5  

    # Degenerate case: both distributions nearly constant
    if np.var(dS_struct) < EPS and np.var(dS_spur) < EPS:
        return (
            1.0
            if np.abs(np.mean(dS_struct) - np.mean(dS_spur)) < EPS
            else 0.0
        )

    # Combined support for robust binning
    all_data = np.concatenate([dS_struct, dS_spur])

    # Robust range (outlier-resistant)
    x_min, x_max = np.percentile(all_data, [1, 99])
    if x_max - x_min < EPS:
        x_min -= 1.0
        x_max += 1.0

    # Adaptive bin count
    n_bins = int(np.clip(np.sqrt(len(all_data)), 10, 50))
    bins = np.linspace(x_min, x_max, n_bins + 1)

    hist_struct, _ = np.histogram(dS_struct, bins=bins, density=True)
    hist_spur, _ = np.histogram(dS_spur, bins=bins, density=True)

    bin_width = bins[1] - bins[0]
    ovl = np.sum(np.minimum(hist_struct, hist_spur)) * bin_width

    return float(np.clip(ovl, 0.0, 1.0))


def collapse(dS_struct: np.ndarray, dS_spur: np.ndarray) -> float:
    """
    COLLAPSE: Bounded standardized mean difference.

    Based on Cohen's d, transformed to a bounded score.

    """
    if len(dS_struct) < 3 or len(dS_spur) < 3:
        return 0.5  

    mean_m, mean_s = np.mean(dS_struct), np.mean(dS_spur)
    n_m, n_s = len(dS_struct), len(dS_spur)

    var_m = np.var(dS_struct, ddof=1)
    var_s = np.var(dS_spur, ddof=1)

    pooled_var = ((n_m - 1) * var_m + (n_s - 1) * var_s) / (n_m + n_s - 2)
    pooled_std = np.sqrt(pooled_var + EPS)

    d = np.abs(mean_m - mean_s) / pooled_std

    # Exponential bounding 
    collapse_score = np.exp(-d)

    return float(np.clip(collapse_score, 0.0, 1.0))


def instability(dS_struct: np.ndarray, dS_spur: np.ndarray) -> float:
    """
    INSTABILITY: Relative dispersion of paired intervention effects.

    Computed as a bounded coefficient of variation of ΔS differences.

    """
    if len(dS_struct) < 5 or len(dS_spur) < 5:
        return 0.5  

    delta = dS_struct - dS_spur
    mean_abs = np.mean(np.abs(delta))

    if mean_abs < EPS:
        return 0.0  # perfectly stable differences

    cv = np.std(delta) / mean_abs

    # Bounded CV transform
    instability_score = (cv ** 2) / (1.0 + cv ** 2)

    return float(np.clip(instability_score, 0.0, 1.0))


# ============================================================
# Convenience wrapper
# ============================================================

def isaac_metrics(
    dS_struct: np.ndarray,
    dS_spur: np.ndarray,
    return_raw_deltas: bool = False,
) -> Dict:
    """
    Compute all ISAAC auditing metrics.

    All metrics are bounded in [0, 1] with consistent interpretation:
        0.0 = well-behaved, structurally consistent reasoning
        1.0 = pathological or collapsed reasoning

    Args:
        dS_struct: ΔS values for structural (mechanistic) interventions
        dS_spur: ΔS values for spurious interventions
        return_raw_deltas: whether to include raw ΔS arrays

    Returns:
        Dictionary containing:
            - entanglement
            - collapse
            - instability
            - (optional) raw ΔS arrays and sample size
    """
    result = {
        "entanglement": entanglement(dS_struct, dS_spur),
        "collapse": collapse(dS_struct, dS_spur),
        "instability": instability(dS_struct, dS_spur),
    }

    if return_raw_deltas:
        result.update({
            "ΔS_struct": (
                dS_struct.tolist()
                if hasattr(dS_struct, "tolist")
                else list(dS_struct)
            ),
            "ΔS_spur": (
                dS_spur.tolist()
                if hasattr(dS_spur, "tolist")
                else list(dS_spur)
            ),
            "n_samples": len(dS_struct),
        })

    return result
