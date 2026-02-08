"""
ISAAC reporting and visualization utilities.

This module provides:
- numerical aggregation utilities (tables, confidence intervals)
- publication-ready tables
- publication-ready plotting (VERTICAL ONLY)

"""

from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
from scipy import stats

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch


# ==================================================
# GLOBAL STYLE (paper-ready, EPS-safe)
# ==================================================

def set_isaac_plot_style():
    """Large-font, EPS-safe plotting style."""
    mpl.rcParams.update({
        "figure.dpi": 100,
        "savefig.dpi": 300,

        "font.family": "sans-serif",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "legend.title_fontsize": 12,

        # EPS / PDF safety
        "ps.fonttype": 3,
        "pdf.fonttype": 42,
        "text.usetex": False,

        "axes.linewidth": 1.2,
        "lines.linewidth": 1.5,
        "patch.linewidth": 1.0,
    })


# ==================================================
# UTILS
# ==================================================

def mean_ci(x, alpha=0.05):
    """
    Mean ± normal-approximation CI.
    Used for AUROC and ISAAC summaries.
    """
    x = np.asarray(x)
    m = x.mean()
    s = x.std(ddof=1) if len(x) > 1 else 0.0
    z = stats.norm.ppf(1 - alpha / 2)
    half = z * s / np.sqrt(len(x)) if len(x) > 1 else 0.0
    return m, s, m - half, m + half


# ==================================================
# TABLE 1 — AUROC SUMMARY
# ==================================================

def make_auroc_table(df_auroc: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (dataset, model), g in df_auroc.groupby(["dataset", "model"]):
        m, s, lo, hi = mean_ci(g["auroc"])
        rows.append({
            "dataset": dataset,
            "model": model,
            "auroc_mean": m,
            "auroc_sd": s,
            "auroc_ci_low": lo,
            "auroc_ci_high": hi,
        })

    return pd.DataFrame(rows)


def aggregate_auroc(df: pd.DataFrame, ci_level: float = 0.95) -> pd.DataFrame:
    records = []

    for (dataset, model), g in df.groupby(["dataset", "model"]):
        values = g["auroc"].values
        n = len(values)

        mean = values.mean()
        sd = values.std(ddof=1)

        alpha = 1 - ci_level
        tval = stats.t.ppf(1 - alpha / 2, df=n - 1)
        half = tval * sd / np.sqrt(n)

        records.append({
            "dataset": dataset,
            "model": model,
            "n_seeds": n,
            "auroc_mean": mean,
            "auroc_sd": sd,
            "ci_low": mean - half,
            "ci_high": mean + half,
        })

    return pd.DataFrame(records)


# ==================================================
# TABLE 2 — ISAAC SUMMARY (per intervention)
# ==================================================

def make_isaac_table_by_intervention(df_results: pd.DataFrame) -> pd.DataFrame:
    metrics = ["entanglement", "collapse", "instability"]
    rows = []

    for (dataset, model, intervention, region), g in df_results.groupby(
        ["dataset", "model", "intervention", "region"]
    ):
        row = {
            "dataset": dataset,
            "model": model,
            "intervention": intervention,
            "region": region,
        }

        for m in metrics:
            mu, sd, lo, hi = mean_ci(g[m])
            row[f"{m}_mean"] = mu
            row[f"{m}_sd"] = sd
            row[f"{m}_ci_low"] = lo
            row[f"{m}_ci_high"] = hi

        rows.append(row)

    return pd.DataFrame(rows)


# ==================================================
# TABLE 3 — ISAAC SUMMARY (aggregated over interventions)
# ==================================================

def make_isaac_table_aggregated(df_results: pd.DataFrame) -> pd.DataFrame:
    metrics = ["entanglement", "collapse", "instability"]

    tmp = (
        df_results
        .groupby(["dataset", "model", "seed"])[metrics]
        .mean()
        .reset_index()
    )

    rows = []

    for (dataset, model), g in tmp.groupby(["dataset", "model"]):
        row = {"dataset": dataset, "model": model}

        for m in metrics:
            mu, sd, lo, hi = mean_ci(g[m])
            row[f"{m}_mean"] = mu
            row[f"{m}_sd"] = sd
            row[f"{m}_ci_low"] = lo
            row[f"{m}_ci_high"] = hi

        rows.append(row)

    return pd.DataFrame(rows)


# ==================================================
# TABLE 4 — INTERVENTION VALIDATION
# ==================================================

def create_validation_table(interv_summary: Dict) -> pd.DataFrame:
    mech = interv_summary["mech"]
    spur = interv_summary["spur"]
    corr = interv_summary["correlations"]

    data = {
        "Metric": [
            "Sample size (n)",
            "Δ binding affinity (mean ± 95% CI)",
            "Relative Δ affinity (mean ± 95% CI)",
            "Intervention effectiveness",
            "Hamming distance from motif (mean ± SD)",
            "New motif sites created",
            "Existing sites affected",
            "Correlation: distance vs Δ affinity",
        ],
        "Mechanistic": [
            mech["n_interventions"],
            f"{mech['mean_delta_affinity']:.2f} "
            f"({mech['ci_delta_affinity'][0]:.2f}, {mech['ci_delta_affinity'][1]:.2f})",
            f"{mech['mean_delta_affinity_rel']:.2f} "
            f"({mech['ci_delta_affinity_rel'][0]:.2f}, {mech['ci_delta_affinity_rel'][1]:.2f})",
            f"{mech['prop_effective_abs']*100:.0f}%",
            "—",
            "—",
            "—",
            f"r = {corr['mech_hamming_vs_delta']:.2f}",
        ],
        "Spurious": [
            spur["n_interventions"],
            f"{spur['mean_delta_affinity']:.2f}",
            "—",
            "—",
            f"{spur['mean_hamming']:.2f}",
            f"{spur['prop_sites_created']*100:.1f}%",
            f"{spur['prop_affinity_changed']*100:.1f}%",
            f"r = {corr['hamming_vs_delta']:.2f}",
        ],
    }

    return pd.DataFrame(data)


# ==================================================
# SAVE FIGURES
# ==================================================

def save_all_formats(fig, basepath):
    basepath = Path(basepath)
    basepath.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(basepath.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.1)
    fig.savefig(basepath.with_suffix(".eps"), format="eps",
                bbox_inches="tight", pad_inches=0.1)
    fig.savefig(basepath.with_suffix(".png"), dpi=300,
                bbox_inches="tight", pad_inches=0.1)


# ==================================================
# PLOT — ISAAC  
# ==================================================

def plot_isaac(
    df_resampling: pd.DataFrame,
    dataset: str | None = None,
    savepath: str | Path | None = None,
):
    """
    Horizontal ISAAC resampling plot (publication figure).
    """
    set_isaac_plot_style()

    df = df_resampling if dataset is None else df_resampling[df_resampling.dataset == dataset]

    INTERVENTION_MAP = {
        "M1": ("PW", "#4C72B0"),
        "M2": ("SS", "#DD8452"),
        "M3": ("CK", "#55A868"),
    }

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
    metrics = ["entanglement", "collapse", "instability"]

    for idx, (ax, metric) in enumerate(zip(axes, metrics)):
        metric_df = df[df["metric"] == metric]

        sns.boxplot(
            data=metric_df,
            x="model",
            y="value",
            hue="intervention",
            ax=ax,
            palette={k: v[1] for k, v in INTERVENTION_MAP.items()},
            width=0.7,
            linewidth=1.5,
            showfliers=True,
        )

        ax.set_title(metric.capitalize(), loc="left", pad=12)
        ax.set_xlabel("")

        if idx == 0:
            ax.set_ylabel("Score")
        else:
            ax.set_ylabel("")

        ax.set_ylim(0.6, 1.0)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

        if ax.get_legend():
            ax.get_legend().remove()

    legend_handles = [
        Patch(facecolor=color, edgecolor="black", label=label)
        for label, color in INTERVENTION_MAP.values()
    ]

    fig.legend(
        handles=legend_handles,
        title="Intervention",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=3,
        frameon=False,
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    if savepath:
        save_all_formats(fig, savepath)

    plt.show()
    return fig, axes



def extract_isaac_results(
    results: Dict,
    dataset: str,
    model: str,
    seed: int,
    auroc: float,
    robust_cfg: Dict,
) -> List[Dict]:
    """
    Flatten ISAAC audit results into row-wise records for CSV / DataFrame.
    """
    rows = []

    for intervention in ["M1", "M2", "M3"]:
        for region in ["mech", "spur"]:

            key = f"entanglement_{intervention}"
            if key not in results:
                continue

            e = results[f"entanglement_{intervention}"]
            c = results[f"collapse_{intervention}"]
            i = results[f"instability_{intervention}"]

            rows.append({
                "dataset": dataset,
                "model": model,
                "seed": seed,
                "intervention": intervention,
                "region": region,

                "entanglement": e["mean"],
                "entanglement_low": e["ci_low"],
                "entanglement_high": e["ci_high"],

                "collapse": c["mean"],
                "collapse_low": c["ci_low"],
                "collapse_high": c["ci_high"],

                "instability": i["mean"],
                "instability_low": i["ci_low"],
                "instability_high": i["ci_high"],

                "auroc": auroc,
                "audit_set_size": results["config"]["total_samples"],
                "subsample_size": robust_cfg["subsample_size"],
                "n_iterations": robust_cfg["n_iterations"],
                "balanced": True,
                "coherent": True,
            })

    return rows


def extract_resampling_distributions(
    results: Dict,
    dataset: str,
    model: str,
    seed: int,
) -> List[Dict]:
    """
    Extract resampling distributions for plotting.
    """
    rows = []

    if "distributions" not in results:
        return rows

    dists = results["distributions"]

    for metric in ["entanglement", "collapse", "instability"]:
        for intervention, values in dists[f"{metric}_by_intervention"].items():
            for v in values:
                rows.append({
                    "dataset": dataset,
                    "model": model,
                    "seed": seed,
                    "intervention": intervention,
                    "metric": metric,
                    "value": v,
                    "balanced": True,
                    "coherent": True,
                })

    return rows
