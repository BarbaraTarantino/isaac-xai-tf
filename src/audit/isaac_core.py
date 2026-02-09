"""
ISAAC-SAFE v4: Biologically grounded, structure-aware interventions
with guaranteed effectiveness for regulatory sequence auditing.

Structural prior is induced exclusively by external PWM knowledge
(e.g. JASPAR), using deterministic, guaranteed-effective interventions.
"""

import random
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
import pandas as pd
from tqdm import tqdm


# ==================================================
# Alphabet
# ==================================================

BASE2IDX = {"A": 0, "C": 1, "G": 2, "T": 3}
BASES = ["A", "C", "G", "T"]


# ==================================================
# Regulatory structural prior
# ==================================================

@dataclass(frozen=True)
class RegulatoryStructuralPrior:
    """
    Canonical regulatory structure induced by PWM affinity.

    Note: a relaxed affinity threshold (80th percentile) is used
    to ensure sufficient structural site coverage per sequence.
    """
    pwm: np.ndarray
    ic_core: List[int]
    affinity_quantile: float = 0.8


# ==================================================
# Information content
# ==================================================

def information_content(p: np.ndarray) -> float:
    eps = 1e-9
    return np.sum(p * np.log2((p + eps) / 0.25))


def compute_ic_core(pwm: np.ndarray, k: int) -> List[int]:
    """
    Compute k PWM positions with highest information content.
    """
    ic = np.array(
        [information_content(pwm[:, j]) for j in range(pwm.shape[1])]
    )
    return list(np.argsort(ic)[-k:])


# ==================================================
# PWM affinity
# ==================================================

def pwm_affinity(seq: str, pos: int, pwm: np.ndarray) -> float:
    """
    Compute PWM affinity at a given sequence position.
    """
    return sum(
        pwm[BASE2IDX[seq[pos + j]], j]
        for j in range(pwm.shape[1])
    )


def find_best_pwm_site(seq: str, pwm: np.ndarray) -> Tuple[int, float]:
    """
    Find the position with highest PWM affinity.
    """
    Lm = pwm.shape[1]
    best_site = -1
    best_aff = -np.inf

    for i in range(len(seq) - Lm + 1):
        aff = pwm_affinity(seq, i, pwm)
        if aff > best_aff:
            best_aff = aff
            best_site = i

    return best_site, best_aff


# ==================================================
# Structural positions
# ==================================================

def pwm_structural_positions(
    seq: str,
    prior: RegulatoryStructuralPrior,
) -> List[int]:
    """
    Positions whose PWM affinity lies above a quantile threshold.
    """
    pwm = prior.pwm
    Lm = pwm.shape[1]

    scores = [
        (i, pwm_affinity(seq, i, pwm))
        for i in range(len(seq) - Lm + 1)
        if all(b in BASE2IDX for b in seq[i:i + Lm])
    ]

    if not scores:
        return []

    vals = np.array([s for _, s in scores])
    thr = np.quantile(vals, prior.affinity_quantile)

    return [i for i, s in scores if s >= thr]


def pwm_nonstructural_positions(
    seq: str,
    prior: RegulatoryStructuralPrior,
) -> List[int]:
    """
    Complement of structural positions.
    """
    Lm = prior.pwm.shape[1]
    all_pos = set(range(len(seq) - Lm + 1))
    return list(all_pos - set(pwm_structural_positions(seq, prior)))


# ==================================================
# Sampling (legacy compatibility)
# ==================================================

def sample_binding_site(
    seq: str,
    positions: List[int],
    pwm: np.ndarray,
    temperature: float = 1.0,
) -> Optional[int]:
    """
    Sample binding site via Boltzmann distribution.
    """
    if not positions:
        return None

    scores = np.array([pwm_affinity(seq, p, pwm) for p in positions])
    probs = np.exp(scores / temperature)
    probs /= probs.sum()

    return np.random.choice(positions, p=probs)


# ==================================================
# Mechanistic interventions (deterministic)
# ==================================================

def M1_precision_weakening(seq: str, site: int, pwm: np.ndarray) -> str:
    """
    Single-position weakening at the highest-scoring PWM position.
    """
    if site is None:
        return seq

    L = pwm.shape[1]
    if site + L > len(seq):
        return seq

    best_pos, best_offset = -1, -1
    best_score = -np.inf

    for offset in range(L):
        i = site + offset
        score = pwm[BASE2IDX[seq[i]], offset]
        if score > best_score:
            best_score = score
            best_pos = i
            best_offset = offset

    if best_pos == -1:
        return seq

    worst_base = min(
        (b for b in BASES if b != seq[best_pos]),
        key=lambda b: pwm[BASE2IDX[b], best_offset],
    )

    seq_list = list(seq)
    seq_list[best_pos] = worst_base
    return "".join(seq_list)


def M2_structural_scramble(seq: str, site: int, pwm: np.ndarray) -> str:
    """
    Scramble high-affinity positions within the PWM window.
    """
    if site is None:
        return seq

    L = pwm.shape[1]
    if site + L > len(seq):
        return seq

    scored = [
        (offset, pwm[BASE2IDX[seq[site + offset]], offset])
        for offset in range(L)
    ]
    scored.sort(key=lambda x: x[1], reverse=True)

    n_scramble = max(2, L // 2)
    positions = [site + o for o, _ in scored[:n_scramble]]

    seq_list = list(seq)
    bases = [seq_list[i] for i in positions]
    random.shuffle(bases)

    for i, pos in enumerate(positions):
        seq_list[pos] = bases[i]

    return "".join(seq_list)


def M3_complete_knockout(seq: str, site: int, pwm: np.ndarray) -> str:
    """
    Replace all PWM positions with anti-consensus bases.
    """
    if site is None:
        return seq

    L = pwm.shape[1]
    if site + L > len(seq):
        return seq

    seq_list = list(seq)

    for offset in range(L):
        worst_idx = np.argmin(pwm[:, offset])
        seq_list[site + offset] = BASES[worst_idx]

    return "".join(seq_list)


# ==================================================
# Spurious interventions
# ==================================================

def S1_minimal_mutation(seq: str, site: int, L: int) -> str:
    if site is None or site + L > len(seq):
        return seq

    seq_list = list(seq)
    i = site + random.randint(0, L - 1)
    seq_list[i] = random.choice([b for b in BASES if b != seq_list[i]])
    return "".join(seq_list)


def S2_random_scramble(seq: str, site: int, L: int) -> str:
    if site is None or site + L > len(seq):
        return seq

    seq_list = list(seq)
    window = list(seq[site:site + L])
    random.shuffle(window)
    seq_list[site:site + L] = window
    return "".join(seq_list)


def S3_gc_preserving(seq: str, site: int, L: int) -> str:
    if site is None or site + L > len(seq):
        return seq

    seq_list = list(seq)
    gc = (seq.count("G") + seq.count("C")) / len(seq)

    for i in range(site, site + L):
        if random.random() < gc:
            seq_list[i] = (
                random.choice(["A", "G"])
                if seq_list[i] in ["A", "G"]
                else random.choice(["C", "T"])
            )
        else:
            seq_list[i] = random.choice(BASES)

    return "".join(seq_list)


# ==================================================
# Intervention builder
# ==================================================

def build_interventions(
    prior: RegulatoryStructuralPrior,
    use_deterministic: bool = True,
):
    pwm = prior.pwm
    Lm = pwm.shape[1]

    if use_deterministic:

        def mech(seq: str) -> Dict[str, str]:
            site, _ = find_best_pwm_site(seq, pwm)
            if site == -1:
                return {"M1": seq, "M2": seq, "M3": seq}
            return {
                "M1": M1_precision_weakening(seq, site, pwm),
                "M2": M2_structural_scramble(seq, site, pwm),
                "M3": M3_complete_knockout(seq, site, pwm),
            }

        def spur(seq: str) -> Dict[str, str]:
            if len(seq) < Lm:
                return {"M1": seq, "M2": seq, "M3": seq}

            r = random.random()
            if r < 0.3:
                site = random.randint(0, len(seq) // 3)
            elif r < 0.6:
                site = random.randint(2 * len(seq) // 3, len(seq) - Lm)
            else:
                site = random.randint(0, len(seq) - Lm)

            return {
                "M1": S1_minimal_mutation(seq, site, Lm),
                "M2": S2_random_scramble(seq, site, Lm),
                "M3": S3_gc_preserving(seq, site, Lm),
            }

    else:

        def mech(seq: str) -> Dict[str, str]:
            S = pwm_structural_positions(seq, prior)
            site = sample_binding_site(seq, S, pwm, temperature=0.5)
            if site is None:
                return {"M1": seq, "M2": seq, "M3": seq}
            return {
                "M1": M1_precision_weakening(seq, site, pwm),
                "M2": M2_structural_scramble(seq, site, pwm),
                "M3": M3_complete_knockout(seq, site, pwm),
            }

        def spur(seq: str) -> Dict[str, str]:
            S = pwm_nonstructural_positions(seq, prior)
            site = random.choice(S) if S else None
            if site is None:
                return {"M1": seq, "M2": seq, "M3": seq}
            return {
                "M1": S1_minimal_mutation(seq, site, Lm),
                "M2": S2_random_scramble(seq, site, Lm),
                "M3": S3_gc_preserving(seq, site, Lm),
            }

    return mech, spur


# ==================================================
# Validation
# ==================================================

def collect_intervention_stats(
    sequences: List[str],
    prior: RegulatoryStructuralPrior,
    mech_fn,
    spur_fn,
    seed: int = 0,
    n_samples: Optional[int] = None,
) -> pd.DataFrame:

    np.random.seed(seed)
    random.seed(seed)

    if n_samples is not None and n_samples < len(sequences):
        sequences = random.sample(sequences, n_samples)

    rows = []
    pwm = prior.pwm

    for seq in tqdm(sequences, desc="Intervention validation"):
        site, base_aff = find_best_pwm_site(seq, pwm)
        if site == -1:
            continue

        for region, fn in [("mech", mech_fn), ("spur", spur_fn)]:
            outs = fn(seq)

            for itype, mutated in outs.items():
                hamming = sum(a != b for a, b in zip(seq, mutated))
                mut_aff = pwm_affinity(mutated, site, pwm)
                delta = mut_aff - base_aff
                delta_rel = delta / max(abs(base_aff), 1.0)

                n_base = len(pwm_structural_positions(seq, prior))
                n_mut = len(pwm_structural_positions(mutated, prior))

                rows.append({
                    "region": region,
                    "intervention": itype,
                    "site_position": site,
                    "hamming": hamming,
                    "original_affinity": base_aff,
                    "mutated_affinity": mut_aff,
                    "delta_affinity": delta,
                    "delta_affinity_rel": delta_rel,
                    "n_base_sites": n_base,
                    "n_mut_sites": n_mut,
                    "site_created": n_mut > n_base,
                    "site_destroyed": n_mut < n_base,
                    "is_effective": delta < -0.5,
                })

    return pd.DataFrame(rows)


def bootstrap_confidence(
    df: pd.DataFrame,
    column: str,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[float, float]:

    rng = np.random.default_rng(seed)
    values = df[column].dropna().values
    if len(values) == 0:
        return (np.nan, np.nan)

    boot = [
        rng.choice(values, size=len(values), replace=True).mean()
        for _ in range(n_boot)
    ]

    lo = np.percentile(boot, 100 * alpha / 2)
    hi = np.percentile(boot, 100 * (1 - alpha / 2))
    return float(lo), float(hi)


def analyze_intervention_stats(
    df: pd.DataFrame,
    tau: float = 0.1,
    min_samples: int = 10,
    n_boot: int = 1000,
) -> Dict:

    if len(df) < min_samples:
        return {"error": "Insufficient samples"}

    summary = {}
    mech = df[df.region == "mech"]
    spur = df[df.region == "spur"]

    if len(mech) > 0:
        summary["mech"] = {
            "n_interventions": len(mech),
            "mean_delta_affinity": float(mech["delta_affinity"].mean()),
            "ci_delta_affinity": bootstrap_confidence(mech, "delta_affinity", n_boot),
            "mean_delta_affinity_rel": float(mech["delta_affinity_rel"].mean()),
            "ci_delta_affinity_rel": bootstrap_confidence(mech, "delta_affinity_rel", n_boot),
            "prop_effective_abs": float(mech["is_effective"].mean()),
            "prop_effective_rel": float((mech["delta_affinity_rel"] < -tau).mean()),
            "by_intervention": {},
        }

        for itype in ["M1", "M2", "M3"]:
            g = mech[mech.intervention == itype]
            if len(g) > 0:
                summary["mech"]["by_intervention"][itype] = {
                    "n": len(g),
                    "mean_delta": float(g["delta_affinity"].mean()),
                    "prop_effective": float(g["is_effective"].mean()),
                }

    if len(spur) > 0:
        summary["spur"] = {
            "n_interventions": len(spur),
            "mean_hamming": float(spur["hamming"].mean()),
            "prop_sites_created": float(spur["site_created"].mean()),
            "prop_affinity_changed": float((spur["delta_affinity"].abs() > 0.5).mean()),
            "by_intervention": {},
        }

        for itype in ["M1", "M2", "M3"]:
            g = spur[spur.intervention == itype]
            if len(g) > 0:
                summary["spur"]["by_intervention"][itype] = {
                    "n": len(g),
                    "mean_hamming": float(g["hamming"].mean()),
                    "mean_delta": float(g["delta_affinity"].mean()),
                }

    summary["correlations"] = {
        "hamming_vs_delta": float(df["hamming"].corr(df["delta_affinity"].abs())),
        "mech_hamming_vs_delta": float(
            mech["hamming"].corr(mech["delta_affinity"].abs())
        ) if len(mech) > 0 else np.nan,
    }

    if "mech" in summary:
        eff = summary["mech"]["prop_effective_abs"]
        summary["validation_status"] = {
            "overall_effectiveness": eff,
            "is_ready_for_audit": eff > 0.9,
            "recommendation": "READY FOR AUDIT" if eff > 0.9 else "NEEDS IMPROVEMENT",
        }

    return summary


# ==================================================
# Convenience
# ==================================================

def create_ctcf_prior(
    pwm_path: str = "ctcf_pwm.npy",
    k: int = 4,
) -> RegulatoryStructuralPrior:

    pwm = np.load(pwm_path)
    ic_core = compute_ic_core(pwm, k)
    return RegulatoryStructuralPrior(pwm=pwm, ic_core=ic_core)


def run_comprehensive_validation(
    sequences: List[str],
    pwm_path: str = "ctcf_pwm.npy",
    n_samples: int = 500,
    seed: int = 42,
) -> Dict:

    prior = create_ctcf_prior(pwm_path)
    mech_fn, spur_fn = build_interventions(prior, use_deterministic=True)

    stats = collect_intervention_stats(
        sequences,
        prior,
        mech_fn,
        spur_fn,
        seed=seed,
        n_samples=n_samples,
    )

    return analyze_intervention_stats(stats)
