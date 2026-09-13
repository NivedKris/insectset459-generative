"""
Statistical rigor protocol:
1. Two One-Sided Tests (TOST) for Equivalence (Lakens, 2017)
2. Paired Student's t-test with Cohen's d effect size
3. Binomial Sign Test
4. Wilson Score Confidence Intervals for Proportions
5. Deflated z-test (accounting for Design Effect ICC / clustering)
"""
import numpy as np
from scipy import stats
from typing import Dict, Tuple


def tost_paired(treatment: np.ndarray, reference: np.ndarray,
                margin: float = 0.020, alpha: float = 0.05) -> Dict[str, float]:
    """
    Two One-Sided Tests (TOST) for paired data.
    treatment = Filtered Generative, reference = SpecAugment.
    H0: |mu_diff| >= margin (nonequivalent)
    H1: -margin < mu_diff < margin (equivalent)
    """
    diff = np.asarray(treatment) - np.asarray(reference)
    n = len(diff)
    mean_diff = float(np.mean(diff))
    se = float(np.std(diff, ddof=1) / np.sqrt(n))

    # Two one-sided t-statistics
    t1 = (mean_diff - (-margin)) / se  # H0: diff <= -margin
    t2 = (mean_diff - margin) / se     # H0: diff >= margin

    p1 = stats.t.sf(t1, df=n - 1)       # one-sided upper
    p2 = stats.t.cdf(t2, df=n - 1)      # one-sided lower
    p_tost = float(max(p1, p2))

    t_crit_90 = stats.t.ppf(1.0 - alpha, df=n - 1)
    ci_lo_90 = float(mean_diff - t_crit_90 * se)
    ci_hi_90 = float(mean_diff + t_crit_90 * se)

    is_equivalent = bool(ci_lo_90 > -margin and ci_hi_90 < margin and p_tost < alpha)

    return {
        "mean_diff": mean_diff,
        "se": se,
        "t1": float(t1),
        "t2": float(t2),
        "p1": float(p1),
        "p2": float(p2),
        "p_tost": p_tost,
        "ci_lo_90": ci_lo_90,
        "ci_hi_90": ci_hi_90,
        "margin": margin,
        "is_equivalent": is_equivalent
    }


def paired_ttest_and_effect_size(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    """Computes paired t-test, Cohen's d, and 95% CI."""
    diff = np.asarray(a) - np.asarray(b)
    n = len(diff)
    t_stat, p_val = stats.ttest_rel(a, b)
    s_d = np.std(diff, ddof=1)
    d = float(np.mean(diff) / s_d) if s_d > 0 else 0.0
    se = float(s_d / np.sqrt(n))

    t_crit_95 = stats.t.ppf(0.975, df=n - 1)
    ci_lo_95 = float(np.mean(diff) - t_crit_95 * se)
    ci_hi_95 = float(np.mean(diff) + t_crit_95 * se)

    wins = int(np.sum(a > b))
    sign_res = stats.binomtest(wins, n, 0.5)

    return {
        "mean_diff": float(np.mean(diff)),
        "t_stat": float(t_stat),
        "p_value": float(p_val),
        "cohens_d": d,
        "ci_lo_95": ci_lo_95,
        "ci_hi_95": ci_hi_95,
        "sign_wins": wins,
        "sign_n": n,
        "sign_p_value": float(sign_res.pvalue)
    }


def wilson_ci(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float, float]:
    """
    Computes Wilson score interval for binomial proportion k/n.
    Returns: (rate_percent, ci_lower_percent, ci_upper_percent)
    """
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    z = stats.norm.ppf(1.0 - (1.0 - confidence) / 2.0)
    denom = 1.0 + (z ** 2) / n
    center = (p + (z ** 2) / (2.0 * n)) / denom
    spread = (z * np.sqrt((p * (1.0 - p) / n) + (z ** 2) / (4.0 * (n ** 2)))) / denom
    return p * 100.0, max(0.0, (center - spread) * 100.0), min(100.0, (center + spread) * 100.0)


def deflated_z_test(p1: float, n1: int, p2: float, n2: int, deff: float = 6.0) -> Tuple[float, float]:
    """
    Computes difference of proportions z-test deflated by survey design effect (DEFF).
    Used in paper to account for chunk-level intra-recording correlation (ICC rho=0.698).
    """
    # Deflate effective sample sizes
    n1_eff = max(1, n1 / deff)
    n2_eff = max(1, n2 / deff)

    p_pool = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = np.sqrt(p_pool * (1.0 - p_pool) * (1.0 / n1_eff + 1.0 / n2_eff))
    z = (p1 - p2) / se if se > 0 else 0.0
    p_val = float(2.0 * (1.0 - stats.norm.cdf(abs(z))))
    return float(z), p_val
