"""
"Does a stricter fraud rule reduce losses?" as an offline A/B simulation on the held-out test period.

Real A/B tests randomise live traffic. Here every test-period transaction is randomly assigned (stable hash of
its id) to control (the cost-optimal threshold) or treatment (a stricter, lower threshold), and its real outcome
under that arm's rule is costed: fraud value let through + a fixed cost per genuine payment declined. Because the
outcome of either rule is known for every transaction, this measures the policy difference honestly, but it is a
simulation, not a live experiment.

Steps: power analysis (sample size needed for the minimum detectable effect) -> assignment -> Welch t-test and
bootstrap confidence interval on the difference in mean cost per transaction -> decision.
"""
from __future__ import annotations

import hashlib
from typing import Dict

import numpy as np
from scipy import stats

from . import config as C


def sample_size_per_arm(sd: float, mean: float, mde: float = C.AB_MDE, alpha: float = C.AB_ALPHA,
                        power: float = C.AB_POWER):
    """Two-sample, two-sided test of means: n = 2 (z_a/2 + z_b)^2 sd^2 / delta^2 with delta = mde * mean."""
    delta = abs(mde * mean)
    if delta == 0 or not np.isfinite(sd):
        return None                         # control already costs nothing: a relative effect is undefined
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return int(np.ceil(2 * z ** 2 * sd ** 2 / delta ** 2))


def assign(txn_ids, salt: str = "stricter-rule-v1") -> np.ndarray:
    return np.array([int(hashlib.sha256(f"{salt}:{t}".encode()).hexdigest()[:8], 16) % 2 for t in txn_ids])


def cost_per_txn(score, y, amount, thr) -> np.ndarray:
    flag = score >= thr
    return np.where(~flag & (y == 1), amount, 0.0) + np.where(flag & (y == 0), C.FALSE_DECLINE_COST_GBP, 0.0)


def simulate(txn_ids, score, y, amount, thr_control: float, factor: float = C.AB_STRICTER_FACTOR,
             n_boot: int = 2000, seed: int = C.SEED) -> Dict:
    score, y, amount = map(np.asarray, (score, y, amount))
    thr_treat = thr_control * factor
    arm = assign(txn_ids)
    c = cost_per_txn(score[arm == 0], y[arm == 0], amount[arm == 0], thr_control)
    t = cost_per_txn(score[arm == 1], y[arm == 1], amount[arm == 1], thr_treat)
    need = sample_size_per_arm(c.std(ddof=1), c.mean())
    welch = stats.ttest_ind(t, c, equal_var=False)
    rng = np.random.default_rng(seed)
    boot = [rng.choice(t, len(t)).mean() - rng.choice(c, len(c)).mean() for _ in range(n_boot)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    diff = float(t.mean() - c.mean())

    def arm_stats(s, yy, aa, thr):
        flag = s >= thr
        return {"n": int(len(s)), "threshold": round(thr, 4), "mean_cost_gbp": round(float(cost_per_txn(s, yy, aa, thr).mean()), 4),
                "fraud_value_lost": round(float(aa[~flag & (yy == 1)].sum()), 2),
                "false_declines": int((flag & (yy == 0)).sum()), "alerts": int(flag.sum())}
    significant = welch.pvalue < C.AB_ALPHA
    return {
        "design": {"unit": "transaction (stable hash of txn_id)", "metric": "cost per transaction (GBP)",
                   "mde": C.AB_MDE, "alpha": C.AB_ALPHA, "power": C.AB_POWER, "required_n_per_arm": need,
                   "powered": need is not None and bool(min(len(c), len(t)) >= need), "simulation": True},
        "control": arm_stats(score[arm == 0], y[arm == 0], amount[arm == 0], thr_control),
        "treatment": arm_stats(score[arm == 1], y[arm == 1], amount[arm == 1], thr_treat),
        "difference_mean_cost_gbp": round(diff, 4), "relative_change": round(diff / c.mean(), 4) if c.mean() else None,
        "ci95": [round(float(lo), 4), round(float(hi), 4)], "welch_t": round(float(welch.statistic), 3),
        "p_value": float(f"{welch.pvalue:.3g}"),
        "decision": ("ship the stricter rule" if significant and diff < 0 else
                     "keep the current rule (stricter rule costs more)" if significant and diff > 0 else
                     "no significant difference: keep the current rule")}
