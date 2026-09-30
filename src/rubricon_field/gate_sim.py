"""Simulation with known ground truth: how does the claim gate behave? (PREREG S0/S1)

Generative model
----------------
* ``n`` items, binary truth ``y`` with prevalence ``pi``.
* Two systems A and B. Their per-item correctness against the TRUTH is drawn
  jointly: P(A right, B wrong) = b, P(A wrong, B right) = c, with b + c = p_disc
  (their disagreement rate) and b - c = delta (the true accuracy gap A - B).
  B's accuracy is fixed at 0.75.
* ``K`` raters (3) label every item; each flips the truth independently with the
  item's error rate ``eta_i``. The gold label is their majority.
* The analyst sees only the gold labels and the rater labels: accuracies are
  scored against gold, alpha is estimated from the rater labels.

Two scenarios: ``iid`` (one eta for every item) and ``het`` (hard items have a
higher eta and the systems' gap differs by item type; see DEVIATIONS.md D4).

An exact result the simulation checks (S0)
------------------------------------------
For binary labels, A agrees with gold iff (A correct) == (gold correct), so the
two systems' observed correctness differs exactly when their true correctness
differs. Label noise therefore leaves the disagreement rate unchanged and scales
the expected accuracy gap by ``1 - 2 * eta_K`` where ``eta_K`` is the probability
that the gold label is wrong. That replaces the earlier ``1 / sqrt(rho)``
heuristic; the two coincide only when prevalence is 0.5 and K = 1.

The gate under test is the shipped ``rubricon.gates.signal.SignalGate``; this
module only prepares its inputs.
"""

from __future__ import annotations

import itertools
import json
import math
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from rubricon.core.schema import Verdict
from rubricon.gates.signal import GatePolicy, SignalGate

from . import assumptions as A

Z_A = 1.959964   # two-sided 5%
Z_B = 0.841621   # 80% power
ACC_B = 0.75
K_RATERS = 3
ALPHA_FLOOR_SWEEP = (0.0, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.667)
Z_THRESHOLDS = (1.96, 2.24, 2.58, 2.8, 3.0, 3.29, 3.5, 4.0, 4.5)

CHECK_ORDER = ("reliability", "precision", "effect_vs_mde", "interval_excludes_zero", "replication")


# --------------------------------------------------------------------------
# small exact helpers
# --------------------------------------------------------------------------


def eta_majority(eta: float, k: int = K_RATERS) -> float:
    """P(majority of k independent raters, each wrong w.p. eta, is wrong); k odd."""
    return sum(math.comb(k, j) * eta**j * (1 - eta) ** (k - j) for j in range(k // 2 + 1, k + 1))


def population_alpha(etas_w: list[tuple[float, float]], pi: float) -> float:
    """Large-sample nominal alpha for symmetric noise; etas_w = [(weight, eta), ...]."""
    d_o = sum(w * 2 * e * (1 - e) for w, e in etas_w)
    eta_bar = sum(w * e for w, e in etas_w)
    p = pi * (1 - eta_bar) + (1 - pi) * eta_bar
    return 1.0 - d_o / (2 * p * (1 - p))


def solve_etas(alpha: float, pi: float, scenario: str, h: float = 0.3, mult: float = 3.0):
    """Item error rates (easy, hard) giving the target population alpha, or None."""
    def alpha_of(s: float) -> float:
        if scenario == "iid":
            return population_alpha([(1.0, s)], pi)
        return population_alpha([(1 - h, s), (h, mult * s)], pi)

    hi = 0.5 if scenario == "iid" else 0.5 / mult
    if alpha_of(hi) > alpha or alpha_of(1e-9) < alpha:
        return None
    lo = 1e-9
    for _ in range(80):
        mid = (lo + hi) / 2
        if alpha_of(mid) > alpha:
            lo = mid
        else:
            hi = mid
    s = (lo + hi) / 2
    return (s, s) if scenario == "iid" else (s, mult * s)


def alpha_binary_fast(s: np.ndarray, k: int) -> np.ndarray:
    """Krippendorff's nominal alpha for binary data with k raters on every unit.

    ``s`` has shape (..., n_units) and holds the number of 1-votes per unit. This is an
    independent vectorised implementation, cross-checked against
    ``rubricon.stats.agreement.krippendorff_alpha`` in the tests.
    """
    n_units = s.shape[-1]
    big_n = n_units * k
    n1 = s.sum(-1)
    n0 = big_n - n1
    d_o = (2.0 * (s * (k - s)).sum(-1) / (k - 1)) / big_n
    d_e = 2.0 * n1 * n0 / (big_n * (big_n - 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        return 1.0 - d_o / d_e


# --------------------------------------------------------------------------
# one cell of the grid
# --------------------------------------------------------------------------


def _cell_probs(delta: float, p_disc: float):
    b = (p_disc + delta) / 2
    c = (p_disc - delta) / 2
    if b < 0 or c < 0 or ACC_B - c < 0 or ACC_B + b > 1:
        return None
    return ACC_B - c, b, c   # q (both right), b (A only), c (B only)


def simulate_cell(scenario: str, n: int, alpha: float, pi: float, delta: float,
                  p_disc: float, reps: int, cell_id: int,
                  h: float = 0.3, mult: float = 3.0, kappa: float = 0.05, debug: bool = False) -> dict | None:
    etas = solve_etas(alpha, pi, scenario, h, mult)
    if etas is None:
        return None
    rng = np.random.default_rng([A.SEED_PAPER, cell_id])
    e_easy, e_hard = etas

    if scenario == "iid":
        hard = np.zeros((reps, n), dtype=bool)
        deltas = {False: delta, True: delta}
    else:
        hard = rng.random((reps, n)) < h
        deltas = {False: delta - kappa * h / (1 - h), True: delta + kappa}
    probs = {t: _cell_probs(deltas[t], p_disc) for t in (False, True)}
    if probs[False] is None or (scenario == "het" and probs[True] is None):
        return None

    u = rng.random((reps, n))
    qe, be, ce = probs[False]
    qh, bh, ch = probs[True]
    q = np.where(hard, qh, qe)
    b = np.where(hard, bh, be)
    c = np.where(hard, ch, ce)
    ca = u < q + b
    cb = (u < q) | ((u >= q + b) & (u < q + b + c))

    eta_i = np.where(hard, e_hard, e_easy)
    y = rng.random((reps, n)) < pi
    flips = rng.random((reps, n, K_RATERS)) < eta_i[..., None]
    votes = y[..., None] ^ flips
    s = votes.sum(-1)
    gold = s >= (K_RATERS // 2 + 1)
    gc = gold == y
    oa = (ca == gc).astype(np.float64)
    ob = (cb == gc).astype(np.float64)

    d = oa - ob
    gap = d.mean(-1)
    sd = d.std(-1, ddof=1)
    se = sd / math.sqrt(n)
    a_hat = alpha_binary_fast(s, K_RATERS)

    claim = gap > 0
    zstat = gap / np.where(se > 0, se, np.inf)
    naive = claim & (zstat > Z_A)
    z_counts = {str(t): int((claim & (zstat > t)).sum()) for t in Z_THRESHOLDS}
    mde = (Z_A + Z_B) * sd / math.sqrt(n)

    # ---- gate v2 statistics (DEVIATIONS.md D6) -------------------------------
    unan = (s == 0) | (s == K_RATERS)
    n_u = unan.sum(-1).astype(np.float64)
    n_s = n - n_u

    def _mean_var(mask, cnt):
        with np.errstate(divide="ignore", invalid="ignore"):
            m = (d * mask).sum(-1) / cnt
            v = (((d - m[..., None]) ** 2) * mask).sum(-1) / (cnt - 1)
        return m, v / cnt

    g_u, v_u = _mean_var(unan, n_u)
    g_s, v_s = _mean_var(~unan, n_s)
    with np.errstate(divide="ignore", invalid="ignore"):
        z_het = (g_u - g_s) / np.sqrt(v_u + v_s)
    opp_sign = (g_u * g_s) < 0
    c3_block = opp_sign & (np.abs(z_het) > 2.58)
    c3_warn = (np.abs(z_het) > 1.96) & ~c3_block
    v2_pub = claim & (zstat > 2.58) & ~c3_block

    dis = ((2.0 * s * (K_RATERS - s) / (K_RATERS * (K_RATERS - 1))).mean(-1))
    eta_hat = (1 - np.sqrt(np.clip(1 - 2 * dis, 0, 1))) / 2
    eta_k_hat = 3 * eta_hat**2 * (1 - eta_hat) + eta_hat**3
    a_hat_att = 1 - 2 * eta_k_hat
    with np.errstate(divide="ignore", invalid="ignore"):
        gap_true_est = gap / a_hat_att

    lo_ci = gap - Z_A * se
    v1_sweep = {f"{f:.2f}": int((claim & (a_hat >= f) & (np.abs(gap) >= mde) & (lo_ci > 0)).sum())
                for f in ALPHA_FLOOR_SWEEP}
    if debug:
        return {"v1_sweep": v1_sweep, "d": d, "unan": unan, "gap": gap, "sd": sd, "zstat": zstat, "claim": claim,
                "v2_pub": v2_pub, "dis": dis, "eta_hat": eta_hat, "eta_k_hat": eta_k_hat}

    gate = SignalGate(GatePolicy(), depth="production")
    verdicts = Counter()
    masks = Counter()
    sum_gap_gate = 0.0
    n_gate_pub = 0
    for i in np.flatnonzero(claim):
        lo, hi = gap[i] - Z_A * se[i], gap[i] + Z_A * se[i]
        checks = [
            gate.check_reliability(float(a_hat[i])),
            gate.check_precision(float(Z_A * se[i])),
            gate.check_effect_vs_mde(float(gap[i]), float(mde[i])),
            gate.check_interval_excludes_zero(float(lo), float(hi)),
            gate.check_replication(float(K_RATERS)),
        ]
        vs = [c.verdict for c in checks]
        mask = sum(1 << j for j, v in enumerate(vs) if v is Verdict.BLOCK)
        masks[mask] += 1
        v = "block" if mask else ("warn" if any(x is Verdict.WARN for x in vs) else "pass")
        verdicts[v] += 1
        if v != "block":
            n_gate_pub += 1
            sum_gap_gate += float(gap[i])

    return {
        "scenario": scenario, "n": n, "alpha_target": alpha, "pi": pi, "delta": delta,
        "p_disc": p_disc, "reps": reps, "eta_easy": e_easy, "eta_hard": e_hard,
        "n_claims": int(claim.sum()),
        "n_naive_pub": int(naive.sum()),
        "z_threshold_counts": z_counts,
        "n_gate_pub": n_gate_pub,
        "verdicts": dict(verdicts),
        "block_masks": {str(k): v for k, v in masks.items()},
        "sum_gap_claims": float(gap[claim].sum()),
        "sum_gap_naive": float(gap[naive].sum()),
        "sum_gap_gate": sum_gap_gate,
        "mean_alpha_hat": float(np.nanmean(a_hat)),
        "v2_n_pub": int(v2_pub.sum()),
        "v1_alpha_floor_sweep": v1_sweep,
        "v2_n_z_only": int((claim & (zstat > 2.58)).sum()),
        "v2_n_c3_block": int((claim & (zstat > 2.58) & c3_block).sum()),
        "v2_n_c3_warn_among_pub": int((v2_pub & c3_warn).sum()),
        "v2_sum_gap_pub": float(gap[v2_pub].sum()),
        "mean_attenuation_est": float(np.nanmean(a_hat_att)),
        "mean_gap_true_est": float(np.nanmean(gap_true_est)),
        "mean_gap_all": float(gap.mean()),
        "mean_disc_obs": float((d != 0).mean()),
    }


# --------------------------------------------------------------------------
# grid runner
# --------------------------------------------------------------------------

GRID = {
    "pi": (0.1, 0.3, 0.5),
    "alpha": (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    "n": (200, 500, 1000, 2000, 5000),
    "delta": (-0.03, 0.0, 0.01, 0.02, 0.03, 0.05),
    "p_disc": (0.1, 0.2, 0.3),
}
HET_ALPHAS = (0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def _work(args):
    return simulate_cell(*args)


def run_grid(scenario: str, out: Path, reps: int = 1000, workers: int = 12, kappa: float = 0.05,
             id_offset: int = 0) -> int:
    alphas = GRID["alpha"] if scenario == "iid" else HET_ALPHAS
    cells = list(itertools.product(GRID["pi"], alphas, GRID["n"], GRID["delta"], GRID["p_disc"]))
    jobs = [(scenario, n, a, pi, d, pd, reps, i + id_offset, 0.3, 3.0, kappa)
            for i, (pi, a, n, d, pd) in enumerate(cells)]
    out.parent.mkdir(parents=True, exist_ok=True)
    done = skipped = 0
    with ProcessPoolExecutor(max_workers=workers) as ex, out.open("w") as f:
        for res in ex.map(_work, jobs, chunksize=4):
            if res is None:
                skipped += 1
                continue
            f.write(json.dumps(res, sort_keys=True) + "\n")
            done += 1
    print(f"{scenario}: {done} cells written, {skipped} infeasible, reps={reps}")
    return done


if __name__ == "__main__":
    import sys
    scen = sys.argv[1] if len(sys.argv) > 1 else "iid"
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    run_grid(scen, Path("results") / f"gate_sim_{scen}.jsonl", reps=reps)
    # post-hoc scenarios (DEVIATIONS.md D6): gap smaller on contested items
    if scen == "het":
        for j, k in enumerate((-0.05, -0.10), start=1):
            run_grid("het", Path("results") / f"gate_sim_het_k{k:+.2f}.jsonl", reps=reps,
                     kappa=k, id_offset=100_000 * j)
