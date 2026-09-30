"""PREREG S2: at a fixed labelling budget, is one rater per item best? Real raters.

Design (PREREG.md section 5; DEVIATIONS.md D9 fixes the details below before any run)
----------------------------------------------------------------------------------
For every item with at least 10 binary ratings, the raters are split by a seeded
per-item shuffle into

* a POOL of 5 raters, from which an evaluation design draws its labels, and
* a REFERENCE: the other raters (5 for Wikipedia Talk, all remaining for DICES).
  The reference label is their majority (coin flip on ties, seeded), and it
  plays the role of the truth.

Two synthetic systems A and B are drawn per item against the REFERENCE label,
with B's accuracy 0.80, disagreement rate p_disc = 0.20 and a true gap delta.
Two system models:

* ``uniform``    -- the same joint distribution on every item;
* ``contested``  -- both systems err more on items whose reference raters split
  (B accuracy 0.60 there, 0.85 elsewhere, rescaled to 0.80 overall) and A's
  advantage lives on the clear items. This is the case where a better system is
  better on exactly the items humans agree on.

A design spends a budget of B labels as n = B/k items x k pool raters. The gold
label is the majority of those k (coin flip on ties for even k). Outcomes, over
2,000 replicates: power (A ahead with z > 1.96), wrong-direction significance,
and the probability that the observed ranking is wrong (gap <= 0).

The prediction the paper tests
------------------------------
With independent symmetric rater noise, power depends on
sqrt(B/k) * (1 - 2 * eta_k), which is largest at k = 1 for any eta < 0.5. The
real raters are not independent or symmetric, and that is the point of running
this on real data.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from rubricon.gates.attenuation import eta_from_disagreement, eta_majority

from . import assumptions as A

POOL = 5
MIN_RATINGS = 10
ALLOCATIONS = (1, 2, 3, 5)
DELTAS = (0.02, 0.03, 0.05)
P_DISC = 0.20
ACC_B = 0.80
REPS = 2000
Z = 1.959964


def _rng(*keys) -> np.random.Generator:
    return np.random.default_rng([A.SEED_PAPER, *keys])


def build_arrays(df, item_col: str, rater_col: str, label_col: str, ref_size: int | None, tag: int):
    """Pool votes (N, 5), reference labels (N,), reference-split flag (N,)."""
    df = df[[item_col, rater_col, label_col]].dropna()
    df = df.drop_duplicates([item_col, rater_col])
    grp = df.groupby(item_col)
    sizes = grp.size()
    keep = sizes[sizes >= MIN_RATINGS].index
    df = df[df[item_col].isin(keep)].sort_values([item_col, rater_col])
    rng = _rng(tag)
    pool, ref, split = [], [], []
    for _, g in df.groupby(item_col, sort=True):
        v = g[label_col].to_numpy().astype(np.int8)
        v = v[rng.permutation(len(v))]
        p, r = v[:POOL], v[POOL:] if ref_size is None else v[POOL:POOL + ref_size]
        ones = int(r.sum())
        if 2 * ones == len(r):
            lab = int(rng.random() < 0.5)
        else:
            lab = int(2 * ones > len(r))
        pool.append(p)
        ref.append(lab)
        minority = min(ones, len(r) - ones) / len(r)
        split.append(minority >= 0.3)
    return np.array(pool), np.array(ref, dtype=np.int8), np.array(split)


def _system_probs(model: str, delta: float, split: np.ndarray):
    """Per-item (q, b, c): P(both right), P(only A right), P(only B right)."""
    if model == "uniform":
        acc_b = np.full(len(split), ACC_B)
        d = np.full(len(split), delta)
        pd = np.full(len(split), P_DISC)
    else:
        # B: 0.60 on contested items, 0.85 on clear items (overall accuracy then
        # depends on the corpus and is reported). A's whole advantage sits on the
        # clear items. On clear items p_disc is lowered where needed so that A's
        # accuracy stays <= 1 (p_disc <= 0.30 - d); cells with d > 0.15 are skipped.
        frac = split.mean()
        acc_b = np.where(split, 0.60, 0.85)
        d_clear = delta / (1 - frac)
        if d_clear > 0.15:
            return None
        d = np.where(split, 0.0, d_clear)
        pd = np.where(split, P_DISC, min(P_DISC, 0.30 - d_clear))
    b = (pd + d) / 2
    c = (pd - d) / 2
    q = acc_b - c
    if (q < -1e-12).any() or (q + b + c > 1 + 1e-12).any() or (c < -1e-12).any():
        raise ValueError("infeasible system probabilities")
    return q, b, c


def run_design(pool, ref, split, budget: int, k: int, model: str, delta: float, cell: int) -> dict:
    n = budget // k
    n_items = len(ref)
    if n > n_items:
        return None
    rng = _rng(1000 + cell)
    probs = _system_probs(model, delta, split)
    if probs is None:
        return None
    q, b, c = probs
    power = wrong_sig = flip = 0
    gaps, eta_g = [], []
    for _ in range(REPS):
        idx = rng.choice(n_items, size=n, replace=False)
        votes = pool[idx, :k]
        ones = votes.sum(1)
        gold = (2 * ones > k).astype(np.int8)
        ties = 2 * ones == k
        if ties.any():
            gold[ties] = rng.random(ties.sum()) < 0.5
        u = rng.random(n)
        qq, bb, cc = q[idx], b[idx], c[idx]
        a_right = u < qq + bb
        b_right = (u < qq) | ((u >= qq + bb) & (u < qq + bb + cc))
        gold_right = gold == ref[idx]
        d = (a_right == gold_right).astype(float) - (b_right == gold_right).astype(float)
        gap = d.mean()
        se = d.std(ddof=1) / math.sqrt(n)
        z = gap / se if se > 0 else 0.0
        power += z > Z
        wrong_sig += z < -Z
        flip += gap <= 0
        gaps.append(gap)
        eta_g.append(1 - gold_right.mean())
    return {"budget": budget, "k": k, "n_items": n, "model": model, "delta": delta,
            "power": power / REPS, "wrong_direction_significant": wrong_sig / REPS,
            "ranking_wrong": flip / REPS, "mean_gap": float(np.mean(gaps)),
            "mean_gold_error_vs_reference": float(np.mean(eta_g))}


def predicted_power(pool, budget, k, delta):
    """Independent-symmetric-noise prediction from the pool's pairwise disagreement."""
    m = pool.shape[1]
    s = pool.sum(1)
    dis = float((2 * s * (m - s) / (m * (m - 1))).mean())
    eta = eta_from_disagreement(dis)
    att = 1 - 2 * eta_majority(eta, k)
    n = budget // k
    sd = math.sqrt(P_DISC - (att * delta) ** 2)
    zmean = att * delta / (sd / math.sqrt(n))
    from statistics import NormalDist
    return 1 - NormalDist().cdf(Z - zmean), eta


def run(data_dir: Path = Path("data"), out: Path = Path("results/claim4.json")) -> dict:
    import pandas as pd

    corpora = {}
    wk = pd.read_csv(data_dir / "wikitalk" / "toxicity_annotations.tsv", sep="\t",
                     usecols=["rev_id", "worker_id", "toxicity"])
    corpora["wikitalk"] = (build_arrays(wk, "rev_id", "worker_id", "toxicity", 5, 1),
                           (1500, 3000, 6000))
    for tag, t in (("350", 2), ("990", 3)):
        d = pd.read_csv(data_dir / "dices" / f"diverse_safety_adversarial_dialog_{tag}.csv",
                        usecols=["item_id", "rater_id", "Q_overall"])
        d["y"] = (d["Q_overall"] == "Yes").astype(int)
        budgets = (300,) if tag == "350" else (450, 900)
        corpora[f"dices{tag}"] = (build_arrays(d, "item_id", "rater_id", "y", None, t), budgets)

    results = {"design": {"pool": POOL, "min_ratings": MIN_RATINGS, "allocations": ALLOCATIONS,
                          "deltas": DELTAS, "p_disc": P_DISC, "acc_b": ACC_B, "reps": REPS,
                          "seed": A.SEED_PAPER}, "corpora": {}}
    cell = 0
    for name, ((pool, ref, split), budgets) in corpora.items():
        rows = []
        for model in ("uniform", "contested"):
            for budget in budgets:
                for delta in DELTAS:
                    for k in ALLOCATIONS:
                        cell += 1
                        r = run_design(pool, ref, split, budget, k, model, delta, cell)
                        if r is None:
                            continue
                        r["predicted_power_iid"], eta = predicted_power(pool, budget, k, delta)
                        rows.append(r)
        results["corpora"][name] = {"n_items": int(len(ref)), "reference_prevalence": float(ref.mean()),
                                    "contested_fraction": float(split.mean()),
                                    "pool_eta_hat": eta, "rows": rows}
        print(name, len(ref), "items", flush=True)
    out.write_text(json.dumps(results, indent=1, sort_keys=True))
    return results


if __name__ == "__main__":
    run()
