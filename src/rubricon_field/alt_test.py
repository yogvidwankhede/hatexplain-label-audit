"""Alternative Annotator Test (Calderon, Reichart & Dror, ACL 2025), implemented from the paper.

Procedure (paper section 3; parameters fixed in PREREG_ADDENDUM.md before this ran):
for each annotator j with at least ``MIN_ITEMS`` items in the sample,
  S(f, x_i, j) = mean over the item's OTHER annotators k of 1{f(x_i) = h_k(x_i)}
  S(h_j, x_i, j) likewise with f replaced by j's own label;
  W^f = 1{S_f >= S_h}, W^h = 1{S_h >= S_f} (ties count for both);
  d_i = W^h - W^f;  t_j = (mean(d) - eps) / (sd(d) / sqrt(n_j));  lower-tail p-value;
Benjamini-Yekutieli at q = 0.05 over the annotators of one corpus;
winning rate omega = share of annotators whose H0 is rejected; pass iff omega >= 0.5;
average advantage probability rho = mean_j mean_i W^f.

Labels are the binarised labels (the primary scale of this study). Items where the
judge produced no valid label are dropped for that judge and counted.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from scipy import stats

from .judge_analysis import binarise, load_judge
from .judges import CORPORA, JUDGES

EPS = 0.10              # crowd workers (paper FAQ p. 15; section B.1)
EPS_SENSITIVITY = (0.05, 0.15, 0.20)
Q = 0.05
MIN_ITEMS = 30
PASS_OMEGA = 0.5


def by_reject(pvals: list[float], q: float = Q) -> list[bool]:
    m = len(pvals)
    if m == 0:
        return []
    c = sum(1 / j for j in range(1, m + 1))
    order = sorted(range(m), key=lambda i: pvals[i])
    k_star = 0
    for rank, i in enumerate(order, start=1):
        if pvals[i] <= rank / m * q / c:
            k_star = rank
    out = [False] * m
    for rank, i in enumerate(order, start=1):
        out[i] = rank <= k_star
    return out


_FULL: dict[str, dict] = {}


def _ratings(corpus: str, it: dict) -> dict:
    """All ratings for DICES-350 (123 raters per item, as the test is designed for);
    the capped sample ratings elsewhere (their caps never bind below 5 raters' worth of
    annotators with >= 30 items anyway)."""
    if corpus != "dices350":
        return it["ratings"]
    if not _FULL:
        from .corpora import load_dices350
        v = {x.name: x for x in load_dices350(Path("data"), cap=None).variants}["three_way"]
        _FULL.update(v.matrix)
    return _FULL[it["item_id"]]


def _annotator_items(sample: dict, corpus: str) -> dict[str, list[tuple[str, int, dict]]]:
    by: dict[str, list] = {}
    for it in sample["items"]:
        lab = {r: binarise(corpus, v) for r, v in _ratings(corpus, it).items()}
        for j in lab:
            by.setdefault(j, []).append((it["item_id"], lab[j], lab))
    return by


def run_corpus(corpus: str, judge: str, eps: float) -> dict | None:
    sample = json.load(open(f"results/judge/sample_{corpus}.json"))
    out = load_judge(judge, corpus)
    if len(out) != len(sample["items"]):
        return None
    pred = {k: binarise(corpus, v["label"]) for k, v in out.items()}
    per = []
    for j, rows in sorted(_annotator_items(sample, corpus).items()):
        rows = [r for r in rows if pred[r[0]] is not None]
        if len(rows) < MIN_ITEMS:
            continue
        d, wf_all = [], []
        for item, hj, lab in rows:
            others = [v for k, v in lab.items() if k != j]
            sf = sum(pred[item] == v for v in others) / len(others)
            sh = sum(hj == v for v in others) / len(others)
            wf, wh = int(sf >= sh), int(sh >= sf)
            d.append(wh - wf)
            wf_all.append(wf)
        n = len(d)
        mean = sum(d) / n
        sd = math.sqrt(sum((x - mean) ** 2 for x in d) / (n - 1))
        if sd == 0:
            p = 0.0 if mean < eps else 1.0
        else:
            t = (mean - eps) / (sd / math.sqrt(n))
            p = float(stats.t.cdf(t, n - 1))
        per.append({"annotator": j, "n": n, "rho_f": sum(wf_all) / n, "p": p})
    rej = by_reject([x["p"] for x in per])
    for x, r in zip(per, rej):
        x["rejected"] = r
    m = len(per)
    omega = sum(rej) / m if m else None
    return {"n_annotators_eligible": m, "omega": omega,
            "passes": (omega is not None and m >= 3 and omega >= PASS_OMEGA),
            "applicable": m >= 3,
            "rho": (sum(x["rho_f"] for x in per) / m) if m else None,
            "per_annotator": per}


def run(out: Path = Path("results/judge/alt_test.json")) -> dict:
    res = {"params": {"eps": EPS, "q": Q, "min_items": MIN_ITEMS, "pass_omega": PASS_OMEGA,
                      "labels": "binary", "correction": "Benjamini-Yekutieli per corpus"}}
    for c in CORPORA:
        res[c] = {}
        for j in JUDGES:
            r = run_corpus(c, j, EPS)
            if r is None:
                continue
            r["sensitivity_omega"] = {str(e): (run_corpus(c, j, e) or {}).get("omega") for e in EPS_SENSITIVITY}
            res[c][j] = r
    out.write_text(json.dumps(res, indent=1, sort_keys=True))
    return res


def eligibility() -> dict:
    return {c: sum(1 for rows in _annotator_items(json.load(open(f"results/judge/sample_{c}.json")), c).values()
                   if len(rows) >= MIN_ITEMS) for c in CORPORA}


if __name__ == "__main__":
    r = run()
    for c in CORPORA:
        for j, v in r[c].items():
            print(f"{c:10s} {j:18s} annotators {v['n_annotators_eligible']:3d} omega {v['omega']} pass {v['passes']} rho {v['rho']}")
