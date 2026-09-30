"""Cross-corpus agreement battery (PREREG.md analyses A1-A3).

For every variant of every corpus this computes, with ``rubricon`` doing all the
numerics: Krippendorff's alpha with an item-cluster bootstrap interval, Gwet AC1,
Fleiss' kappa, raw pairwise agreement, the modal-label agreement ceiling, and the
one-vote-margin / no-majority fractions.

The ceiling is reported two ways. ``ceiling_in_sample`` is the mean modal share
(the quantity used in the earlier HateXplain study): each item's modal label is
computed *including* the vote it is then scored against, so with few raters it is
optimistic by construction. ``ceiling_leave_one_out`` predicts each rater's label
from the modal label of the *other* raters (ties split their credit evenly), which
is the honest version of "a predictor that knows what the other people said".
Both are agreement with a human, not upper bounds on a system.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from rubricon.stats.agreement import (
    fleiss_kappa, gwet_ac1, krippendorff_alpha, percent_agreement,
)

from . import fast_alpha

from . import assumptions as A
from .corpora import LOADERS, MAX_RATINGS_PER_ITEM, Variant

BOOT_REPLICATES = 1000        # PREREG.md A1
# PREREG.md A1 allowed bootstrapping a <=10,000-item subsample; DEVIATIONS.md D8
# replaces that with the full-data bootstrap in fast_alpha, so the point estimate
# and its interval share one basis.


def _r(x, nd=4):
    return None if x is None or x != x else round(x, nd)


def margin_stats(matrix) -> dict:
    n = one = tie = 0
    for cell in matrix.values():
        c = sorted(Counter(cell.values()).values(), reverse=True)
        n += 1
        second = c[1] if len(c) > 1 else 0
        if c[0] == second:
            tie += 1
        elif c[0] - second == 1:
            one += 1
    return {"n_items": n,
            "one_vote_margin_fraction": _r(one / n),
            "no_strict_plurality_fraction": _r(tie / n)}


def ceilings(matrix) -> dict:
    in_sample, loo, total = 0.0, 0.0, 0
    for cell in matrix.values():
        vals = list(cell.values())
        m = len(vals)
        cnt = Counter(vals)
        in_sample += max(cnt.values()) / m
        # leave one rater out: predict their label by the modal label of the others
        hit = 0.0
        for v in vals:
            rest = cnt.copy()
            rest[v] -= 1
            if rest[v] == 0:
                del rest[v]
            top = max(rest.values())
            winners = [k for k, c in rest.items() if c == top]
            hit += (1.0 / len(winners)) if v in winners else 0.0
        loo += hit / m
        total += 1
    return {"ceiling_in_sample": _r(in_sample / total),
            "ceiling_leave_one_out": _r(loo / total)}


def summarise(v: Variant) -> dict:
    m = v.matrix
    a = krippendorff_alpha(m, metric=v.metric)
    ci = fast_alpha.bootstrap(m, metric=v.metric, n_boot=BOOT_REPLICATES, seed=A.SEED_PAPER_BOOTSTRAP)
    if abs(ci["point"] - round(a.value, 4)) > 1e-4:
        raise AssertionError(f"fast alpha {ci['point']} != rubricon {a.value} on {v.name}")
    per_item = Counter(len(c) for c in m.values())
    out = {
        "variant": v.name,
        "metric": v.metric,
        "n_items": len(m),
        "n_raters": len({r for c in m.values() for r in c}),
        "ratings_per_item": {str(k): per_item[k] for k in sorted(per_item)},
        "alpha": _r(a.value),
        "alpha_ci95": ci,
        "ci_basis": {"n_items_bootstrapped": ci["n_clusters"], "replicates": BOOT_REPLICATES},
        "raw_pairwise_agreement": _r(percent_agreement(m).value),
        "notes": v.notes,
    }
    if v.metric == "nominal":
        out["gwet_ac1"] = _r(gwet_ac1(m).value)
        out["fleiss_kappa"] = _r(fleiss_kappa(m).value)
        out.update(ceilings(m))
        out.update(margin_stats(m))
    return out


def run(data_dir: Path, out_dir: Path, corpora: list[str] | None = None,
        cap: int | None = MAX_RATINGS_PER_ITEM) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if cap == MAX_RATINGS_PER_ITEM else "_allratings"
    results = {}
    for name in corpora or list(LOADERS):
        corpus = LOADERS[name](data_dir, cap=cap)
        results[name] = {"provenance": corpus.provenance, "cap": cap,
                         "variants": [summarise(v) for v in corpus.variants]}
        (out_dir / f"cross_{name}{suffix}.json").write_text(
            json.dumps(results[name], indent=2, sort_keys=True))
        print(f"{name}: done", flush=True)
    return results


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    cap = MAX_RATINGS_PER_ITEM
    if args and args[0] == "--all-ratings":
        cap, args = None, args[1:]
    run(Path("data"), Path("results"), args or None, cap=cap)
