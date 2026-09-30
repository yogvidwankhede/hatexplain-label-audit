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
import random
from collections import Counter
from pathlib import Path

from rubricon.stats.agreement import (
    alpha_interval, fleiss_kappa, gwet_ac1, krippendorff_alpha, percent_agreement,
)

from . import assumptions as A
from .corpora import LOADERS, Variant

BOOT_REPLICATES = 1000        # PREREG.md A1
BOOT_MAX_ITEMS = 10_000       # PREREG.md A1


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
    keys = sorted(m)
    if len(keys) > BOOT_MAX_ITEMS:
        rng = random.Random(A.SEED_PAPER_BOOTSTRAP)
        keys = rng.sample(keys, BOOT_MAX_ITEMS)
    sub = {k: m[k] for k in keys}
    ci = alpha_interval(sub, metric=v.metric, n_boot=BOOT_REPLICATES, seed=A.SEED_PAPER_BOOTSTRAP)
    per_item = Counter(len(c) for c in m.values())
    out = {
        "variant": v.name,
        "metric": v.metric,
        "n_items": len(m),
        "n_raters": len({r for c in m.values() for r in c}),
        "ratings_per_item": {str(k): per_item[k] for k in sorted(per_item)},
        "alpha": _r(a.value),
        "alpha_ci95": ci,
        "ci_basis": {"n_items_bootstrapped": len(sub), "replicates": BOOT_REPLICATES},
        "raw_pairwise_agreement": _r(percent_agreement(m).value),
        "notes": v.notes,
    }
    if v.metric == "nominal":
        out["gwet_ac1"] = _r(gwet_ac1(m).value)
        out["fleiss_kappa"] = _r(fleiss_kappa(m).value)
        out.update(ceilings(m))
        out.update(margin_stats(m))
    return out


def run(data_dir: Path, out_dir: Path, corpora: list[str] | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for name in corpora or list(LOADERS):
        corpus = LOADERS[name](data_dir)
        results[name] = {"provenance": corpus.provenance,
                         "variants": [summarise(v) for v in corpus.variants]}
        (out_dir / f"cross_{name}.json").write_text(
            json.dumps(results[name], indent=2, sort_keys=True))
        print(f"{name}: done", flush=True)
    return results


if __name__ == "__main__":
    import sys
    run(Path("data"), Path("results"), sys.argv[1:] or None)
