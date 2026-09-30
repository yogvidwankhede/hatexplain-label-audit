"""Aggregate the gate simulation (PREREG S1 and S1b). Outputs results/gate_sim_summary.json.

Definitions (DEVIATIONS.md D4). A *claim* is an observed gap A - B > 0. A claim is
*false* when the true gap <= 0 and *resolvable* when the true gap >= 0.03.
Rates are per COMPARISON (divided by replicates) unless the name says
"given_claim".
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from .gate_sim import CHECK_ORDER

FALSE_DELTAS = (-0.03, 0.0)
RESOLVABLE_MIN = 0.03


def load(path: Path):
    return [json.loads(l) for l in path.open()]


def publish_count(cell, drop: int | None = None) -> int:
    """Claims not blocked when check index `drop` is removed from the gate."""
    keep = ~(1 << drop) if drop is not None else ~0
    total = 0
    for mask, cnt in cell["block_masks"].items():
        if (int(mask) & keep) == 0:
            total += cnt
    return total   # mask 0 (no BLOCK) is already included above


def aggregate(cells, keyfn):
    acc = defaultdict(lambda: defaultdict(float))
    for c in cells:
        k = keyfn(c)
        a = acc[k]
        a["reps"] += c["reps"]
        a["claims"] += c["n_claims"]
        a["naive"] += c["n_naive_pub"]
        a["gate"] += c["n_gate_pub"]
        a["gap_naive"] += c["sum_gap_naive"]
        a["gap_gate"] += c["sum_gap_gate"]
        for j in range(len(CHECK_ORDER)):
            a[f"drop_{j}"] += publish_count(c, j)
    return acc


def rates(a):
    r = a["reps"]
    return {
        "comparisons": int(r),
        "claim_rate": round(a["claims"] / r, 4),
        "naive_publish_rate": round(a["naive"] / r, 4),
        "gate_publish_rate": round(a["gate"] / r, 4),
        "gate_block_given_claim": round(1 - a["gate"] / a["claims"], 4) if a["claims"] else None,
        "naive_block_given_claim": round(1 - a["naive"] / a["claims"], 4) if a["claims"] else None,
        "mean_gap_published_naive": round(a["gap_naive"] / a["naive"], 4) if a["naive"] else None,
        "mean_gap_published_gate": round(a["gap_gate"] / a["gate"], 4) if a["gate"] else None,
        "ablation_publish_rate_without": {
            CHECK_ORDER[j]: round(a[f"drop_{j}"] / r, 4) for j in range(len(CHECK_ORDER))
        },
    }


def main(results: Path = Path("results")):
    out = {}
    for scen in ("iid", "het"):
        cells = load(results / f"gate_sim_{scen}.jsonl")
        groups = {
            "false_claims": [c for c in cells if c["delta"] in FALSE_DELTAS],
            "resolvable": [c for c in cells if c["delta"] >= RESOLVABLE_MIN],
            "small_true_gap": [c for c in cells if 0 < c["delta"] < RESOLVABLE_MIN],
        }
        s = {}
        for gname, gc in groups.items():
            s[gname] = {
                "overall": rates(aggregate(gc, lambda c: "all")["all"]),
                "by_alpha": {str(k): rates(v) for k, v in sorted(aggregate(gc, lambda c: c["alpha_target"]).items())},
                "by_n": {str(k): rates(v) for k, v in sorted(aggregate(gc, lambda c: c["n"]).items())},
            }
        # sign errors: true overall gap < 0 but claim published
        neg = [c for c in cells if c["delta"] < 0]
        s["sign_error_delta_neg"] = {"by_alpha": {str(k): rates(v) for k, v in sorted(
            aggregate(neg, lambda c: c["alpha_target"]).items())}}
        out[scen] = s
    (results / "gate_sim_summary.json").write_text(json.dumps(out, indent=1, sort_keys=True))
    return out


if __name__ == "__main__":
    main()
