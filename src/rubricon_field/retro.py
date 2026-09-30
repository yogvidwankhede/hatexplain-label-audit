"""PREREG R1 (exploratory): could published HateXplain leaderboard gaps be resolved?

Inputs, all from primary sources:
* accuracies: Mathew et al. (2021), arXiv 2012.10289v1, Table 6 ("Acc." column;
  each model appears once per token method with identical accuracy);
* test size: the official split file Data/post_id_divisions.json in
  hate-alert/HateXplain (1,924 test posts, all with a majority label; the 919
  "undecided" posts are outside every split).

For a paired accuracy comparison on n items with discordance rate p (fraction of
items where exactly one system is right), the z statistic of a gap g is
g / sqrt((p - g^2) / n). A gap is therefore resolvable at threshold z only if the
two systems disagree on fewer than p_max = g^2 * (1 + n / z^2) of the items. We
report p_max, because p is not published: a reader can compare it with how often
two different architectures plausibly disagree (Section 4 of the paper uses the
observed rates between our LLM judges as a reference point).

This says whether the *labels and test size* could support a claimed ordering. It
does not say any claim in the source paper is wrong, and published accuracies are
rounded to three decimals (+/- 0.0005 each), which is carried into every figure.
"""

from __future__ import annotations

import itertools
import json
import random
from collections import Counter
from pathlib import Path

from rubricon.gates.attenuation import mde_observed

from . import assumptions as A

SOURCE = "Mathew et al. 2021, arXiv:2012.10289v1, Table 6, Acc. column"
ACCURACY = {
    "CNN-GRU": 0.627, "BiRNN": 0.595, "BiRNN-Attn": 0.621, "BiRNN-HateXplain": 0.629,
    "BERT": 0.690, "BERT-HateXplain": 0.698,
}
P_DISC_SWEEP = (0.05, 0.10, 0.20, 0.30)


def human_reference(data_dir: Path) -> dict:
    """On official test posts: how often does one annotator match the other two when they agree?"""
    d = json.load(open(data_dir / "hatexplain.json"))
    test = json.load(open(data_dir / "hatexplain_post_id_divisions.json"))["test"]
    hit = tot = margin1 = 0
    for pid in test:
        labs = [a["label"] for a in d[pid]["annotators"]]
        c = Counter(labs)
        if max(c.values()) == 2:
            margin1 += 1
        held = random.Random(f"{A.SEED_PAPER}:retro:{pid}").randrange(3)
        rest = [l for i, l in enumerate(labs) if i != held]
        if rest[0] == rest[1]:
            tot += 1
            hit += labs[held] == rest[0]
    return {"n_test": len(test), "test_items_2_1_split": margin1,
            "test_items_2_1_split_fraction": round(margin1 / len(test), 4),
            "heldout_vs_other_two_when_they_agree": round(hit / tot, 4), "n_where_other_two_agree": tot}


def run(data_dir: Path = Path("data"), out: Path = Path("results/retro_hatexplain.json")) -> dict:
    n = len(json.load(open(data_dir / "hatexplain_post_id_divisions.json"))["test"])
    rows = []
    for a, b in itertools.combinations(ACCURACY, 2):
        hi, lo = (a, b) if ACCURACY[a] >= ACCURACY[b] else (b, a)
        g = round(ACCURACY[hi] - ACCURACY[lo], 3)
        row = {"better": hi, "worse": lo, "gap": g, "gap_range_from_rounding": [round(g - 0.001, 3), round(g + 0.001, 3)]}
        for z in (1.96, 2.58):
            p_max = g * g * (1 + n / z**2)
            row[f"p_max_resolvable_z{z}"] = round(p_max, 4)
        row["mde_by_p_disc"] = {str(p): round(mde_observed(n, (p - g * g) ** 0.5 if p > g * g else 0.0), 4)
                                for p in P_DISC_SWEEP}
        row["resolvable_if_p_disc"] = {str(p): bool(g > 0 and g / (((p - g * g) / n) ** 0.5) > 1.96)
                                       for p in P_DISC_SWEEP if p > g}
        rows.append(row)
    res = {"source": SOURCE, "n_test": n, "accuracy": ACCURACY, "pairs": rows,
           "human_reference": human_reference(data_dir)}
    out.write_text(json.dumps(res, indent=1, sort_keys=True))
    return res


if __name__ == "__main__":
    r = run()
    for p in sorted(r["pairs"], key=lambda x: x["gap"]):
        print(f"{p['better']:17s} > {p['worse']:17s} gap {p['gap']:.3f}  resolvable at z=1.96 only if p_disc < {p['p_max_resolvable_z1.96']:.3f}")
    print(r["human_reference"])
