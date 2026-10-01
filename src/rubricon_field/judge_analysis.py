"""Score the judges against the human panels (PREREG_ADDENDUM.md, Analysis).

M1 (primary): binary agreement with the binary panel majority; the held-out human
is scored on exactly the same items. M2: mean binary agreement with individual
panel raters. M3: native-scale agreement with the panel plurality. Refusals and
unparseable outputs are excluded in the primary analysis and counted as wrong in
a sensitivity analysis. Every interval is a 95% item-cluster percentile bootstrap
(2,000 replicates, seeded), paired where two raters are compared.

Gate v2 is applied to two claim families, with z* = 2.58:
* "judge J agrees with the panel more often than a held-out human" -- the
  direction check runs on d_i = [J right] - [human right];
* "judge J beats judge K" on the same items.
The attenuation report uses the panel raters' own pairwise disagreement.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from rubricon.core.schema import Verdict
from rubricon.gates.signal import GatePolicy, SignalGate
from rubricon.gates.attenuation import (
    PairedGap, attenuation_report, check_contested_consistency, check_direction,
)

from . import assumptions as A
from .judges import CORPORA, JUDGES, parse

BOOT = 2000
Z_A, Z_B = 1.959964, 0.841621


def binarise(corpus: str, v):
    if v is None:
        return None
    if corpus == "hatexplain":
        return int(v != "normal")
    if corpus == "mhs":
        return int(v == 2)
    if corpus == "wikitalk":
        return int(v < 0)
    if corpus == "dices350":
        return int(v == "Yes")
    raise KeyError(corpus)


def _strict_majority(vals):
    c = Counter(vals)
    top = max(c.values())
    winners = [k for k, n in c.items() if n == top]
    return winners[0] if len(winners) == 1 else None


def _boot_mean(x: np.ndarray, rng) -> tuple[float, float]:
    n = len(x)
    idx = rng.integers(0, n, size=(BOOT, n))
    m = x[idx].mean(1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def load_judge(judge: str, corpus: str) -> dict[str, dict]:
    p = Path("results/judge/sample") / f"{judge}__{corpus}.jsonl"
    if not p.exists():
        return {}
    out = {}
    for line in p.open():
        r = json.loads(line)
        if r.get("raw_redacted"):   # long free text kept private; see scripts/redact_judge_outputs.py
            lab, how = r["parsed_label"], r["parsed_how"]
        else:
            lab, how = parse(corpus, r.get("raw"))
        if r.get("stop_reason") == "refusal" or r.get("refusal"):
            how = "refusal"
            lab = None
        out[r["item_id"]] = {"label": lab, "how": how, "in": r.get("in_tokens") or 0,
                             "out": r.get("out_tokens") or 0, "model": r.get("model")}
    return out


def analyse(corpus: str) -> dict:
    sample = json.load(open(f"results/judge/sample_{corpus}.json"))
    rng = np.random.default_rng([A.SEED_PAPER, 77, CORPORA.index(corpus)])
    items = sample["items"]
    judges = {j: load_judge(j, corpus) for j in JUDGES}
    judges = {j: v for j, v in judges.items() if len(v) == len(items)}

    rows = []  # per item
    for it in items:
        held = it["heldout_rater"]
        panel = {r: v for r, v in it["ratings"].items() if r != held}
        pb = [binarise(corpus, v) for v in panel.values()]
        rows.append({
            "item": it["item_id"], "human_native": it["ratings"][held],
            "human_bin": binarise(corpus, it["ratings"][held]),
            "panel_bin_major": _strict_majority(pb), "panel_bin": pb,
            "panel_native_major": _strict_majority(list(panel.values())),
            "unanimous_bin": len(set(pb)) == 1,
        })
    k_panel = Counter(len(r["panel_bin"]) for r in rows)
    k_mode = k_panel.most_common(1)[0][0]
    pan = [r["panel_bin"] for r in rows]
    dis = float(np.mean([sum(a != b for i, a in enumerate(p) for b in p[i + 1:]) / (len(p) * (len(p) - 1) / 2)
                         for p in pan]))

    res = {"corpus": corpus, "n_items": len(rows), "panel_sizes": dict(k_panel),
           "panel_pairwise_disagreement_binary": round(dis, 4), "judges": {}}
    m1_mask = np.array([r["panel_bin_major"] is not None for r in rows])
    res["m1_items_with_panel_majority"] = int(m1_mask.sum())
    human_right = np.array([r["human_bin"] == r["panel_bin_major"] for r in rows], dtype=float)
    h = human_right[m1_mask]
    res["human"] = {"m1": round(h.mean(), 4), "m1_ci": [round(x, 4) for x in _boot_mean(h, rng)],
                    "m2": round(float(np.mean([np.mean([r["human_bin"] == x for x in r["panel_bin"]]) for r in rows])), 4)}
    native_mask = np.array([r["panel_native_major"] is not None for r in rows])
    if corpus != "wikitalk":
        res["human"]["m3"] = round(float(np.mean([r["human_native"] == r["panel_native_major"]
                                                   for r, m in zip(rows, native_mask) if m])), 4)

    correct = {}
    for j, out in judges.items():
        lab = [out[r["item"]]["label"] for r in rows]
        jb = [binarise(corpus, x) for x in lab]
        valid = np.array([x is not None for x in jb])
        right = np.array([x == r["panel_bin_major"] for x, r in zip(jb, rows)], dtype=float)
        mask = m1_mask & valid
        d = right[mask] - human_right[mask]
        diff_ci = _boot_mean(d, rng)
        pg = PairedGap(int(mask.sum()), float(d.mean()), float(d.std(ddof=1)), float((d != 0).mean()))
        unan = np.array([r["unanimous_bin"] for r in rows])[mask]
        checks = [check_direction(pg), check_contested_consistency(d[unan].tolist(), d[~unan].tolist()),
                  attenuation_report(pg, dis, k_mode)]
        verdict = ("block" if any(c.verdict is Verdict.BLOCK for c in checks)
                   else "warn" if any(c.verdict is Verdict.WARN for c in checks) else "pass")
        hows = Counter(out[r["item"]]["how"] for r in rows)
        entry = {
            "model": Counter(out[r["item"]]["model"] for r in rows).most_common(1)[0][0],
            "parse": dict(hows),
            "m1": round(right[mask].mean(), 4), "m1_ci": [round(x, 4) for x in _boot_mean(right[mask], rng)],
            "m1_n": int(mask.sum()),
            "m1_invalid_as_wrong": round(float(np.where(valid, right, 0.0)[m1_mask].mean()), 4),
            "human_m1_same_items": round(human_right[mask].mean(), 4),
            "diff_vs_human": round(float(d.mean()), 4), "diff_vs_human_ci": [round(x, 4) for x in diff_ci],
            "m2": round(float(np.mean([np.mean([x == y for y in r["panel_bin"]])
                                       for x, r, v in zip(jb, rows, valid) if v])), 4),
            "gate_v2_claim_judge_beats_human": {"verdict": verdict, "checks": [c.to_dict() for c in checks]},
            "tokens_in": int(sum(out[r["item"]]["in"] for r in rows)),
            "tokens_out": int(sum(out[r["item"]]["out"] for r in rows)),
        }
        if corpus != "wikitalk":
            ok = [(x == r["panel_native_major"]) for x, r, m in zip(lab, rows, native_mask) if m and x is not None]
            entry["m3"] = round(float(np.mean(ok)), 4)
        res["judges"][j] = entry
        correct[j] = (right, valid)

    # pairwise judge comparisons; gate v1 uses the corpus's capped binary alpha (cross_<corpus>.json)
    var = {"hatexplain": "binary", "mhs": "binary", "wikitalk": "toxicity_binary", "dices350": "binary"}[corpus]
    corpus_alpha = next(v["alpha"] for v in json.load(open(f"results/cross_{corpus}.json"))["variants"]
                        if v["variant"] == var)
    res["corpus_alpha_binary"] = corpus_alpha
    v1_gate = SignalGate(GatePolicy(), depth="production")
    pairs = {}
    names = sorted(correct)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            (ra, va), (rb, vb) = correct[a], correct[b]
            mask = m1_mask & va & vb
            d = ra[mask] - rb[mask]
            pg = PairedGap(int(mask.sum()), float(d.mean()), float(d.std(ddof=1)) if d.std() > 0 else 1e-9,
                           float((d != 0).mean()))
            sign = 1 if pg.gap >= 0 else -1
            pgd = PairedGap(pg.n, abs(pg.gap), pg.sd, pg.disagreement)
            dirc = check_direction(pgd)
            se = pgd.sd / np.sqrt(pgd.n)
            v1_checks = [v1_gate.check_reliability(corpus_alpha),
                         v1_gate.check_effect_vs_mde(pgd.gap, (Z_A + Z_B) * se),
                         v1_gate.check_interval_excludes_zero(pgd.gap - Z_A * se, pgd.gap + Z_A * se)]
            v1_ok = not any(c.verdict is Verdict.BLOCK for c in v1_checks)
            pairs[f"{a}|{b}"] = {"gate_v1_publishable": v1_ok,
                                 "gate_v1_blocking": [c.name for c in v1_checks if c.verdict is Verdict.BLOCK],"gap": round(pg.gap, 4), "ci": [round(x, 4) for x in _boot_mean(d, rng)],
                                 "discordance": round(pg.disagreement, 4), "n": pg.n,
                                 "leader": a if sign > 0 else b,
                                 "gate_v2_direction": dirc.verdict.value, "z": round(pgd.z, 2)}
    res["judge_pairs"] = pairs

    if corpus == "dices350":
        import pandas as pd
        g = pd.read_csv("data/dices/diverse_safety_adversarial_dialog_350.csv",
                        usecols=["item_id", "safety_gold"]).drop_duplicates("item_id")
        gold = {str(i): int(v == "Yes") for i, v in zip(g.item_id, g.safety_gold)}
        exp = {"human_heldout": round(float(np.mean([r["human_bin"] == gold[r["item"]] for r in rows])), 4)}
        for j, out in judges.items():
            xs = [(binarise(corpus, out[r["item"]]["label"]), gold[r["item"]]) for r in rows]
            xs = [(a, b) for a, b in xs if a is not None]
            exp[j] = round(float(np.mean([a == b for a, b in xs])), 4)
        panel_major = [(r["panel_bin_major"], gold[r["item"]]) for r in rows if r["panel_bin_major"] is not None]
        exp["panel_majority"] = round(float(np.mean([a == b for a, b in panel_major])), 4)
        res["expert_gold_agreement"] = exp
        res["expert_gold_prevalence"] = round(float(np.mean(list(gold.values()))), 4)
        rng2 = np.random.default_rng([A.SEED_PAPER, 78])
        matched = {}
        cands = {"human_heldout": [r["human_bin"] for r in rows]}
        for j, out in judges.items():
            cands[j] = [binarise(corpus, out[r["item"]]["label"]) for r in rows]
        for name, labs in cands.items():
            keep = [(x, r) for x, r in zip(labs, rows) if x is not None and r["panel_bin_major"] is not None]
            jr = np.array([x == gold[r["item"]] for x, r in keep], dtype=float)
            pr = np.array([r["panel_bin_major"] == gold[r["item"]] for x, r in keep], dtype=float)
            d = jr - pr
            matched[name] = {"n": len(keep), "rater_vs_expert": round(float(jr.mean()), 4),
                             "panel_vs_expert_same_items": round(float(pr.mean()), 4),
                             "diff": round(float(d.mean()), 4),
                             "diff_ci": [round(x, 4) for x in _boot_mean(d, rng2)]}
        res["expert_gold_matched"] = matched
    return res


def run(out: Path = Path("results/judge/analysis.json")) -> dict:
    res = {c: analyse(c) for c in CORPORA}
    out.write_text(json.dumps(res, indent=1, sort_keys=True))
    return res


if __name__ == "__main__":
    r = run()
    for c, v in r.items():
        print(c, "human M1", v["human"]["m1"], v["human"]["m1_ci"])
        for j, e in v["judges"].items():
            print(f"   {j:18s} M1 {e['m1']:.3f} {e['m1_ci']} diff {e['diff_vs_human']:+.3f} {e['diff_vs_human_ci']} "
                  f"gate {e['gate_v2_claim_judge_beats_human']['verdict']} parse {e['parse']}")
