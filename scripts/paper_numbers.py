"""Generate paper/numbers.tex (LaTeX macros) and CLAIMS.md (the claims ledger).

Every number in the paper text is a macro defined here, and every macro records the
results file and JSON path it came from, plus the script that produces that file. The
commit hash of the results is stamped into CLAIMS.md. tests/test_paper_numbers.py
fails if paper/*.tex contains a statistic-shaped literal that is not a macro.

Formatting is done here, once, so rounding is consistent across the paper.
"""

from __future__ import annotations

import json
import statistics
import subprocess
from pathlib import Path

R = Path("results")
LEDGER: list[tuple[str, str, str, str, str]] = []  # macro, value, file, path, script


def _get(obj, path: str):
    for part in path.split("/"):
        if isinstance(obj, list):
            obj = obj[int(part)]
        else:
            obj = obj[part]
    return obj


def _load(f: str):
    return json.load(open(R / f))


def put(macro: str, value, fmt: str, file: str, path: str, script: str) -> None:
    if value is None:
        raise ValueError(f"{macro}: no value")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if fmt.endswith("d"):
            value = int(round(value))
        text = format(value, fmt)
    else:
        text = str(value)
    LEDGER.append((macro, text, file, path, script))


def src(macro, file, path, fmt, script, scale=1.0, derive=None):
    v = _get(_load(file), path)
    if derive:
        v = derive(v)
    if isinstance(v, (int, float)) and not isinstance(v, bool) and scale != 1.0:
        v = v * scale
    put(macro, v, fmt, file, path, script)


def variant(file: str, name: str) -> tuple[int, dict]:
    vs = _load(file)["variants"]
    for i, v in enumerate(vs):
        if v["variant"] == name:
            return i, v
    raise KeyError(name)


def cross_numbers():
    s = "src/rubricon_field/cross_corpus.py"
    corp = {"Hx": ("cross_hatexplain.json", "binary"), "Hxthree": ("cross_hatexplain.json", "three_way"),
            "Mhs": ("cross_mhs.json", "binary"), "Mhsthree": ("cross_mhs.json", "three_way"),
            "Wiki": ("cross_wikitalk.json", "toxicity_binary"), "Wikiord": ("cross_wikitalk.json", "toxicity_score"),
            "Dthree": ("cross_dices350.json", "binary"), "Dnine": ("cross_dices990.json", "binary"),
            "Dthreethree": ("cross_dices350.json", "three_way")}
    for k, (f, var) in corp.items():
        i, v = variant(f, var)
        p = f"variants/{i}"
        src(f"alpha{k}", f, f"{p}/alpha", ".3f", s)
        src(f"alphaLo{k}", f, f"{p}/alpha_ci95/lo", ".3f", s)
        src(f"alphaHi{k}", f, f"{p}/alpha_ci95/hi", ".3f", s)
        src(f"raw{k}", f, f"{p}/raw_pairwise_agreement", ".3f", s)
        src(f"items{k}", f, f"{p}/n_items", ",d", s)
        src(f"raters{k}", f, f"{p}/n_raters", ",d", s)
        if v["metric"] == "nominal":
            src(f"acOne{k}", f, f"{p}/gwet_ac1", ".3f", s)
            src(f"ceilLoo{k}", f, f"{p}/ceiling_leave_one_out", ".3f", s)
            src(f"ceilIn{k}", f, f"{p}/ceiling_in_sample", ".3f", s)
            src(f"marginOne{k}", f, f"{p}/one_vote_margin_fraction", ".1f", s, scale=100)
    # all-ratings sensitivity for MHS
    i, _ = variant("cross_mhs_allratings.json", "binary")
    src("alphaMhsAll", "cross_mhs_allratings.json", f"variants/{i}/alpha", ".3f", s)
    i3, _ = variant("cross_mhs_allratings.json", "three_way")
    src("alphaMhsthreeAll", "cross_mhs_allratings.json", f"variants/{i3}/alpha", ".3f", s)
    go = _load("cross_goemotions.json")["variants"]
    al = [v["alpha"] for v in go]
    put("alphaGoMedian", statistics.median(al), ".3f", "cross_goemotions.json", "median(variants/*/alpha)", s)
    put("alphaGoMin", min(al), ".3f", "cross_goemotions.json", "min(variants/*/alpha)", s)
    put("alphaGoMax", max(al), ".3f", "cross_goemotions.json", "max(variants/*/alpha)", s)
    top = max(go, key=lambda v: v["alpha"])
    put("goTopEmotion", top["variant"].split(":")[1], "s", "cross_goemotions.json", "argmax alpha", s)
    put("nGoEmotions", len(go), "d", "cross_goemotions.json", "len(variants)", s)


def sim_numbers():
    s = "src/rubricon_field/gate_sim.py"
    files = {"Iid": "gate_sim_iid.jsonl", "Het": "gate_sim_het.jsonl", "Hetneg": "gate_sim_het_k-0.10.jsonl",
             "Hetmild": "gate_sim_het_k-0.05.jsonl"}
    for k, f in files.items():
        cells = [json.loads(l) for l in open(R / f)]
        F = [c for c in cells if c["delta"] <= 0]
        Rz = [c for c in cells if c["delta"] >= 0.03]
        rF, rR = sum(c["reps"] for c in F), sum(c["reps"] for c in Rz)
        def rate(cs, r, key, sub=None):
            return sum((c[key][sub] if sub else c[key]) for c in cs) / r * 100
        put(f"simFalseVone{k}", rate(F, rF, "n_gate_pub"), ".2f", f, "delta<=0: sum n_gate_pub / reps", s)
        put(f"simResVone{k}", rate(Rz, rR, "n_gate_pub"), ".1f", f, "delta>=0.03: sum n_gate_pub / reps", s)
        put(f"simFalseVtwo{k}", rate(F, rF, "v2_n_pub"), ".2f", f, "delta<=0: sum v2_n_pub / reps", s)
        put(f"simResVtwo{k}", rate(Rz, rR, "v2_n_pub"), ".1f", f, "delta>=0.03: sum v2_n_pub / reps", s)
        for z, name in (("1.96", "Zlow"), ("2.58", "Zmid")):
            put(f"simFalse{name}{k}", rate(F, rF, "z_threshold_counts", z), ".2f", f, f"z>{z} false", s)
            put(f"simRes{name}{k}", rate(Rz, rR, "z_threshold_counts", z), ".1f", f, f"z>{z} resolvable", s)
        # z-test resolvable rate interpolated at v1's false rate
        zs = list(cells[0]["z_threshold_counts"])
        pts = sorted((rate(F, rF, "z_threshold_counts", z), rate(Rz, rR, "z_threshold_counts", z)) for z in zs)
        f1 = rate(F, rF, "n_gate_pub")
        est = None
        for (f0, r0), (f2, r2) in zip(pts, pts[1:]):
            if f0 <= f1 <= f2 and f2 > f0:
                est = r0 + (r2 - r0) * (f1 - f0) / (f2 - f0)
        put(f"simResMatched{k}", est, ".1f", f, "linear interpolation of the z-grid at v1's false rate", s)
        n = len(cells)
        put(f"simCells{k}", n, ",d", f, "number of cells", s)
        low = [c for c in Rz if c["alpha_target"] < 0.5]
        if low:
            rl = sum(c["reps"] for c in low)
            put(f"simResVoneLowAlpha{k}", rate(low, rl, "n_gate_pub"), ".2f", f, "alpha<0.5, delta>=0.03", s)
            put(f"simResZlowLowAlpha{k}", rate(low, rl, "z_threshold_counts", "1.96"), ".1f", f, "alpha<0.5 z>1.96", s)
    for chk, name in (("reliability", "Rel"), ("effect_vs_mde", "Mde"), ("interval_excludes_zero", "Ci"),
                      ("precision", "Prec"), ("replication", "Repl")):
        src(f"ablRes{name}", "gate_sim_summary.json",
            f"iid/resolvable/overall/ablation_publish_rate_without/{chk}", ".1f",
            "src/rubricon_field/gate_sim_report.py", scale=100)
    src("ablResNone", "gate_sim_summary.json", "iid/resolvable/overall/gate_publish_rate", ".1f",
        "src/rubricon_field/gate_sim_report.py", scale=100)
    cells = [json.loads(l) for l in open(R / "gate_sim_iid.jsonl")]
    reps = sum(c["reps"] for c in cells)
    put("simComparisons", reps / 1e6, ".2f", "gate_sim_iid.jsonl", "sum reps / 1e6", s)


def claim4_numbers():
    s = "src/rubricon_field/claim4.py"
    d = _load("claim4.json")["corpora"]
    rows = [(c, r) for c, v in d.items() for r in v["rows"]]
    cells = {}
    for c, r in rows:
        cells.setdefault((c, r["model"], r["budget"], r["delta"]), []).append(r)
    wins = sum(1 for rs in cells.values() if max(rs, key=lambda r: r["power"])["k"] == 1)
    put("clFourCells", len(cells), "d", "claim4.json", "number of (corpus, model, budget, delta) cells", s)
    put("clFourWins", wins, "d", "claim4.json", "cells where k=1 has max power", s)
    w = next(r for c, r in rows if c == "wikitalk" and r["model"] == "uniform" and r["budget"] == 6000
             and r["delta"] == 0.03 and r["k"] == 3)
    put("clFourWikiKthree", w["power"], ".3f", "claim4.json", "wikitalk uniform B6000 d0.03 k3 power", s)
    put("clFourWikiKthreePred", w["predicted_power_iid"], ".3f", "claim4.json", "... predicted_power_iid", s)
    w1 = next(r for c, r in rows if c == "wikitalk" and r["model"] == "uniform" and r["budget"] == 6000
              and r["delta"] == 0.03 and r["k"] == 1)
    put("clFourWikiKone", w1["power"], ".3f", "claim4.json", "wikitalk uniform B6000 d0.03 k1 power", s)
    below = sum(1 for c, r in rows if r["k"] > 1 and r["power"] < r["predicted_power_iid"])
    kgt1 = sum(1 for c, r in rows if r["k"] > 1)
    put("clFourBelowPred", below, "d", "claim4.json", "k>1 rows with power < iid prediction", s)
    put("clFourKgtOne", kgt1, "d", "claim4.json", "k>1 rows", s)
    for c, name in (("wikitalk", "Wiki"), ("dices990", "Dnine"), ("dices350", "Dthree")):
        put(f"clFourItems{name}", d[c]["n_items"], ",d", "claim4.json", f"corpora/{c}/n_items", s)
        put(f"clFourEta{name}", d[c]["pool_eta_hat"], ".3f", "claim4.json", f"corpora/{c}/pool_eta_hat", s)


def retro_numbers():
    s = "src/rubricon_field/retro.py"
    r = _load("retro_hatexplain.json")
    put("retroN", r["n_test"], ",d", "retro_hatexplain.json", "n_test", s)
    pairs = r["pairs"]
    put("retroPairs", len(pairs), "d", "retro_hatexplain.json", "len(pairs)", s)
    unres = [p for p in pairs if p["p_max_resolvable_z1.96"] < 0.05]
    put("retroUnresolvable", len(unres), "d", "retro_hatexplain.json", "pairs with p_max(z=1.96) < 0.05", s)
    for p in pairs:
        if p["better"] == "BERT-HateXplain" and p["worse"] == "BERT":
            put("retroBertGap", p["gap"], ".3f", "retro_hatexplain.json", "pairs[BERT-HateXplain>BERT]/gap", s)
            put("retroBertPmax", p["p_max_resolvable_z1.96"] * 100, ".1f", "retro_hatexplain.json",
                "pairs[BERT-HateXplain>BERT]/p_max_resolvable_z1.96", s)
    h = r["human_reference"]
    put("retroHuman", h["heldout_vs_other_two_when_they_agree"] * 100, ".1f", "retro_hatexplain.json",
        "human_reference/heldout_vs_other_two_when_they_agree", s)
    put("retroSplit", h["test_items_2_1_split_fraction"] * 100, ".1f", "retro_hatexplain.json",
        "human_reference/test_items_2_1_split_fraction", s)
    put("retroBestAcc", max(r["accuracy"].values()), ".3f", "retro_hatexplain.json", "max(accuracy)", s)


def judge_numbers():
    p = R / "judge" / "analysis.json"
    if not p.exists():
        return
    s = "src/rubricon_field/judge_analysis.py"
    a = json.load(open(p))
    tag = {"hatexplain": "Hx", "mhs": "Mhs", "wikitalk": "Wiki", "dices350": "Dthree"}
    jt = {"claude-haiku-4-5": "Haiku", "claude-sonnet-5-5": "Sonnet", "gpt-4.1-mini": "Gpt",
          "qwen2.5-14b": "Qwen", "gpt-oss-20b": "Oss"}
    for c, t in tag.items():
        f = "judge/analysis.json"
        src(f"jHuman{t}", f, f"{c}/human/m1", ".3f", s)
        src(f"jHumanLo{t}", f, f"{c}/human/m1_ci/0", ".3f", s)
        src(f"jHumanHi{t}", f, f"{c}/human/m1_ci/1", ".3f", s)
        src(f"jPanelItems{t}", f, f"{c}/m1_items_with_panel_majority", ",d", s)
        for j, jn in jt.items():
            if j not in a[c]["judges"]:
                continue
            base = f"{c}/judges/{j}"
            src(f"j{jn}{t}", f, f"{base}/m1", ".3f", s)
            src(f"jDiff{jn}{t}", f, f"{base}/diff_vs_human", "+.3f", s)
            src(f"jDiffLo{jn}{t}", f, f"{base}/diff_vs_human_ci/0", "+.3f", s)
            src(f"jDiffHi{jn}{t}", f, f"{base}/diff_vs_human_ci/1", "+.3f", s)
            src(f"jGate{jn}{t}", f, f"{base}/gate_v2_claim_judge_beats_human/verdict", "s", s, derive=lambda v: v.upper())
    if "expert_gold_agreement" in a["dices350"]:
        for k, v in a["dices350"]["expert_gold_agreement"].items():
            name = {"human_heldout": "Human", "panel_majority": "Panel"}.get(k) or jt.get(k)
            put(f"jExpert{name}", v, ".3f", "judge/analysis.json", f"dices350/expert_gold_agreement/{k}", s)


def table(name: str, header: list[str], rows: list[list], file: str, script: str, colspec: str) -> None:
    """Write paper/tables/<name>.tex and ledger every numeric cell."""
    out = ["\\begin{tabular}{" + colspec + "}", "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    for r, row in enumerate(rows):
        cells = []
        for c, (val, fmt, path) in enumerate(row):
            text = format(val, fmt) if isinstance(val, (int, float)) and fmt else str(val)
            if path:
                LEDGER.append((f"table:{name}[{r},{c}]", text, file, path, script))
            cells.append(text)
        out.append(" & ".join(cells) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    Path("paper/tables").mkdir(exist_ok=True)
    Path(f"paper/tables/{name}.tex").write_text("\n".join(out) + "\n")


def extra_numbers():
    s = "src/rubricon_field/cross_corpus.py"
    # largest change in alpha from keeping all ratings, excluding MHS
    worst, where = 0.0, ""
    for c in ("hatexplain", "dices350", "dices990", "wikitalk", "goemotions"):
        a = {v["variant"]: v["alpha"] for v in _load(f"cross_{c}.json")["variants"]}
        b = {v["variant"]: v["alpha"] for v in _load(f"cross_{c}_allratings.json")["variants"]}
        for k in a:
            if abs(a[k] - b[k]) > worst:
                worst, where = abs(a[k] - b[k]), f"{c}:{k}"
    put("allRatingsMaxShift", worst, ".3f", "cross_*_allratings.json vs cross_*.json",
        f"max |alpha_all - alpha_capped| excluding MHS (at {where})", s)
    pv = _load("cross_mhs.json")["provenance"]
    put("mhsAnchorItems", pv["items_capped_to_max"], "d", "cross_mhs.json", "provenance/items_capped_to_max", s)
    put("mhsAnchorShare", pv["ratings_on_capped_items"] / pv["n_ratings_raw"] * 100, ".0f", "cross_mhs.json",
        "provenance/ratings_on_capped_items / n_ratings_raw", s)
    # attenuation estimator accuracy (iid, n >= 1000)
    import sys
    sys.path.insert(0, "src")
    from rubricon.gates.attenuation import eta_majority
    cells = [json.loads(l) for l in open(R / "gate_sim_iid.jsonl")]
    err = max(abs(c["mean_attenuation_est"] - (1 - 2 * eta_majority(c["eta_easy"], 3)))
              for c in cells if c["n"] >= 1000)
    put("attEstMaxErr", err, ".3f", "gate_sim_iid.jsonl", "max |mean_attenuation_est - (1-2 eta_3)| over cells n>=1000",
        "src/rubricon_field/gate_sim.py")
    r = _load("retro_hatexplain.json")
    pairs = r["pairs"]
    unres = [p for p in pairs if p["p_max_resolvable_z1.96"] < 0.05]
    put("retroMinGap", min(p["gap"] for p in unres), ".3f", "retro_hatexplain.json", "min gap among unresolvable", "src/rubricon_field/retro.py")
    put("retroMinPmax", min(p["p_max_resolvable_z1.96"] for p in unres) * 100, ".1f", "retro_hatexplain.json",
        "min p_max among unresolvable", "src/rubricon_field/retro.py")
    always = [p["gap"] for p in pairs if p["p_max_resolvable_z1.96"] >= 1]
    put("retroAlwaysGap", min(always), ".3f", "retro_hatexplain.json", "smallest gap with p_max >= 1", "src/rubricon_field/retro.py")
    rows = []
    for i, p in sorted(enumerate(pairs), key=lambda x: -x[1]["gap"]):
        rows.append([(f"{p['better']} $>$ {p['worse']}", None, None), (p["gap"], ".3f", f"pairs/{i}/gap"),
                     (min(p["p_max_resolvable_z1.96"], 1.0) * 100, ".1f", f"pairs/{i}/p_max_resolvable_z1.96"),
                     (min(p["p_max_resolvable_z2.58"], 1.0) * 100, ".1f", f"pairs/{i}/p_max_resolvable_z2.58")])
    table("retro", ["Comparison", "Gap", "$p_{\\max}$ (\\%), $z{=}1.96$", "$z{=}2.58$"], rows,
          "retro_hatexplain.json", "src/rubricon_field/retro.py", "@{}lrrr@{}")
    # full cross-corpus table
    rows = []
    for c, label in (("hatexplain", "HateXplain"), ("mhs", "MHS"), ("wikitalk", "Wikipedia Talk"),
                     ("dices350", "DICES-350"), ("dices990", "DICES-990")):
        for i, v in enumerate(_load(f"cross_{c}.json")["variants"]):
            vb = next(x for x in _load(f"cross_{c}_allratings.json")["variants"] if x["variant"] == v["variant"])
            nom = v["metric"] == "nominal"
            rows.append([(label, None, None), (v["variant"].replace("_", " "), None, None),
                         (v["alpha"], ".3f", f"variants/{i}/alpha"),
                         (f"[{v['alpha_ci95']['lo']:.3f}, {v['alpha_ci95']['hi']:.3f}]", None, f"variants/{i}/alpha_ci95"),
                         (v["gwet_ac1"] if nom else "--", ".3f" if nom else None, f"variants/{i}/gwet_ac1" if nom else None),
                         (v["raw_pairwise_agreement"], ".3f", f"variants/{i}/raw_pairwise_agreement"),
                         (f"{v['ceiling_leave_one_out']:.3f}--{v['ceiling_in_sample']:.3f}" if nom else "--", None,
                          f"variants/{i}/ceiling_*" if nom else None),
                         (vb["alpha"], ".3f", f"{c}_allratings variants/alpha")])
    table("cross", ["Corpus", "Label", "$\\alpha$", "95\\% CI", "AC1", "Raw", "Ceiling", "$\\alpha_{\\text{all}}$"],
          rows, "cross_<corpus>.json", "src/rubricon_field/cross_corpus.py", "@{}llrlrrlr@{}")


def main():
    cross_numbers()
    extra_numbers()
    sim_numbers()
    claim4_numbers()
    retro_numbers()
    judge_numbers()
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    tex = ["% Generated by scripts/paper_numbers.py -- do not edit. Ledger: CLAIMS.md"]
    for m, v, *_ in LEDGER:
        if m.startswith("table:"):
            continue   # table cells live in paper/tables/*.tex; they are ledgered, not macros
        tex.append(f"\\newcommand{{\\{m}}}{{{v}\\xspace}}")
    Path("paper/numbers.tex").write_text("\n".join(tex) + "\n")
    md = ["# CLAIMS.md -- claims ledger (generated by scripts/paper_numbers.py; do not edit)", "",
          f"Every number in paper/*.tex is one of these macros. Results as of commit `{commit}` "
          "(the commit this ledger was generated from; the results files are tracked in git).", "",
          "| macro | value | results file | JSON path / derivation | produced by |", "|---|---|---|---|---|"]
    for m, v, f, p, s in LEDGER:
        name = m if m.startswith("table:") else f"\\{m}"
        md.append(f"| `{name}` | {v} | `results/{f}` | `{p}` | `{s}` |")
    Path("CLAIMS.md").write_text("\n".join(md) + "\n")
    print(len(LEDGER), "macros")


if __name__ == "__main__":
    main()
