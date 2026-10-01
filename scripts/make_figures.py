"""Regenerate every paper figure from results/ (no figure is edited by hand).

Palette: the dataviz reference categorical slots 1-5, validated for CVD and
normal-vision separation on white (scripts/validate_palette.js). Three of the
five fall below 3:1 contrast on white, so every series also has its own marker
shape and a direct label or legend entry: identity never rests on colour alone,
and the figures stay legible in greyscale print.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

R = Path("results")
OUT = Path("paper/figures")
OUT.mkdir(parents=True, exist_ok=True)

C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
M = ["o", "s", "^", "D", "v"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e6e3"

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.5, "lines.linewidth": 1.5, "pdf.fonttype": 42,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
COL = 3.03   # ACL single-column width, inches
FULL = 6.3


def _save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)


# --------------------------------------------------------------------------
def fig_agreement():
    rows = [("HateXplain", "hatexplain", "binary"), ("MHS", "mhs", "binary"),
            ("Wikipedia Talk", "wikitalk", "toxicity_binary"), ("DICES-990", "dices990", "binary"),
            ("DICES-350", "dices350", "binary")]
    fig, ax = plt.subplots(figsize=(COL, 1.9))
    for y, (label, c, var) in enumerate(rows):
        v = next(x for x in json.load(open(R / f"cross_{c}.json"))["variants"] if x["variant"] == var)
        ci = v["alpha_ci95"]
        ax.plot([ci["lo"], ci["hi"]], [y, y], color=C[0], lw=2, solid_capstyle="round")
        ax.plot(v["alpha"], y, M[0], color=C[0], ms=5, zorder=3, label="Krippendorff's α" if y == 0 else None)
        ax.plot(v["gwet_ac1"], y, M[1], color=C[1], ms=5, zorder=3, mfc="white", mew=1.3,
                label="Gwet's AC1" if y == 0 else None)
        ax.plot(v["raw_pairwise_agreement"], y, M[2], color=C[2], ms=5, zorder=3, mfc="white", mew=1.3,
                label="raw agreement" if y == 0 else None)
    go = json.load(open(R / "cross_goemotions.json"))["variants"]
    al = sorted(x["alpha"] for x in go)
    y = len(rows)
    ax.plot([al[0], al[-1]], [y, y], color=C[0], lw=1, ls=(0, (2, 1.5)))
    ax.plot(statistics.median(al), y, M[0], color=C[0], ms=5, zorder=3)
    rows.append(("GoEmotions\n(28 emotions)", None, None))
    for t, s in ((0.667, "0.667"), (0.8, "0.800")):
        ax.axvline(t, color=INK2, lw=0.8, ls=":")
        ax.text(t, -0.75, s, ha="center", va="bottom", fontsize=6, color=INK2)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.set_xlim(0, 1)
    ax.set_xlabel("agreement (binary label)")
    ax.set_ylim(len(rows) - 0.5, -1.0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.28), ncol=3, frameon=False,
              handletextpad=0.2, columnspacing=0.8)
    _save(fig, "agreement")


# --------------------------------------------------------------------------
def fig_gate_operating_points():
    files = {"iid noise": "gate_sim_iid.jsonl", "hard items (specified first)": "gate_sim_het.jsonl",
             "hard items, post hoc (strong)": "gate_sim_het_k-0.10.jsonl"}
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 1.9), sharey=True)
    for ax, (title, f) in zip(axes, files.items()):
        cells = [json.loads(l) for l in open(R / f)]
        F = [c for c in cells if c["delta"] <= 0]
        Rz = [c for c in cells if c["delta"] >= 0.03]
        rF, rR = sum(c["reps"] for c in F), sum(c["reps"] for c in Rz)
        zs = list(cells[0]["z_threshold_counts"])
        xs = [sum(c["z_threshold_counts"][z] for c in F) / rF * 100 for z in zs]
        ys = [sum(c["z_threshold_counts"][z] for c in Rz) / rR * 100 for z in zs]
        ax.plot(xs, ys, "-", color=C[0], marker=M[0], ms=3, label="paired z-test, z* 1.96 to 4.5")
        for z, x, y in zip(zs, xs, ys):
            if z in ("1.96", "2.58", "3.29"):
                ax.annotate(f"z*={z}", (x, y), textcoords="offset points", xytext=(-6, 4), ha="right",
                            fontsize=6, color=INK2)
        v1 = (sum(c["n_gate_pub"] for c in F) / rF * 100, sum(c["n_gate_pub"] for c in Rz) / rR * 100)
        v2 = (sum(c["v2_n_pub"] for c in F) / rF * 100, sum(c["v2_n_pub"] for c in Rz) / rR * 100)
        ax.plot(*v1, M[1], color=C[1], ms=7, label="gate v1 (α floor)")
        ax.plot(*v2, M[2], color=C[2], ms=7, mfc="white", mew=1.5, label="gate v2")
        ax.set_title(title)
        ax.set_xlim(left=-0.05 * max(xs))
        ax.set_xlabel("false claims published (%)")
    axes[0].set_ylabel("resolvable claims\npublished (%)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False)
    _save(fig, "gate_operating_points")


def fig_alpha_floor_sweep():
    cells = [json.loads(l) for l in open(R / "gate_sim_iid.jsonl")]
    floors = list(cells[0]["v1_alpha_floor_sweep"])
    fig, axes = plt.subplots(1, 2, figsize=(COL, 1.5))
    for i, (ax, lab, sel) in enumerate(zip(axes, ("resolvable claims\n(true gap ≥ 0.03)", "false claims\n(true gap ≤ 0)"),
                                           (lambda c: c["delta"] >= 0.03, lambda c: c["delta"] <= 0))):
        cs = [c for c in cells if sel(c)]
        r = sum(c["reps"] for c in cs)
        ys = [sum(c["v1_alpha_floor_sweep"][f] for c in cs) / r * 100 for f in floors]
        ax.plot([float(f) for f in floors], ys, "-", marker=M[i], color=C[i], ms=4)
        ax.axvline(0.5, color=INK2, lw=0.8, ls=":")
        ax.set_title(lab)
        ax.set_xlabel("α floor")
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("published (%)")
    fig.tight_layout(w_pad=1.0)
    _save(fig, "alpha_floor_sweep")


# --------------------------------------------------------------------------
def fig_mde():
    import math
    from rubricon.gates.attenuation import eta_majority, mde_observed
    ns = [200 * 1.2 ** i for i in range(30)]
    ns = [n for n in ns if n <= 20000]
    fig, ax = plt.subplots(figsize=(COL, 1.9))
    for i, (eta, lab) in enumerate(((0.0, "perfect labels"), (0.071, "η=0.07 (Wikipedia Talk)"),
                                    (0.2, "η=0.20 (DICES)"))):
        att = 1 - 2 * eta_majority(eta, 3) if eta > 0 else 1.0
        ys = [mde_observed(int(n), math.sqrt(0.2)) / att for n in ns]
        ax.plot(ns, ys, color=C[i], marker=M[i], markevery=5, ms=4, label=lab)
    ax.axvline(1924, color=INK2, lw=0.8, ls=":")
    ax.text(1924 * 1.06, 0.106, "HateXplain\ntest n = 1,924", fontsize=6, color=INK2, va="top")
    ax.set_xscale("log")
    ax.set_xlabel("test items n")
    ax.set_ylabel("smallest resolvable true gap")
    ax.set_ylim(0, 0.11)
    ax.legend(frameon=False, title="3-rater gold; systems\ndisagree on 20% of items",
              title_fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=3, alignment="center", handlelength=1.5, columnspacing=0.8)
    _save(fig, "mde_vs_n")


# --------------------------------------------------------------------------
def fig_claim4():
    d = json.load(open(R / "claim4.json"))["corpora"]
    fig, axes = plt.subplots(1, 2, figsize=(FULL * 0.72, 1.8), sharey=True)
    for ax, (corpus, budget, title) in zip(axes, (("wikitalk", 3000, "Wikipedia Talk, B=3000"),
                                                  ("dices990", 900, "DICES-990, B=900"))):
        for i, dl in enumerate((0.02, 0.03, 0.05)):
            rs = sorted((r for r in d[corpus]["rows"] if r["model"] == "uniform" and r["budget"] == budget
                         and r["delta"] == dl), key=lambda r: r["k"])
            ax.plot([r["k"] for r in rs], [r["power"] for r in rs], "-", marker=M[i], color=C[i], ms=4,
                    label=f"true gap {dl}")
            ax.plot([r["k"] for r in rs], [r["predicted_power_iid"] for r in rs], ":", color=C[i], lw=1)
        ax.set_title(title)
        ax.set_xticks([1, 2, 3, 5])
        ax.set_xlabel("raters per item k (n = B/k)")
    axes[0].set_ylabel("power (solid: real raters;\ndotted: independent-noise model)")
    axes[1].legend(frameon=False, loc="upper right")
    _save(fig, "claim4_power")


# --------------------------------------------------------------------------
def fig_judges():
    p = R / "judge/analysis.json"
    if not p.exists():
        return
    a = json.load(open(p))
    corpora = [("hatexplain", "HateXplain"), ("mhs", "MHS"), ("wikitalk", "Wiki Talk"), ("dices350", "DICES-350")]
    judges = sorted({j for c, _ in corpora for j in a[c]["judges"]})
    fig, axes = plt.subplots(1, 4, figsize=(FULL, 1.9), sharex=True)
    for ax, (c, title) in zip(axes, corpora):
        h = a[c]["human"]
        ax.axvspan(h["m1_ci"][0], h["m1_ci"][1], color=GRID, zorder=0)
        ax.axvline(h["m1"], color=INK, lw=1)
        for y, j in enumerate(judges):
            e = a[c]["judges"].get(j)
            if not e:
                continue
            ax.plot(e["m1_ci"], [y, y], color=C[y % 5], lw=2)
            ax.plot(e["m1"], y, M[y % 5], color=C[y % 5], ms=5, zorder=3)
        ax.set_title(title)
        nice = {"claude-haiku-4-5": "Claude Haiku 4.5", "claude-sonnet-5-5": "Claude Sonnet 5.5",
                "gpt-4.1-mini": "GPT-4.1-mini", "gpt-oss-20b": "gpt-oss 20B", "qwen2.5-14b": "Qwen2.5 14B"}
        ax.set_yticks(range(len(judges)), [nice[j] for j in judges] if c == "hatexplain" else [""] * len(judges))
        ax.invert_yaxis()
        ax.set_xlim(0.55, 1.0)
        ax.set_xticks([0.6, 0.8, 1.0])
    fig.supxlabel("agreement with the panel majority (binary); black line and band: held-out human with 95% CI",
                  fontsize=7, y=-0.06)
    _save(fig, "judges")


if __name__ == "__main__":
    for f in (fig_agreement, fig_gate_operating_points, fig_alpha_floor_sweep, fig_mde, fig_claim4, fig_judges):
        f()
        print("ok", f.__name__)
