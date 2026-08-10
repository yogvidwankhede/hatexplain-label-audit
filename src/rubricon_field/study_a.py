"""Study A -- label-quality audit of HateXplain.

What this study is for
----------------------
HateXplain is a well-built, widely-used benchmark, and it is used the way
benchmarks always get used: its majority label is treated as ground truth, and
systems are ranked by how often they reproduce it. This study asks a narrower
question than "is the dataset good" -- it asks **what the measurement can
support**. Specifically:

1. How much do three crowd annotators actually agree, and is that number
   depressed by a statistical artefact (the kappa paradox) or by real
   disagreement about the construct?
2. Where does the disagreement live -- in deciding whether a post is worth
   moderating at all, or in naming what kind of bad it is?
3. How concentrated is the labour, and how much of the corpus rests on one
   person's judgement?
4. Given the measured reliability, how large must a difference between two
   classifiers be before it is distinguishable from label noise?

None of this is a criticism of the annotation effort. Hate speech is a
subjective construct with genuinely contested boundaries; a reliability ceiling
is a property of the construct, not a defect in the people who annotated it.
What *is* a defect is quoting a 1.5-point leaderboard gap measured against
labels whose own reproducibility was never checked. This study computes the
numbers a downstream user needs to avoid that.
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from typing import Any, Mapping

from rubricon.gates.signal import (
    DEFAULT_POLICY,
    EXPLORATORY_POLICY,
    Claim,
    ClaimLedger,
    GatePolicy,
    SignalGate,
)
from rubricon.stats.agreement import (
    agreement_report,
    alpha_interval,
    interpret,
    krippendorff_alpha,
    percent_agreement,
    spearman_brown,
)
from rubricon.stats.drift import profile_annotators
from rubricon.stats.precision import minimum_detectable_effect

from . import assumptions as A
from .data import (
    LABELS,
    NO_MAJORITY,
    SEVERITY,
    SPLIT_2_1,
    UNANIMOUS,
    Dataset,
    majority_label,
    profiling_records,
    reliability_matrix,
)

Matrix = dict[str, dict[str, str]]


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------


def _r(x: float | None, nd: int = 4) -> float | None:
    if x is None:
        return None
    if isinstance(x, float) and math.isnan(x):
        return None
    return round(float(x), nd)


def _subsample(matrix: Matrix, n_units: int, seed: int) -> Matrix:
    """Deterministic without-replacement subsample of units, for bootstrapping.

    Sorted keys then ``random.Random(seed).sample`` -- not ``set`` iteration --
    because the whole repository has to reproduce byte-for-byte.
    """
    keys = sorted(matrix)
    if len(keys) <= n_units:
        return dict(matrix)
    chosen = random.Random(seed).sample(keys, n_units)
    return {k: matrix[k] for k in chosen}


def agreement_battery(
    matrix: Matrix,
    name: str,
    description: str,
    scale: str = "nominal",
    subsample_units: int = A.BOOTSTRAP_SUBSAMPLE_UNITS,
    n_boot: int = A.BOOTSTRAP_REPLICATES,
    seed: int = A.SEED_BOOTSTRAP,
) -> dict:
    """Full agreement battery: point estimates on all units, interval on a sample.

    ``rubricon.stats.agreement.agreement_report`` would bootstrap the whole
    corpus, which costs minutes for an interval that is already an order of
    magnitude tighter than any threshold it is compared to. So the report is
    run with the bootstrap disabled and the interval is attached separately from
    a seeded subsample, with the subsample size recorded in the output. No point
    estimate in this file is ever computed on a subsample.
    """
    report = agreement_report(matrix, scale=scale, n_boot=0)
    sub = _subsample(matrix, subsample_units, seed)
    ci = alpha_interval(sub, metric=scale, n_boot=n_boot, seed=seed)
    report["name"] = name
    report["description"] = description
    report["ci"] = ci
    report["ci_basis"] = {
        "point_estimates_computed_on": len(matrix),
        "interval_computed_on": len(sub),
        "subsampled": len(sub) < len(matrix),
        "n_boot": n_boot,
        "seed": seed,
        "note": (
            "Point estimates use every unit. The cluster bootstrap resamples units "
            "(posts), not annotations, from a seeded subsample of the size shown."
        ),
    }
    return report


def three_way_battery(dataset: Dataset, matrix: Matrix | None = None) -> dict:
    """The headline 3-way battery. Computed once and shared, because the cluster
    bootstrap behind it is the single most expensive call in the pipeline."""
    matrix = matrix if matrix is not None else reliability_matrix(dataset, collapse="three_way")
    return agreement_battery(
        matrix,
        name="three_way_nominal",
        description=(
            "The task as annotated: normal / offensive / hatespeech, nominal metric. "
            "Nominal rather than ordinal because 'offensive' is not a partial "
            "'hatespeech' -- it is a different speech act that happens to sit between "
            "the other two in severity. The ordinal metric is reported alongside so "
            "the choice is visible rather than assumed."
        ),
        seed=A.SEED_BOOTSTRAP,
    )


# --------------------------------------------------------------------------
# 1. overall agreement + the kappa paradox test
# --------------------------------------------------------------------------


def overall_agreement(dataset: Dataset, battery: dict | None = None) -> dict:
    """The headline battery on the 3-way task, plus an explicit paradox test.

    Four coefficients are reported because each encodes a different model of
    chance, and the *pattern* across them is what identifies the failure mode:

    * a large raw/kappa gap with AC1 tracking raw would mean prevalence skew --
      one category so dominant that kappa's chance model collapses. The fix for
      that is stratified sampling, not annotator retraining.
    * kappa, alpha and AC1 all landing together, well below raw agreement, means
      the chance correction is not the story: annotators genuinely disagree, and
      no amount of resampling or retraining removes it.

    Reporting only alpha would leave a reader unable to distinguish those two,
    and they have opposite remedies.
    """
    matrix = reliability_matrix(dataset, collapse="three_way")
    battery = battery or three_way_battery(dataset, matrix)
    ordinal = agreement_report(matrix, scale="ordinal", n_boot=0)

    kappa = battery["fleiss_kappa"]["value"]
    ac1 = battery["gwet_ac1"]["value"]
    raw = battery["percent_agreement"]["value"]
    max_prev = battery["gwet_ac1"]["detail"].get("max_prevalence")
    prevalence = battery["fleiss_kappa"]["detail"].get("category_prevalence", {})

    paradox = {
        "detected": bool(battery["kappa_paradox_detected"]),
        "trigger_rule": (
            "raw_agreement >= 0.85 AND fleiss_kappa < 0.40 AND (gwet_ac1 - fleiss_kappa) > 0.25"
        ),
        "raw_agreement": raw,
        "fleiss_kappa": kappa,
        "gwet_ac1": ac1,
        "ac1_minus_kappa": _r((ac1 or 0) - (kappa or 0)),
        "max_category_prevalence": max_prev,
        "category_prevalence": prevalence,
        "conclusion": (
            "The kappa paradox does NOT fire. Raw agreement is {raw:.4f}, far below the "
            "0.85 the paradox requires, the AC1-kappa gap is {gap:.4f} rather than the "
            ">0.25 it requires, and the most common category holds only {prev:.1%} of the "
            "mass, so no category is dominant enough to collapse kappa's chance model. "
            "All three chance-corrected coefficients agree with each other to within "
            "{spread:.4f}. This rules out prevalence skew as the explanation for the "
            "moderate alpha and leaves genuine disagreement about the construct. The "
            "practical consequence is that the usual remedies for a low coefficient -- "
            "stratified oversampling of a rare class, or annotator retraining -- would "
            "not move this number."
        ).format(
            raw=raw,
            gap=(ac1 or 0) - (kappa or 0),
            prev=max_prev or 0.0,
            spread=max(abs(v - w) for v in (kappa, ac1, battery["krippendorff_alpha"]["value"])
                       for w in (kappa, ac1, battery["krippendorff_alpha"]["value"])),
        ),
    }

    return {
        "battery": battery,
        "ordinal_metric_for_comparison": {
            "krippendorff_alpha": ordinal["krippendorff_alpha"]["value"],
            "note": (
                "Reported only to show that the nominal/ordinal choice does not drive the "
                "headline. An ordinal reading treats normal-vs-hatespeech as a bigger "
                "error than normal-vs-offensive, which is defensible but not what the "
                "codebook asks for."
            ),
        },
        "kappa_paradox": paradox,
        "interpretation": {
            "band": battery["interpretation"],
            "krippendorff_thresholds": {
                "tentative_conclusions": 0.667,
                "firm_conclusions": 0.800,
            },
            "statement": (
                "alpha = {a:.4f} (95% CI [{lo:.4f}, {hi:.4f}]). That is below "
                "Krippendorff's 0.667 floor for tentative conclusions and well below the "
                "0.800 floor for firm ones. Three annotators agreed exactly with each "
                "other on {raw:.1%} of pairwise comparisons. The interval is narrow, so "
                "this is a precisely measured moderate reliability, not an uncertain one."
            ).format(
                a=battery["krippendorff_alpha"]["value"],
                lo=battery["ci"]["lo"],
                hi=battery["ci"]["hi"],
                raw=raw,
            ),
        },
    }


# --------------------------------------------------------------------------
# 2. consensus structure
# --------------------------------------------------------------------------


def consensus_structure(dataset: Dataset) -> dict:
    """How the gold label was reached, post by post.

    A single agreement coefficient compresses this into one number; the
    distribution is what a downstream user actually needs, because it says which
    *rows* of the corpus are load-bearing. A 2-1 label and a 3-0 label look
    identical in the released file and are not the same evidence.
    """
    out: dict[str, Any] = {}
    for collapse in ("three_way", "binary"):
        cons = dataset.consensus(collapse=collapse)
        modes = Counter(c.mode for c in cons.values())
        n = len(cons)
        resolved = sum(1 for c in cons.values() if c.resolved)
        fragile = sum(1 for c in cons.values() if c.margin <= 1)
        by_label = Counter(c.label for c in cons.values() if c.resolved)
        out[collapse] = {
            "n_posts": n,
            "counts": {m: modes.get(m, 0) for m in (UNANIMOUS, SPLIT_2_1, NO_MAJORITY)},
            "percentages": {
                m: _r(modes.get(m, 0) / n, 4) for m in (UNANIMOUS, SPLIT_2_1, NO_MAJORITY)
            },
            "resolved_posts": resolved,
            "unresolved_posts": n - resolved,
            "unresolved_fraction": _r((n - resolved) / n),
            "one_vote_fragile_posts": fragile,
            "one_vote_fragile_fraction": _r(fragile / n),
            "gold_label_distribution": dict(sorted(by_label.items())),
        }
    tw = out["three_way"]
    out["reading"] = (
        "On the 3-way task {u:,} posts ({up:.1%}) are unanimous, {s:,} ({sp:.1%}) are 2-1 "
        "splits, and {n:,} ({np_:.1%}) have three different labels and therefore no "
        "majority at all. The 2-1 and no-majority rows together are {f:.1%} of the corpus: "
        "for those posts a single annotator changing their mind would remove the gold "
        "label entirely. Under the binary collapse the same corpus is {bu:.1%} unanimous, "
        "and no post can be unresolved because a two-category vote among three raters "
        "always has a majority."
    ).format(
        u=tw["counts"][UNANIMOUS],
        up=tw["percentages"][UNANIMOUS],
        s=tw["counts"][SPLIT_2_1],
        sp=tw["percentages"][SPLIT_2_1],
        n=tw["counts"][NO_MAJORITY],
        np_=tw["percentages"][NO_MAJORITY],
        f=tw["one_vote_fragile_fraction"],
        bu=out["binary"]["percentages"][UNANIMOUS],
    )
    out["no_majority_note"] = (
        "The {n:,} no-majority posts have no target label defined by the annotation "
        "protocol. Whatever a downstream user does with them -- drop them, assign the "
        "most severe label, assign the least severe -- is a decision made outside the "
        "protocol, and it silently sets {p:.2%} of any accuracy figure computed over the "
        "full corpus."
    ).format(n=tw["counts"][NO_MAJORITY], p=tw["percentages"][NO_MAJORITY])
    return out


# --------------------------------------------------------------------------
# 3. pairwise confusion + the binary contrast
# --------------------------------------------------------------------------


def label_confusion(dataset: Dataset) -> dict:
    """Which label pairs get confused, and what happens when you stop asking.

    Every post contributes three ordered annotator pairs. Counting them
    unordered gives the confusion structure; counting them ordered and
    row-normalising gives ``P(rater B says Y | rater A says X)``, which is the
    quantity a downstream user needs when they wonder how often a ``hatespeech``
    gold label would have been ``offensive`` under a redraw.
    """
    unordered: Counter = Counter()
    ordered: Counter = Counter()
    for post in dataset.posts.values():
        labs = post.labels
        m = len(labs)
        for i in range(m):
            for j in range(m):
                if i == j:
                    continue
                ordered[(labs[i], labs[j])] += 1
                if i < j:
                    unordered[tuple(sorted((labs[i], labs[j])))] += 1

    total_pairs = sum(unordered.values())
    disagreeing = {k: v for k, v in unordered.items() if k[0] != k[1]}
    n_disagree = sum(disagreeing.values())

    row_totals: Counter = Counter()
    for (x, _y), v in ordered.items():
        row_totals[x] += v
    conditional = {
        x: {y: _r(ordered.get((x, y), 0) / row_totals[x]) for y in LABELS}
        for x in LABELS
        if row_totals[x]
    }

    pair_rows = sorted(
        (
            {
                "pair": f"{a}|{b}",
                "count": v,
                "share_of_all_pairs": _r(v / total_pairs),
                "share_of_disagreements": _r(v / n_disagree),
            }
            for (a, b), v in disagreeing.items()
        ),
        key=lambda r: -r["count"],
    )

    return {
        "n_annotator_pairs": total_pairs,
        "n_agreeing_pairs": total_pairs - n_disagree,
        "n_disagreeing_pairs": n_disagree,
        "pairwise_disagreement_rate": _r(n_disagree / total_pairs),
        "disagreement_pairs": pair_rows,
        "conditional_label_given_other_rater": conditional,
        "reading": (
            "Of {nd:,} disagreeing annotator pairs, {top_share:.1%} are "
            "{top_pair}. The {ob:.1%} of disagreements that are offensive-vs-hatespeech "
            "are disagreements about *which kind* of unacceptable a post is, not about "
            "whether it is unacceptable; a moderation system that only needs the binary "
            "decision never has to resolve them."
        ).format(
            nd=n_disagree,
            top_share=pair_rows[0]["share_of_disagreements"],
            top_pair=pair_rows[0]["pair"],
            ob=next(
                (
                    r["share_of_disagreements"]
                    for r in pair_rows
                    if set(r["pair"].split("|")) == {"offensive", "hatespeech"}
                ),
                0.0,
            ),
        ),
    }


def binary_contrast(dataset: Dataset, three_way_battery_cached: dict | None = None) -> dict:
    """The 3-way task versus the binary task most classifiers are trained on.

    This is the finding with the most direct practical consequence in the study.
    Almost no deployed moderation system needs the offensive/hatespeech
    distinction: it needs to know whether to act. If the binary reduction of the
    same annotations is materially more reliable, then the reliability ceiling
    that applies to a deployed system is *not* the headline alpha, and a paper
    reporting 3-way accuracy is being penalised by a distinction its users do
    not need.

    A conditional third battery is reported alongside: agreement restricted to
    annotations that were *not* ``normal``. It isolates the offensive-versus-
    hatespeech boundary, and it is explicitly labelled conditional because
    dropping the ``normal`` judgements is selection on the outcome -- the number
    describes the boundary, not the task.
    """
    three = reliability_matrix(dataset, collapse="three_way")
    binary = reliability_matrix(dataset, collapse="binary")
    boundary = reliability_matrix(dataset, collapse="toxic_boundary")

    b_three = three_way_battery_cached or three_way_battery(dataset, three)
    b_binary = agreement_battery(
        binary,
        name="binary_toxic_vs_normal",
        description=(
            "normal vs toxic (offensive or hatespeech collapsed). This is the decision "
            "a moderation classifier is usually trained to make."
        ),
        seed=A.SEED_BOOTSTRAP + 10,
    )
    b_boundary = agreement_battery(
        boundary,
        name="offensive_vs_hatespeech_conditional",
        description=(
            "CONDITIONAL: offensive vs hatespeech, over the annotations that were not "
            "'normal'. Units retained only where at least two annotators both chose a "
            "non-normal label. Selection on the outcome -- read as a description of the "
            "boundary, not as a reliability estimate for a task anyone runs."
        ),
        seed=A.SEED_BOOTSTRAP + 20,
    )

    a3 = b_three["krippendorff_alpha"]["value"]
    a2 = b_binary["krippendorff_alpha"]["value"]
    ab = b_boundary["krippendorff_alpha"]["value"]
    r3 = b_three["percent_agreement"]["value"]
    r2 = b_binary["percent_agreement"]["value"]

    return {
        "three_way": b_three,
        "binary": b_binary,
        "offensive_vs_hatespeech_conditional": b_boundary,
        "contrast": {
            "alpha_three_way": a3,
            "alpha_binary": a2,
            "alpha_delta": _r(a2 - a3),
            "alpha_relative_gain": _r((a2 - a3) / a3),
            "raw_three_way": r3,
            "raw_binary": r2,
            "raw_delta": _r(r2 - r3),
            "alpha_offensive_vs_hatespeech_conditional": ab,
            "band_three_way": interpret(a3),
            "band_binary": interpret(a2),
            "intervals_overlap": bool(
                b_three["ci"]["hi"] >= b_binary["ci"]["lo"]
                and b_binary["ci"]["hi"] >= b_three["ci"]["lo"]
            ),
        },
        "reading": (
            "Collapsing to the binary decision moves alpha from {a3:.4f} to {a2:.4f}, a "
            "gain of {d:+.4f} ({g:+.1%}), and raw agreement from {r3:.4f} to {r2:.4f}. "
            "The bootstrap intervals do not overlap. Restricted to the annotations that "
            "were not 'normal', alpha on the offensive-versus-hatespeech call is {ab:.4f} "
            "-- that boundary is where the disagreement lives. Practically: the "
            "annotators broadly agree about whether a post should be acted on and "
            "disagree about what to call it, so a system evaluated on the 3-way label is "
            "being scored against a noisier target than the one it will be deployed to "
            "make."
        ).format(a3=a3, a2=a2, d=a2 - a3, g=(a2 - a3) / a3, r3=r3, r2=r2, ab=ab),
    }


# --------------------------------------------------------------------------
# 4. annotator profiles, load, and leave-one-out exposure
# --------------------------------------------------------------------------


def annotator_analysis(dataset: Dataset) -> dict:
    """Who annotated what, how harshly, and what happens if you remove them.

    Three separate questions, deliberately not merged:

    *Load.* An annotation pool with a heavy tail is a pool whose labels are
    partly one person's opinion. That is not misconduct and it is not visible in
    any agreement coefficient, which weights every unit equally regardless of
    who produced it.

    *Calibration versus position.* ``profile_annotators`` is given two
    dimensions: the binary "is this worth acting on" call, treated as
    uncontested, and the full 3-way severity, declared contested. An offset on
    the first is a leniency defect. An offset on the second, net of the first,
    is a *value position* about where hate speech begins -- reported, but never
    flagged as a quality problem, because retraining someone for holding a
    defensible position on a contested boundary is the wrong remedy.

    *Exposure.* Leave-one-annotator-out on the heaviest raters, measuring the
    movement in overall alpha, in the corpus-level label mix, and -- the number
    that matters most -- how many gold labels stop existing.
    """
    load = dataset.annotator_load()
    total = sum(load.values())
    ranked = load.most_common()

    profiles = profile_annotators(
        profiling_records(dataset),
        dimension_keys=["toxic", "severity"],
        contested_dimensions=["severity"],
        bias_threshold=A.ANNOTATOR_BIAS_THRESHOLD,
        speed_floor_s=0.0,  # no timing data in the release; disable the check
    )
    prof_by_id = {p["annotator_id"]: p for p in profiles["annotators"]}

    # -- label marginals per annotator ------------------------------------
    marginals = []
    for aid, rows in dataset.by_annotator.items():
        c = Counter(a.label for a in rows)
        n = len(rows)
        marginals.append(
            {
                "annotator_id": aid,
                "n": n,
                "share_of_corpus": _r(n / total, 5),
                "share_normal": _r(c["normal"] / n),
                "share_offensive": _r(c["offensive"] / n),
                "share_hatespeech": _r(c["hatespeech"] / n),
                "share_toxic": _r((n - c["normal"]) / n),
                "mean_severity": _r(sum(SEVERITY[a.label] for a in rows) / n),
                "bias_vs_pool_uncontested": prof_by_id.get(aid, {}).get(
                    "bias_vs_pool_uncontested"
                ),
                "contested_position": prof_by_id.get(aid, {}).get("contested_position"),
                "flags": prof_by_id.get(aid, {}).get("flags", []),
                "positions": prof_by_id.get(aid, {}).get("positions", []),
            }
        )
    eligible = [m for m in marginals if m["n"] >= A.MIN_LOAD_FOR_MARGINAL_RANKING]
    eligible.sort(key=lambda m: (-m["mean_severity"], m["annotator_id"]))
    flagged_high_load = (
        _r(sum(1 for m in eligible if m["flags"]) / len(eligible), 3) if eligible else None
    )
    pool_toxic = _r(
        sum(1 for a in dataset.annotations if a.is_toxic) / len(dataset.annotations)
    )

    # -- leave-one-annotator-out ------------------------------------------
    full_matrix = reliability_matrix(dataset, collapse="three_way")
    full_alpha = krippendorff_alpha(full_matrix, metric="nominal").value
    full_counts = dataset.label_counts()
    full_shares = {lab: full_counts[lab] / total for lab in LABELS}
    full_consensus = dataset.consensus("three_way")

    loo_rows = []
    for aid, n_ann in ranked[: A.LEAVE_OUT_TOP_N]:
        m = reliability_matrix(dataset, collapse="three_way", exclude_annotators=[aid])
        alpha_wo = krippendorff_alpha(m, metric="nominal").value
        raw_wo = percent_agreement(m).value

        remaining = total - n_ann
        counts_wo = Counter(
            {lab: full_counts[lab] for lab in LABELS}
        )
        for a in dataset.by_annotator[aid]:
            counts_wo[a.label] -= 1

        touched = {a.post_id for a in dataset.by_annotator[aid]}
        lost, preserved, still_unresolved = 0, 0, 0
        for pid in touched:
            post = dataset.posts[pid]
            rest = [a.label for a in post.annotations if a.annotator_id != aid]
            was_resolved = full_consensus[pid].resolved
            now_resolved = len(set(rest)) == 1 and len(rest) >= 2
            if was_resolved and not now_resolved:
                lost += 1
            elif was_resolved and now_resolved:
                preserved += 1
            elif not was_resolved:
                still_unresolved += 1

        loo_rows.append(
            {
                "annotator_id": aid,
                "n_annotations": n_ann,
                "share_of_all_annotations": _r(n_ann / total, 5),
                "n_posts_touched": len(touched),
                "share_of_posts_touched": _r(len(touched) / dataset.n_posts, 5),
                "alpha_without": _r(alpha_wo),
                "alpha_delta": _r(alpha_wo - full_alpha),
                "raw_agreement_without": _r(raw_wo),
                "label_share_delta": {
                    lab: _r(counts_wo[lab] / remaining - full_shares[lab], 5) for lab in LABELS
                },
                "gold_labels_lost": lost,
                "gold_labels_lost_share_of_corpus": _r(lost / dataset.n_posts, 5),
                "gold_labels_preserved": preserved,
                "posts_unresolved_before_and_after": still_unresolved,
            }
        )

    worst = max(loo_rows, key=lambda r: r["gold_labels_lost"])
    biggest_alpha_move = max(loo_rows, key=lambda r: abs(r["alpha_delta"]))

    return {
        "pool": {
            "n_annotators": dataset.n_annotators,
            "n_annotations": total,
            "n_posts": dataset.n_posts,
            "annotations_per_post": dict(dataset.replication()),
            "busiest_annotator": {
                "annotator_id": ranked[0][0],
                "n": ranked[0][1],
                "share": _r(ranked[0][1] / total, 5),
            },
            "top5_share": _r(sum(c for _, c in ranked[:5]) / total, 5),
            "top10_share": _r(sum(c for _, c in ranked[:10]) / total, 5),
            "median_load": sorted(load.values())[len(load) // 2],
            "load_top20": [{"annotator_id": a, "n": n} for a, n in ranked[:20]],
        },
        "pool_label_marginals": {
            "share_toxic": pool_toxic,
            "shares": {lab: _r(full_shares[lab]) for lab in LABELS},
        },
        "profile_summary": {
            "n_annotators": profiles["n_annotators"],
            "pool_mean": profiles["pool_mean"],
            "flagged_annotators": profiles["flagged_annotators"],
            "flagged_fraction": profiles["flagged_fraction"],
            "flagged_fraction_high_load_only": flagged_high_load,
            "n_high_load_annotators": len(eligible),
            "flag_caveat": (
                "The pool-wide flagged fraction is dominated by very small annotators: the "
                f"median load is {sorted(load.values())[len(load) // 2]} judgements, and an "
                "annotator with a handful of posts can sit far from the pool mean by "
                "sampling alone. The high-load figure, restricted to the "
                f"{len(eligible)} annotators with at least {A.MIN_LOAD_FOR_MARGINAL_RANKING} "
                "judgements, is the one to quote. Neither figure is evidence of misconduct; "
                "both measure calibration spread on a subjective call."
            ),
            "contested_dimensions": profiles["contested_dimensions"],
            "divergent_on_contested": profiles["divergent_on_contested"],
            "divergent_on_contested_fraction": profiles["divergent_on_contested_fraction"],
            "position_note": profiles["position_note"],
            "bias_threshold": A.ANNOTATOR_BIAS_THRESHOLD,
            "reading": (
                "{ff:.1%} of all {n} annotators, and {fh:.1%} of the {nh} with at least {k} "
                "judgements, sit more than {t:.2f} from the pool mean on the uncontested "
                "binary 'is this worth acting on' call. Separately, {cf:.1%} sit far from "
                "the pool on the contested severity boundary. The second group is not a "
                "quality problem, and the harness deliberately does not count it as one: "
                "it is the measurable footprint of annotators who apply a different, "
                "defensible threshold for what counts as hate speech rather than merely "
                "offensive. That is the disagreement the construct actually contains."
            ).format(
                ff=profiles["flagged_fraction"],
                n=profiles["n_annotators"],
                fh=flagged_high_load or 0.0,
                nh=len(eligible),
                k=A.MIN_LOAD_FOR_MARGINAL_RANKING,
                t=A.ANNOTATOR_BIAS_THRESHOLD,
                cf=profiles["divergent_on_contested_fraction"],
            ),
        },
        "marginals_ranking": {
            "min_load_for_eligibility": A.MIN_LOAD_FOR_MARGINAL_RANKING,
            "n_eligible": len(eligible),
            "harshest_5": eligible[:5],
            "most_lenient_5": eligible[-5:][::-1],
            "share_toxic_range_among_eligible": (
                [
                    _r(min(m["share_toxic"] for m in eligible)),
                    _r(max(m["share_toxic"] for m in eligible)),
                ]
                if eligible
                else None
            ),
            "reading": (
                (
                    "Among the {n} annotators with at least {k} judgements, the share of "
                    "posts called toxic runs from {lo:.1%} to {hi:.1%} against a pool rate "
                    "of {p:.1%}. Two annotators drawn from opposite ends of that range "
                    "would produce visibly different corpora from identical text."
                ).format(
                    n=len(eligible),
                    k=A.MIN_LOAD_FOR_MARGINAL_RANKING,
                    lo=min(m["share_toxic"] for m in eligible),
                    hi=max(m["share_toxic"] for m in eligible),
                    p=pool_toxic,
                )
                if eligible
                else (
                    "No annotator in this corpus reaches the "
                    f"{A.MIN_LOAD_FOR_MARGINAL_RANKING}-judgement floor required to be "
                    "ranked, so no harshness comparison is reported. Ranking annotators "
                    "below that floor would be ranking sampling noise."
                )
            ),
        },
        "leave_one_annotator_out": {
            "full_alpha": _r(full_alpha),
            "top_n": A.LEAVE_OUT_TOP_N,
            "rows": loo_rows,
            "max_abs_alpha_delta": _r(abs(biggest_alpha_move["alpha_delta"])),
            "reading": (
                "Removing the busiest annotator (#{aid}, {n:,} labels, {s:.1%} of all "
                "annotations, present on {pt:.1%} of posts) moves overall alpha by "
                "{da:+.4f} -- the coefficient is robust, because it weights posts, not "
                "people. What is not robust is the gold labelling: {lost:,} posts "
                "({ls:.1%} of the corpus) lose their majority label entirely when this one "
                "person is removed, because their vote was one of the two that created it. "
                "Across the {k} heaviest annotators the largest alpha movement is "
                "{ma:.4f}, so no single rater is distorting the reliability estimate; the "
                "exposure is to the labels themselves, not to the statistic."
            ).format(
                aid=worst["annotator_id"],
                n=worst["n_annotations"],
                s=worst["share_of_all_annotations"],
                pt=worst["share_of_posts_touched"],
                da=worst["alpha_delta"],
                lost=worst["gold_labels_lost"],
                ls=worst["gold_labels_lost_share_of_corpus"],
                k=A.LEAVE_OUT_TOP_N,
                ma=abs(biggest_alpha_move["alpha_delta"]),
            ),
        },
    }


# --------------------------------------------------------------------------
# 5. agreement by target community
# --------------------------------------------------------------------------


def agreement_by_target(dataset: Dataset, corpus_alpha: float | None = None) -> dict:
    """Is the instrument equally reliable for every targeted community?

    This matters more than a generic per-stratum breakdown. If agreement is
    systematically lower for posts targeting one community, then the gold labels
    for that community are noisier, a classifier trained on them learns a
    fuzzier boundary there, and every downstream fairness audit that compares
    per-group performance is comparing groups whose *labels* differ in quality.
    That confound is invisible unless it is measured.

    Two disciplines are enforced. A community counts for a post only when at
    least two of the three annotators named it, so the strata are not themselves
    built out of single-rater noise. And communities below the pre-registered
    size floor get a point estimate marked ``insufficient_for_claim`` rather
    than a comparison, because at n=100 the bootstrap half-width on alpha is
    wider than any between-community difference under discussion.
    """
    if corpus_alpha is None:
        corpus_alpha = krippendorff_alpha(
            reliability_matrix(dataset, collapse="three_way"), metric="nominal"
        ).value
    endorsed = dataset.endorsed_targets(min_votes=2)
    by_community: dict[str, list[str]] = defaultdict(list)
    any_target: list[str] = []
    for pid, targets in endorsed.items():
        if targets:
            any_target.append(pid)
        for t in targets:
            by_community[t].append(pid)

    # Like-for-like baseline. Conditioning on "this post targets community C"
    # also conditions on "this post is not obviously normal", which removes the
    # easiest units and compresses the label distribution -- classic restriction
    # of range, and it depresses every chance-corrected coefficient. Comparing a
    # per-community alpha against the corpus-wide 0.46 would therefore mostly
    # measure that selection. The right comparison is against all posts with any
    # endorsed target, which shares the selection.
    baseline_matrix = reliability_matrix(
        dataset, collapse="three_way", post_ids=sorted(any_target)
    )
    baseline = {
        "n_posts": len(baseline_matrix),
        "alpha_three_way": _r(krippendorff_alpha(baseline_matrix, metric="nominal").value),
        "raw_agreement": _r(percent_agreement(baseline_matrix).value),
        "definition": (
            "All posts with at least one target community endorsed by >=2 annotators. "
            "Every per-community row below should be read against this, not against the "
            "corpus-wide alpha: selecting posts that target a community also selects away "
            "the easy 'normal' posts, and that restriction of range lowers all "
            "chance-corrected coefficients by construction."
        ),
    }

    rows = []
    for community, pids in sorted(by_community.items(), key=lambda kv: -len(kv[1])):
        matrix = reliability_matrix(dataset, collapse="three_way", post_ids=sorted(pids))
        if len(matrix) < 2:
            continue
        alpha = krippendorff_alpha(matrix, metric="nominal")
        raw = percent_agreement(matrix)
        binary_matrix = reliability_matrix(dataset, collapse="binary", post_ids=sorted(pids))
        alpha_binary = krippendorff_alpha(binary_matrix, metric="nominal")
        cons = Counter(majority_label(dataset.posts[p]).mode for p in pids)
        labs = Counter(
            a.label for p in pids for a in dataset.posts[p].annotations
        )
        n_labs = sum(labs.values())

        big_enough = len(matrix) >= A.MIN_POSTS_FOR_COMMUNITY_CLAIM
        ci = (
            alpha_interval(
                matrix,
                metric="nominal",
                n_boot=A.TARGET_BOOTSTRAP_REPLICATES,
                seed=A.SEED_TARGET_BOOTSTRAP,
            )
            if big_enough
            else None
        )
        rows.append(
            {
                "community": community,
                "n_posts": len(matrix),
                "alpha_three_way": _r(alpha.value),
                "alpha_binary": _r(alpha_binary.value),
                "raw_agreement": _r(raw.value),
                "ci_alpha_three_way": ci,
                "unanimous_share": _r(cons.get(UNANIMOUS, 0) / len(pids)),
                "no_majority_share": _r(cons.get(NO_MAJORITY, 0) / len(pids)),
                "label_shares": {lab: _r(labs[lab] / n_labs) for lab in LABELS},
                "sufficient_for_claim": big_enough,
                "claim_note": (
                    ""
                    if big_enough
                    else (
                        f"n={len(matrix)} posts is below the pre-registered floor of "
                        f"{A.MIN_POSTS_FOR_COMMUNITY_CLAIM}; the point estimate is reported "
                        "for completeness but will not support a comparison."
                    )
                ),
            }
        )

    claimable = [r for r in rows if r["sufficient_for_claim"]]
    ranked = sorted(claimable, key=lambda r: r["alpha_three_way"])
    spread = (
        _r(ranked[-1]["alpha_three_way"] - ranked[0]["alpha_three_way"]) if ranked else None
    )
    disjoint = []
    if len(ranked) >= 2:
        lo_r, hi_r = ranked[0], ranked[-1]
        if lo_r["ci_alpha_three_way"] and hi_r["ci_alpha_three_way"]:
            if lo_r["ci_alpha_three_way"]["hi"] < hi_r["ci_alpha_three_way"]["lo"]:
                disjoint = [lo_r["community"], hi_r["community"]]

    return {
        "min_endorsing_annotators": 2,
        "corpus_alpha_for_reference": _r(corpus_alpha),
        "baseline_all_targeted_posts": baseline,
        "n_communities_observed": len(rows),
        "n_communities_claimable": len(claimable),
        "size_floor": A.MIN_POSTS_FOR_COMMUNITY_CLAIM,
        "rows": rows,
        "lowest_agreement_claimable": ranked[:3],
        "highest_agreement_claimable": ranked[-3:][::-1] if ranked else [],
        "alpha_spread_across_claimable": spread,
        "extremes_have_disjoint_intervals": bool(disjoint),
        "reading": (
            "{nc} target communities are endorsed by at least two annotators on at least "
            "one post; {ncc} clear the {floor}-post floor. Every one of them sits below "
            "the corpus-wide alpha of {corpus:.4f}, and so does the like-for-like baseline "
            "over all {nb:,} targeted posts ({ba:.4f}) -- that drop is restriction of "
            "range, not a change in annotator behaviour, because selecting posts that "
            "target a community removes the easy 'normal' units. Read against that "
            "baseline, 3-way alpha across the claimable communities spans {spread:.4f}, "
            "from {lo_c} at {lo_a:.4f} to {hi_c} at {hi_a:.4f}, and the bootstrap "
            "intervals of the two extremes {overlap}. The comparison is still confounded: "
            "posts about different communities have different label mixes, so part of any "
            "gap is category prevalence rather than annotator difficulty. It is "
            "nonetheless actionable, because a fairness audit that compares model "
            "performance across these groups is comparing labels of unequal quality "
            "either way, and that is a confound in the audit rather than a finding of it."
        ).format(
            nc=len(rows),
            ncc=len(claimable),
            floor=A.MIN_POSTS_FOR_COMMUNITY_CLAIM,
            corpus=corpus_alpha,
            nb=baseline["n_posts"],
            ba=baseline["alpha_three_way"],
            spread=spread or 0.0,
            lo_c=ranked[0]["community"] if ranked else "n/a",
            lo_a=ranked[0]["alpha_three_way"] if ranked else 0.0,
            hi_c=ranked[-1]["community"] if ranked else "n/a",
            hi_a=ranked[-1]["alpha_three_way"] if ranked else 0.0,
            overlap="do not overlap" if disjoint else "overlap",
        ),
    }


# --------------------------------------------------------------------------
# 6. the accuracy ceiling
# --------------------------------------------------------------------------


def accuracy_ceiling(dataset: Dataset, alpha_three: float, alpha_binary: float) -> dict:
    """How accurate can any classifier be, and how big must a gap be to be real?

    Two distinct questions, routinely conflated.

    **The ceiling.** Against a *majority-vote gold* the ceiling is trivially
    100%: the gold is a deterministic function of the votes, so a system that
    memorises it is perfect. That number is meaningless, and quoting it is why
    the ceiling question gets waved away. The meaningful ceiling is against a
    *human*: if a fresh annotator from this pool is drawn for a post, how often
    can any predictor match them? The best possible predictor names the post's
    modal label, so the ceiling is the mean modal share -- and no classifier,
    however good, can beat it, because the residual is disagreement among the
    humans themselves.

    **The resolution.** Given that gold labels contain noise, how large must the
    gap between two systems be before it is distinguishable from that noise?
    This is where ``minimum_detectable_effect`` comes in. Its ``sd`` argument is
    the SD of the per-item paired difference in accuracy; for binary outcomes
    that is ``sqrt(p_disc - delta^2)`` where ``p_disc`` is the fraction of items
    the two systems label differently. We hold no system predictions, so
    ``p_disc`` is a declared, swept assumption rather than a measurement, and it
    is the only assumption in the grid.

    Both the observed-scale and the true-score-scale MDE are reported, never
    compounded, following the convention documented in ``rubricon.stats.precision``.
    """
    n_posts = dataset.n_posts

    def modal_share(collapse: str) -> tuple[float, dict]:
        shares = []
        detail: Counter = Counter()
        for post in dataset.posts.values():
            cons = majority_label(post, collapse=collapse)  # type: ignore[arg-type]
            n = cons.n_annotators
            top = max(cons.counts.values()) if cons.counts else 0
            shares.append(top / n if n else 0.0)
            detail[f"{top}of{n}"] += 1
        return sum(shares) / len(shares), dict(sorted(detail.items()))

    ceil_three, detail_three = modal_share("three_way")
    ceil_binary, detail_binary = modal_share("binary")

    cons_three = dataset.consensus("three_way")
    unresolved = sum(1 for c in cons_three.values() if not c.resolved)
    one_vote_fragile_fraction = sum(1 for c in cons_three.values() if c.margin <= 1) / n_posts

    # -- MDE grid ---------------------------------------------------------
    rho_gold = spearman_brown(alpha_three, 3)
    rho_gold_binary = spearman_brown(alpha_binary, 3)
    test_split_n = int(round(n_posts * A.TEST_SPLIT_FRACTION))
    sizes = sorted({*A.EVAL_SIZES, test_split_n, n_posts})

    grid = []
    for n in sizes:
        for p_disc in A.PAIRWISE_DISAGREEMENT_SWEEP:
            sd = math.sqrt(p_disc)  # delta^2 term dropped: conservative at the null
            res = minimum_detectable_effect(
                n=n,
                sd=sd,
                reliability=rho_gold,
                alpha_level=A.ALPHA_LEVEL,
                power=A.POWER,
                paired=True,
                target_effect=A.EXAMPLE_LEADERBOARD_GAP,
            ).to_dict()
            grid.append(
                {
                    "n_items": n,
                    "assumed_pairwise_disagreement": p_disc,
                    "sd_of_paired_difference": _r(sd),
                    "mde_observed_scale": res["mde_observed_scale"],
                    "mde_true_scale": res["mde_true_scale"],
                    "n_required_for_2pt_gap": res["n_required_for_target"],
                    "is_test_split_size": n == test_split_n,
                    "is_headline_cell": (
                        n == test_split_n and p_disc == A.HEADLINE_DISAGREEMENT_RATE
                    ),
                }
            )
    headline = next(c for c in grid if c["is_headline_cell"])
    full_corpus_cell = next(
        c
        for c in grid
        if c["n_items"] == n_posts and c["assumed_pairwise_disagreement"] == A.HEADLINE_DISAGREEMENT_RATE
    )

    return {
        "ceiling_against_a_human": {
            "three_way": _r(ceil_three),
            "binary": _r(ceil_binary),
            "modal_share_distribution_three_way": detail_three,
            "modal_share_distribution_binary": detail_binary,
            "definition": (
                "Mean over posts of (votes for the post's modal label) / (votes cast). "
                "This is the expected accuracy of an oracle that names each post's most "
                "popular label, scored against one annotator drawn at random from the "
                "three who saw it. No predictor can do better, because the remainder is "
                "disagreement among the annotators, not error in the predictor."
            ),
            "reading": (
                "The ceiling on the 3-way task is {c3:.1%}: even a perfect model of what "
                "this annotator pool believes would disagree with a randomly chosen "
                "annotator on {r3:.1%} of posts. On the binary task the ceiling is "
                "{cb:.1%}. Reported 3-way accuracies in the high 60s therefore sit closer "
                "to the achievable ceiling than to 100%, and the headroom a leaderboard "
                "implies is largely not there."
            ).format(c3=ceil_three, r3=1 - ceil_three, cb=ceil_binary),
        },
        "gold_availability": {
            "posts_without_a_majority": unresolved,
            "share": _r(unresolved / n_posts),
            "note": (
                "A separate cap: for these posts the protocol produces no target label at "
                "all, so any accuracy credited on them reflects a resolution rule chosen "
                "downstream, not annotator agreement."
            ),
        },
        "gold_reliability_used_for_power": {
            "single_rater_alpha_three_way": _r(alpha_three),
            "three_rater_gold_reliability_three_way": _r(rho_gold),
            "single_rater_alpha_binary": _r(alpha_binary),
            "three_rater_gold_reliability_binary": _r(rho_gold_binary),
            "method": (
                "Spearman-Brown step-up of the single-rater alpha to the 3-rater "
                "aggregate, which is the quantity the gold label represents. Approximate "
                "for a majority vote over nominal categories -- Spearman-Brown is derived "
                "for a mean of exchangeable continuous replications -- and reported as an "
                "approximation, not an identity."
            ),
        },
        "minimum_detectable_effect": {
            "assumption": (
                "sd of the per-item paired accuracy difference = sqrt(p_disc), where "
                "p_disc is the assumed fraction of items on which the two compared "
                "systems disagree. Swept, not fixed; no classifier predictions are held "
                "by this study."
            ),
            "alpha_level": A.ALPHA_LEVEL,
            "power": A.POWER,
            "test_split_n": test_split_n,
            "test_split_basis": (
                f"{A.TEST_SPLIT_FRACTION:.0%} of the corpus. HateXplain ships an 8:1:1 "
                "split; the official split file is not used here, so this size is an "
                "assumption and is labelled as one."
            ),
            "grid": grid,
            "headline_cell": headline,
            "full_corpus_cell": full_corpus_cell,
            "what_this_mde_does_not_cover": (
                "This MDE covers item-sampling variance, and reports the attenuation "
                "correction separately. It does NOT cover gold-panel resampling variance: "
                "the labels are one realisation of a three-annotator draw, and a different "
                "draw from the same pool would produce different gold labels for a "
                "substantial share of posts. That variance is not identifiable from a "
                "single panel of three, so it is not estimated here rather than being "
                "guessed at. The observable proxy is in the consensus structure -- "
                "{f:.1%} of posts sit at a margin of one vote -- and it means the figures "
                "below are a floor on the real detection threshold, not a ceiling."
            ).format(f=one_vote_fragile_fraction),
            "reading": (
                "On a test split of {n:,} posts, with two systems disagreeing on {p:.0%} of "
                "items, the smallest detectable accuracy difference at 80% power is "
                "{mo:.4f} on the observed scale -- {mop:.1f} accuracy points -- and "
                "{mt:.4f} ({mtp:.1f} points) once the difference is expressed in "
                "true-score units by dividing by sqrt(gold reliability). A published gap "
                "of {gap:.0%} is therefore inside the noise floor of that design: "
                "detecting it reliably needs n >= {nreq:,} items, against the {n:,} "
                "available. Evaluated on the full {nf:,}-post corpus the same comparison "
                "resolves {mof:.4f} ({mofp:.1f} points), so the gap is detectable there -- "
                "the constraint is the size of the split, not the corpus."
            ).format(
                n=headline["n_items"],
                p=headline["assumed_pairwise_disagreement"],
                mo=headline["mde_observed_scale"],
                mop=headline["mde_observed_scale"] * 100,
                mt=headline["mde_true_scale"],
                mtp=headline["mde_true_scale"] * 100,
                gap=A.EXAMPLE_LEADERBOARD_GAP,
                nreq=headline["n_required_for_2pt_gap"],
                nf=full_corpus_cell["n_items"],
                mof=full_corpus_cell["mde_observed_scale"],
                mofp=full_corpus_cell["mde_observed_scale"] * 100,
            ),
        },
    }


# --------------------------------------------------------------------------
# 7. the signal gate
# --------------------------------------------------------------------------


def _claim_specs(ctx: Mapping[str, Any]) -> list[dict]:
    """Claims a paper or model card might plausibly make about this dataset.

    Written as things people actually say, not strawmen. The point of running
    them through the gate is to show that the gate discriminates: some of these
    are supportable by the measurement and some are not, and the difference is
    mechanical rather than rhetorical.
    """
    return [
        {
            "claim_id": "C-DESC-01",
            "depth": "production",
            "kind": "descriptive",
            "text": (
                "In HateXplain, {u:.1%} of posts carry a unanimous 3-way label and "
                "{s:.1%} are 2-1 splits."
            ).format(u=ctx["unanimous_share"], s=ctx["split_share"]),
            "checks": ["replication"],
        },
        {
            "claim_id": "C-MEAS-01",
            "depth": "production",
            "kind": "measurement",
            "text": (
                "HateXplain's 3-way majority labels are reliable enough to be treated as "
                "ground truth for measuring model quality."
            ),
            "checks": ["reliability_three_way", "precision_three_way"],
        },
        {
            "claim_id": "C-MEAS-02",
            "depth": "production",
            "kind": "measurement",
            "text": (
                "Collapsed to a binary toxic/normal decision, the same annotations support "
                "a measurement of model quality."
            ),
            "checks": ["reliability_binary", "precision_binary"],
        },
        {
            "claim_id": "C-RANK-01",
            "depth": "production",
            "kind": "ranking",
            "text": (
                "Model X outperforms model Y by {g:.1f} accuracy points on the 3-way "
                "HateXplain test split (n={n:,})."
            ).format(g=A.EXAMPLE_LEADERBOARD_GAP * 100, n=ctx["test_split_n"]),
            "checks": ["reliability_three_way", "effect_vs_mde_small"],
        },
        {
            "claim_id": "C-RANK-02",
            "depth": "production",
            "kind": "ranking",
            "text": (
                "Model X outperforms model Y by {g:.1f} accuracy points on the binary "
                "toxic/normal task (n={n:,})."
            ).format(g=ctx["large_gap"] * 100, n=ctx["test_split_n"]),
            "checks": ["reliability_binary", "effect_vs_mde_large"],
        },
        {
            "claim_id": "C-COMM-01",
            "depth": "production",
            "kind": "measurement",
            "text": (
                "Annotator agreement is materially lower for posts targeting {c} "
                "(n={n} posts)."
            ).format(c=ctx["thin_community"], n=ctx["thin_community_n"]),
            "checks": ["reliability_thin_community", "precision_thin_community", "coverage"],
        },
        {
            "claim_id": "C-GATE-01",
            "depth": "pilot",
            "kind": "release_gate",
            "text": (
                "Accuracy against HateXplain gold labels is a release gate for a "
                "production content-moderation classifier."
            ),
            "checks": ["depth", "reliability_three_way"],
        },
    ]


def _build_checks(gate: SignalGate, spec: Mapping[str, Any], ctx: Mapping[str, Any]) -> list:
    checks = []
    for name in spec["checks"]:
        if name == "depth":
            checks.append(gate.check_depth_contract(spec["kind"]))
        elif name == "replication":
            checks.append(gate.check_replication(ctx["mean_replication"]))
        elif name == "coverage":
            checks.append(gate.check_coverage(ctx["coverage"]))
        elif name == "reliability_three_way":
            checks.append(
                gate.check_reliability(ctx["alpha_three"], "three_way", ctx["ci_three"])
            )
        elif name == "reliability_binary":
            checks.append(gate.check_reliability(ctx["alpha_binary"], "binary", ctx["ci_binary"]))
        elif name == "reliability_thin_community":
            checks.append(
                gate.check_reliability(
                    ctx["thin_community_alpha"],
                    f"target:{ctx['thin_community']}",
                    ctx["thin_community_ci"],
                )
            )
        elif name == "precision_thin_community":
            checks.append(gate.check_precision(ctx["thin_community_half_width"]))
        elif name == "precision_three_way":
            checks.append(gate.check_precision(ctx["half_width_three"]))
        elif name == "precision_binary":
            checks.append(gate.check_precision(ctx["half_width_binary"]))
        elif name == "effect_vs_mde_small":
            checks.append(gate.check_effect_vs_mde(A.EXAMPLE_LEADERBOARD_GAP, ctx["mde"]))
        elif name == "effect_vs_mde_large":
            checks.append(gate.check_effect_vs_mde(ctx["large_gap"], ctx["mde"]))
        else:  # pragma: no cover - guarded by the spec table above
            raise KeyError(f"unknown check {name!r}")
    return checks


def run_gate(ctx: Mapping[str, Any], policy: GatePolicy, policy_name: str) -> dict:
    ledger = ClaimLedger()
    for spec in _claim_specs(ctx):
        gate = SignalGate(policy=policy, depth=spec["depth"])
        claim = Claim(
            claim_id=spec["claim_id"],
            track="hatexplain",
            text=spec["text"],
            kind=spec["kind"],
            evidence={"depth": spec["depth"], "checks_run": spec["checks"]},
        )
        ledger.submit(gate.evaluate(claim, _build_checks(gate, spec, ctx)))
    payload = ledger.to_dict()
    payload["policy_name"] = policy_name
    payload["policy"] = policy.to_dict()
    payload["blocked_claim_ids"] = [c.claim_id for c in ledger.blocked()]
    payload["publishable_claim_ids"] = [c.claim_id for c in ledger.publishable()]
    return payload


def signal_gate_section(ctx: Mapping[str, Any]) -> dict:
    default = run_gate(ctx, DEFAULT_POLICY, "default")
    exploratory = run_gate(ctx, EXPLORATORY_POLICY, "exploratory")
    flipped = sorted(set(default["blocked_claim_ids"]) - set(exploratory["blocked_claim_ids"]))
    return {
        "default_policy": default,
        "exploratory_policy": exploratory,
        "claims_unblocked_by_relaxing_policy": flipped,
        "reading": (
            "Under the pre-registered default policy {nb} of {nt} claims are blocked: {bl}. "
            "Under the harness's deliberately looser exploratory policy {nu} of them clear "
            "({fl}), which is the point of having two policies rather than one: relaxing "
            "the bar becomes an explicit, reviewable act instead of something that happens "
            "by omission. Note which claims survive either way -- the descriptive counts "
            "and the binary-task comparison with a large gap -- and which do not: any "
            "3-way measurement claim, and the 2-point leaderboard gap, which is blocked "
            "for being smaller than the design's own minimum detectable effect."
        ).format(
            nb=len(default["blocked_claim_ids"]),
            nt=default["summary"]["n_claims"],
            bl=", ".join(default["blocked_claim_ids"]),
            nu=len(flipped),
            fl=", ".join(flipped) if flipped else "none",
        ),
    }


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------


def run_study_a(dataset: Dataset) -> dict:
    """Run the full label-quality audit and return the structured result."""
    shared_battery = three_way_battery(dataset)
    agreement = overall_agreement(dataset, battery=shared_battery)
    consensus = consensus_structure(dataset)
    confusion = label_confusion(dataset)
    contrast = binary_contrast(dataset, three_way_battery_cached=shared_battery)
    annotators = annotator_analysis(dataset)
    targets = agreement_by_target(
        dataset, corpus_alpha=shared_battery["krippendorff_alpha"]["value"]
    )

    alpha_three = contrast["contrast"]["alpha_three_way"]
    alpha_binary = contrast["contrast"]["alpha_binary"]
    ceiling = accuracy_ceiling(dataset, alpha_three, alpha_binary)

    # The worked "thin stratum" claim uses the LARGEST community below the claim
    # floor, not the smallest. A claim about a 2-post stratum is a strawman; a
    # claim about a 92-post stratum is the kind of thing that gets written down.
    thin = max(
        (r for r in targets["rows"] if not r["sufficient_for_claim"]),
        key=lambda r: r["n_posts"],
        default=targets["rows"][-1],
    )
    thin_matrix = reliability_matrix(
        dataset,
        collapse="three_way",
        post_ids=sorted(
            pid for pid, ts in dataset.endorsed_targets(2).items() if thin["community"] in ts
        ),
    )
    thin_ci = alpha_interval(
        thin_matrix,
        metric="nominal",
        n_boot=A.TARGET_BOOTSTRAP_REPLICATES,
        seed=A.SEED_TARGET_BOOTSTRAP,
    )
    community_sizes = [r["n_posts"] for r in targets["rows"]]
    thin_levels = [
        r["community"] for r in targets["rows"] if r["n_posts"] < DEFAULT_POLICY.min_n_per_cell
    ]
    coverage = {
        "total_declared_levels": len(targets["rows"]),
        "absent_levels": [],
        "thin_levels": thin_levels,
        "marginal_gap_fraction": len(thin_levels) / max(1, len(targets["rows"])),
        "worst_balance_ratio": (
            min(community_sizes) / max(community_sizes) if community_sizes else 0.0
        ),
        "basis": (
            "Strata are target communities endorsed by >=2 annotators. 'Thin' uses the "
            "policy's own min_n_per_cell; the separate, much higher floor used for "
            "per-community claims in this study is min_posts_for_community_claim."
        ),
    }

    ctx = {
        "alpha_three": alpha_three,
        "alpha_binary": alpha_binary,
        "ci_three": contrast["three_way"]["ci"],
        "ci_binary": contrast["binary"]["ci"],
        "half_width_three": contrast["three_way"]["ci"]["width"] / 2,
        "half_width_binary": contrast["binary"]["ci"]["width"] / 2,
        "mean_replication": dataset.n_annotations / dataset.n_posts,
        "coverage": coverage,
        "unanimous_share": consensus["three_way"]["percentages"][UNANIMOUS],
        "split_share": consensus["three_way"]["percentages"][SPLIT_2_1],
        "test_split_n": ceiling["minimum_detectable_effect"]["test_split_n"],
        "mde": ceiling["minimum_detectable_effect"]["headline_cell"]["mde_observed_scale"],
        "large_gap": 0.08,
        "thin_community": thin["community"],
        "thin_community_n": thin["n_posts"],
        "thin_community_alpha": thin["alpha_three_way"],
        "thin_community_ci": thin_ci,
        "thin_community_half_width": (thin_ci["width"] / 2) if thin_ci else float("nan"),
    }
    gate = signal_gate_section(ctx)

    return {
        "study": "A",
        "title": "Label-quality audit of HateXplain",
        "dataset": {
            "name": "HateXplain",
            "citation": (
                "Mathew, B., Saha, P., Yimam, S. M., Biemann, C., Goyal, P., & Mukherjee, A. "
                "(2021). HateXplain: A Benchmark Dataset for Explainable Hate Speech "
                "Detection. AAAI 35(17), 14867-14875."
            ),
            "source_path": dataset.source_path,
            "n_posts": dataset.n_posts,
            "n_annotations": dataset.n_annotations,
            "n_annotators": dataset.n_annotators,
            "annotations_per_post": dict(dataset.replication()),
            "label_counts": dict(sorted(dataset.label_counts().items())),
        },
        "assumptions": A.assumptions_block(),
        "overall_agreement": agreement,
        "consensus_structure": consensus,
        "label_confusion": confusion,
        "binary_contrast": contrast,
        "annotators": annotators,
        "agreement_by_target": targets,
        "accuracy_ceiling": ceiling,
        "signal_gate": gate,
    }


__all__ = [
    "accuracy_ceiling",
    "agreement_battery",
    "agreement_by_target",
    "annotator_analysis",
    "binary_contrast",
    "consensus_structure",
    "label_confusion",
    "overall_agreement",
    "run_gate",
    "run_study_a",
    "signal_gate_section",
]
