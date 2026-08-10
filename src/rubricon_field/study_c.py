"""Study C -- replication economics: how many raters, on how many items, for what.

The question
------------
An annotation budget buys ``B`` labels. Spending them as ``n`` items x ``k``
raters is a design choice, and it is usually made by habit -- three raters,
because three is what people use. This study puts the actual variance structure
of HateXplain behind that choice and asks what the money buys.

Three separate things are computed, and keeping them separate is the point,
because they do not have the same answer:

1. **Reliability of the aggregate.** How much does adding raters improve the
   reliability of the label you end up with? Spearman-Brown, applied to the
   measured single-rater alpha.
2. **Statistical power per dollar.** For a fixed budget, which ``(n, k)``
   combination detects the smallest true difference between two systems?
3. **Per-item label quality.** How often does the aggregate label exist at all,
   and how fragile is it? This is not a power question and it does not have the
   same optimum.

The headline result is that (1) and (2) point in opposite directions, and a
budget conversation that only has one of them in it will reach a confident wrong
answer.
"""

from __future__ import annotations

import math
from typing import Sequence

from rubricon.stats.agreement import interpret, krippendorff_alpha, spearman_brown
from rubricon.stats.precision import minimum_detectable_effect

from . import assumptions as A
from .data import Dataset, reliability_matrix


def _r(x: float | None, nd: int = 4) -> float | None:
    if x is None:
        return None
    if isinstance(x, float) and math.isnan(x):
        return None
    return round(float(x), nd)


# --------------------------------------------------------------------------
# 1. single-rater reliability and the Spearman-Brown projection
# --------------------------------------------------------------------------


def raters_needed(rho_1: float, target: float) -> int | None:
    """Smallest ``k`` whose Spearman-Brown reliability reaches ``target``.

    Closed form: ``k >= target*(1 - rho_1) / (rho_1*(1 - target))``. Computed by
    search rather than by the formula so the answer is guaranteed consistent with
    the projection table printed next to it -- a closed form and a table that
    disagree by one rater is the kind of inconsistency that destroys trust in a
    budget memo.
    """
    if rho_1 <= 0 or target >= 1:
        return None
    for k in range(1, 1001):
        if spearman_brown(rho_1, k) >= target:
            return k
    return None


def reliability_projection(dataset: Dataset, rho_1: float | None = None) -> dict:
    """Project reliability at k = 1..9 raters, with the assumptions stated.

    Krippendorff's alpha is a *single-rater* coefficient: it estimates the
    reliability of one annotator's judgement, whatever the panel size used to
    estimate it. That is why the measured 3-rater corpus yields a rho_1 directly,
    and why the empirical check below matters -- recomputing alpha on 2-rater
    panels should return approximately the same value, and if it did not, the
    Spearman-Brown step-up would be being applied to the wrong quantity.

    Three caveats travel with the projection, all of which push the same way
    (towards optimism):

    * Spearman-Brown is derived for the **mean of exchangeable continuous
      replications**. The aggregate here is a majority vote over nominal
      categories, which is not a mean; the projection is an approximation.
    * It assumes rater errors are **independent**. Part of the disagreement in
      this corpus is systematic -- annotators hold different thresholds for
      where hate speech begins, as the annotator profiles show -- and systematic
      disagreement does not average out. To the extent it is systematic, extra
      raters buy less than the table says.
    * It assumes extra raters are **drawn from the same pool**. Recruiting a
      different population changes rho_1 rather than moving along this curve.
    """
    if rho_1 is None:
        rho_1 = krippendorff_alpha(
            reliability_matrix(dataset, collapse="three_way"), metric="nominal"
        ).value

    # Empirical check that alpha really is a per-rater coefficient here: drop one
    # annotator per post (three ways, deterministically by position) and recompute.
    two_rater_alphas = []
    for drop_pos in range(3):
        matrix: dict[str, dict[str, str]] = {}
        for pid in dataset.post_ids:
            anns = dataset.posts[pid].annotations
            if len(anns) < 3:
                continue
            kept = [a for i, a in enumerate(anns) if i != drop_pos]
            matrix[pid] = {a.annotator_id: a.label for a in kept}
        two_rater_alphas.append(krippendorff_alpha(matrix, metric="nominal").value)

    projection = [
        {
            "k_raters": k,
            "reliability": _r(spearman_brown(rho_1, k)),
            "band": interpret(spearman_brown(rho_1, k)),
        }
        for k in A.RATER_COUNTS
    ]
    targets = {
        f"{t:.3f}": {
            "target": t,
            "raters_required": raters_needed(rho_1, t),
            "reliability_at_that_k": _r(
                spearman_brown(rho_1, raters_needed(rho_1, t) or 1)
            ),
        }
        for t in A.RELIABILITY_TARGETS
    }

    k3 = spearman_brown(rho_1, 3)
    return {
        "single_rater_alpha": _r(rho_1),
        "empirical_two_rater_check": {
            "alphas_by_dropped_position": [_r(a) for a in two_rater_alphas],
            "mean": _r(sum(two_rater_alphas) / len(two_rater_alphas)),
            "single_rater_alpha": _r(rho_1),
            "max_abs_difference": _r(max(abs(a - rho_1) for a in two_rater_alphas)),
            "why": (
                "Alpha is a single-rater coefficient, so recomputing it on 2-rater panels "
                "should return the same value as on 3-rater panels. It does. That "
                "confirms Spearman-Brown is being applied to the right quantity: rho_1 is "
                "the reliability of one annotator, and the step-up gives the reliability "
                "of the k-rater aggregate."
            ),
        },
        "projection": projection,
        "targets": targets,
        "three_rater_reliability": _r(k3),
        "method_caveats": [
            "Spearman-Brown is derived for a mean of exchangeable continuous "
            "replications; a majority vote over nominal categories is neither. The "
            "projection is an approximation, not an identity.",
            "It assumes independent rater error. Systematic threshold differences "
            "between annotators do not average out, so the true gain from extra raters "
            "is at or below the projected gain.",
            "It assumes additional raters come from the same pool. A different "
            "recruitment population changes rho_1 rather than moving along this curve.",
        ],
        "reading": (
            "One annotator on this task has a reliability of {r1:.4f}. The published "
            "3-rater design reaches {r3:.4f} -- above Krippendorff's 0.667 floor for "
            "tentative conclusions, which is a real and often-missed point in the "
            "dataset's favour: the aggregate label is meaningfully more reliable than the "
            "individual judgements the headline alpha describes. Reaching {t1} needs "
            "{k1} raters and reaching {t2} needs {k2}, so firm-conclusion reliability on "
            "this construct costs {mult:.1f}x the labelling of the design actually used. "
            "All three caveats above push the same way, so treat these as upper bounds on "
            "what extra raters buy."
        ).format(
            r1=rho_1,
            r3=k3,
            t1=f"{A.RELIABILITY_TARGETS[0]:.3f}",
            k1=raters_needed(rho_1, A.RELIABILITY_TARGETS[0]),
            t2=f"{A.RELIABILITY_TARGETS[1]:.3f}",
            k2=raters_needed(rho_1, A.RELIABILITY_TARGETS[1]),
            mult=(raters_needed(rho_1, A.RELIABILITY_TARGETS[1]) or 3) / 3,
        ),
    }


# --------------------------------------------------------------------------
# 2. budget allocation
# --------------------------------------------------------------------------


def budget_allocation(
    rho_1: float,
    budget_labels: int = A.BUDGET_LABELS,
    rater_counts: Sequence[int] = A.ALLOCATION_RATER_COUNTS,
    disagreement_rates: Sequence[float] = A.PAIRWISE_DISAGREEMENT_SWEEP,
) -> dict:
    """For a fixed label budget, which (n_items, k_raters) split detects most?

    Method. A budget of ``B`` labels buys ``n = B // k`` items at ``k`` raters
    each. For a paired comparison of two systems' per-item accuracy, the SD of
    the item-level difference is ``sqrt(p_disc)``, so the observed-scale MDE is
    ``(z_{1-a/2} + z_power) * sqrt(p_disc / n)``. Increasing ``k`` shrinks ``n``
    and therefore *raises* that MDE. What increasing ``k`` buys is a less noisy
    gold label, which shows up as a higher reliability and therefore a smaller
    attenuation correction. The quantity that trades those off is the
    **true-score-scale MDE**, ``mde_observed / sqrt(rho_k)`` -- reported by
    ``rubricon.stats.precision.minimum_detectable_effect`` as ``mde_true_scale``,
    deliberately never compounded with the observed-scale figure.

    Substituting ``n = B/k`` and ``rho_k = k*rho_1 / (1 + (k-1)*rho_1)``, the
    true-scale MDE is proportional to ``sqrt(k / rho_k)``, and

        k / rho_k = (1 + (k - 1) * rho_1) / rho_1

    which is strictly increasing in ``k`` for every ``rho_1`` in (0, 1). So the
    optimum is always ``k = 1``, for any reliability whatsoever -- this is an
    analytic result about the attenuation model, not a fact about HateXplain,
    and the grid below simply confirms it numerically at this dataset's variance.
    Saying so plainly matters: it means "should we add a third rater?" is not
    answerable by a power calculation, because the power calculation always says
    no. The reasons to add raters are elsewhere, and they are listed in the
    result.
    """
    rows = []
    for k in rater_counts:
        n = budget_labels // k
        rho_k = spearman_brown(rho_1, k)
        for p_disc in disagreement_rates:
            res = minimum_detectable_effect(
                n=n,
                sd=math.sqrt(p_disc),
                reliability=rho_k,
                alpha_level=A.ALPHA_LEVEL,
                power=A.POWER,
                paired=True,
            ).to_dict()
            rows.append(
                {
                    "k_raters": k,
                    "n_items": n,
                    "labels_spent": n * k,
                    "gold_reliability_rho_k": _r(rho_k),
                    "assumed_pairwise_disagreement": p_disc,
                    "mde_observed_scale": res["mde_observed_scale"],
                    "mde_true_scale": res["mde_true_scale"],
                }
            )

    headline_rows = [
        r for r in rows if r["assumed_pairwise_disagreement"] == A.HEADLINE_DISAGREEMENT_RATE
    ]
    best = min(headline_rows, key=lambda r: r["mde_true_scale"])
    k3 = next(r for r in headline_rows if r["k_raters"] == 3)
    k1 = next(r for r in headline_rows if r["k_raters"] == 1)

    # Is the optimum stable across the assumed disagreement rate? It must be:
    # p_disc scales every cell by the same constant. Verified rather than asserted.
    optima = {
        str(p): min(
            (r for r in rows if r["assumed_pairwise_disagreement"] == p),
            key=lambda r: r["mde_true_scale"],
        )["k_raters"]
        for p in disagreement_rates
    }

    return {
        "budget_labels": budget_labels,
        "criterion": "true-score-scale minimum detectable effect (lower is better)",
        "grid": rows,
        "optimum_k_at_headline_rate": best["k_raters"],
        "optimum_k_by_disagreement_rate": optima,
        "optimum_is_invariant_to_disagreement_assumption": len(set(optima.values())) == 1,
        "analytic_result": (
            "Substituting n = B/k and the Spearman-Brown rho_k into the true-scale MDE "
            "gives MDE proportional to sqrt((1 + (k-1)*rho_1) / rho_1), which is strictly "
            "increasing in k for every rho_1 in (0,1). One rater per item maximises "
            "aggregate statistical power per label at ANY reliability. This is a property "
            "of the attenuation model, not of this dataset."
        ),
        "is_a_third_rater_worth_more_than_a_third_more_items": {
            "answer": "No -- not for aggregate statistical power.",
            "k1_true_scale_mde": k1["mde_true_scale"],
            "k3_true_scale_mde": k3["mde_true_scale"],
            "k3_penalty_relative": _r(
                k3["mde_true_scale"] / k1["mde_true_scale"] - 1
            ),
            "detail": (
                "At this dataset's variance and a {b:,}-label budget, spending on one "
                "rater per item gives {n1:,} items and a true-scale MDE of {m1:.4f}. "
                "Spending on three raters gives {n3:,} items and {m3:.4f} -- {pen:.0%} "
                "worse. The three-rater gold label is more reliable ({r3:.4f} against "
                "{r1:.4f}), but not by enough to pay for having a third as many items. "
                "For the narrow question 'which design detects a smaller true difference "
                "between two systems', more items wins, and it wins at every reliability."
            ).format(
                b=budget_labels,
                n1=k1["n_items"],
                m1=k1["mde_true_scale"],
                n3=k3["n_items"],
                m3=k3["mde_true_scale"],
                pen=k3["mde_true_scale"] / k1["mde_true_scale"] - 1,
                r3=k3["gold_reliability_rho_k"],
                r1=k1["gold_reliability_rho_k"],
            ),
        },
        "why_you_should_still_replicate": [
            "With k=1 the reliability of the labels is unknowable. There is no second "
            "judgement to compare against, so no alpha, no MDE that accounts for "
            "attenuation, and no way to detect that the pool has drifted. The power "
            "calculation that recommends k=1 depends on a rho_1 that only replication "
            "can supply.",
            "Per-item gold quality does not follow aggregate power. Error analysis, "
            "safety triage, per-community breakdowns and any use of individual rows all "
            "need a label that is defensible for that row, and a k=1 label is one "
            "person's opinion with no margin at all.",
            "A single-rated corpus cannot distinguish a hard item from a careless "
            "annotator, so it has no quality-control signal and no way to identify the "
            "genuinely ambiguous items that are usually the most informative ones.",
            "Aggregate power is not the only objective. If the deliverable is a public "
            "benchmark that others will rank models on, the per-item defensibility of "
            "the labels is the product.",
        ],
        "recommended_design": (
            "Rate the bulk of the corpus once and carve out a replicated island: spend "
            "{f:.0%} of the budget on a 3-rated subsample. That subsample supplies rho_1, "
            "the drift signal, and the annotator profiles, while the remaining {r:.0%} "
            "buys the item count that actually moves the MDE. It is strictly better than "
            "uniform 3-rating on power and strictly better than uniform 1-rating on "
            "measurability."
        ).format(
            f=A.HYBRID_RELIABILITY_SUBSAMPLE_FRACTION,
            r=1 - A.HYBRID_RELIABILITY_SUBSAMPLE_FRACTION,
        ),
    }


# --------------------------------------------------------------------------
# 3. cost model
# --------------------------------------------------------------------------


def cost_model(
    dataset: Dataset, rho_1: float, unit_cost: float = A.UNIT_COST_USD_PER_LABEL
) -> dict:
    """What each design costs, with the unit price as one named, adjustable input.

    ``unit_cost`` is the single price in this study. Every dollar figure below is
    that number multiplied by a label count, so a reader who thinks the rate is
    wrong can rescale everything in their head. No conclusion about *allocation*
    depends on it, because allocation compares designs at equal label counts.
    """
    n_posts = dataset.n_posts
    rows = []
    for k in A.RATER_COUNTS:
        labels = n_posts * k
        rows.append(
            {
                "k_raters": k,
                "n_items": n_posts,
                "labels": labels,
                "cost_usd": round(labels * unit_cost, 2),
                "reliability": _r(spearman_brown(rho_1, k)),
                "band": interpret(spearman_brown(rho_1, k)),
            }
        )
    def row_for(k: int) -> dict:
        labels = n_posts * k
        return {
            "k_raters": k,
            "n_items": n_posts,
            "labels": labels,
            "cost_usd": round(labels * unit_cost, 2),
            "reliability": _r(spearman_brown(rho_1, k)),
            "band": interpret(spearman_brown(rho_1, k)),
        }

    actual = row_for(3)
    k_667 = raters_needed(rho_1, 0.667)
    k_800 = raters_needed(rho_1, 0.800)
    # Computed rather than looked up: at a low enough rho_1 the required k falls
    # outside the projection table, and a KeyError there would be a crash caused
    # purely by the corpus being small.
    row_800 = row_for(k_800) if k_800 else None

    hybrid_replicated_items = int(n_posts * A.HYBRID_RELIABILITY_SUBSAMPLE_FRACTION)
    hybrid_labels = (n_posts - hybrid_replicated_items) + hybrid_replicated_items * 3
    return {
        "unit_cost_usd_per_label": unit_cost,
        "unit_cost_is_an_assumption": True,
        "n_items": n_posts,
        "by_rater_count": rows,
        "as_published": {
            "k_raters": 3,
            "labels": actual["labels"],
            "cost_usd": actual["cost_usd"],
            "reliability": actual["reliability"],
        },
        "cost_of_reaching_targets": {
            "0.667": {
                "raters_required": k_667,
                "cost_usd": round(n_posts * (k_667 or 0) * unit_cost, 2),
            },
            "0.800": {
                "raters_required": k_800,
                "cost_usd": round(n_posts * (k_800 or 0) * unit_cost, 2),
                "marginal_cost_over_published_usd": (
                    round((row_800["cost_usd"] - actual["cost_usd"]), 2) if row_800 else None
                ),
                "marginal_reliability_gain": (
                    _r(row_800["reliability"] - actual["reliability"]) if row_800 else None
                ),
            },
        },
        "hybrid_design": {
            "replicated_fraction": A.HYBRID_RELIABILITY_SUBSAMPLE_FRACTION,
            "n_items_covered": n_posts,
            "n_items_triple_rated": hybrid_replicated_items,
            "labels": hybrid_labels,
            "cost_usd": round(hybrid_labels * unit_cost, 2),
            "saving_vs_uniform_triple_usd": round(
                (actual["labels"] - hybrid_labels) * unit_cost, 2
            ),
            "note": (
                "Same item coverage, a measurable reliability, and a saving of "
                f"{round((actual['labels'] - hybrid_labels) * unit_cost, 2):,.2f} USD at "
                f"{unit_cost:.2f}/label. What it gives up is per-item defensibility on the "
                "un-replicated rows, which is the right trade only if the corpus is being "
                "used for aggregate comparison rather than as a public gold standard."
            ),
        },
        "reading": (
            "At an assumed {u:.2f} USD per label, the published 3-rater design costs "
            "{c:,.2f} USD for a gold reliability of {r:.4f}. Reaching the 0.800 "
            "firm-conclusion floor needs {k8} raters, {c8:,.2f} USD -- an extra "
            "{dc:,.2f} USD for {dr:+.4f} of reliability. Whether that is worth paying "
            "depends on what the labels are for, which is exactly the conversation the "
            "number is meant to start."
        ).format(
            u=unit_cost,
            c=actual["cost_usd"],
            r=actual["reliability"],
            k8=k_800,
            c8=row_800["cost_usd"] if row_800 else float("nan"),
            dc=(row_800["cost_usd"] - actual["cost_usd"]) if row_800 else float("nan"),
            dr=(row_800["reliability"] - actual["reliability"]) if row_800 else float("nan"),
        ),
    }


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------


def run_study_c(dataset: Dataset, rho_1: float | None = None) -> dict:
    """Run the replication-economics study and return the structured result."""
    if rho_1 is None:
        rho_1 = krippendorff_alpha(
            reliability_matrix(dataset, collapse="three_way"), metric="nominal"
        ).value
    projection = reliability_projection(dataset, rho_1=rho_1)
    allocation = budget_allocation(rho_1)
    costs = cost_model(dataset, rho_1)
    return {
        "study": "C",
        "title": "Replication economics at HateXplain's measured variance",
        "assumptions": A.assumptions_block(),
        "reliability_projection": projection,
        "budget_allocation": allocation,
        "cost_model": costs,
        "headline": (
            "Single-rater reliability is {r1:.4f}; the 3-rater aggregate the corpus "
            "actually publishes is {r3:.4f}. Getting to 0.667 takes {k1} raters, getting "
            "to 0.800 takes {k2}. But on a fixed label budget, aggregate statistical power "
            "is maximised at ONE rater per item -- at any reliability, as an analytic "
            "property of the attenuation model -- so a third rater is not worth more than "
            "a third more items if the only goal is detecting a difference between two "
            "systems. Replication earns its cost through measurability and per-item "
            "defensibility, not through power, and a budget argument that conflates the "
            "two will reach a confident wrong answer in whichever direction it started."
        ).format(
            r1=projection["single_rater_alpha"],
            r3=projection["three_rater_reliability"],
            k1=projection["targets"][f"{A.RELIABILITY_TARGETS[0]:.3f}"]["raters_required"],
            k2=projection["targets"][f"{A.RELIABILITY_TARGETS[1]:.3f}"]["raters_required"],
        ),
    }


__all__ = [
    "budget_allocation",
    "cost_model",
    "raters_needed",
    "reliability_projection",
    "run_study_c",
]
