"""Every assumption this study makes, in one file, named and adjustable.

A field study that mixes measured quantities with modelling assumptions and does
not separate them is not auditable: the reader cannot tell which numbers would
change if they disagreed with you. Everything in this module is an assumption,
a seed, or a computational budget. Nothing here was measured from the data.

Each constant records *why* it has the value it has, and every result file that
depends on one echoes it back under an ``assumptions`` key, so a number can
never be quoted without the premise it rests on.
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# determinism
# --------------------------------------------------------------------------

#: Master seed. Every derived seed is this plus a small fixed offset, so the
#: whole pipeline is reproducible from one number and no two resamplings share a
#: stream by accident.
SEED = 20210201  # HateXplain's AAAI-2021 publication year, for memorability only

#: Master seed of the paper analyses (PREREG.md section 8). The HateXplain-only
#: studies above keep SEED so their committed results stay reproducible byte for
#: byte; everything added for the cross-corpus paper derives from this one.
SEED_PAPER = 20260929
SEED_PAPER_BOOTSTRAP = SEED_PAPER + 1

SEED_BOOTSTRAP = SEED + 1
SEED_TARGET_BOOTSTRAP = SEED + 2
SEED_JUDGE_HOLDOUT = SEED + 3
SEED_PERMUTATION = SEED + 4

# --------------------------------------------------------------------------
# computational budgets
# --------------------------------------------------------------------------
#
# Krippendorff's alpha on the full corpus takes ~0.08s, but the cluster
# bootstrap needs several hundred recomputations, and the interval is wanted for
# several sub-analyses. Bootstrapping the full 20,148 units everywhere would put
# the pipeline in the tens of minutes for no gain: at n=6,000 the interval on
# alpha is already about +/-0.015, an order of magnitude below any threshold it
# is compared against. So point estimates are always computed on the FULL
# corpus, and only the intervals are computed on a seeded subsample, whose size
# is written into every result that uses it.

#: Units drawn (without replacement, seeded) for bootstrap intervals on the
#: headline agreement batteries.
BOOTSTRAP_SUBSAMPLE_UNITS = 6000

#: Bootstrap replicates for the headline intervals.
BOOTSTRAP_REPLICATES = 400

#: Replicates for the smaller per-community intervals.
TARGET_BOOTSTRAP_REPLICATES = 250

#: Communities with fewer than this many posts get a point estimate and an
#: explicit "insufficient to support a claim" marker rather than an interval.
#: At n=250 the bootstrap half-width on alpha is already around +/-0.07, which is
#: wider than most of the between-community differences being discussed.
MIN_POSTS_FOR_COMMUNITY_CLAIM = 250

#: How many of the heaviest annotators to remove one at a time.
LEAVE_OUT_TOP_N = 10

#: Annotators below this load are excluded from the "harshest / most lenient"
#: rankings. A 6-annotation annotator can top any marginal-share leaderboard by
#: chance; ranking them against annotator #4's 5,730 judgements is not a
#: comparison, it is a sampling artefact.
MIN_LOAD_FOR_MARGINAL_RANKING = 100

#: Offset from the pool mean, on a 0-1 dimension, above which
#: ``profile_annotators`` raises ``systematic_bias``. 0.15 means "calls posts
#: toxic 15 percentage points more or less often than the pool", against a pool
#: base rate near 0.60. Chosen to be large enough that it is not reachable by
#: sampling noise at the ranking-eligible load floor above.
ANNOTATOR_BIAS_THRESHOLD = 0.15

# --------------------------------------------------------------------------
# power analysis: the classifier comparison being modelled
# --------------------------------------------------------------------------
#
# We hold no classifier predictions, so the minimum-detectable-effect analysis
# has to be told how much two compared systems differ item-by-item. For paired
# binary accuracy the SD of the per-item difference is sqrt(p_disc - delta^2),
# where p_disc is the fraction of items on which the two systems disagree --
# the discordant cells of McNemar's table. That single quantity is the only
# assumption the MDE grid needs, and it is swept rather than fixed.

#: Sweep of assumed item-level disagreement rates between two compared systems.
#: 0.20 is the headline: two systems at ~0.70 accuracy each and substantially
#: correlated errors typically disagree on 15-25% of items.
PAIRWISE_DISAGREEMENT_SWEEP: tuple[float, ...] = (0.05, 0.10, 0.20, 0.30)

#: The cell quoted in prose.
HEADLINE_DISAGREEMENT_RATE = 0.20

#: HateXplain ships an 8:1:1 train/val/test split. We do not have the official
#: split file in this repository, so the test-split size is treated as an
#: assumption: 10% of the corpus, rounded. Every MDE quoted for "a HateXplain
#: test split" uses this n and says so.
TEST_SPLIT_FRACTION = 0.10

#: Evaluation sizes the MDE grid is evaluated at, alongside the derived test
#: split size. 1,000 is a typical hand-curated probe set.
EVAL_SIZES: tuple[int, ...] = (1000, 5000)

#: Two-sided significance level and target power for every MDE in this study.
ALPHA_LEVEL = 0.05
POWER = 0.80

#: The gap, in accuracy points, used as the worked example of a leaderboard
#: claim ("model X beats model Y by two points").
EXAMPLE_LEADERBOARD_GAP = 0.02

# --------------------------------------------------------------------------
# replication economics
# --------------------------------------------------------------------------

#: Unit cost of one label from one annotator, in USD. THIS IS AN ASSUMPTION and
#: it is the only price in the model: every cost in study C is this number times
#: a label count. It is set to a plausible crowd rate for a short-text
#: three-way classification with a codebook, inclusive of platform fee. Change
#: it here and every cost in the results moves proportionally; no conclusion in
#: study C about *allocation* depends on its value, because allocation compares
#: designs at equal label counts.
UNIT_COST_USD_PER_LABEL = 0.08

#: Rater counts projected by Spearman-Brown.
RATER_COUNTS: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8, 9)

#: Reliability targets. 0.667 is Krippendorff's floor for tentative
#: conclusions; 0.800 is his floor for firm ones.
RELIABILITY_TARGETS: tuple[float, ...] = (0.667, 0.800)

#: Fixed budget for the allocation grid, expressed in labels. Set to the actual
#: spend of the published corpus (20,148 posts x 3 raters) so the allocation
#: question is the one the authors actually faced.
BUDGET_LABELS = 60_444

#: Rater counts considered in the allocation grid.
ALLOCATION_RATER_COUNTS: tuple[int, ...] = (1, 2, 3, 4, 5)

#: Share of a budget set aside for a 3-rated reliability subsample in the
#: hybrid design study C recommends. You cannot estimate reliability at all
#: without replication, so a k=1 bulk design needs a replicated island.
HYBRID_RELIABILITY_SUBSAMPLE_FRACTION = 0.10

# --------------------------------------------------------------------------
# judge validation
# --------------------------------------------------------------------------

#: Environment variable that selects the judge implementation in study B.
#: Unset (or ``heldout``) selects the held-out-human stand-in, which is the only
#: judge exercised in this repository.
JUDGE_ENV_VAR = "RUBRICON_FIELD_JUDGE"

DEFAULT_JUDGE = "heldout"


def assumptions_block() -> dict:
    """The subset of this module that gets echoed into every result file."""
    return {
        "seed": SEED,
        "bootstrap_subsample_units": BOOTSTRAP_SUBSAMPLE_UNITS,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "alpha_level": ALPHA_LEVEL,
        "power": POWER,
        "assumed_pairwise_disagreement_sweep": list(PAIRWISE_DISAGREEMENT_SWEEP),
        "headline_pairwise_disagreement_rate": HEADLINE_DISAGREEMENT_RATE,
        "test_split_fraction": TEST_SPLIT_FRACTION,
        "unit_cost_usd_per_label": UNIT_COST_USD_PER_LABEL,
        "annotator_bias_threshold": ANNOTATOR_BIAS_THRESHOLD,
        "min_posts_for_community_claim": MIN_POSTS_FOR_COMMUNITY_CLAIM,
        "min_load_for_marginal_ranking": MIN_LOAD_FOR_MARGINAL_RANKING,
        "note": (
            "Point estimates are computed on the full corpus. Bootstrap intervals are "
            "computed on a seeded subsample of the size given above; every interval "
            "records its own n_clusters."
        ),
    }


__all__ = [name for name in dir() if not name.startswith("_")]
