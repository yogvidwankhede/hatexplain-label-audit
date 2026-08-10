"""Study B -- validating a judge harness end to end, on real data, at zero cost.

The problem this solves
-----------------------
Everyone building an LLM-judge pipeline wants to know two things before they
spend money on it: does the scoring harness work, and what score would count as
good. The usual answer to the second is "as close to 1.0 as possible", which is
wrong in a way that costs teams months. A judge is scored against human labels.
If the humans agree with each other only moderately, then a judge that models
those humans *perfectly* still does not match them perfectly, because they do
not match each other. The achievable maximum is set by the panel, not by the
judge.

The design used here
--------------------
Hold out one real annotator per post and put them in the judge's seat, scored
against the remaining two. This is deliberately not an LLM. It is a
**methodological stand-in**, and it buys three things:

1. It exercises the entire scoring path -- protocol, panel construction,
   agreement, disattenuation, reporting -- against real human labels, so a bug
   in the harness shows up before any inference budget is spent.
2. It measures the ceiling directly. The held-out annotator *is* a competent
   human judge drawn from the same pool that produced the gold labels. Whatever
   they score is, to a first approximation, what a perfect judge would score.
3. It costs nothing and is exactly reproducible.

**What it is not.** It is not an evaluation of any LLM, and nothing here should
be read as one. No model was called. The judge slot is filled by a person whose
label was already in the file. Swapping in a real judge is a one-line change --
see :class:`JudgeProtocol` and the ``RUBRICON_FIELD_JUDGE`` environment variable
-- and the numbers below are what that judge would have to be compared against.

A note on the panel
-------------------
The panel is two annotators, not three, because one was removed to fill the
judge slot. A two-annotator panel produces a usable label only when the two
agree; when they split, there is no target and the post is excluded from
exact-match scoring rather than resolved by a tiebreak. The excluded share is
reported, because it is large and because a harness that silently dropped it
would report a flattering number computed only on the easy posts.
"""

from __future__ import annotations

import math
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence, runtime_checkable

from rubricon.stats.agreement import krippendorff_alpha, percent_agreement, spearman_brown
from rubricon.stats.precision import attenuated_correlation, paired_permutation_test

from . import assumptions as A
from .data import BINARY_MAP, LABELS, Dataset, Post, reliability_matrix


def _r(x: float | None, nd: int = 4) -> float | None:
    if x is None:
        return None
    if isinstance(x, float) and math.isnan(x):
        return None
    return round(float(x), nd)


# --------------------------------------------------------------------------
# the judge interface
# --------------------------------------------------------------------------


@runtime_checkable
class JudgeProtocol(Protocol):
    """Anything that can put a label on a post.

    Deliberately minimal. A judge sees a post and returns one of
    :data:`rubricon_field.data.LABELS`, or ``None`` to abstain. Abstentions are
    counted separately and never scored as errors, because an abstaining judge
    and a wrong judge call for different responses.

    Everything else -- prompt construction, retries, batching, caching, rate
    limits -- belongs inside the implementation. The scoring code in this module
    touches nothing but ``name`` and ``label``, which is what makes the held-out
    human and a hypothetical API judge interchangeable.
    """

    #: Short identifier written into the results.
    name: str

    def label(self, post: Post) -> str | None:
        """Return a label for ``post``, or ``None`` to abstain."""
        ...


@dataclass(frozen=True)
class HoldoutPlan:
    """Which annotator is held out for each post.

    Built once from a seed and passed around, rather than re-drawn at each use.
    Re-drawing would make the panel and the judge disagree about who was
    excluded -- a bug that produces a *better* judge score, because the judge's
    own label leaks into their panel, and is therefore the kind of bug that does
    not look like a bug.
    """

    held_out: Mapping[str, str]  # post_id -> annotator_id
    seed: int

    def annotator_for(self, post_id: str) -> str | None:
        return self.held_out.get(post_id)


def build_holdout_plan(dataset: Dataset, seed: int = A.SEED_JUDGE_HOLDOUT) -> HoldoutPlan:
    """Pick one annotator per post, uniformly at random, with a fixed seed.

    Uniform rather than "always the first" or "always the heaviest": holding out
    a fixed position would confound the judge score with whatever ordering the
    release happens to use, and holding out the heaviest annotator would measure
    that one person rather than the pool.
    """
    rng = random.Random(seed)
    plan: dict[str, str] = {}
    for pid in dataset.post_ids:
        anns = dataset.posts[pid].annotations
        if len(anns) < 3:
            # Fewer than three raters leaves no panel after removing one; the
            # post is simply not part of this design, and is counted as such.
            continue
        plan[pid] = anns[rng.randrange(len(anns))].annotator_id
    return HoldoutPlan(held_out=plan, seed=seed)


@dataclass
class HeldOutAnnotatorJudge:
    """The stand-in judge: a real human, removed from the panel that scores them.

    This is the only judge exercised in this repository. It is a human, not a
    model, and every number it produces is labelled as a human-ceiling estimate
    rather than as judge performance.
    """

    dataset: Dataset
    plan: HoldoutPlan
    name: str = "held_out_human"

    def label(self, post: Post) -> str | None:
        aid = self.plan.annotator_for(post.post_id)
        if aid is None:
            return None
        for a in post.annotations:
            if a.annotator_id == aid:
                return a.label
        return None


@dataclass
class ConstantJudge:
    """Always returns the same label. The floor any real judge must clear.

    A judge that beats nothing is a judge that has learned the prior. Reporting
    a judge's accuracy without this baseline alongside it is how a 62%-accurate
    judge gets celebrated on a task whose majority class is 61%.
    """

    constant: str = "normal"
    name: str = "constant_baseline"

    def label(self, post: Post) -> str | None:  # noqa: D102 - see class docstring
        return self.constant


@dataclass
class APIJudge:
    """Adapter shape for a real LLM judge. **Not exercised in this repository.**

    Nothing in this class is called anywhere in this study, no network request
    is made by this package, and no results file contains a number produced by
    it. It exists to fix the integration surface, so that "plug in a real judge"
    is a matter of filling in one method rather than reworking the scoring code.

    To use it, implement ``_call_model`` against whatever client you have and
    select it with the ``RUBRICON_FIELD_JUDGE`` environment variable
    (``RUBRICON_FIELD_JUDGE=api:<model-name>``). Everything downstream --
    panels, exact match, per-label recall, alpha, disattenuation, the ceiling
    comparison -- runs unchanged, because it only ever sees ``name`` and
    ``label``.

    Three things this shape deliberately forces:

    * ``temperature`` is explicit and defaults to 0. A judge that is not
      deterministic has a test-retest reliability below 1, and the
      disattenuation in this study assumes a value for it; guessing that value
      silently would be worse than declaring it.
    * unparseable responses return ``None`` (abstain) rather than a guessed
      label, so parser failures cannot be laundered into apparent accuracy.
    * the prompt is a field, not a literal, because a judge's prompt is part of
      the instrument and belongs in the results next to the numbers it produced.
    """

    model: str
    prompt_template: str = (
        "Classify the following social media post as exactly one of: "
        "normal, offensive, hatespeech.\n\nPost: {text}\n\nAnswer with one word."
    )
    temperature: float = 0.0
    name: str = "api_judge"

    def __post_init__(self) -> None:
        self.name = f"api_judge:{self.model}"

    def _call_model(self, prompt: str) -> str:  # pragma: no cover - never exercised
        raise NotImplementedError(
            "APIJudge is an unexercised adapter shape. Implement _call_model against "
            "your model client to use it. This repository makes no network calls."
        )

    def label(self, post: Post) -> str | None:  # pragma: no cover - never exercised
        raw = self._call_model(self.prompt_template.format(text=post.text))
        candidate = raw.strip().lower()
        return candidate if candidate in LABELS else None


def judge_from_env(dataset: Dataset, plan: HoldoutPlan) -> JudgeProtocol:
    """Select the judge from ``RUBRICON_FIELD_JUDGE``.

    ============================  =========================================
    value                         judge
    ============================  =========================================
    unset / ``heldout``           :class:`HeldOutAnnotatorJudge` (default)
    ``constant:<label>``          :class:`ConstantJudge`
    ``api:<model>``               :class:`APIJudge` -- raises on use
    ============================  =========================================

    The default is the held-out human because that is the only judge this study
    validates. An unrecognised value is an error rather than a silent fallback:
    quietly scoring the wrong judge is the failure this whole module exists to
    make impossible.
    """
    spec = os.environ.get(A.JUDGE_ENV_VAR, A.DEFAULT_JUDGE).strip()
    if spec in ("", A.DEFAULT_JUDGE):
        return HeldOutAnnotatorJudge(dataset=dataset, plan=plan)
    if spec.startswith("constant:"):
        return ConstantJudge(constant=spec.split(":", 1)[1])
    if spec.startswith("api:"):
        return APIJudge(model=spec.split(":", 1)[1])
    raise ValueError(
        f"{A.JUDGE_ENV_VAR}={spec!r} is not recognised. Use 'heldout', "
        "'constant:<label>', or 'api:<model>'."
    )


# --------------------------------------------------------------------------
# panels and scoring
# --------------------------------------------------------------------------


@dataclass
class PanelItem:
    """One post's judge label and the panel it is scored against."""

    post_id: str
    judge_label: str | None
    panel_labels: tuple[str, ...]
    panel_consensus: str | None


def build_panels(
    dataset: Dataset, judge: JudgeProtocol, plan: HoldoutPlan, collapse: str = "three_way"
) -> list[PanelItem]:
    """Score-ready items: judge label plus the two-annotator panel behind it."""
    mapper = BINARY_MAP if collapse == "binary" else {lab: lab for lab in LABELS}
    items: list[PanelItem] = []
    for pid in dataset.post_ids:
        held = plan.annotator_for(pid)
        if held is None:
            continue
        post = dataset.posts[pid]
        raw = judge.label(post)
        jl = mapper.get(raw) if raw is not None else None
        panel = tuple(
            mapper[a.label] for a in post.annotations if a.annotator_id != held
        )
        consensus = panel[0] if len(set(panel)) == 1 and panel else None
        items.append(PanelItem(pid, jl, panel, consensus))
    return items


def score_judge(items: Sequence[PanelItem], categories: Sequence[str]) -> dict:
    """Exact match, per-label recall and precision, confusion, and alpha.

    Scored only over posts where the two-annotator panel agreed. That is a
    genuine restriction and it is reported, not hidden: posts where the panel
    split have no target label, and inventing one by tiebreak would score the
    judge against a coin.
    """
    scored = [i for i in items if i.panel_consensus is not None and i.judge_label is not None]
    abstained = sum(1 for i in items if i.judge_label is None)
    no_consensus = sum(1 for i in items if i.panel_consensus is None)

    hits = sum(1 for i in scored if i.judge_label == i.panel_consensus)
    confusion: dict[str, Counter] = defaultdict(Counter)
    for i in scored:
        confusion[i.panel_consensus][i.judge_label] += 1

    recall, precision = {}, {}
    judge_counts: Counter = Counter(i.judge_label for i in scored)
    for cat in categories:
        row = confusion.get(cat, Counter())
        n_gold = sum(row.values())
        recall[cat] = {
            "n_panel_consensus": n_gold,
            "recall": _r(row.get(cat, 0) / n_gold) if n_gold else None,
        }
        n_pred = judge_counts.get(cat, 0)
        precision[cat] = {
            "n_judge_predicted": n_pred,
            "precision": _r(row.get(cat, 0) / n_pred) if n_pred else None,
        }

    matrix = {i.post_id: {"judge": i.judge_label, "panel": i.panel_consensus} for i in scored}
    alpha = krippendorff_alpha(matrix, metric="nominal")
    raw = percent_agreement(matrix)

    # Unconditioned anchor. Exact match and alpha above are computed only where
    # the panel agreed, which selects the easy posts and inflates both. Scoring
    # the judge against each panel member individually, over EVERY post, removes
    # that selection, and the resulting number should land on the corpus-wide
    # pairwise agreement -- which is the check that the harness is wired up
    # correctly, and the honest denominator for the conditioned figure.
    pair_hits = pair_n = 0
    for i in items:
        if i.judge_label is None:
            continue
        for member in i.panel_labels:
            pair_n += 1
            pair_hits += int(member == i.judge_label)

    return {
        "judge_vs_individual_panel_members_unconditioned": {
            "exact_match": _r(pair_hits / pair_n) if pair_n else None,
            "n_pairs": pair_n,
            "note": (
                "Judge against each panel member separately, over all posts, with no "
                "conditioning on panel agreement. This is the selection-free comparison; "
                "the headline exact match above is conditioned on the panel agreeing and "
                "is optimistic relative to it."
            ),
        },
        "n_items": len(items),
        "n_scored": len(scored),
        "n_panel_without_consensus": no_consensus,
        "panel_consensus_rate": _r((len(items) - no_consensus) / len(items)) if items else None,
        "n_judge_abstained": abstained,
        "exact_match": _r(hits / len(scored)) if scored else None,
        "krippendorff_alpha_judge_vs_panel": _r(alpha.value),
        "percent_agreement_judge_vs_panel": _r(raw.value),
        "per_label_recall": recall,
        "per_label_precision": precision,
        "confusion_panel_by_judge": {
            g: {j: confusion[g].get(j, 0) for j in categories} for g in categories
        },
        "scoring_note": (
            "Exact match and alpha are computed only over posts where the two-annotator "
            "panel agreed. Posts where it split have no target label and are excluded "
            "rather than resolved by tiebreak; their count is reported above."
        ),
    }


# --------------------------------------------------------------------------
# ceilings and disattenuation
# --------------------------------------------------------------------------


def judge_ceiling(
    dataset: Dataset,
    items: Sequence[PanelItem],
    plan: HoldoutPlan,
    single_rater_alpha: float,
    measured_exact_match: float,
    measured_alpha: float,
    collapse: str = "three_way",
) -> dict:
    """The maximum any judge can score against this panel, and why.

    Three reference points, and one of them is deliberately shown to be useless:

    ``random_rater_ceiling``
        What a competent human drawn from the annotation pool actually scores.
        This is the measured held-out number, and it is the honest target: an
        LLM judge that reaches it is performing as well as the people who made
        the data.
    ``single_rater_modal_ceiling``
        The best expected match of any predictor against **one** randomly drawn
        annotator, over every post: the mean modal vote share. Unlike the
        conditioned figure above it involves no selection on panel agreement, so
        it is the cleaner bound, and it is strictly higher because naming the
        modal label beats being a random rater.
    ``degenerate_consensus_bound``
        The obvious construction -- an oracle that names the modal label across
        all three annotators -- scores exactly 1.0 against a two-annotator
        consensus, mechanically: whenever two of three raters agree, their label
        *is* the modal label. It is reported here with its value because
        computing it and discovering it is degenerate is the point. A
        two-rater-consensus target admits a perfect score in principle, so "the
        ceiling is 100%" is technically true and completely uninformative. It is
        uninformative because the consensus is not a function of the text, and a
        judge only sees the text. That is why the achievable ceiling has to be
        estimated by putting a human in the judge's seat rather than derived.

    The alpha reference is separate. A judge should not be *expected* to reach an
    alpha against the panel exceeding the humans' own single-rater alpha, but it
    is not a hard bound: a systematic judge can track the panel's systematic
    component better than a second noisy human does. It is a reference point and
    is labelled as one.
    """
    mapper = BINARY_MAP if collapse == "binary" else {lab: lab for lab in LABELS}
    by_id = {i.post_id: i for i in items}
    oracle_hits, oracle_n = 0, 0
    for pid, item in by_id.items():
        if item.panel_consensus is None:
            continue
        labs = [mapper[a.label] for a in dataset.posts[pid].annotations]
        top = Counter(labs).most_common()
        if len(top) > 1 and top[0][1] == top[1][1]:
            continue  # no modal label to name
        oracle_n += 1
        if top[0][0] == item.panel_consensus:
            oracle_hits += 1

    # Best expected agreement with ONE randomly drawn annotator, over all posts.
    modal_shares = []
    for post in dataset.posts.values():
        labs = [mapper[a.label] for a in post.annotations]
        if not labs:
            continue
        modal_shares.append(max(Counter(labs).values()) / len(labs))
    single_rater_ceiling = sum(modal_shares) / len(modal_shares)

    rho_panel = spearman_brown(single_rater_alpha, 2)
    return {
        "random_rater_ceiling": {
            "value": _r(measured_exact_match),
            "basis": (
                "measured: a real held-out annotator scored against the other two, over "
                "the posts where those two agreed"
            ),
            "n": sum(1 for i in items if i.panel_consensus is not None and i.judge_label),
        },
        "single_rater_modal_ceiling": {
            "value": _r(single_rater_ceiling),
            "basis": (
                "mean modal vote share over all posts: the best expected match of any "
                "predictor against one randomly drawn annotator. No conditioning on panel "
                "agreement, so no selection effect."
            ),
            "n": len(modal_shares),
        },
        "degenerate_consensus_bound": {
            "value": _r(oracle_hits / oracle_n) if oracle_n else None,
            "n": oracle_n,
            "is_degenerate": True,
            "basis": (
                "An oracle naming the modal label of all three annotators, scored against "
                "the two-annotator consensus. Mechanically 1.0: if two of three raters "
                "agree, their shared label is the modal one. Reported to show that the "
                "'what is the maximum score' question has a trivial answer against a "
                "consensus target and therefore has to be asked differently."
            ),
        },
        "alpha_reference": {
            "human_single_rater_alpha": _r(single_rater_alpha),
            "measured_judge_vs_panel_alpha": _r(measured_alpha),
            "two_rater_panel_reliability_spearman_brown": _r(rho_panel),
            "note": (
                "A reference point, not a hard bound: a systematic judge can track the "
                "panel's systematic component better than a second noisy human does."
            ),
        },
        "headline": (
            "A judge on this task should be measured against {c:.1%}, not against 100%. "
            "That is what a real human annotator from the same pool scores against two of "
            "their peers on the posts where those peers agreed. Against a single randomly "
            "drawn annotator across all posts, the best any predictor can do is {s:.1%}. "
            "The missing {gap:.1%} is disagreement among the humans, and no judge, prompt, "
            "or model scale removes it. The arithmetic upper bound against a two-rater "
            "consensus is exactly 1.0 and is meaningless, because the consensus is not a "
            "function of the text while a judge is."
        ).format(
            c=measured_exact_match,
            s=single_rater_ceiling,
            gap=1 - measured_exact_match,
        ),
    }


def disattenuate(
    observed_alpha: float, single_rater_alpha: float, judge_test_retest: float = 1.0
) -> dict:
    """Correct judge-panel agreement for the panel's own unreliability.

    Spearman's 1904 correction: an observed association between two noisy
    measures understates the association between the underlying quantities by a
    factor of ``sqrt(rel_x * rel_y)``. Here ``y`` is the two-annotator panel,
    whose reliability is the single-rater alpha stepped up by Spearman-Brown,
    and ``x`` is the judge.

    Two variants are reported because the judge's own reliability depends on
    what the judge is:

    * **judge as a human rater** -- ``rel_x`` is the single-rater alpha. This is
      the correct correction for the held-out-annotator stand-in used here.
    * **judge as a deterministic model** -- ``rel_x = 1.0``. A model sampled at
      temperature 0 returns the same label every time, so it contributes no
      random error of its own; only the panel's noise attenuates the observed
      number. That is an *assumption about the judge*, stated here so it can be
      checked by re-running the judge and measuring its test-retest agreement,
      which any real deployment should do.

    Why the uncorrected number is misleading
    ----------------------------------------
    An uncorrected judge-panel agreement answers "how well does this judge
    reproduce these particular noisy labels", when the question being asked is
    almost always "how well does this judge measure the underlying construct".
    Those differ by the panel's unreliability, and at this dataset's reliability
    the difference is large. Quoting the uncorrected number penalises the judge
    for the panel's noise and makes every judge look worse than it is -- which
    in practice leads teams to keep tuning a judge that already hit the ceiling.

    Caveat, stated plainly: the correction is derived for product-moment
    correlations. Applying it to a nominal-scale alpha is a standard and useful
    approximation, not an identity, and the corrected values should be read as
    "roughly this much higher", not as precise estimates.
    """
    rho_panel = spearman_brown(single_rater_alpha, 2)
    as_human = attenuated_correlation(observed_alpha, single_rater_alpha, rho_panel)
    as_model = attenuated_correlation(observed_alpha, judge_test_retest, rho_panel)
    # ``attenuated_correlation`` clamps at 1.0. The unclamped ratio is kept
    # because an overshoot is diagnostic, not cosmetic -- see ``overshoot_note``.
    unclamped_human = observed_alpha / math.sqrt(
        max(1e-9, single_rater_alpha) * max(1e-9, rho_panel)
    )
    overshoot = unclamped_human > 1.0
    return {
        "observed_alpha_judge_vs_panel": _r(observed_alpha),
        "panel_reliability_rho2": _r(rho_panel),
        "human_single_rater_alpha": _r(single_rater_alpha),
        "corrected_if_judge_is_a_human_rater": _r(as_human),
        "corrected_if_judge_is_a_human_rater_unclamped": _r(unclamped_human),
        "corrected_if_judge_is_deterministic": _r(as_model),
        "assumed_judge_test_retest_reliability": judge_test_retest,
        "correction_overshoots_unity": bool(overshoot),
        "reading": (
            "Observed judge-panel alpha is {o:.4f}. The panel it is scored against has a "
            "reliability of only {p:.4f}, so the observed number is attenuated: it is "
            "measuring the judge through a noisy instrument. Correcting for the panel "
            "alone -- the right correction when the judge is a deterministic model -- "
            "gives {m:.4f}. That gap, {o:.4f} to {m:.4f}, is the size of the mistake made "
            "by quoting the uncorrected number: a judge sitting at {o:.2f} looks mediocre "
            "and is in fact tracking the construct at roughly {m:.2f}, and teams keep "
            "tuning judges that already hit the ceiling because of it."
        ).format(o=observed_alpha, p=rho_panel, m=as_model),
        "overshoot_note": (
            (
                "Correcting for BOTH sides -- right for the human stand-in, whose own "
                "single-rater reliability is {r:.4f} -- gives {u:.4f} unclamped, which "
                "exceeds 1.0 and is reported clamped to {c:.4f}. An overshoot is not a "
                "rounding artefact, it is the model telling you one of its inputs is "
                "wrong. Here the cause is identifiable: the observed alpha is computed "
                "only over posts where the two panel members agreed, and that selection "
                "keeps the easy posts. The judge's agreement on a selected-easy subset is "
                "being divided by a reliability estimated on the whole corpus, so the "
                "numerator and denominator are not describing the same population. The "
                "unconditioned judge-versus-individual-panel-member figure reported "
                "alongside is the selection-free comparison; the deterministic-judge "
                "correction is the one to quote."
            ).format(r=single_rater_alpha, u=unclamped_human, c=as_human)
            if overshoot
            else (
                "The two-sided correction is within [0, 1], so the attenuation model is "
                "not being pushed past its range here."
            )
        ),
        "method_caveat": (
            "Spearman's disattenuation is derived for product-moment correlations; alpha "
            "on a nominal scale is not one. Treat these as approximations -- 'roughly this "
            "much higher', not precise estimates."
        ),
    }


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------


def run_study_b(
    dataset: Dataset,
    judge: JudgeProtocol | None = None,
    single_rater_alpha: float | None = None,
    permutation_sample: int = 2000,
    n_perm: int = 1000,
) -> dict:
    """Run the judge-validation harness and return the structured result."""
    plan = build_holdout_plan(dataset)
    judge = judge or judge_from_env(dataset, plan)
    baseline_label = dataset.label_counts().most_common(1)[0][0]
    baseline = ConstantJudge(constant=baseline_label)

    if single_rater_alpha is None:
        single_rater_alpha = krippendorff_alpha(
            reliability_matrix(dataset, collapse="three_way"), metric="nominal"
        ).value
    binary_alpha = krippendorff_alpha(
        reliability_matrix(dataset, collapse="binary"), metric="nominal"
    ).value

    results: dict[str, dict] = {}
    for collapse, cats, rater_alpha in (
        ("three_way", list(LABELS), single_rater_alpha),
        ("binary", ["normal", "toxic"], binary_alpha),
    ):
        items = build_panels(dataset, judge, plan, collapse=collapse)
        scores = score_judge(items, cats)
        base_items = build_panels(dataset, baseline, plan, collapse=collapse)
        base_scores = score_judge(base_items, cats)

        # Paired permutation test: is the human judge actually better than the
        # prior-only baseline? Run on a seeded subsample because the test is
        # O(n_perm * n) and the answer does not need 17,000 items to be clear.
        paired_ids = [
            i.post_id
            for i in items
            if i.panel_consensus is not None and i.judge_label is not None
        ]
        rng = random.Random(A.SEED_PERMUTATION)
        sample_ids = (
            rng.sample(paired_ids, permutation_sample)
            if len(paired_ids) > permutation_sample
            else paired_ids
        )
        j_by_id = {i.post_id: i for i in items}
        b_by_id = {i.post_id: i for i in base_items}
        a_vec = [1.0 if j_by_id[p].judge_label == j_by_id[p].panel_consensus else 0.0
                 for p in sample_ids]
        b_vec = [1.0 if b_by_id[p].judge_label == b_by_id[p].panel_consensus else 0.0
                 for p in sample_ids]
        perm = paired_permutation_test(a_vec, b_vec, n_perm=n_perm, seed=A.SEED_PERMUTATION)
        perm["n_sampled_from"] = len(paired_ids)
        perm["note"] = (
            "Paired permutation test on per-item exact-match, held-out human judge minus "
            f"the constant '{baseline_label}' baseline, on a seeded subsample of "
            f"{len(sample_ids)} scored posts."
        )

        ceiling = judge_ceiling(
            dataset,
            items,
            plan,
            rater_alpha,
            scores["exact_match"],
            scores["krippendorff_alpha_judge_vs_panel"],
            collapse=collapse,
        )
        results[collapse] = {
            "judge": scores,
            "constant_baseline": {
                "label": baseline_label,
                "exact_match": base_scores["exact_match"],
                "n_scored": base_scores["n_scored"],
            },
            "judge_minus_baseline": _r(
                (scores["exact_match"] or 0) - (base_scores["exact_match"] or 0)
            ),
            "paired_permutation_vs_baseline": perm,
            "ceiling": ceiling,
            "disattenuation": disattenuate(
                scores["krippendorff_alpha_judge_vs_panel"], rater_alpha
            ),
        }

    three = results["three_way"]
    corpus_raw = percent_agreement(
        reliability_matrix(dataset, collapse="three_way")
    ).value
    uncond = three["judge"]["judge_vs_individual_panel_members_unconditioned"]["exact_match"]
    validation = {
        "corpus_pairwise_agreement": _r(corpus_raw),
        "judge_vs_panel_member_unconditioned": uncond,
        "absolute_difference": _r(abs(uncond - corpus_raw)),
        "passes": bool(abs(uncond - corpus_raw) < 0.01),
        "why_this_is_the_right_check": (
            "The held-out human is just an annotator, so scoring them against each panel "
            "member with no conditioning must reproduce the corpus-wide pairwise agreement "
            "to within sampling noise. It uses two of the three pairs per post rather than "
            "all three, so exact equality is not expected. If these two numbers diverged, "
            "the panel construction would be leaking the judge's own label or dropping "
            "posts non-randomly -- the two failure modes that make a judge harness report "
            "a flattering number. They agree, so the scoring path is sound end to end, "
            "which is the entire purpose of running a stand-in judge before spending "
            "inference budget on a real one."
        ),
    }
    return {
        "study": "B",
        "title": "Judge-validation harness, validated with a held-out human at zero budget",
        "honesty_statement": (
            "The judge in this study is a held-out human annotator, not a language model. "
            "No model was called and this package makes no network requests. The purpose "
            "is to exercise the scoring harness end to end on real labels and to establish "
            "the ceiling an LLM judge would be measured against. Nothing here is an "
            "evaluation of any model's performance."
        ),
        "assumptions": A.assumptions_block(),
        "design": {
            "judge_name": judge.name,
            "holdout_seed": plan.seed,
            "n_posts_in_plan": len(plan.held_out),
            "panel_size": 2,
            "selection": (
                "One of the three annotators is chosen uniformly at random per post with a "
                "fixed seed and moved into the judge slot; the other two form the panel. "
                "The plan is built once so the judge's own label can never leak into the "
                "panel that scores it."
            ),
            "judge_env_var": A.JUDGE_ENV_VAR,
            "judge_env_values": ["heldout (default)", "constant:<label>", "api:<model>"],
            "swap_in_a_real_judge": (
                "Implement JudgeProtocol.label and select it with "
                f"{A.JUDGE_ENV_VAR}=api:<model>. Nothing else in this study changes: the "
                "panels, exact match, per-label recall, alpha, disattenuation and ceiling "
                "comparison all run against the protocol, not against the human."
            ),
        },
        "harness_validation": validation,
        "three_way": results["three_way"],
        "binary": results["binary"],
        "headline": (
            "A real human annotator, scored against two of their peers, matches the panel "
            "on {em:.1%} of the {n:,} posts where those two peers agreed, with a "
            "judge-panel alpha of {a:.4f}. The two-annotator panel itself reaches "
            "consensus on only {pc:.1%} of posts, so more than a third of the corpus has "
            "no target label under this design at all. With no conditioning on panel "
            "agreement the same human matches an individual panel member {un:.1%} of the "
            "time. Correcting the judge-panel alpha for the panel's own unreliability "
            "lifts it to {corr:.4f}. Those are the numbers an LLM judge on this task "
            "should be measured against -- not 1.0, and not {em:.1%} read as though the "
            "missing {gap:.1%} were the judge's fault."
        ).format(
            em=three["judge"]["exact_match"],
            n=three["judge"]["n_scored"],
            a=three["judge"]["krippendorff_alpha_judge_vs_panel"],
            pc=three["judge"]["panel_consensus_rate"],
            un=three["judge"]["judge_vs_individual_panel_members_unconditioned"][
                "exact_match"
            ],
            corr=three["disattenuation"]["corrected_if_judge_is_deterministic"],
            gap=1 - three["judge"]["exact_match"],
        ),
    }


__all__ = [
    "APIJudge",
    "ConstantJudge",
    "HeldOutAnnotatorJudge",
    "HoldoutPlan",
    "JudgeProtocol",
    "PanelItem",
    "build_holdout_plan",
    "build_panels",
    "disattenuate",
    "judge_ceiling",
    "judge_from_env",
    "run_study_b",
    "score_judge",
]
