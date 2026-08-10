"""End-to-end tests: the three studies run on a small slice and produce sane shapes.

Run on a 600-post slice rather than the full corpus so the suite stays fast. The
slice is deterministic (the first 600 post ids in sorted order), so a failure is
reproducible.
"""

from __future__ import annotations

import json

import pytest

from rubricon_field.data import Dataset
from rubricon_field.study_a import run_study_a
from rubricon_field.study_b import (
    ConstantJudge,
    HeldOutAnnotatorJudge,
    JudgeProtocol,
    build_holdout_plan,
    build_panels,
    disattenuate,
    judge_from_env,
    run_study_b,
    score_judge,
)
from rubricon_field.study_c import run_study_c
from rubricon_field.assumptions import JUDGE_ENV_VAR


# --------------------------------------------------------------------------
# study A
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def study_a(slice_dataset: Dataset) -> dict:
    return run_study_a(slice_dataset)


def test_study_a_runs_and_is_json_serialisable(study_a: dict) -> None:
    text = json.dumps(study_a, sort_keys=True)
    assert len(text) > 5000
    assert json.loads(text)["study"] == "A"


def test_study_a_sections_present(study_a: dict) -> None:
    for key in (
        "overall_agreement",
        "consensus_structure",
        "label_confusion",
        "binary_contrast",
        "annotators",
        "agreement_by_target",
        "accuracy_ceiling",
        "signal_gate",
        "assumptions",
    ):
        assert key in study_a, key


def test_study_a_consensus_partitions_the_slice(study_a: dict, slice_dataset: Dataset) -> None:
    counts = study_a["consensus_structure"]["three_way"]["counts"]
    assert sum(counts.values()) == slice_dataset.n_posts


def test_study_a_records_bootstrap_subsampling(study_a: dict) -> None:
    basis = study_a["overall_agreement"]["battery"]["ci_basis"]
    assert basis["point_estimates_computed_on"] >= basis["interval_computed_on"]
    assert basis["n_boot"] > 0
    assert "seed" in basis


def test_study_a_ceiling_is_between_chance_and_one(study_a: dict) -> None:
    ceiling = study_a["accuracy_ceiling"]["ceiling_against_a_human"]
    assert 1 / 3 < ceiling["three_way"] < 1.0
    assert ceiling["binary"] > ceiling["three_way"]


def test_study_a_mde_shrinks_with_n(study_a: dict) -> None:
    grid = study_a["accuracy_ceiling"]["minimum_detectable_effect"]["grid"]
    cells = sorted(
        (g for g in grid if g["assumed_pairwise_disagreement"] == 0.20),
        key=lambda g: g["n_items"],
    )
    values = [g["mde_observed_scale"] for g in cells]
    assert values == sorted(values, reverse=True)
    # The attenuation correction always makes the true-scale MDE the larger one,
    # and the two must never be compounded into a single figure.
    assert all(g["mde_true_scale"] > g["mde_observed_scale"] for g in cells)


def test_study_a_gate_produces_verdicts(study_a: dict) -> None:
    gate = study_a["signal_gate"]["default_policy"]
    verdicts = {c["claim_id"]: c["verdict"] for c in gate["claims"]}
    assert len(verdicts) == 7
    assert set(verdicts.values()) <= {"pass", "warn", "block"}
    # A 3-way measurement claim must not survive an alpha below the blocking floor.
    assert verdicts["C-MEAS-01"] == "block"
    # Blocked claims stay in the ledger rather than disappearing.
    assert len(gate["claims"]) == gate["summary"]["n_claims"]


def test_study_a_exploratory_policy_is_not_stricter(study_a: dict) -> None:
    default = set(study_a["signal_gate"]["default_policy"]["blocked_claim_ids"])
    exploratory = set(study_a["signal_gate"]["exploratory_policy"]["blocked_claim_ids"])
    assert exploratory <= default


def test_study_a_thin_communities_are_marked(study_a: dict) -> None:
    rows = study_a["agreement_by_target"]["rows"]
    floor = study_a["agreement_by_target"]["size_floor"]
    for r in rows:
        assert r["sufficient_for_claim"] == (r["n_posts"] >= floor)
        if not r["sufficient_for_claim"]:
            assert r["claim_note"]
            assert r["ci_alpha_three_way"] is None


# --------------------------------------------------------------------------
# study B
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def study_b(slice_dataset: Dataset) -> dict:
    return run_study_b(slice_dataset, permutation_sample=200, n_perm=200)


def test_study_b_runs(study_b: dict) -> None:
    assert study_b["study"] == "B"
    assert json.loads(json.dumps(study_b, sort_keys=True))


def test_study_b_states_it_is_not_an_llm(study_b: dict) -> None:
    statement = study_b["honesty_statement"].lower()
    assert "not a language model" in statement
    assert study_b["design"]["judge_name"] == "held_out_human"


def test_holdout_plan_is_deterministic(slice_dataset: Dataset) -> None:
    a = build_holdout_plan(slice_dataset)
    b = build_holdout_plan(slice_dataset)
    assert a.held_out == b.held_out
    assert len(a.held_out) == slice_dataset.n_posts


def test_judge_label_is_never_in_its_own_panel(slice_dataset: Dataset) -> None:
    """The leak that would silently inflate every judge score."""
    plan = build_holdout_plan(slice_dataset)
    judge = HeldOutAnnotatorJudge(slice_dataset, plan)
    for pid in slice_dataset.post_ids:
        post = slice_dataset.posts[pid]
        held = plan.annotator_for(pid)
        panel_ids = [a.annotator_id for a in post.annotations if a.annotator_id != held]
        assert held not in panel_ids
        assert len(panel_ids) == 2
        assert judge.label(post) is not None


def test_panels_exclude_split_panels_from_scoring(slice_dataset: Dataset) -> None:
    plan = build_holdout_plan(slice_dataset)
    judge = HeldOutAnnotatorJudge(slice_dataset, plan)
    items = build_panels(slice_dataset, judge, plan)
    scored = score_judge(items, ["normal", "offensive", "hatespeech"])
    n_split = sum(1 for i in items if i.panel_consensus is None)
    assert scored["n_scored"] == len(items) - n_split
    assert scored["n_panel_without_consensus"] == n_split


def test_study_b_harness_validation_passes(study_b: dict) -> None:
    """The stand-in judge must reproduce the corpus pairwise agreement."""
    v = study_b["harness_validation"]
    assert v["passes"], v
    assert v["absolute_difference"] < 0.01


def test_study_b_beats_the_constant_baseline(study_b: dict) -> None:
    three = study_b["three_way"]
    assert three["judge"]["exact_match"] > three["constant_baseline"]["exact_match"]
    assert three["paired_permutation_vs_baseline"]["p_value"] < 0.05


def test_study_b_binary_is_easier_than_three_way(study_b: dict) -> None:
    assert study_b["binary"]["judge"]["exact_match"] > study_b["three_way"]["judge"][
        "exact_match"
    ]


def test_disattenuation_raises_the_observed_value() -> None:
    out = disattenuate(observed_alpha=0.50, single_rater_alpha=0.4597)
    assert out["corrected_if_judge_is_deterministic"] > 0.50
    assert out["panel_reliability_rho2"] == 0.6299  # 2*0.4597 / (1 + 0.4597)


def test_disattenuation_flags_an_overshoot() -> None:
    out = disattenuate(observed_alpha=0.95, single_rater_alpha=0.4597)
    assert out["correction_overshoots_unity"] is True
    assert out["corrected_if_judge_is_a_human_rater"] == 1.0
    assert out["corrected_if_judge_is_a_human_rater_unclamped"] > 1.0
    assert "selection" in out["overshoot_note"]


def test_constant_judge_conforms_to_protocol(slice_dataset: Dataset) -> None:
    judge = ConstantJudge("normal")
    assert isinstance(judge, JudgeProtocol)
    post = slice_dataset.posts[slice_dataset.post_ids[0]]
    assert judge.label(post) == "normal"


def test_heldout_judge_conforms_to_protocol(slice_dataset: Dataset) -> None:
    plan = build_holdout_plan(slice_dataset)
    assert isinstance(HeldOutAnnotatorJudge(slice_dataset, plan), JudgeProtocol)


def test_judge_from_env_default(slice_dataset: Dataset, monkeypatch) -> None:
    monkeypatch.delenv(JUDGE_ENV_VAR, raising=False)
    plan = build_holdout_plan(slice_dataset)
    assert isinstance(judge_from_env(slice_dataset, plan), HeldOutAnnotatorJudge)


def test_judge_from_env_constant(slice_dataset: Dataset, monkeypatch) -> None:
    monkeypatch.setenv(JUDGE_ENV_VAR, "constant:hatespeech")
    plan = build_holdout_plan(slice_dataset)
    judge = judge_from_env(slice_dataset, plan)
    assert isinstance(judge, ConstantJudge) and judge.constant == "hatespeech"


def test_judge_from_env_rejects_unknown(slice_dataset: Dataset, monkeypatch) -> None:
    monkeypatch.setenv(JUDGE_ENV_VAR, "gpt-please")
    plan = build_holdout_plan(slice_dataset)
    with pytest.raises(ValueError):
        judge_from_env(slice_dataset, plan)


def test_api_judge_is_not_exercised(slice_dataset: Dataset, monkeypatch) -> None:
    """The adapter exists but must refuse to run: this repo makes no network calls."""
    monkeypatch.setenv(JUDGE_ENV_VAR, "api:some-model")
    plan = build_holdout_plan(slice_dataset)
    judge = judge_from_env(slice_dataset, plan)
    assert judge.name == "api_judge:some-model"
    with pytest.raises(NotImplementedError):
        judge.label(slice_dataset.posts[slice_dataset.post_ids[0]])


# --------------------------------------------------------------------------
# study C
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def study_c(slice_dataset: Dataset) -> dict:
    return run_study_c(slice_dataset)


def test_study_c_runs(study_c: dict) -> None:
    assert study_c["study"] == "C"
    assert json.loads(json.dumps(study_c, sort_keys=True))


def test_study_c_projection_is_monotone(study_c: dict) -> None:
    values = [r["reliability"] for r in study_c["reliability_projection"]["projection"]]
    assert values == sorted(values)


def test_study_c_allocation_prefers_one_rater(study_c: dict) -> None:
    """The analytic result: k=1 minimises the true-scale MDE at any reliability."""
    alloc = study_c["budget_allocation"]
    assert alloc["optimum_k_at_headline_rate"] == 1
    assert alloc["optimum_is_invariant_to_disagreement_assumption"] is True
    rows = sorted(
        (r for r in alloc["grid"] if r["assumed_pairwise_disagreement"] == 0.20),
        key=lambda r: r["k_raters"],
    )
    mdes = [r["mde_true_scale"] for r in rows]
    assert mdes == sorted(mdes)


def test_study_c_costs_scale_with_the_named_unit_cost(study_c: dict) -> None:
    costs = study_c["cost_model"]
    unit = costs["unit_cost_usd_per_label"]
    for row in costs["by_rater_count"]:
        assert row["cost_usd"] == round(row["labels"] * unit, 2)


def test_study_c_hybrid_is_cheaper_than_uniform_triple(study_c: dict) -> None:
    costs = study_c["cost_model"]
    assert costs["hybrid_design"]["cost_usd"] < costs["as_published"]["cost_usd"]
    assert costs["hybrid_design"]["saving_vs_uniform_triple_usd"] > 0
