"""Statistical checks: the verified headline numbers and the projection maths.

The agreement coefficients themselves are tested in the ``rubricon`` repository
against independent reference implementations. What is tested here is that this
study *applies* them to the right object and reproduces the published shape of
the corpus -- a wrong matrix produces a perfectly valid alpha of the wrong thing.
"""

from __future__ import annotations

import math

from rubricon.stats.agreement import (
    fleiss_kappa,
    gwet_ac1,
    krippendorff_alpha,
    percent_agreement,
    spearman_brown,
)

from rubricon_field.data import Dataset, reliability_matrix
from rubricon_field.study_c import raters_needed, reliability_projection

# Verified headline values for the 3-way nominal task.
ALPHA = 0.4597
RAW = 0.6439
KAPPA = 0.4597
AC1 = 0.4689


def test_headline_agreement_reproduces(dataset: Dataset) -> None:
    m = reliability_matrix(dataset, collapse="three_way")
    assert round(krippendorff_alpha(m, metric="nominal").value, 4) == ALPHA
    assert round(percent_agreement(m).value, 4) == RAW
    assert round(fleiss_kappa(m).value, 4) == KAPPA
    assert round(gwet_ac1(m).value, 4) == AC1


def test_no_kappa_paradox(dataset: Dataset) -> None:
    """The paradox needs high raw agreement and a large AC1-kappa gap. Neither holds."""
    m = reliability_matrix(dataset, collapse="three_way")
    raw = percent_agreement(m).value
    kappa = fleiss_kappa(m).value
    ac1 = gwet_ac1(m).value
    assert raw < 0.85
    assert ac1 - kappa < 0.25
    assert not (raw >= 0.85 and kappa < 0.40 and (ac1 - kappa) > 0.25)


def test_binary_collapse_is_more_reliable(dataset: Dataset) -> None:
    """The key contrast: the binary task the classifiers train on is the cleaner one."""
    three = krippendorff_alpha(
        reliability_matrix(dataset, collapse="three_way"), metric="nominal"
    ).value
    binary = krippendorff_alpha(
        reliability_matrix(dataset, collapse="binary"), metric="nominal"
    ).value
    assert binary > three
    assert round(binary - three, 3) == 0.102


def test_alpha_is_a_single_rater_coefficient(dataset: Dataset) -> None:
    """Dropping one rater per post leaves alpha where it was, within 0.005.

    This is the premise of the whole Spearman-Brown projection in study C: alpha
    estimates the reliability of ONE annotator regardless of panel size. If it
    moved with panel size, stepping it up by Spearman-Brown would be
    double-counting.
    """
    full = krippendorff_alpha(
        reliability_matrix(dataset, collapse="three_way"), metric="nominal"
    ).value
    for drop_pos in range(3):
        matrix = {}
        for pid in dataset.post_ids:
            anns = dataset.posts[pid].annotations
            kept = [a for i, a in enumerate(anns) if i != drop_pos]
            matrix[pid] = {a.annotator_id: a.label for a in kept}
        assert abs(krippendorff_alpha(matrix, metric="nominal").value - full) < 0.005


# --------------------------------------------------------------------------
# Spearman-Brown
# --------------------------------------------------------------------------


def test_spearman_brown_hand_worked_value() -> None:
    """rho_1 = 0.5, k = 3 -> 3*0.5 / (1 + 2*0.5) = 1.5 / 2.0 = 0.75, exactly."""
    assert math.isclose(spearman_brown(0.5, 3), 0.75, rel_tol=0, abs_tol=1e-12)


def test_spearman_brown_hand_worked_on_this_corpus() -> None:
    """rho_1 = 0.4597, k = 3 -> 1.3791 / 1.9194 = 0.718506...

    Worked by hand rather than snapshotted, so the test fails if the formula
    changes rather than merely recording whatever it currently produces.
    """
    expected = 3 * 0.4597 / (1 + 2 * 0.4597)
    assert math.isclose(spearman_brown(0.4597, 3), expected, abs_tol=1e-12)
    assert round(expected, 4) == 0.7185


def test_spearman_brown_identity_at_k_one() -> None:
    for rho in (0.1, 0.4597, 0.9):
        assert math.isclose(spearman_brown(rho, 1), rho, abs_tol=1e-12)


def test_spearman_brown_is_monotone_and_bounded() -> None:
    for rho in (0.05, 0.2, 0.4597, 0.8):
        values = [spearman_brown(rho, k) for k in range(1, 21)]
        assert all(b >= a for a, b in zip(values, values[1:])), rho
        assert all(0.0 <= v <= 1.0 for v in values)
        assert values[-1] > values[0]


def test_raters_needed_matches_closed_form() -> None:
    for rho in (0.2, 0.4597, 0.6):
        for target in (0.667, 0.80, 0.90):
            k = raters_needed(rho, target)
            assert k is not None
            exact = target * (1 - rho) / (rho * (1 - target))
            closed = max(1, math.ceil(round(exact, 9)))
            # Two knife-edges make an exact equality assertion wrong here, and
            # both are floating point rather than arithmetic:
            #   * at rho=0.2, target=0.8 the closed form is exactly 16 but
            #     evaluates to 16.000000000000004, so a bare ceil() gives 17;
            #   * at rho=0.6, target=0.9 the closed form is exactly 6 and
            #     Spearman-Brown at k=6 evaluates to 0.8999999999999999, so the
            #     search correctly refuses it and returns 7.
            # The search is the authority; it can only ever be off by one from
            # the closed form, and only when the closed form is an exact integer.
            assert k in (closed, closed + 1)
            if k != closed:
                assert math.isclose(exact, round(exact), abs_tol=1e-9)
            assert spearman_brown(rho, k) >= target
            if k > 1:
                assert spearman_brown(rho, k - 1) < target + 1e-9


def test_projection_targets_on_real_data(dataset: Dataset) -> None:
    proj = reliability_projection(dataset)
    assert proj["single_rater_alpha"] == ALPHA
    assert proj["three_rater_reliability"] == 0.7185
    assert proj["targets"]["0.667"]["raters_required"] == 3
    assert proj["targets"]["0.800"]["raters_required"] == 5
    values = [row["reliability"] for row in proj["projection"]]
    assert values == sorted(values)
