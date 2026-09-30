"""Property-based tests: the vectorised alpha must equal rubricon's reference on any input."""
import math

import numpy as np
from hypothesis import given, settings, strategies as st
from rubricon.stats.agreement import krippendorff_alpha

from rubricon_field import fast_alpha as F
from rubricon_field.corpora import build_matrix

cells = st.lists(st.lists(st.integers(-2, 2), min_size=2, max_size=6), min_size=3, max_size=40)


def _m(rows):
    return {f"u{i}": {f"r{j}": v for j, v in enumerate(r)} for i, r in enumerate(rows)}


@given(cells, st.sampled_from(["nominal", "ordinal", "interval"]))
@settings(max_examples=300, deadline=None)
def test_fast_alpha_equals_reference(rows, metric):
    m = _m(rows)
    ref = krippendorff_alpha(m, metric).value
    got = F.alpha(m, metric)
    assert (math.isnan(ref) and math.isnan(got)) or abs(ref - got) < 1e-9


@given(cells, st.integers(0, 2**31 - 1))
@settings(max_examples=100, deadline=None)
def test_integer_weights_equal_duplicated_units(rows, seed):
    m = _m(rows)
    cats, counts = F.unit_counts(m)
    w = np.random.default_rng(seed).integers(0, 3, size=len(counts))
    if w.sum() == 0:
        return
    dup = {f"{k}#{r}": m[k] for i, k in enumerate(m) for r in range(w[i])}
    ref = krippendorff_alpha(dup, "nominal").value
    got = F.alpha_from_weights(F._pair_tensor(counts), w[None, :].astype(float), "nominal")[0]
    assert (math.isnan(ref) and math.isnan(got)) or abs(ref - got) < 1e-9


@given(st.lists(st.tuples(st.sampled_from("abcdef"), st.sampled_from("pqrstuvw"), st.integers(0, 1)),
                max_size=120), st.integers(2, 6))
@settings(max_examples=200, deadline=None)
def test_build_matrix_respects_cap_and_minimum(rows, cap):
    m, info = build_matrix(rows, "prop", cap=cap)
    assert all(2 <= len(c) <= cap for c in m.values())
    assert info["n_items_used"] == len(m)
