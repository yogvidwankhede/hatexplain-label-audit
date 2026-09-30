"""fast_alpha must agree with rubricon's validated implementation."""
import random

import numpy as np
import pytest
from rubricon.stats.agreement import krippendorff_alpha

from rubricon_field import fast_alpha as F


def _random_matrix(seed, n_units=400, cats=(0, 1, 2), max_r=6):
    rng = random.Random(seed)
    m = {}
    for u in range(n_units):
        k = rng.randint(1, max_r)
        base = rng.choice(cats)
        m[f"u{u}"] = {f"r{rng.randint(0, 40)}_{j}": (base if rng.random() < 0.6 else rng.choice(cats))
                      for j in range(k)}
    return m


@pytest.mark.parametrize("metric", ["nominal", "ordinal", "interval"])
def test_point_matches_rubricon(metric):
    for seed in range(4):
        m = _random_matrix(seed, cats=(-2, -1, 0, 1, 2) if metric != "nominal" else ("a", "b", "c"))
        ref = krippendorff_alpha(m, metric).value
        assert abs(F.alpha(m, metric) - ref) < 1e-9


@pytest.mark.parametrize("metric", ["nominal", "ordinal"])
def test_weighted_replicate_equals_duplicated_units(metric):
    m = _random_matrix(9, n_units=120, cats=(0, 1, 2, 3))
    cats, counts = F.unit_counts(m)
    keys = [k for k, v in m.items() if len(v) >= 2]
    rng = np.random.default_rng(3)
    w = rng.multinomial(len(keys), np.full(len(keys), 1 / len(keys)))
    dup = {}
    for i, key in enumerate(keys):
        for r in range(w[i]):
            dup[f"{key}#{r}"] = m[key]
    ref = krippendorff_alpha(dup, metric).value
    got = F.alpha_from_weights(F._pair_tensor(counts), w[None, :].astype(float), metric,
                               np.array(cats, float) if metric != "nominal" else None)[0]
    assert abs(got - ref) < 1e-9


def test_bootstrap_contains_point_and_is_deterministic():
    m = _random_matrix(5, n_units=800)
    a = F.bootstrap(m, n_boot=200, seed=1)
    b = F.bootstrap(m, n_boot=200, seed=1)
    assert a == b and a["lo"] <= a["point"] <= a["hi"]
