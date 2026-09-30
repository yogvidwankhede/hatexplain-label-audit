"""Vectorised Krippendorff's alpha and its item-cluster bootstrap.

Why this exists
---------------
``rubricon.stats.agreement.alpha_interval`` is pure Python and takes minutes per
interval on 10^5 items, so the first cross-corpus run bootstrapped a 10,000-item
subsample while the point estimate used every item. The two were then on
different bases, and on GoEmotions:amusement the full-data point fell outside
its own subsample interval. This module makes the full-data bootstrap cheap.

How
---
Alpha is a ratio of two sums over units, and a cluster bootstrap only re-weights
units. With per-unit category counts ``n_uc`` and ``m_u = sum_c n_uc``:

    coincidence  o_ck = sum_u w_u * n_uc * (n_uk - [c == k]) / (m_u - 1)
    marginals    n_c  = sum_c o_ck

so a replicate is two matrix products with a weight vector ``w`` (multinomial
counts). Alpha = 1 - D_o / D_e with the metric's distances, which for the
ordinal metric depend on the replicate's own marginals and are recomputed per
replicate, as Krippendorff defines them.

Correctness is not assumed: the tests compare the point estimate with
``rubricon.stats.agreement.krippendorff_alpha`` (itself validated against
Krippendorff's published reference values) on real corpora, and compare an
integer-weighted replicate with rubricon's value on the explicitly duplicated
units.
"""

from __future__ import annotations

import numpy as np


def unit_counts(matrix: dict, categories: list | None = None):
    """Return (categories, counts[U, C]) for units with >= 2 ratings."""
    cells = [c for c in matrix.values() if len(c) >= 2]
    if categories is None:
        categories = sorted({v for c in cells for v in c.values()}, key=lambda v: (str(type(v)), v))
    index = {c: i for i, c in enumerate(categories)}
    counts = np.zeros((len(cells), len(categories)), dtype=np.float64)
    for u, cell in enumerate(cells):
        for v in cell.values():
            counts[u, index[v]] += 1
    return categories, counts


def _pair_tensor(counts: np.ndarray) -> np.ndarray:
    """Per-unit coincidence contributions, shape (U, C, C)."""
    m = counts.sum(1)
    outer = counts[:, :, None] * counts[:, None, :]
    idx = np.arange(counts.shape[1])
    outer[:, idx, idx] -= counts
    return outer / (m - 1)[:, None, None]


def _delta2(metric: str, n_c: np.ndarray, values: np.ndarray | None) -> np.ndarray:
    """Squared distance matrix for one replicate, shape (C, C)."""
    c = len(n_c)
    if metric == "nominal":
        return 1.0 - np.eye(c)
    if metric == "interval":
        return (values[:, None] - values[None, :]) ** 2
    if metric == "ordinal":
        # delta_ck = (sum_{g=c..k} n_g - (n_c + n_k)/2)^2, categories in rank order
        cum = np.concatenate([[0.0], np.cumsum(n_c)])
        lo = np.minimum.outer(np.arange(c), np.arange(c))
        hi = np.maximum.outer(np.arange(c), np.arange(c))
        s = cum[hi + 1] - cum[lo]
        return (s - (n_c[:, None] + n_c[None, :]) / 2.0) ** 2
    raise ValueError(metric)


def alpha_from_weights(pairs: np.ndarray, weights: np.ndarray, metric: str = "nominal",
                       values: np.ndarray | None = None) -> np.ndarray:
    """Alpha for each row of ``weights`` (B, U). ``pairs`` from ``_pair_tensor``."""
    u, c, _ = pairs.shape
    o = (weights @ pairs.reshape(u, c * c)).reshape(-1, c, c)  # (B, C, C)
    n_c = o.sum(2)
    n = n_c.sum(1)
    out = np.empty(len(weights))
    for b in range(len(weights)):
        d2 = _delta2(metric, n_c[b], values)
        d_o = (o[b] * d2).sum() / n[b]
        d_e = (np.outer(n_c[b], n_c[b]) * d2).sum() / (n[b] * (n[b] - 1))
        out[b] = 1.0 - d_o / d_e if d_e > 0 else np.nan
    return out


def alpha(matrix: dict, metric: str = "nominal") -> float:
    cats, counts = unit_counts(matrix)
    values = np.array(cats, dtype=float) if metric != "nominal" else None
    return float(alpha_from_weights(_pair_tensor(counts), np.ones((1, len(counts))), metric, values)[0])


def bootstrap(matrix: dict, metric: str = "nominal", n_boot: int = 1000, seed: int = 0,
              level: float = 0.95, chunk: int = 50) -> dict:
    """Percentile item-cluster bootstrap over ALL pairable units."""
    cats, counts = unit_counts(matrix)
    values = np.array(cats, dtype=float) if metric != "nominal" else None
    pairs = _pair_tensor(counts)
    u = len(counts)
    rng = np.random.default_rng(seed)
    draws = []
    for start in range(0, n_boot, chunk):
        b = min(chunk, n_boot - start)
        w = rng.multinomial(u, np.full(u, 1.0 / u), size=b).astype(np.float64)
        draws.append(alpha_from_weights(pairs, w, metric, values))
    draws = np.sort(np.concatenate(draws))
    point = float(alpha_from_weights(pairs, np.ones((1, u)), metric, values)[0])
    a = (1 - level) / 2
    lo, hi = np.quantile(draws, [a, 1 - a])
    return {"point": round(point, 4), "lo": round(float(lo), 4), "hi": round(float(hi), 4),
            "level": level, "n_boot": n_boot, "n_clusters": u,
            "method": "cluster_bootstrap_percentile_all_units"}
