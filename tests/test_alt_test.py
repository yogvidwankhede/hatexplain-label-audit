import random

from statsmodels.stats.multitest import multipletests

from rubricon_field.alt_test import by_reject


def test_by_matches_statsmodels():
    r = random.Random(1)
    for m in (1, 5, 40, 120):
        p = [r.random() ** 3 for _ in range(m)]
        assert by_reject(p) == list(multipletests(p, alpha=0.05, method="fdr_by")[0])
