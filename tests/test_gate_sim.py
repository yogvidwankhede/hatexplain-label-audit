"""Tests for the gate simulation: numerics are cross-checked, not trusted."""
import numpy as np
from rubricon.stats.agreement import krippendorff_alpha

from rubricon_field import gate_sim as G


def test_fast_alpha_matches_rubricon_reference():
    rng = np.random.default_rng(1)
    for k in (2, 3, 5):
        s = rng.binomial(k, rng.choice([0.2, 0.5, 0.8], size=300)).astype(int)
        matrix = {}
        for i, si in enumerate(s):
            votes = [1] * int(si) + [0] * (k - int(si))
            matrix[f"u{i}"] = {f"r{j}": v for j, v in enumerate(votes)}
        ref = krippendorff_alpha(matrix, "nominal").value
        assert abs(ref - float(G.alpha_binary_fast(s, k))) < 1e-9


def test_solved_eta_hits_target_alpha():
    for scen in ("iid", "het"):
        for target in (0.4, 0.6, 0.9):
            e = G.solve_etas(target, 0.3, scen)
            w = [(1.0, e[0])] if scen == "iid" else [(0.7, e[0]), (0.3, e[1])]
            assert abs(G.population_alpha(w, 0.3) - target) < 1e-6


def test_iid_noise_scales_gap_by_one_minus_two_eta_and_keeps_disagreement():
    r = G.simulate_cell("iid", 4000, 0.6, 0.3, 0.05, 0.2, reps=400, cell_id=7)
    eta = G.solve_etas(0.6, 0.3, "iid")[0]
    pred = (1 - 2 * G.eta_majority(eta)) * 0.05
    assert abs(r["mean_gap_all"] - pred) < 0.0015
    assert abs(r["mean_disc_obs"] - 0.2) < 0.003


def test_deterministic():
    a = G.simulate_cell("iid", 200, 0.5, 0.5, 0.02, 0.1, reps=50, cell_id=3)
    b = G.simulate_cell("iid", 200, 0.5, 0.5, 0.02, 0.1, reps=50, cell_id=3)
    assert a == b


def test_infeasible_cells_return_none():
    assert G.simulate_cell("het", 200, 0.2, 0.5, 0.0, 0.1, reps=10, cell_id=1) is None


def test_ablation_baseline_equals_gate_publish_count():
    from rubricon_field.gate_sim_report import publish_count
    for scen in ("iid", "het"):
        c = G.simulate_cell(scen, 500, 0.7, 0.3, 0.03, 0.2, reps=200, cell_id=11)
        assert publish_count(c, None) == c["n_gate_pub"]
        # removing a check can only publish more, never fewer
        assert all(publish_count(c, j) >= c["n_gate_pub"] for j in range(5))
