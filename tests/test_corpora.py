"""The uniform matrix-building rules every corpus goes through (PREREG.md section 1)."""
from rubricon_field.corpora import build_matrix, collapse


def test_drops_single_rated_items_and_counts_them():
    rows = [("a", "r1", 1), ("a", "r2", 0), ("b", "r1", 1)]
    m, info = build_matrix(rows, "t")
    assert set(m) == {"a"} and info["items_dropped_lt2_ratings"] == 1


def test_duplicate_item_rater_keeps_first_and_counts():
    rows = [("a", "r1", 1), ("a", "r1", 0), ("a", "r2", 1)]
    m, info = build_matrix(rows, "t")
    assert m["a"] == {"r1": 1, "r2": 1} and info["duplicate_item_rater_rows"] == 1


def test_cap_is_deterministic_and_independent_of_row_order():
    rows = [("a", f"r{i}", i % 2) for i in range(20)]
    m1, info = build_matrix(rows, "t", cap=5)
    m2, _ = build_matrix(list(reversed(rows)), "t", cap=5)
    assert m1 == m2 and len(m1["a"]) == 5
    assert info["items_capped_to_max"] == 1 and info["ratings_on_capped_items"] == 20


def test_no_cap_keeps_everything():
    rows = [("a", f"r{i}", 1) for i in range(12)]
    m, _ = build_matrix(rows, "t", cap=None)
    assert len(m["a"]) == 12


def test_collapse_drops_none_and_underfilled_items():
    m = {"a": {"r1": "x", "r2": "y"}, "b": {"r1": "x", "r2": "z", "r3": "z"}}
    out = collapse(m, lambda v: None if v == "z" else v)
    assert out == {"a": {"r1": "x", "r2": "y"}}
