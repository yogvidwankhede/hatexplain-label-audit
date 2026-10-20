"""Guard the ordering, stored intervals and column width of the emotion plot."""

import importlib.util
import json
from pathlib import Path

import pytest


def load_figures():
    script = Path(__file__).resolve().parents[1] / "scripts" / "make_figures.py"
    spec = importlib.util.spec_from_file_location("make_figures", script)
    figures = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(figures)
    return figures


def test_goemotions_uses_sorted_results_and_single_column(monkeypatch, tmp_path):
    figures = load_figures()
    rows = [
        {"variant": "emotion:joy", "alpha": 0.7, "alpha_ci95": {"lo": 0.6, "hi": 0.8}},
        {"variant": "emotion:fear", "alpha": 0.2, "alpha_ci95": {"lo": 0.1, "hi": 0.3}},
    ]
    (tmp_path / "cross_goemotions.json").write_text(json.dumps({"variants": rows}))
    monkeypatch.setattr(figures, "R", tmp_path)
    captured = []
    monkeypatch.setattr(figures, "_save", lambda fig, name, **kwargs: captured.append((fig, name)))
    try:
        figures.fig_goemotions()
        fig, name = captured[0]
        ax = fig.axes[0]
        assert name == "goemotions"
        assert fig.get_size_inches()[0] == pytest.approx(figures.COL)
        assert [label.get_text() for label in ax.get_yticklabels()] == ["fear", "joy"]
        assert ax.yaxis_inverted()
        for i, row in enumerate(sorted(rows, key=lambda row: row["alpha"])):
            assert list(ax.lines[2 * i].get_xdata()) == [row["alpha_ci95"]["lo"], row["alpha_ci95"]["hi"]]
            assert list(ax.lines[2 * i + 1].get_xdata()) == [row["alpha"]]
            assert ax.lines[2 * i + 1].get_marker() == figures.M[0]
    finally:
        for fig, _ in captured:
            figures.plt.close(fig)


def test_goemotions_pdf_is_independent_of_creation_date(monkeypatch, tmp_path):
    figures = load_figures()
    monkeypatch.setattr(figures, "OUT", tmp_path)
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "0")
    figures.fig_goemotions()
    first = (tmp_path / "goemotions.pdf").read_bytes()
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "86400")
    figures.fig_goemotions()
    assert (tmp_path / "goemotions.pdf").read_bytes() == first
