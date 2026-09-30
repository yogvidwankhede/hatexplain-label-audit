"""CLI, determinism, and the README-provenance check.

The last of these is the one that matters most for a study like this. A report
whose prose has drifted away from its results is worse than no report, because
it is confidently wrong and nothing flags it. The test extracts every
statistic-shaped number from the README and requires it to appear in the
results, so a stale sentence fails the build.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from rubricon_field.cli import main

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results"
README = REPO_ROOT / "README.md"


@pytest.fixture(scope="module")
def small_corpus(tmp_path_factory, raw_records: dict) -> Path:
    """A 400-post JSON file, deterministic in content and ordering."""
    keys = sorted(raw_records)[:400]
    out = tmp_path_factory.mktemp("corpus") / "small.json"
    out.write_text(
        json.dumps({k: raw_records[k] for k in keys}, sort_keys=True), encoding="utf-8"
    )
    return out


def test_run_writes_all_artifacts(small_corpus: Path, tmp_path: Path) -> None:
    out = tmp_path / "results"
    assert main(["run", "--data", str(small_corpus), "--out", str(out)]) == 0
    for name in ("study_a.json", "study_b.json", "study_c.json", "summary.json",
                 "REPORT.md", "_manifest.json"):
        assert (out / name).exists(), name
    manifest = json.loads((out / "_manifest.json").read_text())
    assert set(manifest) >= {"study_a.json", "study_b.json", "study_c.json"}
    assert all(entry["sha256_12"] for entry in manifest.values())


def test_run_is_deterministic(small_corpus: Path, tmp_path: Path) -> None:
    """Two runs, byte-identical outputs.

    No timestamps, no unseeded randomness, no dict ordering leaking into the
    files. If this fails, no number in the repository is reproducible.
    """
    first, second = tmp_path / "a", tmp_path / "b"
    main(["run", "--data", str(small_corpus), "--out", str(first)])
    main(["run", "--data", str(small_corpus), "--out", str(second)])
    for name in ("study_a.json", "study_b.json", "study_c.json", "summary.json",
                 "REPORT.md"):
        assert (first / name).read_bytes() == (second / name).read_bytes(), name


def test_report_command_renders_from_results(small_corpus: Path, tmp_path: Path) -> None:
    out = tmp_path / "results"
    main(["run", "--data", str(small_corpus), "--out", str(out)])
    (out / "REPORT.md").unlink()
    assert main(["report", "--results", str(out)]) == 0
    text = (out / "REPORT.md").read_text(encoding="utf-8")
    assert "Study A" in text and "Study B" in text and "Study C" in text
    assert "Mathew" in text


def test_run_rejects_a_missing_dataset(tmp_path: Path) -> None:
    assert main(["run", "--data", str(tmp_path / "nope.json"), "--out", str(tmp_path)]) == 2


def test_report_without_results_fails_cleanly(tmp_path: Path) -> None:
    assert main(["report", "--results", str(tmp_path / "empty")]) == 2


def test_ascii_only() -> None:
    """No emoji, no smart quotes, no stray unicode anywhere in the repository."""
    for path in list((REPO_ROOT / "src").rglob("*.py")) + list(
        (REPO_ROOT / "tests").rglob("*.py")
    ) + [README]:
        if not path.exists():
            continue
        raw = path.read_bytes()
        try:
            raw.decode("ascii")
        except UnicodeDecodeError as exc:  # pragma: no cover - failure path
            pytest.fail(f"non-ASCII byte in {path}: {exc}")


# --------------------------------------------------------------------------
# README provenance
# --------------------------------------------------------------------------

# Statistic-shaped tokens: decimals, percentages, and thousands-separated
# integers. Bare small integers are excluded because they are almost always
# structural (section numbers, list counts, "three annotators") rather than
# results, and requiring them to appear verbatim in a JSON file would produce
# noise rather than signal.
_NUMBER = re.compile(r"\d[\d,]*\.\d+%?|\d[\d,]*%|\d{1,3}(?:,\d{3})+")

# Numbers that legitimately appear in the README without being results.
_ALLOWED = {
    "1.0",       # a reference point, not a measurement
    "0.667",     # Krippendorff's published thresholds
    "0.800",
    "0.05",      # significance level, declared in assumptions
    "0.80",      # target power
    "8:1:1",     # the dataset's documented split ratio
    "4.1",       # part of a model name (GPT-4.1-mini), not a measurement
}


def _readme_numbers() -> set[str]:
    text = README.read_text(encoding="utf-8")
    # Ignore fenced code blocks: they are commands, not claims.
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    return {m.group(0).rstrip(".") for m in _NUMBER.finditer(text)} - _ALLOWED


@pytest.mark.skipif(not RESULTS.exists(), reason="results/ not generated yet")
def test_every_readme_number_appears_in_results() -> None:
    """No number in the README that was not computed into results/."""
    blob = "\n".join(
        p.read_text(encoding="utf-8") for p in sorted(RESULTS.glob("*.json"))
    )
    missing = sorted(n for n in _readme_numbers() if n not in blob)
    assert not missing, (
        "README quotes numbers absent from results/: "
        + ", ".join(missing)
        + ". Every figure in the README must be traceable to a results file."
    )


@pytest.mark.skipif(not RESULTS.exists(), reason="results/ not generated yet")
def test_readme_covers_the_headline_findings() -> None:
    summary = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
    text = README.read_text(encoding="utf-8")
    for key in (
        "krippendorff_alpha_nominal",
        "percent_agreement",
        "fleiss_kappa",
        "gwet_ac1",
        "alpha_binary",
        "ceiling_three_way",
    ):
        assert str(summary["study_a"][key]) in text, key
