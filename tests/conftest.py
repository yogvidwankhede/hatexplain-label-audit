"""Shared fixtures.

The full corpus is loaded once per session and shared read-only. Loading 12MB of
JSON per test would dominate the suite's runtime and would encourage writing
fewer tests, which is the wrong trade.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rubricon_field.data import Dataset, load_hatexplain

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "data" / "hatexplain.json"


@pytest.fixture(scope="session")
def data_path() -> Path:
    if not DATA_PATH.exists():  # pragma: no cover - environment guard
        pytest.skip(f"corpus not present at {DATA_PATH}")
    return DATA_PATH


@pytest.fixture(scope="session")
def dataset(data_path: Path) -> Dataset:
    return load_hatexplain(data_path)


@pytest.fixture(scope="session")
def slice_dataset(dataset: Dataset) -> Dataset:
    """A small, deterministic slice for end-to-end study runs.

    First 600 post ids in sorted order -- deterministic and independent of the
    JSON writer's insertion order. Large enough that the bootstrap and the
    per-community strata have something to work with, small enough that the
    whole suite stays fast.
    """
    return dataset.subset(dataset.post_ids[:600])


@pytest.fixture(scope="session")
def raw_records(data_path: Path) -> dict:
    return json.loads(data_path.read_text(encoding="utf-8"))
