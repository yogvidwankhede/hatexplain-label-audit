"""Loaders that turn each per-annotator corpus into reliability matrices.

Every agreement statistic in this package consumes ``unit -> {rater -> value}``.
This module builds that object for each corpus named in PREREG.md section 1, with
the same rules everywhere:

* items with fewer than 2 ratings are dropped (they carry no pairing information)
  and the number dropped is recorded;
* items with more than ``MAX_RATINGS_PER_ITEM`` ratings keep that many, chosen by
  a seeded draw that depends only on (seed, item id), never on Python's salted
  ``hash()``;
* a repeated (item, rater) pair keeps its first row and is counted;
* nothing here reads, stores or prints free text or user names.

The label definitions below are fixed by PREREG.md (tag ``prereg-v1``). Changing
one is a deviation and belongs in DEVIATIONS.md.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Mapping

from . import assumptions as A

Matrix = dict[str, dict[str, object]]

#: PREREG.md section 1: cap on ratings kept per item.
MAX_RATINGS_PER_ITEM = 5


@dataclass
class Variant:
    """One reliability matrix plus what was done to build it."""

    name: str
    matrix: Matrix
    metric: str = "nominal"          # krippendorff metric
    binary_of: str | None = None     # name of the parent variant if this is a collapse
    notes: dict = field(default_factory=dict)


@dataclass
class Corpus:
    name: str
    variants: list[Variant]
    provenance: dict


def _item_rng(item_id: str, salt: str) -> random.Random:
    digest = hashlib.sha256(f"{A.SEED_PAPER}:{salt}:{item_id}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def build_matrix(
    rows: Iterable[tuple[str, str, object]],  # (item, rater, value)
    salt: str,
    cap: int | None = MAX_RATINGS_PER_ITEM,
) -> tuple[Matrix, dict]:
    """Group rows into a matrix under the uniform rules above."""
    per_item: dict[str, dict[str, object]] = {}
    duplicates = 0
    for item, rater, value in rows:
        cell = per_item.setdefault(item, {})
        if rater in cell:
            duplicates += 1
            continue
        cell[rater] = value

    n_items_raw = len(per_item)
    n_ratings_raw = sum(len(v) for v in per_item.values())
    dropped_lt2 = 0
    capped_items = 0
    out: Matrix = {}
    for item, cell in per_item.items():
        if len(cell) < 2:
            dropped_lt2 += 1
            continue
        if cap is not None and len(cell) > cap:
            keys = sorted(cell)
            keep = _item_rng(item, salt).sample(keys, cap)
            cell = {k: cell[k] for k in keep}
            capped_items += 1
        out[item] = cell
    info = {
        "n_items_raw": n_items_raw,
        "n_ratings_raw": n_ratings_raw,
        "items_dropped_lt2_ratings": dropped_lt2,
        "items_capped_to_max": capped_items,
        "max_ratings_per_item": cap,
        "duplicate_item_rater_rows": duplicates,
        "n_items_used": len(out),
        "n_ratings_used": sum(len(v) for v in out.values()),
    }
    return out, info


def collapse(matrix: Matrix, fn: Callable[[object], object | None]) -> Matrix:
    """Relabel every rating; ratings mapped to None are dropped; <2 left drops the item."""
    out: Matrix = {}
    for item, cell in matrix.items():
        new = {r: fn(v) for r, v in cell.items()}
        new = {r: v for r, v in new.items() if v is not None}
        if len(new) >= 2:
            out[item] = new
    return out


# --------------------------------------------------------------------------
# corpus-specific loaders
# --------------------------------------------------------------------------


def load_hatexplain_corpus(data_dir: Path) -> Corpus:
    from .data import load_hatexplain, reliability_matrix

    ds = load_hatexplain(data_dir / "hatexplain.json")
    three = reliability_matrix(ds, collapse="three_way")
    binary = reliability_matrix(ds, collapse="binary")
    return Corpus(
        "hatexplain",
        [
            Variant("three_way", three),
            Variant("binary", binary, binary_of="three_way"),
        ],
        {"source": "hate-alert/HateXplain dataset.json", "cap": None,
         "note": "existing loader; exactly 3 ratings per post so no cap applies"},
    )


def load_mhs(data_dir: Path) -> Corpus:
    import pandas as pd

    df = pd.read_parquet(data_dir / "mhs" / "mhs.parquet", columns=["comment_id", "annotator_id", "hatespeech"])
    df = df.dropna(subset=["hatespeech"])
    rows = zip(df.comment_id.astype(str), df.annotator_id.astype(str), df.hatespeech.astype(int))
    m3, info = build_matrix(rows, "mhs")
    m2 = collapse(m3, lambda v: "hate" if v == 2 else "other")
    return Corpus(
        "mhs",
        [Variant("three_way", m3, notes=info), Variant("binary", m2, binary_of="three_way")],
        {"source": "ucberkeley-dlab/measuring-hate-speech",
         "coding": "0 neutral/counter, 1 unclear, 2 hate speech (from a secondary source; primary codebook not seen)",
         **info},
    )


def _load_dices(data_dir: Path, tag: str) -> Corpus:
    import pandas as pd

    df = pd.read_csv(data_dir / "dices" / f"diverse_safety_adversarial_dialog_{tag}.csv",
                     usecols=["item_id", "rater_id", "Q_overall"])
    df = df.dropna(subset=["Q_overall"])
    rows = zip(df.item_id.astype(str), df.rater_id.astype(str), df.Q_overall.astype(str))
    m3, info = build_matrix(rows, f"dices{tag}")
    m2 = collapse(m3, lambda v: "yes" if v == "Yes" else "other")
    return Corpus(
        f"dices{tag}",
        [Variant("three_way", m3, notes=info), Variant("binary", m2, binary_of="three_way")],
        {"source": f"google-research-datasets/dices-dataset {tag}", "label": "Q_overall", **info},
    )


def load_dices350(data_dir: Path) -> Corpus:
    return _load_dices(data_dir, "350")


def load_dices990(data_dir: Path) -> Corpus:
    return _load_dices(data_dir, "990")


def load_wikitalk(data_dir: Path) -> Corpus:
    import pandas as pd

    df = pd.read_csv(data_dir / "wikitalk" / "toxicity_annotations.tsv", sep="\t",
                     usecols=["rev_id", "worker_id", "toxicity", "toxicity_score"])
    ids = df.rev_id.astype(str)
    workers = df.worker_id.astype(str)
    mb, info = build_matrix(zip(ids, workers, df.toxicity.astype(int)), "wiki")
    ms, info_s = build_matrix(zip(ids, workers, df.toxicity_score.astype(int)), "wiki")
    return Corpus(
        "wikitalk",
        [Variant("toxicity_binary", mb, notes=info),
         Variant("toxicity_score", ms, metric="ordinal", notes=info_s)],
        {"source": "figshare 4563973 (Wulczyn et al. 2017)", **info},
    )


def load_goemotions(data_dir: Path) -> Corpus:
    import pandas as pd

    frames = [pd.read_csv(data_dir / "goemotions" / f"goemotions_{i}.csv") for i in (1, 2, 3)]
    df = pd.concat(frames, ignore_index=True)
    meta = {"id", "text", "author", "subreddit", "link_id", "parent_id", "created_utc",
            "rater_id", "example_very_unclear"}
    emotions = [c for c in df.columns if c not in meta]
    ids = df["id"].astype(str)
    raters = df["rater_id"].astype(str)
    variants = []
    infos = {}
    for emo in emotions:
        m, info = build_matrix(zip(ids, raters, df[emo].astype(int)), "goemo")
        variants.append(Variant(f"emotion:{emo}", m, notes=info))
        infos[emo] = info["n_items_used"]
    return Corpus(
        "goemotions", variants,
        {"source": "google-research/goemotions raw CSVs",
         "note": "alpha per emotion on a binary present/absent label; usernames never read into results",
         "n_emotions": len(emotions)},
    )


LOADERS: Mapping[str, Callable[[Path], Corpus]] = {
    "hatexplain": load_hatexplain_corpus,
    "mhs": load_mhs,
    "dices350": load_dices350,
    "dices990": load_dices990,
    "wikitalk": load_wikitalk,
    "goemotions": load_goemotions,
}
