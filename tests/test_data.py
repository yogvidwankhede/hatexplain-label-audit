"""Loader tests: round-trip fidelity, the verified counts, and the tie branch.

The verified counts are hard-coded on purpose. They are the published shape of
the corpus, and a loader change that silently alters them -- deduplicating a
repeated annotator, coercing a label, dropping a post with an empty rationale --
would otherwise pass unnoticed and quietly move every downstream number.
"""

from __future__ import annotations

from collections import Counter

import pytest

from rubricon_field.data import (
    LABELS,
    NO_MAJORITY,
    SPLIT_2_1,
    UNANIMOUS,
    Annotation,
    Dataset,
    Post,
    majority_label,
    profiling_records,
    reliability_matrix,
)

# Verified against the published HateXplain release.
N_POSTS = 20148
N_ANNOTATIONS = 60444
N_ANNOTATORS = 253
N_UNANIMOUS = 9845
N_SPLIT_2_1 = 9384
N_NO_MAJORITY = 919
BUSIEST_ANNOTATOR = "4"
BUSIEST_LOAD = 5730


def test_corpus_shape(dataset: Dataset) -> None:
    assert dataset.n_posts == N_POSTS
    assert dataset.n_annotations == N_ANNOTATIONS
    assert dataset.n_annotators == N_ANNOTATORS
    # Every post has exactly three annotations; anything else would break the
    # panel design in study B without raising.
    assert dict(dataset.replication()) == {3: N_POSTS}


def test_only_known_labels(dataset: Dataset) -> None:
    assert set(dataset.label_counts()) == set(LABELS)


def test_label_counts_sum(dataset: Dataset) -> None:
    assert sum(dataset.label_counts().values()) == N_ANNOTATIONS


def test_annotator_load(dataset: Dataset) -> None:
    load = dataset.annotator_load()
    top = load.most_common(1)[0]
    assert top == (BUSIEST_ANNOTATOR, BUSIEST_LOAD)
    assert sum(load.values()) == N_ANNOTATIONS
    top5 = sum(c for _, c in load.most_common(5))
    assert round(top5 / N_ANNOTATIONS, 4) == 0.2229


def test_round_trip_against_raw_json(dataset: Dataset, raw_records: dict) -> None:
    """Every raw record survives parsing with its labels and targets intact."""
    assert len(raw_records) == dataset.n_posts
    for pid, rec in list(raw_records.items())[:500]:
        post = dataset.posts[rec["post_id"]]
        assert list(post.labels) == [a["label"] for a in rec["annotators"]]
        assert [a.annotator_id for a in post.annotations] == [
            str(a["annotator_id"]) for a in rec["annotators"]
        ]
        assert [list(a.targets) for a in post.annotations] == [
            [str(t) for t in a["target"]] for a in rec["annotators"]
        ]
        assert list(post.tokens) == rec["post_tokens"]


def test_annotator_ids_are_strings(dataset: Dataset) -> None:
    assert all(isinstance(a.annotator_id, str) for a in dataset.annotations[:1000])
    assert all(isinstance(k, str) for k in dataset.by_annotator)


def test_consensus_counts(dataset: Dataset) -> None:
    modes = Counter(c.mode for c in dataset.consensus("three_way").values())
    assert modes[UNANIMOUS] == N_UNANIMOUS
    assert modes[SPLIT_2_1] == N_SPLIT_2_1
    assert modes[NO_MAJORITY] == N_NO_MAJORITY
    assert sum(modes.values()) == N_POSTS


def test_binary_collapse_has_no_ties(dataset: Dataset) -> None:
    """Three raters over two categories always produce a majority."""
    modes = Counter(c.mode for c in dataset.consensus("binary").values())
    assert modes[NO_MAJORITY] == 0


def _post(labels: list[str]) -> Post:
    return Post(
        post_id="p",
        tokens=("a",),
        annotations=tuple(
            Annotation("p", str(i), lab, ("None",)) for i, lab in enumerate(labels)
        ),
    )


def test_majority_label_unanimous() -> None:
    c = majority_label(_post(["normal", "normal", "normal"]))
    assert (c.label, c.mode, c.margin, c.resolved) == ("normal", UNANIMOUS, 3, True)


def test_majority_label_split() -> None:
    c = majority_label(_post(["normal", "normal", "hatespeech"]))
    assert (c.label, c.mode, c.margin, c.resolved) == ("normal", SPLIT_2_1, 1, True)


def test_majority_label_three_way_does_not_pick_silently() -> None:
    """The tie branch: no label, flagged, and counted."""
    c = majority_label(_post(["normal", "offensive", "hatespeech"]))
    assert c.label is None
    assert c.mode == NO_MAJORITY
    assert c.resolved is False
    assert c.margin == 0
    assert c.counts == {"normal": 1, "offensive": 1, "hatespeech": 1}


def test_majority_label_tie_is_order_independent() -> None:
    """Permuting the annotations cannot conjure a winner out of a tie.

    This is the specific failure ``majority_label`` exists to prevent:
    ``Counter.most_common`` resolves ties by insertion order, so a naive
    implementation returns a different 'gold' label for the same votes in a
    different order.
    """
    orders = [
        ["normal", "offensive", "hatespeech"],
        ["hatespeech", "normal", "offensive"],
        ["offensive", "hatespeech", "normal"],
    ]
    assert {majority_label(_post(o)).label for o in orders} == {None}


def test_majority_label_binary_collapse() -> None:
    c = majority_label(_post(["offensive", "hatespeech", "normal"]), collapse="binary")
    assert (c.label, c.mode) == ("toxic", SPLIT_2_1)


def test_reliability_matrix_shape(dataset: Dataset) -> None:
    m = reliability_matrix(dataset, collapse="three_way")
    assert len(m) == N_POSTS
    assert all(len(v) == 3 for v in m.values())
    assert set(next(iter(m.values())).values()) <= set(LABELS)


def test_reliability_matrix_binary_vocabulary(dataset: Dataset) -> None:
    m = reliability_matrix(dataset, collapse="binary")
    values = {v for row in m.values() for v in row.values()}
    assert values == {"normal", "toxic"}


def test_reliability_matrix_excludes_annotator(dataset: Dataset) -> None:
    m = reliability_matrix(dataset, exclude_annotators=[BUSIEST_ANNOTATOR])
    assert all(BUSIEST_ANNOTATOR not in row for row in m.values())
    # Every post keeps two raters, so none is dropped for being unpairable.
    assert len(m) == N_POSTS


def test_reliability_matrix_drops_unpairable_units(dataset: Dataset) -> None:
    """The toxic-boundary collapse discards 'normal', so some units fall below two."""
    m = reliability_matrix(dataset, collapse="toxic_boundary")
    assert len(m) < N_POSTS
    assert all(len(row) >= 2 for row in m.values())


def test_endorsed_targets_requires_corroboration(dataset: Dataset) -> None:
    endorsed = dataset.endorsed_targets(min_votes=2)
    assert all("None" not in t for t in endorsed.values())
    loose = dataset.endorsed_targets(min_votes=1)
    assert sum(len(v) for v in loose.values()) > sum(len(v) for v in endorsed.values())


def test_profiling_records_shape(dataset: Dataset) -> None:
    recs = profiling_records(dataset)
    assert len(recs) == N_ANNOTATIONS
    r = recs[0]
    assert set(r) == {"annotator_id", "item_id", "scores"}
    assert set(r["scores"]) == {"toxic", "severity"}
    assert all(0.0 <= v <= 1.0 for v in r["scores"].values())


def test_subset_reindexes(dataset: Dataset) -> None:
    sub = dataset.subset(dataset.post_ids[:10])
    assert sub.n_posts == 10
    assert sub.n_annotations == 30
    assert sum(len(v) for v in sub.by_annotator.values()) == 30


def test_unknown_collapse_raises(dataset: Dataset) -> None:
    with pytest.raises(KeyError):
        reliability_matrix(dataset, collapse="nonsense")  # type: ignore[arg-type]
