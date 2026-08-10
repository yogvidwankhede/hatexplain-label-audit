"""Typed loader for the HateXplain annotation corpus.

Why this module is separate from the analyses
---------------------------------------------
Every reliability statistic in this repository is a function of one object: the
*reliability matrix*, ``unit -> {annotator -> value}``. Almost every mistake in
applied agreement work happens before that matrix is built, not after:

* silently dropping units where the annotators tied, which deletes precisely the
  hardest items and inflates every coefficient computed afterwards;
* keying the matrix on something that is not the annotator (a row index, a
  position within the record), which turns a repeated-measures design into an
  apparently-independent one;
* collapsing categories in an undeclared place, so a reader cannot tell whether
  a reported alpha describes the 3-way task or a binary reduction of it.

This module therefore does the parsing once, into frozen dataclasses, and
exposes the matrix construction as an explicit, parameterised call. The
category collapse is a named argument with a named vocabulary, so a number
computed on the binary reduction cannot be confused with the 3-way number.

Dataset
-------
Mathew, B., Saha, P., Yimam, S. M., Biemann, C., Goyal, P., & Mukherjee, A.
(2021). HateXplain: A Benchmark Dataset for Explainable Hate Speech Detection.
Proceedings of the AAAI Conference on Artificial Intelligence, 35(17),
14867-14875. Data: https://github.com/hate-alert/HateXplain (MIT licence).

Record shape (``dataset.json``), one entry per post::

    {"post_id": str,
     "annotators": [{"annotator_id": int, "label": str, "target": [str, ...]}, x3],
     "rationales": [[0/1 per token], ...],
     "post_tokens": [str, ...]}
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal, Mapping, Sequence

# --------------------------------------------------------------------------
# vocabulary
# --------------------------------------------------------------------------

#: The three labels of the annotation task, in ascending order of severity.
#: Order matters: it is the order used for the ordinal projection and for every
#: table that reports the label marginals, so tables from different analyses
#: line up column-for-column.
LABELS: tuple[str, str, str] = ("normal", "offensive", "hatespeech")

#: Severity projection onto [0, 1]. Used only where a *numeric* per-annotation
#: score is required (annotator profiling). The primary reliability statistics
#: are computed on the nominal labels, because the 3-way task is not
#: convincingly ordinal: "offensive" is not a partial "hatespeech", it is a
#: different speech act that happens to sit between the other two in severity.
SEVERITY: dict[str, float] = {"normal": 0.0, "offensive": 0.5, "hatespeech": 1.0}

#: The binary reduction that most downstream classifiers actually train on:
#: is this post something a moderation system should act on, yes or no.
BINARY_MAP: dict[str, str] = {
    "normal": "normal",
    "offensive": "toxic",
    "hatespeech": "toxic",
}

#: Collapse the two "act on it" categories apart from each other, discarding
#: ``normal``. Used for the conditional analysis of where the disagreement
#: actually lives. Annotations mapped to ``None`` are dropped from the matrix.
TOXIC_BOUNDARY_MAP: dict[str, str | None] = {
    "normal": None,
    "offensive": "offensive",
    "hatespeech": "hatespeech",
}

Collapse = Literal["three_way", "binary", "toxic_boundary"]

_COLLAPSES: dict[str, Mapping[str, str | None]] = {
    "three_way": {lab: lab for lab in LABELS},
    "binary": BINARY_MAP,
    "toxic_boundary": TOXIC_BOUNDARY_MAP,
}

#: Consensus modes returned by :func:`majority_label`.
UNANIMOUS = "unanimous"
SPLIT_2_1 = "split_2_1"
NO_MAJORITY = "no_majority"

#: The ``target`` vocabulary includes a sentinel for "this post targets nobody".
#: It is not a community and must never appear in a per-community table.
NO_TARGET = "None"


# --------------------------------------------------------------------------
# records
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Annotation:
    """One annotator's judgement of one post.

    ``annotator_id`` is carried as a string throughout. The source encodes it as
    an integer, but every agreement routine keys dictionaries on it, and mixing
    ``4`` with ``"4"`` in a key position is a silent, catastrophic bug class:
    it would split one annotator into two and inflate apparent independence.
    Normalising at the boundary makes that impossible downstream.
    """

    post_id: str
    annotator_id: str
    label: str
    targets: tuple[str, ...]

    @property
    def severity(self) -> float:
        return SEVERITY[self.label]

    @property
    def is_toxic(self) -> bool:
        """True when the annotator did not call the post ``normal``."""
        return self.label != "normal"


@dataclass(frozen=True)
class Consensus:
    """The outcome of aggregating one post's annotations into a gold label.

    ``label`` is ``None`` exactly when no label held a strict plurality. That
    case is *not* resolved here: silently picking a winner (first-seen, lowest
    severity, most severe) is the single most common way a corpus acquires
    labels nobody ever agreed to, and it is invisible afterwards because the
    resulting file looks exactly like a file where everyone agreed.
    """

    label: str | None
    mode: str  # UNANIMOUS | SPLIT_2_1 | NO_MAJORITY
    counts: dict[str, int]
    n_annotators: int

    @property
    def resolved(self) -> bool:
        return self.label is not None

    @property
    def margin(self) -> int:
        """Votes for the winner minus votes for the runner-up.

        A margin of 1 means the gold label is one annotator's opinion away from
        not existing. Roughly half of this corpus sits at margin <= 1, which is
        the concrete meaning of the headline alpha.
        """
        ordered = sorted(self.counts.values(), reverse=True)
        if not ordered:
            return 0
        runner_up = ordered[1] if len(ordered) > 1 else 0
        return ordered[0] - runner_up

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "mode": self.mode,
            "counts": dict(self.counts),
            "n_annotators": self.n_annotators,
            "margin": self.margin,
        }


@dataclass(frozen=True)
class Post:
    """One social-media post and the judgements attached to it."""

    post_id: str
    tokens: tuple[str, ...]
    annotations: tuple[Annotation, ...]
    rationales: tuple[tuple[int, ...], ...] = ()

    @property
    def text(self) -> str:
        return " ".join(self.tokens)

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(a.label for a in self.annotations)

    def target_votes(self) -> Counter:
        """How many annotators named each target community for this post.

        Targets are a multi-select field: one annotator may name several
        communities. The vote count is therefore per (post, community), capped
        at the number of annotators.
        """
        votes: Counter = Counter()
        for a in self.annotations:
            for t in set(a.targets):
                votes[t] += 1
        return votes

    def endorsed_targets(self, min_votes: int = 2) -> tuple[str, ...]:
        """Communities named by at least ``min_votes`` annotators, excluding the
        ``None`` sentinel.

        Requiring corroboration matters: a single annotator's target guess is
        measured on the same noisy instrument as the label itself, so
        per-community agreement computed over singly-endorsed targets would be
        partly a measurement of target-field noise.
        """
        return tuple(
            sorted(t for t, v in self.target_votes().items() if v >= min_votes and t != NO_TARGET)
        )


@dataclass
class Dataset:
    """The parsed corpus plus the indices every analysis needs.

    Held as plain dicts rather than a dataframe on purpose: the whole point of
    the parent harness is that its numbers are traceable to bytes on disk by
    someone with no access to the code, and an intermediate representation
    nobody can print is a step away from that.
    """

    posts: dict[str, Post]
    source_path: str = ""

    # -- derived indices (built in __post_init__) --------------------------
    annotations: list[Annotation] = field(default_factory=list, repr=False)
    by_annotator: dict[str, list[Annotation]] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if not self.annotations:
            self.annotations = [a for p in self.posts.values() for a in p.annotations]
        if not self.by_annotator:
            idx: dict[str, list[Annotation]] = defaultdict(list)
            for a in self.annotations:
                idx[a.annotator_id].append(a)
            self.by_annotator = dict(idx)

    # -- basic sizes -------------------------------------------------------

    @property
    def n_posts(self) -> int:
        return len(self.posts)

    @property
    def n_annotations(self) -> int:
        return len(self.annotations)

    @property
    def n_annotators(self) -> int:
        return len(self.by_annotator)

    @property
    def post_ids(self) -> list[str]:
        """Post ids in a deterministic order.

        Sorted, not insertion-ordered: every subsample in this repository is
        drawn from this list with a fixed seed, and insertion order depends on
        the JSON writer that produced the file.
        """
        return sorted(self.posts)

    def label_counts(self) -> Counter:
        return Counter(a.label for a in self.annotations)

    def annotator_load(self) -> Counter:
        """Annotations produced per annotator, descending."""
        return Counter({aid: len(rows) for aid, rows in self.by_annotator.items()})

    def replication(self) -> Counter:
        """Distribution of annotations-per-post. Should be a single spike at 3."""
        return Counter(len(p.annotations) for p in self.posts.values())

    def subset(self, post_ids: Iterable[str]) -> "Dataset":
        keep = {pid: self.posts[pid] for pid in post_ids if pid in self.posts}
        return Dataset(posts=keep, source_path=self.source_path)

    # -- aggregation -------------------------------------------------------

    def consensus(self, collapse: Collapse = "three_way") -> dict[str, Consensus]:
        return {pid: majority_label(p, collapse=collapse) for pid, p in self.posts.items()}

    def endorsed_targets(self, min_votes: int = 2) -> dict[str, tuple[str, ...]]:
        return {pid: p.endorsed_targets(min_votes) for pid, p in self.posts.items()}


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------


def load_hatexplain(path: str | Path) -> Dataset:
    """Parse ``dataset.json`` from the HateXplain release into a :class:`Dataset`.

    No filtering, no repair, no imputation. If the file contains a post with two
    annotations or an unexpected label, that fact survives into the parsed object
    and is reported by the audit rather than quietly normalised away. The only
    transformation applied is the string-coercion of ``annotator_id`` described
    on :class:`Annotation`.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(
            f"expected a JSON object keyed by post_id at {path}; got {type(raw).__name__}"
        )

    posts: dict[str, Post] = {}
    for pid, rec in raw.items():
        post_id = str(rec.get("post_id", pid))
        anns = tuple(
            Annotation(
                post_id=post_id,
                annotator_id=str(a["annotator_id"]),
                label=str(a["label"]),
                targets=tuple(str(t) for t in a.get("target", ())),
            )
            for a in rec.get("annotators", ())
        )
        posts[post_id] = Post(
            post_id=post_id,
            tokens=tuple(str(t) for t in rec.get("post_tokens", ())),
            annotations=anns,
            rationales=tuple(tuple(int(x) for x in r) for r in rec.get("rationales", ())),
        )
    return Dataset(posts=posts, source_path=str(path))


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------


def majority_label(post: Post, collapse: Collapse = "three_way") -> Consensus:
    """Aggregate one post's annotations into a gold label, honestly.

    Returns the label *and* how it was reached. Three outcomes:

    ``unanimous``
        Every annotator chose the same category. The gold label is as strong as
        this design can make it.
    ``split_2_1``
        A strict majority. The gold label exists, but it rests on a single
        annotator's vote: change one of the two majority votes and the label
        stops existing. Roughly 47% of this corpus is here.
    ``no_majority``
        Every annotator chose something different. **No label is returned.**
        Downstream code must decide what to do and must do it visibly.

    The third branch is the reason this function exists rather than a
    ``Counter(...).most_common(1)[0][0]`` at each call site: ``most_common``
    resolves a three-way tie by insertion order, which is a coin flip dressed as
    a decision, and it leaves no trace in the output.
    """
    mapper = _COLLAPSES[collapse]
    mapped = [mapper[a.label] for a in post.annotations if mapper.get(a.label) is not None]
    counts = Counter(mapped)
    n = len(mapped)
    if n == 0:
        return Consensus(None, NO_MAJORITY, {}, 0)

    ranked = counts.most_common()
    top_count = ranked[0][1]
    winners = [lab for lab, c in ranked if c == top_count]

    if len(winners) > 1:
        # A tie. No strict plurality exists, so no label is produced.
        return Consensus(None, NO_MAJORITY, dict(counts), n)
    if top_count == n:
        return Consensus(winners[0], UNANIMOUS, dict(counts), n)
    return Consensus(winners[0], SPLIT_2_1, dict(counts), n)


# --------------------------------------------------------------------------
# reliability matrix
# --------------------------------------------------------------------------


def reliability_matrix(
    dataset: Dataset,
    collapse: Collapse = "three_way",
    post_ids: Sequence[str] | None = None,
    exclude_annotators: Iterable[str] = (),
    min_annotations_per_unit: int = 2,
) -> dict[str, dict[str, str]]:
    """Build ``post_id -> {annotator_id -> label}``, the input every agreement
    coefficient in ``rubricon.stats.agreement`` consumes.

    Parameters
    ----------
    collapse
        ``three_way`` (the task as annotated), ``binary`` (``normal`` vs
        ``toxic``, the reduction most deployed classifiers are trained on), or
        ``toxic_boundary`` (``offensive`` vs ``hatespeech`` only, which drops
        every ``normal`` judgement and is therefore a *conditional* measurement
        -- see the caveat recorded alongside it in the results).
    exclude_annotators
        Used by the leave-one-annotator-out analysis. Units that fall below
        ``min_annotations_per_unit`` after exclusion are dropped, because a unit
        with one rating carries no pairing information; the count of dropped
        units is what makes the exposure visible, so callers should compare
        ``len(result)`` against the unfiltered size rather than assume it held.

    Units below ``min_annotations_per_unit`` are dropped rather than passed
    through. ``krippendorff_alpha`` would ignore them anyway, but it counts them
    in ``n_units``, and a coverage number that silently includes unusable rows
    is worse than one that does not.
    """
    mapper = _COLLAPSES[collapse]
    drop = set(exclude_annotators)
    ids = list(post_ids) if post_ids is not None else dataset.post_ids

    matrix: dict[str, dict[str, str]] = {}
    for pid in ids:
        post = dataset.posts.get(pid)
        if post is None:
            continue
        row: dict[str, str] = {}
        for a in post.annotations:
            if a.annotator_id in drop:
                continue
            value = mapper.get(a.label)
            if value is None:
                continue
            row[a.annotator_id] = value
        if len(row) >= min_annotations_per_unit:
            matrix[pid] = row
    return matrix


def profiling_records(dataset: Dataset) -> list[dict]:
    """Annotations in the shape ``rubricon.stats.drift.profile_annotators`` wants.

    Two numeric dimensions are exposed, and the split between them is the whole
    point of using that function here:

    ``toxic``
        0/1: did this annotator decline to call the post ``normal``. This is a
        calibration property. Two annotators who disagree about *whether a post
        needs moderating at all* are not both right, and a systematic offset here
        is the kind of thing annotator re-anchoring actually fixes.
    ``severity``
        0 / 0.5 / 1: the full 3-way judgement on a common scale. Declared as a
        *contested* dimension, so ``profile_annotators`` reports an annotator's
        offset on it as a ``contested_position`` rather than a quality flag.

    That mapping is faithful to the construct. Where an individual draws the line
    between "offensive" and "hate speech" is a definitional stance that reasonable
    trained annotators hold differently and that the codebook does not fully
    determine; flagging it as bias would recommend retraining people for
    disagreeing about a genuinely contested boundary. Both dimensions are placed
    on [0, 1] so a single ``bias_threshold`` means the same thing on each.
    """
    return [
        {
            "annotator_id": a.annotator_id,
            "item_id": a.post_id,
            "scores": {"toxic": 1.0 if a.is_toxic else 0.0, "severity": SEVERITY[a.label]},
        }
        for a in dataset.annotations
    ]


__all__ = [
    "Annotation",
    "BINARY_MAP",
    "Consensus",
    "Dataset",
    "LABELS",
    "NO_MAJORITY",
    "NO_TARGET",
    "Post",
    "SEVERITY",
    "SPLIT_2_1",
    "TOXIC_BOUNDARY_MAP",
    "UNANIMOUS",
    "load_hatexplain",
    "majority_label",
    "profiling_records",
    "reliability_matrix",
]
