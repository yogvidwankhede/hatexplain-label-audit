"""Build the fixed item samples for the LLM-judge experiment (PREREG J1).

For each corpus: items with >= 3 ratings in the capped reliability matrix, a
seeded sample of 1,000 (all 350 for DICES-350), and for every item one held-out
rater chosen by a seeded draw. The remaining raters form the panel. The same
panels score the human held-out judge and every model judge, so the comparison
is like for like.

Texts are written only under data/judge/ (gitignored): the corpora's licences
cover research use, but re-releasing platform text is not something this repo
does. The committed file results/judge/sample_<corpus>.json carries item ids,
ratings and the held-out rater, never text.

A separate 5-item pilot per corpus is drawn from items NOT in the sample, for
checking output formatting only (PREREG J1).
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from . import assumptions as A
from .corpora import LOADERS

SAMPLE_SIZE = {"hatexplain": 1000, "mhs": 1000, "wikitalk": 1000, "dices350": 350}
NATIVE_VARIANT = {"hatexplain": "three_way", "mhs": "three_way", "wikitalk": "toxicity_score",
                  "dices350": "three_way"}
PILOT_SIZE = 5
MAX_CHARS = 6000   # PREREG_ADDENDUM.md: longer texts are cut with an explicit marker


def _texts(corpus: str, data_dir: Path, ids: set[str]) -> dict[str, str]:
    import pandas as pd

    if corpus == "hatexplain":
        from .data import load_hatexplain
        ds = load_hatexplain(data_dir / "hatexplain.json")
        return {p.post_id: " ".join(p.tokens) for p in ds.posts.values() if p.post_id in ids}
    if corpus == "mhs":
        m = pd.read_parquet(data_dir / "mhs" / "mhs.parquet", columns=["comment_id", "text"])
        m["comment_id"] = m.comment_id.astype(str)
        m = m[m.comment_id.isin(ids)].drop_duplicates("comment_id")
        return dict(zip(m.comment_id, m.text))
    if corpus == "wikitalk":
        c = pd.read_csv(data_dir / "wikitalk" / "toxicity_annotated_comments.tsv", sep="\t",
                        usecols=["rev_id", "comment"])
        c["rev_id"] = c.rev_id.astype(str)
        c = c[c.rev_id.isin(ids)]
        clean = c.comment.str.replace("NEWLINE_TOKEN", "\n").str.replace("TAB_TOKEN", "\t").str.strip()
        return dict(zip(c.rev_id, clean))
    if corpus == "dices350":
        d = pd.read_csv(data_dir / "dices" / "diverse_safety_adversarial_dialog_350.csv",
                        usecols=["item_id", "context", "response"]).drop_duplicates("item_id")
        d["item_id"] = d.item_id.astype(str)
        d = d[d.item_id.isin(ids)]
        return {i: f"Conversation:\n{c}\n\nChatbot's final response:\n{r}"
                for i, c, r in zip(d.item_id, d.context, d.response)}
    raise KeyError(corpus)


def _clip(t: str) -> tuple[str, bool]:
    if len(t) <= MAX_CHARS:
        return t, False
    return t[:MAX_CHARS] + "\n[... text truncated for length ...]", True


def build(corpus: str, data_dir: Path = Path("data"), out_dir: Path = Path("results/judge")) -> dict:
    variant = {v.name: v for v in LOADERS[corpus](data_dir).variants}[NATIVE_VARIANT[corpus]]
    eligible = sorted(k for k, c in variant.matrix.items() if len(c) >= 3)
    rng = random.Random(f"{A.SEED_PAPER}:judge:{corpus}")
    n = min(SAMPLE_SIZE[corpus], len(eligible))
    sample = sorted(rng.sample(eligible, n))
    rest = sorted(set(eligible) - set(sample))
    pilot = sorted(rng.sample(rest, min(PILOT_SIZE, len(rest))))
    texts = _texts(corpus, data_dir, set(sample) | set(pilot))

    items = []
    for item in sample:
        ratings = {r: v for r, v in sorted(variant.matrix[item].items())}
        held = random.Random(f"{A.SEED_PAPER}:heldout:{corpus}:{item}").choice(sorted(ratings))
        items.append({"item_id": item, "ratings": ratings, "heldout_rater": held})
    missing = [i["item_id"] for i in items if i["item_id"] not in texts]
    if missing:
        raise RuntimeError(f"{corpus}: {len(missing)} sampled items have no text")

    private = data_dir / "judge"
    private.mkdir(parents=True, exist_ok=True)
    n_trunc = 0
    with (private / f"sample_{corpus}.jsonl").open("w") as f:
        for it in items:
            t, cut = _clip(str(texts[it["item_id"]]))
            n_trunc += cut
            f.write(json.dumps({"item_id": it["item_id"], "text": t}) + "\n")
    with (private / f"pilot_{corpus}.jsonl").open("w") as f:
        for item in pilot:
            f.write(json.dumps({"item_id": item, "text": _clip(str(texts[item]))[0]}) + "\n")

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "corpus": corpus, "variant": NATIVE_VARIANT[corpus], "n_eligible": len(eligible),
        "n_sample": len(items), "n_truncated": n_trunc, "max_chars": MAX_CHARS,
        "pilot_item_ids": pilot,
        "text_sha256": hashlib.sha256(
            (private / f"sample_{corpus}.jsonl").read_bytes()).hexdigest(),
        "items": items,
    }
    (out_dir / f"sample_{corpus}.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
    return manifest


if __name__ == "__main__":
    for c in SAMPLE_SIZE:
        m = build(c)
        print(c, m["n_eligible"], "eligible;", m["n_sample"], "sampled;", m["n_truncated"], "truncated")
