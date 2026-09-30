"""Keep model free text out of the public repository.

Judge answers longer than MAX_PUBLIC_CHARS are sentences, and some quote or paraphrase the
(offensive) item being judged. The full outputs are moved to data/judge/raw_outputs/
(gitignored); the committed rows keep the SHA-256 and length of the raw text plus the
label and parse tier computed by judges.parse, so every analysis reproduces from the
committed files. Idempotent.
"""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")
from rubricon_field.judges import parse  # noqa: E402

MAX_PUBLIC_CHARS = 25

for split in ("sample", "pilot"):
    for f in sorted(Path("results/judge", split).glob("*__*.jsonl")):
        corpus = f.stem.split("__")[1]
        private = Path("data/judge/raw_outputs", split)
        private.mkdir(parents=True, exist_ok=True)
        rows, n = [json.loads(l) for l in f.open()], 0
        full = private / f.name
        if not full.exists():
            full.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
        out = []
        for r in rows:
            raw = r.get("raw")
            if raw is not None and len(raw) > MAX_PUBLIC_CHARS:
                label, how = parse(corpus, raw)
                r = dict(r, raw=None, raw_redacted=True, raw_sha256=hashlib.sha256(raw.encode()).hexdigest(),
                         raw_len=len(raw), parsed_label=label, parsed_how=how)
                n += 1
            out.append(r)
        f.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in out))
        if n:
            print(f, "redacted", n)
