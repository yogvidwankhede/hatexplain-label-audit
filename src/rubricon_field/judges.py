"""Run the LLM judges over the fixed samples (PREREG J1, PREREG_ADDENDUM.md).

Every judge sees the same prompt file (hashed in the addendum) as its system
prompt and the item text as the user message. Raw outputs are kept; parsing is
a separate, pre-registered rule (``parse``) so a parser change can be re-run
without new model calls.

No server-side model fallback is enabled for Claude: a refusal is recorded as a
refusal of *this* judge. Silently answering with another model would change the
judge's identity mid-experiment.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CORPORA = ("hatexplain", "mhs", "wikitalk", "dices350")
PROMPT_FILE = {"hatexplain": "hatexplain.txt", "mhs": "mhs.txt", "wikitalk": "wikitalk.txt",
               "dices350": "dices.txt"}
LABELS = {
    "hatexplain": {"hatespeech": "hatespeech", "offensive": "offensive", "normal": "normal"},
    "mhs": {"yes": 2, "unclear": 1, "no": 0},
    "wikitalk": {"very_toxic": -2, "toxic": -1, "neither": 0, "healthy": 1, "very_healthy": 2},
    "dices350": {"yes": "Yes", "no": "No", "unsure": "Unsure"},
}

JUDGES = {
    "claude-haiku-4-5": {"provider": "anthropic", "model": "claude-haiku-4-5-20251001",
                         "params": {"max_tokens": 16, "temperature": 0}},
    "claude-sonnet-5-5": {"provider": "anthropic", "model": "claude-sonnet-5-5",
                          "params": {"max_tokens": 64, "thinking": {"type": "between_tools"},
                                     "output_config": {"effort": "low"}}},
    "gpt-4.1-mini": {"provider": "openai", "model": "gpt-4.1-mini",
                     "params": {"max_tokens": 16, "temperature": 0, "seed": 0}},
    "qwen2.5-14b": {"provider": "ollama", "model": "qwen2.5:14b",
                    "params": {"options": {"temperature": 0, "seed": 0, "num_predict": 16}}},
    "gpt-oss-20b": {"provider": "ollama", "model": "gpt-oss:20b",
                    "params": {"think": "low", "options": {"temperature": 0, "seed": 0,
                                                           "num_predict": 4096}}},
}

# USD per 1M tokens (input, output); Anthropic at Batch rates. Sources in PREREG_ADDENDUM.md.
PRICES = {"claude-haiku-4-5": (0.5, 2.5), "claude-sonnet-5-5": (1.0, 5.0), "gpt-4.1-mini": (0.40, 1.60)}
CAP_USD = {"anthropic": 8.0, "openai": 4.0}


# --------------------------------------------------------------------------
# parsing (pre-registered)
# --------------------------------------------------------------------------


def parse(corpus: str, raw: str | None) -> tuple[object | None, str]:
    """Strict: the whole answer, minus quotes/punctuation/case, is one label.
    First word: the answer starts with a label (then explains).
    Lenient: exactly one distinct label word occurs in the answer. Else None."""
    if raw is None:
        return None, "no_output"
    labels = LABELS[corpus]
    norm = re.sub(r"[^a-z_ ]", "", re.sub(r"\s+", " ", raw.strip().lower()).replace("-", "_")).strip()
    norm = norm.replace("very toxic", "very_toxic").replace("very healthy", "very_healthy")
    if norm in labels:
        return labels[norm], "strict"
    first = norm.split()[0] if norm.split() else ""
    if first in labels:          # added after the pilot (DEVIATIONS.md D10)
        return labels[first], "first_word"
    words = set(norm.split())
    hits = [k for k in labels if k in words]
    if corpus == "wikitalk" and "very_toxic" in hits:
        hits = [h for h in hits if h != "toxic"]
    if corpus == "wikitalk" and "very_healthy" in hits:
        hits = [h for h in hits if h != "healthy"]
    if len(hits) == 1:
        return labels[hits[0]], "lenient"
    return None, "unparseable"


# --------------------------------------------------------------------------
# io
# --------------------------------------------------------------------------


def _items(corpus: str, split: str) -> list[dict]:
    return [json.loads(l) for l in open(f"data/judge/{split}_{corpus}.jsonl")]


def _prompt(corpus: str) -> str:
    return Path("prompts", PROMPT_FILE[corpus]).read_text()


def _out_path(judge: str, corpus: str, split: str) -> Path:
    p = Path("results/judge") / split / f"{judge}__{corpus}.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {json.loads(l)["item_id"] for l in path.open()}


def _append(path: Path, rows: list[dict]) -> None:
    with path.open("a") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")


def _load_env() -> None:
    for line in Path(".env").read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def estimate_cost(judge: str, corpora=CORPORA, split: str = "sample") -> float:
    pin, pout = PRICES.get(judge, (0, 0))
    chars = sum(len(_prompt(c)) * len(_items(c, split)) + sum(len(i["text"]) for i in _items(c, split))
                for c in corpora)
    n = sum(len(_items(c, split)) for c in corpora)
    out_tokens = n * JUDGES[judge]["params"].get("max_tokens", 16)   # upper bound
    return chars / 3.0 / 1e6 * pin + out_tokens / 1e6 * pout


# --------------------------------------------------------------------------
# providers
# --------------------------------------------------------------------------


def _anthropic_sync(judge: str, corpus: str, item: dict) -> dict:
    import anthropic
    cfg = JUDGES[judge]
    client = anthropic.Anthropic()
    params = dict(cfg["params"])
    extra = {k: params.pop(k) for k in ("temperature",) if k in params}
    # anthropic>=1 removed the sampling keywords from messages.create(); models that
    # still honour them (Haiku 4.5) take them through extra_body. In batch params the
    # key is forwarded as-is.
    msg = client.messages.create(model=cfg["model"], system=_prompt(corpus),
                                 messages=[{"role": "user", "content": item["text"]}],
                                 extra_body=extra or None, **params)
    return _anthropic_row(item["item_id"], msg)


def _anthropic_row(item_id: str, msg) -> dict:
    text = "".join(b.text for b in msg.content if b.type == "text") if msg.stop_reason != "refusal" else None
    return {"item_id": item_id, "raw": text, "stop_reason": msg.stop_reason, "model": msg.model,
            "refusal_category": getattr(getattr(msg, "stop_details", None), "category", None),
            "in_tokens": msg.usage.input_tokens, "out_tokens": msg.usage.output_tokens}


def run_anthropic_batch(judge: str, corpora=CORPORA) -> None:
    import anthropic
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    cfg = JUDGES[judge]
    client = anthropic.Anthropic()
    state = Path("results/judge/sample") / f"{judge}__batch.json"
    state.parent.mkdir(parents=True, exist_ok=True)
    if not state.exists():
        reqs, index = [], {}
        for c in corpora:
            for i, it in enumerate(_items(c, "sample")):
                cid = f"{c}-{i}"
                index[cid] = [c, it["item_id"]]
                reqs.append(Request(custom_id=cid, params=MessageCreateParamsNonStreaming(
                    model=cfg["model"], system=_prompt(c),
                    messages=[{"role": "user", "content": it["text"]}], **cfg["params"])))
        batch = client.messages.batches.create(requests=reqs)
        state.write_text(json.dumps({"batch_id": batch.id, "index": index}))
        print(judge, "batch", batch.id, len(reqs), "requests", flush=True)
    st = json.loads(state.read_text())
    while True:
        b = client.messages.batches.retrieve(st["batch_id"])
        if b.processing_status == "ended":
            break
        time.sleep(60)
    rows = {c: [] for c in corpora}
    for r in client.messages.batches.results(st["batch_id"]):
        c, item_id = st["index"][r.custom_id]
        if r.result.type == "succeeded":
            rows[c].append(_anthropic_row(item_id, r.result.message))
        else:
            rows[c].append({"item_id": item_id, "raw": None, "stop_reason": f"batch_{r.result.type}",
                            "model": cfg["model"], "in_tokens": 0, "out_tokens": 0})
    for c in corpora:
        path = _out_path(judge, c, "sample")
        path.unlink(missing_ok=True)
        _append(path, sorted(rows[c], key=lambda x: x["item_id"]))
    print(judge, "done", {c: len(v) for c, v in rows.items()}, flush=True)


def _openai_one(judge: str, corpus: str, item: dict) -> dict:
    cfg = JUDGES[judge]
    body = json.dumps({"model": cfg["model"], "messages": [
        {"role": "system", "content": _prompt(corpus)}, {"role": "user", "content": item["text"]}],
        **cfg["params"]}).encode()
    req = urllib.request.Request("https://api.openai.com/v1/chat/completions", data=body, headers={
        "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}", "Content-Type": "application/json"})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                d = json.load(resp)
            ch = d["choices"][0]
            return {"item_id": item["item_id"], "raw": ch["message"].get("content"),
                    "stop_reason": ch.get("finish_reason"), "model": d.get("model"),
                    "refusal": ch["message"].get("refusal"),
                    "system_fingerprint": d.get("system_fingerprint"),
                    "in_tokens": d["usage"]["prompt_tokens"], "out_tokens": d["usage"]["completion_tokens"]}
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < 5:
                time.sleep(2 ** attempt * 5)
                continue
            return {"item_id": item["item_id"], "raw": None, "stop_reason": f"http_{e.code}",
                    "model": cfg["model"], "error": e.read().decode()[:300], "in_tokens": 0, "out_tokens": 0}
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt < 5:
                time.sleep(2 ** attempt * 5)
                continue
            raise


def _ollama_one(judge: str, corpus: str, item: dict) -> dict:
    cfg = JUDGES[judge]
    body = json.dumps({"model": cfg["model"], "stream": False, "messages": [
        {"role": "system", "content": _prompt(corpus)}, {"role": "user", "content": item["text"]}],
        **cfg["params"]}).encode()
    req = urllib.request.Request("http://localhost:11434/api/chat", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as resp:
        d = json.load(resp)
    return {"item_id": item["item_id"], "raw": d["message"].get("content"),
            "stop_reason": d.get("done_reason"), "model": d.get("model"),
            "in_tokens": d.get("prompt_eval_count"), "out_tokens": d.get("eval_count")}


def run_sync(judge: str, split: str, corpora=CORPORA, workers: int = 1) -> None:
    provider = JUDGES[judge]["provider"]
    fn = {"openai": _openai_one, "ollama": _ollama_one, "anthropic": _anthropic_sync}[provider]
    for c in corpora:
        path = _out_path(judge, c, split)
        todo = [it for it in _items(c, split) if it["item_id"] not in _done(path)]
        with ThreadPoolExecutor(workers) as ex:
            for k in range(0, len(todo), 50):
                chunk = todo[k:k + 50]
                _append(path, list(ex.map(lambda it: fn(judge, c, it), chunk)))
        print(judge, c, split, "done", flush=True)


if __name__ == "__main__":
    _load_env()
    judge, split = sys.argv[1], sys.argv[2]
    provider = JUDGES[judge]["provider"]
    if provider in CAP_USD and split == "sample":
        est = estimate_cost(judge)
        print(f"{judge}: projected upper-bound cost ${est:.2f} (cap ${CAP_USD[provider]:.2f})")
        if est > CAP_USD[provider]:
            sys.exit("projected cost exceeds cap; aborting")
    if provider == "anthropic" and split == "sample":
        run_anthropic_batch(judge)
    else:
        run_sync(judge, split, workers={"openai": 8, "anthropic": 4, "ollama": 1}[provider])
