# Contributing to hatexplain-label-audit

Thanks for your interest. This repository is the reproducibility package for a label-reliability audit across several annotation corpora.

## Ground rules
- **Numbers must be traceable.** Any change that alters a statistic must regenerate the
  affected files in `results/` in the same pull request, and the tests must pass.
- **Statistics are validated, not trusted.** New estimators need a test against an
  independent reference (a published value, `statsmodels`/`scipy`, or exact enumeration).
- **No silent assumption changes.** Constants belong in one clearly documented place with the reason for their value.

## Workflow
1. Open an issue describing the change (for anything beyond a typo).
2. Branch from `main`, keep the change focused, and add tests.
3. Run `make test` locally.
4. Open a pull request using the template; CI must pass before merge.

## Adding a per-annotator corpus

Keep the addition focused and discuss it in an issue first:

1. Check the primary source's licence and terms before downloading. Record the
   source, licence and any ambiguity in the `source` entry of the `provenance`
   dict that the loader returns with the `Corpus` (see `load_mhs`) and in the data
   section of `README.md`; do not assume a dataset card and the upstream
   repository agree. Do not commit corpus
   text or user names.
2. Add a loader in `src/rubricon_field/corpora.py` and register it in `LOADERS`.
   Follow an existing loader such as `load_mhs`: return a `Corpus` containing
   named `Variant` objects, label definitions and source provenance. Feed
   `(item, rater, value)` rows through `build_matrix` so duplicate handling,
   dropping single-rated items and seeded capping stay consistent. Support both
   the default cap and `cap=None`. A new corpus is outside the pre-registered
   analysis: note it in `CHANGELOG.md`. Use `DEVIATIONS.md` only for changes to
   the label definitions of the six existing corpora.
3. Add the download steps to `scripts/fetch_corpora.sh`, using the loader's
   expected paths under gitignored `data/`. Record the downloaded files' SHA-256
   checksums in `scripts/data.sha256`; the script verifies them before analysis.
   Keep tests independent of network downloads and paid judge API calls.
4. Add synthetic, text-free fixture tests under `tests/`. Check the actual
   loader's labels, variants and provenance, as well as capped and uncapped
   behaviour. Use `tests/test_corpora.py` for the shared matrix rules; run
   `make test` before opening the PR.
5. Add the corpus's `LOADERS` key to `CORPORA` in `Makefile`. The `cross` target
   runs both capped and `--all-ratings` analyses for each key. If reported
   statistics change, regenerate the affected `results/` files and keep their
   claim provenance traceable. `make judges` is not required: it makes paid
   model calls.

## AI-assisted contributions
AI assistance is allowed. Say in the PR description what was assisted, and make sure you
have read, run and understood every line you submit. You are responsible for its correctness.

## Adding a corpus

Every agreement statistic consumes the same `unit -> {rater -> value}` object,
built by one loader under the uniform rules in [`src/rubricon_field/corpora.py`](src/rubricon_field/corpora.py)
(items with fewer than two ratings dropped and counted; per-item ratings capped by a
seeded draw that depends only on `(seed, item id)`; repeated `(item, rater)` rows
deduped and counted; no free text or user names ever read). To add a new corpus,
walk through these steps in order.

1. **Loader.** Add `load_<name>(data_dir, cap=...) -> Corpus` in
   `src/rubricon_field/corpora.py`. It must produce one or more `Variant` objects
   (each with a `name`, a `matrix`, and the `metric` — `nominal` default, `ordinal`
   where a scale is ordered) and a `provenance` dict naming the source and, where a
   value is relabelled, how. Register the loader in the module-level `LOADERS`
   mapping, keyed by the corpus name.
2. **Licence check.** Read the corpus's licence from the primary source and record it
   as a comment at the top of [`scripts/fetch_corpora.sh`](scripts/fetch_corpora.sh),
   alongside the date you read it. If a licence is ambiguous (for example, a raw-data
   repo that states no data licence), record it as `AMBIGUOUS` and note where, rather
   than guessing.
3. **Fetch script + checksums.** Add a `get` call in `scripts/fetch_corpora.sh` that
   writes the file to `data/<name>/...`, then add its `shasum -a 256` line to
   [`scripts/data.sha256`](scripts/data.sha256) so an upstream change is caught before
   any analysis runs. Data is not vendored (see `.gitignore`).
4. **Test.** Add a test in [`tests/test_corpora.py`](tests/test_corpora.py) that builds
   a matrix with `build_matrix` / `collapse` and asserts one behaviour of the new loader
   or the uniform rules it relies on. Statistics changes must also regenerate the
   affected `results/` files in the same pull request.
5. **Register in the build.** Add the corpus name to the `CORPORA` variable in the
   [`Makefile`](Makefile) so `make cross` sweeps it in both the capped and
   all-ratings passes.

Then run `./scripts/fetch_corpora.sh`, `make test`, and
`python -m rubricon_field.cross_corpus <name>` (plus `--all-ratings`) and confirm the
new `results/cross_<name>.json` is reproducible and consistent with
[`DEVIATIONS.md`](DEVIATIONS.md).

## Code of conduct
Participation is governed by [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
