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
   source, licence and any ambiguity in the provenance documentation; do not
   assume a dataset card and the upstream repository agree. Do not commit corpus
   text or user names.
2. Add a loader in `src/rubricon_field/corpora.py` and register it in `LOADERS`.
   Follow an existing loader such as `load_mhs`: return a `Corpus` containing
   named `Variant` objects, label definitions and source provenance. Feed
   `(item, rater, value)` rows through `build_matrix` so duplicate handling,
   dropping single-rated items and seeded capping stay consistent. Support both
   the default cap and `cap=None`. A new label definition changes the preregistered
   assumptions: explain it in `DEVIATIONS.md` rather than changing them silently.
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

## Code of conduct
Participation is governed by [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
