# Draft "good first issue" tickets (not yet filed; the maintainer files them)

## 1. Loader for the EPIC irony corpus
Add `load_epic` to `src/rubricon_field/corpora.py` for EPIC (Frenda et al.; per-annotator
labels, CC BY-NC-SA 4.0 - check the licence on the dataset card first and record it in the
docstring). Use `build_matrix` so the uniform rules apply, register it in `LOADERS`, add it to
`scripts/fetch_corpora.sh`, and add a small test. Do not commit data.

## 2. Figure: per-emotion alpha for GoEmotions
`results/cross_goemotions.json` has 28 per-emotion alphas with intervals. Add
`fig_goemotions()` to `scripts/make_figures.py` (sorted dot plot with intervals, same palette
and marker rules as the other figures) and check it renders at column width.

## 3. `make check-data`: verify downloaded corpora against the manifest
`scripts/fetch_corpora.sh` writes `data/MANIFEST.sha256`. Add a Makefile target that runs
`shasum -a 256 -c` against it and prints which corpus differs, so a changed upstream file is
caught before any analysis runs.

## 4. Document how to add a corpus
Add a short section to CONTRIBUTING.md walking through the steps in issue 1 (loader,
licence check, fetch script, test, `CORPORA` list in the Makefile).

## 5. Report per-corpus judge refusal counts in the paper appendix
`results/judge/analysis.json` records parse outcomes per judge (`parse`). Extend
`scripts/paper_numbers.py` to emit a small table of refusals/unparseable outputs per judge and
corpus into `paper/tables/`, ledgered like the other tables.
