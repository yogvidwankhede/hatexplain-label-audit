# Changelog

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow [Semantic Versioning](https://semver.org/).

## [2.0.1] - 2026-09-30
### Changed
- Simulation outputs round derived floats to 12 decimals so they reproduce across CPUs (DEVIATIONS.md D12); no reported number changed.
- Preprint author block; ORCID in CITATION.cff; `.zenodo.json` archive metadata.
### Added
- Property-based tests (Hypothesis); OpenSSF Scorecard workflow and badges.

## [2.0.0] - 2026-09-30
### Added
- Cross-corpus audit of six per-annotator corpora (HateXplain, Measuring Hate Speech, Wikipedia Talk toxicity, DICES-350, DICES-990, GoEmotions): `corpora.py`, `cross_corpus.py`, full-data cluster bootstrap in `fast_alpha.py` (validated exactly against rubricon), capped and all-ratings variants.
- Known-truth simulation of claim gates (`gate_sim.py`, `gate_sim_report.py`) with ablation, alpha-floor sweep and matched significance-test baselines.
- Real-rater test of the one-label-per-item result (`claim4.py`).
- HateXplain retrospective on published gaps using the official test split (`retro.py`).
- LLM-judge experiment: fixed samples (`judge_sample.py`), runner (`judges.py`), scoring against held-out humans with gate v2 (`judge_analysis.py`), Alternative Annotator Test (`alt_test.py`).
- Paper sources, generated number macros, `CLAIMS.md` ledger, `PREREG.md`, `PREREG_ADDENDUM.md`, `DEVIATIONS.md`.
- Reproduction `Makefile`, pinned `requirements.lock`, CI with a byte-for-byte reproduction job, community files.

### Changed (breaking for anyone quoting v1)
- The v1 "accuracy ceiling" (mean modal share) is reported as the upper end of a bracket; see README "Corrections to version 1".
- HateXplain test-split size is taken from the official split file (1,924 posts) instead of an assumed 10%.
- The one-rater-per-item finding is credited to Dorner & Hardt (ICML 2024).
- Requires rubricon >= 0.5.0.

## [1.0.0]
- Single-corpus HateXplain audit (studies A-C).
