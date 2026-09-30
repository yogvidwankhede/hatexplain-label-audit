# What is and is not new (updated 2026-09-30 after full reads; see related_work_notes.md)

| Component | Status | Prior work we must credit |
|---|---|---|
| One label per item maximises power for comparing two binary classifiers at a fixed budget | **Known theorem** | Dorner & Hardt, ICML 2024 (verified on proceedings.mlr.press/v235/dorner24a) |
| Observed gap = (1 - 2 eta) x true gap; correction for noisy test labels | **Known** | Lam & Stork, IJCAI 2003, Eq. 2; also Dorner & Hardt 2024 |
| Held-out-annotator comparison for LLM judges | **Known procedure** | Calderon et al. ACL 2025 (alt-test); Resnick et al. 2021 |
| Reliability thresholds for evaluation claims | **Proposed as guidance** | Caban 2026 (arXiv 2608.00794); Krippendorff 2004 |
| Known-truth calibration of a reliability-threshold claim gate vs a plain significance test at matched false-claim rates | Not found in prior work | — |
| Estimating eta from the benchmark's own replicated labels and propagating it into published leaderboard gaps (HateXplain) | Not found as an end-to-end procedure | builds on Lam & Stork |
| Real-annotator test of the one-label theorem (3 corpora, contested-item system model) | Empirical replication/extension | Dorner & Hardt 2024 (theory, independent noise); Pandita et al. 2026 (Toxicity, K=1 best) |
| Six-corpus reliability audit with alpha/AC1/raw side by side, capped vs all-ratings sensitivity | Descriptive contribution | Klie et al. 2024 surveys reporting practice |
| Judge-vs-human-ceiling on 4 corpora with 5 judges, gate v2 applied | Empirical | Bavaresco et al. 2025; Calderon et al. 2025 |

Consequence for the paper: claim 7 of the old README ("one rater per item maximises power") is re-stated as a replication of Dorner & Hardt 2024 on real raters, never as our result.
