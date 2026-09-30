# PREREG.md — analysis plan (DRAFT, not yet frozen)

Status: **DRAFT v0.1, 2026-09-29.** It becomes binding when the author approves it and it is tagged `prereg-v1`.
After that tag, every change goes in `DEVIATIONS.md` with the date and whether results had been seen.

Working title: *What can your labels support? A claim-gated reliability audit of annotation benchmarks.*

## 0. What was known before this file (honest disclosure)

- **HateXplain** results (alpha 0.4597, binary 0.5613, ceiling 0.8143, 51.1% one-vote margin, MDE 0.0279 at n=2015, judge-stand-in 76.1%) were published in `results/` **before** this plan. HateXplain analyses are therefore *replications*, not pre-registered tests.
- For the other corpora only file schemas, value codings, row counts, rater counts and raters-per-item summaries were inspected (2026-09-29). **No agreement statistic, ceiling, margin or judge result has been computed on them.**
- The existing MDE (assumed system-disagreement rate + `1/sqrt(rho)` rescaling) is replaced by model M1 below **before** any new-corpus run. The old figure stays in the repo as history.
- Claim 4 ("one rater per item maximises power") is treated as a hypothesis to test, not a finding (see S2).

## 1. Corpora and label definitions (fixed now)

| Corpus | Unit | Label used | Collapse | Licence (primary source) |
|---|---|---|---|---|
| HateXplain | post | 3-way label | binary: normal vs offensive/hatespeech | MIT |
| Measuring Hate Speech (MHS) | comment | `hatespeech` item, codes 0/1/2 as provided (meaning of codes to be confirmed from the dataset card and recorded before the run) | binary: code 2 vs {0,1} | CC-BY-4.0 |
| DICES-350 | conversation | `Q_overall` (Yes/No/Unsure) | binary: Yes vs {No,Unsure} | CC BY 4.0 |
| DICES-990 | conversation | `Q_overall` (replication of the above) | same | CC BY 4.0 |
| Wikipedia Talk (toxicity) | comment | `toxicity` (0/1) primary; `toxicity_score` (-2..2) secondary, ordinal | — | CC0 |
| GoEmotions | comment | each of the 27 emotions + neutral as a binary label, alpha per emotion | — | Apache-2.0 per HF card; **AMBIGUOUS** on the Google repo, so only derived statistics are redistributed |

Rules applied uniformly:
- Items with fewer than 2 ratings are dropped from agreement statistics, and the number dropped is reported.
- If an item has more than 5 ratings (for example MHS anchor items with hundreds), keep 5 chosen by seeded random draw. A sensitivity run keeps all.
- Ties are never resolved by insertion order. Items with no strict plurality have no majority label and are counted.
- GoEmotions contains Reddit usernames (`author`). They are never printed, stored in results or redistributed.

## 2. Primary analyses (per corpus, identical code path)

- **A1 Agreement.** Krippendorff's alpha (nominal; ordinal for `toxicity_score`) with a 95% item-cluster bootstrap CI (1,000 replicates, on at most 10,000 items, seeded). Reported side by side with Gwet AC1, Fleiss kappa (fixed-rater subsets only) and raw pairwise agreement. The point estimate always uses all items.
- **A2 Ceiling.** Mean modal share of an item's ratings, meaning the expected agreement of the best possible predictor with a randomly chosen annotator. It is called "the modal-label agreement ceiling", **not** an upper bound on system performance (Boguslav & Cohen 2017 argue agreement is not a strict bound).
- **A3 Margin.** Fraction of items where the top label leads the runner-up by exactly one rating, and fraction with no majority.
- **A4 MDE (model M1, replaces the old one).**
  - Binary labels, class-independent symmetric noise with single-rater error rate eta.
  - eta is estimated from pairwise agreement: P(two raters agree) = (1-eta)^2 + eta^2.
  - A gold label from the majority of k raters has error eta_k (computed exactly from the binomial). The observed accuracy gap between two systems is attenuated: E[observed gap] = (1 - 2·eta_k) × true gap.
  - The paired-difference MDE uses the assumed system-disagreement rate p_disc from {0.05, 0.10, 0.20, 0.30}, swept and never fixed, at n in {500, 1000, 2000, 5000}. The MDE on the true-gap scale = observed-scale MDE / (1 - 2·eta_k).
  - Model M1 is validated by simulation (S0) before it is used. If the simulation shows bias, the bias is reported and M1's claims are limited accordingly.
  - Multi-class extension: not attempted. Multi-class corpora are reported on their binary collapse for MDE.

## 3. Claim gate

Applicable checks for real corpora: **reliability, precision, effect_vs_mde, interval_excludes_zero, replication** (5 of the 11). The other six (depth_contract, coverage, annotator_pool, drift, rubric_specification, gold_calibration) depend on Rubricon's simulated pipeline and are reported as *not applicable* to external corpora, not silently skipped.

Default thresholds (configurable; not presented as universal): alpha block < 0.50, warn < 0.667, firm ≥ 0.800; CI half-width ≤ 0.075. Threshold-sweep plots: alpha_block in {0.30…0.60}, alpha_warn in {0.50…0.75}.
The 0.667/0.800 values are attributed to Krippendorff only after I have checked his primary text (the repository PDF returned HTTP 403 during Phase 0).

## 4. Simulation studies (known ground truth)

- **S0 Model check.** Simulate binary truth, K raters with symmetric noise, and two systems with a known true gap. Compare the predicted attenuation (1 - 2·eta_k) and MDE with the simulated ones.
- **S1 Gate error rates.**
  - Grid: prevalence {0.1, 0.3, 0.5}; target single-rater alpha {0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}; n {200, 500, 1000, 2000, 5000}; true gap {0, 0.01, 0.02, 0.03, 0.05}; p_disc {0.1, 0.2, 0.3}; 1,000 replicates per cell, master seed 20260929.
  - A "claim" is "A beats B". **False publication** = claim not BLOCKed although true gap ≤ 0 (or of the opposite sign). **False block** = claim BLOCKed although true gap ≥ 0.03.
  - Baselines compared on the same draws: no gate, and a plain paired test at 0.05 on the noisy labels. The gate is reported as useful only where it beats or complements these; if it does not, that is reported.
- **S1b Ablation.** Drop each of the 5 applicable checks in turn and report the change in false-publication and false-block rates and in the real-corpus block rate.

## 5. Claim 4 study (S2): items vs raters at a fixed budget

- Corpora with many raters per item: Wikipedia Talk (median 10), DICES (123 or 172 per item), MHS where ≥ 5.
- Fixed budget B labels; allocations k in {1, 2, 3, 5} raters per item (n = B/k items). Reference truth = majority of held-out raters disjoint from the sampled ones. Synthetic systems are built by flipping reference labels at controlled rates. Outcome: power to detect a true gap and ranking-flip rate.
- Stated in the paper as **conditional** (independent, item-homogeneous noise; paired accuracy difference). Contradicting results are reported. It is compared against Homan et al. (2026) and Pandita et al. (2026) after I have read them in full.

## 6. LLM-judge experiment (J1)

- **Items:** 1,000 items per corpus, seeded random sample of items with ≥ 3 raters, for HateXplain, MHS, Wikipedia Talk; all 350 for DICES-350.
- **Human reference:** leave-one-annotator-out human judge scored against the remaining raters on the same items (the ceiling), plus an Alternative Annotator Test (Calderon et al., ACL 2025) implementation following the paper (parameters recorded once I have read it in full).
- **Judges:** primary = one Claude model through the API (≤ $8 cap) and local qwen2.5:14b (Ollama). Secondary (labelled as such) = one OpenAI model (≤ $4), Google AI Studio free tier, Groq free tier. Exact model IDs and dates are written into `PREREG_ADDENDUM.md` **before** the first call. Temperature 0.
- **Prompts:** written once, hashed, stored under `prompts/`; formatting-only pilot on ≤ 20 items **outside** the evaluation sample, disclosed. No prompt is edited after any evaluation-sample output has been seen.
- **Metric:** judge-vs-panel exact match on items where the panel agrees, agreement per corpus with bootstrap CIs, and the same for the human LOO judge. Refusals and unparseable outputs are their own category, counted, and analysed both as missing and as errors.
- **Spend rule:** abort a run if projected cost exceeds the cap.

## 7. Retrospective check (R1) — exploratory

"Could this gap have been resolved by the labels?" applied only to public leaderboard numbers where test-set size and label reliability are known. No paper is accused. If fewer than 5 usable gaps are found, R1 is dropped from the paper and said so in the limitations.

## 8. Reproducibility and reporting rules

- Every number in the paper traces claim → results file → script → commit in `CLAIMS.md`; untraceable claims are deleted.
- Every citation is verified on a primary page before entering `references.bib`.
- Master seed 20260929; derived seeds by fixed offsets recorded in `assumptions.py`.
- Nothing is tuned on outcomes. Negative and null results are reported.
- Deviations: `DEVIATIONS.md`, with date and whether results had been seen.
