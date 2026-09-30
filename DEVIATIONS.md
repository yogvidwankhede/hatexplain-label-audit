# DEVIATIONS.md — changes relative to PREREG.md (tag prereg-v1)

Each entry says whether any result on the affected corpus had been seen.

## D1 — Leave-one-out ceiling added alongside the in-sample ceiling (2026-09-29)
- PREREG A2 defined the ceiling as the mean modal share. That estimator includes each rater's own vote in the modal label, so it is optimistic when raters per item is small (3 in HateXplain).
- Added `ceiling_leave_one_out`: predict each rater from the modal label of the *other* raters, ties split evenly. Both are reported as a bracket; neither is an upper bound on a system.
- Results seen before adding: none on the new corpora; HateXplain in-sample 0.8143 was already published. The added estimator was decided before the new-corpus runs completed.
- Consequence: the earlier README sentence "ceiling 0.8143" must be reported as the upper end of a bracket (leave-one-out HateXplain 0.6439).

## D2 — Separate paper seed (2026-09-29)
- PREREG says master seed 20260929. `assumptions.py` kept `SEED` (20210201) for the committed HateXplain studies so they reproduce byte for byte, and added `SEED_PAPER = 20260929` for all new analyses. No results seen.

## D3 — One-vote margin definition (2026-09-29)
- Reported as two numbers: exactly-one-vote lead, and no strict plurality. The earlier "51.1% at one-vote margin" equals their sum on HateXplain (46.58% + 4.56%), i.e. "lead of at most one vote".

## D4 — Gate simulation: heterogeneous-noise scenario and outcome definitions (2026-09-29, before any S1 run)
- PREREG S1 specified one generative model (independent symmetric rater noise). Added a second scenario, "het": 30% of items are hard (single-rater error rate 3x the easy items'), and the systems' gap differs by item type (gap on hard items = overall + 0.05, on easy items = overall - 0.0214, so the overall gap is unchanged). Reason: under independent noise the observed gap is only attenuated, so a reliability floor has little to protect against; het is the mechanism by which low reliability can flip a ranking.
- Grid unchanged from PREREG for iid; true-gap grid extended to {-0.03, 0, 0.01, 0.02, 0.03, 0.05} so false claims (true gap <= 0) exist. Het uses alpha in {0.4..0.9}; infeasible cells are skipped and counted.
- "Claim" = observed gap A - B > 0. False claim = true gap <= 0. Resolvable claim = true gap >= 0.03. Gate "publishes" a claim iff no check BLOCKs (ClaimLedger.publishable semantics).
- Interval for interval_excludes_zero and precision is the normal-approximation interval of the paired difference (vectorised), not a bootstrap; agreement between the two is tested on a sample of cells.
- No S1 result had been computed when this entry was written.

## D5 — Matched-operating-point baseline for the gate (POST HOC, 2026-09-29)
- After seeing aggregate S1 results (the gate publishes ~30% of resolvable claims vs ~66% for an unadjusted z>1.96 test, and only effect_vs_mde and reliability ever bind), added a fairer comparison: plain z-thresholds {1.96, 2.24, 2.58, 2.80, 3.00, 3.29}, so the gate can be compared with a significance test at the SAME resolvable-claim publish rate or the SAME false-claim rate.
- This is exploratory and labelled as such in the paper. Results seen before adding: S1 aggregates for iid and het (not the matched comparison itself). Simulation seeds are unchanged, so the earlier numbers do not move.
- A ablation bug (double-counted unblocked claims) was found and fixed before interpretation; a regression test covers it.

## D6 — Gate v2 and post-hoc scenarios (POST HOC, 2026-09-29)
- Decision (author, 2026-09-29): report the negative S1 result and design/validate a replacement gate.
- Gate v2 (all quantities computed from gold labels, rater votes and system outputs):
  (C1) direction evidence: z of the paired accuracy gap must exceed z* = 2.58;
  (C2) attenuation report: eta_hat from pairwise rater disagreement, eta_K for the K-rater majority, true-scale gap = observed gap / (1 - 2 eta_K) and true-scale MDE, reported, not used to block;
  (C3) contested-item consistency: the gap is computed separately on items where raters were unanimous and on items where they split; BLOCK a direction claim if the two gaps have opposite signs and their difference has |z| > 2.58; WARN if |z| > 1.96.
  z* = 2.58 and the C3 cut-offs were fixed here, before any v2 run.
- New scenarios, added because the pre-registered `het` (systems' gap larger on hard items) only hides real gaps and cannot produce false claims: `het_k-0.05` and `het_k-0.10`, where the systems' gap is smaller on the hard/contested items (kappa < 0). The direction was chosen after seeing that the original `het` produced no extra false claims. Reported as post hoc.
- Cells with infeasible joint probabilities are skipped and counted.

## D7 — Wider z-threshold grid for the matched comparison (POST HOC, 2026-09-29)
- In het k=-0.10 the v1 gate's operating point (false 0.0043) lies below the smallest false-claim rate the grid {1.96..3.29} reaches (0.0078), so a matched comparison was not possible. Added 3.5, 4.0, 4.5 to Z_THRESHOLDS. Results seen before adding: all S1/v2 aggregates. Seeds and earlier columns are unchanged.
- C3 was NOT modified after seeing that it adds little (D6 fixed its cut-offs); the paper reports it as is.

## D8 — Full-data bootstrap replaces the 10,000-item subsample (2026-09-30)
- PREREG A1 bootstrapped at most 10,000 items while the point estimate used all items; on GoEmotions:amusement the point (0.451) fell outside its subsample interval [0.453, 0.506]. `fast_alpha.py` computes the same cluster bootstrap on all units (validated exactly against rubricon, including ordinal and weighted replicates). Results seen: first-run CIs (all corpora). Point estimates are unaffected.

## D9 — S2 details fixed before running (2026-09-30)
- Pool = 5 raters per item; reference = majority of 5 other raters (Wikipedia Talk) or of all remaining raters (DICES); items need >= 10 ratings. MHS is excluded from S2: only 70 items have >= 10 ratings.
- Systems: B accuracy 0.80, p_disc 0.20, deltas {0.02, 0.03, 0.05}; two system models, `uniform` and `contested` (both systems err more where reference raters split, minority share >= 0.3; A's advantage only on clear items). The `contested` model was added here, before any S2 run, because PREREG asked that contradicting results be sought.
- Budgets: Wikipedia Talk {1500, 3000, 6000}; DICES-350 {300}; DICES-990 {450, 900} (n cannot exceed the item count). 2,000 replicates per cell.
- D9 addendum (same day, after a crash and before any S2 output existed): the `contested` model's rescaling to overall B accuracy 0.80 was infeasible on corpora where most items are contested. Replaced by fixed B accuracy 0.60 (contested) / 0.85 (clear), p_disc on clear items lowered to min(0.20, 0.30 - d) where needed, and cells needing d > 0.15 on clear items skipped and counted.

## D10 — Judge pilot details (2026-09-30, no evaluation-sample output seen)
- DICES-350 has no items outside the sample, so its 5 pilot items come from DICES-990 (seeded), which the same prompt applies to.
- anthropic SDK 1.x removed `temperature` from `messages.create()`; Haiku 4.5 still honours it, so it is sent via `extra_body` (sync pilot) and in the batch params (forwarded). The request sent to the API is unchanged from the addendum.
- Parser: after the pilot showed Haiku 4.5 answering "yes" followed by an explanation (truncated at max_tokens) on 2 of 5 DICES pilot items, a "first_word" tier was added between strict and lenient: if the first word of the answer is a label, that label is taken. Prompts were not changed. No evaluation-sample output existed.
- The first-word tier initially failed because newlines were deleted before splitting ("yes

The" -> "yesthe"); whitespace is now normalised first. Covered by tests/test_judges.py.

## D11 — Wording of the judge claim tested by gate v2 (2026-09-30)
- PREREG_ADDENDUM says gate v2 is applied to "judge X agrees with the panel at least as well as the held-out human". Gate v2's blocking check is a directional (superiority) test, so what it actually evaluates is "judge X agrees with the panel MORE often than the held-out human". The code was always the superiority test; only the label is corrected (results key renamed to gate_v2_claim_judge_beats_human). Noticed while reading partial results (OpenAI and qwen judges); no analysis choice changed.

## D12 — Rounding simulation floats for cross-hardware reproducibility (2026-09-30)
- The CI reproduction job found one simulation cell whose `mean_gap_true_est` differed in the 16th significant digit between two GitHub runners (0.0004777733053698053 vs ...054): numpy's summation order depends on the CPU's SIMD support. Derived floats in `gate_sim` outputs are now rounded to 12 decimals. No reported number changed: `gate_sim_summary.json` is byte-identical and every macro in `paper/numbers.tex` is unchanged.
