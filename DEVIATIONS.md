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
