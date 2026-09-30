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
