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

## AI-assisted contributions
AI assistance is allowed. Say in the PR description what was assisted, and make sure you
have read, run and understood every line you submit. You are responsible for its correctness.

## Code of conduct
Participation is governed by [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
