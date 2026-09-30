# What the human author must do

Nothing below has been done on your behalf. Nothing has been pushed, published, filed or submitted.

## Before any push
1. **Review both branches.** Both `rubricon` and `hatexplain-label-audit` are on `paper/claim-gated-audit`. Read `DEVIATIONS.md` in full; you must be able to defend every entry.
2. **Code of conduct contact.** `CODE_OF_CONDUCT.md` sends reports to "the maintainer, privately, using the contact details on their GitHub profile". Confirm that is what you want, or give an address.
3. **Push order.**
   1. Push rubricon's branch, open a PR and merge it after CI passes.
   2. Tag rubricon `v0.5.0` and publish a GitHub release using the CHANGELOG text.
   3. Push the audit repo. Its CI checks out `rubricon@v0.5.0`, so it fails until that tag exists.
4. **GitHub settings** (per repo):
   - enable private vulnerability reporting (SECURITY.md relies on it);
   - enable Dependabot alerts;
   - add topics such as `annotation`, `inter-annotator-agreement`, `krippendorff-alpha`, `llm-as-a-judge`, `evaluation`, `reproducibility`;
   - optionally enable Discussions.
5. **Good first issues.** Done: filed as issues #3-#6 (this repo) and #8-#12 (rubricon).
6. **Badges.**
   - Add a CI badge only after the workflow has run on GitHub.
   - Add a release badge only after the release exists.
   - Add a Zenodo DOI badge only after Zenodo mints one (connect the repo in Zenodo, then publish the release).
   - Do not add a PyPI badge. Nothing is on PyPI.
7. **OpenSSF Scorecard.** After pushing, run it against the real repositories, or add the official Scorecard action. The local results after fixes (on clean clones):
   - rubricon: 10 on binary artifacts, dangerous workflow, dependency updates, pinned dependencies, SAST, security policy, token permissions and vulnerabilities; License 9.
   - audit repo: the same, except Pinned-Dependencies 9 (checked before the fetch-script fix).
   - Fuzzing and Packaging are not addressed.

## Paper
1. **Author block and personal details.** Author name, affiliation (WashU), email and ORCID go in the preprint version only (`\usepackage[preprint]{acl}`). The review version must stay anonymous.
2. **Acknowledgements.** Add funding and thanks. Keep the AI-assistance paragraph; ACL requires it. Check it against how you actually used the tools.
3. **Anonymised code for review.** ARR is double-blind, so GitHub links reveal you. Use an anonymisation service (for example anonymous.4open.science) for the review version, or upload a zip as supplementary material.
4. **Read every cited paper you rely on.**
   - `paper/related_work_notes.md` marks which ones were read in full, from the abstract only, or from metadata only.
   - Krippendorff's book page for the 0.667/0.800 thresholds was not seen directly. Check it, or cite only the 2004 article.
   - ACL desk-rejects papers with hallucinated references. All 48 bib entries were fetched from primary sources, but verify any you are unsure of.
5. **Responsible NLP checklist.** Re-check `paper/responsible_nlp_checklist.md` against the final PDF, then enter it in the ARR form.
6. **Venue.** The primary target is ARR January 2027, then commit to ACL 2027.
   - The exact January 2027 date was not yet published on aclrollingreview.org/dates on 2026-09-30. Check it.
   - Every author must register as an ARR reviewer by the cycle's reviewer-registration deadline.
   - Backup 1: a 2027 workshop that accepts ARR-reviewed papers. Proposals were notified on 2026-10-02, so look for evaluation or human-label-variation workshops.
   - Backup 2: TMLR, for the long version.
7. **arXiv.** ARR allows non-anonymous preprints unless you tick its optional "no preprint" commitment. Decide before submitting. If you post, use the preprint version and fill in the preprint identifier in both `CITATION.cff` files.
8. **Licences.** Your code is MIT. Before any public release, confirm you are comfortable with the GoEmotions data licence being ambiguous. The repository releases only derived statistics for it.

## Money and keys
- Anthropic, OpenAI, Google and Groq keys are in `.env`, which is gitignored and was never printed. Rotate them if you ever suspect exposure.
- Spend: the three API judges used about $1.49 at list prices (paper macro `\jCostUsd`, computed from logged token counts). Confirm against your provider dashboards. The caps were $8 for Anthropic and $4 for OpenAI.
