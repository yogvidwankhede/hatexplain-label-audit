# Distribution versions

Three short forms of the HateXplain label-quality audit. Every number here appears in `results/*.json` or is a declared assumption in `src/rubricon_field/assumptions.py`.

---

## (a) 280-character post

> HateXplain label audit: Krippendorff's alpha = 0.4597 across 253 annotators, 20,148 posts. 51.1% of gold labels rest on one vote. On a 2,015-item test split the MDE is 2.8 accuracy points — so a reported 2-point gap between two classifiers is a tie. Full writeup + code:

(270 characters including the trailing colon; add the link.)

---

## (b) LinkedIn version

I ran a label-quality audit of HateXplain (Mathew et al., AAAI 2021) — 20,148 posts, 253 crowd annotators, 60,444 individual judgements, three per post. Krippendorff's alpha on the three-way task is 0.4597, 95% CI [0.4452, 0.4729]. Fleiss' kappa is 0.4597 and Gwet's AC1 is 0.4689, so all three coefficients agree to within 0.0092: the moderate figure is not the kappa paradox, and no amount of rebalancing or annotator retraining would move it. Just under half the corpus (48.86%) is unanimous, and 919 posts carry three different labels, meaning no majority label exists for them at all.

Two findings that change how you read a leaderboard. First, the binary decision — normal versus offensive-or-hateful — is materially more reliable than the three-way one: alpha 0.5613 against 0.4597, a 22.1% relative improvement with non-overlapping bootstrap intervals. The offensive-versus-hatespeech boundary alone accounts for 40.7% of all disagreeing annotator pairs. Systems are being scored on a distinction that is noisier than the decision most deployments actually make. Second, there is a ceiling: an oracle naming each post's modal label scores 0.8143 against a randomly drawn annotator, so no classifier can exceed that, and an accuracy near it is measuring agreement with an annotator population rather than correctness.

The practical consequence: on a 2,015-item test split (10% of the corpus — a declared assumption, since I did not use the official split file), the minimum detectable effect at 80% power is 0.0279. A two-point gap needs 3,925 items. On all 20,148 posts it resolves at 0.0088. So a reported two-point difference on a standard-sized split cannot be distinguished from label noise. None of this is a criticism of the dataset — a reliability ceiling is a property of a subjective construct, and HateXplain is unusually transparent in publishing per-annotator labels, which is the only reason this was computable at all. Most datasets do not. Writeup and reproducible code in the comments.

---

## (c) Cold email to a benchmark maintainer / eval lead

**Subject: minimum detectable effect on a HateXplain-sized test split**

Hi [name] — I ran a label-quality audit on HateXplain's published per-annotator labels (20,148 posts, 253 annotators, 3 labels each) and found that on a 2,015-item test split, the minimum detectable effect for a paired accuracy comparison at 80% power is 0.0279, so a reported two-point gap between two classifiers is inside the noise floor; you need 3,925 items to resolve it.

The underlying reliability is Krippendorff's alpha = 0.4597 on the three-way task, and I checked that this is not the kappa paradox — Fleiss' kappa (0.4597) and Gwet's AC1 (0.4689) agree with alpha to within 0.0092, so prevalence skew is ruled out.

The more actionable part is that collapsing to the binary normal-versus-toxic decision raises alpha to 0.5613, a 22.1% relative improvement with non-overlapping intervals, with the offensive-versus-hatespeech boundary accounting for 40.7% of all disagreeing pairs.

None of this says the dataset is bad — the ceiling is a property of the construct, and HateXplain is one of the few benchmarks that publishes individual annotator labels at all, which is what made the audit possible.

My question: when you evaluate on this dataset, do you use the official 8:1:1 test split, or the full corpus — and would a reported MDE alongside each leaderboard delta be something you would find useful, or something you would consider noise in the results table?
