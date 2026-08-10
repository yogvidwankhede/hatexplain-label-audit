# A two-point gap on HateXplain is not a result

If you are comparing two hate-speech classifiers on a HateXplain-sized test split, here is the number you need before you read the leaderboard: **0.0279**.

That is the minimum detectable effect, at 80% power and a two-sided 0.05 level, for a paired accuracy comparison on 2,015 items where the two systems disagree on 20% of them. Anything smaller than 2.8 accuracy points is inside the noise floor of that design. A reported gap of two points needs **n >= 3,925** items to be resolved reliably, and a 10% test split gives you 2,015.

The reason is not that the models are close. It is that the target they are scored against is itself noisy. Three crowd annotators labelled each of HateXplain's 20,148 posts, and they agree with each other at **Krippendorff's alpha = 0.4597**. Just over half the gold labels in the corpus rest on a single vote.

This post is a label-quality audit of HateXplain (Mathew et al., AAAI 2021), computed from the published per-annotator labels: 20,148 posts, 253 annotators, 60,444 individual judgements, exactly three per post. Everything below is measured from that file.

A note on register first. This is not a criticism of the dataset or its authors. A reliability ceiling is a property of a subjective construct, not a defect in the people who measured it. HateXplain is also one of the few benchmarks that publishes every annotator's individual label rather than only the aggregated gold — which is the only reason this audit is possible.

## The measurement

| coefficient | value | units |
|---|---|---|
| Krippendorff's alpha (nominal, 3-way) | **0.4597** | 20,148 |
| 95% cluster-bootstrap CI | [0.4452, 0.4729] | 6,000 |
| Raw pairwise agreement | 0.6439 | 20,148 |
| Fleiss' kappa | 0.4597 | 20,148 |
| Gwet's AC1 | 0.4689 | 20,148 |

Point estimates use every unit. The interval is a cluster bootstrap (400 replicates) resampling posts, not annotations, from a seeded 6,000-post subsample; that is a compute budget, and it is recorded next to every interval in the results.

Krippendorff's own thresholds are 0.800 for firm conclusions and 0.667 as the floor below which he would not draw even tentative ones (Krippendorff, *Content Analysis*). At 0.4597 with an interval 0.0277 wide, HateXplain's three-way task sits well below both — and the narrowness matters. This is not an uncertain estimate that might turn out fine with more data. It is a precisely measured moderate reliability.

## Three coefficients, one answer

The usual objection to a low kappa is the **kappa paradox**: when one category dominates, the chance-correction term inflates and the coefficient can collapse toward zero even when raters agree on 95% of items. It is a real failure mode that has cost teams weeks of unnecessary retraining. Gwet's AC1 (Gwet, 2008) is stable under prevalence skew, which makes the kappa-AC1 gap a diagnostic.

Here, the paradox does not fire, and it is worth being explicit about why:

- Raw agreement is **0.6439**, nowhere near the ~0.85 region where the paradox bites.
- The AC1-minus-kappa gap is **0.0092**, not the >0.25 a paradox would produce.
- The most common category holds **40.5%** of the annotation mass — no category is dominant enough to distort the chance model.

All three chance-corrected coefficients land within 0.0092 of each other. This is a negative result, and it strengthens everything that follows, because it removes the easy explanation. The moderate alpha is not a statistical artefact of an unbalanced label distribution. It is genuine disagreement about the construct. Practically: neither stratified oversampling of a rare class nor another round of annotator training would move this number.

## What the votes actually look like

| consensus mode | posts | share |
|---|---|---|
| unanimous (3/3) | 9,845 | 48.86% |
| 2-1 split | 9,384 | 46.58% |
| three different labels | 919 | 4.56% |

Fewer than half of HateXplain's posts have a label all three annotators chose. **51.1%** — 10,303 posts — sit at a margin of one vote: change one annotator's mind and the gold label changes or disappears.

The 919 three-way splits deserve precision, because this is where a downstream user can quietly invent data. On those posts the annotation protocol produces *no* majority and therefore no target label. Whatever happens to them next — dropped, resolved to the most severe label, resolved to the least — is a decision made outside the protocol. It silently determines 4.56% of any accuracy figure computed over the full corpus.

Under the binary collapse the picture changes: 68.3% unanimous, and no post can be unresolved at all, because a two-category vote among three raters always produces a majority.

## The three-way task is where the noise lives

This is the most immediately useful finding in the audit.

| task | alpha | raw agreement | 95% CI |
|---|---|---|---|
| 3-way (normal / offensive / hatespeech) | 0.4597 | 0.6439 | [0.4452, 0.4729] |
| binary (normal vs toxic) | **0.5613** | 0.7887 | [0.5382, 0.5723] |
| offensive vs hatespeech (conditional) | 0.4181 | 0.7138 | [0.4065, 0.4451] |

Collapsing the three-way label to the binary decision — is this post normal, or is it offensive-or-hateful — moves alpha by **+0.1016**, a **22.1% relative improvement**, and raw agreement from 0.6439 to 0.7887. The bootstrap intervals do not overlap.

Restricted to the 12,334 posts where at least two annotators both chose a non-normal label, alpha on the offensive-versus-hatespeech call is **0.4181**. (That conditioning selects on the outcome, so read it as a description of the boundary rather than a reliability estimate for a task anyone runs.) Of the 21,525 disagreeing annotator pairs in the corpus, **40.7%** are offensive-versus-hatespeech — disagreements about *which kind* of unacceptable a post is, not about whether it is unacceptable.

The practical conclusion: annotators broadly agree on whether a post should be acted on, and disagree on what to call it. Most deployed moderation systems only need the first decision. Systems on this benchmark are being scored on a distinction that is materially less reliable than the one their deployment actually makes.

## The accuracy ceiling

Take the best possible predictor — an oracle that names each post's modal label — and score it against one of the three annotators drawn at random. It gets **0.8143** on the three-way task and **0.8943** on the binary one.

No classifier can beat that, because the remaining 18.6% is disagreement among the humans, not error in the predictor. Model scale does not remove it; a better prompt does not remove it.

Two consequences follow. First, three-way accuracies in the high 60s are much closer to the achievable ceiling than to 100%, so the headroom a leaderboard implies is largely not there. Second, a reported accuracy at or above 0.8143 is not measuring correctness. It is measuring agreement with this particular annotator population — which may be exactly what you want, but it is a different claim and should be written as one.

## What a test split can resolve

| n items | MDE (observed scale) | MDE (true-score scale) |
|---|---|---|
| 1,000 | 0.0396 | 0.0467 |
| 2,015 (assumed 10% test split) | **0.0279** | 0.0329 |
| 5,000 | 0.0177 | 0.0209 |
| 20,148 (full corpus) | 0.0088 | 0.0104 |

All cells assume two systems disagreeing on 20% of items, 80% power, alpha = 0.05. The true-score column divides by the square root of the three-rater gold reliability (0.7185 by Spearman-Brown step-up), which is what you want if the question is about the underlying construct rather than agreement with these labels.

HateXplain ships an 8:1:1 split. The official split file was not used here, so the test-split size is a **declared assumption** — 10% of the corpus, 2,015 posts — and it is named as one in the study's assumptions module alongside every other modelling choice.

This is the section that cashes out the opening. At n = 2,015 the smallest resolvable difference is 2.8 accuracy points, so a published two-point gap is inside the noise floor: you would need 3,925 items. On all 20,148 posts the same comparison resolves at 0.0088, under one point. The constraint is the size of the split, not the size of the corpus.

## One annotator produced 9.5% of the labels

Annotator #4 contributed **5,730 labels**, 9.48% of all 60,444, and appears on 28.4% of posts. The top five annotators produced **22.3%** of the corpus between them. Median load across the pool is 41 judgements.

Remove annotator #4 entirely and alpha moves by **-0.0207**. Across the ten heaviest annotators the largest movement is the same 0.0207. The coefficient is robust to any single person, and the reason is structural: alpha averages disagreement over *posts*, and no individual is on enough posts to shift that average, even at 28.4% coverage.

The number that actually matters is different. Removing that one annotator destroys **2,300 gold labels — 11.4% of the corpus** — because their vote was one of the two that created the majority. The statistic is stable; the labels are not. If you are reasoning about robustness to annotator turnover, sampling error in the pool, or a single rater's idiosyncratic reading of the codebook, alpha is the wrong instrument to check. Gold-label exposure is the right one.

## What this does not show

- **It does not show that HateXplain is bad work, or that the authors erred.** A reliability ceiling is a property of the construct being measured. 0.4597 is an unremarkable figure for subjective content moderation; the problem is that it rarely gets propagated into the conclusions built on top of the data. No comparison corpus was analysed here, so nothing supports a claim that HateXplain is noisier than its peers.
- **It does not show that any specific published result is wrong.** The MDE analysis says what size of gap a design of a given size can resolve. It does not evaluate anybody's paper.
- **The test split is assumed, not read.** 2,015 = 10% of the corpus. The official split file was not reachable from this repository. Use the real split size and the MDE moves.
- **The MDEs are a floor, not a ceiling.** They cover item-sampling variance and report the attenuation correction separately. They do *not* cover gold-panel resampling variance: these labels are one realisation of a three-annotator draw, and a different draw from the same pool would produce different gold labels for a substantial share of posts. That variance is not identifiable from a single three-rater panel, so it is left unestimated rather than guessed at. The observable proxy is the 51.1% of posts at a one-vote margin.
- **Per-community agreement is read against a like-for-like baseline.** Every targeted community sits below the corpus alpha — but so does the baseline over all 12,466 targeted posts, at 0.3171. Conditioning on a target removes the easy `normal` units and restricts range, which lowers every chance-corrected coefficient by construction. Against that baseline the eleven communities clearing a 250-post floor span 0.1216 in alpha. The comparison is still confounded with label mix and is reported as such.
- **No LLM judge was run.** The "judge" in the companion study is a held-out human annotator, used to exercise the scoring path on real labels and to establish a ceiling. No model was called anywhere in this work.

## What to do about it

Four things, all cheap:

1. **Report the binary collapse alongside the three-way score.** It costs one extra line, it is 22.1% more reliable, and it is closer to the decision most deployments make.
2. **Publish per-annotator labels.** HateXplain does. Most datasets do not, and without them none of this is computable by anyone downstream — not the ceiling, not the MDE, not the consensus structure.
3. **Put an MDE next to any leaderboard delta.** It is one line of arithmetic given the eval size and an assumed system-disagreement rate, and it converts "we improved by 2 points" into a claim a reader can check.
4. **Treat sub-3-point gaps on a HateXplain-sized test split as ties.** Not as small wins. As ties.

## The tooling

This audit ran on [Rubricon](https://github.com/yogvidwankhede/rubricon), an open evaluation harness whose agreement implementation is validated against Krippendorff's published 2011 reference dataset — nominal 0.7434, ordinal 0.8154, interval 0.8491, ratio 0.7974, matching to within 0.0004 across all four distance metrics. Every assumption in this study, from the seed to the 10% split fraction, lives in one file, is echoed into every result JSON that depends on it, and can be changed in one place.

## Reproduce this

```bash
git clone https://github.com/yogvidwankhede/rubricon
# place HateXplain's dataset.json at field/data/hatexplain.json
cd field
make all      # install -> run the three studies -> tests
make report   # regenerate results/REPORT.md from results/*.json
```

Every number in this post is read from `results/study_a.json`, `study_b.json`, `study_c.json` or `summary.json`, or is a declared assumption in `src/rubricon_field/assumptions.py`. None was typed in by hand.

---

**Dataset.** Mathew, B., Saha, P., Yimam, S. M., Biemann, C., Goyal, P., & Mukherjee, A. (2021). HateXplain: A Benchmark Dataset for Explainable Hate Speech Detection. *Proceedings of the AAAI Conference on Artificial Intelligence*, 35(17), 14867-14875.

**Coefficients.** Krippendorff, K. *Content Analysis: An Introduction to Its Methodology* (alpha; the 0.667 and 0.800 thresholds). Gwet, K. L. (2008), *British Journal of Mathematical and Statistical Psychology* 61(1), 29-48 (AC1). Fleiss, J. L. (1971), for kappa.
