# Related-work reading notes

Compiled 2026-09-30. Every entry is in `references.bib` under the same key, with the URL it was checked against.

**Legend**
- **FULL**: I read the full text in this session. Page and section pointers refer to the PDF or HTML I read, as stated.
- **PARTIAL**: I read only the named sections.
- **ABSTRACT-ONLY**: I read only the abstract, from the Anthology page, Crossref, the proceedings page or the arXiv API. Do not cite numbers from these papers beyond what the abstract states.
- **METADATA-ONLY**: I verified the bibliographic record but read no abstract or text in this session.

Verbatim quotes are kept under 15 words and put in quotation marks. Everything else is paraphrase.

---

## 0. Novelty check. Read this first.

### (i) Correcting an accuracy gap for gold-label noise with a (1 − 2η) factor, binary majority labels

**This is already in the literature, twice.**

1. **`lam2003evaluating`** (IJCAI 2003), Eq. (2), printed p. 514, §3.
   - They assume binary labels and a labeler whose errors are independent of the classifier's errors.
   - They derive `Pr[y ≠ ŷ] = (Pr[ŷ ≠ ỹ] − Pr[y ≠ ỹ]) / (1 − 2 Pr[y ≠ ỹ])`. In words: true error = (apparent error − mislabel rate η) / (1 − 2η).
   - I read this equation from the typeset image in the PDF. The text extraction dropped it.
   - This is a per-classifier correction. Applied to two classifiers against the same noisy labels, it gives observed gap = (1 − 2η) × true gap. That is exactly the leaderboard-gap correction.
   - They also derive how many noisy labels are worth one clean label (Eq. 3, p. 515), plus bounds for the case where labeler and classifier errors are dependent (§6, per the abstract).
2. **`dorner2024dont`** (ICML 2024), §3, Proposition 1.
   - They assume independent classifier and labeler errors and label accuracy q.
   - The expected gap indicator is E[G] = (2q − 1)ε, where ε is the true accuracy margin. With η = 1 − q, that is (1 − 2η)ε.
   - Proposition 2 (§3.2) gives the version for correlated classifier and label errors.
   - Their target is the true label. In their Related Work paragraph on disagreement, they say they focus on the case where the target is a "(fictitious) majority vote over the whole crowdworker population".

**What might still be new (my assessment):**
- Using the (1 − 2η) factor as a published, per-benchmark de-attenuation of reported leaderboard gaps.
- Estimating η from the benchmark's own reliability (for example from α or split-half agreement, via Spearman–Brown-style reasoning).
- Combining this with a significance or decision rule.

I did not find a paper that does this end to end for LLM leaderboards. Related but different work:
- `northcutt2021pervasive` and `nahum2025llms` correct labels empirically and re-score models (ABSTRACT-ONLY). Neither applies an analytic (1 − 2η) correction.
- `gordon2021disagreement` (METADATA-ONLY) adjusts metrics for disagreement. I have not read its method.

**Recommended framing:** cite Lam & Stork 2003 and Dorner & Hardt 2024 as the source of the attenuation identity. Claim novelty only for how it is operationalised or applied.

### (ii) A reliability-threshold publish/block gate for evaluation claims

**No exact precedent found.** Closest items:

- **`caban2026measurement`** (arXiv preprint, Aug 2026; PARTIAL: §2.3 and §8 read).
  - Proposes tiered Krippendorff-α thresholds for agentic-AI evaluation: α ≥ 0.67 for exploratory capability evaluation, α ≥ 0.70 for safety, bias, fairness and risk scoring, and α ≥ 0.80 for scoring that "gates model deployment, compliance certification, or regulatory reporting" (Prescription 5, §8).
  - Prescription 8 asks for IRR metric, threshold and rationale to be required fields in evaluation reporting cards.
  - This is prescriptive guidance, not an algorithmic gate that blocks a specific leaderboard claim. It is the closest prior art and must be cited.
  - Its α = 0.67 floor cites Krippendorff's book, pp. 241–243.
- **Judge-acceptance gates**, which are a different object: they gate whether an LLM judge may replace humans, not whether a claim may be published.
  - The alt-test's pass rule ω ≥ 0.5 (`calderon2025alternative`).
  - The simplified leave-one-out (LOO) rule in `li2026llm`.
  - Two 2026 preprints I skimmed but did **not** put in the bib: GroundedGEO, arXiv 2609.25189, uses a "frozen gate" of macro-F1 ≥ 0.75 and κ ≥ 0.60 for automatic judges. Jarmak, arXiv 2608.13867, gates two judges on Cohen's κ.
- **Guidance only:** Krippendorff (`krippendorff2004reliability`) sets reliability standards for accepting data, not for gating downstream model-comparison claims.

### (iii) Our claim that, at a fixed label budget, one rating per item maximises power for a paired accuracy comparison against a majority label

**Already proven for binary labels by `dorner2024dont`** (Theorem 1 informal in §1; Theorem 2 in §4).

- For a large enough budget k, the chance of identifying the better of two binary classifiers is highest at m = 1 label per item.
- Assumptions (§3.2):
  - Assumption 1: no biased label accuracy, q_b ≥ q_w.
  - Assumption 2: no biased heterogeneity.
- They conjecture the result for all n and checked it numerically over billions of parameter settings (§1).
- Exceptions they list in §5:
  - when estimating the exact risk matters more than ranking;
  - when unlabelled data is costly;
  - when the assumptions are violated.
- Multiclass (plurality vote) is left open (§5).

Our paper must cite this as prior work, not claim it. Homan et al. does **not** contradict it; see §2(b).

---

## 1. Papers read in full (positioning-critical)

### calderon2025alternative: The Alternative Annotator Test for LLM-as-a-Judge (ACL 2025)

**Status:** FULL. Read in the ACL Anthology PDF (2025.acl-long.782, proceedings pp. 16051–16081). PDF page p corresponds to proceedings page 16050 + p. I also read arXiv v4 HTML.
- The arXiv v4 numbers the appendices differently: its D.x corresponds to the ACL version's C.x.
- The arXiv HTML also drops text at the "n < 30" sentence, so use the ACL PDF.

**Summary.** The alt-test asks whether an LLM can replace human annotators. It needs a small subset of items annotated by at least three humans. For each human annotator j in turn:
1. Hold j out.
2. Score how well the LLM matches the remaining annotators on each item.
3. Score how well j matches the same remaining annotators.
4. Test whether the LLM's advantage probability beats j's, minus a cost-benefit margin ε.

The per-annotator p-values get a false-discovery-rate correction (Benjamini–Yekutieli). The winning rate ω is the share of annotators the LLM "beats", and the LLM passes if ω ≥ 0.5.

The paper also proposes the Average Advantage Probability ρ for comparing judges. Across 10 datasets, 6 LLMs and 4 prompting methods:
- at least one LLM passes on 9 datasets;
- no LLM passes on MT-Bench or SummEval (Table 2);
- closed models beat the open ones they tried;
- few-shot prompting helps.

**Exact procedure (for implementation):**

- **Notation (§3, p. 4 / 16054).**
  - n items x_i; m annotators h_j; annotation h_j(x_i); LLM prediction f(x_i).
  - `[-j]` means all annotator indices except j.
  - 𝕀_j is the set of items annotated by h_j.
  - ℍ_i is the set of annotators who annotated x_i.
- **Alignment scoring function, classification (§3.1, p. 4).** This is the version in the arXiv HTML; the ACL PDF shows the same formula.
  ```latex
  \texttt{ACC}(f,x_{i},j)=\frac{1}{|{\mathbb{H}}_{i}|-1}\sum_{k\in{\mathbb{H}}_{i}[-j]}\bm{1}\{f(x_{i})=h_{k}(x_{i})\}
  ```
  - For the held-out human, use the same function with f replaced by h_j, i.e. ACC(h_j, x_i, j).
  - Implementation note: the normaliser |ℍ_i| − 1 assumes h_j ∈ ℍ_i, i.e. the item is one that h_j annotated (i ∈ 𝕀_j). Items need at least one other annotator. The same formulas are confirmed in the ACL PDF, p. 4.
  - Other variants: −RMSE for continuous labels and SIM for free text.
  ```latex
  -\texttt{RMSE}(f,x_{i},j)=-\sqrt{\frac{1}{|{\mathbb{H}}_{i}|-1}\sum_{k\in{\mathbb{H}}_{i}[-j]}(f(x_{i})-h_{k}(x_{i}))^{2}}
  ```
- **Per-item win indicators (§3.2, p. 4).**
  ```latex
  W_{i,j}^{f}=\begin{cases}1,&\text{if }S(f,x_{i},j)\geq S(h_{j},x_{i},j)\\ 0,&\text{otherwise}\end{cases}
  ```
  - W^h_{i,j} is the same with ≤. Ties therefore give both indicators the value 1, so d = 0.
- **Advantage probabilities.**
  ```latex
  \rho_{j}^{f}=\hat{\mathbb{P}}(\text{LLM}\succeq h_{j})=\hat{\mathbb{E}}[W_{i,j}^{f}]=\frac{1}{|{\mathbb{I}}_{j}|}\sum_{i\in{\mathbb{I}}_{j}}{W_{i,j}^{f}}
  ```
  - ρ^h_j is defined the same way from W^h.
- **Hypotheses (§3.3, p. 4).** One per annotator, one-sided.
  ```latex
  \mathbf{H_{0j}:}\rho_{j}^{f}\leq\rho_{j}^{h}-\varepsilon\quad\text{vs.}\quad\mathbf{H_{1j}:}\rho_{j}^{f}>\rho_{j}^{h}-\varepsilon
  ```
- **Test (§3.3, pp. 4–5).** The paper calls it a "paired t-test" (citing `dror2018hitchhikers`) on d_{i,j} = W^h_{i,j} − W^f_{i,j}. The Summary paragraph (p. 5) calls it "one-sample proportion t-tests".
  ```latex
  t_{j}=\frac{\bar{d}_{j}-\varepsilon}{s_{j}/\sqrt{n}}\quad s_{j}=\sqrt{\frac{\sum_{i=1}^{n}\left(d_{i,j}-\bar{d}_{j}\right)^{2}}{n-1}}
  ```
  - H1 means d̄_j < ε, so the p-value is the **lower-tail** Student-t probability of t_j.
  - The formula writes n and sums over i = 1..n. In practice n should be |𝕀_j|, the items annotated by h_j. Our reading is that the paper leaves this implicit.
  - If n < 30, use a non-parametric test such as the Wilcoxon signed-rank (p. 5; FAQ p. 16).
  - Reject when p < α, typically α = 0.05.
- **Winning rate and pass rule (p. 5).**
  ```latex
  \omega=\frac{1}{m}\sum_{j=1}^{m}{\bm{1}\{H_{0j}\text{ is rejected}\}}
  ```
  - Pass if ω ≥ 0.5.
  - Footnote 5: the 0.5 is a hyperparameter; stricter thresholds can be used in high-stakes domains.
  - C.3 gives a quality-weighted version: ω = Σ_j Q_j·1{reject} / Σ_j Q_j.
- **Multiple-comparison correction (p. 5).**
  - Benjamini–Yekutieli (BY) FDR control, because the m hypotheses are dependent. Each h_j's score depends on the other annotators.
  - Target q = 0.05.
  - Algorithm 1 (App. C.6, p. 21): sort p-values; threshold(i) = (i/m) · q / Σ_{j=1}^m (1/j); reject p_(1)..p_(i*) for the largest i* with p_(i*) ≤ threshold(i*).
  - This is BY, not Benjamini–Hochberg (BH).
  - For several domains, apply BY across all domain × annotator p-values and report "passes in X of D domains" (FAQ p. 16).
- **ε defaults (FAQ p. 15; §B.1 p. 17; §5 p. 7).**
  - Experts: ε = 0.2.
  - Skilled annotators (undergraduates, trained workers): ε = 0.15.
  - Crowd workers: ε = 0.1.
  - Effective range is 0.05–0.3. Above 0.3 almost every LLM passes; below 0.05 almost none do (§B.1).
  - Use ε = 0 when testing whether an LLM *outperforms* humans against gold labels (§C.5).
- **Minimum sizes.**
  - At least 3 annotators (§1 p. 2; FAQ p. 15). The procedure runs with 2 but is "less reliable".
  - At least 30 items for the t-test normality assumption (§5.1 p. 9; FAQ).
  - 50–100 items suffice in most cases (FAQ p. 16; §6).
  - Bootstrap results (§5.1, Fig. 2, 3 annotators, best LLM per dataset by ρ): with ε = 0.2, in most cases the LLM starts passing before 100 items, and in half the datasets before 50. With ε = 0.1 it takes about double, typically 100–150 items. On LGBTeen, MT-Bench and SummEval it never passes, however many items are used.
- **Average Advantage Probability (§3.4, p. 5).**
  ```latex
  \rho=\frac{1}{m}\sum_{j=1}^{m}{\rho_{j}^{f}}
  ```
  - Interpretation: the probability that the LLM is as good as or better than a randomly chosen annotator.
  - ρ involves no hypothesis test, so it does not depend on ε, and in expectation not on n.
- **Theorem 1 (§3.4; proof in App. D).** With S = ACC the optimal judge is the majority vote MV(x_i). With −RMSE it is the mean annotation. Both reach ρ = 1.
- **Extensions.**
  - C.1: class imbalance, via inverse-probability weighting by the leave-one-out majority label MV_j(x_i). This gives weighted ρ^{f,π}_j and an effective sample size n^π = (Σπ)² / Σπ².
  - C.2: single expert; the LLM and non-experts are each compared with the expert.
  - C.3: annotator quality weights Q_k.
  - C.4: subjective tasks.
  - C.5: LLM versus humans against gold labels, with ε = 0.
- **Known weakness (§C.1, p. 18).** With skewed labels, an LLM that always predicts the majority class can pass because of the many ties. SummEval Consistency has 89% '5' labels.

**Numbers we may cite.**
- Table 1 (p. 6): dataset agreement. Examples: MT-Bench, 3 experts, Fleiss κ 0.49; WAX, κ 0.26.
- Table 2 (p. 6): ω and ρ per LLM.

**Relation to our paper.**
- We implement the alt-test as a judge-validation baseline.
- It is a *replacement* test for judges. It is not a test of whether a model-vs-model leaderboard gap is reliable.
- Its leave-one-annotator-out design needs multiply annotated items.
- Its pass rule ω ≥ 0.5 with BY correction is one example of a gate on judges, not on evaluation claims.
- Its Limitations section (p. 10) says high human disagreement makes passing unlikely. That is consistent with a reliability-gated view.

---

### homan2026many: How Many Ratings per Item are Necessary for Reliable Significance Testing? (EACL 2026 Findings)

**Status:** FULL. Read in arXiv v3 (HTML and PDF); page numbers below are arXiv PDF pages. The Anthology version is 2026.findings-eacl.223, pp. 4258–4273. Note that the arXiv PDF lists authors in the order Homan, Korn, Pandita, Welty.

**Summary.** The paper extends the VET simulator (Wein et al. 2023) to ask how to split a fixed budget N × K between items N and responses per item K when comparing two models A and B against gold data G.

Setup (§4, App. A.1):
- G, A **and** B are all matrices of N × K responses.
- A is drawn from the same fitted item distributions as G, so it is an ideal model.
- B is perturbed by δ_i ~ Unif(−ε, ε) on the item means.
- p-values come from a multistage bootstrap over items and responses, one-sided, with α = 0.05 (App. A.2).

Datasets and metrics:
- Parametric fits (truncated or folded normal, etc.) to Stanford Toxicity (5 raters per item) and MultiDomain Agreement (5 per item), plus Amazon, HS-Brexit, ConvAbuse, ArMIS and MHS in App. B.
- Metrics:
  - Γ_MAE: the difference in absolute error between each model's per-item mean and the gold per-item mean.
  - Γ_Wins: the fraction of items where A's absolute error is smaller than B's.
  - Γ_MEMD: mean Earth mover's distance.

**Main claims.**
- For many metrics, even 5–10 responses per item (from each model and from the human team) is not enough (abstract).
- For a fixed N × K, p-values fall as K rises for MAE, with an inflection before K = 500 (§5.3, p. 6, Figs. 3–4).
- "as many as 100 responses per item" can beat fewer responses per item (§1 contributions, p. 2).
- Conclusion (§7, p. 9):
  - "Even when using 1000 items, at least 25 raters are needed" for significance with MAE.
  - For MAE and mean EMD, more responses per item is the better split.
  - For **Wins the opposite holds: more items and fewer responses per item give lower p-values.**
  - In real data, MAE was more sensitive than Wins.
- App. B, Tables 1–3: the minimum p-value is usually reached at K = 100, except the MHS dataset, where it is K ∈ {5, 10}.
- Limitations: responses are treated as independent across items, i.e. no rater identity (fixed in `pandita2026improving`). The simulator compares a near-perfect model with a perturbed one.

**Does it contradict our claim** ("at a fixed label budget, one rating per item maximises power for a paired accuracy comparison against a majority label")?
- **No.** Their target and metrics differ.
  - Models are stochastic and are themselves summarised by K responses.
  - The metrics (MAE, EMD) are sensitive to the per-item mean or distribution of human responses. More K sharpens the target of a *continuous* distance.
- Their one discrete, per-item comparison metric (Wins) behaves like our setting and favours more N and fewer K (§6 Discussion, pp. 7–8; §7).
- They do not study deterministic classifiers scored by accuracy against a majority label, which is the setting of `dorner2024dont`.
- Suggested wording: "Homan et al. find that K matters for distribution-sensitive metrics (MAE, EMD). For per-item discrete comparisons (Wins) they find the opposite, consistent with our result and with Dorner & Hardt (2024)."

**Relation to our paper.** We should acknowledge it as the main voice for "more K". We should also be explicit about the metric and target (majority label versus response distribution) under which our m = 1 result holds.

---

### pandita2026improving: Improving Reproducibility in Evaluation through Multi-Level Annotator Modeling (arXiv 2605.13801)

**Status:** FULL. Read in arXiv v2 PDF, 4 Aug 2026.

**Summary.** The paper replaces the parametric VET simulator with a non-parametric multi-level bootstrap that keeps rater identity. Three schemes:
- S1: resample items and raters in a fully crossed design.
- S2: resample items, then raters within each item.
- S3: stratified batches, where the same raters see each batch.

Model A reuses G's items with resampled raters. Model B perturbs a fraction ε of responses (§3.2). The data are:
- DICES-350: 123 raters, fully crossed.
- Stanford Toxicity: 107,620 items, 17,280 raters, batches of 20.
- D3code: 4,554 items, 4,309 raters.

They test 8 metrics (Accuracy on plurality vote, MAE, Wins, precision, recall, F1, KL divergence, Jensen–Shannon distance) over N × K from 100 to 50,000 and ε ∈ {0.1, …, 0.4} (§4.3).

**Findings.**
- RQ1 (§5, p. 6–7): modelling rater behaviour across items (S1, S3) raises p-values and needs larger budgets than S2. Ignoring rater identity **underestimates** p-values (§1, p. 2).
- RQ2: the optimal K rises. For MAE, DICES goes from K = 40 (S2) to 100 (S1); Toxicity from K = 9 (S2) to 20 (S3).
- RQ3: batch structure (S3) roughly multiplies the required budget. Toxicity accuracy needs N × K = 500 under S2 but 2500 under S3 (§4.4).
- Table 1 (p. 5), ε = 0.3, minimum N × K with p < 0.05:
  - For **Accuracy on Toxicity the optimal K = 1** (S2 and S3), and the same for Precision, Recall and F1 in several cells.
  - For DICES, Accuracy optimum is K = 40.
  - MAE, Wins and JSD often want K = 40–100.
- Conclusion: distribution-sensitive metrics plus higher K is an efficient route to significance.

**Relation to our paper.**
- Partly supportive: on Toxicity, accuracy against a plurality label is best at K = 1.
- Caution: in their setup the *models* are also summarised by K responses, so "accuracy" here is not a deterministic classifier's accuracy.
- Their point that ignoring rater dependence overstates power applies to any bootstrap CI we report. We should cluster by rater where rater IDs exist.

---

### li2026llm: LLM Judge Validation Under Sparse Overlap: From Inference to Design (arXiv 2609.31857)

**Status:** FULL (main text; appendices skimmed). The paper exists: v2 dated 29 Sep 2026. Authors: Junxuan Li and Arko Mukherjee (Adobe), Soumyabrata Pal (Adobe Research). arXiv comment: "Accepted at NeurIPS 2026".

**Summary.** The paper asks how the alt-test-style leave-one-out validation of an LLM judge degrades when only a fraction ρ of items get more than one human label. Its version of the validation rule (§2.3, Eqs. 3–6) is:
- a pooled pairwise observed agreement S(j; r_k) of judge versus held-in humans, compared with S(r_{−k}; r_k) of the held-out human;
- T_k = 1[S(j; r_k) ≥ S(r_{−k}; r_k) − ε] with **ε = 0.05 fixed**;
- accept the judge if ω = mean T_k ≥ 0.5.

This is **a simplified, test-free variant of the alt-test.** It has no per-annotator t-test and no BY correction, and it does not say so explicitly.

Results:
- Theorem 1 (§2.2, p. 4): the sparse-estimator variance is γ_F(π)/m, where m is the overlap count.
  - γ is bounded for raw agreement p̂_o and for Gwet's AC1.
  - γ diverges under label skew for α (as (1 − p_e)^{-2}) and for κ (as (1 − p_e)^{-4}).
- Theorem 2 (§2.4, p. 5): overlap-planning formula m* = ⌈z²σ²/δ²⌉, with certification and ranking variants.
- Stratified shared overlap (STRAT, §3) roughly halves false rejections.
- Real-data claims, over 10 judges and 4 matrices (WAX, CeBaB-aspects, CeBaB-stars, SummEval):
  - At ρ = 0.05, mean top-1 wrong-best-judge rate is 65%; it is 36% at ρ = 0.25 (Table 2, p. 8).
  - STRAT cuts the mean false-rejection rate from 29% to 14% at ρ = 0.05 (§4.4, p. 8).
  - Borderline judges (ω = 0.5) stay at 30–44% wrong decisions (p. 9).
- Recipe (§5): raw agreement p̂_o, STRAT, ρ ≥ 0.25, 3–5 raters, and treat ω ≈ 0.5 as inconclusive.
- Assumes i.i.d. items and non-self-selected annotation masks.

**Relation to our paper.** This is the most recent work on design for judge validation. It supports the claims that sparse overlap and label skew make κ/α-based decisions unstable and that borderline verdicts should be abstained on. It does not address leaderboard gaps or noise correction. If we compare against it, note that its "LOO pipeline" is not the full alt-test.

---

### resnick2021principled: Principled Evaluation with Human Labels: One Rater at a Time and Rater Equivalence (arXiv 2106.01254 v3, Apr 2026)

**Status:** PARTIAL. Read §1–§4 fully in arXiv HTML; §5–§8 section headings only. **No peer-reviewed version found** (no arXiv journal_ref; the Crossref title search returned nothing).

**Summary.** The paper defines two properties of an evaluation setting:
- **Objectivity:** a ground truth exists.
- **Equanimity:** all misclassifications are equally harmful.

Under the "subjective utility model" (no objectivity, or objectivity without equanimity):
- Scoring against one randomly chosen rater is an unbiased estimate of utility (Claim 1, §4.1).
- Scoring against a k > 1 majority vote is biased, because the most frequent label is over-represented (Claim 2).
- Only k_e = 1 guarantees a reliable ordering of classifiers (Claims 4–5, §4.2).
- Recommendation: "average the scores, don't score against the average label" (§1).

Under the objective model with no ground-truth labels there is no justified panel size, and larger panels can reverse the true ordering (App. H, per §1).

It also introduces power curves and **rater equivalence**: the smallest human benchmark panel whose combined label beats the classifier (§5). It gives an optimal "Anonymous Bayesian Combiner" for cross-entropy scoring (§6.3).

**Relation to our paper.**
- It argues against majority-label evaluation on *validity* grounds (the subjective utility model).
- Our m = 1 argument, like Dorner & Hardt's, is about *power* under an objective majority target.
- We should state which utility model we assume.
- Rater equivalence is a natural alternative reporting unit. `wulczyn2017ex` already reported a classifier "as good as the aggregate of 3 crowd-workers" (abstract).

---

### boguslav2017inter: Inter-Annotator Agreement and the Upper Limit on Machine Performance (MEDINFO 2017)

**Status:** ABSTRACT-ONLY. The full text was not openly retrievable: the IOS Press page gave the abstract only, and the PubMed record has no PMC copy. DOI 10.3233/978-1-61499-830-3-298 resolves, and Stud Health Technol Inform vol. 245, pp. 298–302, was confirmed via PubMed (PMID 29295103) and IOS Press.

**Summary (from abstract).**
- Traces the "IAA is the ceiling on system performance" assumption to logical-positivist motivations for measuring agreement.
- Shows the assumption is widespread.
- Presents data suggesting IAA is **not** in fact an upper bound on NLP system performance.

**Relation to our paper.** Cite it when rejecting "IAA = ceiling". With a majority-vote target, a model can exceed pairwise human agreement (compare the alt-test's Theorem 1: the majority vote is the optimal judge).

---

### Krippendorff sources (thresholds)

#### krippendorff2011computing: Computing Krippendorff's Alpha-Reliability (UPenn ScholarlyCommons, asc_papers/43)

**Status:** FULL (10 pp.). Retrieved from the repository via the DSpace REST bitstream link. The legacy URL repository.upenn.edu/asc_papers/43 now redirects to an item page. Dated 2011.1.25.

**Content.**
- Defines α = 1 − D_o/D_e.
- Gives worked computations for binary, nominal, ordinal, interval and other metrics, including multiple coders and missing data.
- α = 1 means perfect reliability; α = 0 means absence of reliability (p. 1).

**It states no acceptability thresholds.** For those it points to the book, Ch. 11, pp. 211–256 (references, p. 10). So do not cite this note for 0.800 or 0.667.

#### krippendorff2004reliability: Reliability in Content Analysis (HCR 30(3):411–433)

**Status:** FULL, read in the author's manuscript posted on the UPenn repository. Page numbers are **manuscript** pages, not journal pages.

On manuscript p. 12, recommendation (iii):
- The acceptable level "must be chosen depending on the costs of drawing invalid conclusions" from the data.
- Where lives or war-and-peace decisions hang on the results, criteria must be far higher than for scholarly arguments.
- For scholarly use he "suggested elsewhere" requiring α ≥ .800, and α ≥ .667 where tentative conclusions are still acceptable, citing **Krippendorff 2004 [book], p. 241**.
- He adds that, except for perfect agreement, there are "no magical numbers", and that the suggested values should be checked by experiments.
- Footnote 14: these standards were derived for α only and other coefficients may need different standards.
- Same page: confidence intervals should be consulted; testing against chance "has no bearing on reliability".

Manuscript p. 13, (iv): with several variables, take the **smallest** variable's reliability as the reliability of the whole system. Do not average.

#### krippendorff2004content: Content Analysis: An Introduction to Its Methodology (2nd ed., Sage, 2004)

**Status:** METADATA-ONLY. Edition confirmed via Open Library (2nd ed., Sage, Thousand Oaks, 2004, 413 pp.). **I could not access the book text.**

- The page number for the .800 / .667 guidance (**p. 241**) is Krippendorff's *own* citation in his 2004 HCR article. I did not see p. 241 myself.
- Caban 2026 cites the same book for α ≥ 0.67 as "pp. 241–243".
- The book's exact wording is not verified. Cite the HCR article for the wording and the book for the "p. 241" attribution, and flag it as a secondary pointer.
- I did not check later editions (3rd 2013, 4th 2019); their page numbers will differ.

---

### lam2003evaluating: Evaluating Classifiers by Means of Test Data with Noisy Labels (IJCAI 2003)

**Status:** FULL (6 pp., printed pp. 513–518). Equations are embedded as images and were read from them directly.

**Summary.**
- Setting: binary classes and a labeler with a known mislabeling rate η, independent of the classifier.
- Result (Eq. 1–2, p. 514): true error = (apparent error − η) / (1 − 2η).
- Table 1 (p. 515): even 1% label noise inflates a 6% true error to an apparent 6.88%, a 15% relative increase (text, p. 514).
- Rule of thumb for small η: the (1 − 2η) denominator is approximately 1, so η is approximately additive (p. 514).
- §4, Eq. 3 (p. 515): the ratio of noisy to perfect labels needed for the same variance. When η is below the true error rate, one perfect label is worth fewer than four noisy ones (Fig. 1 caption).
- It also gives upper and lower bounds on true error when labeler and classifier are dependent (abstract; §6 not read in detail).

**Relation to our paper.** This is the primary source of the (1 − 2η) correction. Cite it for (i).

---

### dorner2024dont: Don't Label Twice: Quantity Beats Quality when Comparing Binary Classifiers on a Budget (ICML 2024)

**Status:** FULL (main text; appendices not read). Read in arXiv v3 HTML.

See §0 for the core claims. Additional details:
- The gap indicator G ∈ {1, −1, 0} gives (1/n)ΣG_i = Acc_test(c_b) − Acc_test(c_w) (§2).
- The Hoeffding comparison needs √m > (2M_m(q) − 1)/(2q − 1). This always holds for m = 3 because the right side is at most 1.5 (§3.1).
- With a union bound over k classifiers, the Cramér-based bound certifies exponentially more testable classifiers. Example: budget 1500, margin 0.1, accuracy 0.75: more than 17 models with single labels, while Hoeffding cannot guarantee even 2 (§3.3, Fig. 1).
- They still recommend multiple labels on a small pilot sample to fix guidelines (§5).
- An online calculator is linked from §1.

**Relation.** This is prior art for our m = 1 claim and for the (2q − 1) attenuation. Our contribution must be positioned relative to it: for example LLM-scale benchmarks, multiclass or plurality labels, pairing with reliability estimation, gating.

---

## 2. Other papers (abstract-level unless noted)

### Agreement, reliability and statistics

- **artstein2008survey** (CL 34(4):555–596). ABSTRACT-ONLY (Crossref).
  - Survey of agreement coefficients (α, π, κ), their mathematics and assumptions.
  - Argues weighted α-like coefficients may suit many CL tasks, while noting that interpreting the values gets harder.
  - Relation: standard reference for IAA in NLP.
- **gwet2008computing** (BJMSP 61(1):29–48; DOI 10.1348/000711006X126600 verified). ABSTRACT-ONLY.
  - Explains the kappa paradoxes and introduces AC1, a more stable coefficient.
  - Gives variance estimators for multi-rater π and AC1 without assuming rater independence.
  - Relation: cite for the prevalence-robust coefficient. `li2026llm` shows AC1's variance amplification is bounded.
- **feinstein1990high** (J Clin Epidemiol 43(6):543–549; DOI verified). METADATA-ONLY. Classic citation for "high agreement but low kappa" under skewed prevalence.
- **james2026counting** (LREC 2026, pp. 4434–4446; arXiv 2603.06865). ABSTRACT-ONLY (arXiv).
  - Guide to choosing IAA measures by task type.
  - Covers label imbalance and missing data.
  - Recommends confidence intervals and disagreement analysis.
- **spearman1904proof** (Am J Psychol 15(1); starts p. 72; DOI verified). METADATA-ONLY.
  - Usual citation for the correction for attenuation.
  - The end page is not in Crossref and is not verified, so I left it out.
  - I did not read the text. Cite for attenuation only by convention.
- **spearman1910correlation** (Brit J Psychol 3(3):271–295) and **brown1910some** (Brit J Psychol 3(3):296–322). METADATA-ONLY. These are the joint origin of the Spearman–Brown prophecy formula. Text not read.
- **dror2018hitchhikers** (ACL 2018, pp. 1383–1392). ABSTRACT-ONLY.
  - Protocol for choosing significance tests in NLP.
  - Surveys ACL and TACL 2017 papers and finds testing often ignored or misused.
  - The alt-test cites it for the paired t-test.
- **card2020little** (EMNLP 2020, pp. 9263–9274). ABSTRACT-ONLY.
  - Underpowered experiments are common in NLP, and many GLUE comparisons are underpowered.
  - Typical human-rating designs are underpowered for small differences.
  - MT test sets of 2000 sentences have about 75% power for a 1 BLEU difference.
- **miller2024adding** (arXiv 2411.00640). ABSTRACT-ONLY.
  - Treats evals as experiments on a super-population of questions.
  - Gives formulas for CIs, paired differences and experiment planning.
  - No peer-reviewed version found in the arXiv record.
- **riley2024finding** (NAACL 2024, pp. 4908–4919). ABSTRACT-ONLY.
  - Uses MQM machine-translation human evaluation to study item-to-rater allocation, ratings per item and normalisation for stable system rankings, measured as "stable ranking probability".
  - Releases about 140k segment annotations.
  - Relation: a direct empirical study of ratings per item for ranking stability, in a regime different from ours.
- **pandita2026forest** (AAAI 2026, 40(29):24736–24744; DOI verified). ABSTRACT-ONLY (Crossref). Found via `pandita2026improving`.
  - Categorical datasets: the minimal N × K is ≤ 1000 for at least one metric on every dataset tested.
  - The minimum "almost always occurred for K > 10".
  - Distribution-sensitive metrics do better at high K; the trade-off depends on the metric.
  - Relation: the strongest "more K" claim for categorical data. The Wins/accuracy caveat from Homan et al. likely applies again, but I only read the abstract, so we must read the full paper before contrasting in detail.

### Disagreement and perspectivism

- **paun2018comparing** (TACL 6:571–585). ABSTRACT-ONLY. Compares six Bayesian annotation models against majority vote and agreement coefficients, with guidelines.
- **plank2022problem** (EMNLP 2022, pp. 10671–10682). ABSTRACT-ONLY. Position paper: human label variation is signal, not noise, and affects data, modelling and evaluation.
- **davani2022dealing** (TACL 10:92–110). ABSTRACT-ONLY. Multi-annotator multitask models match or beat majority-label training on 7 binary subjective tasks, and give uncertainty estimates.
- **basile2021need** (BPPF 2021, pp. 15–21). ABSTRACT-ONLY. Argues evaluation must account for disagreement from annotator, data and context sources.
- **uma2021learning** (JAIR 72:1385–1470). ABSTRACT-ONLY (JAIR page).
  - Survey and systematic comparison of learning from disagreement.
  - Even without a gold standard, a consensus on how to evaluate is needed, because method rankings depend on the evaluation form.
- **gordon2021disagreement** (CHI 2021, pp. 1–14; DOI verified). ABSTRACT checked 2026-10-01 via Semantic Scholar: "compares each test set prediction to the individual stable opinions from each annotator".
  - I did not read the abstract or text here.
  - It is known to adjust metrics for annotator disagreement; read it before claiming anything about its method.
- **klie2024analyzing** (CL 50(3):817–866). ABSTRACT-ONLY.
  - Annotates 591 dataset papers for quality management.
  - Most are good or excellent, but 30% are subpar.
  - Common errors in using IAA and computing error rates.
- **kunilovskaya2026who** (arXiv 2606.02255; comment says EMNLP 2026 Main camera-ready). ABSTRACT-ONLY.
  - Audit of 1,603 ACL-venue papers (2018–2025) and 2,667 annotation tasks.
  - Agreement values, training, compensation and similar details are often missing, "especially in model-evaluation studies".
  - Their LLM extractor reaches α 0.606 versus 0.585 human–human on a 41-paper gold set.
  - Relation: motivates reliability reporting and gating.

### Datasets

- **sachdeva2022measuring** (NLPerspectives @ LREC 2022, pp. 83–94). ABSTRACT-ONLY.
  - This is the peer-reviewed Measuring Hate Speech corpus paper.
  - 50,070 comments from YouTube, Reddit and Twitter; 11,143 MTurk annotators.
  - 10 ordinal labels plus a 3-valued hate speech label.
  - Faceted Rasch scaling gives a continuous score and annotator strictness.
  - A linked design across annotators.
- **kennedy2020measuring** (arXiv 2009.10277). ABSTRACT-ONLY. The earlier preprint with the same 50,070 / 11,143 figures, faceted Rasch IRT and RoBERTa multitask model. Prefer `sachdeva2022measuring` when citing the corpus.
- **mathew2021hatexplain** (AAAI 35(17):14867–14875). ABSTRACT-ONLY.
  - Posts labelled hate, offensive or normal, with target community and rationales.
  - Rationale-trained models reduce unintended bias.
- **aroyo2023dices** (NeurIPS 2023 D&B, 36:53330–53342). ABSTRACT-ONLY. Verified as a Datasets and Benchmarks track paper on proceedings.neurips.cc.
  - Conversational-AI safety ratings with rater demographics and high replication per item.
  - `pandita2026improving` reports DICES-350 has 123 raters on every item.
- **wulczyn2017ex** (WWW 2017, pp. 1391–1399). ABSTRACT-ONLY (arXiv 1610.08914 abstract).
  - Wikipedia personal-attack corpus: over 100k human-labelled comments and 63M machine-labelled ones.
  - The classifier is "as good as the aggregate of 3 crowd-workers" (AUC and Spearman).
  - Relation: an early example of rater-equivalence-style reporting.
- **demszky2020goemotions** (ACL 2020, pp. 4040–4054). ABSTRACT-ONLY. 58k Reddit comments, 27 emotions plus Neutral; BERT average F1 0.46.
- **northcutt2021pervasive** (NeurIPS 2021 D&B, vol. 1). ABSTRACT-ONLY.
  - At least 3.3% average test-label errors across 10 datasets; at least 6% on the ImageNet validation set.
  - Rankings flip: ResNet-18 beats ResNet-50 once the share of mislabelled test items rises by 6%.
- **nahum2025llms** (EMNLP 2025, pp. 26782–26809). ABSTRACT-ONLY.
  - LLM-ensemble detection of label errors in TRUE and SummEval.
  - Correcting them "induce[s] a significant upward shift in reported model performance".
  - Relation: empirical cousin of the (1 − 2η) correction.

### LLM-as-judge and judge validation

- **zheng2023judging** (NeurIPS 2023 **Datasets and Benchmarks** track, 36:46595–46623). Venue verified: listed under datasets_and_benchmarks on proceedings.neurips.cc. ABSTRACT-ONLY.
  - MT-Bench and Chatbot Arena.
  - GPT-4 judges reach "over 80% agreement", the same level as agreement between humans.
- **chiang2023large** (ACL 2023, pp. 15607–15631). ABSTRACT-ONLY. LLM evaluation agrees with expert human evaluation on story generation and adversarial attacks, and is stable across instruction formats.
- **bavaresco2025llms** (ACL 2025 **short**, 2025.acl-short.20, pp. 238–255). ABSTRACT-ONLY.
  - JUDGE-BENCH: 20 datasets and 11 LLMs.
  - Large variance across models and datasets.
  - LLMs "should be carefully validated against human judgments".
- **bowman2021will** (NAACL 2021, pp. 4843–4855). ABSTRACT-ONLY.
  - Four criteria for NLU benchmarks.
  - Fixing benchmarking needs, among other things, progress in annotation reliability and dataset size.

### Using model predictions to cut label cost

- **angelopoulos2023prediction** (Science 382(6671):669–674). ABSTRACT-ONLY. Prediction-powered inference gives valid CIs by combining a small labelled set with ML predictions.
- **boyeau2025autoeval** (ICML 2025, PMLR 267:5276–5290). ABSTRACT-ONLY.
  - The ICML version adds author **Tianle Li**, who is not on arXiv 2403.07008.
  - Unbiased autoevaluation with synthetic labels; up to 50% larger effective human sample size with GPT-4 (arXiv abstract).
- **dorner2025limits** (ICLR 2025, pp. 26467–26491). ABSTRACT-ONLY.
  - If the judge is no more accurate than the evaluated model, no debiasing method can cut the number of ground-truth labels needed by more than half.
  - Practical savings are smaller still.

### Label noise and repeated labelling

- **sheng2008get** (KDD 2008, pp. 614–622; DOI verified). METADATA-ONLY here.
  - `dorner2024dont` summarises it (§1.1): repeated labelling helps some decision-tree learners when labels are very noisy.
  - That concerns *training*, not evaluation.

---

## 3. Items requested but not fully verified or not read

- **Krippendorff book**: edition verified, text not accessed. The "p. 241" is Krippendorff's self-citation, not seen by me.
- **Spearman 1904**: the end page could not be verified from Crossref, so it is omitted from the bib.
- **Gordon et al. 2021, Feinstein & Cicchetti 1990, Sheng et al. 2008, Spearman 1910, Brown 1910**: bibliographic data verified via DOI, content not read.
- **Boguslav & Cohen 2017**: abstract only; the full text was not open.
- **Resnick et al.**: no peer-reviewed version found.
- **Kunilovskaya et al.**: I did not look up the EMNLP 2026 Anthology entry.

## Added 2026-10-04 (missed by the first search)

- **duan2025exploring** (NAACL 2025 long, 2025.naacl-long.119, pp. 2359-2372; DOI 10.18653/v1/2025.naacl-long.119 from the Anthology bib). FULL TEXT READ. Crowdsourced toxicity on 120 comments; perspective-taking annotations (estimate a subgroup's opinion) vs. direct polling of the subgroup; lower variance, higher bias; best under a limited budget when calibrated with a few direct annotations.
- **amin2026fallback** (Findings of ACL 2026, 2026.findings-acl.2124, pp. 42798-42830; DOI 10.18653/v1/2026.findings-acl.2124 from the Anthology bib). ABSTRACT, INTRODUCTION, SECTION 3, CONCLUSION AND LIMITATIONS READ. Frames perspective-taking as estimating a latent group-level judgement; bias-variance-correlation analysis of when LLMs beat human annotators, including in-group ones; uses the Duan et al. data and DICES-350 (DICES for regime behaviour, not direct human-LLM comparison). Overlap with our Section on judges: same corpus (DICES-350) and the same question of whether an LLM can beat a human annotator, but their target is a subgroup mean and ours is agreement with a panel majority, with a held-out human scored on the same items.
