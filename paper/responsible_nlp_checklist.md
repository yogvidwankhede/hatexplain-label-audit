# ARR Responsible NLP checklist: draft answers

Questions fetched from aclrollingreview.org/responsibleNLPresearch on 2026-09-30. The author must re-check these answers against the final PDF before submitting. Wrong or misleading answers can lead to desk rejection. Section references are to `main.tex`.

| Item | Answer | Where / justification |
|---|---|---|
| **A1** Limitations | Yes | "Limitations" section. It covers noise-model assumptions, simulation scope, post-hoc scenarios, the retrospective, corpus coverage, the judges, and the applicability of the alt-test. |
| **A2** Risks | Yes | "Ethics statement". It covers the offensive content, exposure of text to API providers, and the risk that "cannot resolve" is misread as "wrong". |
| **B1** Cite artefact creators | Yes | §Data cites every corpus. Rubricon, the ACL style files and the models are listed with versions in the appendix and the repository. |
| **B2** Licences | Yes | Table 1 (licence per corpus). The GoEmotions licence ambiguity is stated. Code is released under MIT. |
| **B3** Consistent with intended use | Yes | §Data and Ethics: research use under each licence, with no text redistributed. DICES is used for rater-agreement analysis, which its authors intend. |
| **B4** PII / offensive content | Yes | Ethics: we use the corpora's pseudonymous rater IDs, never read GoEmotions user names into results, and HateXplain user names are already masked as `<user>`. The content is offensive by design, and we quote none of it. |
| **B5** Documentation of artefacts | Yes | §Data and Table 1 give domain, language (English) and label schemes. |
| **B6** Statistics and splits | Yes | Table 1, §§5 and 7 (HateXplain official split, n = 1,924), and the judge sample sizes in the appendix. |
| **C1** Parameters, compute, infrastructure | Yes | Appendix "Judge experiment details": model identifiers; local models run on one Apple M4 Max (qwen2.5 14B, gpt-oss 20B); API spend; and the simulation compute (CPU only, minutes). |
| **C2** Experimental setup / hyperparameters | Yes | No models were trained. Judge settings (temperature, max tokens, thinking), prompts and the parser were fixed in PREREG_ADDENDUM before the calls. The pilot is described. |
| **C3** Descriptive statistics / error bars | Yes | 95% item-cluster bootstrap intervals throughout. The simulations report rates over 1,000 replicates per cell. Every judge result comes from a single run per item, and we say so. |
| **C4** Packages and settings | Yes | Rubricon 0.5.0 with a pinned commit; `requirements.lock` is hash-pinned; the SDK versions are in the lockfile. |
| **D1–D5** Human annotators | N/A | We collected no new annotations. We re-analyse existing per-annotator labels under their licences. The original papers describe annotator recruitment, pay and demographics, and we cite them. |
| **E1** AI assistants | Yes | Acknowledgements, "Use of AI assistants". It states that an AI coding assistant was used for code, experiments, literature search and verification, and drafting; that the author reviewed everything; and that LLMs as objects of study are separate (§9, Appendix). |
