# PREREG_ADDENDUM.md — LLM-judge experiment (J1), fixed before the first model call

Date: 2026-09-30. Written and committed before any judge output (pilot or sample) existed.

## Judges (author decisions 2026-09-29/30)
| id | provider | model string sent | settings | role |
|---|---|---|---|---|
| claude-haiku-4-5 | Anthropic Message Batches API | claude-haiku-4-5-20251001 (from the Models API list) | temperature 0, max_tokens 16 | primary |
| claude-sonnet-5-5 | Anthropic Message Batches API | claude-sonnet-5-5 | thinking {type: between_tools} (thinking off), effort low, max_tokens 64; this model rejects non-default temperature, so sampling is the default (deviation from PREREG "temperature 0", decided before any call) | primary |
| gpt-4.1-mini | OpenAI Chat Completions | gpt-4.1-mini (returned snapshot id is logged per call) | temperature 0, seed 0, max_tokens 16 | secondary |
| qwen2.5-14b | local Ollama | qwen2.5:14b (digest 7cdf5a0187d5) | temperature 0, seed 0 | primary (local) |
| gpt-oss-20b | local Ollama | gpt-oss:20b (digest logged when pulled) | think "low", temperature 0, seed 0 | primary (local) |

Dropped by author decision: Google AI Studio free tier (its terms say submitted content is used to improve Google products and may be read by human reviewers, and ask users not to submit personal information); Groq free tier (200K tokens/day would take >8 days; replaced by local gpt-oss-20b). No server-side model fallback is enabled for Claude, so refusals stay refusals.

Price sources (fetched 2026-09-30): Anthropic model table in the Claude API skill (Haiku 4.5 $1/$5, Sonnet 5.5 $2/$10 per MTok; Batch = 50%); OpenAI pricing page developers.openai.com/api/docs/pricing (gpt-4.1-mini $0.40/$1.60). Caps: Anthropic $8, OpenAI $4; projected upper bounds $0.54 + $1.88 and $0.41.

## Prompts (system prompt = file; user message = item text; no few-shot examples)
    69c09c0404e16a00ee486c5ebd73aea9c21d9b09be04b00413a8476bc4c2c074  prompts/dices.txt
    110f28c0e9ed6767a39cd20f7be2591584e0b3270707b25a5c3071226511d58f  prompts/hatexplain.txt
    3c63b262de04e2c6e5937e3edb016924a96010fdcb3c67cc22da1c77236444d8  prompts/mhs.txt
    0ae95bfa3ce7da59c019861b0a94d9bc7e61efe860bed119ce11995cbec8a37a  prompts/wikitalk.txt
The label definitions paraphrase each corpus's task; they are not the original annotator guidelines, which the paper states.

## Samples (judge_sample.py; seeds derived from SEED_PAPER = 20260929)
    hatexplain: n=1000 of 20148 eligible; text sha256 2f395379abd9114856138c40fd98f90ccb2a9d33286df38a17354d29e845d038
    mhs: n=1000 of 17352 eligible; text sha256 b0f5105774b00e111232d10fbfaa4ac0572c93c9dd55d1e6b706cc9814f9635b
    wikitalk: n=1000 of 159686 eligible; text sha256 5f277b6a5927b539dc7c52810e9e0e00636236cdee86cf5d994efba22a754d6b
    dices350: n=350 of 350 eligible; text sha256 e58ed1e7939d86caef771d592745c7bc0c4a5d276c8bb7dc0c4cae320f23cf43
Items need >= 3 ratings in the capped matrix. One held-out rater per item (seeded). Texts > 6,000 characters would be cut with a marker; none were.

## Parsing rule (judges.parse)
Strict: the whole answer, lower-cased with punctuation removed, equals one label. Lenient: exactly one distinct label word occurs. Otherwise unparseable. Refusals (Anthropic stop_reason "refusal", OpenAI "refusal" field) and unparseable outputs are their own categories.

## Pilot
5 items per corpus drawn from eligible items outside the sample (20 per judge), used only to check that outputs parse. If a prompt must change after the pilot, the change and the reason go in DEVIATIONS.md, and no evaluation-sample output will have been seen.

## Analysis (fixed now)
- Panel = the item's ratings minus the held-out rater. Binary panel label = strict majority of the binarised panel ratings (items without one are excluded and counted).
- **M1 (primary):** agreement of each judge's binary label with the binary panel label; the held-out human is scored identically on the same items. Per corpus: judge, human, and paired difference (judge - human) with 95% item-cluster bootstrap CIs (2,000 replicates).
- **M2:** mean agreement with individual panel raters (binary), all items.
- **M3:** native-scale exact match with the panel plurality (HateXplain 3-way, MHS 3-way, DICES 3-way).
- Refusals/unparseable: excluded in the primary analysis (counts reported); sensitivity analysis scores them as wrong.
- Alternative Annotator Test (Calderon et al., ACL 2025): implemented following the paper; its epsilon, test, FDR level and minimum items per annotator are copied from the paper into this file before the alt-test analysis is run.
- Gate v2 (rubricon.gates.attenuation) is applied to each claim "judge X agrees with the panel at least as well as the held-out human" and "judge X beats judge Y", with z* = 2.58.
- DICES-350 expert label (safety_gold): secondary reference for all judges and the human.
