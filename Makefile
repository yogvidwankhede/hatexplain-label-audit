# Reproduction targets for the paper "What can your labels support?".
# `make all` regenerates every table, figure and number in the paper from the raw
# corpora, without calling any LLM API: judge outputs are committed under
# results/judge/sample/ and are re-scored, not re-queried. `make judges` re-queries
# the models (needs API keys in .env and local Ollama models; costs money).

PY      ?= python3
DATA    ?= data/hatexplain.json
RESULTS ?= results
CORPORA := hatexplain mhs dices350 dices990 wikitalk goemotions

.PHONY: all install data corpora study cross sims claim4 retro judge-samples judges \
        judge-analysis alt-test figures paper test lint help

help:  ## list targets
	@grep -E '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-16s %s\n",$$1,$$2}'

all: data corpora study cross sims claim4 retro judge-samples judge-analysis alt-test figures test paper  ## regenerate everything (no API calls)

install:  ## install pinned dependencies and both packages
	pip install --require-hashes -r requirements.lock
	pip install --no-deps -e ../rubricon -e .   # CI checks rubricon out at tag v0.5.0

data:  ## HateXplain (12 MB)
	test -s $(DATA) || ./scripts/fetch_data.sh

corpora:  ## the other per-annotator corpora (~250 MB) and the HateXplain split file
	./scripts/fetch_corpora.sh

study:  ## original single-corpus HateXplain studies A-C
	rubricon-field run --data $(DATA) --out $(RESULTS)

cross:  ## cross-corpus agreement battery, capped (primary) and all-ratings (sensitivity)
	for c in $(CORPORA); do $(PY) -m rubricon_field.cross_corpus $$c && \
	  $(PY) -m rubricon_field.cross_corpus --all-ratings $$c || exit 1; done

sims:  ## known-truth gate simulation (iid, prereg het, post-hoc het) and its summary
	$(PY) -m rubricon_field.gate_sim iid 1000
	$(PY) -m rubricon_field.gate_sim het 1000
	$(PY) -m rubricon_field.gate_sim_report

claim4:  ## real-rater budget allocation study
	$(PY) -m rubricon_field.claim4

retro:  ## HateXplain published-gap retrospective
	$(PY) -m rubricon_field.retro

judge-samples:  ## fixed judge item samples (texts go to data/judge/, ids to results/judge/)
	$(PY) -m rubricon_field.judge_sample

judges:  ## RE-QUERY all judges (costs money; needs .env and Ollama)
	for j in claude-haiku-4-5 claude-sonnet-5-5 gpt-4.1-mini qwen2.5-14b gpt-oss-20b; do \
	  $(PY) -m rubricon_field.judges $$j sample || exit 1; done

judge-analysis:  ## score committed judge outputs against the human panels
	$(PY) -m rubricon_field.judge_analysis

alt-test:  ## Alternative Annotator Test (Calderon et al. 2025)
	$(PY) -m rubricon_field.alt_test

figures:  ## every paper figure
	$(PY) scripts/make_figures.py

paper:  ## compile both paper versions with tectonic
	cd paper && tectonic -X compile main.tex && tectonic -X compile short.tex

test:  ## test suite
	pytest -q

lint:
	$(PY) -m pyflakes src tests scripts
