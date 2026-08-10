data:  ## download the HateXplain corpus (12MB, not vendored)
	./scripts/fetch_data.sh

# Convenience targets. Everything here is a thin wrapper over the CLI or pytest;
# nothing in the pipeline requires make.

DATA    ?= data/hatexplain.json
RESULTS ?= results

.PHONY: install run report test lint clean all

all: install run test

install:
	pip install -e ../rubricon
	pip install -e .

run:
	rubricon-field run --data $(DATA) --out $(RESULTS)

report:
	rubricon-field report --results $(RESULTS)

test:
	pytest -q

lint:
	python -m pyflakes src tests

# Results are generated artefacts, but they are committed on purpose: the README
# asserts that every number it quotes is traceable to them, and a test enforces
# that. Use this only when regenerating from scratch.
clean:
	rm -rf $(RESULTS)
