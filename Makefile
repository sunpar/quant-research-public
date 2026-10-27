override TARGET := $(value TARGET)
override HEAD := $(value HEAD)
override BASE := $(value BASE)
override TITLE_FILE := $(value TITLE_FILE)
override BODY_FILE := $(value BODY_FILE)
override REPOSITORY := $(value REPOSITORY)
override REMOTE := $(value REMOTE)
override ARGS := $(value ARGS)
override DESTINATION := $(value DESTINATION)
override PR := $(value PR)
export TARGET HEAD BASE TITLE_FILE BODY_FILE REPOSITORY REMOTE ARGS DESTINATION PR

RUN = common=$$(git rev-parse --path-format=absolute --git-common-dir) && gate="$$common/repo-gates/repo.py"; test -f "$$gate" || { echo 'Run make hooks or make setup before repository commands' >&2; exit 1; }; uv run --no-project --no-config --offline --no-python-downloads --python 3.12.13 python -B -I -S "$$gate"

.PHONY: setup deps hooks smoke-index verify coverage pr merge run example worker-image

setup: hooks
	$(MAKE) deps

deps:
	$(RUN) deps

hooks:
	uv run --no-project --no-config --python 3.12.13 python -B -I -S tools/repo.py hooks

smoke-index:
	$(RUN) smoke-index

verify:
	$(RUN) verify

coverage:
	$(RUN) coverage

pr:
	$(RUN) pr

merge:
	$(RUN) merge

run:
	$(RUN) run

example:
	$(RUN) example

worker-image:
	docker build --provenance=false --build-arg SOURCE_DATE_EPOCH=0 --output type=image,name=quant-research-worker:2,rewrite-timestamp=true,unpack=false worker
