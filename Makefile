# Entry points. Run from the repository root; everything runs through uv, which picks up .venv.
# Runs and judging go through the command line: uv run python -m src --help
PY := uv run python

.PHONY: help ui ui-stop smoke smoke-swe smoke-kw check test lint

help:  ## list the targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/' | column -t -s "$$(printf '\t')"

ui:  ## open the results browser (starts the server when needed)
	@$(PY) -m src ui

ui-stop:  ## stop the results browser started by make ui
	@$(PY) -m src ui stop

smoke:  ## openmath mock-model scenarios on the real sandbox, no model spend
	$(PY) tests/smoke.py

smoke-swe:  ## SWE mock-model scenarios on the flipt-02e2 sandbox, no model spend
	$(PY) tests/smoke_swe.py

smoke-kw:  ## knowledge-work mock-model scenarios on the office sandbox, no model spend
	$(PY) tests/smoke_kw.py

check:  ## refuse unless the eleven swarm runs are complete and judged
	$(PY) tests/check_runs.py

test:  ## offline unit tests of the turn rules, the config, the access policy, the model registry and the examples
	$(PY) -m pytest tests/test_rules.py tests/test_config.py tests/test_access.py tests/test_models.py \
		tests/test_examples.py -q

lint:  ## ruff over src, tests and examples (ruff.toml: 120 columns, sorted imports)
	uvx ruff check src tests examples
