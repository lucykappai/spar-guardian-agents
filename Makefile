# Entry points. Run from the repository root; everything uses .venv.
# Runs and judging go through the command line: .venv/bin/python -m swarm_contagion --help
PY := .venv/bin/python

.PHONY: help ui ui-stop smoke smoke-swe check test

help:  ## list the targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/' | column -t -s "$$(printf '\t')"

ui:  ## open the results browser (starts the server when needed)
	@$(PY) -m swarm_contagion ui

ui-stop:  ## stop the results browser started by make ui
	@$(PY) -m swarm_contagion ui stop

smoke:  ## openmath mock-model scenarios on the real sandbox, no model spend
	$(PY) tests/smoke.py

smoke-swe:  ## SWE mock-model scenarios on the flipt-02e2 sandbox, no model spend
	$(PY) tests/smoke_swe.py

check:  ## refuse unless the eleven swarm runs are complete and judged
	$(PY) tests/check_runs.py

test:  ## offline unit tests of the turn rules and the config (mock model, no Docker)
	$(PY) -m pytest tests/test_rules.py tests/test_config.py -q
