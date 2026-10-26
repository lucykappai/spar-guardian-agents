# Entry points. Run from the repository root; everything uses .venv.
# Runs and judging go through the command line: .venv/bin/python -m guardian --help
PY := .venv/bin/python

.PHONY: help ui ui-stop smoke check

help:  ## list the targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/' | column -t -s "$$(printf '\t')"

ui:  ## open the results browser (starts the server when needed)
	@$(PY) -m guardian ui

ui-stop:  ## stop the results browser started by make ui
	@$(PY) -m guardian ui stop

smoke:  ## six mock-model scenarios on the real sandbox, no model spend
	$(PY) tests/smoke.py

check:  ## refuse unless the eleven swarm runs are complete and judged
	$(PY) tests/check_runs.py
