# Entry points. Run from the repository root; everything uses .venv.
PY := .venv/bin/python
SWARM := R0D0,R10D0,R0D10,R1D1,R1D2,R2D2,R3D3,R2D5,R5D2,R0D3,R3D0

.PHONY: help ui ui-stop smoke check

help:  ## list the targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/' | column -t -s "$$(printf '\t')"

ui:  ## open the results browser (starts the server when needed)
	@scripts/ui

ui-stop:  ## stop the results browser started by make ui
	@scripts/ui stop

smoke:  ## six mock-model scenarios on the real sandbox, no model spend
	$(PY) -m guardian.checks.smoke_test

check:  ## refuse unless the eleven swarm runs are complete and judged
	$(PY) -m guardian.checks.check_sweep --expect $(SWARM)
