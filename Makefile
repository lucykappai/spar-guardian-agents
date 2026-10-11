# Entry points. Run from the repository root; everything uses .venv.
PY := .venv/bin/python
SWARM := R0D0,R10D0,R0D10,R1D1,R1D2,R2D2,R3D3,R2D5,R5D2,R0D3,R3D0
CTRL := $(addprefix runs/,ctrl_vaccination ctrl_secure_model ctrl_tool_wrapper ctrl_dual_llm ctrl_monitor_stock ctrl_guardian_stock ctrl_monitor_task ctrl_guardian_task ctrl_regex_task ctrl_regex_task_hard ctrlb_monitor_stock ctrlb_guardian_stock ctrlb_monitor_task ctrlb_guardian_task)

.PHONY: help ui ui-stop smoke check tables figures doc

help:  ## list the targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/' | column -t -s "$$(printf '\t')"

ui:  ## open the results browser (starts the server when needed)
	@scripts/ui

ui-stop:  ## stop the results browser started by make ui
	@scripts/ui stop

smoke:  ## four mock-model scenarios on the real sandbox, no model spend
	$(PY) -m integration.checks.smoke_test

check:  ## refuse unless the eleven swarm runs are complete and judged
	$(PY) -m integration.checks.check_sweep --expect $(SWARM)

tables:  ## regenerate docs/generated/*.tex from the run directories
	$(PY) -m integration.report.report_pilot runs/gemini_riemann runs/gemini_board
	$(PY) -m integration.report.report_pilot $(CTRL) --macro ctrl --out docs/generated/controls_results.tex
	$(PY) -m integration.report.report_swarm

figures:  ## regenerate docs/figures from the run directories
	$(PY) -m integration.report.figures --runs runs --out docs/figures --swimlane gemini_board
	$(PY) -m integration.report.figures --runs runs --out /tmp/figures_swarm --swimlane swarm_R3D3
	cp /tmp/figures_swarm/fig_swimlane.pdf docs/figures/fig_swimlane_swarm.pdf
	cp /tmp/figures_swarm/fig_swimlane.png docs/figures/fig_swimlane_swarm.png

doc: tables  ## build docs/cheatbench_orbit_integration.pdf
	cd docs && latexmk -pdf -interaction=nonstopmode cheatbench_orbit_integration.tex
