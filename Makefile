# Entry points. Run from the repository root; everything uses .venv.
PY := .venv/bin/python
SWARM := R0D0,R10D0,R0D10,R1D1,R1D2,R2D2,R3D3,R2D5,R5D2,R0D3,R3D0
CTRL := $(addprefix runs/,ctrl_vaccination ctrl_secure_model ctrl_tool_wrapper ctrl_dual_llm ctrl_monitor_stock ctrl_guardian_stock ctrl_monitor_task ctrl_guardian_task ctrl_regex_task ctrl_regex_task_hard ctrlb_monitor_stock ctrlb_guardian_stock ctrlb_monitor_task ctrlb_guardian_task)

.PHONY: help ui ui-stop share share-browser smoke check tables figures doc

help:  ## list the targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/' | column -t -s "$$(printf '\t')"

ui:  ## open the results browser (starts the server when needed)
	@scripts/ui

ui-stop:  ## stop the results browser started by make ui
	@scripts/ui stop

share:  ## export the results browser as a static bundle (guardian/build/share) and check it
	$(PY) -m guardian.ui.export
	$(PY) -m guardian.checks.check_share

share-browser:  ## headless pass over the exported bundle (needs node and Playwright with Chromium)
	@if lsof -nP -iTCP:8798 -sTCP:LISTEN -t >/dev/null 2>&1; then echo "port 8798 is busy; stop that server first" >&2; exit 1; fi; \
	python3 -m http.server 8798 --bind 127.0.0.1 --directory guardian/build/share >/dev/null 2>&1 & echo $$! > guardian/build/share-http.pid; sleep 1; \
	NODE_PATH=$${NODE_PATH:-/opt/homebrew/lib/node_modules} node guardian/checks/share_browser.js http://127.0.0.1:8798/; rc=$$?; \
	kill $$(cat guardian/build/share-http.pid); rm -f guardian/build/share-http.pid; exit $$rc

smoke:  ## six mock-model scenarios on the real sandbox, no model spend
	$(PY) -m guardian.checks.smoke_test

check:  ## refuse unless the eleven swarm runs are complete and judged
	$(PY) -m guardian.checks.check_sweep --expect $(SWARM)

tables:  ## regenerate docs/generated/*.tex from the run directories
	$(PY) -m guardian.report.report_pilot runs/gemini_riemann runs/gemini_board
	$(PY) -m guardian.report.report_pilot $(CTRL) --macro ctrl --out docs/generated/controls_results.tex
	$(PY) -m guardian.report.report_swarm

figures:  ## regenerate docs/figures from the run directories
	$(PY) -m guardian.report.figures --runs runs --out docs/figures --swimlane gemini_board
	$(PY) -m guardian.report.figures --runs runs --out /tmp/figures_swarm --swimlane swarm_R3D3
	cp /tmp/figures_swarm/fig_swimlane.pdf docs/figures/fig_swimlane_swarm.pdf
	cp /tmp/figures_swarm/fig_swimlane.png docs/figures/fig_swimlane_swarm.png

doc: tables  ## build docs/cheatbench_orbit_integration.pdf
	cd docs && latexmk -pdf -interaction=nonstopmode cheatbench_orbit_integration.tex
