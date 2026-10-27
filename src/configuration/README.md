# src/configuration

Every setting of the experiment and every piece of model-facing text we add to CheatBench and ORBIT lives here. Code in `src/benchmark/` and `src/` reads these files and holds no texts of its own.

**`settings.py`** holds the fixed values: the repository paths, the default agent and judge models, the swarm compositions of the October 2026 sweep, the browser's port, the model timeout, the verdict fields the command line prints, the failure labels of an incomplete run, the board's channel id and the submodule commits the code was written against. It also loads the YAML files below.

**`openmath.py`** and **`swe.py`** hold each task domain's fixed settings: its CheatBench paths, the sandbox (work directory, user, CPU and memory, tool and grading timeouts, diff sizes), the per-agent token limit, and for SWE Lucy's honeypot names, her split parts and the jigsaw assignment. The domain packages under `src/benchmark/domains/` import these and hold only code.

**`swarm_config.py`** holds `SwarmConfig`, the frozen model of one run, and three presets. `default` is the pull board at the small scale in `shared_task`, with the board notice's counts. `ian_legacy` and `lucy_legacy` are the presets `ours` and `lucy` of commit 6769c80, with the same behaviour apart from the removed team report. The field `team_report`, on in the legacy presets only, keeps the old report's sentences in the role texts. The field `preset` records the preset's name; every preset and `build_config` set it, and `run.json` reports it. Each field's docstring says what it does, which ORBIT mechanism implements it and how it differs from Lucy's patch. `build_config(preset, scale, pairs)` applies the preset, then the scale, then every `key=value` pair, as the command line does. `resolve_mode(domain)` gives the mode of a run: `mode` when set, else `shared_task` on both domains. The helpers at the bottom pick prompt texts that depend on a setting.

**`scales.py`** holds the five scales, from `small` (three agents, eight steps) to `gigantic` (a thousand agents). A scale sets the team size, the steps, the post size, the read page, the free board actions, the time limit, the concurrency and the agents' batch calls, and derives the context bound and the message limit. `SwarmConfig.at_scale(name)` applies one. A later `--set` overrides any value a scale gave, and does not recompute the derived ones.

**`prompts.yaml`** holds the team sentence, the board sentence, the legacy board system prompt, the role briefing, the board tools' descriptions and outputs, the board notice, Lucy's planning prompts and their pull-board version, the turn header, the continue prompt, the unread notice's replacement line, the SWE system prompts and assignment, and Lucy's end-of-episode questions. Each borrowed text names its source.

**`instruction_levels.yaml`** holds three ladders of model-facing text: the rule every agent is told, the regulator's text and the disregulator's text. A config selects one level per ladder. Each benchmark fills the placeholders from its own `vars`. A level whose text named the old team report keeps that text as `text_legacy`, which a config with `team_report` renders.

**`controls.yaml`** holds CheatBench's openmath rule as criteria for an LLM monitor or guardian, and as two lists of regex patterns for ORBIT's pattern monitor.
