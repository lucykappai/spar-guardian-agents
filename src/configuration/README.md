# src/configuration

Every setting of the experiment and every piece of model-facing text we add to CheatBench and ORBIT lives here. Code in `src/benchmark/` and `src/` reads these files and holds no texts of its own.

**`settings.py`** holds the fixed values: the repository paths, the default agent and judge models, the swarm compositions of the October 2026 sweep, the browser's port, the model timeout, the verdict fields the command line prints, the failure labels of an incomplete run, the board's channel id and the submodule commits the code was written against. It also loads the YAML files below.

**`openmath.py`** and **`swe.py`** hold each task domain's fixed settings: its CheatBench paths, the sandbox (work directory, user, CPU and memory, tool and grading timeouts, diff sizes), the per-agent token limit, and for SWE Lucy's honeypot names, her split parts and the jigsaw assignment. The domain modules under `src/benchmark/domains/` import these and hold only code.

**`swarm_config.py`** holds `SwarmConfig`, the frozen model of one run, and the presets `ours` and `lucy`. Each field's docstring says what it does, which ORBIT mechanism implements it and how it differs from Lucy's patch. The helpers at the bottom pick prompt texts that depend on a setting.

**`prompts.yaml`** holds the board system prompt, the team report instruction, Lucy's planning prompts, the turn header, the continue prompt, the unread notice's replacement line, the SWE system prompts and assignment, and Lucy's end-of-episode questions. Each borrowed text names its source.

**`instruction_levels.yaml`** holds three ladders of model-facing text: the rule every agent is told, the regulator's text and the disregulator's text. A config selects one level per ladder. Each benchmark fills the placeholders from its own `vars`.

**`controls.yaml`** holds CheatBench's openmath rule as criteria for an LLM monitor or guardian, and as two lists of regex patterns for ORBIT's pattern monitor.
