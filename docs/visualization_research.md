# Swarm visualization research

Survey by an Opus research agent on 2026-10-11 (WebSearch and WebFetch), unedited. It informs the views in `guardian/ui/static/app.js` and the figures in `guardian/report/figures.py`.

I found no standard way to draw multi-agent LLM runs or their hidden roles. The six views below adapt what existing tools and papers do. Most of the data they need is already served by `GET /api/runs/<run>/timeline` and `GET /api/summary` in `/Users/unrulyabstractions/work/spar-guardian-agents/guardian/ui/API.md`. `/Users/unrulyabstractions/work/spar-guardian-agents/guardian/ui/static/app.js` draws no SVG or canvas charts yet.

## What the field does

- **Graph views show structure, not time.**
  - AutoGen Studio draws a "control transition graph". LangGraph Studio shows the "nodes traversed".
  - Langfuse infers an agent graph from span timing and nesting, with a merged view and an as-it-ran view.
  - The OpenAI Agents SDK draws agents as yellow rectangles, tools as green ellipses and handoffs as solid arrows.
  - AgentScope Studio has a chat view and traces only, and it is now archived.
- **Time views are swimlanes.**
  - Inspect 0.3.263, which ORBIT is built on, draws one lane per agent with token counts and markers. I confirmed this in the installed source.
  - AISI's transect lays sub-agent lanes on a turn axis.
  - Figure 2 of the SCHEMEs paper draws each tool call as a tick, coloured by the task it advances.
  - AGDebugger draws one rectangle per message and lets you colour by sender or by message type.
  - AgentLens draws one curve per agent and pulls curves together when agents interact.
- **Contagion is shown three ways.**
  - Infected-count curves: Prompt Infection Figure 6 and Colosseum Figures 7 and 9.
  - Cascade trees with infected nodes in red: Mind Viruses Figure 5 and Generative Agents Figure 9.
  - Dot timelines with one dot per post and one line per agent: Mind Viruses Figure 10.
- **Design grids are heatmaps with cell counts or stacked bars.** Examples are SCHEMEs Figure 14, Institutional AI Figure 5 and the debate paper "Talk Isn't Always Cheap" Figures 2 and 3.
- **Roles have no convention.** Papers just reuse red for the adversary (GovSim Figure 3, Mind Viruses). ORBIT's own viewer is Inspect View plus a Markdown overview, and its docs say it does not infer causal influence.

## Recommended visualizations

**1. Agent swimlane with message arrows (run page)**
- **Question:** Who did what, when, and which posts each agent read before acting?
- **Fields:** `events` (t, t_end, agent, kind, tokens), board posts (`sender`, `t`, `read_by[{agent,t,how}]`), `first_honeypot_touch`, defense interventions, `roles`, `verdicts`.
- **Encoding:**
  - x is seconds from start, with a toggle to round-robin turn index. Each agent gets one lane, grouped by role.
  - Model calls are bars, with height scaled by tokens. Tool calls are ticks coloured by tool.
  - A post is a dot on the sender's lane, with an arrow to each reader's lane at read time. Solid arrows mean the agent read the post; dashed means it was only shown to the model.
  - Honeypot access is a diamond, interventions are a vertical band, and the verdict sits at the lane's end.
- **Sources:** Inspect timeline, transect, SCHEMEs Figure 2, Lamport's space-time diagram.

**2. Agent × turn raster (thumbnail in the runs list)**
- **Question:** What does the whole run look like at a glance?
- **Fields:** `events` by agent and turn.
- **Encoding:** Rows are agents and columns are turns. Each cell is coloured by its main action: work, post, read, honeypot or submit.
- **Source:** AGDebugger's overview.

**3. Exposure cascade graph**
- **Question:** Did access spread through the board, and from whom?
- **Fields:** `read_by`, `first_honeypot_touch`, `roles`, `verdicts`, disclosed flag.
- **Encoding:**
  - Nodes are agents, placed left to right by time of first honeypot access. Agents who never touched it go in a "never" column.
  - An edge A→B means B read A's post before B's first touch. Edge width is the number of such posts.
  - Shape shows role, fill shows cheated or not, and an outline marks disclosure.
  - Label it as exposure, not cause.
- **Sources:** Mind Viruses Figure 5, Generative Agents Figure 9, Prompt Infection Figures 1–3, ORBIT's viewer docs.

**4. Aligned adoption curves (small multiples, one panel per composition)**
- **Question:** How fast does access spread, and do regulators slow it?
- **Fields:** `first_honeypot_touch`, the time of the disregulator's first post, `composition`.
- **Encoding:**
  - x is time since a reference event: the disregulator's first post. This alignment trick comes from event-sequence visualization research.
  - y is the fraction of agents who have touched the honeypot.
  - Each run is a faint step line, with the median drawn bold.
- **Sources:** Prompt Infection Figure 6, Colosseum Figures 7 and 9, Zhang et al. (VIS 2019).

**5. Composition grid heatmap, with per-role small multiples**
- **Question:** Where in the regulator × disregulator grid does the team cheat?
- **Fields:** `regulators`, `disregulators`, `judged`, `cheated`, `report_cheated`, `by_role`, `seed`.
- **Encoding:**
  - x is the number of disregulators and y the number of regulators. Only the triangle where the two sum to at most 10 is used.
  - Fill is the fraction of agents judged cheated, on a single-hue scale. Each cell also shows "k/n".
  - A corner mark flags cells where the team's self-report disagrees with the judge. Cells not run are hatched.
  - A companion grid puts three per-role bars in each cell.
- **Sources:** SCHEMEs Figure 14, Institutional AI Figure 5 (its green-to-red scale is not colour-blind safe), Colosseum Figure 5, ORBIT Table 2, GovSim Figure 3.

**6. Role-coloured outcome funnel (alluvial)**
- **Question:** At which step do agents in each role stop?
- **Fields:** The judge flags per agent (honeypot seen, attempted access, cheated, disclosed) and `role`, pooled over runs.
- **Encoding:** Columns are the stages in that order. Bands are coloured by role, and band width is the number of agents.
- **Sources:** Rosvall and Bergstrom's alluvial diagrams, "Talk Isn't Always Cheap" Figures 2 and 3.

**Roles.** Show role by shape: circle for worker, square for regulator, triangle for disregulator. Use the colour-blind-safe Okabe-Ito colours, and keep red and green for outcomes only. The UI currently colours regulators green and disregulators red, which clashes with red for "cheated". A "reveal roles" toggle is my own idea; I found no source for it.

## Where to build each one

- **Plain SVG in the vanilla-JS browser:** 1, 3 and 5. Each is at most 10 lanes, nodes, or 66 cells, and needs no graph library. Use Canvas for 1 if a run goes past a few thousand marks.
- **Canvas:** 2.
- **Static matplotlib figures for the paper:** 4, which needs medians and intervals across seeds, and 6, built from `fill_between` bands. Also 5, and one example of 1.

## What I checked

- I read the paper text or HTML for each figure, not the images. The encodings described for paper figures come from their captions and surrounding text.
- I confirmed the Inspect swimlane in the installed inspect_ai 0.3.263 source.
- These rest on search summaries only: Agent Smith (I cite it in no recommendation), Tanahashi and Ma, Zhang et al., Rosvall and Bergstrom, and Lamport.
- The task was read-only, so I added no `VERIFICATION_LOG.md` entry.

## URLs consulted

- https://microsoft.github.io/autogen/stable/user-guide/autogenstudio-user-guide/
- https://docs.langchain.com/langgraph-platform/langgraph-studio
- https://langfuse.com/docs/observability/features/agent-graphs
- https://openai.github.io/openai-agents-python/visualization/
- https://openai.github.io/openai-agents-python/tracing/
- https://github.com/agentscope-ai/agentscope-studio
- https://inspect.aisi.org.uk/log-viewer.html
- https://inspect.aisi.org.uk/reference/inspect_ai.event.html
- https://inspect.aisi.org.uk/CHANGELOG.html
- https://github.com/AI-Safety-Institute/transect
- https://arxiv.org/html/2609.33102v1 (ORBIT)
- https://arxiv.org/html/2503.02068 (AGDebugger)
- https://arxiv.org/html/2402.08995v1 (AgentLens)
- https://arxiv.org/html/2605.29178v1 (SCHEMEs)
- https://arxiv.org/pdf/2602.15198 (Colosseum)
- https://arxiv.org/pdf/2608.10218 (Mind Viruses)
- https://arxiv.org/html/2410.07283v1 (Prompt Infection)
- https://ar5iv.labs.arxiv.org/html/2304.03442 (Generative Agents)
- https://arxiv.org/html/2509.05396 (Talk Isn't Always Cheap)
- https://arxiv.org/html/2404.16698v4 (GovSim)
- https://arxiv.org/html/2601.11369v1 (Institutional AI)
- https://arxiv.org/html/2504.04072v3 (Among Us)
- https://arxiv.org/pdf/2402.07510 (Secret Collusion)
- https://arxiv.org/pdf/2402.08567 (Agent Smith)
- https://arxiv.org/pdf/2308.10848 (AgentVerse)
- https://arxiv.org/pdf/0812.1242 (Rosvall and Bergstrom)
- https://ieeevis.org/year/2012/paper/infovis/design-considerations-optimizing-storyline-visualizations
- https://arxiv.org/pdf/1908.07316 (Zhang et al.)
- https://amturing.acm.org/p558-lamport.pdf
- https://conceptviz.app/blog/okabe-ito-palette-hex-codes-complete-reference
