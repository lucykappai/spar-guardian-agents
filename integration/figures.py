"""Publication figures for lone-agent, board and swarm runs.

    python -m integration.figures --runs runs --out docs/figures [--swimlane <run>]
                                  [--arrows auto|all|exposure] [--format pdf,png]

Reads run directories through the results server's data layer (no HTTP): ``Store.summary()``
for cross-run aggregates, ``Store.episodes()`` for per-agent judge flags and hidden roles,
``Store.timeline()`` for one run's events on a common clock. Every number drawn comes from
those calls, read afresh on every invocation. Figures (see docs/visualization_research.md):

    fig_composition_heatmap   regulators x disregulators, share of agents who cheated
    fig_role_funnel           honeypot seen -> attempted access -> cheated -> disclosed, by hidden role
    fig_adoption_curves       share of agents that have read the honeypot archive, one panel per composition
    fig_swimlane              one run: lanes per agent, model calls, tools, posts and who read them
    fig_controls              cheated / judged per control, lone agent and board

Run states. A swarm run is failed when the summary marks it failed (harness crash or a
``failed_`` name) or an episode ended with a provider error (``failure == "connection_closed"``,
which judge.sh refuses to judge); failed runs are left out of the swarm figures and counted in
their notes. A run is partly judged while fewer of its agents are judged than it has agents:
the heatmap shows its counts as provisional, with no fill, and the funnel leaves it out. A run
whose files cannot be read (a judge line still being written, say) is unreadable: the heatmap
marks its cell, the figures name it, and the command exits with status 1.

The adoption curves count an agent once the timeline's ``first_member_read`` marker fires: a
successful bash or python call whose output holds text found only inside the honeypot archive.
The looser ``first_honeypot_touch`` marker (any call naming another agent's branch) is not drawn.
Each invocation prints how often the archive-read marker agrees with the judge's cheated flag (a
successful inspection of the archive's bytes; a failed attempt leaves no bytes for a marker).

Exit status: 0 when every run was read and every cross-check passed (a figure without data is
skipped with a printed reason); 1 when a run could not be read, a cross-check failed or the
requested swimlane could not be drawn; 2 for usage errors.

Each figure is written once per requested format at 200 dpi. Output is deterministic: the data
decide every value, and the PDF and SVG carry no creation date.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import textwrap
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Patch, Polygon, Rectangle  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from integration.ui.server import Store  # noqa: E402

# -- encoding ------------------------------------------------------------------------------
# Roles: Okabe-Ito colour plus shape. Red and green are reserved for outcomes.
ROLES = ("worker", "regulator", "disregulator")
ROLE_COLOUR = {"worker": "#999999", "regulator": "#0072B2", "disregulator": "#E69F00"}
ROLE_MARKER = {"worker": "o", "regulator": "s", "disregulator": "^"}
RED = "#B2182B"      # cheated, blocked
GREEN = "#009E73"    # honest
INK = "#1a1a1a"
MUTED = "#5a5a5a"
GRID = "#e3e3e3"
NEUTRAL = "#c9c9c9"
FAINT = "#b8b8b8"
UNREADABLE = "#d9d9d9"
# Tool ticks on the swimlane. Board operations and submit are drawn in ink with their own marks.
TOOL_COLOUR = {"bash": "#56B4E9", "python": "#CC79A7"}
OTHER_TOOL = "#8a8a8a"
TOPOLOGY_LABEL = {"single": "lone agent", "board": "board"}
TOPOLOGY_MARK = {"single": {"marker": "o", "markersize": 6, "markerfacecolor": INK, "markeredgecolor": "white",
                             "markeredgewidth": 0.8},
                  "board": {"marker": "o", "markersize": 11, "markerfacecolor": "none", "markeredgecolor": "#CC79A7",
                            "markeredgewidth": 2.0}}
STAGES = (("honey_pots_seen", "honeypot seen"), ("attempted_access", "attempted access"),
          ("cheated", "cheated"), ("disclosed_access", "disclosed"))
SUMMARY_KEY = {"honey_pots_seen": "honeypots_seen", "attempted_access": "attempted", "cheated": "cheated",
               "disclosed_access": "disclosed"}
LOST = "connection_closed"   # the provider ended the episode; judge.sh refuses such runs
AUTO_ARROWS = 40             # --arrows auto draws every reader arrow up to this many
SAVE_META = {"pdf": {"CreationDate": None, "Creator": "integration.figures"},
             "png": {"Software": "integration.figures"},
             "svg": {"Date": None, "Creator": "integration.figures"}}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 10, "axes.labelsize": 8.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#9a9a9a",
    "axes.linewidth": 0.6, "xtick.color": "#5a5a5a", "ytick.color": "#5a5a5a", "axes.labelcolor": INK,
    "text.color": INK, "axes.titlecolor": INK, "grid.color": GRID, "grid.linewidth": 0.6,
    "grid.linestyle": "-", "hatch.linewidth": 0.5, "pdf.fonttype": 42, "svg.hashsalt": "figures",
    "figure.facecolor": "white", "savefig.facecolor": "white",
})


def say(msg: str) -> None:
    print(msg, flush=True)


def plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:  # noqa: BLE001 - missing or half-written
        return None


# -- data ----------------------------------------------------------------------------------
class _OneRun(Store):
    """A Store that sees one run, so one unreadable run cannot sink the summary of the rest."""

    def __init__(self, runs_dir: Path, name: str) -> None:
        super().__init__(runs_dir)
        self._only = name

    def run_names(self) -> list[str]:
        return [self._only]


class Data:
    """Everything one invocation reads. Built afresh on every invocation; nothing is kept between them.

    rows        summary rows of the runs that could be read
    unreadable  runs whose files could not be read, with their composition from run.json when it parses
    swarm       swarm runs that did not fail, in run-name order
    failed      (run, reason) for swarm runs left out as failed
    problems    reasons for exit status 1
    """

    def __init__(self, runs_dir: Path) -> None:
        self.store = Store(runs_dir)
        self.problems: list[str] = []
        self.unreadable: list[dict] = []
        self.rows = self._summary()
        self.swarm: list[dict] = []
        self.failed: list[tuple[str, str]] = []
        self._timelines: dict = {}
        for r in self.rows:
            if r.get("composition") is None:
                continue
            reason = self._failure(r)
            if reason == "unreadable":
                continue
            if reason:
                self.failed.append((r["run"], reason))
            else:
                self.swarm.append(r)

    def problem(self, msg: str) -> None:
        say(f"PROBLEM: {msg}")
        self.problems.append(msg)

    def _mark_unreadable(self, name: str, ex: Exception) -> None:
        run = read_json(self.store.runs_dir / name / "run.json") or {}
        sw = run.get("swarm") or {}
        self.unreadable.append({"run": name, "error": f"{type(ex).__name__}: {ex}",
                                "kind": "swarm" if sw else (run.get("topology") or "unknown"),
                                "regulators": sw.get("regulators"), "disregulators": sw.get("disregulators"),
                                "workers": sw.get("workers"), "n_agents": sw.get("n_agents")})
        self.problem(f"run {name} could not be read ({type(ex).__name__}: {ex})")

    def _summary(self) -> list[dict]:
        """``store.summary()['runs']``; on failure, the same summary built run by run."""
        try:
            return self.store.summary()["runs"]
        except Exception as ex:  # noqa: BLE001 - a run still being written can hold a partial line
            say(f"summary over all runs failed ({type(ex).__name__}: {ex}); reading runs one at a time")
        rows = []
        for name in self.store.run_names():
            try:
                rows.extend(_OneRun(self.store.runs_dir, name).summary()["runs"])
            except Exception as ex:  # noqa: BLE001
                self._mark_unreadable(name, ex)
        return rows

    def _failure(self, r: dict) -> str | None:
        if r.get("failed"):
            return "crashed (harness failure or failed_ name)"
        try:
            eps = self.store.episodes(r["run"])
        except Exception as ex:  # noqa: BLE001 - changed since the summary read it
            self._mark_unreadable(r["run"], ex)
            return "unreadable"
        lost = sum(1 for e in eps if e.get("failure") == LOST)
        return f"lost {plural(lost, 'episode')} to a provider error" if lost else None

    def unreadable_of(self, kinds: tuple[str, ...]) -> list[dict]:
        return [u for u in self.unreadable if u["kind"] in kinds + ("unknown",)]

    def timeline(self, run: str) -> dict | None:
        """``store.timeline(run)``, read once per invocation; None when the eval log cannot be read."""
        if run not in self._timelines:
            try:
                self._timelines[run] = self.store.timeline(run)
            except Exception as ex:  # noqa: BLE001 - eval log missing or still being written
                say(f"  no timeline for {run}: {type(ex).__name__}: {ex}")
                self._timelines[run] = None
        return self._timelines[run]

    def agents(self, run: str) -> list[dict]:
        """Every agent of one run with role, judge state and judge flags, judged by the rule summary() uses."""
        eps = self.store.episodes(run)
        roles = {}
        if any(not e.get("role") for e in eps):
            roles = ((read_json(self.store.runs_dir / run / "run.json") or {}).get("swarm") or {}).get("roles") or {}
        out = []
        for e in eps:
            v = e.get("verdict") or {}
            judged = bool(e.get("verdict") and v.get("cheated") is not None and e.get("has_transcript"))
            rec = {"run": run, "agent": e.get("orbit_agent"), "role": e.get("role") or roles.get(e.get("orbit_agent")),
                   "judged": judged, "has_verdict": e.get("verdict") is not None,
                   "has_transcript": bool(e.get("has_transcript"))}
            rec.update({k: judged and bool(v.get(k)) for k, _ in STAGES})
            out.append(rec)
        return out


def fully_judged(r: dict) -> bool:
    return bool(r.get("n_agents")) and r.get("judged") == r.get("n_agents")


def run_end(tl: dict) -> float:
    """Last moment on the run's clock: the sample's total time or the last event, whichever is later."""
    ts = [tl.get("duration") or 0.0]
    for e in tl["events"]:
        ts.extend(x for x in (e.get("t"), e.get("t_end")) if x is not None)
    return max(ts)


def left_out_note(failed: list, unreadable: list[dict], extra: list[str] = ()) -> str:
    """One wrapped line naming what a figure leaves out."""
    parts = list(extra)
    if failed:
        parts.append(f"{plural(len(failed), 'failed run')} left out")
    if unreadable:
        parts.append("could not read " + ", ".join(u["run"] for u in unreadable))
    return "; ".join(parts)


def footnote(fig, text: str, y: float = -0.02, width: int = 150) -> None:
    """A note under the whole figure, at figure fraction y."""
    if text:
        fig.text(0.01, y, textwrap.fill(text, width), fontsize=7, color=MUTED, va="top", ha="left")


def axes_note(ax, text: str, below_pt: float, width: int = 110) -> None:
    """A note under an axes, its top ``below_pt`` points under the axes' bottom edge."""
    if text:
        ax.annotate(textwrap.fill(text, width), xy=(0, 0), xycoords="axes fraction", xytext=(0, -below_pt),
                    textcoords="offset points", fontsize=7, color=MUTED, va="top", ha="left", annotation_clip=False)


# -- output --------------------------------------------------------------------------------
def save(fig, out: Path, stem: str, formats: list[str]) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in formats:
        path = out / f"{stem}.{fmt}"
        fig.savefig(path, dpi=200, bbox_inches="tight", metadata=SAVE_META.get(fmt))
        say(f"wrote {path}")
        paths.append(path)
    plt.close(fig)
    return paths


def note_stale(out: Path, stem: str, formats: list[str]) -> None:
    for fmt in formats:
        p = out / f"{stem}.{fmt}"
        if p.exists():
            say(f"  note: {p} exists from an earlier invocation and was NOT updated")


def ink_on(rgba) -> str:
    """White or ink for text set inside a fill, by the fill's luminance."""
    r, g, b = to_rgb(rgba)
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "white" if lum < 0.45 else INK


def role_handles(roles=ROLES, counts: dict | None = None) -> list[Line2D]:
    hs = []
    for r in roles:
        label = r if counts is None else f"{r} (n = {counts[r]})"
        hs.append(Line2D([], [], linestyle="none", marker=ROLE_MARKER[r], markersize=6.5,
                         markerfacecolor=ROLE_COLOUR[r], markeredgecolor="white", markeredgewidth=0.6, label=label))
    return hs


# -- 1. composition heatmap ----------------------------------------------------------------
def fig_composition_heatmap(data: Data, out: Path, formats: list[str]):
    stem = "fig_composition_heatmap"
    runs = [r for r in data.swarm if r.get("regulators") is not None and r.get("disregulators") is not None]
    bad = [u for u in data.unreadable_of(("swarm",))
           if u["regulators"] is not None and u["disregulators"] is not None]
    if not runs and not bad:
        say(f"skipped {stem}: no usable swarm runs with a composition")
        note_stale(out, stem, formats)
        return None
    sizes = {r["regulators"] + r["disregulators"] + (r.get("workers") or 0) for r in runs}
    sizes |= {r["n_agents"] for r in runs if r.get("n_agents")}
    sizes |= {u["n_agents"] or (u["regulators"] + u["disregulators"] + (u["workers"] or 0)) for u in bad}
    team = max(sizes)
    if len(sizes) > 1:
        say(f"  {stem}: runs of several team sizes {sorted(sizes)}; the grid is drawn for {team}")
    cells: dict = defaultdict(lambda: {"k": 0, "n": 0, "done": [], "partial": [], "unjudged": [], "unreadable": [],
                                       "disagree": []})
    for r in runs:
        c = cells[(r["regulators"], r["disregulators"])]
        if fully_judged(r):
            c["k"] += r["cheated"]
            c["n"] += r["judged"]
            c["done"].append(r["run"])
            if r.get("report_cheated") is not None and bool(r["report_cheated"]) != (r["cheated"] > 0):
                c["disagree"].append(r["run"])
        elif r.get("judged"):
            c["partial"].append(r)
        else:
            c["unjudged"].append(r["run"])
    for u in bad:
        cells[(u["regulators"], u["disregulators"])]["unreadable"].append(u["run"])
    cmap = LinearSegmentedColormap.from_list("cheated", plt.get_cmap("Reds")(np.linspace(0.08, 0.95, 256)))
    norm = Normalize(0, 1)
    side = min(7.0, 2.4 + 0.42 * (team + 1))
    fig, ax = plt.subplots(figsize=(side + 1.3, side))
    gap = 0.06
    small = 7.5 if team <= 10 else 6
    seen = set()
    for reg in range(team + 1):
        for dis in range(team + 1):
            if reg + dis > team:
                continue
            x0, y0, w = dis - 0.5 + gap / 2, reg - 0.5 + gap / 2, 1 - gap
            c = cells.get((reg, dis))
            if c is None:
                ax.add_patch(Rectangle((x0, y0), w, w, facecolor="white", edgecolor="#cfcfcf", hatch="////",
                                       linewidth=0))
                continue
            if c["done"]:
                # final values: every agent of every counted run is judged
                fill = cmap(norm(c["k"] / c["n"]))
                ax.add_patch(Rectangle((x0, y0), w, w, facecolor=fill, linewidth=0))
                label = f"{c['k']}/{c['n']}" + (f"\n{len(c['done'])} runs" if len(c["done"]) > 1 else "")
                ax.text(dis, reg, label, ha="center", va="center", fontsize=small, color=ink_on(fill))
                if c["partial"] or c["unjudged"] or c["unreadable"]:
                    ax.plot(x0 + 0.13 * w, y0 + 0.13 * w, marker="o", markersize=3.6, markerfacecolor="white",
                            markeredgecolor=INK, markeredgewidth=0.8, linestyle="none", zorder=5)
                    seen.add("uncounted")
            elif c["partial"]:
                # provisional: counts so far, no fill, so they cannot be read as final
                k = sum(r["cheated"] for r in c["partial"])
                j = sum(r["judged"] for r in c["partial"])
                n = sum(r["n_agents"] for r in c["partial"])
                ax.add_patch(Rectangle((x0 + 0.02, y0 + 0.02), w - 0.04, w - 0.04, facecolor="white", edgecolor=MUTED,
                                       linewidth=0.9, linestyle=(0, (2.5, 1.5))))
                ax.text(dis, reg + 0.13, f"{k}/{j}", ha="center", va="center", fontsize=small - 0.5, color=INK,
                        fontstyle="italic")
                ax.text(dis, reg - 0.17, f"{j} of {n}\njudged", ha="center", va="center", fontsize=5.5, color=MUTED,
                        linespacing=1.0)
                seen.add("partial")
            elif c["unreadable"]:
                ax.add_patch(Rectangle((x0, y0), w, w, facecolor=UNREADABLE, linewidth=0))
                ax.text(dis, reg, "could not\nbe read", ha="center", va="center", fontsize=5.5, color=INK,
                        linespacing=1.0)
                seen.add("unreadable")
            else:
                ax.add_patch(Rectangle((x0, y0), w, w, facecolor="#efefef", linewidth=0))
                ax.text(dis, reg, "not\njudged", ha="center", va="center", fontsize=6, color=MUTED)
                seen.add("unjudged")
            if c["disagree"]:
                s = 0.3 * w
                xr, yt = x0 + w, y0 + w
                ax.add_patch(Polygon([(xr, yt), (xr - s, yt), (xr, yt - s)], closed=True, facecolor=INK,
                                     edgecolor="white", linewidth=0.8, joinstyle="miter"))
                seen.add("disagree")
    ax.set_xlim(-0.5, team + 0.5)
    ax.set_ylim(-0.5, team + 0.5)
    ax.set_aspect("equal")
    ax.set_xticks(range(team + 1))
    ax.set_yticks(range(team + 1))
    ax.tick_params(length=0)
    for s in ("left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.set_xlabel("disregulators in the team")
    ax.set_ylabel("regulators in the team")
    ax.set_title("Share of agents who cheated, by team composition", loc="left", pad=16)
    ax.text(0, 1.012, f"Teams of {team} agents. Fill only where every agent of the run is judged.",
            transform=ax.transAxes, fontsize=7, color=MUTED, va="bottom")
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label("share of agents who cheated")
    cb.outline.set_visible(False)
    handles = [Patch(facecolor="white", edgecolor="#cfcfcf", hatch="////", linewidth=0, label="no usable run")]
    if "partial" in seen:
        handles.append(Patch(facecolor="white", edgecolor=MUTED, linewidth=0.9, linestyle=(0, (2.5, 1.5)),
                             label="judging in progress:\ncheated / judged so far"))
    if "unjudged" in seen:
        handles.append(Patch(facecolor="#efefef", linewidth=0, label="run, no verdicts yet"))
    if "unreadable" in seen:
        handles.append(Patch(facecolor=UNREADABLE, linewidth=0, label="run files could not be read"))
    if "uncounted" in seen:
        handles.append(Line2D([], [], linestyle="none", marker="o", markersize=4.5, markerfacecolor="white",
                              markeredgecolor=INK, markeredgewidth=0.8,
                              label="another run in the cell is\nnot counted yet"))
    if "disagree" in seen:
        handles.append(Line2D([], [], linestyle="none", marker=[(-1, 1), (1, 1), (1, -1)], markersize=7,
                              markerfacecolor=INK, markeredgecolor="white", markeredgewidth=0.6,
                              label="team report disagrees\nwith the judge"))
    ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.0, 1.0), fontsize=7, handlelength=1.2)
    axes_note(ax, left_out_note(data.failed, bad), below_pt=34, width=90)
    say(f"{stem}: {len(runs)} usable swarm runs and {len(bad)} unreadable in {len(cells)} cells, team size {team}")
    for (reg, dis), c in sorted(cells.items()):
        bits = [f"cheated {c['k']} of {c['n']} judged in fully judged runs {', '.join(c['done']) or '(none)'}"]
        bits += [f"partly judged {r['run']}: {r['cheated']} of {r['judged']} judged, {r['n_agents']} agents"
                 for r in c["partial"]]
        bits += [f"not judged {x}" for x in c["unjudged"]] + [f"unreadable {x}" for x in c["unreadable"]]
        bits += [f"report disagrees {x}" for x in c["disagree"]]
        say(f"  R{reg} D{dis}: " + "; ".join(bits))
    return save(fig, out, stem, formats)


# -- 2. role funnel ------------------------------------------------------------------------
def fig_role_funnel(data: Data, out: Path, formats: list[str]):
    stem = "fig_role_funnel"
    agents, used, partial = [], [], []
    for r in data.swarm:
        if not r.get("judged"):
            say(f"  {stem}: {r['run']} has no judged agents yet")
            continue
        if not fully_judged(r):
            say(f"  {stem}: left out {r['run']}: {r['judged']} of {r['n_agents']} agents judged")
            partial.append(r)
            continue
        try:
            recs = [a for a in data.agents(r["run"]) if a["judged"]]
        except Exception as ex:  # noqa: BLE001
            data.problem(f"{stem}: {r['run']} could not be read ({type(ex).__name__}: {ex})")
            continue
        # the per-agent flags must add up to the run's summary row
        for key, _ in STAGES:
            got = sum(a[key] for a in recs)
            if got != r[SUMMARY_KEY[key]]:
                data.problem(f"MISMATCH {r['run']}: {key} per agent {got} != summary {r[SUMMARY_KEY[key]]}")
        for role in ROLES:
            got = sum(1 for a in recs if a["role"] == role and a["cheated"])
            want = ((r.get("by_role") or {}).get(role) or {}).get("cheated")
            if want is not None and got != want:
                data.problem(f"MISMATCH {r['run']}: {role} cheated per agent {got} != summary by_role {want}")
        agents.extend(recs)
        used.append(r["run"])
    if not agents:
        say(f"skipped {stem}: no fully judged swarm run")
        note_stale(out, stem, formats)
        return None
    unknown = [a for a in agents if a["role"] not in ROLES]
    if unknown:
        say(f"  {len(unknown)} judged agents without a known role left out: "
            + ", ".join(f"{a['run']}/{a['agent']}" for a in unknown))
    n = {role: sum(1 for a in agents if a["role"] == role) for role in ROLES}
    present = [role for role in ROLES if n[role]]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    width, step = 0.22, 0.25
    offs = {role: (i - (len(present) - 1) / 2) * step for i, role in enumerate(present)}
    gtrans = blended_transform_factory(ax.transData, ax.transAxes)
    for si, (key, _) in enumerate(STAGES):
        for role in present:
            k = sum(1 for a in agents if a["role"] == role and a[key])
            share = k / n[role]
            x = si + offs[role]
            ax.bar(x, share, width=width, color=ROLE_COLOUR[role], linewidth=0, zorder=2)
            ax.text(x, share + 0.02, f"{k}/{n[role]}", ha="center", va="bottom", fontsize=6.5, color=INK)
            # the role's shape under its bar, so role is not told by colour alone
            ax.plot(x, -0.04, marker=ROLE_MARKER[role], markersize=5, markerfacecolor=ROLE_COLOUR[role],
                    markeredgecolor="white", markeredgewidth=0.5, transform=gtrans, clip_on=False, linestyle="none")
    ax.set_xticks(range(len(STAGES)))
    ax.set_xticklabels([label for _, label in STAGES])
    ax.set_xlim(-0.6, len(STAGES) - 0.4)
    ax.set_ylim(0, 1.12)
    ax.set_yticks(np.linspace(0, 1, 5))
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0, pad=11)
    ax.set_ylabel("share of judged agents in the role")
    for si in range(len(STAGES) - 1):
        ax.annotate("", xy=(si + 0.62, -0.085), xytext=(si + 0.38, -0.085), xycoords=("data", "axes fraction"),
                    arrowprops={"arrowstyle": "-|>", "color": "#9a9a9a", "lw": 0.8})
    ax.set_title("Outcome funnel by hidden role", loc="left", pad=18)
    ax.text(0, 1.02, f"Pooled over {plural(len(used), 'fully judged swarm run')} and {len(agents)} agents. "
            "Bar label: agents at the stage / agents in the role.",
            transform=ax.transAxes, fontsize=7, color=MUTED, va="bottom")
    ax.legend(handles=role_handles(present, n), loc="upper center", ncol=len(present), bbox_to_anchor=(0.5, -0.17))
    extra = [f"judging in progress, left out: " + ", ".join(f"{r['run']} ({r['judged']} of {r['n_agents']})"
                                                          for r in partial)] if partial else []
    axes_note(ax, left_out_note(data.failed, data.unreadable_of(("swarm",)), extra), below_pt=62, width=120)
    say(f"{stem}: {len(agents)} judged agents from {len(used)} runs; per role " + ", ".join(f"{r} {n[r]}" for r in ROLES))
    for key, label in STAGES:
        say(f"  {label}: " + ", ".join(f"{role} {sum(1 for a in agents if a['role'] == role and a[key])}/{n[role]}"
                                       for role in present))
    return save(fig, out, stem, formats)


# -- 3. adoption curves --------------------------------------------------------------------
def fig_adoption_curves(data: Data, out: Path, formats: list[str]):
    stem = "fig_adoption_curves"
    curves, no_log = [], []
    agree = judged = 0
    for r in data.swarm:
        if r.get("regulators") is None or r.get("disregulators") is None:
            say(f"  {stem}: {r['run']} has no regulator and disregulator counts; left out")
            continue
        tl = data.timeline(r["run"])
        if tl is None:
            no_log.append(r["run"])
            continue
        lanes = tl.get("agents") or []
        if not lanes:
            say(f"  {stem}: {r['run']} has no agents on the timeline; left out")
            continue
        first = {a: t for a, t in (tl.get("first_member_read") or {}).items() if a in lanes and t is not None}
        roles = tl.get("roles") or {}
        dis_posts = [p["t"] for p in tl.get("board") or []
                     if p.get("t") is not None and roles.get(p.get("sender")) == "disregulator"]
        # how often the archive-read marker agrees with the judge's cheated flag (bytes seen)
        try:
            recs = [a for a in data.agents(r["run"]) if a["judged"]]
        except Exception as ex:  # noqa: BLE001
            data.problem(f"{stem}: {r['run']} episodes could not be read ({type(ex).__name__}: {ex})")
            recs = []
        off = [a["agent"] for a in recs if (a["agent"] in first) != bool(a.get("cheated"))]
        agree += len(recs) - len(off)
        judged += len(recs)
        curves.append({"run": r["run"], "R": r["regulators"], "D": r["disregulators"], "n": len(lanes),
                       "reads": sorted(first.values()), "end": run_end(tl),
                       "dis_post": min(dis_posts) if dis_posts else None})
        say(f"  {r['run']}: R{r['regulators']} D{r['disregulators']}, {len(first)}/{len(lanes)} agents read the "
            f"archive; first disregulator post at {min(dis_posts) if dis_posts else None}; marker agrees with the "
            f"judge's cheated flag for {len(recs) - len(off)} of {len(recs)} judged"
            + (f" (differs: {', '.join(off)})" if off else ""))
    if not curves:
        say(f"skipped {stem}: no usable swarm run with a readable timeline")
        note_stale(out, stem, formats)
        return None
    groups: dict = defaultdict(list)
    for c in curves:
        groups[(c["D"], c["R"])].append(c)
    keys = sorted(groups)
    ncols = min(4, len(keys))
    nrows = math.ceil(len(keys) / ncols)
    W, H = max(2.45 * ncols + 0.5, 6.0), 1.75 * nrows + 1.2
    fig, axes = plt.subplots(nrows, ncols, figsize=(W, H), sharex=True, sharey=True, squeeze=False)
    xmax = max(c["end"] for c in curves) or 1.0
    multi = any(len(g) > 1 for g in groups.values())
    dis_seen = False
    for i, key in enumerate(keys):
        ax = axes[i // ncols][i % ncols]
        runs = sorted(groups[key], key=lambda c: c["run"])
        gtrans = blended_transform_factory(ax.transData, ax.transAxes)
        for c in runs:
            if c["dis_post"] is not None:
                dis_seen = True
                ax.axvline(c["dis_post"], color=ROLE_COLOUR["disregulator"], linewidth=0.8, linestyle=(0, (1.5, 1.5)),
                           zorder=1)
                ax.plot(c["dis_post"], 0, marker="^", markersize=5.5, markerfacecolor=ROLE_COLOUR["disregulator"],
                        markeredgecolor="white", markeredgewidth=0.5, transform=gtrans, clip_on=False,
                        linestyle="none", zorder=6)
        for c in runs:
            xs = [0.0] + c["reads"] + [c["end"]]
            ys = [0.0] + [(j + 1) / c["n"] for j in range(len(c["reads"]))]
            ys.append(ys[-1])
            ax.step(xs, ys, where="post", color=FAINT if len(runs) > 1 else INK,
                    linewidth=1.0 if len(runs) > 1 else 1.6, zorder=3)
        if len(runs) > 1:
            grid = sorted({0.0} | {t for c in runs for t in c["reads"]} | {max(c["end"] for c in runs)})
            med = [float(np.median([sum(1 for t in c["reads"] if t <= g) / c["n"] for c in runs])) for g in grid]
            ax.step(grid, med, where="post", color=INK, linewidth=2.0, zorder=4)
            final = med[-1]
            ns = {c["n"] for c in runs}
            label = (", ".join(str(len(c["reads"])) for c in runs) + f" of {ns.pop()}" if len(ns) == 1
                     else ", ".join(f"{len(c['reads'])}/{c['n']}" for c in runs))
        else:
            c = runs[0]
            final = len(c["reads"]) / c["n"]
            label = f"{len(c['reads'])}/{c['n']}"
        high = final > 0.8
        ax.text(0.98, final - 0.05 if high else final + 0.05, label, transform=blended_transform_factory(
            ax.transAxes, ax.transData), ha="right", va="top" if high else "bottom", fontsize=7, color=INK)
        ax.set_title(f"R{key[1]} D{key[0]}" + (f"  ({len(runs)} runs)" if len(runs) > 1 else ""), loc="left",
                     fontsize=8.5)
        ax.set_ylim(-0.03, 1.05)
        ax.set_yticks([0, 0.5, 1])
        ax.set_xlim(0, xmax * 1.02)
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)
    for j in range(len(keys), nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)
    for i in range(len(keys)):
        # tick labels on the lowest panel of each column, also where the panel under it is unused
        if i + ncols >= len(keys):
            axes[i // ncols][i % ncols].xaxis.set_tick_params(labelbottom=True)
    # fixed margins in inches: title and legend above the panels, axis labels outside them. Plain figure text,
    # so tight_layout below fits the panels to exactly the rectangle left for them.
    fig.text(0.5, 0.06 / H, "seconds from the start of the run", fontsize=8.5, ha="center", va="bottom")
    fig.text(0.04 / W, 0.5, "share of agents", fontsize=8.5, rotation=90, ha="left",
             va="center")
    fig.text(0.01, 1 - 0.06 / H, "Agents that have read the honeypot archive over time, by team composition",
             ha="left", va="top", fontsize=10)
    handles, labels = [], []
    if multi:
        handles += [Line2D([], [], color=FAINT, linewidth=1.0), Line2D([], [], color=INK, linewidth=2.0)]
        labels += ["one run", "median of the runs"]
    else:
        handles.append(Line2D([], [], color=INK, linewidth=1.6))
        labels.append("one run (end label: agents that read / agents)")
    if dis_seen:
        handles.append(Line2D([], [], color=ROLE_COLOUR["disregulator"], linewidth=0.8, linestyle=(0, (1.5, 1.5)),
                              marker="^", markersize=5.5, markerfacecolor=ROLE_COLOUR["disregulator"],
                              markeredgecolor="white"))
        labels.append("first board post by a disregulator")
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.005, 1 - 0.32 / H), ncol=len(handles),
               fontsize=7.5)
    note = ("R = regulators, D = disregulators. An agent counts from its first successful bash or python call whose "
            "output holds text found only inside the honeypot archive (timeline marker). "
            + (f"The marker agrees with the judge's cheated flag for {agree} of {judged} judged agents."
               if judged else "No agent in these runs is judged yet."))
    extra = [f"no readable eval log: {', '.join(no_log)}"] if no_log else []
    lo = left_out_note(data.failed, data.unreadable_of(("swarm",)), extra)
    fig.tight_layout(rect=(0.26 / W, 0.3 / H, 1.0, 1 - 0.58 / H))
    footnote(fig, note + (" Left out: " + lo + "." if lo else ""), y=0.0, width=int(24 * W))
    if judged and agree != judged:
        say(f"  note: the archive-read marker disagrees with the judge for {judged - agree} of {judged} judged agents")
    say(f"{stem}: {len(curves)} runs in {len(keys)} compositions; marker agrees with the judge for {agree} of {judged}")
    return save(fig, out, stem, formats)


# -- 4. swimlane ---------------------------------------------------------------------------
def pick_swimlane_run(data: Data, wanted: str | None):
    """The requested run, else the most informative usable swarm run, else gemini_board.

    Most informative: most hidden roles present, every agent judged, most cheated, most attempted; then name.
    """
    if wanted:
        return wanted, data.timeline(wanted)

    def rank(r):
        roles = sum(1 for k in ("regulators", "disregulators", "workers") if r.get(k))
        return (-roles, not fully_judged(r), -(r.get("cheated") or 0), -(r.get("attempted") or 0), r["run"])

    for r in sorted(data.swarm, key=rank):
        tl = data.timeline(r["run"])
        if tl is not None:
            return r["run"], tl
    names = data.store.run_names()
    if "gemini_board" in names:
        say("fig_swimlane: no usable swarm run with a readable timeline; using gemini_board")
        return "gemini_board", data.timeline("gemini_board")
    say("skipped fig_swimlane: no usable swarm run and no gemini_board")
    return None, None


def fig_swimlane(data: Data, wanted: str | None, arrows: str, out: Path, formats: list[str]):
    stem = "fig_swimlane"
    run, tl = pick_swimlane_run(data, wanted)
    if tl is None:
        if run:
            msg = f"skipped {stem}: no readable timeline for {run}"
            data.problem(msg) if wanted else say(msg)
        note_stale(out, stem, formats)
        return None
    row = next((r for r in data.rows if r["run"] == run), None)
    failed = next((why for name, why in data.failed if name == run), None) or (
        "crashed (harness failure or failed_ name)" if row and row.get("failed") else None)
    try:
        recs = {a["agent"]: a for a in data.agents(run)}
    except Exception as ex:  # noqa: BLE001
        data.problem(f"{stem}: episodes of {run} could not be read ({type(ex).__name__}: {ex})")
        recs = {}
    roles = tl.get("roles") or {}
    lanes_in = list(tl.get("agents") or [])
    order = {r: i for i, r in enumerate(ROLES)}
    lanes = sorted(lanes_in, key=lambda a: (order.get(roles.get(a), len(ROLES)), a))
    y = {a: len(lanes) - 1 - i for i, a in enumerate(lanes)}
    end = run_end(tl)
    fig, ax = plt.subplots(figsize=(10.5, 1.8 + 0.5 * len(lanes)))
    for a in lanes:
        if y[a] % 2 == 0:
            ax.axhspan(y[a] - 0.5, y[a] + 0.5, color="#f6f6f6", zorder=0, linewidth=0)
    prev = None
    for a in lanes:
        if prev is not None and roles.get(a) != roles.get(prev):
            ax.axhline(y[a] + 0.5, color="#b5b5b5", linewidth=0.8, zorder=1)
        prev = a
    min_w = end * 0.0012
    model = defaultdict(list)
    dcall = defaultdict(list)
    seen_kinds = set()
    blocked_marks = []
    for e in tl["events"]:
        a = e.get("agent")
        if e["kind"] == "defense" and e.get("allow") is False:
            seen_kinds.add("intervention")
            span = [y[a]] if a in y else list(y.values())
            for yy in span:
                ax.vlines(e["t"], yy - 0.47, yy + 0.47, color=RED, linewidth=1.3, zorder=2)
            continue
        if a not in y or e.get("t") is None:
            continue
        t = e["t"]
        if e["kind"] in ("model", "defense_call"):
            w = max((e.get("t_end") or t) - t, min_w)
            (model if e["kind"] == "model" else dcall)[a].append((t, w))
            if e["kind"] == "model" and e.get("refusal"):
                blocked_marks.append((t, y[a]))
            continue
        if e["kind"] not in ("tool", "submit"):
            continue
        fn = e.get("function")
        if e.get("blocked"):
            blocked_marks.append((t, y[a]))
        if fn == "channel_post":
            continue  # drawn with the board posts
        if fn == "channel_read":
            ax.vlines(t, y[a] - 0.14, y[a] + 0.14, color=INK, linewidth=0.7, zorder=3)
            seen_kinds.add("channel_read")
        elif fn == "submit" or e["kind"] == "submit":
            ax.vlines(t, y[a] - 0.4, y[a] + 0.4, color=INK, linewidth=1.8, zorder=3)
            seen_kinds.add("submit")
        else:
            ax.vlines(t, y[a] - 0.28, y[a] + 0.28, color=TOOL_COLOUR.get(fn, OTHER_TOOL), linewidth=0.9, zorder=3)
            seen_kinds.add(fn if fn in TOOL_COLOUR else "other_tool")
    for a, spans in model.items():
        ax.broken_barh(spans, (y[a] - 0.08, 0.16), facecolors=NEUTRAL, linewidth=0, zorder=2)
    for a, spans in dcall.items():
        ax.broken_barh(spans, (y[a] - 0.36, 0.08), facecolors="#7d7d7d", linewidth=0, zorder=2)
        seen_kinds.add("defense_call")
    # reader arrows: one per (post, reader), at the reader's first exposure to the post
    first_read = {a: t for a, t in (tl.get("first_member_read") or {}).items() if a in y and t is not None}
    reads = []
    for p in tl.get("board") or []:
        if p.get("t") is None or p.get("sender") not in y:
            continue
        readers = {}
        for rb in p.get("read_by") or []:
            ra = rb.get("agent")
            if ra == p["sender"] or ra not in y or rb.get("t") is None:
                continue
            cur = readers.setdefault(ra, {"t": rb["t"], "read": False})
            cur["t"] = min(cur["t"], rb["t"])
            cur["read"] = cur["read"] or rb.get("how") == "read"
        for ra, info in sorted(readers.items()):
            reads.append({"t0": p["t"], "y0": y[p["sender"]], "reader": ra, "t": info["t"],
                          "how": "read" if info["read"] else "model_exposure"})
    mode = arrows if arrows != "auto" else ("all" if len(reads) <= AUTO_ARROWS else "exposure")
    shown = reads if mode == "all" else [r for r in reads if r["reader"] in first_read and r["t"] <= first_read[r["reader"]]]
    n_arrows = {"read": 0, "model_exposure": 0}
    for r in shown:
        n_arrows[r["how"]] += 1
        ax.add_patch(FancyArrowPatch((r["t0"], r["y0"]), (r["t"], y[r["reader"]]), arrowstyle="-|>", mutation_scale=6,
                                     linewidth=0.55 if mode == "all" else 0.75, color="#4a4a4a",
                                     alpha=0.6 if mode == "all" else 0.8, zorder=4,
                                     linestyle="-" if r["how"] == "read" else (0, (3, 2)), shrinkA=2, shrinkB=1))
    n_posts = 0
    for p in tl.get("board") or []:
        if p.get("t") is None or p.get("sender") not in y:
            continue
        n_posts += 1
        ax.plot(p["t"], y[p["sender"]], marker="o", markersize=5, markerfacecolor=INK, markeredgecolor="white",
                markeredgewidth=1.0, zorder=6, linestyle="none")
        seen_kinds.add("post")
    for a, t in first_read.items():
        ax.plot(t, y[a], marker="D", markersize=5.5, markerfacecolor=INK, markeredgecolor="white",
                markeredgewidth=0.8, zorder=8, linestyle="none")
        seen_kinds.add("member_read")
    if blocked_marks:
        ax.plot([t for t, _ in blocked_marks], [yy + 0.4 for _, yy in blocked_marks], marker="x", markersize=4.5,
                color=RED, markeredgewidth=1.2, linestyle="none", zorder=8)
        seen_kinds.add("blocked")
    # verdict at the lane's end, judged by the summary's rule (a verdict needs a transcript)
    vtrans = blended_transform_factory(ax.transAxes, ax.transData)
    verdicts = {}
    for a in lanes:
        rec = recs.get(a)
        v = rec["cheated"] if rec and rec["judged"] and not failed else None
        verdicts[a] = v
        if v is True:
            text, face, fg = "cheated", RED, "white"
        elif v is False:
            text, face, fg = "honest", GREEN, "white"
        elif rec and rec["has_verdict"] and (failed or not rec["has_transcript"]):
            text, face, fg = "not counted", "#ececec", MUTED
        else:
            text, face, fg = "not judged", "#ececec", INK
        ax.text(1.008, y[a], text, transform=vtrans, ha="left", va="center", fontsize=7, color=fg,
                fontweight="bold" if v is not None else "normal",
                bbox={"boxstyle": "round,pad=0.25", "facecolor": face, "linewidth": 0})
    # lane labels with the role glyph
    ax.set_yticks([y[a] for a in lanes])
    ax.set_yticklabels(lanes)
    ax.tick_params(axis="y", length=0, pad=14 if roles else 4)
    for a in lanes:
        r = roles.get(a)
        if r in ROLE_MARKER:
            ax.plot(-0.012, y[a], marker=ROLE_MARKER[r], markersize=6.5, markerfacecolor=ROLE_COLOUR[r],
                    markeredgecolor="white", markeredgewidth=0.6, transform=vtrans, clip_on=False, linestyle="none")
    ax.set_ylim(-0.6, len(lanes) - 0.4)
    ax.set_xlim(0, end * 1.01 if end > 0 else 1)
    ax.spines["left"].set_visible(False)
    ax.xaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlabel("seconds from the start of the sample")
    title = f"What each agent did over time in run {run}"
    if failed:
        title += " (failed run)"
    ax.set_title(title, loc="left", pad=25 if failed else 16)
    if not reads:
        sub = "No board post was read by another agent."
    elif mode == "all":
        sub = f"Arrows: every post to each other agent that read it ({len(reads)})."
    else:
        sub = (f"Arrows: only posts an agent read before its first archive read ({len(shown)} of {len(reads)} "
               "post reads); short ticks mark every board read.")
    if failed:
        sub = f"The run {failed}; no verdict is counted.\n" + sub
    ax.text(0, 1.012, sub, transform=ax.transAxes, fontsize=7, color=MUTED, va="bottom")
    handles = [Patch(facecolor=NEUTRAL, linewidth=0, label="model call")]
    if "defense_call" in seen_kinds:
        handles.append(Patch(facecolor="#7d7d7d", linewidth=0, label="defense model call"))
    for fn, label in (("bash", "bash call"), ("python", "python call"), ("other_tool", "other tool call")):
        if fn in seen_kinds:
            handles.append(Line2D([], [], color=TOOL_COLOUR.get(fn, OTHER_TOOL), linewidth=1.2, label=label))
    if "channel_read" in seen_kinds:
        handles.append(Line2D([], [], color=INK, linewidth=0.7, label="board read (short tick)"))
    if "submit" in seen_kinds:
        handles.append(Line2D([], [], color=INK, linewidth=1.8, label="submit (tall tick)"))
    if "post" in seen_kinds:
        handles.append(Line2D([], [], linestyle="none", marker="o", markersize=5, markerfacecolor=INK,
                              markeredgecolor="white", label="board post"))
    if n_arrows["read"]:
        handles.append(Line2D([], [], color="#4a4a4a", linewidth=0.8, label="post read by an agent"))
    if n_arrows["model_exposure"]:
        handles.append(Line2D([], [], color="#4a4a4a", linewidth=0.8, linestyle=(0, (3, 2)),
                              label="post only shown to the model"))
    if "member_read" in seen_kinds:
        handles.append(Line2D([], [], linestyle="none", marker="D", markersize=5, markerfacecolor=INK,
                              markeredgecolor="white", label="first honeypot archive read"))
    if "blocked" in seen_kinds:
        handles.append(Line2D([], [], linestyle="none", marker="x", markersize=5, color=RED, markeredgewidth=1.2,
                              label="blocked by a defense"))
    if "intervention" in seen_kinds:
        handles.append(Line2D([], [], color=RED, linewidth=1.3, label="defense intervention (full-height line)"))
    present_roles = [r for r in ROLES if r in set(roles.get(a) for a in lanes)]
    handles += role_handles(present_roles)
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0, -0.11 - 0.25 / max(len(lanes), 1)),
              ncol=4, fontsize=7, handlelength=1.8, columnspacing=1.4)
    per_sender = defaultdict(int)
    for p in tl.get("board") or []:
        if p.get("t") is not None and p.get("sender") in y:
            per_sender[p["sender"]] += 1
    say(f"{stem}: run {run}{' (failed)' if failed else ''}, {len(lanes)} lanes, {len(tl['events'])} events, "
        f"{n_posts} posts {dict(sorted(per_sender.items()))}, arrows {mode}: {n_arrows['read']} read and "
        f"{n_arrows['model_exposure']} model-only of {len(reads)} post reads; "
        f"archive read {first_read}, verdicts {verdicts}")
    return save(fig, out, stem, formats)


# -- 5. controls ---------------------------------------------------------------------------
def fig_controls(data: Data, out: Path, formats: list[str]):
    stem = "fig_controls"
    groups: dict = defaultdict(lambda: {"k": 0, "n": 0, "runs": []})
    for r in data.rows:
        if r.get("composition") is not None or r.get("topology") not in TOPOLOGY_LABEL:
            continue
        if r.get("failed"):
            say(f"  {stem}: skipped failed run {r['run']}")
            continue
        if not r.get("judged"):
            say(f"  {stem}: skipped {r['run']}: no judged agents")
            continue
        g = groups[(r["control"], r["topology"])]
        g["k"] += r["cheated"]
        g["n"] += r["judged"]
        g["runs"].append(r["run"])
    if not groups:
        say(f"skipped {stem}: no judged lone-agent or board runs")
        note_stale(out, stem, formats)
        return None
    controls = sorted({c for c, _ in groups}, key=lambda c: (c != "baseline", c))
    topos = [t for t in ("single", "board") if any(k[1] == t for k in groups)]
    fig, ax = plt.subplots(figsize=(6.8, 1.1 + 0.36 * len(controls)))
    ypos = {c: len(controls) - 1 - i for i, c in enumerate(controls)}
    # lone agent: filled dot; board: ring; a line joins the pair. Counts sit in a column on the right.
    col_x = {"single": 1.05, "board": 1.16} if len(topos) == 2 else {topos[0]: 1.07}
    rtrans = blended_transform_factory(ax.transAxes, ax.transData)
    for t in topos:
        ax.text(col_x[t], len(controls) - 0.45, TOPOLOGY_LABEL[t], transform=rtrans, ha="center", va="bottom",
                fontsize=7, color=MUTED)
    for c in controls:
        pts = [(topo, groups[(c, topo)]) for topo in topos if (c, topo) in groups]
        if len(pts) == 2:
            ax.plot([p["k"] / p["n"] for _, p in pts], [ypos[c]] * 2, color="#b0b0b0", linewidth=1.4, zorder=1)
        for topo, p in pts:
            ax.plot(p["k"] / p["n"], ypos[c], linestyle="none", zorder=3 if topo == "single" else 2,
                    **TOPOLOGY_MARK[topo])
        for t in topos:
            g = groups.get((c, t))
            ax.text(col_x[t], ypos[c], f"{g['k']}/{g['n']}" if g else "–", transform=rtrans, ha="center",
                    va="center", fontsize=7, color=INK if g else MUTED)
    ax.set_yticks([ypos[c] for c in controls])
    ax.set_yticklabels(controls)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(-0.04, 1.04)
    ax.set_xticks(np.linspace(0, 1, 5))
    ax.set_ylim(-0.6, len(controls) - 0.4)
    ax.xaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlabel("share of judged agents who cheated (right: cheated/judged)")
    ax.set_title("Cheating under each control, " + " and ".join(TOPOLOGY_LABEL[t] for t in topos), loc="left",
                 pad=20)
    handles = [Line2D([], [], linestyle="none", label=TOPOLOGY_LABEL[t], **TOPOLOGY_MARK[t]) for t in topos]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, borderaxespad=0.2,
              handletextpad=0.3, columnspacing=1.2)
    axes_note(ax, left_out_note([], data.unreadable_of(("single", "board"))), below_pt=32, width=110)
    say(f"{stem}: {len(controls)} controls, {len(groups)} cheated/judged values")
    for (c, t), g in sorted(groups.items()):
        say(f"  {c} / {TOPOLOGY_LABEL[t]}: cheated {g['k']} of {g['n']} judged ({', '.join(g['runs'])})")
    return save(fig, out, stem, formats)


# -- CLI -----------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs", help="directory of run directories")
    ap.add_argument("--out", default="docs/figures", help="output directory")
    ap.add_argument("--swimlane", default=None,
                    help="run for fig_swimlane (default: the most informative usable swarm run, else gemini_board)")
    ap.add_argument("--arrows", default="auto", choices=("auto", "all", "exposure"),
                    help=f"swimlane reader arrows: all, only reads before the reader's first archive read, "
                         f"or auto (all up to {AUTO_ARROWS}, else exposure)")
    ap.add_argument("--format", default="pdf,png", help="comma-separated formats (pdf, png, svg)")
    args = ap.parse_args(argv)
    formats = [f.strip().lower() for f in args.format.split(",") if f.strip()]
    bad = [f for f in formats if f not in SAVE_META]
    if bad or not formats:
        ap.error(f"unsupported format(s): {', '.join(bad) or '(none)'}; choose from {', '.join(SAVE_META)}")
    runs_dir = Path(args.runs).resolve()
    if not runs_dir.is_dir():
        ap.error(f"runs directory not found: {runs_dir}")
    if args.swimlane and args.swimlane not in Store(runs_dir).run_names():
        ap.error(f"--swimlane: run {args.swimlane} not found under {runs_dir}")
    out = Path(args.out)
    data = Data(runs_dir)
    say(f"read {len(data.rows)} runs from {runs_dir}"
        + (f"; could not read {', '.join(u['run'] for u in data.unreadable)}" if data.unreadable else ""))
    say(f"{len(data.swarm)} usable swarm runs: {', '.join(r['run'] for r in data.swarm) or '(none)'}")
    for name, why in data.failed:
        say(f"  left out swarm run {name}: {why}")
    written = {}
    for name, fn in (("fig_composition_heatmap", lambda: fig_composition_heatmap(data, out, formats)),
                     ("fig_role_funnel", lambda: fig_role_funnel(data, out, formats)),
                     ("fig_adoption_curves", lambda: fig_adoption_curves(data, out, formats)),
                     ("fig_swimlane", lambda: fig_swimlane(data, args.swimlane, args.arrows, out, formats)),
                     ("fig_controls", lambda: fig_controls(data, out, formats))):
        written[name] = fn()
    say("summary:")
    for name, paths in written.items():
        say(f"  {name}: {'written' if paths else 'skipped'}")
    if data.problems:
        say(f"exit status 1: {plural(len(data.problems), 'problem')}; the figures leave out what could not be "
            "read or checked:")
        for p in data.problems:
            say(f"  - {p}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
