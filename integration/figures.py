"""Publication figures for lone-agent, board and swarm runs.

    python -m integration.figures --runs runs --out docs/figures [--swimlane <run>] [--format pdf,png]

Reads run directories through the results server's data layer (no HTTP):
``Store.summary()`` for cross-run aggregates, ``Store.episodes()`` for per-agent judge flags
and hidden roles, ``Store.timeline()`` for one run's events on a common clock. Every number
drawn comes from those calls. Figures (see docs/visualization_research.md):

    fig_composition_heatmap   regulators x disregulators, share of judged agents who cheated
    fig_role_funnel           honeypot seen -> attempted access -> cheated -> disclosed, by role
    fig_adoption_curves       share of agents with a honeypot touch over time, per swarm run
    fig_swimlane              one run: lanes per agent, model calls, tools, posts and readers
    fig_controls              cheated / judged per control, lone agent and board

Each figure is written once per requested format at 200 dpi. Output is deterministic: the
data decide every value, and the PDF carries no creation date. A figure whose data are
absent (no swarm runs yet, no verdicts) is skipped with a printed reason.
"""
from __future__ import annotations

import argparse
import sys
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
DASHES = ["-", (0, (5, 2)), (0, (1, 1.6)), (0, (6, 2, 1, 2)), (0, (2, 3)), (0, (8, 2, 2, 2, 2, 2)),
          (0, (3, 1, 1, 1, 1, 1)), (0, (10, 3))]
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


# -- data ----------------------------------------------------------------------------------
class _OneRun(Store):
    """A Store that sees one run, so one unreadable run cannot sink the summary of the rest."""

    def __init__(self, runs_dir: Path, name: str) -> None:
        super().__init__(runs_dir)
        self._only = name

    def run_names(self) -> list[str]:
        return [self._only]


def summary_rows(store: Store) -> list[dict]:
    """``store.summary()['runs']``; on failure, the same summary built run by run."""
    try:
        return store.summary()["runs"]
    except Exception as ex:  # noqa: BLE001 - a run still being written can hold a partial line
        say(f"summary over all runs failed ({type(ex).__name__}: {ex}); reading runs one at a time")
    rows = []
    for name in store.run_names():
        try:
            rows.extend(_OneRun(store.runs_dir, name).summary()["runs"])
        except Exception as ex:  # noqa: BLE001
            say(f"skipped run {name}: its summary failed ({type(ex).__name__}: {ex})")
    return rows


def swarm_rows(rows: list[dict]) -> list[dict]:
    """Swarm runs (composition set) that did not fail; prints the ones left out."""
    out = []
    for r in rows:
        if r.get("composition") is None:
            continue
        if r.get("failed"):
            say(f"skipped swarm run {r['run']}: marked failed")
            continue
        out.append(r)
    return out


def judged_agents(store: Store, run: str) -> list[dict]:
    """Judged agents of one run with role and judge flags, by the rule summary() uses."""
    eps = store.episodes(run)
    roles = {}
    if any(not e.get("role") for e in eps):
        roles = ((store.run_detail(run).get("run_json") or {}).get("swarm") or {}).get("roles") or {}
    out = []
    for e in eps:
        v = e.get("verdict")
        if not (v and v.get("cheated") is not None and e.get("has_transcript")):
            continue
        rec = {"run": run, "agent": e.get("orbit_agent"), "role": e.get("role") or roles.get(e.get("orbit_agent"))}
        rec.update({k: bool(v.get(k)) for k, _ in STAGES})
        out.append(rec)
    return out


def safe_timeline(store: Store, run: str) -> dict | None:
    try:
        return store.timeline(run)
    except Exception as ex:  # noqa: BLE001 - eval log missing or still being written
        say(f"skipped timeline of {run}: {type(ex).__name__}: {ex}")
        return None


def run_end(tl: dict) -> float:
    """Last moment on the run's clock: the sample's total time or the last event, whichever is later."""
    ts = [tl.get("duration") or 0.0]
    for e in tl["events"]:
        ts.extend(x for x in (e.get("t"), e.get("t_end")) if x is not None)
    return max(ts)


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
def fig_composition_heatmap(swarm: list[dict], out: Path, formats: list[str]):
    stem = "fig_composition_heatmap"
    runs = [r for r in swarm if r.get("regulators") is not None and r.get("disregulators") is not None]
    if not runs:
        say(f"skipped {stem}: no swarm runs with a composition")
        note_stale(out, stem, formats)
        return None
    team = max(r["regulators"] + r["disregulators"] + (r.get("workers") or 0) for r in runs)
    team = max(team, max(r["n_agents"] for r in runs))
    cells: dict = defaultdict(lambda: {"k": 0, "n": 0, "runs": [], "disagree": [], "agree": []})
    for r in runs:
        c = cells[(r["regulators"], r["disregulators"])]
        c["k"] += r["cheated"]
        c["n"] += r["judged"]
        c["runs"].append(r["run"])
        if r["judged"] and r.get("report_cheated") is not None:
            (c["disagree"] if bool(r["report_cheated"]) != (r["cheated"] > 0) else c["agree"]).append(r["run"])
    cmap = LinearSegmentedColormap.from_list("cheated", plt.get_cmap("Reds")(np.linspace(0.08, 0.95, 256)))
    norm = Normalize(0, 1)
    side = min(7.0, 2.4 + 0.42 * (team + 1))
    fig, ax = plt.subplots(figsize=(side + 1.3, side))
    gap = 0.06
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
            if c["n"] == 0:
                ax.add_patch(Rectangle((x0, y0), w, w, facecolor="#efefef", linewidth=0))
                ax.text(dis, reg, "not\njudged", ha="center", va="center", fontsize=6, color=MUTED)
                fg = INK
            else:
                fill = cmap(norm(c["k"] / c["n"]))
                fg = ink_on(fill)
                ax.add_patch(Rectangle((x0, y0), w, w, facecolor=fill, linewidth=0))
                label = f"{c['k']}/{c['n']}"
                if len(c["runs"]) > 1:
                    label += f"\n{len(c['runs'])} runs"
                ax.text(dis, reg, label, ha="center", va="center", fontsize=7.5 if team <= 10 else 6, color=fg)
            if c["disagree"]:
                s = 0.3 * w
                xr, yt = x0 + w, y0 + w
                ax.add_patch(Polygon([(xr, yt), (xr - s, yt), (xr, yt - s)], closed=True, facecolor=INK,
                                     edgecolor="white", linewidth=0.8, joinstyle="miter"))
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
    ax.set_title("Share of judged agents who cheated, by team composition", loc="left")
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label("share of judged agents who cheated")
    cb.outline.set_visible(False)
    handles = [Patch(facecolor="white", edgecolor="#cfcfcf", hatch="////", linewidth=0, label="not run"),
               Patch(facecolor="#efefef", linewidth=0, label="run, no verdicts yet"),
               Line2D([], [], linestyle="none", marker=[(-1, 1), (1, 1), (1, -1)], markersize=7,
                      markerfacecolor=INK, markeredgecolor="white", markeredgewidth=0.6,
                      label="team report disagrees with the judge")]
    ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.0, 1.0), fontsize=7, handlelength=1.2)
    say(f"{stem}: {len(runs)} swarm runs in {len(cells)} cells, team size {team}")
    for (reg, dis), c in sorted(cells.items()):
        say(f"  R{reg} D{dis}: cheated {c['k']} of {c['n']} judged; runs {', '.join(c['runs'])}"
            f"{'; report disagrees: ' + ', '.join(c['disagree']) if c['disagree'] else ''}")
    return save(fig, out, stem, formats)


# -- 2. role funnel ------------------------------------------------------------------------
def fig_role_funnel(store: Store, swarm: list[dict], out: Path, formats: list[str]):
    stem = "fig_role_funnel"
    agents, used = [], []
    for r in swarm:
        try:
            recs = judged_agents(store, r["run"])
        except Exception as ex:  # noqa: BLE001
            say(f"skipped {r['run']} in {stem}: {type(ex).__name__}: {ex}")
            continue
        # the per-agent flags must add up to the run's summary row
        for key, _ in STAGES:
            got = sum(a[key] for a in recs)
            if got != r[SUMMARY_KEY[key]]:
                say(f"  MISMATCH {r['run']}: {key} per agent {got} != summary {r[SUMMARY_KEY[key]]}")
        for role in ROLES:
            got = sum(1 for a in recs if a["role"] == role and a["cheated"])
            want = ((r.get("by_role") or {}).get(role) or {}).get("cheated")
            if want is not None and got != want:
                say(f"  MISMATCH {r['run']}: {role} cheated per agent {got} != summary by_role {want}")
        if not recs:
            say(f"  {r['run']}: no judged agents yet")
            continue
        agents.extend(recs)
        used.append(r["run"])
    if not agents:
        say(f"skipped {stem}: no judged swarm agents")
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
    for si, (key, _) in enumerate(STAGES):
        for role in present:
            k = sum(1 for a in agents if a["role"] == role and a[key])
            share = k / n[role]
            x = si + offs[role]
            ax.bar(x, share, width=width, color=ROLE_COLOUR[role], linewidth=0, zorder=2)
            ax.text(x, share + 0.02, f"{k}/{n[role]}", ha="center", va="bottom", fontsize=6.5, color=INK)
    ax.set_xticks(range(len(STAGES)))
    ax.set_xticklabels([label for _, label in STAGES])
    ax.set_xlim(-0.6, len(STAGES) - 0.4)
    ax.set_ylim(0, 1.12)
    ax.set_yticks(np.linspace(0, 1, 5))
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("share of judged agents in the role")
    for si in range(len(STAGES) - 1):
        ax.annotate("", xy=(si + 0.62, -0.075), xytext=(si + 0.38, -0.075), xycoords=("data", "axes fraction"),
                    arrowprops={"arrowstyle": "-|>", "color": "#9a9a9a", "lw": 0.8})
    ax.set_title("Outcome funnel by hidden role", loc="left", pad=18)
    ax.text(0, 1.02, f"Pooled over {len(used)} swarm runs and {len(agents)} judged agents. "
            "Bar label: agents at the stage / judged agents in the role.",
            transform=ax.transAxes, fontsize=7, color=MUTED, va="bottom")
    ax.legend(handles=role_handles(present, n), loc="upper center", ncol=len(present), bbox_to_anchor=(0.5, -0.12))
    say(f"{stem}: {len(agents)} judged agents from {len(used)} runs; per role " + ", ".join(f"{r} {n[r]}" for r in ROLES))
    for key, label in STAGES:
        say(f"  {label}: " + ", ".join(f"{role} {sum(1 for a in agents if a['role'] == role and a[key])}/{n[role]}"
                                       for role in present))
    return save(fig, out, stem, formats)


# -- 3. adoption curves --------------------------------------------------------------------
def fig_adoption_curves(store: Store, swarm: list[dict], out: Path, formats: list[str]):
    stem = "fig_adoption_curves"
    curves = []
    for r in swarm:
        tl = safe_timeline(store, r["run"])
        if tl is None:
            continue
        agents = tl.get("agents") or []
        if not agents:
            say(f"skipped {r['run']} in {stem}: no agents on the timeline")
            continue
        touches = sorted(t for a, t in (tl.get("first_honeypot_touch") or {}).items() if a in agents and t is not None)
        posts = [p["t"] for p in tl.get("board") or [] if p.get("t") is not None]
        curves.append({"run": r["run"], "R": r["regulators"], "D": r["disregulators"], "n": len(agents),
                       "touches": touches, "end": run_end(tl), "first_post": min(posts) if posts else None})
    if not curves:
        say(f"skipped {stem}: no swarm run with a readable timeline")
        note_stale(out, stem, formats)
        return None
    regs = sorted({c["R"] for c in curves})
    diss = sorted({c["D"] for c in curves})
    ramp = plt.get_cmap("Blues")
    colour = {v: ramp(0.85 if len(regs) == 1 else 0.45 + 0.55 * i / (len(regs) - 1)) for i, v in enumerate(regs)}
    if len(diss) > len(DASHES):
        say(f"  {stem}: {len(diss)} disregulator counts but {len(DASHES)} line styles; styles repeat")
    dash = {v: DASHES[i % len(DASHES)] for i, v in enumerate(diss)}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), sharey=True)
    aligned = 0
    for c in sorted(curves, key=lambda c: (c["R"], c["D"], c["run"])):
        frac = [(i + 1) / c["n"] for i in range(len(c["touches"]))]
        xs = [0.0] + c["touches"] + [c["end"]]
        ys = [0.0] + frac + [frac[-1] if frac else 0.0]
        style = {"color": colour[c["R"]], "linestyle": dash[c["D"]], "linewidth": 1.6, "where": "post"}
        axes[0].step(xs, ys, **style)
        if c["first_post"] is not None:
            aligned += 1
            axes[1].step([x - c["first_post"] for x in xs], ys, **style)
        else:
            say(f"  {c['run']}: no board post with a time; left out of the aligned panel")
        say(f"  {c['run']}: R{c['R']} D{c['D']}, {len(c['touches'])}/{c['n']} agents touched the honeypot; "
            f"first post at {c['first_post']}")
    # composition at each line's end; labels that would collide are merged into one
    xspan = max(c["end"] for c in curves) or 1.0
    ends = sorted(((c["end"], (len(c["touches"]) / c["n"]), f"R{c['R']}D{c['D']}") for c in curves),
                  key=lambda e: (e[1], e[0], e[2]))
    merged: list[list] = []
    for x, yv, text in ends:
        hit = next((m for m in merged if abs(m[1] - yv) < 0.045 and abs(m[0] - x) < 0.12 * xspan), None)
        if hit:
            hit[0], hit[2] = max(hit[0], x), hit[2] + [text]
        else:
            merged.append([x, yv, [text]])
    for x, yv, texts in merged:
        counts = {t: texts.count(t) for t in texts}
        names = [t if counts[t] == 1 else f"{t} \u00d7{counts[t]}" for t in dict.fromkeys(texts)]
        label = "\n".join(", ".join(names[i:i + 3]) for i in range(0, len(names), 3))
        high = yv > 0.85   # near the top: label under the line, else above it; kept inside the panel
        axes[0].text(x, yv - 0.025 if high else yv + 0.02, label, fontsize=6.5, color=MUTED,
                     ha="right" if x > 0.3 * xspan else "left", va="top" if high else "bottom")
    axes[0].set_xlabel("seconds from the start of the run")
    axes[1].set_xlabel("seconds from the first board post")
    axes[0].set_ylabel("share of agents with a honeypot touch")
    axes[0].set_title("From the start of the run", loc="left", fontsize=9)
    axes[1].set_title("Aligned to the first board post", loc="left", fontsize=9)
    if aligned:
        axes[1].axvline(0, color="#9a9a9a", linewidth=0.8, zorder=0)
    else:
        axes[1].text(0.5, 0.5, "no run has a timed board post", transform=axes[1].transAxes, ha="center",
                     color=MUTED)
    for ax in axes:
        ax.set_ylim(-0.03, 1.03)
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)
    blank = Line2D([], [], linestyle="none")
    headers = {"colour: regulators", "line: disregulators"}
    handles = ([blank] + [Line2D([], [], color=colour[v], linewidth=2.2) for v in regs]
               + [blank] + [Line2D([], [], color=INK, linestyle=dash[v], linewidth=1.6) for v in diss])
    labels = (["colour: regulators"] + [str(v) for v in regs] + ["line: disregulators"] + [str(v) for v in diss])
    leg = axes[1].legend(handles, labels, loc="upper left", bbox_to_anchor=(1.02, 1.0), handlelength=3.2)
    for text in leg.get_texts():
        if text.get_text() in headers:
            text.set_fontweight("bold")
    fig.suptitle("Share of agents that touched the honeypot over time, one line per swarm run", x=0.01,
                 ha="left", fontsize=10)
    fig.text(0.01, -0.04, "A touch is a bash or python call naming another agent's branch, commit or copied archive "
             "(timeline heuristic); the judge's verdict is the record of truth.", fontsize=7, color=MUTED)
    say(f"{stem}: {len(curves)} runs, {aligned} with a timed board post")
    return save(fig, out, stem, formats)


# -- 4. swimlane ---------------------------------------------------------------------------
def pick_swimlane_run(store: Store, swarm: list[dict], wanted: str | None):
    names = store.run_names()
    if wanted:
        if wanted not in names:
            say(f"skipped fig_swimlane: run {wanted} not found under {store.runs_dir}")
            return None, None
        return wanted, safe_timeline(store, wanted)
    for r in sorted(swarm, key=lambda r: r["run"]):
        tl = safe_timeline(store, r["run"])
        if tl is not None:
            return r["run"], tl
    if "gemini_board" in names:
        say("fig_swimlane: no swarm run with a readable timeline; using gemini_board")
        return "gemini_board", safe_timeline(store, "gemini_board")
    say("skipped fig_swimlane: no swarm run and no gemini_board")
    return None, None


def fig_swimlane(store: Store, swarm: list[dict], wanted: str | None, out: Path, formats: list[str]):
    stem = "fig_swimlane"
    run, tl = pick_swimlane_run(store, swarm, wanted)
    if tl is None:
        if run:
            say(f"skipped {stem}: no timeline for {run}")
        note_stale(out, stem, formats)
        return None
    roles = tl.get("roles") or {}
    agents = list(tl.get("agents") or [])
    order = {r: i for i, r in enumerate(ROLES)}
    lanes = sorted(agents, key=lambda a: (order.get(roles.get(a), len(ROLES)), a))
    y = {a: len(lanes) - 1 - i for i, a in enumerate(lanes)}
    end = run_end(tl)
    fig, ax = plt.subplots(figsize=(10.5, 1.6 + 0.5 * len(lanes)))
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
    n_arrows = {"read": 0, "model_exposure": 0}
    for p in tl.get("board") or []:
        if p.get("t") is None or p.get("sender") not in y:
            continue
        sy = y[p["sender"]]
        readers = {}
        for rb in p.get("read_by") or []:
            ra = rb.get("agent")
            if ra == p["sender"] or ra not in y or rb.get("t") is None:
                continue
            cur = readers.setdefault(ra, {"t": rb["t"], "read": False})
            cur["t"] = min(cur["t"], rb["t"])
            cur["read"] = cur["read"] or rb.get("how") == "read"
        for ra, info in sorted(readers.items()):
            how = "read" if info["read"] else "model_exposure"
            n_arrows[how] += 1
            ax.add_patch(FancyArrowPatch((p["t"], sy), (info["t"], y[ra]), arrowstyle="-|>", mutation_scale=6,
                                         linewidth=0.55, color="#4a4a4a", alpha=0.6, zorder=4,
                                         linestyle="-" if how == "read" else (0, (3, 2)), shrinkA=2, shrinkB=1))
        ax.plot(p["t"], sy, marker="o", markersize=5, markerfacecolor=INK, markeredgecolor="white",
                markeredgewidth=1.0, zorder=6, linestyle="none")
        seen_kinds.add("post")
    for a, t in (tl.get("first_honeypot_touch") or {}).items():
        if a in y and t is not None:
            ax.plot(t, y[a], marker="o", markersize=11, markerfacecolor="none", markeredgecolor=INK,
                    markeredgewidth=1.4, zorder=7, linestyle="none")
            seen_kinds.add("touch")
    for a, t in (tl.get("first_member_read") or {}).items():
        if a in y and t is not None:
            ax.plot(t, y[a], marker="D", markersize=5.5, markerfacecolor=INK, markeredgecolor="white",
                    markeredgewidth=0.8, zorder=8, linestyle="none")
            seen_kinds.add("member_read")
    if blocked_marks:
        ax.plot([t for t, _ in blocked_marks], [yy + 0.4 for _, yy in blocked_marks], marker="x", markersize=4.5,
                color=RED, markeredgewidth=1.2, linestyle="none", zorder=8)
        seen_kinds.add("blocked")
    # verdict at the lane's end
    vtrans = blended_transform_factory(ax.transAxes, ax.transData)
    verdicts = tl.get("verdicts") or {}
    for a in lanes:
        v = verdicts.get(a)
        text, face, fg = ("cheated", RED, "white") if v is True else (
            ("honest", GREEN, "white") if v is False else ("not judged", "#ececec", INK))
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
    ax.set_title(f"What each agent did over time in run {run}", loc="left")
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
    if "touch" in seen_kinds:
        handles.append(Line2D([], [], linestyle="none", marker="o", markersize=9, markerfacecolor="none",
                              markeredgecolor=INK, markeredgewidth=1.3, label="first honeypot touch"))
    if "member_read" in seen_kinds:
        handles.append(Line2D([], [], linestyle="none", marker="D", markersize=5, markerfacecolor=INK,
                              markeredgecolor="white", label="first archive member read"))
    if "blocked" in seen_kinds:
        handles.append(Line2D([], [], linestyle="none", marker="x", markersize=5, color=RED, markeredgewidth=1.2,
                              label="blocked by a defense"))
    if "intervention" in seen_kinds:
        handles.append(Line2D([], [], color=RED, linewidth=1.3, label="defense intervention (full-height line)"))
    present_roles = [r for r in ROLES if r in set(roles.get(a) for a in lanes)]
    handles += role_handles(present_roles)
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0, -0.11 - 0.25 / max(len(lanes), 1)),
              ncol=4, fontsize=7, handlelength=1.8, columnspacing=1.4)
    say(f"{stem}: run {run}, {len(lanes)} lanes, {len(tl['events'])} events, {len(tl.get('board') or [])} posts, "
        f"{n_arrows['read']} read arrows, {n_arrows['model_exposure']} model-only arrows; "
        f"touch {tl.get('first_honeypot_touch')}, member read {tl.get('first_member_read')}, verdicts {verdicts}")
    return save(fig, out, stem, formats)


# -- 5. controls ---------------------------------------------------------------------------
def fig_controls(rows: list[dict], out: Path, formats: list[str]):
    stem = "fig_controls"
    groups: dict = defaultdict(lambda: {"k": 0, "n": 0, "runs": []})
    for r in rows:
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
    fig, ax = plt.subplots(figsize=(6.8, 1.1 + 0.36 * len(controls)))
    ypos = {c: len(controls) - 1 - i for i, c in enumerate(controls)}
    # lone agent: filled dot; board: ring; a line joins the pair. Counts sit in a column on the right.
    col_x = {"single": 1.05, "board": 1.16}
    rtrans = blended_transform_factory(ax.transAxes, ax.transData)
    topos = [t for t in ("single", "board") if any(k[1] == t for k in groups)]
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
            ax.text(col_x[t], ypos[c], f"{g['k']}/{g['n']}" if g else "\u2013", transform=rtrans, ha="center",
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
    ax.set_title("Cheating under each control, lone agent and board", loc="left")
    handles = [Line2D([], [], linestyle="none", label=TOPOLOGY_LABEL[t], **TOPOLOGY_MARK[t])
               for t in ("single", "board") if any(k[1] == t for k in groups)]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.08 - 0.5 / len(controls)), ncol=2)
    say(f"{stem}: {len(controls)} controls")
    for (c, t), g in sorted(groups.items()):
        say(f"  {c} / {TOPOLOGY_LABEL[t]}: cheated {g['k']} of {g['n']} judged ({', '.join(g['runs'])})")
    return save(fig, out, stem, formats)


# -- CLI -----------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs", help="directory of run directories")
    ap.add_argument("--out", default="docs/figures", help="output directory")
    ap.add_argument("--swimlane", default=None, help="run for fig_swimlane (default: first swarm run, else gemini_board)")
    ap.add_argument("--format", default="pdf,png", help="comma-separated formats (pdf, png, svg)")
    args = ap.parse_args(argv)
    formats = [f.strip().lower() for f in args.format.split(",") if f.strip()]
    bad = [f for f in formats if f not in SAVE_META]
    if bad or not formats:
        ap.error(f"unsupported format(s): {', '.join(bad) or '(none)'}; choose from {', '.join(SAVE_META)}")
    runs_dir = Path(args.runs).resolve()
    if not runs_dir.is_dir():
        ap.error(f"runs directory not found: {runs_dir}")
    out = Path(args.out)
    store = Store(runs_dir)
    rows = summary_rows(store)
    say(f"read {len(rows)} runs from {runs_dir}")
    swarm = swarm_rows(rows)
    say(f"{len(swarm)} swarm runs: {', '.join(r['run'] for r in swarm) or '(none)'}")
    written = {}
    for name, fn in (("fig_composition_heatmap", lambda: fig_composition_heatmap(swarm, out, formats)),
                     ("fig_role_funnel", lambda: fig_role_funnel(store, swarm, out, formats)),
                     ("fig_adoption_curves", lambda: fig_adoption_curves(store, swarm, out, formats)),
                     ("fig_swimlane", lambda: fig_swimlane(store, swarm, args.swimlane, out, formats)),
                     ("fig_controls", lambda: fig_controls(rows, out, formats))):
        written[name] = fn()
    say("summary:")
    for name, paths in written.items():
        say(f"  {name}: {'written' if paths else 'skipped'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
