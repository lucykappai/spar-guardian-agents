"""Export the results browser as a static bundle: the page with the UI inlined and every API response as a file.

    python -m integration.ui.export [--runs runs] [--out integration/build/share] [--allow-errors]

The bundle needs no server. It is written into a fresh directory that replaces --out on success, so nothing from an
earlier export survives. Layout:

    index.html              the complete page: serve the directory over HTTP (python -m http.server 8000 in it, then
                            http://127.0.0.1:8000/); opened from disk, the browser refuses the data requests
    artifact.html           the same page without the document skeleton, for a publisher that adds its own (a
                            claude.ai artifact): <title>, the stylesheet, the app and the script
    data/runs.json          /api/runs
    data/summary.json       /api/summary
    data/figures.json       /api/figures
    data/run/<run>/run.json          /api/runs/<run>
    data/run/<run>/sample.json       /api/runs/<run>/sample
    data/run/<run>/context.json      /api/runs/<run>/context
    data/run/<run>/timeline.json     /api/runs/<run>/timeline
    data/run/<run>/transcripts.json  {<episode>: /api/runs/<run>/transcript/<episode>} for every episode with one
    figures/<png>           docs/figures/*.png
    manifest.json           every file with its size and SHA-256, the counts, the source hashes, what was left out

An endpoint that would have answered 404 or 500 is stored in its place as {"__error": {"status", "body"}}, and the
export then exits 1 unless --allow-errors: a bundle built without the eval logs would otherwise look complete.
Left out on purpose: /api/runs/<run>/context/<i>, the exact provider input of one model call (about 200 KB each,
hundreds per run). The context page states that and points at the local browser.
The stylesheet gains the [data-theme] variants an artifact viewer sets (an explicit light or dark choice) from its
prefers-color-scheme blocks, so the three viewer states (system, light, dark) all read the right tokens.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from integration.ui.server import FIGURES, STATIC, Store, figures_list  # noqa: E402

TITLE = "ORBIT x CheatBench results"
OMITTED = "/api/runs/<run>/context/<i>: the exact provider input of one model call is read from the eval log and is not exported"
DARK_BLOCK = re.compile(r"@media \(prefers-color-scheme: dark\) \{\n  :root \{\n(.*?)\n  \}\n\}", re.S)
# The publisher's own skeleton, read back from a published copy on 2026-10-12: index.html is the page exactly as a
# viewer of the artifact gets it, so a local check of index.html is a check of the published page.
SKELETON_HEAD = (
    '<!doctype html><html><head><meta charset=utf8><meta name=viewport content="width=device-width,initial-scale=1,'
    'viewport-fit=cover"><style>:root{color-scheme:light;box-sizing:border-box;padding-top:env(safe-area-inset-top,0px);'
    'padding-bottom:env(safe-area-inset-bottom,0px)}html{scroll-padding-top:env(safe-area-inset-top,0px)}'
    'body{margin:0;padding:0;font:14px -apple-system,BlinkMacSystemFont,sans-serif;background:#faf9f5;color:#141413}'
    'img{max-width:100%}[hidden]:not([hidden=until-found i]){display:none!important}</style></head><body>\n')
SKELETON_TAIL = "\n</body></html>"
SOURCES = ("static/app.js", "static/style.css", "export.py")


def themed_css(css: str) -> str:
    """Add the [data-theme] variants: the dark tokens apply under the system preference unless the viewer chose
    light, and always when the viewer chose dark."""
    def repl(m):
        body = m.group(1)
        return ('@media (prefers-color-scheme: dark) {\n  :root:not([data-theme="light"]) {\n' + body + '\n  }\n}\n'
                ':root[data-theme="dark"] {\n' + body.replace("\n    ", "\n  ")[2:] + '\n}')
    out, n = DARK_BLOCK.subn(repl, css)
    if n != 2:
        raise RuntimeError(f"expected 2 dark token blocks in style.css, found {n}")
    if (out.count("@media (prefers-color-scheme: dark)") != 2 or out.count('[data-theme="dark"]') != 2
            or out.count('[data-theme="light"]') != 2):
        raise RuntimeError("the themed stylesheet does not have the three-state structure")
    return out


def page_body(css: str, js: str, config: dict) -> str:
    """The page without a document skeleton: the publisher adds doctype, head and body."""
    if "</style" in css.lower():
        raise RuntimeError("style.css contains '</style'")
    if "</script" in js.lower() or "<!--" in js:
        raise RuntimeError("app.js contains '</script' or '<!--', which would end the inline script")
    cfg = json.dumps(config).replace("<", "\\u003c")
    return (f"<title>{TITLE}</title>\n<style>\n{css}\n</style>\n"
            '<div id="app">Loading… (if this stays, the page script did not run)</div>\n'
            f"<script>window.RESULTS_STATIC = {cfg};</script>\n"
            f"<script>\n{js}\n</script>\n")


def full_page(fragment: str) -> str:
    return SKELETON_HEAD + fragment + SKELETON_TAIL


def dumps(obj) -> str:
    return json.dumps(obj, default=str)        # the server's encoding (server.py _send_json)


def endpoint(fn, *args):
    """What the server would have answered: the value, or an error record in its place."""
    try:
        return fn(*args)
    except KeyError as ex:
        return {"__error": {"status": 404, "body": {"error": f"not found: {ex}"}}}
    except Exception as ex:  # noqa: BLE001 - the bundle records the fault where the server would have
        return {"__error": {"status": 500, "body": {"error": f"{type(ex).__name__}: {ex}"}}}


def is_error(v) -> bool:
    return isinstance(v, dict) and "__error" in v


def git_stamp() -> str | None:
    """The commit the sources come from, with -dirty when the working tree differs from it."""
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "describe", "--always", "--dirty"], text=True).strip()
    except Exception:  # noqa: BLE001
        return None


def source_hashes() -> dict:
    here = Path(__file__).resolve().parent
    return {name: hashlib.sha256((here / name).read_bytes()).hexdigest() for name in SOURCES}


def export(runs_dir: Path, out: Path) -> dict:
    store = Store(runs_dir.resolve())
    tmp = out.parent / (out.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    (tmp / "data" / "run").mkdir(parents=True)
    (tmp / "figures").mkdir()
    generated = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    written: list[Path] = []

    def write(rel: str, text: str) -> None:
        p = tmp / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        written.append(p)

    names = store.run_names()
    write("data/runs.json", dumps([store.run_summary(n) for n in names]))
    write("data/summary.json", dumps(store.summary()))
    write("data/figures.json", dumps(figures_list()))
    n_transcripts = 0
    errors: dict[str, list[str]] = {}
    for name in names:
        parts = {"run": endpoint(store.run_detail, name)}
        for key, fn in (("sample", store.sample), ("context", store.context_list), ("timeline", store.timeline)):
            parts[key] = endpoint(fn, name)
        transcripts = {}
        for ep in (e["episode"] for e in store.episodes(name) if e.get("has_transcript")):
            transcripts[ep] = endpoint(store.transcript, name, ep)
            if is_error(transcripts[ep]):
                errors.setdefault(name, []).append(f"transcript {ep}: {transcripts[ep]['__error']['status']}")
            else:
                n_transcripts += 1
        for key, v in parts.items():
            if is_error(v):
                errors.setdefault(name, []).append(f"{key}: {v['__error']['status']}")
            write(f"data/run/{name}/{key}.json", dumps(v))
        write(f"data/run/{name}/transcripts.json", dumps(transcripts))
        store.forget(name)
        print(f"  {name}: {len(transcripts)} transcripts" + (f"; ERRORS {', '.join(errors[name])}" if name in errors else ""),
              flush=True)
    for fig in sorted(FIGURES.glob("*.png")) if FIGURES.is_dir() else []:
        p = tmp / "figures" / fig.name
        p.write_bytes(fig.read_bytes())
        written.append(p)

    css = themed_css((STATIC / "style.css").read_text())
    js = (STATIC / "app.js").read_text()
    config = {"data": "data/", "figures": "figures/", "generated": generated, "git": git_stamp(), "omitted": OMITTED}
    fragment = page_body(css, js, config)
    write("artifact.html", fragment)
    write("index.html", full_page(fragment))

    files = []
    for p in sorted(written):
        data = p.read_bytes()
        files.append({"path": p.relative_to(tmp).as_posix(), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    manifest = {"generated": generated, "git": config["git"], "sources": source_hashes(), "runs": len(names),
                "transcripts": n_transcripts, "endpoint_errors": errors, "omitted": [OMITTED], "files": files,
                "total_bytes": sum(f["bytes"] for f in files)}
    (tmp / "manifest.json").write_text(json.dumps(manifest, indent=1))
    if out.exists():
        shutil.rmtree(out)
    tmp.rename(out)
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=str(ROOT / "runs"))
    ap.add_argument("--out", default=str(ROOT / "integration" / "build" / "share"))
    ap.add_argument("--allow-errors", action="store_true", help="exit 0 even when an endpoint was stored as an error record")
    args = ap.parse_args()
    out = Path(args.out)
    m = export(Path(args.runs), out)
    print(f"exported {m['runs']} runs, {m['transcripts']} transcripts, {len(m['files'])} files, "
          f"{m['total_bytes'] / 1e6:.1f} MB -> {out} (git {m['git']})")
    if m["endpoint_errors"]:
        n = sum(len(v) for v in m["endpoint_errors"].values())
        print(f"{n} endpoint(s) stored as error records in {len(m['endpoint_errors'])} run(s): "
              + "; ".join(f"{k}: {', '.join(v)}" for k, v in m["endpoint_errors"].items()), file=sys.stderr)
        if not args.allow_errors:
            print("refusing: pass --allow-errors to ship a bundle with error records", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
