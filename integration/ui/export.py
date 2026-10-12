"""Export the results browser as a static bundle: one page with the UI inlined and every API response as a file.

    python -m integration.ui.export [--runs runs] [--out integration/build/share]

The bundle runs with no server (a claude.ai artifact, any static host). Layout of --out:

    index.html          the page for the artifact publisher: <title>, the stylesheet, the app and the script,
                        with no document skeleton (the publisher wraps it)
    preview.html        the same page inside a full document, for a local check:
                        cd integration/build/share && python -m http.server 8799, then open /preview.html
    data/runs.json      /api/runs
    data/summary.json   /api/summary
    data/figures.json   /api/figures
    data/run/<run>.json {run: /api/runs/<run>, sample, context, timeline, transcripts: {<episode>: ...}}
                        (an endpoint that would have answered 404 or 500 is stored as {"__error": {status, body}})
    figures/<png>       docs/figures/*.png
    manifest.json       every file with its size and SHA-256, the counts, what was left out

Left out on purpose: /api/runs/<run>/context/<i>, the exact provider input of one model call (about 200 KB each,
hundreds per run). The page tells the reader so and points at the local browser.

The stylesheet gains the [data-theme] variants an artifact viewer sets (an explicit light or dark choice) from its
prefers-color-scheme blocks, so the three viewer states (system, light, dark) all read the right tokens.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
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
PREVIEW_HEAD = (
    '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
    '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
    "<style>:root{color-scheme:light;padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}"
    "body{margin:0;font:14px system-ui,sans-serif;background:#fafaf8}img{max-width:100%}[hidden]{display:none!important}</style>\n"
    "</head><body>\n")
PREVIEW_TAIL = "\n</body></html>\n"


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
    if out.count("@media (prefers-color-scheme: dark)") != 2 or out.count('[data-theme="dark"]') != 2 or out.count('[data-theme="light"]') != 2:
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


def dumps(obj) -> str:
    return json.dumps(obj, default=str)        # the server's encoding (server.py _send_json)


def endpoint(fn, *args):
    try:
        return fn(*args)
    except KeyError as ex:
        return {"__error": {"status": 404, "body": {"error": f"not found: {ex}"}}}
    except Exception as ex:  # noqa: BLE001 - the bundle records the fault where the server would have
        return {"__error": {"status": 500, "body": {"error": f"{type(ex).__name__}: {ex}"}}}


def git_head() -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return None


def export(runs_dir: Path, out: Path) -> dict:
    store = Store(runs_dir.resolve())
    out.mkdir(parents=True, exist_ok=True)
    (out / "data" / "run").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)
    generated = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    written: list[Path] = []

    def write(rel: str, text: str) -> None:
        p = out / rel
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
        bundle = {"run": store.run_detail(name), "transcripts": {}}
        for ep in bundle["run"]["episodes"]:
            if ep.get("has_transcript"):
                bundle["transcripts"][ep["episode"]] = store.transcript(name, ep["episode"])
                n_transcripts += 1
        for key, fn in (("sample", store.sample), ("context", store.context_list), ("timeline", store.timeline)):
            bundle[key] = endpoint(fn, name)
            if isinstance(bundle[key], dict) and "__error" in bundle[key]:
                errors.setdefault(name, []).append(f"{key}: {bundle[key]['__error']['status']}")
        write(f"data/run/{name}.json", dumps(bundle))
        store.forget(name)
        print(f"  {name}: {len(bundle['transcripts'])} transcripts" + (f"; {', '.join(errors[name])}" if name in errors else ""), flush=True)
    for fig in sorted(FIGURES.glob("*.png")) if FIGURES.is_dir() else []:
        p = out / "figures" / fig.name
        p.write_bytes(fig.read_bytes())
        written.append(p)

    css = themed_css((STATIC / "style.css").read_text())
    js = (STATIC / "app.js").read_text()
    config = {"data": "data/", "figures": "figures/", "generated": generated, "git": git_head(), "omitted": OMITTED}
    body = page_body(css, js, config)
    write("index.html", body)
    write("preview.html", PREVIEW_HEAD + body + PREVIEW_TAIL)

    files = []
    for p in sorted(written):
        data = p.read_bytes()
        files.append({"path": p.relative_to(out).as_posix(), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    manifest = {"generated": generated, "git": config["git"], "runs": len(names), "transcripts": n_transcripts,
                "endpoint_errors": errors, "omitted": [OMITTED], "files": files,
                "total_bytes": sum(f["bytes"] for f in files)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=str(ROOT / "runs"))
    ap.add_argument("--out", default=str(ROOT / "integration" / "build" / "share"))
    args = ap.parse_args()
    out = Path(args.out)
    m = export(Path(args.runs), out)
    print(f"exported {m['runs']} runs, {m['transcripts']} transcripts, {len(m['files'])} files, "
          f"{m['total_bytes'] / 1e6:.1f} MB -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
