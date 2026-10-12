"""Check a static export of the results browser. Exit 1 on any difference.

    python -m integration.checks.check_share [--runs runs] [--bundle integration/build/share]

Checks, each against the thing itself: every file in manifest.json exists with its recorded size and SHA-256;
data/runs.json, data/summary.json, data/figures.json and every data/run/<run>.json equal what Store answers now
for the same paths (the error records too); the figures equal docs/figures byte for byte; index.html starts with
the title, carries the stylesheet and the script verbatim, declares the static switch and has no document
skeleton and no external resource; no file exceeds the publisher's 16 MB limit and the bundle stays under 256 MB
and 511 files; no credential pattern appears in any text file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from integration.ui.export import endpoint, dumps, themed_css  # noqa: E402
from integration.ui.server import FIGURES, STATIC, Store, figures_list  # noqa: E402

MAX_FILE = 16 * 1024 * 1024
MAX_TOTAL = 256 * 1024 * 1024
MAX_FILES = 511
SECRETS = [re.compile(p) for p in (r"AIza[0-9A-Za-z_-]{35}", r"\bsk-[A-Za-z0-9_-]{20,}", r"\bsk-proj-[A-Za-z0-9_-]{20,}",
                                   r"(?i)\b(api[_-]?key|secret[_-]?key|authorization)\b\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{24,}")]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=str(ROOT / "runs"))
    ap.add_argument("--bundle", default=str(ROOT / "integration" / "build" / "share"))
    args = ap.parse_args()
    bundle = Path(args.bundle)
    store = Store(Path(args.runs).resolve())
    problems: list[str] = []
    bad = problems.append

    manifest = json.loads((bundle / "manifest.json").read_text())
    listed = {f["path"]: f for f in manifest["files"]}
    for rel, f in listed.items():
        p = bundle / rel
        if not p.is_file():
            bad(f"missing file {rel}")
            continue
        data = p.read_bytes()
        if len(data) != f["bytes"] or hashlib.sha256(data).hexdigest() != f["sha256"]:
            bad(f"{rel}: size or hash differs from the manifest")
        if len(data) > MAX_FILE:
            bad(f"{rel}: {len(data)} bytes exceeds the 16 MB limit")
    on_disk = {p.relative_to(bundle).as_posix() for p in bundle.rglob("*") if p.is_file()} - {"manifest.json"}
    for extra in sorted(on_disk - set(listed)):
        bad(f"not in the manifest: {extra}")
    total = sum(f["bytes"] for f in listed.values())
    if total > MAX_TOTAL:
        bad(f"bundle is {total} bytes, over 256 MB")
    if len(listed) > MAX_FILES:
        bad(f"bundle has {len(listed)} files, over {MAX_FILES}")

    def same(rel: str, expected_text: str) -> None:
        p = bundle / rel
        if not p.is_file():
            bad(f"missing {rel}")
        elif json.loads(p.read_text()) != json.loads(expected_text):
            bad(f"{rel} differs from the live data")

    names = store.run_names()
    same("data/runs.json", dumps([store.run_summary(n) for n in names]))
    same("data/summary.json", dumps(store.summary()))
    same("data/figures.json", dumps(figures_list()))
    n_transcripts = 0
    for name in names:
        rel = f"data/run/{name}.json"
        p = bundle / rel
        if not p.is_file():
            bad(f"missing {rel}")
            continue
        got = json.loads(p.read_text())
        if got.get("run") != json.loads(dumps(store.run_detail(name))):
            bad(f"{rel}: run differs")
        want_eps = {e["episode"] for e in store.episodes(name) if e.get("has_transcript")}
        if set(got.get("transcripts", {})) != want_eps:
            bad(f"{rel}: transcripts {sorted(got.get('transcripts', {}))} != {sorted(want_eps)}")
        for ep in want_eps & set(got.get("transcripts", {})):
            if got["transcripts"][ep] != json.loads(dumps(store.transcript(name, ep))):
                bad(f"{rel}: transcript {ep} differs")
            n_transcripts += 1
        for key, fn in (("sample", store.sample), ("context", store.context_list), ("timeline", store.timeline)):
            if got.get(key) != json.loads(dumps(endpoint(fn, name))):
                bad(f"{rel}: {key} differs")
        store.forget(name)
        print(f"  {name}: checked", flush=True)

    figs = sorted(FIGURES.glob("*.png")) if FIGURES.is_dir() else []
    if {f"figures/{f.name}" for f in figs} != {r for r in listed if r.startswith("figures/")}:
        bad("the figure set differs from docs/figures/*.png")
    for f in figs:
        p = bundle / "figures" / f.name
        if p.is_file() and p.read_bytes() != f.read_bytes():
            bad(f"figures/{f.name} differs from docs/figures")

    index = (bundle / "index.html").read_text()
    if not index.startswith("<title>"):
        bad("index.html does not start with <title>")
    for tag in ("<!doctype", "<html", "<head", "<body", "<script src=", "<link "):
        if tag in index.lower():
            bad(f"index.html contains {tag}")
    if "window.RESULTS_STATIC = {" not in index:
        bad("index.html does not set window.RESULTS_STATIC")
    if (STATIC / "app.js").read_text() not in index:
        bad("index.html does not carry app.js verbatim")
    if themed_css((STATIC / "style.css").read_text()) not in index:
        bad("index.html does not carry the themed stylesheet verbatim")
    preview = (bundle / "preview.html").read_text()
    if index not in preview or not preview.startswith("<!doctype html>"):
        bad("preview.html does not wrap index.html in a document")

    for rel in listed:
        if rel.endswith(".png"):
            continue
        text = (bundle / rel).read_text(errors="replace")
        for pat in SECRETS:
            m = pat.search(text)
            if m:
                bad(f"{rel}: credential-like text at offset {m.start()}: {m.group(0)[:12]}…")

    print(f"{len(names)} runs, {n_transcripts} transcripts, {len(listed)} files, {total / 1e6:.1f} MB; "
          f"{len(problems)} problem(s)")
    for pr in problems:
        print(f"  BROKEN: {pr}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
