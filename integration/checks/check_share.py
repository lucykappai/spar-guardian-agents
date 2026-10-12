"""Check a static export of the results browser. Exit 1 on any problem.

    python -m integration.checks.check_share [--runs runs] [--bundle integration/build/share] [--allow-errors]

Every check is against the thing itself:
  * every file in manifest.json exists with its recorded size and SHA-256, nothing else is in the directory, no text
    file is over 16 MB, no image over 15 MB, the bundle is under 256 MB and 511 files (the publisher's limits per
    version; one publish takes at most 255 files and 64 MB, so the check prints how many publishes the bundle needs);
  * data/runs.json, data/summary.json, data/figures.json and every data/run/<run>/*.json equal what Store answers
    now for the same paths (the error records too), the transcripts are exactly the episodes that have one, and the
    manifest's counts and error list match;
  * no endpoint is an error record unless --allow-errors;
  * the figures equal docs/figures/*.png byte for byte;
  * artifact.html is exactly page_body(themed stylesheet, app.js, config) with the config's roots relative and its
    stamp equal to the manifest's, index.html is exactly artifact.html in the publisher's skeleton, the themed
    stylesheet has both [data-theme] dark blocks with color-scheme dark, and the page loads nothing external;
  * no credential pattern appears in any decoded JSON string or page text, and no value of a credential-like
    environment variable (API keys, tokens, secrets, passwords) appears anywhere in the bundle.
A headless browser pass over the served bundle is integration/checks/check_share_browser.js (make share-browser).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from integration.ui.export import (OMITTED, dumps, endpoint, full_page, is_error, page_body,  # noqa: E402
                                   source_hashes, themed_css)
from integration.ui.server import FIGURES, STATIC, Store, figures_list  # noqa: E402

MAX_TEXT = 16 * 1000 * 1000
MAX_BINARY = 15 * 1000 * 1000
MAX_TOTAL = 256 * 1000 * 1000
MAX_FILES = 511
PUBLISH_FILES = 255
PUBLISH_BYTES = 64 * 1000 * 1000
SECRET_PATTERNS = [(re.compile(p), what) for p, what in (
    (r"AIza[0-9A-Za-z_-]{35}", "Google API key"),
    (r"(?<![A-Za-z0-9])sk-(?:ant-|proj-)?[A-Za-z0-9_-]{20,}", "OpenAI or Anthropic key"),
    (r"\bAKIA[0-9A-Z]{16}\b", "AWS access key id"),
    (r"(?i)aws_secret_access_key\W{0,4}[A-Za-z0-9/+=]{30,}", "AWS secret key"),
    (r"(?<![A-Za-z0-9])(?:gh[pousr]|github_pat)_[A-Za-z0-9_]{20,}", "GitHub token"),
    (r"(?<![A-Za-z0-9])hf_[A-Za-z0-9]{30,}", "Hugging Face token"),
    (r"(?<![A-Za-z0-9])xox[abprs]-[A-Za-z0-9-]{10,}", "Slack token"),
    (r"-----BEGIN [A-Z ]*PRIVATE KEY-----", "private key"),
    (r"(?im)^\s*(authorization|cookie|set-cookie|x-api-key|x-goog-api-key)\s*:\s*\S{8,}", "credential header"),
    (r"(?i)\bbearer\s+[A-Za-z0-9._~+/-]{16,}", "bearer token"),
    (r"\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}", "JWT"),
    (r"(?i)[A-Za-z0-9_]*(?:api[_-]?key|secret|token|password|passwd)[A-Za-z0-9_]*[\"']?\s*[:=]\s*[\"']?(?:bearer\s+)?"
     r"(?=[A-Za-z0-9_\-./+=]*[A-Za-z])(?=[A-Za-z0-9_\-./+=]*[0-9])[A-Za-z0-9_\-./+=]{20,}", "key = value"),
)]
ENV_CREDENTIAL = re.compile(r"(API_KEY|TOKEN|SECRET|PASSWORD|PASSWD)$", re.IGNORECASE)


def strings_of(v):
    """Every string inside a decoded JSON value."""
    if isinstance(v, str):
        yield v
    elif isinstance(v, dict):
        for k, x in v.items():
            yield k
            yield from strings_of(x)
    elif isinstance(v, list):
        for x in v:
            yield from strings_of(x)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=str(ROOT / "runs"))
    ap.add_argument("--bundle", default=str(ROOT / "integration" / "build" / "share"))
    ap.add_argument("--allow-errors", action="store_true", help="accept endpoints stored as error records")
    args = ap.parse_args()
    bundle = Path(args.bundle)
    store = Store(Path(args.runs).resolve())
    problems: list[str] = []
    bad = problems.append

    # -- files, sizes, limits ---------------------------------------------------------------------
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
        limit = MAX_BINARY if rel.endswith(".png") else MAX_TEXT
        if len(data) > limit:
            bad(f"{rel}: {len(data)} bytes, over the {limit // 1000000} MB limit")
    on_disk = {p.relative_to(bundle).as_posix() for p in bundle.rglob("*") if p.is_file()} - {"manifest.json"}
    for extra in sorted(on_disk - set(listed)):
        bad(f"not in the manifest: {extra}")
    total = sum(f["bytes"] for f in listed.values())
    if total > MAX_TOTAL:
        bad(f"bundle is {total} bytes, over 256 MB")
    if len(listed) > MAX_FILES:
        bad(f"bundle has {len(listed)} files, over {MAX_FILES}")
    published = {rel: f for rel, f in listed.items() if rel != "index.html"}
    publishes = max(math.ceil(len(published) / PUBLISH_FILES), math.ceil(sum(f["bytes"] for f in published.values()) / PUBLISH_BYTES))
    if manifest.get("sources") != source_hashes():
        bad("manifest sources differ from app.js, style.css and export.py on disk: re-export")

    # -- data parity with the run directories -------------------------------------------------------
    def same(rel: str, expected) -> None:
        p = bundle / rel
        if not p.is_file():
            bad(f"missing {rel}")
        elif json.loads(p.read_text()) != json.loads(dumps(expected)):
            bad(f"{rel} differs from the live data")

    names = store.run_names()
    same("data/runs.json", [store.run_summary(n) for n in names])
    same("data/summary.json", store.summary())
    same("data/figures.json", figures_list())
    n_transcripts = 0
    errors: dict[str, list[str]] = {}
    for name in names:
        d = f"data/run/{name}"
        parts = {"run": endpoint(store.run_detail, name), "sample": endpoint(store.sample, name),
                 "context": endpoint(store.context_list, name), "timeline": endpoint(store.timeline, name)}
        for key, want in parts.items():
            same(f"{d}/{key}.json", want)
            if is_error(want):
                errors.setdefault(name, []).append(f"{key}: {want['__error']['status']}")
        p = bundle / d / "transcripts.json"
        if not p.is_file():
            bad(f"missing {d}/transcripts.json")
        else:
            got = json.loads(p.read_text())
            want_eps = {e["episode"] for e in store.episodes(name) if e.get("has_transcript")}
            if set(got) != want_eps:
                bad(f"{d}/transcripts.json holds {sorted(got)} but the run has transcripts for {sorted(want_eps)}")
            for ep in sorted(want_eps & set(got)):
                want = endpoint(store.transcript, name, ep)
                if got[ep] != json.loads(dumps(want)):
                    bad(f"{d}/transcripts.json: {ep} differs")
                if is_error(want):
                    errors.setdefault(name, []).append(f"transcript {ep}: {want['__error']['status']}")
                else:
                    n_transcripts += 1
        store.forget(name)
        print(f"  {name}: checked", flush=True)
    if manifest.get("runs") != len(names) or manifest.get("transcripts") != n_transcripts:
        bad(f"manifest counts {manifest.get('runs')} runs, {manifest.get('transcripts')} transcripts; "
            f"found {len(names)} and {n_transcripts}")
    if manifest.get("endpoint_errors") != errors:
        bad(f"manifest endpoint_errors {manifest.get('endpoint_errors')} differ from the records found {errors}")
    if errors and not args.allow_errors:
        bad("endpoints stored as error records (pass --allow-errors to accept): "
            + "; ".join(f"{k}: {', '.join(v)}" for k, v in errors.items()))
    if manifest.get("omitted") != [OMITTED]:
        bad("manifest omitted list differs from the exporter's")

    # -- figures ---------------------------------------------------------------------------------------
    figs = sorted(FIGURES.glob("*.png")) if FIGURES.is_dir() else []
    if {f"figures/{f.name}" for f in figs} != {r for r in listed if r.startswith("figures/")}:
        bad("the figure set differs from docs/figures/*.png")
    for f in figs:
        p = bundle / "figures" / f.name
        if p.is_file() and p.read_bytes() != f.read_bytes():
            bad(f"figures/{f.name} differs from docs/figures")

    # -- the page -------------------------------------------------------------------------------------
    fragment = (bundle / "artifact.html").read_text()
    m = re.search(r"<script>window\.RESULTS_STATIC = (\{.*?\});</script>\n", fragment)
    cfg = None
    if not m:
        bad("artifact.html does not set window.RESULTS_STATIC")
    else:
        cfg = json.loads(m.group(1))
        for key in ("data", "figures"):
            v = cfg.get(key)
            if not isinstance(v, str) or v.startswith("/") or "://" in v or not v.endswith("/"):
                bad(f"RESULTS_STATIC.{key} = {v!r} is not a relative directory")
        if cfg.get("generated") != manifest.get("generated") or cfg.get("git") != manifest.get("git"):
            bad("RESULTS_STATIC stamp differs from the manifest")
        if cfg.get("omitted") != OMITTED:
            bad("RESULTS_STATIC.omitted differs from the exporter's")
    css = themed_css((STATIC / "style.css").read_text())
    js = (STATIC / "app.js").read_text()
    if cfg is not None and fragment != page_body(css, js, cfg):
        bad("artifact.html is not exactly the page the exporter builds from style.css, app.js and its config")
    if (bundle / "index.html").read_text() != full_page(fragment):
        bad("index.html is not exactly artifact.html in the publisher's skeleton")
    if not fragment.startswith("<title>"):
        bad("artifact.html does not start with <title>")
    for tag in ("<!doctype", "<html", "<head", "<body", "<script src=", "<link ", "<iframe", "<object", "<embed"):
        if tag in fragment.lower():
            bad(f"artifact.html contains {tag}")
    for needle in ("@import", "url(http", "url('http", 'url("http', "url(//", "url('//", 'url("//'):
        if needle in css:
            bad(f"the stylesheet loads something external: {needle}")
    for block in (':root:not([data-theme="light"]) {', ':root[data-theme="dark"] {'):
        i = css.find(block)
        if i < 0 or "color-scheme: dark" not in css[i:i + 200]:
            bad(f"the themed stylesheet lacks color-scheme: dark in its {block} block")
    if "</script" in js.lower():
        bad("app.js contains '</script'")

    # -- secrets ---------------------------------------------------------------------------------------
    env_values = {k: v for k, v in os.environ.items() if ENV_CREDENTIAL.search(k) and len(v) >= 16}
    for rel in sorted(listed):
        if rel.endswith(".png"):
            continue
        raw = (bundle / rel).read_text(errors="replace")
        texts = list(strings_of(json.loads(raw))) if rel.endswith(".json") else [raw]
        hits: dict[str, int] = {}
        for text in texts:
            for pat, what in SECRET_PATTERNS:
                for mm in pat.finditer(text):
                    hits[f"{what} ({mm.group(0)[:10]}…)"] = hits.get(f"{what} ({mm.group(0)[:10]}…)", 0) + 1
            for k, v in env_values.items():
                if v in text:
                    hits[f"value of ${k}"] = hits.get(f"value of ${k}", 0) + 1
        for what, n in hits.items():
            bad(f"{rel}: {n} credential-like match(es): {what}")

    print(f"{len(names)} runs, {n_transcripts} transcripts, {len(listed)} files, {total / 1e6:.1f} MB "
          f"({publishes} publish(es) of at most {PUBLISH_FILES} files and {PUBLISH_BYTES // 1000000} MB); "
          f"{len(problems)} problem(s)")
    for pr in problems:
        print(f"  BROKEN: {pr}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
