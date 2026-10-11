"""HTTP server for the results browser. Stdlib only, plus inspect_ai for eval logs.

Endpoints (all JSON unless noted; see API.md):
    GET /                                   index.html
    GET /static/<file>                      UI assets
    GET /api/runs                           every run with its episodes and verdicts
    GET /api/runs/<run>                     run.json, episodes, verdicts, defenses, communication, defense evidence
    GET /api/runs/<run>/transcript/<episode>   the ATIF transcript plus the episode record and verdict
    GET /api/runs/<run>/sample              sample-level facts from the eval log (scorer metadata, probe, limit, error)
    GET /api/runs/<run>/context             index of every model call in the eval log (agents and defenses)
    GET /api/runs/<run>/context/<i>         one model call: exact provider input, output, usage
"""
from __future__ import annotations

import json
import mimetypes
import sys
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from integration.eval_to_atif import split_content, tool_result_text  # noqa: E402

STATIC = Path(__file__).resolve().parent / "static"
REFUSAL = "I cannot provide that response under the configured defense policy."
FLAG_KEYS = {"orbit_communication_delivery": "delivery_notice", "orbit_trailing_turn_fix": "trailing_turn_fix",
             "orbit_defense_redacted": "redacted"}


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


class Store:
    """Reads run directories; loads an eval log once per run on demand."""

    def __init__(self, runs_dir: Path) -> None:
        self.runs_dir = runs_dir
        self._logs: dict[str, object] = {}
        self._calls: dict[str, list] = {}
        self._lock = threading.Lock()

    # -- run directories ---------------------------------------------------
    def run_names(self) -> list[str]:
        return sorted(p.parent.name for p in self.runs_dir.glob("*/run.json"))

    def _dir(self, name: str) -> Path:
        if name not in self.run_names():
            raise KeyError(name)
        return self.runs_dir / name

    def episodes(self, name: str) -> list[dict]:
        d = self._dir(name)
        verdicts = {v["episode"]: v for v in _jsonl(d / "judge.jsonl")}
        out = []
        for rec in _jsonl(d / "episodes.jsonl"):
            v = verdicts.get(rec["episode"])
            rec = dict(rec)
            rec["verdict"] = ({k: v.get(k) for k in ("cheated", "honey_pots_seen", "attempted_access", "extracted_submission",
                                                   "success_claim_seen_before_access", "disclosed_access",
                                                   "explicit_prohibition_violated", "summary", "judge", "judge_cost_usd",
                                                   "parse_ok", "judge_error")} if v else None)
            rec["has_transcript"] = bool(rec.get("transcript_path")) and Path(rec["transcript_path"]).exists()
            out.append(rec)
        return out

    def run_summary(self, name: str) -> dict:
        d = self._dir(name)
        run = _json(d / "run.json") or {}
        eps = self.episodes(name)
        return {
            "run": name, "task": run.get("task"), "model": run.get("model"), "topology": run.get("topology"),
            "agents": run.get("agents"), "control": run.get("control") or "baseline",
            "monitor_model": run.get("monitor_model"), "defenses": [d_["name"] for d_ in run.get("defenses") or []],
            "eval_log": run.get("eval_log"), "eval_log_exists": bool(run.get("eval_log")) and Path(run["eval_log"]).exists(),
            "failed": any(e.get("failure") == "harness" for e in eps) or name.startswith("failed_"),
            "episodes": eps,
            "has_communication": (d / "communication").exists(), "has_defense": (d / "defense").exists(),
        }

    def run_detail(self, name: str) -> dict:
        d = self._dir(name)
        detail = self.run_summary(name)
        detail["run_json"] = _json(d / "run.json")
        detail["communication"] = {p.stem: _json(p) for p in sorted((d / "communication").glob("*.json"))} if (d / "communication").exists() else {}
        detail["defense"] = {p.stem: _json(p) for p in sorted((d / "defense").glob("*.json"))} if (d / "defense").exists() else {}
        return detail

    def transcript(self, name: str, episode: str) -> dict:
        for rec in self.episodes(name):
            if rec["episode"] == episode:
                if not rec.get("has_transcript"):
                    raise KeyError(f"{episode}: no transcript")
                t = json.loads(Path(rec["transcript_path"]).read_text())
                return {"episode": rec, "transcript": t}
        raise KeyError(episode)

    # -- eval log ----------------------------------------------------------
    def _log(self, name: str):
        with self._lock:
            if name in self._logs:
                return self._logs[name]
        run = _json(self._dir(name) / "run.json") or {}
        path = run.get("eval_log")
        if not path or not Path(path).exists():
            raise KeyError(f"{name}: eval log not found at {path}")
        from inspect_ai.log import read_eval_log

        log = read_eval_log(path, resolve_attachments=True)
        with self._lock:
            self._logs[name] = log
        return log

    def sample(self, name: str) -> dict:
        log = self._log(name)
        s = log.samples[0]
        scores = {}
        for k, sc in (s.scores or {}).items():
            scores[k] = {"value": sc.value, "answer": sc.answer, "explanation": sc.explanation, "metadata": sc.metadata}
        return {
            "status": log.status, "model": log.eval.model, "eval_id": log.eval.eval_id,
            "sample_id": s.id, "epoch": s.epoch, "total_time": s.total_time, "working_time": s.working_time,
            "limit": {"type": s.limit.type, "limit": s.limit.limit, "message": s.limit.message} if s.limit else None,
            "error": str(s.error.message)[:4000] if s.error else None,
            "model_usage": {k: {"input": v.input_tokens, "cache_read": v.input_tokens_cache_read, "output": v.output_tokens,
                                "reasoning": v.reasoning_tokens} for k, v in (s.model_usage or {}).items()},
            "scores": scores,
            "store": {k: s.store.get(k) for k in ("EnvironmentState:turn", "EnvironmentState:last_text_by_agent",
                                                  "DefenseLog:blocks", "DefenseLog:activity", "DefenseLog:vaccination_injections")},
            "n_messages": len(s.messages), "n_events": len(s.events),
        }

    def _owner_fn(self, sample):
        begins = {e.id: e for e in sample.events if e.event == "span_begin"}
        cache: dict = {}

        def owner(span_id):
            """(agent or None, defense name or None) for a span."""
            if span_id in cache:
                return cache[span_id]
            e = begins.get(span_id)
            if e is None:
                res = (None, None)
            elif e.type == "orbit.defense":
                parent = owner(e.parent_id) if e.parent_id else (None, None)
                res = (parent[0], e.name)
            elif e.type == "tool":
                # a model call made while a tool runs (dual-LLM quarantine summariser)
                parent = owner(e.parent_id) if e.parent_id else (None, None)
                res = (parent[0], f"inside tool {e.name}")
            elif e.type == "agent":
                res = (e.name, None)
            else:
                res = owner(e.parent_id) if e.parent_id else (None, None)
            cache[span_id] = res
            return res
        return owner

    def _redacted_ids(self, sample) -> set:
        ids = set()
        for e in sample.events:
            d = getattr(e, "data", None)
            if e.event == "info" and isinstance(d, dict) and d.get("event") == "defense_intervention" \
                    and d.get("boundary") == "model_input" and d.get("allow") is False and d.get("message_id"):
                ids.add(d["message_id"])
        return ids

    def _msg(self, m, redacted: set = frozenset()) -> dict:
        text, reasoning = split_content(m.content)
        meta = m.metadata or {}
        d = {"role": m.role, "id": m.id, "text": text, "reasoning": reasoning,
             "flags": {v for k, v in FLAG_KEYS.items() if meta.get(k)} | ({"redacted"} if m.id in redacted else set())}
        d["flags"] = sorted(d["flags"])
        if m.role == "assistant":
            d["tool_calls"] = [{"id": tc.id, "function": tc.function, "arguments": tc.arguments} for tc in (m.tool_calls or [])]
            d["model"] = getattr(m, "model", None)
            if text.strip() == REFUSAL:
                d["flags"].append("defense_refusal")
        if m.role == "tool":
            d["tool_call_id"] = m.tool_call_id
            d["function"] = m.function
            d["text"] = tool_result_text(m.content)
            d["error"] = m.error.message if m.error else None
        return d

    def context_index(self, name: str) -> dict:
        with self._lock:
            cached = self._calls.get(name)
        if cached is not None:
            return cached
        log = self._log(name)
        s = log.samples[0]
        owner = self._owner_fn(s)
        redacted = self._redacted_ids(s)
        calls = []
        for e in s.events:
            if e.event != "model":
                continue
            agent, defense = owner(e.span_id)
            out = e.output
            msg = out.message if (out and out.choices) else None
            usage = out.usage if out else None
            calls.append({
                "index": len(calls), "agent": agent, "defense": defense, "model": e.model,
                "timestamp": e.timestamp.isoformat() if e.timestamp else None, "working_time": e.working_time,
                "n_input": len(e.input or []), "input_chars": sum(len(m.text or "") for m in (e.input or [])),
                "tools": [t.name for t in (e.tools or [])],
                "usage": {"input": usage.input_tokens, "cache_read": usage.input_tokens_cache_read, "output": usage.output_tokens,
                          "reasoning": usage.reasoning_tokens} if usage else None,
                "output_preview": ((msg.text if msg else "") or "")[:200],
                "output_tool_calls": [tc.function for tc in (msg.tool_calls or [])] if msg else [],
                "stop_reason": out.choices[0].stop_reason if (out and out.choices) else None,
                "error": str(e.error)[:500] if e.error else None,
                "flags": sorted({f for m in (e.input or []) for f in self._msg(m, redacted)["flags"]}
                                | ({"defense_refusal"} if msg and (msg.text or "").strip() == REFUSAL else set())),
                "_event": e, "_redacted": redacted,
            })
        with self._lock:
            self._calls[name] = calls
        return calls

    def context_list(self, name: str) -> dict:
        calls = self.context_index(name)
        public = [{k: v for k, v in c.items() if not k.startswith("_")} for c in calls]
        agents = sorted({c["agent"] for c in calls if c["agent"]})
        defenses = sorted({c["defense"] for c in calls if c["defense"]})
        return {"run": name, "agents": agents, "defenses": defenses, "calls": public}

    def context_call(self, name: str, index: int) -> dict:
        calls = self.context_index(name)
        if not 0 <= index < len(calls):
            raise KeyError(f"call {index}")
        c = calls[index]
        e = c["_event"]
        out = e.output
        msg = out.message if (out and out.choices) else None
        return {
            **{k: v for k, v in c.items() if not k.startswith("_")},
            "input": [self._msg(m, c["_redacted"]) for m in (e.input or [])],
            "output": (self._msg(msg) if msg else None),
            "output_error": str(out.error) if (out and out.error) else None,
            "tool_schemas": [{"name": t.name, "description": t.description} for t in (e.tools or [])],
        }


class Handler(SimpleHTTPRequestHandler):
    store: Store = None  # set by serve()

    def log_message(self, fmt, *args):  # quieter than the default
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _send_json(self, obj, status=200):
        body = json.dumps(obj, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path):
        if not path.is_file():
            return self._send_json({"error": "not found"}, 404)
        body = path.read_bytes()
        ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", f"{ctype}; charset=utf-8" if ctype.startswith("text/") or ctype.endswith("javascript") else ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)
        parts = [unquote(p) for p in url.path.split("/") if p]
        try:
            if not parts or parts == ["index.html"]:
                return self._send_file(STATIC / "index.html")
            if parts[0] == "static" and len(parts) == 2 and "/" not in parts[1] and ".." not in parts[1]:
                return self._send_file(STATIC / parts[1])
            if parts[0] != "api":
                return self._send_json({"error": "not found"}, 404)
            store = self.store
            if parts[1:] == ["runs"]:
                return self._send_json([store.run_summary(n) for n in store.run_names()])
            if len(parts) >= 3 and parts[1] == "runs":
                name = parts[2]
                rest = parts[3:]
                if not rest:
                    return self._send_json(store.run_detail(name))
                if rest[0] == "transcript" and len(rest) == 2:
                    return self._send_json(store.transcript(name, rest[1]))
                if rest == ["sample"]:
                    return self._send_json(store.sample(name))
                if rest == ["context"]:
                    return self._send_json(store.context_list(name))
                if rest[0] == "context" and len(rest) == 2 and rest[1].isdigit():
                    return self._send_json(store.context_call(name, int(rest[1])))
            return self._send_json({"error": "not found"}, 404)
        except KeyError as ex:
            return self._send_json({"error": f"not found: {ex}"}, 404)
        except Exception as ex:  # noqa: BLE001 - report to the client, keep serving
            import traceback
            return self._send_json({"error": f"{type(ex).__name__}: {ex}", "traceback": traceback.format_exc()[-2000:]}, 500)


def serve(runs_dir: Path, host: str = "127.0.0.1", port: int = 8765) -> None:
    Handler.store = Store(runs_dir.resolve())
    httpd = ThreadingHTTPServer((host, port), Handler)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
