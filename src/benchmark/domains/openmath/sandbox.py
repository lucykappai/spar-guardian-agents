"""The openmath sandbox: the staging tarball, the setup scripts, the docker context and the per-agent tools.

The docker context holds the agent image, the mirror sidecar image and the compose file under build/openmath_agent/.
The setup scripts and the two Dockerfiles live in scripts/ and are read at import.

Two layouts share the one image and compose file. CheatBench's own layout (a lone agent, and every agent of a
shared_task team) stages the row under /workspace for the user `user`. The solo_task layout of a team gives each
agent a Linux user of its own name, a home /home/<name> and a copy of the row under /workspace/<name>, both mode 700,
and binds its tools to that user and directory.
"""
from __future__ import annotations

import datetime as dt
import io
import os
import tarfile
from pathlib import Path

import yaml
from inspect_ai.tool import ToolDef
from inspect_ai.util import sandbox
from orbit.tools import ToolContext, ToolFactory

from src.configuration.openmath import AGENT_NAME, AGENT_UID, BUILD, MIRROR_IP, TOOL_TIMEOUT_S, WORKDIR

SCRIPTS = Path(__file__).parent / "scripts"
SETUP_SCRIPT = (SCRIPTS / "setup.sh").read_text().format(workdir=WORKDIR, agent_name=AGENT_NAME)
SOLO_SETUP = (SCRIPTS / "setup_solo.sh").read_text()
SOLO_SETUP_AGENT = (SCRIPTS / "setup_solo_agent.sh").read_text()
DOCKERFILE_AGENT = (SCRIPTS / "Dockerfile.agent").read_text()
DOCKERFILE_MIRROR = (SCRIPTS / "Dockerfile.mirror").read_text()
# ORBIT reserves the names bash and python for its built-in tools, which take one user for every agent. The per-agent
# tools of the solo_task layout therefore take SWE's names. Their descriptions are Inspect's bash and python texts.
SOLO_TOOLS = ["shell", "run_python"]
BASH_DESCRIPTION = "Use this function to execute bash commands."
PYTHON_DESCRIPTION = """Use the python function to execute Python code.

The Python tool executes single-run Python scripts. Important notes:
1. Each execution is independent - no state is preserved between runs
2. You must explicitly use print() statements to see any output
3. Simply writing expressions (like in notebooks) will not display results
4. The script cannot accept interactive input during execution
5. Return statements alone won't produce visible output
6. All variables and imports are cleared between executions
7. Standard output (via print()) is the only way to see results"""


def agent_workdir(name: str) -> str:
    """An agent's copy of the row in the solo_task layout."""
    return f"{WORKDIR}/{name}"


def agent_layout(agents: list[str]) -> dict[str, dict]:
    """The solo_task layout: each agent's Linux user, uid, home and workspace. Uids follow `user`'s 1000."""
    return {a: {"user": a, "uid": AGENT_UID + 1 + i, "home": f"/home/{a}", "workdir": agent_workdir(a)}
            for i, a in enumerate(agents)}


def solo_setup_script(agents: list[str]) -> str:
    """The solo_task setup: /workspace becomes root's, and /workspace and /home mode 711, so an agent can reach its own
    directories but cannot list the others'; then per agent a user, a home and a copy of the staged row.

    The tarball is extracted once per agent with its members' modes and mtimes; chown changes the owner only."""
    layout = agent_layout(agents)
    blocks = "".join(SOLO_SETUP_AGENT.format(workdir=WORKDIR, name=a, uid=layout[a]["uid"]) for a in agents)
    return SOLO_SETUP.format(workdir=WORKDIR, agents=blocks.rstrip("\n"))


def parse_mtime(value) -> int:
    """CheatBench core/stage.py: 'YYYY-MM-DD[ HH:MM[:SS]]' or epoch, UTC."""
    if isinstance(value, (int, float)):
        return int(value)
    s = str(value).strip()
    if s.isdigit():
        return int(s)
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return int(dt.datetime.strptime(s, fmt).replace(tzinfo=dt.timezone.utc).timestamp())
        except ValueError:
            continue
    raise ValueError(f"unparseable file_mtimes value {value!r}")


def make_stage_tar(row: dict, env_dir: Path, out_path: Path) -> Path:
    """Pack row['files'] (dest relative to /workspace) with CheatBench's staging semantics.

    Files 0644 (or file_modes), directories created for them 0755, everything
    owned by uid/gid 1000, mtimes from file_mtimes; a created directory takes
    the newest mtime staged beneath it (core/stage.py:write).
    """
    if row.get("files_abs"):
        raise NotImplementedError("files_abs staging is not needed for openmath_agent")
    mtimes = {k: parse_mtime(v) for k, v in (row.get("file_mtimes") or {}).items()}
    modes = {k: int(str(v), 8) for k, v in (row.get("file_modes") or {}).items()}
    dir_mtime: dict[str, int] = {}
    files: list[tuple[str, Path]] = []
    for dest, src in sorted(row["files"].items()):
        if dest.startswith("/") or ".." in Path(dest).parts:
            raise ValueError(f"bad destination {dest!r}")
        src_path = (env_dir / src).resolve()
        if not src_path.is_relative_to(env_dir.resolve()) or not src_path.is_file():
            raise ValueError(f"bad source {src!r}")
        files.append((dest, src_path))
        when = mtimes.get(dest)
        if when is not None:
            parent = Path(dest).parent
            while str(parent) not in (".", ""):
                dir_mtime[str(parent)] = max(dir_mtime.get(str(parent), when), when)
                parent = parent.parent
    now = int(dt.datetime.now(dt.timezone.utc).timestamp())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Written to a private temp file and renamed into place: several task
    # builds may run at once, and a reader must never see a partial tarball.
    tmp_path = out_path.with_name(f"{out_path.name}.tmp{os.getpid()}")
    with tarfile.open(tmp_path, "w:gz") as tar:
        for d in sorted(dir_mtime, key=lambda p: (p.count("/"), p)):
            info = tarfile.TarInfo(d)
            info.type = tarfile.DIRTYPE
            info.mode = 0o755
            info.uid = info.gid = AGENT_UID
            info.uname = info.gname = AGENT_NAME
            info.mtime = dir_mtime[d]
            tar.addfile(info)
        for dest, src_path in files:
            data = src_path.read_bytes()
            info = tarfile.TarInfo(dest)
            info.size = len(data)
            info.mode = modes.get(dest, 0o644)
            info.uid = info.gid = AGENT_UID
            info.uname = info.gname = AGENT_NAME
            info.mtime = mtimes.get(dest, now)
            tar.addfile(info, io.BytesIO(data))
    os.replace(tmp_path, out_path)
    return out_path


def write_docker_context() -> Path:
    env = BUILD / "environment"
    site = env / "site"
    docker_dir = BUILD / "docker"
    docker_dir.mkdir(parents=True, exist_ok=True)
    _write_atomic(docker_dir / "ca.crt", (site / "tls" / "ca.crt").read_bytes())
    _write_atomic(docker_dir / "Dockerfile.agent", DOCKERFILE_AGENT.encode())
    _write_atomic(docker_dir / "Dockerfile.mirror", DOCKERFILE_MIRROR.encode())
    compose = {
        "services": {
            "default": {
                "build": {"context": ".", "dockerfile": "Dockerfile.agent"},
                "init": True,
                "command": ["tail", "-f", "/dev/null"],
                "working_dir": WORKDIR,
                "network_mode": "none",
                "extra_hosts": [f"openai.com:{MIRROR_IP}"],
                "stop_grace_period": "1s",
            },
            "mirror": {
                "build": {"context": ".", "dockerfile": "Dockerfile.mirror"},
                "init": True,
                "network_mode": "service:default",
                "cap_add": ["NET_ADMIN"],
                "depends_on": ["default"],
                "volumes": [f"{site.resolve()}:/mirror:ro",
                            f"{(env / 'server.py').resolve()}:/mirror_server.py:ro"],
                "command": ["bash", "-c",
                            f"ip addr add {MIRROR_IP}/32 dev lo 2>/dev/null || true; "
                            "exec python /mirror_server.py --mirror /mirror "
                            "--cert /mirror/tls/site.crt --key /mirror/tls/site.key "
                            "--port 443 --bind 0.0.0.0"],
                "stop_grace_period": "1s",
            },
        }
    }
    compose_path = docker_dir / "compose.yaml"
    _write_atomic(compose_path, yaml.safe_dump(compose, sort_keys=False).encode())
    return compose_path


def _write_atomic(path: Path, data: bytes) -> None:
    """Write through a private temp file and rename, so concurrent task builds never expose a partial file."""
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _output(result) -> str:
    """Inspect's bash and python tools' output: stderr, then stdout."""
    return (f"{result.stderr}\n" if result.stderr else "") + f"{result.stdout}"


def _shell(ctx: ToolContext) -> ToolDef:
    workdir = agent_workdir(ctx.agent_name)

    async def execute(command: str) -> str:
        result = await sandbox().exec(["bash", "--login", "-c", command], timeout=TOOL_TIMEOUT_S,
                                      user=ctx.agent_name, cwd=workdir)
        return _output(result)

    return ToolDef(execute, name="shell", description=BASH_DESCRIPTION,
                   parameters={"command": "The bash command to execute."})


def _run_python(ctx: ToolContext) -> ToolDef:
    workdir = agent_workdir(ctx.agent_name)

    async def execute(code: str) -> str:
        result = await sandbox().exec(["bash", "--login", "-c", "python3 -"], input=code, timeout=TOOL_TIMEOUT_S,
                                      user=ctx.agent_name, cwd=workdir)
        return _output(result)

    return ToolDef(execute, name="run_python", description=PYTHON_DESCRIPTION,
                   parameters={"code": "The python code to execute."})


def solo_tool_bindings() -> dict[str, ToolFactory]:
    """`shell` and `run_python` as Inspect's bash and python, run as the agent's own user in its own copy."""
    return {"shell": ToolFactory(_shell), "run_python": ToolFactory(_run_python)}
