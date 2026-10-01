"""The studio's Verify page: runbook checks, and running the safe ones.

A runbook is a list of checks: a heading, a ```bash block of commands, and usually the expected
output right after it. This module finds them, decides which can run from the browser, runs those,
and remembers the last result of each in $SLM_HOME/studio/verify.json (never in the repo).

**What may run.** Only read-only commands on a fixed list (`classify`): inspecting runs, models,
lineage and configs; the test and lint suites; projects' `check_*.py` analysis scripts; git and
GPU status; `ls`/`cat`-style reads inside $SLM_HOME or the repo. Nothing that trains, evaluates,
exports, serves, writes or deletes. Commands are split into arguments and matched against the list,
then run directly, never through a shell, so pipes, redirections, `;` and `$(…)` can't sneak in;
a block that uses them is shown with a Copy button instead, and the reason.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from slmkit.studio.data import runbooks

SHELL_SYNTAX = re.compile(r"[|;&<>`]|\$\(")
FENCE = re.compile(r"^(```+|~~~+)\s*([\w-]*)")
TIMEOUT_S = 600  # CONTRIBUTING.md rule 9: nothing launched from here may run longer than ~10 min
OUTPUT_LIMIT = 200_000

SLM_READ_ONLY = {
    ("runs", "list"), ("runs", "summary"), ("runs", "compare"), ("models", "list"),
    ("lineage",), ("config",), ("sample",), ("studio", "status"),
}  # fmt: skip
WRITES = {"ingest", "prepare", "tokenize", "pack", "pretrain", "sft", "run", "eval", "export",
          "serve", "doctor", "model"}  # fmt: skip
READERS = {"ls", "cat", "head", "tail", "wc", "du", "stat"}
SYSTEM = {"nvidia-smi", "nproc", "free", "uname", "df"}
GIT_READ = {"status", "log", "show", "diff", "ls-files", "rev-parse"}


@dataclass
class Check:
    id: str
    runbook: str
    section: str
    heading: str
    commands: list[str]
    expected: str | None
    runnable: bool
    reason: str  # why it can't run here, or what it does
    line: int = 0
    text: str = ""  # the block exactly as written, for display and copying


def _strip_comment(line: str) -> str:
    """`make test   # 205 passed` -> `make test`, leaving a `#` inside quotes alone."""
    quote = ""
    for k, ch in enumerate(line):
        if quote:
            quote = "" if ch == quote else quote
        elif ch in "'\"":
            quote = ch
        elif ch == "#" and (k == 0 or line[k - 1].isspace()):
            return line[:k].rstrip()
    return line.rstrip()


def _commands(block: str) -> list[str]:
    """Commands in a ```bash block: continuations joined, comments and blank lines dropped."""
    out: list[str] = []
    pending = ""
    for raw in block.splitlines():
        line = raw.rstrip()
        if line.endswith("\\"):
            pending += line[:-1].strip() + " "
            continue
        command = _strip_comment(pending + line.strip())
        pending = ""
        if command:
            out.append(command)
    return out


def _expand(arg: str, home: Path) -> str:
    return os.path.expanduser(arg.replace("$SLM_HOME", str(home)).replace("${SLM_HOME}", str(home)))


def _inside(path: str, roots: list[Path]) -> bool:
    try:
        resolved = Path(path).resolve()
    except (OSError, RuntimeError):
        return False
    return any(resolved == r or r in resolved.parents for r in roots)


def classify(command: str, home: Path, repo: Path) -> tuple[bool, str]:
    """(may run here, why or why not). The allowlist is deliberately short and literal."""
    if SHELL_SYNTAX.search(command):
        return (
            False,
            "uses shell syntax (pipes, redirection, ;, &&, $(…), heredocs): copy it to a terminal",
        )
    try:
        argv = shlex.split(command, comments=True)
    except ValueError:
        return False, "can't be parsed safely"
    if not argv:
        return False, "empty"
    if "=" in argv[0] and not argv[0].startswith("-"):
        return False, "sets environment variables: copy it to a terminal"
    head = argv[0]
    roots = [home.resolve(), repo.resolve()]
    if head == "uv" and argv[1:2] == ["run"]:
        rest = argv[2:]
        if rest[:1] == ["slm"]:
            sub = rest[1:]
            if any(tuple(sub[: len(p)]) == p for p in SLM_READ_ONLY):
                return True, "read-only: inspects what's on disk"
            if (
                sub[:1]
                and sub[0] in WRITES
                or sub[:2] in (["studio", "start"], ["studio", "stop"], ["studio", "run"])
            ):
                return (
                    False,
                    "trains, evaluates, writes to $SLM_HOME or starts a process: run it in your terminal",
                )
            return False, "not on the read-only list"
        if rest[:1] == ["pytest"]:
            return True, "the test suite: CPU-only unless marked gpu, uses temporary directories"
        if rest[:1] == ["mypy"] or rest[:2] == ["ruff", "check"] and "--fix" not in rest:
            return True, "a linter: reads the code"
        if rest[:2] == ["ruff", "format"] and "--check" in rest:
            return True, "a formatter in check mode: reads the code"
        if rest[:1] == ["ruff"]:
            return False, "ruff without --check rewrites files: run it in your terminal"
        if rest[:1] == ["python"] and len(rest) > 1:
            script = rest[1]
            if (
                re.fullmatch(r"projects/[\w-]+/check_[\w-]+\.py", script)
                or script == "scripts/val_metrics.py"
            ):
                return True, "an analysis script: reads runs and prints"
            return False, "runs Python code that isn't one of the read-only check scripts"
        return False, "not on the read-only list"
    if head == "make":
        if argv[1:] and all(t in {"test", "lint", "test-gpu"} for t in argv[1:]):
            return True, "the test or lint suite"
        return (
            False,
            "this make target writes or measures (doctor, figures): run it in your terminal",
        )
    if head in READERS:
        paths = [_expand(a, home) for a in argv[1:] if not a.startswith("-")]
        if paths and all(_inside(p, roots) for p in paths):
            return True, "reads files in $SLM_HOME or the repo"
        return False, "reads outside $SLM_HOME and the repo"
    if head == "git" and argv[1:2] and argv[1] in GIT_READ:
        if any(a.startswith("--output") for a in argv):
            return False, "--output writes a file"
        return True, "read-only git"
    if head in SYSTEM:
        return True, "reads system status"
    return False, "not on the read-only list"


def parse(repo: Path, relative: str, home: Path) -> list[Check]:
    """Every ```bash block in a runbook, with its heading and the expected output after it."""
    lines = (repo / relative).read_text().splitlines()
    checks: list[Check] = []
    section = heading = ""
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("# "):
            section = heading = line[2:].strip()
        elif line.startswith(("## ", "### ")):
            heading = line.lstrip("#").strip()
            if line.startswith("## "):
                section = heading
        m = FENCE.match(line)
        if not m:
            i += 1
            continue
        fence, lang = m.group(1), m.group(2)
        end = i + 1
        while end < len(lines) and not lines[end].startswith(fence[:3]):
            end += 1
        body = "\n".join(lines[i + 1 : end])
        if lang in ("bash", "sh", "shell"):
            expected = None
            j = end + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            nm = FENCE.match(lines[j]) if j < len(lines) else None
            if nm and nm.group(2) in ("", "text", "console", "output"):
                k = j + 1
                while k < len(lines) and not lines[k].startswith(nm.group(1)[:3]):
                    k += 1
                expected = "\n".join(lines[j + 1 : k])
                end = k
            cmds = _commands(body)
            if "<<" in body:
                runnable, reason = (
                    False,
                    "contains an inline script (heredoc): copy it to a terminal",
                )
            else:
                verdicts = [classify(c, home, repo) for c in cmds]
                runnable = bool(cmds) and all(ok for ok, _ in verdicts)
                reason = next(
                    (r for ok, r in verdicts if not ok), verdicts[0][1] if verdicts else "empty"
                )
            cid = hashlib.sha256(f"{relative}|{heading}|{body}".encode()).hexdigest()[:12]
            checks.append(
                Check(
                    cid, relative, section, heading, cmds, expected, runnable, reason, i + 1, body
                )
            )
        i = end + 1
    return checks


# ------------------------------------------------------------------------------ running


@dataclass
class Execution:
    id: str
    check_id: str
    started: float
    output: str = ""
    exit_code: int | None = None
    seconds: float | None = None
    done: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)


class Runner:
    """Runs one check at a time in a background thread; the page polls for output."""

    def __init__(self, home: Path, repo: Path) -> None:
        self.home, self.repo = home, repo
        self.state_file = home / "studio" / "verify.json"
        self._runs: dict[str, Execution] = {}
        self._busy = threading.Lock()

    def results(self) -> dict[str, Any]:
        try:
            data: dict[str, Any] = json.loads(self.state_file.read_text())
            return data
        except (OSError, ValueError):
            return {}

    def _save(self, check_id: str, ex: Execution) -> None:
        results = self.results()
        results[check_id] = {"time": time.time(), "exit_code": ex.exit_code, "seconds": ex.seconds,
                             "output": ex.output[-20_000:]}  # fmt: skip
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(results))
        tmp.replace(self.state_file)

    def start(self, check: Check) -> Execution:
        if not check.runnable:
            raise PermissionError(check.reason)
        if not self._busy.acquire(blocking=False):
            raise BlockingIOError("another check is running")
        ex = Execution(uuid.uuid4().hex[:12], check.id, time.time())
        self._runs[ex.id] = ex
        threading.Thread(target=self._run, args=(check, ex), daemon=True).start()
        return ex

    def get(self, run_id: str) -> Execution | None:
        return self._runs.get(run_id)

    def _run(self, check: Check, ex: Execution) -> None:
        try:
            for command in check.commands:
                ok, reason = classify(
                    command, self.home, self.repo
                )  # re-checked right before running
                if not ok:
                    raise PermissionError(reason)
                argv = [_expand(a, self.home) for a in shlex.split(command, comments=True)]
                exe = shutil.which(argv[0])
                if exe is None:
                    self._append(ex, f"$ {command}\n{argv[0]}: not found on PATH\n")
                    ex.exit_code = 127
                    break
                self._append(ex, f"$ {command}\n")
                proc = subprocess.Popen([exe, *argv[1:]], cwd=self.repo, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, text=True,
                                        errors="replace", env={**os.environ, "SLM_HOME": str(self.home)})  # fmt: skip
                assert proc.stdout is not None
                deadline = time.time() + TIMEOUT_S
                for out_line in proc.stdout:
                    self._append(ex, out_line)
                    if time.time() > deadline:
                        proc.kill()
                        self._append(ex, f"\n(stopped after {TIMEOUT_S} s)\n")
                        break
                ex.exit_code = proc.wait()
                if ex.exit_code != 0:
                    break
        except Exception as exc:  # noqa: BLE001 (any failure is reported on the page, not raised)
            self._append(ex, f"\n{exc}\n")
            ex.exit_code = ex.exit_code or 1
        finally:
            ex.seconds = round(time.time() - ex.started, 1)
            ex.done = True
            self._save(check.id, ex)
            self._busy.release()

    @staticmethod
    def _append(ex: Execution, text: str) -> None:
        with ex.lock:
            if len(ex.output) < OUTPUT_LIMIT:
                ex.output += text


def checks_for(repo: Path, home: Path, project: str | None) -> list[dict[str, Any]]:
    """The project's runbooks (and the engine's), each with its checks."""
    out = []
    for book in runbooks(repo, project):
        checks = parse(repo, book["file"], home)
        out.append({**book, "checks": [c.__dict__ for c in checks]})
    return out
