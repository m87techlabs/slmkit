"""Start, stop and check the studio as a background process.

State lives in $SLM_HOME/studio/ (never in the repo): `studio.json` records the process ID and
port, `studio.log` collects its output, `vendor/` holds the browser libraries. The studio holds
no state of its own, so a power-off loses nothing: `slm studio start` again (a stale record from
a dead process is detected and replaced). Nothing depends on it running, which is why a
background server doesn't conflict with "no daemons" (CONTRIBUTING.md): it is a convenience you start
and stop, not a job the machine must keep alive.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
import webbrowser
from collections.abc import Callable
from pathlib import Path
from typing import Any

from slmkit import artifacts
from slmkit.studio import vendor

DEFAULT_PORT = 8765
Log = Callable[[str], None]


def state_dir() -> Path:
    path = artifacts.slm_home() / "studio"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _record() -> dict[str, Any] | None:
    try:
        data: dict[str, Any] = json.loads((state_dir() / "studio.json").read_text())
        return data
    except (OSError, ValueError):
        return None


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def health(port: int, timeout: float = 2.0) -> dict[str, Any] | None:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=timeout) as r:
            data: dict[str, Any] = json.loads(r.read())
            return data
    except (OSError, ValueError):
        return None


def status() -> dict[str, Any] | None:
    """The running studio's record plus its health, or None. Clears a stale record."""
    rec = _record()
    if rec is None:
        return None
    if not _alive(rec["pid"]):
        (state_dir() / "studio.json").unlink(missing_ok=True)
        return None
    return {**rec, "url": f"http://localhost:{rec['port']}/", "health": health(rec["port"])}


def _in_wsl() -> bool:
    try:
        return "microsoft" in Path("/proc/version").read_text().lower()
    except OSError:
        return False


def open_browser(url: str) -> None:
    """The Windows default browser from WSL, otherwise Python's `webbrowser`."""
    if _in_wsl():
        # cwd on a Windows drive: cmd.exe warns about UNC paths when started from a Linux path.
        subprocess.Popen(["cmd.exe", "/c", "start", "", url], cwd="/mnt/c",
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # fmt: skip
    else:
        webbrowser.open(url)


def ensure_vendor(log: Log) -> None:
    missing = vendor.ensure(state_dir() / "vendor", log=log)
    if missing:
        log("  (the studio works without them; charts show a notice instead)")


def start(port: int = DEFAULT_PORT, *, open_: bool = True, log: Log = print) -> dict[str, Any]:
    running = status()
    if running:
        log(f"already running: {running['url']} (pid {running['pid']})")
        return running
    ensure_vendor(log)
    logfile = state_dir() / "studio.log"
    with logfile.open("a") as out:
        out.write(
            f"\n--- start {_dt.datetime.now().astimezone().isoformat(timespec='seconds')} port {port}\n"
        )
        out.flush()
        proc = subprocess.Popen(
            [sys.executable, "-m", "slmkit.studio", "--port", str(port)],
            stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            start_new_session=True,  # survives the terminal closing; `slm studio stop` ends it
        )  # fmt: skip
    rec = {"pid": proc.pid, "port": port,
           "started": _dt.datetime.now().astimezone().isoformat(timespec="seconds")}  # fmt: skip
    (state_dir() / "studio.json").write_text(json.dumps(rec) + "\n")
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            (state_dir() / "studio.json").unlink(missing_ok=True)
            raise RuntimeError(f"the studio exited during startup; see {logfile}")
        if health(port, timeout=1):
            break
        time.sleep(0.5)
    else:
        raise RuntimeError(f"the studio didn't answer within 60 s; see {logfile}")
    url = f"http://localhost:{port}/"
    log(f"slm studio is running: {url}  (pid {proc.pid}, log {logfile})")
    if open_:
        open_browser(url)
    return {**rec, "url": url}


def stop(log: Log = print) -> bool:
    rec = status()
    if rec is None:
        log("slm studio is not running")
        return False
    os.kill(rec["pid"], signal.SIGTERM)
    for _ in range(40):
        if not _alive(rec["pid"]):
            break
        time.sleep(0.25)
    else:
        os.kill(rec["pid"], signal.SIGKILL)
    (state_dir() / "studio.json").unlink(missing_ok=True)
    log(f"stopped slm studio (pid {rec['pid']})")
    return True
