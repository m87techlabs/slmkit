"""The machine the models were built on: fixed facts, and live readings for the studio.

Fixed facts come from `slm doctor`'s doctor.json (measured once), /proc and nvidia-smi. Live
readings are one nvidia-smi query and two /proc reads, cheap enough to poll every couple of
seconds. Nothing here needs torch, so the studio's home page loads fast. Every reading is
optional: a machine without a GPU, or without nvidia-smi, just shows fewer numbers.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

_GPU_FIELDS = ("name", "driver_version", "utilization.gpu", "memory.used", "memory.total",
               "temperature.gpu", "power.draw", "power.limit")  # fmt: skip


def _nvidia_smi() -> dict[str, str] | None:
    exe = shutil.which("nvidia-smi") or "/usr/lib/wsl/lib/nvidia-smi"
    try:
        out = subprocess.run(
            [exe, f"--query-gpu={','.join(_GPU_FIELDS)}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.splitlines()  # fmt: skip
    except (OSError, subprocess.SubprocessError):
        return None
    if not out:
        return None
    values = [v.strip() for v in out[0].split(",")]
    return dict(zip(_GPU_FIELDS, values, strict=False))


def _num(value: str | None) -> float | None:
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None  # "[N/A]" on some GPUs


def _meminfo() -> dict[str, int]:
    out = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, _, rest = line.partition(":")
            out[key] = int(rest.split()[0]) * 1024
    except OSError:
        pass
    return out


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


_size_cache: dict[Path, tuple[float, int]] = {}


def dir_size(path: Path, max_age: float = 60.0) -> int:
    """Bytes under `path`, cached for a minute (walking a large $SLM_HOME takes a moment)."""
    hit = _size_cache.get(path)
    if hit and time.monotonic() - hit[0] < max_age:
        return hit[1]
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.lstat(os.path.join(root, name)).st_size
            except OSError:
                pass
    _size_cache[path] = (time.monotonic(), total)
    return total


def facts(home: Path) -> dict[str, Any]:
    """What doesn't change while the studio runs."""
    doctor: dict[str, Any] = {}
    try:
        doctor = json.loads((home / "doctor.json").read_text())
    except (OSError, ValueError):
        pass
    gpu = _nvidia_smi() or {}
    disk = shutil.disk_usage(home)
    kernel = platform.release()
    return {
        "gpu": gpu.get("name") or doctor.get("gpu"),
        "driver": gpu.get("driver_version"),
        "vram_gib": doctor.get("vram_gib") or (_num(gpu.get("memory.total")) or 0) / 1024 or None,
        "capability": doctor.get("capability"),
        "cuda": doctor.get("cuda"),
        "torch": doctor.get("torch"),
        "bf16_tflops": doctor.get("bf16_tflops"),  # measured by `slm doctor --bench`
        "cpu": _cpu_model(),
        "cores": os.cpu_count(),
        "ram_gib": round(_meminfo().get("MemTotal", 0) / 2**30, 1),
        "os": f"{platform.system()} {kernel}"
        + (" (WSL2)" if "microsoft" in kernel.lower() else ""),
        "python": platform.python_version(),
        "slm_home": str(home),
        "slm_home_gib": round(dir_size(home) / 2**30, 2),
        "disk_free_gib": round(disk.free / 2**30, 1),
        "disk_total_gib": round(disk.total / 2**30, 1),
    }


class CpuMeter:
    """CPU busy % between consecutive calls, from /proc/stat (what `top` does)."""

    def __init__(self) -> None:
        self._last: tuple[int, int] | None = None
        self._lock = threading.Lock()

    @staticmethod
    def _read() -> tuple[int, int] | None:
        try:
            fields = [int(x) for x in Path("/proc/stat").read_text().split("\n", 1)[0].split()[1:]]
        except (OSError, ValueError):
            return None
        idle = fields[3] + (fields[4] if len(fields) > 4 else 0)  # idle + iowait
        return idle, sum(fields)

    def percent(self) -> float | None:
        with self._lock:
            now = self._read()
            last, self._last = self._last, now
        if now is None or last is None or now[1] == last[1]:
            return None
        return round(100 * (1 - (now[0] - last[0]) / (now[1] - last[1])), 1)


_cpu = CpuMeter()


def live() -> dict[str, Any]:
    """One reading: GPU load, memory, temperature and power; CPU and RAM use."""
    gpu = _nvidia_smi()
    mem = _meminfo()
    total, available = mem.get("MemTotal", 0), mem.get("MemAvailable", 0)
    return {
        "time": time.time(),
        "gpu_util": _num(gpu.get("utilization.gpu")) if gpu else None,
        "vram_used_mib": _num(gpu.get("memory.used")) if gpu else None,
        "vram_total_mib": _num(gpu.get("memory.total")) if gpu else None,
        "gpu_temp_c": _num(gpu.get("temperature.gpu")) if gpu else None,
        "gpu_power_w": _num(gpu.get("power.draw")) if gpu else None,
        "gpu_power_limit_w": _num(gpu.get("power.limit")) if gpu else None,
        "cpu_percent": _cpu.percent(),
        "ram_used_gib": round((total - available) / 2**30, 2) if total else None,
        "ram_total_gib": round(total / 2**30, 2) if total else None,
    }
