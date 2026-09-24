"""Environment checks that must pass before any training command runs.

Why this exists: nearly every wasted hour on a new ML box comes from an
environment problem that only surfaces twenty minutes into a run -- a PyTorch
build without kernels for your GPU, a cache directory on a slow filesystem, or
an optimizer that silently is not installed. Each check below corresponds to a
failure that has cost someone a day.

Run with `slm doctor`. Add `--bench` to measure achieved bf16 throughput, which
is the number every compute budget in docs/DESIGN.md should be based on.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

# The compute capability this project targets. Blackwell consumer cards report
# sm_120; a PyTorch wheel built without it will import fine and then fail at the
# first kernel launch, which is why this is checked explicitly rather than
# assumed from `torch.cuda.is_available()`.
REQUIRED_ARCH = "sm_120"
MIN_FREE_GB = 50.0
FAST_FILESYSTEMS = {"ext4", "btrfs", "xfs", "zfs", "overlay"}


class Status(str, Enum):
    OK = "ok"
    WARN = "warn"
    FAIL = "fail"


@dataclass
class Check:
    name: str
    status: Status
    detail: str
    # Only populated for checks that produce a number worth recording in
    # docs/decisions/0001-stack-versions.md.
    data: dict[str, object] = field(default_factory=dict)


def _fs_type(path: Path) -> str:
    """Filesystem type for `path`, by longest-prefix match in /proc/mounts."""
    best_len, best_type = -1, "unknown"
    try:
        for line in Path("/proc/mounts").read_text().splitlines():
            parts = line.split()
            if len(parts) < 3:
                continue
            mount, fstype = parts[1], parts[2]
            if str(path).startswith(mount) and len(mount) > best_len:
                best_len, best_type = len(mount), fstype
    except OSError:
        pass
    return best_type


# --------------------------------------------------------------------------
# GPU and PyTorch
# --------------------------------------------------------------------------


def check_torch() -> list[Check]:
    try:
        import torch
    except ImportError as exc:
        return [Check("torch import", Status.FAIL, f"{exc} -- run `uv sync`")]

    out = [
        Check(
            "torch",
            Status.OK,
            f"{torch.__version__} (CUDA {torch.version.cuda})",
            {"torch": torch.__version__, "cuda": torch.version.cuda},
        )
    ]

    if not torch.cuda.is_available():
        out.append(
            Check(
                "CUDA available",
                Status.FAIL,
                "no CUDA device. Inside WSL this usually means the Windows driver is "
                "too old, or a Linux NVIDIA driver was installed in the distro -- "
                "never do that; see ADR 0003.",
            )
        )
        return out

    name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    out.append(
        Check(
            "GPU",
            Status.OK,
            f"{name}, sm_{major}{minor}, {vram:.1f} GiB",
            {"gpu": name, "capability": f"sm_{major}{minor}", "vram_gib": round(vram, 1)},
        )
    )

    # The decisive check: was this wheel compiled with kernels for our card?
    arch_list = torch.cuda.get_arch_list()
    device_arch = f"sm_{major}{minor}"
    if device_arch in arch_list:
        out.append(
            Check("arch support", Status.OK, f"{device_arch} in wheel", {"arch_list": arch_list})
        )
    else:
        out.append(
            Check(
                "arch support",
                Status.FAIL,
                f"{device_arch} missing from wheel (has: {', '.join(arch_list)}). "
                "Install torch from a CUDA 12.8+ index -- see pyproject.toml.",
                {"arch_list": arch_list},
            )
        )

    try:
        a = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
        (a @ a).float().sum().item()
        out.append(Check("bf16 matmul", Status.OK, "works"))
    except Exception as exc:  # noqa: BLE001 - report anything at all
        out.append(Check("bf16 matmul", Status.FAIL, str(exc)[:200]))

    # SDPA is this project's only attention implementation (ADR 0002 rationale
    # in DESIGN 3); if it is broken, nothing trains.
    try:
        import torch.nn.functional as F

        q = torch.randn(1, 4, 64, 32, device="cuda", dtype=torch.bfloat16)
        F.scaled_dot_product_attention(q, q, q, is_causal=True)
        out.append(Check("SDPA", Status.OK, "works"))
    except Exception as exc:  # noqa: BLE001
        out.append(Check("SDPA", Status.FAIL, str(exc)[:200]))

    out.append(check_compile())
    return out


def check_compile() -> Check:
    """torch.compile end to end: trace, generate a Triton kernel, build it, run it.

    Each step has its own dependency (Triton, a C compiler, Python's headers), and every one
    of them is invisible until the first compile. Found the hard way: without `python3-dev`
    this fails with `Python.h: No such file or directory` deep inside Inductor.
    """
    try:
        import torch
        import torch.nn.functional as F

        fn = torch.compile(lambda x: F.silu(x) * x)
        x = torch.randn(1024, device="cuda")
        torch.testing.assert_close(fn(x), F.silu(x) * x)
        return Check("torch.compile", Status.OK, "Triton kernel built and ran")
    except Exception as exc:  # noqa: BLE001
        text = str(exc)
        # Inductor wraps the real failure in boilerplate; show the line that names it.
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        detail = next(
            (ln for ln in lines if "error" in ln.lower() and "TORCHDYNAMO" not in ln),
            lines[0] if lines else type(exc).__name__,
        )
        hint = " -- install python3-dev (scripts/setup-ml-distro.sh)" if "Python.h" in text else ""
        return Check("torch.compile", Status.FAIL, detail[:160] + hint)


def check_bitsandbytes() -> Check:
    """8-bit Adam halves optimizer memory. Optional, so a failure is a warning.

    Never trust it without an A/B against fp32 AdamW -- see DESIGN 3.
    """
    try:
        import bitsandbytes as bnb
        import torch
    except ImportError:
        return Check("bitsandbytes", Status.WARN, "not installed (8-bit Adam unavailable)")

    try:
        p = torch.nn.Parameter(torch.randn(256, 256, device="cuda"))
        opt = bnb.optim.AdamW8bit([p], lr=1e-4)  # type: ignore[attr-defined,no-untyped-call]
        (p * p).sum().backward()  # type: ignore[no-untyped-call]
        opt.step()  # type: ignore[no-untyped-call]
        return Check(
            "bitsandbytes",
            Status.OK,
            f"{bnb.__version__}, 8-bit Adam step OK",
            {"bitsandbytes": bnb.__version__},
        )
    except Exception as exc:  # noqa: BLE001
        return Check("bitsandbytes", Status.WARN, f"present but failed: {str(exc)[:160]}")


# --------------------------------------------------------------------------
# Storage and memory
# --------------------------------------------------------------------------


def check_paths() -> list[Check]:
    out: list[Check] = []
    raw = os.environ.get("SLM_HOME")
    if not raw:
        return [Check("SLM_HOME", Status.FAIL, "not set -- export SLM_HOME=~/slm")]

    home = Path(raw).expanduser().resolve()
    if not home.exists():
        return [Check("SLM_HOME", Status.FAIL, f"{home} does not exist")]

    # The single most expensive mistake available here. Windows drives reached
    # over 9P are 20-70x slower on small files, which is what a venv, a Triton
    # cache and checkpoint writes all are.
    if str(home).startswith("/mnt/"):
        out.append(
            Check("SLM_HOME location", Status.FAIL, f"{home} is on a Windows drive -- see DESIGN 4")
        )
    else:
        fs = _fs_type(home)
        status = Status.OK if fs in FAST_FILESYSTEMS else Status.WARN
        out.append(
            Check("SLM_HOME location", status, f"{home} ({fs})", {"slm_home": str(home), "fs": fs})
        )

    free_gb = shutil.disk_usage(home).free / 1024**3
    out.append(
        Check(
            "free space",
            Status.OK if free_gb >= MIN_FREE_GB else Status.WARN,
            f"{free_gb:.0f} GiB free",
            {"free_gib": round(free_gb, 1)},
        )
    )

    # Caches default to ~/.cache and ~/.triton; if SLM_HOME is on a different
    # (faster, larger) filesystem they must be moved explicitly.
    for var in ("HF_HOME", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR"):
        val = os.environ.get(var)
        if not val:
            out.append(Check(var, Status.WARN, f"unset -- should point under {home}/cache"))
        elif not Path(val).expanduser().resolve().is_relative_to(home):
            out.append(Check(var, Status.WARN, f"{val} is outside SLM_HOME"))
        else:
            out.append(Check(var, Status.OK, val))
    return out


def check_memory() -> list[Check]:
    """WSL memory pressure. All WSL2 distros share one VM and one cap, so other
    workloads (containers, another distro) eat directly into what training gets.
    """
    out: list[Check] = []
    try:
        info = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, _, rest = line.partition(":")
            info[key] = int(rest.split()[0]) * 1024
        total = info["MemTotal"] / 1024**3
        out.append(
            Check(
                "RAM",
                Status.OK,
                f"{total:.1f} GiB visible to this VM",
                {"ram_gib": round(total, 1)},
            )
        )

        swap_total = info.get("SwapTotal", 0)
        if swap_total:
            used_frac = 1 - info.get("SwapFree", 0) / swap_total
            out.append(
                Check(
                    "swap",
                    Status.OK if used_frac < 0.5 else Status.WARN,
                    f"{used_frac * 100:.0f}% used"
                    + ("" if used_frac < 0.5 else " -- stop other containers before training"),
                )
            )
    except (OSError, KeyError, ZeroDivisionError) as exc:
        out.append(Check("memory", Status.WARN, f"could not read /proc/meminfo: {exc}"))
    return out


# --------------------------------------------------------------------------
# Throughput benchmark
# --------------------------------------------------------------------------


def benchmark_bf16(seconds: float = 4.0, size: int = 8192) -> Check:
    """Measure achieved bf16 matmul throughput in TFLOPS.

    This is M0's headline deliverable. Every GPU-hour estimate in DESIGN 2 is
    currently derived from a spec sheet; this replaces it with a measurement.

    A matmul of two NxN matrices is 2*N^3 floating point operations. We run it
    in a loop and divide by elapsed time. Consumer GeForce cards run bf16 with
    fp32 accumulate at half the fp16-accumulate rate, and PyTorch uses fp32
    accumulate -- which is why the achieved number is far below the marketing
    figure, and why measuring matters.
    """
    try:
        import torch
    except ImportError as exc:
        return Check("bf16 TFLOPS", Status.WARN, str(exc))

    if not torch.cuda.is_available():
        return Check("bf16 TFLOPS", Status.WARN, "no CUDA device")

    torch.backends.cuda.matmul.allow_tf32 = True
    a = torch.randn(size, size, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(size, size, device="cuda", dtype=torch.bfloat16)

    for _ in range(5):  # warm up: first calls include autotuning and allocation
        a @ b
    torch.cuda.synchronize()

    iters, start = 0, time.perf_counter()
    while time.perf_counter() - start < seconds:
        a @ b
        iters += 1
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    tflops = (2 * size**3 * iters) / elapsed / 1e12
    return Check(
        "bf16 TFLOPS",
        Status.OK,
        f"{tflops:.1f} achieved ({size}x{size}, {iters} iters)",
        {"bf16_tflops": round(tflops, 1), "matmul_size": size},
    )


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------

_SYMBOL = {Status.OK: "PASS", Status.WARN: "WARN", Status.FAIL: "FAIL"}


def run(bench: bool = False) -> tuple[list[Check], bool]:
    checks: list[Check] = []
    checks.extend(check_torch())
    checks.append(check_bitsandbytes())
    checks.extend(check_paths())
    checks.extend(check_memory())
    if bench:
        checks.append(benchmark_bf16())
    ok = not any(c.status is Status.FAIL for c in checks)
    return checks, ok


def report(checks: list[Check], ok: bool) -> None:
    width = max(len(c.name) for c in checks)
    for c in checks:
        print(f"  {_SYMBOL[c.status]}  {c.name.ljust(width)}  {c.detail}")
    print()
    print("slm doctor: " + ("all clear" if ok else "FAILED -- fix the above before training"))

    data: dict[str, object] = {}
    for c in checks:
        data.update(c.data)
    if data and (raw := os.environ.get("SLM_HOME")):
        target = Path(raw).expanduser() / "doctor.json"
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            # Merge rather than overwrite: a plain `slm doctor` must not erase the throughput
            # an earlier `--bench` measured, which the trainer uses for MFU.
            previous = json.loads(target.read_text()) if target.is_file() else {}
            target.write_text(json.dumps({**previous, **data}, indent=2, sort_keys=True) + "\n")
            print(f"measurements written to {target}")
        except OSError as exc:
            print(f"(could not write {target}: {exc})")


def main(bench: bool = False) -> int:
    checks, ok = run(bench=bench)
    report(checks, ok)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(bench="--bench" in sys.argv))
