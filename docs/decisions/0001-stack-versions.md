# ADR 0001 — Stack versions

**Status:** pending (fill in at M0)

## Context

Blackwell (`sm_120`) support is recent across the training stack. Pinning and recording exact
versions is what makes a run reproducible after a months-long gap.

## Decision

Record at install time, and re-record whenever any of these change:

| Component | Version | Notes |
|---|---|---|
| NVIDIA driver (Windows) | | `nvidia-smi` |
| CUDA (UMD, via WSL) | | |
| WSL | | `wsl --version` |
| Ubuntu | | |
| Python | | |
| torch | | must be a cu128+ wheel |
| `torch.cuda.get_arch_list()` | | **must include `sm_120`** |
| triton | | |
| bitsandbytes | | 8-bit Adam optional; A/B before trusting |
| numpy / pydantic / tokenizers | | see `uv.lock` |

## Measured baselines (M0 exit criteria)

| Measurement | Value | Method |
|---|---|---|
| Achieved BF16 TFLOPS | | large matmul benchmark |
| ext4 sequential write | | `fio --direct=1 --size=16G` |
| ext4 4K random read | | `fio --direct=1 --size=16G` |

The achieved TFLOPS number replaces the ~112 TFLOPS spec-sheet estimate in every GPU-hour
budget in DESIGN §2.

## Consequences

Every ETA before M0 completes is a guess.
