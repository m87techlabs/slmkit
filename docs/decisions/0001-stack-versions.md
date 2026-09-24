# ADR 0001 — Stack versions and measured baselines

**Status:** accepted (M0 complete)

## Context

Blackwell (`sm_120`) support is recent across the training stack, and every compute estimate in
`docs/DESIGN.md` began as a spec-sheet derivation. Pinning exact versions and recording real
measurements is what makes a run reproducible after a months-long gap — and what turns a budget
from a guess into a number.

## Versions

| Component | Version |
|---|---|
| GPU | NVIDIA GeForce RTX 5080, 16 GB, `sm_120` |
| NVIDIA driver (Windows) | 616.92 |
| CUDA (user-mode, via WSL) | 13.4 |
| WSL | 2.7.11.0 |
| WSL kernel | 6.18.33.2-microsoft-standard-WSL2 |
| Distro | Ubuntu 24.04 LTS (`Ubuntu-ML`) |
| Python | 3.12 |
| torch | **2.11.0+cu128** |
| `torch.cuda.get_arch_list()` | `sm_75, sm_80, sm_86, sm_90, sm_100, **sm_120**` |
| triton | 3.6.0 |
| bitsandbytes | 0.50.2 |

Exact resolved versions of everything else are in `uv.lock`.

## Measured baselines

| Measurement | Value | Method |
|---|---|---|
| **Achieved bf16 matmul** | **121.6 TFLOPS** (re-runs: 111.7–114.3) | `slm doctor --bench`, 8192×8192, fp32 accumulate |
| Spec-sheet estimate it replaces | ~112 TFLOPS | derivation in DESIGN §1 |
| Planning figure (≈35% MFU) | **~43 TFLOPS** | used for every GPU-hour budget in DESIGN §2 |
| ext4 sequential write | 4,655 MB/s | `fio --direct=1 --bs=1M --iodepth=16` |
| ext4 4K random read | 631 MB/s (161,592 IOPS) | `fio --direct=1 --bs=4k --iodepth=32` |
| `/mnt/c` sequential write | 229 MB/s | `dd`, buffered |
| `/mnt/c` 4K random read | 26 MB/s | `dd`, buffered |
| VRAM visible to torch | 15.9 GiB | `slm doctor` |
| RAM visible to the WSL VM | 11.7 GiB → **19.5 GiB** after `.wslconfig` `memory=20GB` | `/proc/meminfo` |

## Decision

Pin torch to a CUDA 12.8+ wheel index in `pyproject.toml` under `[tool.uv.sources]`, so the
requirement is structural rather than a documentation note someone can miss. Treat
`sm_120 in torch.cuda.get_arch_list()` as a hard `slm doctor` failure.

Offer `bitsandbytes` as the optional `optim8bit` extra rather than a default dependency: 8-bit
Adam halves optimizer memory but is not needed at the model sizes this project actually trains,
and memory is not the binding constraint (DESIGN §2).

## Consequences

- **8-bit Adam works on consumer Blackwell.** This was the single largest unknown in the plan;
  `slm doctor` performs a real `AdamW8bit` step on the GPU and it passes. Still A/B it against
  fp32 AdamW before trusting a long run — "it executes" is not "it converges the same".
- The spec-sheet estimate was **9% conservative**, so the design's budgets needed only a mild
  adjustment rather than a rethink. 350M × 7B moved from ~100–110 to ~95–105 GPU-hours.
- Achieved TFLOPS varies by ~8% between sessions (111.7–121.6 measured) with GPU clocks,
  temperature and background load. The planning figure is ~35% of that range, 40–43 TFLOPS;
  budgets use 43, and the trainer replaces it with measured tokens/sec on every run anyway.
  Re-measurement steps are in `docs/runbooks/m0-environment.md`.
- **Gap found in M1:** `torch.compile` needs `python3-dev` (for `Python.h`) as well as a C
  compiler, and M0 never compiled anything, so the gap went unnoticed. The setup script now
  installs it and `slm doctor` runs a real compile.
- The earlier buffered `dd` figure of 1.8 GB/s for 4K reads was page cache, not disk. The honest
  `O_DIRECT` number is 631 MB/s — still ~24× faster than `/mnt/c`, which is what the rule in
  DESIGN §4 rests on.
- `torch` ships kernels for `sm_75` through `sm_120`, so the same lockfile works on hardware
  from Turing onward.

## Alternatives considered

Letting `uv` resolve `torch` from PyPI's default index — rejected: the default wheel may lack
`sm_120` kernels, and the failure appears at the first kernel launch rather than at install time.
