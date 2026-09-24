# The environment

*Milestone M0. Read `../GLOSSARY.md` for any unfamiliar term. To check each piece on your own
machine, follow `../runbooks/m0-environment.md`.*

Before any of the interesting work, you need a machine that can actually train. This page
explains what we set up, why each piece is there, and what breaks without it.

The short version: **a GPU inside WSL works differently than a GPU on native Linux**, and almost
every setup failure comes from not knowing how.

---

## 1. Why a separate distro

We created `Ubuntu-ML` rather than training in an existing distro. The reason is not isolation of
resources — it is isolation of *userland*.

All WSL2 distros share **one virtual machine, one Linux kernel, and one memory cap**. A second
distro gets you:

- a separate filesystem, so datasets and a multi-gigabyte virtual environment do not entangle
  with anything else;
- a separate Python and CUDA installation, so upgrading PyTorch cannot break unrelated tooling;
- a clean "delete and start over" when a dependency upgrade goes wrong.

It does **not** get you separate memory. If another distro is running containers, they are eating
from the same pool. That is why the training routine starts with "stop the other containers" —
separation here is by *time*, not allocation.

---

## 2. How the GPU gets into Linux

On a normal Linux box you install an NVIDIA driver and `/dev/nvidia0` appears. **Under WSL,
none of that happens**, and trying to make it happen is the most common way to break everything.

What actually exists:

```
/dev/dxg                          a paravirtualized GPU device
/usr/lib/wsl/lib/libcuda.so       a stub library, mounted in by WSL itself
kernel: ...-microsoft-standard-WSL2   with dxgkrnl compiled in
```

`libcuda.so` here is not a real driver. It is a shim: CUDA calls made inside Linux are marshalled
across `/dev/dxg` to the **Windows** driver, which owns the hardware. The Linux side never talks
to the GPU directly.

Two consequences:

**Never install a Linux NVIDIA driver inside WSL.** It will overwrite the shim with a real
`libcuda.so` that expects hardware it cannot reach, and CUDA stops working. Recovery means
reinstalling the distro. Your GPU driver is the **Windows** one; update it on the Windows side.

**This is also why a hand-built Hyper-V Linux VM gets no GPU at all.** The plumbing above is
specific to WSL, not a general Hyper-V feature — see `../decisions/0003-wsl-distro.md`.

---

## 3. Why the CUDA version matters so much

A GPU has a **compute capability** — a generation identifier. Blackwell consumer cards report
`sm_120`. When PyTorch is built, it compiles GPU kernels for a specific list of these.

The trap: a PyTorch wheel built without `sm_120` **imports perfectly and reports
`torch.cuda.is_available() == True`**, then fails at the first real kernel launch. The error is
usually cryptic and arrives twenty minutes into a job.

So `slm doctor` checks the list directly:

```python
>>> torch.cuda.get_arch_list()
['sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120']
```

If your device's capability is missing there, nothing else matters. This is why `pyproject.toml`
pins the CUDA 12.8+ wheel index structurally rather than mentioning it in a README:

```toml
[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu128" }
```

A requirement you can forget is a requirement you will forget.

---

## 4. Where files live, and why it is not a detail

WSL can see Windows drives at `/mnt/c`. It is tempting to keep your repo there so Windows tools
can reach it. **Do not.** Measured on the reference machine:

| | Sequential write | 4K random read |
|---|---|---|
| ext4 (inside the distro) | 4,655 MB/s | **631 MB/s** (161K IOPS) |
| `/mnt/c` (Windows drive) | 229 MB/s | **26 MB/s** |

Every file operation on `/mnt/c` crosses the VM boundary over a protocol called 9P, paying a
cost *per operation*. Large sequential reads survive it; huge numbers of small files do not.

The things that hurt are exactly the things ML does constantly: a virtual environment with tens
of thousands of small files, compiler caches, and checkpoint writes that block training while
they save.

So `$SLM_HOME` lives on ext4, and `slm doctor` **fails** if it is under `/mnt/`. The code enforces
it too, because a rule that only exists in documentation is a rule that gets broken at 1am.

One subtlety worth internalizing. An early buffered `dd` test reported 1.8 GB/s for 4K reads —
nearly 3× the honest figure. That was the **page cache**: the file had just been written and was
still in RAM. Benchmarking storage requires `O_DIRECT` (`fio --direct=1`) to bypass it. Measuring
the wrong thing confidently is worse than not measuring.

### The cache variables

PyTorch, Triton and Hugging Face default their caches to `~/.cache` and `~/.triton`. Those
defaults are fine — until someone puts the repo somewhere unusual and the caches follow. We point
them explicitly under `$SLM_HOME`:

```bash
export HF_HOME="$SLM_HOME/cache/hf"
export TRITON_CACHE_DIR="$SLM_HOME/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$SLM_HOME/cache/inductor"
```

---

## 5. What `slm doctor` actually protects against

Each check exists because of a specific failure that costs real time:

| Check | The failure it catches |
|---|---|
| `sm_120` in arch list | Wheel without kernels for your GPU — fails at first launch, not at install |
| bf16 matmul | The precision the whole project trains in |
| SDPA | slmkit's only attention implementation; broken means nothing trains |
| `torch.compile` | Triton, a C compiler and Python's headers are all needed at the first compile, and none is checked at install time |
| bitsandbytes step | 8-bit Adam present but non-functional on new hardware |
| `$SLM_HOME` not under `/mnt/` | A 24× storage slowdown you would blame on the GPU |
| Cache variables | Compiler caches quietly landing on a slow filesystem |
| Swap usage | Other workloads have the RAM; preprocessing will thrash |

It writes everything it learns to `$SLM_HOME/doctor.json`, which is what populates
`../decisions/0001-stack-versions.md`.

---

## 6. Reading the benchmark

`slm doctor --bench` multiplies two 8192×8192 bf16 matrices repeatedly and divides total
floating-point operations by elapsed time. A matmul of two N×N matrices costs `2 × N³` operations.

On the reference machine: **121.6 TFLOPS**, and 112–114 on later sessions. Consumer cards adjust
their clocks to temperature and power, so expect a spread of several percent between runs.

That number is a **ceiling, not a throughput prediction**, and confusing the two will make every
estimate you produce roughly three times too optimistic.

The benchmark does nothing but large dense matmuls — the single operation GPUs are best at. Real
training also runs attention, normalization, the optimizer step, and moves data around. The
fraction of the ceiling you actually achieve is called **MFU** (Model FLOPs Utilization), and
35–45% is respectable.

So the planning figure is **~43 TFLOPS** — about 35% of measured peak — and that is what every
GPU-hour budget in `../DESIGN.md` §2 uses.

MFU also falls with model size. Tensor cores need large matrices to stay busy; a model with
`d_model=256` leaves most of the hardware idle no matter how good your code is. Expect 10–20% at
the smallest presets. **Do not treat a low MFU on a tiny model as a bug.**

### Was the estimate any good?

Before measuring, the design derived ~112 TFLOPS from the spec sheet, reasoning that consumer
GeForce cards run bf16 with fp32 accumulate at half the fp16-accumulate rate, and that PyTorch
uses fp32 accumulate.

Measured: 121.6. The estimate was **9% conservative** — sound enough that the budgets needed
adjusting, not rethinking.

This is the useful lesson. Estimating from first principles is worth doing, because it tells you
whether a plan is plausible before you spend a weekend on it. But you replace the estimate the
moment you can, and the trainer re-estimates again from measured tokens/sec in the first ten
minutes of every run — because even a measured ceiling is not a prediction of *your* model's
throughput.

---

## 7. The session routine

Training does not run unattended here. The workstation is shared and gets powered off, so each
session starts deliberately:

1. Stop other containers, or quit Docker Desktop, to free RAM and VRAM.
2. Pause Windows Update, or set active hours to cover the session. Update reboots end more runs
   than hardware faults do.
3. Prevent the machine from sleeping.
4. Optionally cap the GPU at ~280–300 W from the Windows side. You lose 5–10% throughput and run
   much cooler and quieter — a good trade on a machine you are sitting next to.
5. Start the run inside `tmux`.

`tmux` survives a disconnected terminal, **not a power-off**. The thing that survives a power-off
is the checkpoint, which is why the trainer writes full state every 20 minutes and why resume is
the default path rather than an error path. See `the-training-loop.md` when it lands.
