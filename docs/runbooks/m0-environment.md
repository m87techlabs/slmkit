# Runbook M0: the environment

*Companion to [`../concepts/environment.md`](../concepts/environment.md) (the ideas) and
[ADR 0001](../decisions/0001-stack-versions.md) / [ADR 0003](../decisions/0003-wsl-distro.md)
(the decisions). This page tells you how to **check it yourself**.*

M0 adds no ML code. What it produces is a machine you can **trust**: when a training run fails
later, you want to know it failed because of the model and not because of the platform. Every
section below follows the same pattern:

- **What** was set up.
- **How** it was done, meaning the command or file responsible.
- **Verify**: commands to run, with the output the reference machine gives.
- **Why** it was done this way, and what was rejected.

Unless a step says otherwise, run the commands inside the `Ubuntu-ML` distro from the repo
root.

---

## 0. The 60-second check

If you only run one thing, run this:

```bash
make doctor          # = uv run slm doctor
```

Expected (reference machine):

```
  PASS  torch                    2.11.0+cu128 (CUDA 12.8)
  PASS  GPU                      NVIDIA GeForce RTX 5080, sm_120, 15.9 GiB
  PASS  arch support             sm_120 in wheel
  PASS  bf16 matmul              works
  PASS  SDPA                     works
  PASS  bitsandbytes             0.50.2, 8-bit Adam step OK
  PASS  SLM_HOME location        /home/<you>/slm (ext4)
  PASS  free space               945 GiB free
  PASS  HF_HOME                  /home/<you>/slm/cache/hf
  PASS  TRITON_CACHE_DIR         /home/<you>/slm/cache/triton
  PASS  TORCHINDUCTOR_CACHE_DIR  /home/<you>/slm/cache/inductor
  PASS  RAM                      19.5 GiB visible to this VM
  PASS  swap                     2% used

slm doctor: all clear
```

The exit code is `0` when every line is PASS or WARN and `1` if any line is FAIL. That makes
it usable as a gate in a script (`make doctor && uv run slm pretrain ...`), the same way
`terraform validate` gates a `terraform plan`.

The rest of this page explains each of those lines and how to check it without relying on
`doctor`. The goal is that you understand what `doctor` checks instead of taking its word
for it.

---

## 1. A dedicated WSL distro

**What.** A separate WSL2 distro named `Ubuntu-ML` (Ubuntu 24.04) that holds the repo, the
Python environment and all data.

**How.** From Windows PowerShell:

```powershell
wsl --install -d Ubuntu-24.04 --name Ubuntu-ML --no-launch
wsl -d Ubuntu-ML
```

**Verify.**

```powershell
# Windows side
wsl -l -v
```
```
  NAME          STATE     VERSION
  Ubuntu-ML     Running   2          <- must be VERSION 2; WSL1 has no GPU
```

```bash
# inside the distro
echo $WSL_DISTRO_NAME                  # Ubuntu-ML
grep PRETTY /etc/os-release            # Ubuntu 24.04.x LTS
df -T . | tail -1 | awk '{print $2}'   # ext4   <- the repo must not be on /mnt/c
```

**Why.** It gives the environment its own userland: its own filesystem, Python and CUDA
libraries. A PyTorch upgrade here cannot break anything else on the machine, and the whole
environment can be deleted and rebuilt. It does **not** give it its own memory, because all
WSL2 distros share one VM. We rejected a Hyper-V VM (it gets no GPU on client Windows),
Windows-native Python (Triton, bitsandbytes and vLLM are Linux-first) and dual-boot (the
machine is shared). The details are in ADR 0003.

---

## 2. The GPU path: Windows driver in, no Linux driver

**What.** CUDA inside Linux works by forwarding calls to the **Windows** NVIDIA driver. No
NVIDIA driver is installed inside the distro, and none should ever be.

**How.** Nothing needs installing. WSL provides the device `/dev/dxg` and mounts a
`libcuda.so` shim at `/usr/lib/wsl/lib/`. You update the driver on the Windows side, like any
Windows driver.

**Verify.**

```bash
grep -o 'microsoft-standard-WSL2' /proc/version    # the WSL kernel, which has dxgkrnl built in
ls -l /dev/dxg                                     # crw-rw-rw- ... /dev/dxg
ls /usr/lib/wsl/lib/ | grep libcuda                # libcuda.so, libcuda.so.1, ...
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
```
```
name, driver_version, memory.total [MiB]
NVIDIA GeForce RTX 5080, 616.92, 16303 MiB
```

The version `nvidia-smi` shows is the **Windows** driver version. That is expected.

Also confirm that no Linux driver is present:

```bash
dpkg -l | grep -iE 'nvidia-(driver|dkms|kernel)' || echo "none installed (good)"
lsmod | grep -i nvidia                            || echo "no nvidia module (good)"
```

**Why.** A Linux NVIDIA driver replaces the shim with a real `libcuda.so` that expects hardware
it cannot reach, and CUDA stops working until the distro is reinstalled. Guides written for
native Linux say to install the driver first, and that advice is wrong here. The details are in
`../concepts/environment.md` §2.

---

## 3. System packages and `uv`

**What.** A few system tools, plus `uv` as the only way to manage Python.

**How.** Run `scripts/setup-ml-distro.sh`. It is **idempotent**, so running it again changes
nothing, which is the same property you expect from an Ansible play.

**Verify.**

```bash
uv --version
for t in git tmux fio zstd aria2c abc2midi jq; do printf '%-9s ' $t; command -v $t || echo MISSING; done
```

**Why each one is there:**

| Tool | Used for | Milestone |
|---|---|---|
| `uv` | Python environments, locked by `uv.lock` | all |
| `tmux` | a run survives a closed terminal (not a power-off; checkpoints handle that) | M1+ |
| `fio` | honest disk benchmarks (§7) | M0 |
| `zstd`, `aria2c` | download and stream-decompress large corpora without storing them twice | M2, M3 |
| `abc2midi` | grading ABC music output, and turning it into audio you can listen to | M2 |

We chose `uv` over pip/venv/conda because it gives one tool, a lockfile, and fast, reproducible
installs. **Never `pip install` into the system Python**, because that breaks the lockfile
guarantee in the same way that editing live infrastructure by hand breaks Terraform state.

---

## 4. `$SLM_HOME` and the cache variables

**What.** One directory (default `~/slm`, on ext4) that holds all data, artifacts, runs,
exported models and caches. It lives **outside** the repo.

**How.** The setup script creates the layout and appends a marked block to `~/.profile`:

```bash
export SLM_HOME="$HOME/slm"
export HF_HOME="$SLM_HOME/cache/hf"
export TRITON_CACHE_DIR="$SLM_HOME/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$SLM_HOME/cache/inductor"
```

**Verify.**

```bash
env | grep -E '^(SLM_HOME|HF_HOME|TRITON_CACHE_DIR|TORCHINDUCTOR_CACHE_DIR)='
ls $SLM_HOME          # cache datasets doctor.json models packed raw runs tokenizers
df -hT $SLM_HOME      # Type must be ext4, not 9p/drvfs
```

**Why.**
- **Outside the repo**, because artifacts are like Terraform state: large and machine-local, and
  they never go into git. `git clean` in the repo cannot delete a week of training.
- **On ext4, never `/mnt/c`.** §7 below measures the difference. It is about 20–130×, depending
  on file size.
- **Caches set explicitly.** Triton and TorchInductor (the PyTorch compiler) write thousands of
  small compiled-kernel files. The defaults (`~/.cache`, `~/.triton`) happen to be fine today,
  but setting the variables means that moving `$SLM_HOME` moves the caches with it.

---

## 5. PyTorch with Blackwell (`sm_120`) kernels

**What.** `torch 2.11.0+cu128`, installed from PyTorch's CUDA 12.8 wheel index, with the
index pinned in `pyproject.toml`.

**How.**

```toml
# pyproject.toml
[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu128" }
```

then `uv sync --all-extras` (the extra installs bitsandbytes).

**Verify.**

```bash
uv run python -c "import torch
print(torch.__version__, torch.version.cuda)
print(torch.cuda.get_arch_list())
print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
```
```
2.11.0+cu128 12.8
['sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120']
NVIDIA GeForce RTX 5080 (12, 0)
```

`(12, 0)` is the card's **compute capability**, meaning its GPU generation. PyTorch spells it
`sm_120`. The decisive check is that **your card's `sm_XY` appears in the arch list.**

**Why.** A PyTorch build **without** `sm_120` kernels imports fine and reports
`cuda.is_available() == True`, then crashes at the first real GPU operation, often many
minutes into a job. PyTorch's default PyPI wheel is not guaranteed to include it. Pinning the
index in `pyproject.toml` makes the requirement **structural**, so a fresh `uv sync` cannot get
it wrong. Putting it in a README was rejected: a requirement you can forget is one you will
forget.

---

## 6. `slm doctor`, line by line

**What.** [`src/slmkit/doctor.py`](../../src/slmkit/doctor.py) runs one check per known failure
mode and writes what it measured to `$SLM_HOME/doctor.json`.

| Line | What it actually does | The failure it prevents |
|---|---|---|
| `torch` | imports torch, reports the CUDA build | a broken or missing `uv sync` |
| `GPU` | device name, capability, VRAM | wrong GPU, or no GPU visible |
| `arch support` | device `sm_XY` ∈ `get_arch_list()` | §5: a crash at the first kernel launch |
| `bf16 matmul` | a 512×512 bf16 matrix multiply on the GPU | bf16 is the precision slmkit trains in |
| `SDPA` | one causal attention call | SDPA is slmkit's only attention path; if it fails, nothing trains |
| `bitsandbytes` | one **real** `AdamW8bit` optimizer step | 8-bit Adam installed but unable to run on new hardware (WARN only, because it is optional) |
| `SLM_HOME location` | refuses anything under `/mnt/` | §7: a 20–130× I/O slowdown you would blame on the GPU |
| `free space` | warns below 50 GiB | a checkpoint write failing halfway through |
| cache vars | set, and inside `$SLM_HOME` | compiler caches landing on a slow or wrong disk |
| `RAM`, `swap` | reads `/proc/meminfo` | other workloads already using the shared VM memory |

**Verify.** Look at the machine-readable record:

```bash
cat $SLM_HOME/doctor.json
```

Those values are what ADR 0001 records. If you rebuild the environment, run `doctor` and
compare against the ADR. Any difference is the first thing to investigate.

**Why check behaviour, not presence.** Several checks *execute* something (a matmul, an SDPA
call, an optimizer step) rather than just checking that an import works. Each of those
libraries can import cleanly and still fail on a GPU it wasn't built for. This is the same idea
as a readiness probe that sends a real request instead of checking that the process exists.

### Try breaking it

The quickest way to trust a guard is to watch it fail. Point `$SLM_HOME` at a Windows drive for
a single command:

```bash
SLM_HOME=/mnt/c/Users/Public uv run slm doctor; echo "exit=$?"
```
```
  FAIL  SLM_HOME location        /mnt/c/Users/Public is on a Windows drive -- see DESIGN 4
  WARN  HF_HOME                  /home/<you>/slm/cache/hf is outside SLM_HOME
  ...
slm doctor: FAILED -- fix the above before training
exit=1
```

Nothing is written to `/mnt/c`, and your real `$SLM_HOME` is untouched.

---

## 7. Disk: why `/mnt/c` is banned

**What.** Real disk measurements, which back the "no pipeline I/O under `/mnt/`" rule.

**Verify: the demo that makes the point.** This writes and reads 2,000 small files (4 KB each)
on both filesystems, which roughly matches how a venv or a compiler cache behaves:

```bash
uv run python - <<'EOF'
import os, time, pathlib
for base in [pathlib.Path(os.environ["SLM_HOME"]) / "cache", pathlib.Path("/mnt/c/Users/Public")]:
    d = base / "slm_smallfile_test"; d.mkdir(exist_ok=True)
    t = time.perf_counter()
    for i in range(2000): (d / str(i)).write_bytes(b"x" * 4096)
    for i in range(2000): (d / str(i)).read_bytes()
    el = time.perf_counter() - t
    for i in range(2000): (d / str(i)).unlink()
    d.rmdir(); print(f"{base}: {el:.2f}s")
EOF
```
```
/home/<you>/slm/cache: 0.08s
/mnt/c/Users/Public: 10.51s        <- ~130x slower
```

**Verify: the careful benchmark** (the method used for ADR 0001; about 30 seconds, and it cleans
up after itself):

```bash
F=$SLM_HOME/cache/fio.test
fio --name=seqwrite --filename=$F --size=2G --rw=write    --bs=1M --iodepth=16 \
    --ioengine=libaio --direct=1 --runtime=15 --time_based | grep -E 'WRITE:'
fio --name=randread --filename=$F --size=2G --rw=randread --bs=4k --iodepth=32 \
    --ioengine=libaio --direct=1 --runtime=15 --time_based | grep -E 'IOPS='
rm -f $F
```

| | Recorded (ADR 0001) | Re-run 2026-09-23 |
|---|---|---|
| ext4 sequential write | 4,655 MB/s | 4,414 MB/s |
| ext4 4K random read | 631 MB/s, 161K IOPS | 609 MB/s, 149K IOPS |

A variation of about 5% between runs is normal for disk benchmarks.

**Why `--direct=1`.** Without it you are measuring RAM. An early test with buffered `dd`
reported 1.8 GB/s for 4K reads, three times the honest figure, because the file had just been
written and was still in the Linux **page cache** (RAM used to hold recently used file data).
`O_DIRECT` bypasses that cache. The general lesson is to check what a benchmark actually
measures before you trust its number.

**Why `/mnt/c` is so slow.** Every file operation crosses the VM boundary over the 9P protocol,
and each operation pays a fixed cost. Large sequential copies can absorb that cost. Thousands of
small files cannot, and a venv, a Triton cache and checkpoint shards are all thousands of small
files. The only command allowed to write there is `slm export --to-windows`, and it runs once
at the end.

---

## 8. The compute benchmark

**What.** A measured bf16 throughput figure, which replaces a spec-sheet estimate in every
GPU-hour budget.

**How / Verify.**

```bash
uv run slm doctor --bench      # adds one line, ~5 extra seconds
```
```
  PASS  bf16 TFLOPS              113.1 achieved (8192x8192, 1441 iters)
```

**How to read the number.** Multiplying two N×N matrices costs `2·N³` floating-point operations.
For N = 8192, that is 1.1 TFLOP per multiply. The benchmark counts how many multiplies complete
in about 4 seconds and divides.

**Measurements so far** (reference machine):

| When | Result |
|---|---|
| M0 session (ADR 0001) | 121.6 TFLOPS |
| 2026-09-23, four runs | 111.7, 112.1, 113.1, 114.3 TFLOPS |

The spec-sheet estimate was about 112, so the real range is **~112–122 TFLOPS**. Consumer GPUs
adjust their clock speed continuously based on temperature, power draw and the Windows power
plan. A card that is already warm, or a desktop doing other work, costs a few percent. This is
why the ADR states a planning figure rather than a precise peak.

**Why this number matters, and why it is not what training will achieve.** It is a **ceiling**:
large dense matrix multiplies are the best case for a GPU. Real training also spends time on
attention, normalization, the optimizer and data movement. The fraction of the ceiling you
actually get is called **MFU** (see the glossary). Budgets use about 35% of peak: **~43 TFLOPS** in
DESIGN §2, or ~40 if you take today's lower readings. That difference is small next to the
uncertainty in MFU itself. Also, slmkit's trainer will replace even that estimate with measured tokens/sec after 50
steps. A low MFU on the tiny M1 models (10–20%) is expected and is **not** a bug. The full
explanation is in `../concepts/environment.md` §6.

---

## 9. `.wslconfig`: the memory ceiling

**What.** The WSL VM gets up to 20 GB of RAM and 8 GB of swap.

**How.** On Windows, create or edit `%UserProfile%\.wslconfig` (contents in `../DESIGN.md` §3),
then run `wsl --shutdown` from PowerShell. **`wsl --shutdown` stops every distro**, including
any running containers, so do it between sessions.

**Verify.**

```bash
free -h
```
```
               total        used        free      shared  buff/cache   available
Mem:            19Gi       3.7Gi        12Gi        45Mi       3.2Gi        15Gi
Swap:          8.0Gi       117Mi       7.9Gi
```

A total of about 19 GiB means the 20 GB cap is in effect. Before the change, WSL's default gave
11.7 GiB.

**Why.** `memory` is a **ceiling, not a reservation**: an idle WSL gives memory back to
Windows, so a high cap costs nothing when you aren't using it. A cap that is too low pushes
data preprocessing into swap, which looks like a slow pipeline rather than an obvious error.
Remember that **every distro and every Docker Desktop container share this one limit.** That is
why a training session starts by stopping other containers.

---

## 10. Repo tooling

```bash
make lint      # ruff check + ruff format --check + mypy (strict) on src/
make test      # CPU unit tests
```

- `make lint` should pass (`All checks passed!` / `Success: no issues found`).
- `make test` currently exits with **code 5, "no tests ran"**. That is expected, because M0 ships
  no engine code to test. The first unit tests arrive in M1.

Note: newer ruff versions also format Python code blocks inside Markdown. The code blocks in
`docs/` are illustrative sketches with hand-aligned comments, so `pyproject.toml` excludes
`*.md` from `ruff format`.

---

## 11. Before every training session

M0 also defines the routine that starts every session (full reasoning in
`../concepts/environment.md` §7):

1. Stop other containers or quit Docker Desktop, because they share the VM's memory.
2. Pause Windows Update or set active hours. Update reboots end more runs than hardware faults.
3. Stop the machine from sleeping.
4. Optionally cap GPU power at ~280–300 W on the Windows side. You lose 5–10% throughput and
   the machine runs much quieter.
5. `make doctor`, then start the run inside `tmux`.

---

## If something fails

| Symptom | Likely cause | Fix |
|---|---|---|
| `CUDA available: FAIL` | Windows driver too old, or a Linux NVIDIA driver was installed | update the driver on Windows; if a Linux driver was installed, rebuild the distro |
| `arch support: FAIL` | torch came from the wrong index | check `[tool.uv.sources]`; `uv sync --reinstall-package torch` |
| `SLM_HOME: not set` | `~/.profile` not loaded (e.g. a non-login shell) | `source ~/.profile` |
| `bitsandbytes: WARN not installed` | extras not synced | `uv sync --all-extras` |
| `swap: WARN` | other workloads are holding memory | stop containers; check `free -h` |
| TFLOPS well below 110 | GPU busy, hot, power-limited, or on a laptop power plan | close GPU apps; check `nvidia-smi` power and temperature |
