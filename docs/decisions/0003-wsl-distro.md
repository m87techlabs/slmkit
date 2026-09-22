# ADR 0003 — Train in a dedicated WSL distro, not Hyper-V or Windows-native

**Status:** accepted (2026-09-22)

## Context

The target platform is a consumer NVIDIA GPU in a Windows workstation that also runs other
workloads (Docker containers, desktop applications). Three options were evaluated for where
training should run.

## Decision

A dedicated WSL2 distro, `Ubuntu-ML`.

**Hyper-V VM — rejected.** It gets no GPU at all. DDA (`Dismount-VMHostAssignableDevice`) is a
Windows Server feature; the cmdlets are present on client Windows but the platform does not
support it. GPU-P (`Add-VMGpuPartitionAdapter`) works on Windows 11 client but targets Windows
guests via WDDM. WSL's GPU access is a WSL-specific arrangement — `/dev/dxg`, `dxgkrnl` built
into the `microsoft-standard-WSL2` kernel, and a Microsoft-supplied `libcuda.so` shim mounted at
`/usr/lib/wsl/lib` — which a stock Ubuntu kernel does not have. Note that WSL2 *is* a Hyper-V
VM; the choice is between a specially-plumbed one and a plain one.

**Windows-native — rejected.** Triton/`torch.compile`, bitsandbytes and vLLM are Linux-first or
Linux-only (vLLM has no native Windows build at all), and the Windows side has no Python, CUDA
toolkit or MSVC installed. The full 31 GB of host RAM is the only real advantage, and it is
recoverable by raising the `.wslconfig` cap.

**Dual-boot native Linux — rejected for now.** Full GPU and full host RAM with no
virtualization overhead, but every other workload on the machine is unavailable until you reboot
back. That suits a dedicated training box, not a shared workstation. It remains the right answer
for anyone whose machine does nothing else.

## Consequences

- All WSL2 distros share one utility VM, one kernel and one memory cap, so a dedicated distro
  isolates the Python/CUDA **userland** only. Separation from other workloads is by **time**:
  stop the other containers, `wsl --shutdown`, raise the cap, run only `Ubuntu-ML`.
- Never install a Linux NVIDIA driver inside WSL; CUDA comes from the Windows driver.
- `/mnt/c` is 20–70× slower than ext4, so all pipeline I/O stays under `$SLM_HOME`.
