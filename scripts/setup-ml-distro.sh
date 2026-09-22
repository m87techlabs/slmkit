#!/usr/bin/env bash
# Prepare a WSL2 distro for slmkit training. Idempotent: safe to re-run.
#
#   wsl --install -d Ubuntu-24.04 --name Ubuntu-ML --no-launch
#   wsl -d Ubuntu-ML
#   bash scripts/setup-ml-distro.sh
#
# What it does NOT do, on purpose:
#   - install an NVIDIA driver. CUDA comes from the Windows driver through
#     /dev/dxg. Installing a Linux driver inside WSL breaks GPU access. See
#     docs/decisions/0003-wsl-distro.md.
#   - edit .wslconfig. That lives on the Windows side and needs `wsl --shutdown`
#     to apply, which terminates every distro. See docs/DESIGN.md section 3.
set -euo pipefail

SLM_HOME_DEFAULT="$HOME/slm"
SLM_HOME="${SLM_HOME:-$SLM_HOME_DEFAULT}"

say() { printf '\n==> %s\n' "$1"; }

say "Checking this is WSL with a GPU"
grep -qi microsoft /proc/version || { echo "not running under WSL"; exit 1; }
[ -e /dev/dxg ] || echo "WARNING: /dev/dxg missing -- the GPU will not be visible"

say "Installing system packages"
# fio      -- measure real disk throughput (M0 exit criterion)
# aria2/zstd -- fetch and stream-decompress large corpora without staging them
# abcmidi  -- abc2midi, for grading and listening to the ABC music project
# tmux     -- long runs survive a disconnected terminal (not a power-off; that
#             is what checkpoints are for)
export DEBIAN_FRONTEND=noninteractive
sudo apt-get update -qq
sudo apt-get install -y -qq --no-install-recommends \
    git build-essential tmux aria2 zstd fio curl ca-certificates abcmidi jq unzip

say "Installing uv"
if ! command -v uv >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/uv" ]; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

say "Creating \$SLM_HOME at $SLM_HOME"
mkdir -p "$SLM_HOME"/{raw,datasets,tokenizers,packed,runs,models,cache/{hf,triton,inductor}}

say "Configuring the shell environment"
MARKER="# --- slmkit ---"
if grep -qF "$MARKER" "$HOME/.profile" 2>/dev/null; then
    echo "already present in ~/.profile, leaving it alone"
else
    cat >> "$HOME/.profile" <<EOF

$MARKER
export PATH="\$HOME/.local/bin:\$PATH"
export SLM_HOME="$SLM_HOME"
# Every cache stays on ext4 under \$SLM_HOME. Defaults would put them in
# ~/.cache and ~/.triton, and on a Windows drive they are 20-70x slower.
export HF_HOME="\$SLM_HOME/cache/hf"
export TRITON_CACHE_DIR="\$SLM_HOME/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="\$SLM_HOME/cache/inductor"
EOF
    echo "appended to ~/.profile"
fi

say "Done"
cat <<EOF

Next:
  1. source ~/.profile
  2. Set up git access if you have not already (an SSH key for this distro, or
     \`gh auth login\`), then clone the repo somewhere on ext4 -- never /mnt/c.
  3. cd into the repo and run:
         uv sync
         uv run slm doctor --bench
  4. Record the results in docs/decisions/0001-stack-versions.md.

EOF
