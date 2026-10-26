#!/usr/bin/env bash
# Install the GR00T N1.7 policy env that ships in RoboTwin's XPolicyLab.
#
# Upstream install.sh fails on x86_64 at its `uv run` check: uv run re-syncs the
# project for every platform it declares and wants an aarch64-only torchcodec
# wheel. These are the same install steps, without the sync.
set -euo pipefail
cd "$(dirname "$0")/../RoboTwin/XPolicyLab/policy/GR00T_N17/gr00t_n17"

export CUDA_HOME=/usr/local/cuda-12.9 PATH=/usr/local/cuda-12.9/bin:$PATH MAX_JOBS=48

uv venv --clear --python 3.10
export VIRTUAL_ENV="$PWD/.venv"
uv pip install -e .
uv pip install -e ../../..  # XPolicyLab
uv pip install h5py pyyaml
.venv/bin/python -c "import gr00t, XPolicyLab; print('GR00T + XPolicyLab ok')"
