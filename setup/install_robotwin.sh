#!/usr/bin/env bash
# Clone RoboTwin 2.0 (+ XPolicyLab submodule), build its venv, download assets (~15 GB).
#
# RoboTwin's install script uses pip, which reads ~/.config/pip/pip.conf (the company's
# CodeArtifact mirror). That token lasts 12h; refresh it first:
#   aws codeartifact login --tool pip --domain <DOMAIN> --domain-owner <AWS_ACCOUNT_ID> \
#     --repository <REPOSITORY> --region us-west-2
set -euo pipefail
cd "$(dirname "$0")/.."

[ -d RoboTwin ] || git clone --depth 1 https://github.com/RoboTwin-Platform/RoboTwin.git
cd RoboTwin
git submodule update --init --recursive --depth 1 XPolicyLab

uv venv --python 3.10 --seed .venv
source .venv/bin/activate

# pytorch3d and cuRobo compile CUDA extensions: match torch 2.4.1's CUDA 12.1 and
# build for H100 (sm_90) only.
export CUDA_HOME=/usr/local/cuda-12.1 PATH=/usr/local/cuda-12.1/bin:$PATH
export TORCH_CUDA_ARCH_LIST="9.0" MAX_JOBS=48 FORCE_CUDA=1

# _install.sh keeps going after errors; check its output for "ERROR".
bash scripts/_install.sh
bash scripts/_download_assets.sh
