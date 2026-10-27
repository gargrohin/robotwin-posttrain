#!/usr/bin/env bash
# Fine-tune GR00T N1.7 on the 5-task RoboTwin dataset through XPolicyLab's train.sh.
#
# usage: RUN=5task_sanity GPUS=5,6 MAX_STEPS=2000 GLOBAL_BATCH_SIZE=32 gr00t/train.sh
#        DATASET=50task picks the prepared dataset (default 5task).
#
# XPolicyLab names both the dataset and the checkpoint dir after <ckpt_name>, so each
# RUN gets a symlink to a prepared dataset (RoboTwin-<DATASET>-aloha_agilex-joint) and
# its own checkpoints/RoboTwin-<RUN>-aloha_agilex-joint-<SEED>/.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

RUN="${RUN:-5task_sanity}"
SEED="${SEED:-0}"
GPUS="${GPUS:-5,6}"

export CUDA_DEVICE_ORDER=PCI_BUS_ID  # GPU ids match nvidia-smi
export GR00T_LEROBOT_HOME="$ROOT/data/lerobot"
export MAX_STEPS="${MAX_STEPS:-2000}" SAVE_STEPS="${SAVE_STEPS:-1000}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
export USE_WANDB=0                       # logs stay local
export MASTER_PORT="${MASTER_PORT:-29617}"  # avoid torchrun's default on a shared box
# Cosmos-Reason2-2B is gated. $HF_HOME (/mnt/nvme/...) has no token file, but the
# default location does.
[ -n "${HF_TOKEN:-}" ] || export HF_TOKEN="$(cat ~/.cache/huggingface/token)"

dataset="$GR00T_LEROBOT_HOME/RoboTwin-$RUN-aloha_agilex-joint"
[ -e "$dataset" ] || ln -s "RoboTwin-${DATASET:-5task}-aloha_agilex-joint" "$dataset"

mkdir -p "$ROOT/logs"
cd "$ROOT/RoboTwin/XPolicyLab/policy/GR00T_N17"
bash train.sh RoboTwin "$RUN" aloha_agilex joint "$SEED" "$GPUS" 2>&1 \
  | tee "$ROOT/logs/train_${RUN}_seed${SEED}.log"
