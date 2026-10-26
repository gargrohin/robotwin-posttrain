#!/usr/bin/env bash
# Evaluate checkpoints of a (possibly still running) training run as they appear.
#
# usage: CKPT=RoboTwin-5task_30k-aloha_agilex-joint-0 STEPS="10000 20000" \
#        TASK=beat_block_hammer TEST_NUM=10 gr00t/eval_checkpoints.sh
# Other eval.sh settings (POLICY_GPU, ENV_GPU, TASK_CONFIG, ...) pass through.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
: "${CKPT:?set CKPT}" "${STEPS:?set STEPS}"

run_dir="$ROOT/RoboTwin/XPolicyLab/policy/GR00T_N17/checkpoints/$CKPT/$CKPT"
for step in $STEPS; do
  # The trainer writes trainer_state.json last when it saves a checkpoint.
  until [ -f "$run_dir/checkpoint-$step/trainer_state.json" ]; do sleep 60; done
  sleep 60
  echo "[eval_checkpoints] $(date -u +%H:%M) evaluating checkpoint-$step"
  CHECKPOINT_NUM="$step" bash "$ROOT/gr00t/eval.sh"
done
