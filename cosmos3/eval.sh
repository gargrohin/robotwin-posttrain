#!/usr/bin/env bash
# Evaluate a Cosmos3-Nano checkpoint on one RoboTwin task, like gr00t/eval.sh: the Cosmos3_Nano
# XPolicyLab policy server (Cosmos venv) + the RoboTwin sim client (RoboTwin venv).
#
# usage: CHECKPOINT_NUM=1500 TASK=beat_block_hammer POLICY_GPU=3 ENV_GPU=3 TEST_NUM=25 \
#        TASK_CONFIG=demo_clean cosmos3/eval.sh
#   CHECKPOINT_NUM: training iteration, or last (resolved here from latest_checkpoint.txt, so
#     logs and episode<N>.json record the real iteration).
#   CKPT: label in log names and eval_result/ (default cosmos3_nano_mix200).
#   RUN_DIR: cosmos-framework training output dir (default: the mix200 Nano run).
#   POLICY_OVERRIDES: extra deploy.yml overrides, e.g. "num_steps=10 actions_per_call=8".
#   SEED=i: shard i (RoboTwin seeds from 100000*(1+i)); also seeds the sampling noise.
# The GPUs are required (no default): each policy server needs ~35-40 GB.
# Logs: logs/eval_<CKPT>_ckpt-<N>_<TASK>_<CONFIG>[_seed<i>].{server,client}.log, as GR00T's, so
# gr00t/benchmark.sh (EVAL_SH=cosmos3/eval.sh), benchmark_worker.sh and summarize_eval.py work.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

: "${TASK:?set TASK}" "${POLICY_GPU:?set POLICY_GPU}" "${ENV_GPU:?set ENV_GPU}"
CKPT="${CKPT:-cosmos3_nano_mix200}"
RUN_DIR="${RUN_DIR:-/mnt/efs/rohingarg/cri/cosmos3/runs/cosmos3_action_robotwin/action_sft/action_policy_robotwin_nano}"
SEED="${SEED:-0}"
TEST_NUM="${TEST_NUM:-100}"
TASK_CONFIG="${TASK_CONFIG:-demo_clean}"
CHECKPOINT_NUM="${CHECKPOINT_NUM:-last}"
COSMOS_PY=/opt/dlami/nvme/rohingarg/cosmos3-venv/bin/python

if [ "$CHECKPOINT_NUM" = last ]; then
  latest="$(cat "$RUN_DIR/checkpoints/latest_checkpoint.txt")"  # e.g. iter_000001500
  CHECKPOINT_NUM=$((10#${latest#iter_}))
fi

export CUDA_DEVICE_ORDER=PCI_BUS_ID
[ -n "${HF_TOKEN:-}" ] || export HF_TOKEN="$(cat ~/.cache/huggingface/token)"

XPL="$ROOT/RoboTwin/XPolicyLab"
port="$(bash "$XPL/utils/get_free_port.sh")"
action_dim="$(bash "$XPL/utils/get_action_dim.sh" "$ROOT/RoboTwin" aloha_agilex)"
mkdir -p "$ROOT/logs"
log="$ROOT/logs/eval_${CKPT}_ckpt-${CHECKPOINT_NUM}_${TASK}_${TASK_CONFIG}"
[ "$SEED" = 0 ] || log="${log}_seed${SEED}"  # parallel shards (benchmark.sh)

# 1. Cosmos3_Nano policy server (Cosmos venv). LD_LIBRARY_PATH cleared as for training.
(
  cd "$XPL/policy/Cosmos3_Nano"
  CUDA_VISIBLE_DEVICES="$POLICY_GPU" LD_LIBRARY_PATH= PYTHONWARNINGS=ignore::UserWarning \
  exec "$COSMOS_PY" "$XPL/setup_policy_server.py" \
    --config_path "$XPL/policy/Cosmos3_Nano/deploy.yml" --overrides \
    port="$port" host=localhost bench_name=RoboTwin task_name="$TASK" \
    ckpt_name="$CKPT" env_cfg_type=aloha_agilex seed="$SEED" policy_name=Cosmos3_Nano \
    action_type=joint action_dim="$action_dim" checkpoint_num="$CHECKPOINT_NUM" run_dir="$RUN_DIR" \
    ${POLICY_OVERRIDES:-}
) > "$log.server.log" 2>&1 &
server=$!
trap 'kill "$server" 2>/dev/null || true' EXIT
# Loading the EMA weights (tens of GB) from EFS plus the torch.compile warm-up takes minutes.
bash "$XPL/utils/wait_for_policy_server.sh" localhost "$port" "$server" "Policy server" 2400

# 2. RoboTwin simulator client (RoboTwin venv).
source "$ROOT/RoboTwin/.venv/bin/activate"
cd "$ROOT/RoboTwin"
CUDA_VISIBLE_DEVICES="$ENV_GPU" bash scripts/eval_policy.sh \
  --bench_name RoboTwin --task_name "$TASK" --env_cfg_type aloha_agilex \
  --policy_name Cosmos3_Nano --host localhost --port "$port" --protocol ws \
  --eval_batch false --root_dir "$ROOT/RoboTwin" --device_id 0 \
  --additional_info "ckpt_name=$CKPT,action_type=joint,checkpoint_num=$CHECKPOINT_NUM" --seed "$SEED" \
  --task_config "$TASK_CONFIG" --test_num "$TEST_NUM" 2>&1 | tee "$log.client.log"
