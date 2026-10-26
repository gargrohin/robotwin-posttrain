#!/usr/bin/env bash
# Evaluate a GR00T checkpoint on one RoboTwin task.
#
# usage: CKPT=RoboTwin-5task_sanity-aloha_agilex-joint-0 TASK=beat_block_hammer \
#        POLICY_GPU=6 ENV_GPU=6 TEST_NUM=100 TASK_CONFIG=demo_clean gr00t/eval.sh
#   CHECKPOINT_NUM=<step|last> picks a checkpoint; TASK_CONFIG=demo_randomized for "hard".
#
# Same two halves as XPolicyLab's eval.sh (GR00T policy server + RoboTwin sim client),
# launched directly because its sim client runs `conda activate` and our RoboTwin env
# is a uv venv. Also points the Cosmos backbone at its HF repo: deploy.yml's
# shared/Cosmos-Reason2-2B doesn't exist here.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

: "${CKPT:?set CKPT}" "${TASK:?set TASK}"
SEED="${SEED:-0}"
TEST_NUM="${TEST_NUM:-100}"
TASK_CONFIG="${TASK_CONFIG:-demo_clean}"
CHECKPOINT_NUM="${CHECKPOINT_NUM:-last}"
POLICY_GPU="${POLICY_GPU:-6}"
ENV_GPU="${ENV_GPU:-6}"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
[ -n "${HF_TOKEN:-}" ] || export HF_TOKEN="$(cat ~/.cache/huggingface/token)"

XPL="$ROOT/RoboTwin/XPolicyLab"
GR00T_DIR="$XPL/policy/GR00T_N17"
port="$(bash "$XPL/utils/get_free_port.sh")"
action_dim="$(bash "$XPL/utils/get_action_dim.sh" "$ROOT/RoboTwin" aloha_agilex)"
mkdir -p "$ROOT/logs"
log="$ROOT/logs/eval_${CKPT}_ckpt-${CHECKPOINT_NUM}_${TASK}_${TASK_CONFIG}"

# 1. GR00T policy server (GR00T venv).
(
  cd "$GR00T_DIR"
  PYTHONPATH="$GR00T_DIR/gr00t_n17:${PYTHONPATH:-}" GR00T_VIDEO_BACKEND=pyav \
  CUDA_VISIBLE_DEVICES="$POLICY_GPU" PYTHONWARNINGS=ignore::UserWarning \
  exec gr00t_n17/.venv/bin/python "$XPL/setup_policy_server.py" \
    --config_path deploy.yml --overrides \
    port="$port" host=localhost bench_name=RoboTwin task_name="$TASK" \
    ckpt_name="$CKPT" env_cfg_type=aloha_agilex seed="$SEED" policy_name=GR00T_N17 \
    action_type=joint action_dim="$action_dim" checkpoint_num="$CHECKPOINT_NUM" \
    cosmos_model_path=nvidia/Cosmos-Reason2-2B
) > "$log.server.log" 2>&1 &
server=$!
trap 'kill "$server" 2>/dev/null || true' EXIT
bash "$XPL/utils/wait_for_policy_server.sh" localhost "$port" "$server" "Policy server" 900

# 2. RoboTwin simulator client (RoboTwin venv).
source "$ROOT/RoboTwin/.venv/bin/activate"
cd "$ROOT/RoboTwin"
CUDA_VISIBLE_DEVICES="$ENV_GPU" bash scripts/eval_policy.sh \
  --bench_name RoboTwin --task_name "$TASK" --env_cfg_type aloha_agilex \
  --policy_name GR00T_N17 --host localhost --port "$port" --protocol ws \
  --eval_batch false --root_dir "$ROOT/RoboTwin" --device_id 0 \
  --additional_info "ckpt_name=$CKPT,action_type=joint" --seed "$SEED" \
  --task_config "$TASK_CONFIG" --test_num "$TEST_NUM" 2>&1 | tee "$log.client.log"
