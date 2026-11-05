#!/usr/bin/env bash
# Evaluate a FLUX 3 Action checkpoint on one RoboTwin task, like gr00t/eval.sh: the Flux3_Action
# XPolicyLab policy server (flux-action venv) + the RoboTwin sim client (RoboTwin venv).
#
# usage: CHECKPOINT_NUM=1500 TASK=beat_block_hammer POLICY_GPU=3 ENV_GPU=3 TEST_NUM=25 \
#        TASK_CONFIG=demo_clean flux3/eval.sh
#   CHECKPOINT_NUM: training step, or last (resolved here to the highest step-<N> in EXPORT_ROOT, so
#     logs and episode<N>.json record the real step).
#   CKPT: label in log names and eval_result/ (default flux3_mix200_b512).
#   EXPORT_ROOT: folder of inference exports step-<N>/ (flux-action export-checkpoint --profile ema_0p10
#     --dtype bfloat16).
#   FLUX_WEIGHTS: flux-3-action-base folder with video_vae.safetensors and text_encoder/.
#   POLICY_OVERRIDES: extra deploy.yml overrides, e.g. "num_inference_steps=8 actions_per_call=16".
#   SEED=i: shard i (RoboTwin seeds from 100000*(1+i)); also sets the noise seeds.
# The GPUs are required (no default): each policy server needs ~30 GB.
# Logs: logs/eval_<CKPT>_ckpt-<N>_<TASK>_<CONFIG>[_seed<i>].{server,client}.log, as GR00T's, so
# gr00t/benchmark.sh (EVAL_SH=flux3/eval.sh), benchmark_worker.sh and summarize_eval.py work.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

: "${TASK:?set TASK}" "${POLICY_GPU:?set POLICY_GPU}" "${ENV_GPU:?set ENV_GPU}"
CKPT="${CKPT:-flux3_mix200_b512}"
EXPORT_ROOT="${EXPORT_ROOT:-/opt/dlami/nvme/rohingarg/flux3-eval/exports/f3a-mix200-b512-2n-v2}"
FLUX_WEIGHTS="${FLUX_WEIGHTS:-/opt/dlami/nvme/rohingarg/flux3-weights}"
FLUX_PY="${FLUX_PY:-/opt/dlami/nvme/rohingarg/flux3-venv/bin/python}"
SEED="${SEED:-0}"
TEST_NUM="${TEST_NUM:-100}"
TASK_CONFIG="${TASK_CONFIG:-demo_clean}"
CHECKPOINT_NUM="${CHECKPOINT_NUM:-last}"

if [ "$CHECKPOINT_NUM" = last ]; then
  CHECKPOINT_NUM="$(ls -d "$EXPORT_ROOT"/step-*/manifest.json | sed -E 's|.*/step-([0-9]+)/manifest.json|\1|' | sort -n | tail -1)"
fi
[ -f "$EXPORT_ROOT/step-$CHECKPOINT_NUM/manifest.json" ] || { echo "no export $EXPORT_ROOT/step-$CHECKPOINT_NUM" >&2; exit 1; }

export CUDA_DEVICE_ORDER=PCI_BUS_ID
[ -n "${HF_TOKEN:-}" ] || export HF_TOKEN="$(cat ~/.cache/huggingface/token)"

XPL="$ROOT/RoboTwin/XPolicyLab"
port="$(bash "$XPL/utils/get_free_port.sh")"
action_dim="$(bash "$XPL/utils/get_action_dim.sh" "$ROOT/RoboTwin" aloha_agilex)"
mkdir -p "$ROOT/logs"
log="$ROOT/logs/eval_${CKPT}_ckpt-${CHECKPOINT_NUM}_${TASK}_${TASK_CONFIG}"
[ "$SEED" = 0 ] || log="${log}_seed${SEED}"  # parallel shards (benchmark.sh)

# 1. Flux3_Action policy server (flux-action venv; local encoders only, no Hub downloads).
(
  cd "$XPL/policy/Flux3_Action"
  CUDA_VISIBLE_DEVICES="$POLICY_GPU" LD_LIBRARY_PATH= HF_HUB_OFFLINE=1 PYTHONWARNINGS=ignore::UserWarning \
  exec "$FLUX_PY" "$XPL/setup_policy_server.py" \
    --config_path "$XPL/policy/Flux3_Action/deploy.yml" --overrides \
    port="$port" host=localhost bench_name=RoboTwin task_name="$TASK" \
    ckpt_name="$CKPT" env_cfg_type=aloha_agilex seed="$SEED" policy_name=Flux3_Action \
    action_type=joint action_dim="$action_dim" checkpoint_num="$CHECKPOINT_NUM" export_root="$EXPORT_ROOT" \
    video_vae_path="$FLUX_WEIGHTS/video_vae.safetensors" text_encoder_path="$FLUX_WEIGHTS/text_encoder" \
    ${POLICY_OVERRIDES:-}
) > "$log.server.log" 2>&1 &
server=$!
trap 'kill "$server" 2>/dev/null || true' EXIT
# Loading the DiT (14 GB), the text encoder (9 GB) and the VAE, plus one warm-up request.
bash "$XPL/utils/wait_for_policy_server.sh" localhost "$port" "$server" "Policy server" 2400

# 2. RoboTwin simulator client (RoboTwin venv).
source "$ROOT/RoboTwin/.venv/bin/activate"
cd "$ROOT/RoboTwin"
CUDA_VISIBLE_DEVICES="$ENV_GPU" bash scripts/eval_policy.sh \
  --bench_name RoboTwin --task_name "$TASK" --env_cfg_type aloha_agilex \
  --policy_name Flux3_Action --host localhost --port "$port" --protocol ws \
  --eval_batch false --root_dir "$ROOT/RoboTwin" --device_id 0 \
  --additional_info "ckpt_name=$CKPT,action_type=joint,checkpoint_num=$CHECKPOINT_NUM" --seed "$SEED" \
  --task_config "$TASK_CONFIG" --test_num "$TEST_NUM" 2>&1 | tee "$log.client.log"
