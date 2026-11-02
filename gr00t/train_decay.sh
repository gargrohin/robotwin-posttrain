#!/usr/bin/env bash
# Learning-rate decay run from a checkpoint of an earlier run: load its weights and train
# MAX_STEPS more on the same data while the learning rate falls to 0, to separate
# "the model needed the low-learning-rate end of training" from "the model needed more
# passes over the data" (plan: LOG.md, 2026-11-01).
#
# Same command and settings as gr00t/train.sh -> XPolicyLab train.sh -> finetune.sh, except:
#   --base_model_path FROM (weights only: DeepSpeed can't load ZeRO optimizer state saved on
#     a different number of GPUs, so the AdamW state restarts)
#   --learning_rate LR, --max_steps MAX_STEPS, --save_steps SAVE_STEPS; GR00T's schedule then
#     warms up over 5% of the steps (absorbs the optimizer restart) and decays to 0 (cosine).
#
# usage: FROM=<checkpoint dir> RUN=mix200_decay60k GPUS=4,5 LR=7.0e-5 MAX_STEPS=10000 \
#        SAVE_STEPS=5000 bash gr00t/train_decay.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
: "${FROM:?}" "${RUN:?}" "${GPUS:?}" "${LR:?}" "${MAX_STEPS:?}"
SEED=0
POLICY="$ROOT/RoboTwin/XPolicyLab/policy/GR00T_N17"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPUS"
export GR00T_LEROBOT_HOME="$ROOT/data/lerobot" GR00T_COSMOS_MODEL=nvidia/Cosmos-Reason2-2B
export GR00T_VIDEO_BACKEND=pyav TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
[ -n "${HF_TOKEN:-}" ] || export HF_TOKEN="$(cat ~/.cache/huggingface/token)"
NUM_GPUS=$(tr ',' '\n' <<< "$GPUS" | wc -l)
name="RoboTwin-$RUN-aloha_agilex-joint"
dataset="$GR00T_LEROBOT_HOME/$name"
[ -e "$dataset" ] || ln -s "RoboTwin-${DATASET:-mix200}-aloha_agilex-joint" "$dataset"
out="$POLICY/checkpoints/$name-$SEED"
mkdir -p "$out" "$ROOT/logs"
python3 "$ROOT/gr00t/write_run_info.py" "$out" run="$RUN" started_from="$FROM (weights only)" \
  dataset="$(readlink -f "$dataset")" learning_rate="$LR peak, 5% warm-up, cosine to 0" \
  max_steps="$MAX_STEPS" save_steps="${SAVE_STEPS:-$MAX_STEPS}" gpus="$GPUS" global_batch=64 \
  command="FROM=$FROM RUN=$RUN GPUS=$GPUS LR=$LR MAX_STEPS=$MAX_STEPS SAVE_STEPS=${SAVE_STEPS:-$MAX_STEPS} bash gr00t/train_decay.sh"
cd "$POLICY/gr00t_n17"
source .venv/bin/activate
uv run --no-sync torchrun --nproc_per_node="$NUM_GPUS" --master_port="${MASTER_PORT:-29618}" \
  gr00t/experiment/launch_finetune.py \
  --base_model_path "$FROM" --dataset_path "$dataset" --embodiment_tag NEW_EMBODIMENT \
  --num_gpus "$NUM_GPUS" --output_dir "$out" --save_steps "${SAVE_STEPS:-$MAX_STEPS}" \
  --save_total_limit 10 --max_steps "$MAX_STEPS" --warmup_ratio 0.05 --weight_decay 1e-5 \
  --learning_rate "$LR" --global_batch_size 64 \
  --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-32}" --shard_size 1024 \
  --num_shards_per_epoch 100000 --episode_sampling_rate 0.1 \
  --modality_config_path "$POLICY/configs/aloha_agilex_config.py" --experiment_name "$name-$SEED" \
  2>&1 | tee "$ROOT/logs/train_${RUN}_seed${SEED}.log"
