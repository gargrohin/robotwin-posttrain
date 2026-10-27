#!/usr/bin/env bash
# Benchmark a GR00T checkpoint: EPS episodes per task, split into SHARDS parallel
# shards. Each shard is a full eval.sh (its own policy server + sim client) with
# SEED=i, i.e. RoboTwin seeds from 100000*(1+i).
#
# usage: CKPT=RoboTwin-5task_30k-aloha_agilex-joint-0 CHECKPOINT_NUM=30000 \
#        TASKS="beat_block_hammer lift_pot" TASK_CONFIG=demo_clean GPUS="6" \
#        SHARDS=4 EPS=100 gr00t/benchmark.sh
#
# Don't replace this with RoboTwin's --eval_batch/--num_workers: those workers share
# one policy server, and the client sends update_obs and get_action as separate
# calls, so concurrent workers can get actions computed from each other's
# observations.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
: "${CKPT:?set CKPT}"
TASKS="${TASKS:-beat_block_hammer lift_pot handover_block stack_blocks_two open_laptop}"
SHARDS="${SHARDS:-4}"
EPS="${EPS:-100}"
SEED_OFFSET="${SEED_OFFSET:-0}"  # shards use SEED=OFFSET..OFFSET+SHARDS-1 (new seeds to top up a run)
read -r -a gpus <<< "${GPUS:-6}"
per_shard=$(( (EPS + SHARDS - 1) / SHARDS ))

for task in $TASKS; do
  echo "[benchmark] $(date -u +%H:%M) $task: $SHARDS shards x $per_shard eps"
  pids=()
  for ((i = 0; i < SHARDS; i++)); do
    gpu="${gpus[$((i % ${#gpus[@]}))]}"
    TASK="$task" SEED="$((SEED_OFFSET + i))" TEST_NUM="$per_shard" POLICY_GPU="$gpu" ENV_GPU="$gpu" \
      bash "$ROOT/gr00t/eval.sh" > /dev/null 2>&1 &
    pids+=($!)
    sleep 20  # stagger startup, so shards don't race for the same free port
  done
  for pid in "${pids[@]}"; do wait "$pid" || echo "[benchmark] shard pid $pid failed"; done
  echo "[benchmark] $(date -u +%H:%M) $task done"
done
python3 "$ROOT/gr00t/summarize_eval.py" "$CKPT" "${CHECKPOINT_NUM:-last}" "${TASK_CONFIG:-demo_clean}"
