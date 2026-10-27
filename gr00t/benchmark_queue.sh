#!/usr/bin/env bash
# Benchmark worker for one GPU. Pops "<task> <task_config>" lines from a shared queue
# file and runs benchmark.sh on each, so several GPUs share one list of jobs and each
# picks up the next job as it frees up.
#
# usage: QUEUE=logs/queue_30k.txt GPU=1 CKPT=RoboTwin-5task_30k-aloha_agilex-joint-0 \
#        CHECKPOINT_NUM=30000 gr00t/benchmark_queue.sh
#   WAIT_PID=<pid>: start only after that process exits (e.g. a run already on the GPU).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
: "${QUEUE:?set QUEUE}" "${GPU:?set GPU}" "${CKPT:?set CKPT}"
export CKPT CHECKPOINT_NUM="${CHECKPOINT_NUM:-last}"

if [ -n "${WAIT_PID:-}" ]; then
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
fi

pop() {  # print and delete the first line of the queue, under a lock
  flock "$QUEUE.lock" bash -c \
    'line=$(head -n 1 "$1"); [ -z "$line" ] || sed -i 1d "$1"; printf "%s" "$line"' _ "$QUEUE"
}

# A line may also set the episode count and seed offset: "<task> <config> [eps] [seed_offset]".
while job="$(pop)" && [ -n "$job" ]; do
  read -r task config eps seed_offset <<< "$job"
  echo "[queue gpu$GPU] $(date -u +%H:%M) start $task $config ${eps:-} ${seed_offset:-}"
  TASKS="$task" TASK_CONFIG="$config" GPUS="$GPU" EPS="${eps:-${EPS:-100}}" \
    SEED_OFFSET="${seed_offset:-0}" bash "$ROOT/gr00t/benchmark.sh"
  echo "[queue gpu$GPU] $(date -u +%H:%M) done $task $config"
done
echo "[queue gpu$GPU] $(date -u +%H:%M) queue empty"
