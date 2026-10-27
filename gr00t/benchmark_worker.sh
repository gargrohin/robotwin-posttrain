#!/usr/bin/env bash
# Benchmark worker for one GPU that sizes each job to the GPU memory that is free.
# Before popping a job it checks the GPU's free memory and runs as many eval streams
# (shards) as fit, up to MAX_SHARDS. If not even one fits (other users' jobs), it waits
# and checks again, so it can sit on a shared GPU and use whatever is left.
#
# QUEUES is a space-separated list of queue=ckpt=checkpoint_num entries, popped in
# order (the first non-empty queue wins). Queue lines: "<task> <config> [eps] [seed_offset]".
#
# usage: GPU=3 QUEUES="logs/queue_50task_v2.txt=RoboTwin-50task_60k-aloha_agilex-joint-0=60000" \
#        gr00t/benchmark_worker.sh
#   WAIT_PID=<pid>: start only after that process exits (e.g. an older worker on the GPU).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
: "${GPU:?set GPU}" "${QUEUES:?set QUEUES}"
PER_STREAM_MB="${PER_STREAM_MB:-14000}"  # policy server ~7 GB + sim client ~7 GB
HEADROOM_MB="${HEADROOM_MB:-4000}"       # room for other users' jobs to grow
MAX_SHARDS="${MAX_SHARDS:-4}"            # the GPU is ~100% busy rendering at 4

if [ -n "${WAIT_PID:-}" ]; then
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
fi

log() { echo "[worker gpu$GPU] $(TZ=America/New_York date '+%F %H:%M ET') $*"; }

fitting_shards() {
  local free
  free=$(nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader,nounits)
  local n=$(( (free - HEADROOM_MB) / PER_STREAM_MB ))
  [ "$n" -gt "$MAX_SHARDS" ] && n=$MAX_SHARDS
  [ "$n" -lt 0 ] && n=0
  echo "$n"
}

pop() {  # print and delete the first line of a queue, under a lock
  flock "$1.lock" bash -c \
    'line=$(head -n 1 "$1"); [ -z "$line" ] || sed -i 1d "$1"; printf "%s" "$line"' _ "$1"
}

while true; do
  shards=$(fitting_shards)
  if [ "$shards" -lt 1 ]; then sleep 60; continue; fi
  job="" queue="" ckpt="" num=""
  for entry in $QUEUES; do
    IFS== read -r queue ckpt num <<< "$entry"
    job="$(pop "$queue")"
    [ -n "$job" ] && break
  done
  if [ -z "$job" ]; then log "all queues empty"; exit 0; fi
  read -r task config eps seed_offset <<< "$job"
  log "start $task $config ${eps:-100} eps, $shards shards, seed offset ${seed_offset:-0} ($ckpt)"
  CKPT="$ckpt" CHECKPOINT_NUM="$num" TASKS="$task" TASK_CONFIG="$config" GPUS="$GPU" \
    SHARDS="$shards" EPS="${eps:-100}" SEED_OFFSET="${seed_offset:-0}" \
    bash "$ROOT/gr00t/benchmark.sh" || log "benchmark.sh exited with $?"
  log "done $task $config"
done
