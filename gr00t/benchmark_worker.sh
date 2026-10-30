#!/usr/bin/env bash
# Benchmark worker for one GPU that sizes each job to the GPU memory that is free.
# Before popping a job it checks the GPU's free memory and runs as many eval streams
# (shards) as fit, up to MAX_SHARDS. If not even one fits (other users' jobs), it waits
# and checks again, so it can sit on a shared GPU and use whatever is left.
# Free memory is the lowest of 6 samples over ~1 min, with a bigger margin when other
# users are on the GPU (their memory use moves). After each job it counts the episodes
# actually recorded; if streams crashed, it requeues the shortfall on new seeds (at most
# MAX_RETRIES times), at seed offset +1000 so it can't reuse a seed index (and log file)
# that another queued job owns. A watchdog stops any stream whose client log has been silent for
# HANG_MIN minutes (healthy streams log every few minutes); its episodes are requeued
# like a crash. It re-execs itself between jobs, so edits to this file apply from the
# next job without a restart.
#
# QUEUES is a space-separated list of queue=ckpt=checkpoint_num entries, popped in
# order (the first non-empty queue wins). Queue lines:
# "<task> <config> [eps] [seed_offset] [retry]".
#
# usage: GPU=3 QUEUES="logs/queue_50task_v2.txt=RoboTwin-50task_60k-aloha_agilex-joint-0=60000" \
#        gr00t/benchmark_worker.sh
#   WAIT_PID=<pid>: start only after that process exits (e.g. an older worker's job).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
: "${GPU:?set GPU}" "${QUEUES:?set QUEUES}"
PER_STREAM_MB="${PER_STREAM_MB:-14000}"        # policy server ~7 GB + sim client ~7 GB
HEADROOM_MB="${HEADROOM_MB:-4000}"             # margin on a GPU only we use
SHARED_HEADROOM_MB="${SHARED_HEADROOM_MB:-10000}"  # margin when other users are on it
MAX_SHARDS="${MAX_SHARDS:-4}"                  # the GPU is ~100% busy rendering at 4
MAX_RETRIES="${MAX_RETRIES:-2}"
HANG_MIN="${HANG_MIN:-60}"
WATCH_SEC="${WATCH_SEC:-300}"   # how often the watchdog checks

if [ -n "${WAIT_PID:-}" ]; then
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
fi
unset WAIT_PID

log() { echo "[worker gpu$GPU] $(TZ=America/New_York date '+%F %H:%M ET') $*"; }

others_on_gpu() {  # number of other users' processes on this GPU
  local uuid pid n=0
  uuid=$(nvidia-smi -i "$GPU" --query-gpu=uuid --format=csv,noheader)
  for pid in $(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader \
               | awk -F', ' -v u="$uuid" '$1 == u {print $2}'); do
    [ "$(ps -o user= -p "$pid" 2>/dev/null)" = "$USER" ] || n=$((n + 1))
  done
  echo "$n"
}

fitting_shards() {
  local free min=999999 i headroom=$HEADROOM_MB
  for i in 1 2 3 4 5 6; do
    free=$(nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader,nounits)
    [ "$free" -lt "$min" ] && min=$free
    [ "$i" -lt 6 ] && sleep 10
  done
  [ "$(others_on_gpu)" -gt 0 ] && headroom=$SHARED_HEADROOM_MB
  local n=$(( (min - headroom) / PER_STREAM_MB ))
  [ "$n" -gt "$MAX_SHARDS" ] && n=$MAX_SHARDS
  [ "$n" -lt 0 ] && n=0
  echo "$n"
}

pop() {  # print and delete the first line of a queue, under a lock
  flock "$1.lock" bash -c \
    'line=$(head -n 1 "$1"); [ -z "$line" ] || sed -i 1d "$1"; printf "%s" "$line"' _ "$1"
}

descendants() { local p; for p in $(ps --ppid "$1" -o pid= 2>/dev/null); do descendants "$p"; echo "$p"; done; }

watchdog() {  # stop streams of benchmark.sh $1 whose client log is silent for HANG_MIN
  local bench=$1 ckpt=$2 num=$3 task=$4 config=$5 now e seed f last pids
  while kill -0 "$bench" 2>/dev/null; do
    sleep "$WATCH_SEC"
    now=$(date +%s)
    for e in $(ps --ppid "$bench" -o pid= 2>/dev/null); do
      seed=$(tr '\0' '\n' < /proc/"$e"/environ 2>/dev/null | sed -n 's/^SEED=//p')
      [ -n "$seed" ] || continue
      f="$ROOT/logs/eval_${ckpt}_ckpt-${num}_${task}_${config}"
      [ "$seed" -eq 0 ] || f+="_seed$seed"
      f+=".client.log"
      grep -q 'Final success rate' "$f" 2>/dev/null && continue
      last=$(stat -c %Y "$f" 2>/dev/null || stat -c %Y /proc/"$e" 2>/dev/null || echo "$now")
      if [ $(( now - last )) -gt $(( HANG_MIN * 60 )) ]; then
        pids="$(descendants "$e") $e"
        log "watchdog: $task $config seed $seed silent for $(( (now - last) / 60 )) min, stopping it"
        kill -TERM $pids 2>/dev/null || true; sleep 20; kill -KILL $pids 2>/dev/null || true
      fi
    done
  done
}

recorded_episodes() {  # episodes with a final result for seeds offset..offset+shards-1
  local ckpt=$1 num=$2 task=$3 config=$4 offset=$5 shards=$6 seed f n total=0
  for ((seed = offset; seed < offset + shards; seed++)); do
    f="$ROOT/logs/eval_${ckpt}_ckpt-${num}_${task}_${config}"
    [ "$seed" -eq 0 ] || f+="_seed$seed"
    f+=".client.log"
    n=$(sed 's/\x1b\[[0-9;]*m//g' "$f" 2>/dev/null \
        | grep -oE 'Final success rate: [0-9]+/[0-9]+' | tail -1 | cut -d/ -f2)
    total=$((total + ${n:-0}))
  done
  echo "$total"
}

shards=0
while [ "$shards" -lt 1 ]; do
  shards=$(fitting_shards)
  [ "$shards" -lt 1 ] && sleep 50
done
job="" queue="" ckpt="" num=""
for entry in $QUEUES; do
  IFS== read -r queue ckpt num <<< "$entry"
  job="$(pop "$queue")"
  [ -n "$job" ] && break
done
if [ -z "$job" ]; then log "all queues empty"; exit 0; fi
read -r task config eps seed_offset retry <<< "$job"
eps=${eps:-100} seed_offset=${seed_offset:-0} retry=${retry:-0}
log "start $task $config $eps eps, $shards shards, seed offset $seed_offset, retry $retry ($ckpt)"
CKPT="$ckpt" CHECKPOINT_NUM="$num" TASKS="$task" TASK_CONFIG="$config" GPUS="$GPU" \
  SHARDS="$shards" EPS="$eps" SEED_OFFSET="$seed_offset" \
  bash "$ROOT/gr00t/benchmark.sh" &
bench=$!
watchdog "$bench" "$ckpt" "$num" "$task" "$config" &
dog=$!
wait "$bench" || log "benchmark.sh exited with $?"
kill "$dog" 2>/dev/null || true
got=$(recorded_episodes "$ckpt" "$num" "$task" "$config" "$seed_offset" "$shards")
if [ "$got" -lt "$eps" ] && [ "$retry" -lt "$MAX_RETRIES" ]; then
  line="$task $config $((eps - got)) $((seed_offset + 1000 * (retry + 1))) $((retry + 1))"
  flock "$queue.lock" bash -c 'echo "$2" >> "$1"' _ "$queue" "$line"
  log "done $task $config: $got/$eps episodes recorded; requeued \"$line\""
else
  log "done $task $config: $got/$eps episodes recorded"
fi
exec bash "$ROOT/gr00t/benchmark_worker.sh"  # pick up edits to this file
