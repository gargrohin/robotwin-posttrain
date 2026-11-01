#!/usr/bin/env bash
# Start 4 single-stream eval workers (MAX_SHARDS=1) on each given GPU once a training run
# has exited with its final checkpoint saved, so the GPUs go straight from training to
# evaluation. Queue lines come from gr00t/paired_queue.py (one line per shard of a
# reference eval, so MAX_SHARDS must stay 1). Meant to run detached:
#   GPUS="4 5" WAIT_PID=<training pid> CHECKPOINT=<final checkpoint dir> TAG=<name> \
#   QUEUES="<queue file>=<ckpt name>=<checkpoint num> ..." \
#     setsid -f nohup bash gr00t/eval_after.sh > logs/eval_<TAG>.log 2>&1 < /dev/null
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
: "${GPUS:?}" "${WAIT_PID:?}" "${CHECKPOINT:?}" "${QUEUES:?}" "${TAG:?}"
log() { echo "[eval_after $TAG] $(TZ=America/New_York date '+%F %H:%M ET') $*"; }

log "waiting for training (pid $WAIT_PID) to exit"
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
[ -f "$CHECKPOINT/trainer_state.json" ] || { log "no complete $CHECKPOINT; not starting"; exit 1; }
for g in $GPUS; do
  for i in 1 2 3 4; do
    GPU=$g MAX_SHARDS=1 QUEUES="$QUEUES" setsid -f nohup bash gr00t/benchmark_worker.sh \
      > "logs/worker_${TAG}_gpu${g}_$i.log" 2>&1 < /dev/null
    sleep 20  # stagger startups (free-port picks, checkpoint loads from EFS)
  done
done
log "started 4 workers on each of GPUs $GPUS"
