#!/usr/bin/env bash
# Evals of the mix200_150k checkpoints (plan agreed with the user 2026-10-30, see LOG.md),
# meant to run detached:
#   GPUS="0 3 5 7" WAIT_PID=<pid of gr00t/run_mix200.sh> \
#     setsid -f nohup bash gr00t/eval_mix200.sh > logs/eval_mix200.log 2>&1 < /dev/null
# Waits for training to exit with checkpoint-150000 saved, then starts 4 single-stream
# workers per GPU (MAX_SHARDS=1), so each queue line runs as exactly one shard and
# replays the 50task_60k eval's seeds (queues made by gr00t/paired_queue.py).
# Queues, in priority order:
#   1. checkpoints 150k and 60k on 10 tasks (pick_dual_bottles, place_object_stand,
#      move_pillbottle_pad, place_empty_cup, place_phone_stand, move_can_pot,
#      dump_bin_bigbin, handover_mic, grab_roller, click_bell), clean + randomized,
#      ~20 episodes each on the old eval's seeds
#   2. checkpoint 150k on the other 40 tasks: randomized (~20 episodes) + clean (the
#      first 10 episodes)
#   3. checkpoints 120k, 90k and 30k on the same 10 tasks as in 1, randomized only
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
: "${GPUS:?set GPUS, e.g. GPUS=\"0 3 5 7\"}" "${WAIT_PID:?set WAIT_PID}"
CK=RoboTwin-mix200_150k-aloha_agilex-joint-0
CKPTS="RoboTwin/XPolicyLab/policy/GR00T_N17/checkpoints/$CK/$CK"
log() { echo "[eval_mix200] $(TZ=America/New_York date '+%F %H:%M ET') $*"; }

log "waiting for training (pid $WAIT_PID) to exit"
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
[ -f "$CKPTS/checkpoint-150000/trainer_state.json" ] || { log "no complete checkpoint-150000; not starting"; exit 1; }

Q=logs/queue_mix200
QUEUES="${Q}_s1_150k.txt=$CK=150000 ${Q}_s1_60k.txt=$CK=60000 ${Q}_s2_150k.txt=$CK=150000"
QUEUES+=" ${Q}_s3_120k.txt=$CK=120000 ${Q}_s3_90k.txt=$CK=90000 ${Q}_s3_30k.txt=$CK=30000"
for g in $GPUS; do
  for i in 1 2 3 4; do
    GPU=$g MAX_SHARDS=1 QUEUES="$QUEUES" setsid -f nohup bash gr00t/benchmark_worker.sh \
      > "logs/worker_mix200_gpu${g}_$i.log" 2>&1 < /dev/null
    sleep 20  # stagger startups (free-port picks, checkpoint loads from EFS)
  done
done
log "started 4 workers on each of GPUs $GPUS"
