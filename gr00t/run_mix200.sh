#!/usr/bin/env bash
# Clean + randomized training run (2026-10-30), end to end, meant to run detached:
#   setsid -f nohup bash gr00t/run_mix200.sh > logs/run_mix200.log 2>&1 < /dev/null
# 1. Fetch 200 randomized demos per task from RoboTwin's raw zips, in XPolicyLab format
#    (gr00t/fetch_randomized.py).
# 2. Convert them to LeRobot and merge with the already-converted 50-task clean demos:
#    12,500 episodes -> data/lerobot/RoboTwin-mix200-aloha_agilex-joint (prepare_data.sh).
# 3. GPUs are fixed: 3, 0, 4, 7 (the user's choice). GPU 3's vLLM server is stopped by
#    hand, not by this script. Training starts only when every GPU has >= MIN_FREE_MB free
#    (our ~22 GB + >= 23 GB spare for the other users' jobs); until then it waits and
#    re-checks every 10 min, and never switches to other GPUs.
# 4. Train 150k steps, checkpoint every 15k, keep all; on a crash, wait and resume from the
#    last checkpoint (up to 3 attempts).
# Each step is skipped when its output exists, so the script can simply be re-run.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
RUN=mix200_150k DATASET=mix200 MAX_STEPS=150000
GPUS=3,0,4,7
MIN_FREE_MB="${MIN_FREE_MB:-45000}"
CKPTS="RoboTwin/XPolicyLab/policy/GR00T_N17/checkpoints/RoboTwin-$RUN-aloha_agilex-joint-0/RoboTwin-$RUN-aloha_agilex-joint-0"
export HF_TOKEN="$(cat ~/.cache/huggingface/token)"
log() { echo "[mix200] $(TZ=America/New_York date '+%F %H:%M ET') $*"; }
TASKS="$(ls RoboTwin/data/demo_clean | tr '\n' ' ')"

# 1-2. Data.
if [ ! -f "data/lerobot/RoboTwin-$DATASET-aloha_agilex-joint/meta/video_frames.json" ]; then
  log "fetching 200 randomized demos per task"
  RoboTwin/.venv/bin/python gr00t/fetch_randomized.py --n 200 --group 50 --config demo_randomized \
    --jobs 24 $TASKS || { log "fetch failed"; exit 1; }
  log "building the merged dataset"
  TASKS="$TASKS" CONFIG="demo_randomized0 demo_randomized1 demo_randomized2 demo_randomized3" \
    SRC=robotwin_mix200_aloha CKPT_NAME=$DATASET JOBS=24 \
    EXTRA_PARTS=robotwin_50task_aloha_parts EXPECTED=12500 bash gr00t/prepare_data.sh \
    || { log "dataset prep failed"; exit 1; }
fi
log "dataset ready"

min_free() {  # lowest free memory (MiB) on GPU $1 over ~1 min
  local i m=999999 f
  for i in 1 2 3 4 5 6; do
    f=$(nvidia-smi -i "$1" --query-gpu=memory.free --format=csv,noheader,nounits)
    [ "$f" -lt "$m" ] && m=$f; sleep 10
  done
  echo "$m"
}

others_on() {  # other users' PIDs on GPU $1
  local u; u=$(nvidia-smi -i "$1" --query-gpu=uuid --format=csv,noheader)
  nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader \
    | awk -F', ' -v u="$u" '$1 == u {print $2}' \
    | while read -r p; do [ "$(ps -o user= -p "$p")" = "$USER" ] || echo "$p"; done
}

gpus_have_room() {  # every training GPU has >= MIN_FREE_MB free
  local g free ok=0
  for g in ${GPUS//,/ }; do
    free=$(min_free "$g"); log "GPU $g: min free ${free} MiB (need $MIN_FREE_MB)"
    [ "$free" -ge "$MIN_FREE_MB" ] || ok=1
  done
  return $ok
}

# 3-4. Train, resuming from the last checkpoint after a crash.
for attempt in 1 2 3; do
  [ -d "$CKPTS/checkpoint-$MAX_STEPS" ] && { log "training already finished"; exit 0; }
  until gpus_have_room; do log "not enough room on GPUs $GPUS; re-checking in 10 min"; sleep 600; done
  log "attempt $attempt: training on GPUs $GPUS"
  [ -f "logs/train_${RUN}_seed0.log" ] && mv "logs/train_${RUN}_seed0.log" "logs/train_${RUN}_seed0.before_attempt$attempt.log"
  OMP_NUM_THREADS=8 DATALOADER_NUM_WORKERS=16 RUN=$RUN DATASET=$DATASET GPUS="$GPUS" \
    MAX_STEPS=$MAX_STEPS SAVE_STEPS=15000 SAVE_TOTAL_LIMIT=10 GLOBAL_BATCH_SIZE=64 \
    bash gr00t/train.sh
  [ -d "$CKPTS/checkpoint-$MAX_STEPS" ] && { log "training finished"; exit 0; }
  log "training exited before step $MAX_STEPS; resuming in 5 min"; sleep 300
done
log "giving up after 3 attempts"; exit 1
