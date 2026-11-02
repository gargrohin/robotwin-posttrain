#!/usr/bin/env bash
# Cosmos3-Nano action-policy SFT on RoboTwin mix200 (NVIDIA's LIBERO recipe, run card
# notes/runs/cosmos3_nano_mix200.md), meant to run detached:
#   GPUS=0,4,5,6,7 setsid -f nohup bash cosmos3/run_robotwin_nano.sh > logs/cosmos3_nano_mix200.log 2>&1 < /dev/null
# Runs NVIDIA cosmos-framework's launcher (our `robotwin` branch, /mnt/efs/rohingarg/cri/cosmos3/
# cosmos-framework). If training exits before MAX_ITER, it waits 5 min, waits until every GPU
# in GPUS is nearly empty (a 16B full fine-tune needs ~all of an 80 GB GPU), and relaunches;
# the trainer resumes from the run folder's latest_checkpoint.txt (up to 3 attempts).
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CF=/mnt/efs/rohingarg/cri/cosmos3/cosmos-framework
: "${GPUS:?set GPUS, e.g. GPUS=0,4,5,6,7}"
MAX_ITER=5000
OUTPUT_ROOT=/mnt/efs/rohingarg/cri/cosmos3/runs
RUN_DIR="$OUTPUT_ROOT/cosmos3_action_robotwin/action_sft/action_policy_robotwin_nano"
MIN_FREE_MB=78000
log() { echo "[cosmos3_nano] $(TZ=America/New_York date '+%F %H:%M %Z') $*"; }
latest() { cat "$RUN_DIR/checkpoints/latest_checkpoint.txt" 2>/dev/null; }
gpus_empty() {  # every GPU has >= MIN_FREE_MB free (lowest of 6 samples over ~1 min)
  local g i f min ok=0
  for g in ${GPUS//,/ }; do
    min=999999; for i in 1 2 3 4 5 6; do
      f=$(nvidia-smi -i "$g" --query-gpu=memory.free --format=csv,noheader,nounits); [ "$f" -lt "$min" ] && min=$f; sleep 10
    done
    log "GPU $g: min free $min MiB (need $MIN_FREE_MB)"; [ "$min" -ge "$MIN_FREE_MB" ] || ok=1
  done
  return $ok
}
for attempt in 1 2 3; do
  [ "$(latest)" = "iter_$(printf %09d $MAX_ITER)" ] && { log "training already finished"; exit 0; }
  [ "$attempt" = 1 ] || until gpus_empty; do log "GPUs $GPUS not free; re-checking in 10 min"; sleep 600; done
  log "attempt $attempt on GPUs $GPUS (latest checkpoint: $(latest || echo none))"
  (cd "$CF" && source .venv/bin/activate && export LD_LIBRARY_PATH= HF_TOKEN="$(cat ~/.cache/huggingface/token)" &&
    CUDA_VISIBLE_DEVICES="$GPUS" \
    ROBOTWIN_ROOT="$ROOT/data/lerobot/RoboTwin-mix200-aloha_agilex-joint" \
    ROBOTWIN_INDEX_CACHE=/mnt/efs/rohingarg/cri/cosmos3/robotwin_cache/RoboTwin-mix200-aloha_agilex-joint.index.npz \
    OUTPUT_ROOT="$OUTPUT_ROOT" LOG_FILENAME="action_policy_robotwin_nano_attempt$attempt.log" \
    bash examples/launch_sft_action_policy_robotwin_nano.sh)
  [ "$(latest)" = "iter_$(printf %09d $MAX_ITER)" ] && { log "training finished"; exit 0; }
  log "training exited before iteration $MAX_ITER (latest: $(latest || echo none)); retrying in 5 min"; sleep 300
done
log "giving up after 3 attempts"; exit 1
