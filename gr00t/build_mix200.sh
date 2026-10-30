#!/usr/bin/env bash
# Build the clean + randomized GR00T dataset (12,500 episodes) without the slow
# merge-then-split round trip of prepare_data.sh. Meant to run detached:
#   setsid -f nohup bash gr00t/build_mix200.sh > logs/build_mix200.log 2>&1 < /dev/null
# 1. Wait for (or resume) the fetch of 200 randomized demos per task.
# 2. Convert each (demo group, task) to its own small LeRobot v3.0 part, 24 at a time.
# 3. Downgrade every part to v2.1 on its own, 24 at a time; the 50 clean parts are copied
#    from robotwin_50task_aloha_parts first (originals untouched).
# 4. Merge the v2.1 parts by renumbering + copying per-episode files (gr00t/merge_v2.py),
#    straight into the final dataset dir, so GR00T's process_data.sh skips its own copy.
#    Waits for scratch/merge_v2_validated (written once merge_v2.py is verified).
# 5. GR00T stats/modality (process_data.sh), then check_dataset.py (writes
#    meta/video_frames.json only if every video matches and there are 12,500 episodes).
# Training is started by hand afterwards (GPU 3 is freed by hand right before).
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export HF_LEROBOT_HOME="$ROOT/data/lerobot" GR00T_LEROBOT_HOME="$ROOT/data/lerobot"
export HF_TOKEN="$(cat ~/.cache/huggingface/token)"
PY="$ROOT/.venv-lerobot/bin/python"
GR00T_DIR="$ROOT/RoboTwin/XPolicyLab/policy/GR00T_N17"
V3TOV2="$GR00T_DIR/gr00t_n17/scripts/lerobot_conversion/convert_v3_to_v2.py"
PARTS="$HF_LEROBOT_HOME/robotwin_mix200_aloha_parts"
OUT=RoboTwin-mix200-aloha_agilex-joint
CONFIGS="demo_randomized0 demo_randomized1 demo_randomized2 demo_randomized3"
TASKS="$(ls RoboTwin/data/demo_clean | tr '\n' ' ')"
log() { echo "[build_mix200] $(TZ=America/New_York date '+%F %H:%M ET') $*"; }

# 1. Fetch (resumes; skips episodes already on disk).
while pgrep -u "$USER" -f 'gr00t/fetch_randomized[.]py' >/dev/null; do sleep 30; done
n=$(find RoboTwin/data/demo_randomized[0-3] -name 'episode_*.hdf5' | wc -l)
if [ "$n" -lt 10000 ]; then
  log "fetch has $n/10000 episodes; resuming it"
  RoboTwin/.venv/bin/python gr00t/fetch_randomized.py --n 200 --group 50 \
    --config demo_randomized --jobs 12 $TASKS || { log "fetch failed"; exit 1; }
fi
log "fetch complete: $(find RoboTwin/data/demo_randomized[0-3] -name 'episode_*.hdf5' | wc -l) episodes"

# 2. Parts (v3.0).
log "converting 200 randomized parts"
"$PY" gr00t/convert_parallel.py --repo_id robotwin_mix200_aloha --jobs 24 \
  --config $CONFIGS --no_aggregate -- $TASKS || { log "part conversion failed"; exit 1; }
for t in $TASKS; do
  [ -d "$PARTS/clean__$t" ] || [ -d "$PARTS/clean__${t}_v3.0" ] || cp -a "$HF_LEROBOT_HOME/robotwin_50task_aloha_parts/$t" "$PARTS/clean__$t"
done

# 3. Downgrade each part to v2.1 (skip parts already downgraded).
log "downgrading 250 parts to v2.1"
ls "$PARTS" | grep -vE '^_logs$|_v3\.0$' | while read -r part; do
  v=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['codebase_version'])" "$PARTS/$part/meta/info.json")
  [ "$v" = v3.0 ] && echo "$part"
done | xargs -r -P 24 -I{} sh -c '"$0" "$1" --root "$2" --repo-id "$3" > "$2/_logs/v2__$3.log" 2>&1 || echo "downgrade failed: $3"' "$PY" "$V3TOV2" "$PARTS" {}
bad=$(ls "$PARTS" | grep -vE '^_logs$|_v3\.0$' | while read -r part; do "$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['codebase_version'])" "$PARTS/$part/meta/info.json" | grep -qx v2.1 || echo "$part"; done)
[ -z "$bad" ] || { log "parts not downgraded: $bad"; exit 1; }

# 4. Merge (waits until merge_v2.py has been verified).
until [ -f scratch/merge_v2_validated ]; do sleep 60; done
if [ ! -f "$HF_LEROBOT_HOME/$OUT/meta/info.json" ]; then
  log "merging 250 parts into $OUT"
  order=""; for t in $TASKS; do order="$order $PARTS/clean__$t"; done
  for c in $CONFIGS; do for t in $TASKS; do order="$order $PARTS/${c}__$t"; done; done
  "$PY" gr00t/merge_v2.py --out "$HF_LEROBOT_HOME/$OUT" --jobs 24 $order || { log "merge failed"; exit 1; }
fi

# 5. GR00T modality + stats (no copy: source = output), then the full check.
log "GR00T stats"
GR00T_SRC_DATASET=$OUT bash "$GR00T_DIR/process_data.sh" RoboTwin mix200 aloha_agilex joint \
  || { log "process_data failed"; exit 1; }
log "checking every video"
"$PY" gr00t/check_dataset.py "$HF_LEROBOT_HOME/$OUT" 12500 || { log "check failed"; exit 1; }
log "dataset ready"
