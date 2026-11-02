#!/usr/bin/env bash
# Build the clean + 495-randomized GR00T dataset, data/lerobot/RoboTwin-mix500-aloha_agilex-joint:
# 50 tasks × (50 clean + randomized episodes 0-494) = 27,250 episodes. Randomized episodes
# 495-499 stay out (RoboTwin/data/demo_randomized_heldout500) for the memorization check.
# Meant to run detached:
#   setsid -f nohup bash gr00t/build_mix500.sh > logs/build_mix500.log 2>&1 < /dev/null
# Reuses mix200's converted v2.1 pieces (clean, randomized 0-199) and only converts the
# new groups demo_randomized4..9 (episodes 200-494; group 9 has 45). Same steps as
# gr00t/build_mix200.sh: convert per (group, task) with short-piece re-conversion,
# downgrade each piece to v2.1, merge (gr00t/merge_v2.py), GR00T stats, check every video,
# then compare every episode with its source file (gr00t/verify_dataset.py).
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export HF_LEROBOT_HOME="$ROOT/data/lerobot" GR00T_LEROBOT_HOME="$ROOT/data/lerobot"
export HF_TOKEN="$(cat ~/.cache/huggingface/token)"
PY="$ROOT/.venv-lerobot/bin/python"
GR00T_DIR="$ROOT/RoboTwin/XPolicyLab/policy/GR00T_N17"
V3TOV2="$GR00T_DIR/gr00t_n17/scripts/lerobot_conversion/convert_v3_to_v2.py"
OLD_PARTS="$HF_LEROBOT_HOME/robotwin_mix200_aloha_parts"   # clean__*, demo_randomized0..3__* (v2.1)
PARTS="$HF_LEROBOT_HOME/robotwin_mix500_aloha_parts"       # demo_randomized4..9__*
OUT=RoboTwin-mix500-aloha_agilex-joint
OLD_CONFIGS="demo_randomized0 demo_randomized1 demo_randomized2 demo_randomized3"
CONFIGS="demo_randomized4 demo_randomized5 demo_randomized6 demo_randomized7 demo_randomized8 demo_randomized9"
TASKS="$(ls RoboTwin/data/demo_clean | tr '\n' ' ')"
log() { echo "[build_mix500] $(TZ=America/New_York date '+%F %H:%M %Z') $*"; }
expected() { [ "$1" = demo_randomized9 ] && echo 45 || echo 50; }  # episodes per piece

# 1. Fetch (resumes; skips episodes already on disk).
while pgrep -u "$USER" -f 'gr00t/fetch_randomized[.]py' >/dev/null; do sleep 30; done
n=$(find RoboTwin/data/demo_randomized[4-9] -name 'episode_*.hdf5' | wc -l)
if [ "$n" -lt 14750 ]; then
  log "fetch has $n/14750 episodes; resuming it"
  RoboTwin/.venv/bin/python gr00t/fetch_randomized.py --n 295 --start 200 --group 50 \
    --config demo_randomized --jobs 12 $TASKS || { log "fetch failed"; exit 1; }
fi
log "fetch complete: $(find RoboTwin/data/demo_randomized[4-9] -name 'episode_*.hdf5' | wc -l) episodes"

# 2. New pieces (v3.0), re-converting pieces the converter left short (see build_mix200.sh).
episodes() { grep -o '"total_episodes": *[0-9]*' "$PARTS/$1/meta/info.json" 2>/dev/null | grep -o '[0-9]*$'; }
for round in 1 2 3 4; do
  log "converting new randomized pieces (round $round)"
  "$PY" gr00t/convert_parallel.py --repo_id robotwin_mix500_aloha --jobs 24 \
    --config $CONFIGS --no_aggregate -- $TASKS || { log "piece conversion failed"; exit 1; }
  short=$(for c in $CONFIGS; do for t in $TASKS; do [ "$(episodes "${c}__$t")" = "$(expected $c)" ] || echo "${c}__$t"; done; done)
  [ -z "$short" ] && break
  log "short pieces, re-converting:" $short
  for p in $short; do rm -rf "${PARTS:?}/$p" "$PARTS/${p}_v3.0"; done
done
[ -z "$short" ] || { log "pieces still short after 4 rounds: $short"; exit 1; }

# 3. Downgrade each new piece to v2.1 (skip pieces already downgraded).
log "downgrading new pieces to v2.1"
ls "$PARTS" | grep -vE '^_logs$|_v3\.0$' | while read -r part; do
  v=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['codebase_version'])" "$PARTS/$part/meta/info.json")
  [ "$v" = v3.0 ] && echo "$part"
done | xargs -r -P 24 -I{} sh -c '"$0" "$1" --root "$2" --repo-id "$3" > "$2/_logs/v2__$3.log" 2>&1 || echo "downgrade failed: $3"' "$PY" "$V3TOV2" "$PARTS" {}
bad=$(ls "$PARTS" | grep -vE '^_logs$|_v3\.0$' | while read -r part; do "$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['codebase_version'])" "$PARTS/$part/meta/info.json" | grep -qx v2.1 || echo "$part"; done)
[ -z "$bad" ] || { log "pieces not downgraded: $bad"; exit 1; }

# 4. Merge 550 pieces: clean, then randomized groups 0..9, tasks sorted within each.
if [ ! -f "$HF_LEROBOT_HOME/$OUT/meta/info.json" ]; then
  log "merging 550 pieces into $OUT"
  order=""; for t in $TASKS; do order="$order $OLD_PARTS/clean__$t"; done
  for c in $OLD_CONFIGS; do for t in $TASKS; do order="$order $OLD_PARTS/${c}__$t"; done; done
  for c in $CONFIGS; do for t in $TASKS; do order="$order $PARTS/${c}__$t"; done; done
  "$PY" gr00t/merge_v2.py --out "$HF_LEROBOT_HOME/$OUT" --jobs 24 $order || { log "merge failed"; exit 1; }
fi

# 5. GR00T modality + stats (no copy: source = output), check every video, then compare
# every episode with its source file.
log "GR00T stats"
GR00T_SRC_DATASET=$OUT bash "$GR00T_DIR/process_data.sh" RoboTwin mix500 aloha_agilex joint \
  || { log "process_data failed"; exit 1; }
log "checking every video"
"$PY" gr00t/check_dataset.py "$HF_LEROBOT_HOME/$OUT" 27250 || { log "check failed"; exit 1; }
log "comparing every episode with its source file"
src=""; for t in $TASKS; do src="$src RoboTwin/data/demo_clean/$t/aloha_agilex/data"; done
for c in $OLD_CONFIGS $CONFIGS; do for t in $TASKS; do src="$src RoboTwin/data/$c/$t/aloha_agilex/data"; done; done
"$PY" gr00t/verify_dataset.py "$HF_LEROBOT_HOME/$OUT" $src > logs/verify_mix500.log 2>&1 \
  || { log "verify failed (logs/verify_mix500.log)"; exit 1; }
tail -12 logs/verify_mix500.log
log "dataset ready"
