#!/usr/bin/env bash
# RoboTwin demos (HDF5) -> LeRobot v3.0 -> LeRobot v2.1 -> GR00T-ready dataset.
#
# Steps 1-2 run in .venv-lerobot; step 3 is GR00T's own process_data.sh (GR00T venv),
# which writes meta/modality.json, configs/aloha_agilex_config.py and dataset stats.
# Doing the v3.0 -> v2.1 downgrade here keeps process_data.sh from installing lerobot
# into the GR00T venv (that would break its numpy/av/datasets pins).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Override for another task set, e.g. TASKS="$(ls RoboTwin/data/demo_clean)"
# SRC=robotwin_50task_aloha CKPT_NAME=50task gr00t/prepare_data.sh
read -r -a TASKS <<< "${TASKS:-beat_block_hammer lift_pot handover_block stack_blocks_two open_laptop}"
SRC="${SRC:-robotwin_5task_aloha}"  # converter output; becomes v2.1 in place (v3.0 kept as *_v3.0)
CKPT_NAME="${CKPT_NAME:-5task}"     # -> GR00T dataset RoboTwin-<CKPT_NAME>-aloha_agilex-joint

export HF_LEROBOT_HOME="$ROOT/data/lerobot" GR00T_LEROBOT_HOME="$ROOT/data/lerobot"
PY="$ROOT/.venv-lerobot/bin/python"
GR00T_DIR="$ROOT/RoboTwin/XPolicyLab/policy/GR00T_N17"
mkdir -p "$HF_LEROBOT_HOME"

# 1. HDF5 -> LeRobot v3.0 with XPolicyLab's official converter (all tasks, one dataset),
#    run through convert_to_lerobot.py so video frames aren't dropped and EFS read
#    errors are retried.
#    JOBS>1 converts that many tasks at a time and merges them (convert_parallel.py).
if [ ! -d "$HF_LEROBOT_HOME/$SRC" ] && [ "${JOBS:-1}" -gt 1 ]; then
  "$PY" "$ROOT/gr00t/convert_parallel.py" --repo_id "$SRC" --jobs "$JOBS" "${TASKS[@]}"
elif [ ! -d "$HF_LEROBOT_HOME/$SRC" ]; then
  patterns=()
  for t in "${TASKS[@]}"; do patterns+=("demo_clean.$t.aloha_agilex"); done
  "$PY" "$ROOT/gr00t/convert_to_lerobot.py" "${patterns[@]}" --repo_id "$SRC"
fi

# 2. LeRobot v3.0 -> v2.1, the version GR00T reads.
version=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['codebase_version'])" \
  "$HF_LEROBOT_HOME/$SRC/meta/info.json")
if [ "$version" = "v3.0" ]; then
  "$PY" "$GR00T_DIR/gr00t_n17/scripts/lerobot_conversion/convert_v3_to_v2.py" \
    --root "$HF_LEROBOT_HOME" --repo-id "$SRC"
fi

# 3. GR00T prep: copy, modality.json + modality config, normalization stats.
GR00T_SRC_DATASET="$SRC" bash "$GR00T_DIR/process_data.sh" RoboTwin "$CKPT_NAME" aloha_agilex joint

# 4. Every episode present, and one video frame per step for every camera.
"$PY" "$ROOT/gr00t/check_dataset.py" \
  "$GR00T_LEROBOT_HOME/RoboTwin-$CKPT_NAME-aloha_agilex-joint" $((${#TASKS[@]} * 50))
