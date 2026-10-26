#!/usr/bin/env bash
# Show benchmark progress: workers, queue, episodes per running task, results so far.
# usage: CKPT=RoboTwin-5task_30k-aloha_agilex-joint-0 CHECKPOINT_NUM=30000 \
#        QUEUE=logs/queue_30k.txt gr00t/benchmark_status.sh
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CKPT="${CKPT:-RoboTwin-5task_30k-aloha_agilex-joint-0}"
NUM="${CHECKPOINT_NUM:-30000}"
QUEUE="${QUEUE:-$ROOT/logs/queue_30k.txt}"
cd "$ROOT"

echo "== $(date -u '+%F %H:%M UTC')"
echo "== workers"
for log in "${QUEUE%.txt}"_gpu*.log; do
  [ -f "$log" ] && echo "$(basename "$log"): $(grep -E '^\[queue' "$log" | tail -1)"
done
echo "running shards: $(pgrep -u "$USER" -fc 'eval_policy_xpolicylab[.]py')"
echo "== queue ($(grep -c . "$QUEUE" 2>/dev/null || echo 0) jobs left)"
cat "$QUEUE" 2>/dev/null
echo "== episodes recorded per running task (videos; includes episodes in progress)"
for d in RoboTwin/eval_result/*/GR00T_N17/*/"$CKPT"/; do
  [ -d "$d" ] || continue
  running=$(find "$d" -mindepth 1 -maxdepth 1 -type d ! -exec test -e '{}/_result.txt' \; -print)
  [ -n "$running" ] || continue
  n=$(for r in $running; do ls "$r" | grep -c mp4; done | paste -sd+ | bc)
  echo "$(echo "$d" | cut -d/ -f3,5): $n"
done
echo "== GPUs"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
for config in demo_clean demo_randomized; do
  echo "== results: $config"
  python3 gr00t/summarize_eval.py "$CKPT" "$NUM" "$config" | tail -n +2
done
