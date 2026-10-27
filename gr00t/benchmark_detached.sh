#!/usr/bin/env bash
# Start one benchmark_queue.sh worker per GPU, detached (setsid + nohup), so the jobs
# survive closing the terminal or Claude session that launched them. Workers start
# 75 s apart to spread out checkpoint loading from EFS.
#
# usage: QUEUE=logs/queue_30k.txt GPUS="1 5 6" CKPT=RoboTwin-5task_30k-aloha_agilex-joint-0 \
#        CHECKPOINT_NUM=30000 gr00t/benchmark_detached.sh
# Stop: pkill -u "$USER" -f 'benchmark_queue[.]sh', then the same for 'gr00t/eval[.]sh',
#       'setup_policy_server[.]py' and 'eval_policy_xpolicylab[.]py'.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
: "${QUEUE:?set QUEUE}" "${GPUS:?set GPUS}" "${CKPT:?set CKPT}"
queue="$(realpath "$QUEUE")"

delay=0
for gpu in $GPUS; do
  log="$ROOT/logs/$(basename "$queue" .txt)_gpu$gpu.log"
  QUEUE="$queue" GPU="$gpu" CKPT="$CKPT" CHECKPOINT_NUM="${CHECKPOINT_NUM:-last}" \
    setsid -f nohup bash -c "sleep $delay; exec bash '$ROOT/gr00t/benchmark_queue.sh'" \
    > "$log" 2>&1 < /dev/null
  echo "GPU $gpu worker starts in ${delay}s, log: $log"
  delay=$((delay + ${STAGGER:-75}))
done
