"""Sum per-shard eval results into a per-task success table.

usage: summarize_eval.py <ckpt> <checkpoint_num> <task_config>
Reads logs/eval_<ckpt>_ckpt-<n>_<task>_<config>[_seed<i>].client.log.
"""

import re
import sys
from collections import defaultdict
from pathlib import Path

ckpt, num, config = sys.argv[1:4]
logs = Path(__file__).resolve().parents[1] / "logs"
pattern = re.compile(
    rf"eval_{re.escape(ckpt)}_ckpt-{re.escape(num)}_(.+)_{re.escape(config)}"
    r"(?:_seed(\d+))?\.client\.log$"
)
final = re.compile(r"Final success rate: (\d+)/(\d+)")

totals = defaultdict(lambda: [0, 0, 0])  # successes, episodes, shards
for path in sorted(logs.glob(f"eval_{ckpt}_ckpt-{num}_*.client.log")):
    m = pattern.match(path.name)
    hits = final.findall(re.sub(r"\x1b\[[0-9;]*m", "", path.read_text(errors="ignore")))
    if not m or not hits:
        continue
    successes, episodes = map(int, hits[-1])
    t = totals[m.group(1)]
    t[0] += successes
    t[1] += episodes
    t[2] += 1

print(f"{ckpt} checkpoint-{num}, {config}")
print("| task | success | episodes | shards |")
print("|---|---|---|---|")
for task, (s, n, k) in sorted(totals.items()):
    print(f"| {task} | {100 * s / n:.0f}% ({s}/{n}) | {n} | {k} |")
if totals:
    rates = [s / n for s, n, _ in totals.values()]
    print(f"| **mean** | **{100 * sum(rates) / len(rates):.1f}%** | | |")
