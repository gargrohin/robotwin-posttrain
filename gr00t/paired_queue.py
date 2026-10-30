"""Write eval queue lines that replay another eval's seed layout, so two models (or two
checkpoints) are scored on the same scenes.

Shard k of an eval runs consecutive valid seeds from 100000*(k+1) (gr00t/benchmark.sh),
so the same scenes need the same seed index and episode count per shard. This reads the
reference eval's client logs (logs/ and logs/preempted/) and prints one queue line per
shard, "<task> <config> <episodes> <seed index>", for gr00t/benchmark_worker.sh with
MAX_SHARDS=1 (each line runs as exactly one shard). Lines are sorted longest first
(episodes x the task's step limit), so slow single-stream jobs start early.

usage: paired_queue.py <ref ckpt> <ref checkpoint num> <config> [--max-eps N] <task> ...
  --max-eps N: replay only each task's first N episodes, in seed-index order.
"""

import argparse
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"
LIMITS = yaml.safe_load((ROOT / "RoboTwin/env_cfg/task_config/_eval_step_limit.yml").read_text())

p = argparse.ArgumentParser()
p.add_argument("ckpt")
p.add_argument("num")
p.add_argument("config")
p.add_argument("--max-eps", type=int)
p.add_argument("tasks", nargs="+")
a = p.parse_args()

lines = []
for task in a.tasks:
    name = re.compile(rf"eval_{re.escape(a.ckpt)}_ckpt-{a.num}_{task}_{a.config}(?:_seed(\d+))?\.client\.log")
    shards = {}
    for d in (LOGS / "preempted", LOGS):  # logs/ wins if both have the same shard
        for f in d.glob(f"eval_{a.ckpt}_ckpt-{a.num}_{task}_{a.config}*.client.log"):
            m = name.fullmatch(f.name)
            text = re.sub(r"\x1b\[[0-9;]*m", "", f.read_text(errors="replace"))
            n = len(re.findall(r"current seed: \d+", text))  # one line per finished episode
            if m and n:
                shards[int(m[1] or 0)] = n
    if not shards:
        raise SystemExit(f"no reference episodes for {task} {a.config}")
    left = a.max_eps
    for k in sorted(shards):
        n = shards[k] if left is None else min(shards[k], left)
        if n <= 0:
            break
        lines.append((n * LIMITS[task], f"{task} {a.config} {n} {k}"))
        if left is not None:
            left -= n

for _, line in sorted(lines, reverse=True):
    print(line)
