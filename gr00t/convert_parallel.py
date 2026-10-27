"""Convert many RoboTwin tasks to one LeRobot v3.0 dataset, several tasks at a time.

Converting tasks one after another (convert_to_lerobot.py with every task) takes
~7 min per task, ~6 h for 50. This converts each task into its own dataset under
$HF_LEROBOT_HOME/<repo_id>_parts/<task> with up to --jobs conversions in parallel,
then merges them with LeRobot's aggregate_datasets into $HF_LEROBOT_HOME/<repo_id>.

usage: convert_parallel.py --repo_id robotwin_50task_aloha --jobs 10 <task> [<task> ...]
"""

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from lerobot.datasets.aggregate import aggregate_datasets

parser = argparse.ArgumentParser()
parser.add_argument("tasks", nargs="+")
parser.add_argument("--repo_id", required=True)
parser.add_argument("--jobs", type=int, default=10)
args = parser.parse_args()

home = Path(os.environ["HF_LEROBOT_HOME"])
parts = f"{args.repo_id}_parts"
logs = home / parts / "_logs"
logs.mkdir(parents=True, exist_ok=True)
converter = Path(__file__).with_name("convert_to_lerobot.py")


def convert(task):
    if (home / parts / task / "meta/info.json").exists():
        return task, 0  # converted by an earlier run
    with open(logs / f"{task}.log", "w") as log:
        returncode = subprocess.call(
            [sys.executable, str(converter), f"demo_clean.{task}.aloha_agilex",
             "--repo_id", f"{parts}/{task}"],
            stdout=log, stderr=subprocess.STDOUT,
        )
    print(f"[convert] {task}: exit {returncode}", flush=True)
    return task, returncode


with ThreadPoolExecutor(args.jobs) as pool:
    failed = [task for task, code in pool.map(convert, args.tasks) if code != 0]
if failed:
    sys.exit(f"conversion failed for {failed}; see {logs}")

print(f"[aggregate] {len(args.tasks)} datasets -> {home / args.repo_id}", flush=True)
aggregate_datasets(
    repo_ids=[f"{parts}/{task}" for task in args.tasks],
    aggr_repo_id=args.repo_id,
    roots=[home / parts / task for task in args.tasks],
    aggr_root=home / args.repo_id,
)
