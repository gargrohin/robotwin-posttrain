"""Convert many RoboTwin tasks to one LeRobot v3.0 dataset, several tasks at a time.

Converting tasks one after another (convert_to_lerobot.py with every task) takes
~7 min per task, ~6 h for 50. This converts each task into its own dataset under
$HF_LEROBOT_HOME/<repo_id>_parts/<task> with up to --jobs conversions in parallel,
then merges them with LeRobot's aggregate_datasets into $HF_LEROBOT_HOME/<repo_id>.
--config picks the demo set(s) (RoboTwin/data/<config>; with several, parts are named
<config>__<task>); --extra_parts adds already
converted per-task datasets (e.g. robotwin_50task_aloha_parts) to the merge.

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
parser.add_argument("--config", nargs="+", default=["demo_clean"])
parser.add_argument("--extra_parts", nargs="*", default=[])
args = parser.parse_args()

home = Path(os.environ["HF_LEROBOT_HOME"])
parts = f"{args.repo_id}_parts"
logs = home / parts / "_logs"
logs.mkdir(parents=True, exist_ok=True)
converter = Path(__file__).with_name("convert_to_lerobot.py")


jobs = [(c, t) for c in args.config for t in args.tasks]
part = (lambda c, t: f"{parts}/{t}") if len(args.config) == 1 else (lambda c, t: f"{parts}/{c}__{t}")


def convert(job):
    config, task = job
    if (home / part(config, task) / "meta/info.json").exists():
        return job, 0  # converted by an earlier run
    with open(logs / f"{config}__{task}.log", "w") as log:
        returncode = subprocess.call(
            [sys.executable, str(converter), f"{config}.{task}.aloha_agilex",
             "--repo_id", part(config, task)],
            stdout=log, stderr=subprocess.STDOUT,
        )
    print(f"[convert] {config} {task}: exit {returncode}", flush=True)
    return job, returncode


with ThreadPoolExecutor(args.jobs) as pool:
    failed = [job for job, code in pool.map(convert, jobs) if code != 0]
if failed:
    sys.exit(f"conversion failed for {failed}; see {logs}")

sources = [part(c, t) for c, t in jobs]
sources += [f"{extra}/{task}" for extra in args.extra_parts for task in args.tasks]
print(f"[aggregate] {len(sources)} datasets -> {home / args.repo_id}", flush=True)
aggregate_datasets(
    repo_ids=sources,
    aggr_repo_id=args.repo_id,
    roots=[home / source for source in sources],
    aggr_root=home / args.repo_id,
)
