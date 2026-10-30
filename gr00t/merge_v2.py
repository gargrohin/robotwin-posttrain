"""Merge LeRobot v2.1 datasets into one v2.1 dataset, episode files in parallel.

v2.1 stores one parquet and one mp4 per camera per episode, so merging is renumbering
plus copying. This replaces "aggregate the v3.0 parts, then convert_v3_to_v2.py the
merged dataset", which is single-threaded and cuts the merged videos back into
per-episode files.

Episodes are renumbered 0..N-1 in input order and go to chunk i // chunks_size. Tasks
are de-duplicated in order of first appearance (like LeRobot's aggregate_datasets), and
each parquet gets the new episode_index and task_index and a contiguous global index;
every other column is kept as-is. Videos are copied unchanged. meta/stats.json is
aggregated from the inputs' stats.json like aggregate_datasets does, except for the
renumbered columns, whose per-episode and global stats are recomputed.

usage: merge_v2.py --out <dir> [--jobs 24] <v2.1 dataset dir> [<dir> ...]
"""

import argparse
import json
import math
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import jsonlines
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from lerobot.datasets.compute_stats import aggregate_stats, get_feature_stats
from lerobot.datasets.utils import cast_stats_to_numpy, load_stats, serialize_dict, write_info, write_stats

RENUMBERED = ("episode_index", "index", "task_index")
# info.json fields that must be identical across inputs (the counts are recomputed).
SHARED_INFO = ("codebase_version", "robot_type", "chunks_size", "fps", "data_path", "video_path", "features")


def read_jsonl(path):
    with jsonlines.open(path) as reader:
        return list(reader)


def load_input(root):
    """Read one input's metadata and check that it is self-consistent."""
    info = json.loads((root / "meta/info.json").read_text())
    if info["codebase_version"] != "v2.1":
        sys.exit(f"{root}: codebase_version {info['codebase_version']}, expected v2.1")
    tasks = sorted(read_jsonl(root / "meta/tasks.jsonl"), key=lambda t: t["task_index"])
    episodes = read_jsonl(root / "meta/episodes.jsonl")
    ep_stats = read_jsonl(root / "meta/episodes_stats.jsonl")
    n = info["total_episodes"]
    problems = []
    if [t["task_index"] for t in tasks] != list(range(len(tasks))):
        problems.append("task indices are not 0..k-1")
    if [e["episode_index"] for e in episodes] != list(range(n)):
        problems.append(f"episodes.jsonl episode indices are not 0..{n - 1}")
    if [s["episode_index"] for s in ep_stats] != list(range(n)):
        problems.append(f"episodes_stats.jsonl episode indices are not 0..{n - 1}")
    if sum(e["length"] for e in episodes) != info["total_frames"]:
        problems.append("episode lengths don't add up to total_frames")
    if info["splits"] != {"train": f"0:{n}"}:
        problems.append(f"unexpected splits {info['splits']}")
    if not (root / "meta/stats.json").exists():
        problems.append("no meta/stats.json")
    if problems:
        sys.exit(f"{root}: " + "; ".join(problems))
    return {"root": root, "info": info, "tasks": tasks, "episodes": episodes, "ep_stats": ep_stats}


def merge_episode(job):
    """Write one episode's renumbered parquet and copy its videos; return the new
    per-episode stats of the renumbered columns. Retries EFS read errors (errno 5)."""
    for attempt in range(3):
        try:
            return _merge_episode(**job)
        except OSError:
            if attempt == 2:
                raise
            time.sleep(1)


def _merge_episode(src_root, src_ep, dst_root, ep, first_index, length, task_map, task_names,
                   ep_tasks, chunks_size, data_path, video_path, video_keys):
    src_chunk, chunk = src_ep // chunks_size, ep // chunks_size
    src = src_root / data_path.format(episode_chunk=src_chunk, episode_index=src_ep)
    table = pq.read_table(src)
    old_task = table["task_index"].to_numpy()
    if table.num_rows != length or (table["episode_index"].to_numpy() != src_ep).any():
        raise ValueError(f"{src}: expected {length} rows of episode {src_ep}")
    if {task_names[i] for i in np.unique(old_task)} != set(ep_tasks):
        raise ValueError(f"{src}: task_index values don't match episodes.jsonl tasks {ep_tasks}")

    new = {
        "episode_index": np.full(length, ep, dtype=np.int64),
        "index": np.arange(first_index, first_index + length, dtype=np.int64),
        "task_index": task_map[old_task],
    }
    for name, values in new.items():
        i = table.schema.get_field_index(name)
        field = table.schema.field(i)
        table = table.set_column(i, field, pa.array(values, type=field.type))
    dst = dst_root / data_path.format(episode_chunk=chunk, episode_index=ep)
    dst.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, dst)

    for key in video_keys:
        src = src_root / video_path.format(episode_chunk=src_chunk, video_key=key, episode_index=src_ep)
        dst = dst_root / video_path.format(episode_chunk=chunk, video_key=key, episode_index=ep)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)

    # Same computation LeRobot uses for 1-D columns when it records an episode.
    return {name: serialize_dict(get_feature_stats(values, axis=0, keepdims=True))
            for name, values in new.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", type=Path, help="v2.1 dataset dirs, merged in this order")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--jobs", type=int, default=24)
    args = parser.parse_args()
    start = time.time()

    if args.out.exists():
        sys.exit(f"{args.out} already exists")
    inputs = [load_input(root) for root in args.inputs]
    info = inputs[0]["info"]
    for inp in inputs[1:]:
        for key in SHARED_INFO:
            if inp["info"][key] != info[key]:
                sys.exit(f"{inp['root']}: info.json '{key}' differs from {inputs[0]['root']}")

    # Tasks in order of first appearance, as aggregate_datasets numbers them.
    task_index = {}
    for inp in inputs:
        for task in inp["tasks"]:
            task_index.setdefault(task["task"], len(task_index))

    video_keys = [key for key, ft in info["features"].items() if ft["dtype"] == "video"]
    jobs, episodes, ep_stats = [], [], []
    total_frames = 0
    for inp in inputs:
        task_names = [task["task"] for task in inp["tasks"]]
        task_map = np.array([task_index[name] for name in task_names], dtype=np.int64)
        for episode, stats in zip(inp["episodes"], inp["ep_stats"]):
            ep = len(episodes)
            jobs.append(dict(
                src_root=inp["root"], src_ep=episode["episode_index"], dst_root=args.out, ep=ep,
                first_index=total_frames, length=episode["length"], task_map=task_map,
                task_names=task_names, ep_tasks=episode["tasks"], chunks_size=info["chunks_size"],
                data_path=info["data_path"], video_path=info["video_path"], video_keys=video_keys,
            ))
            episodes.append({**episode, "episode_index": ep})
            ep_stats.append({"episode_index": ep, "stats": stats["stats"]})
            total_frames += episode["length"]

    (args.out / "meta").mkdir(parents=True)
    with ProcessPoolExecutor(args.jobs) as pool:
        for stats, renumbered in zip(ep_stats, pool.map(merge_episode, jobs, chunksize=4)):
            stats["stats"].update(renumbered)  # keeps the key order

    with jsonlines.open(args.out / "meta/tasks.jsonl", mode="w") as writer:
        for task, i in task_index.items():
            writer.write({"task_index": i, "task": task})
    with jsonlines.open(args.out / "meta/episodes.jsonl", mode="w") as writer:
        writer.write_all(episodes)
    with jsonlines.open(args.out / "meta/episodes_stats.jsonl", mode="w") as writer:
        writer.write_all(ep_stats)

    stats = aggregate_stats([load_stats(inp["root"]) for inp in inputs])
    stats.update(aggregate_stats(
        [cast_stats_to_numpy({name: s["stats"][name] for name in RENUMBERED}) for s in ep_stats]))
    write_stats(stats, args.out)

    n = len(episodes)
    info = {**info, "total_episodes": n, "total_frames": total_frames, "total_tasks": len(task_index),
            "splits": {"train": f"0:{n}"}, "total_chunks": math.ceil(n / info["chunks_size"]),
            "total_videos": n * len(video_keys)}
    write_info(info, args.out)
    print(f"merged {len(inputs)} datasets -> {args.out}: {n} episodes, {total_frames} frames, "
          f"{len(task_index)} tasks in {time.time() - start:.0f} s", flush=True)


if __name__ == "__main__":
    main()
