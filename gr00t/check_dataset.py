"""Check a LeRobot v2.1 dataset: every camera video has exactly one frame per step.

usage: check_dataset.py <dataset_dir> [expected_episodes]
"""

import json
import sys
from multiprocessing import Pool
from pathlib import Path

import av

root = Path(sys.argv[1])
expected = int(sys.argv[2]) if len(sys.argv) > 2 else None
info = json.loads((root / "meta/info.json").read_text())
episodes = [json.loads(line) for line in (root / "meta/episodes.jsonl").open()]
video_keys = [k for k, f in info["features"].items() if f["dtype"] == "video"]


def check(job):
    ep, key = job
    i = ep["episode_index"]
    path = root / info["video_path"].format(
        episode_chunk=i // info["chunks_size"], video_key=key, episode_index=i
    )
    with av.open(str(path)) as container:
        n = sum(1 for _ in container.decode(video=0))
    if n != ep["length"]:
        return f"episode {i} {key}: {n} frames, {ep['length']} steps"


if __name__ == "__main__":
    jobs = [(ep, key) for ep in episodes for key in video_keys]
    with Pool(32) as pool:
        bad = [r for r in pool.imap(check, jobs, chunksize=8) if r]

    print(f"{len(episodes)} episodes, {len(video_keys)} cameras, {len(bad)} mismatches")
    for line in bad[:20]:
        print("  " + line)
    if bad or (expected is not None and len(episodes) != expected):
        sys.exit(f"FAILED (expected {expected} episodes)" if expected else "FAILED")
