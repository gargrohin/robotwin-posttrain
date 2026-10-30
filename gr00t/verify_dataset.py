"""Check a merged GR00T training dataset against the source HDF5 demos it was built from.

- Every episode: same length as its source file; state and action exactly equal to the
  source's joint states (left arm, left gripper, right arm, right gripper); the
  instruction GR00T reads (task_index -> tasks.jsonl) is one of the source's sentences.
- First episode of each source dir: the middle frame of each camera is close to the
  source image (the videos are lossy) and closer to it than to the other cameras' images.
- GR00T's normalization stats (stats.json, relative_stats.json) are all finite.

usage: verify_dataset.py <dataset_dir> <source dir> [<source dir> ...]
  source dirs in merge order, each holding episode_*.hdf5 files
  (e.g. RoboTwin/data/demo_clean/<task>/aloha_agilex/data)
"""

import json
import math
import re
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import av
import cv2
import h5py
import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "RoboTwin"))
from XPolicyLab.utils.process_data import decode_image_bit  # noqa: E402

JOINTS = ["left_arm_joint_states", "left_ee_joint_states", "right_arm_joint_states", "right_ee_joint_states"]
CAMERAS = {"cam_high": "cam_head", "cam_left_wrist": "cam_left_wrist", "cam_right_wrist": "cam_right_wrist"}
MAX_IMAGE_MAE = 8.0  # mean abs pixel difference (0-255) allowed for video compression

root = Path(sys.argv[1])
info = json.loads((root / "meta/info.json").read_text())
episodes = [json.loads(line) for line in (root / "meta/episodes.jsonl").open()]
tasks = {t["task_index"]: t["task"] for t in map(json.loads, (root / "meta/tasks.jsonl").open())}
source_dirs = [Path(d) for d in sys.argv[2:]]
jobs = []
for d in source_dirs:
    files = sorted(d.glob("episode_*.hdf5"))
    jobs += [(len(jobs) + n, str(f), n == 0) for n, f in enumerate(files)]


def path(key, i, video_key=None):
    return root / info[key].format(episode_chunk=i // info["chunks_size"], episode_index=i, video_key=video_key)


def video_frame(file, k):
    with av.open(str(file)) as container:
        for n, frame in enumerate(container.decode(video=0)):
            if n == k:
                return frame.to_ndarray(format="rgb24")


def check(job):
    i, src, sample = job
    ep, errors, maes = episodes[i], [], []
    k = ep["length"] // 2
    with h5py.File(src, "r") as f:
        state = np.concatenate([f["state/" + j][()] for j in JOINTS], axis=1).astype(np.float32)
        action = np.concatenate([f["action/" + j][()] for j in JOINTS], axis=1).astype(np.float32)
        sentences = json.loads(f["instructions"][()])
        sample = sample and len(state) == ep["length"]  # else the length error below says it
        images = {cam: decode_image_bit(f[f"vision/{s}/colors"][k]) for cam, s in CAMERAS.items()} if sample else {}
    table = pq.read_table(path("data_path", i))
    if not len(table) == ep["length"] == len(state):
        errors.append(f"length: {len(table)} rows, {ep['length']} in meta, {len(state)} in source")
    else:
        for name, ref in (("observation.state", state), ("action", action)):
            got = np.array(table[name].to_pylist(), dtype=np.float32)
            if not np.array_equal(got, ref):
                errors.append(f"{name} differs from source (max {np.abs(got - ref).max():.3g})")
    if set(table["episode_index"].to_pylist()) != {i}:
        errors.append("episode_index column")
    seen = {tasks[t] for t in set(table["task_index"].to_pylist())}
    if len(seen) != 1 or not seen <= set(sentences) or ep["tasks"] != list(seen):
        errors.append(f"instruction {seen} not one of the source's sentences")
    for cam, img in images.items():
        if img.shape[:2] != (240, 320):
            img = images[cam] = cv2.resize(img, (320, 240), interpolation=cv2.INTER_LINEAR)
    for cam in images:
        frame = video_frame(path("video_path", i, f"observation.images.{cam}"), k).astype(np.int16)
        mae = {other: float(np.abs(frame - img.astype(np.int16)).mean()) for other, img in images.items()}
        maes.append(mae[cam])
        if mae[cam] > MAX_IMAGE_MAE or any(mae[o] <= mae[cam] for o in mae if o != cam):
            errors.append(f"{cam} frame {k}: {', '.join(f'{o} {v:.1f}' for o, v in mae.items())}")
    return i, src, errors, maes


def finite(x):
    if isinstance(x, dict):
        return all(finite(v) for v in x.values())
    if isinstance(x, list):
        return all(finite(v) for v in x)
    return not isinstance(x, float) or math.isfinite(x)


if __name__ == "__main__":
    problems = []
    if len(jobs) != len(episodes):
        problems.append(f"{len(jobs)} source files, {len(episodes)} episodes in the dataset")
    with Pool(32) as pool:
        results = list(pool.imap(check, jobs[: len(episodes)], chunksize=16))
    problems += [f"episode {i} ({src}): {e}" for i, src, errs, _ in results for e in errs]
    maes = [m for *_, ms in results for m in ms] or [float("nan")]
    if len(maes) < len(source_dirs) * len(CAMERAS):
        problems.append(f"only {len(maes)} image spot checks ran")

    counts = defaultdict(Counter)  # task -> setting -> episodes
    for d in source_dirs:  # .../<config>/<task>/aloha_agilex/data
        counts[d.parents[1].name][re.sub(r"\d+$", "", d.parents[2].name)] += len(list(d.glob("episode_*.hdf5")))
    for name in ("stats.json", "relative_stats.json"):
        if (root / "meta" / name).exists() and not finite(json.loads((root / "meta" / name).read_text())):
            problems.append(f"meta/{name} has a NaN or inf")
    stats = json.loads((root / "meta/stats.json").read_text())

    print(f"{len(episodes)} episodes checked against {len(jobs)} source files in {len(source_dirs)} dirs")
    print(f"tasks: {len(counts)}; episodes per task by setting: "
          f"{dict(Counter(tuple(sorted(c.items())) for c in counts.values()))}")
    print(f"instructions: {len(tasks)} distinct")
    print(f"image spot checks: {len(maes)} frames, mean abs diff {np.mean(maes):.2f} (max {max(maes):.2f}) of 255")
    for key in ("observation.state", "action"):
        print(f"{key}: min {np.round(stats[key]['min'], 2).tolist()}\n{' ' * len(key)}  max {np.round(stats[key]['max'], 2).tolist()}")
    print(f"{len(problems)} problems")
    for line in problems[:20]:
        print("  " + line)
    sys.exit("FAILED" if problems else 0)
