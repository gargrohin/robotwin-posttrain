"""Fetch RoboTwin raw-format demos from the HF zips and write them in XPolicyLab format.

RoboTwin publishes randomized demos only in its raw collection format
(`<task>/aloha-agilex_randomized_500.zip`, 500 episodes, ~8 GB per task), while
XPolicyLab's LeRobot converter reads only its own "xspark v1.0" layout (the format of
`demo_clean.zip`). This reads the zip remotely (HfFileSystem range requests), takes
episodes 0..N-1 and converts each one:

  state/*[t]  = raw[t],  action/*[t] = raw[t + 1]  (so T - 1 steps), float32
    *_arm_joint_states <- joint_action/*_arm, *_ee_joint_states <- joint_action/*_gripper,
    *_ee_poses <- endpose/*_endpose
  vision/cam_{head,left_wrist,right_wrist,third_view} <- observation/{head,left,right,front}_camera
    colors: decode_image_bit -> encode_image_bit (XPolicyLab's standard RGB JPEG),
    extrinsics_matrix <- cam2world_gl, intrinsic_matrix <- intrinsic_cv, shape = (H, W, 3)
  instructions = JSON list of the episode's "seen" instructions ("unseen" stay for eval)
  data_format_version = "v1.0", additional_info/frequency = 15

The mapping was derived from, and is checked against, the clean demos that RoboTwin
publishes in both formats (`--verify`: raw aloha-agilex_clean_50 vs demo_clean).

Writes RoboTwin/data/<config>/<task>/aloha_agilex/data/episode_XXXXXXX.hdf5, the layout
transform_lerobot_v30_format.py expects ("<config>.<task>.aloha_agilex").

--group G puts episode i in <config><i // G> (e.g. demo_randomized0..3 for 200/50), so the
LeRobot conversion can run many small per-group jobs in parallel.

usage: fetch_randomized.py --n 200 --group 50 --config demo_randomized --jobs 24 <task> ...
       fetch_randomized.py --verify --n 2 <task> [<task> ...]   (compare with demo_clean)
"""

import argparse
import io
import json
import os
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import h5py
import numpy as np
from huggingface_hub import HfFileSystem

ROBOTWIN = Path(__file__).resolve().parents[1] / "RoboTwin"
sys.path.insert(0, str(ROBOTWIN))
from XPolicyLab.utils.process_data import decode_image_bit, encode_image_bit  # noqa: E402

ZIP = "datasets/TianxingChen/RoboTwin2.0/dataset/{task}/aloha-agilex_{kind}.zip"
CAMERAS = {"cam_head": "head_camera", "cam_left_wrist": "left_camera",
           "cam_right_wrist": "right_camera", "cam_third_view": "front_camera"}


def to_xspark(raw_bytes: bytes, instructions: dict, dst: Path) -> None:
    raw = h5py.File(io.BytesIO(raw_bytes), "r")
    ja, ep = raw["joint_action"], raw["endpose"]
    parts = {
        "left_arm_joint_states": ja["left_arm"][()],
        "left_ee_joint_states": ja["left_gripper"][()].reshape(-1, 1),
        "left_ee_poses": ep["left_endpose"][()],
        "right_arm_joint_states": ja["right_arm"][()],
        "right_ee_joint_states": ja["right_gripper"][()].reshape(-1, 1),
        "right_ee_poses": ep["right_endpose"][()],
    }
    tmp = dst.with_suffix(".tmp")
    with h5py.File(tmp, "w") as out:
        for name, value in parts.items():
            out[f"state/{name}"] = value[:-1].astype(np.float32)
            out[f"action/{name}"] = value[1:].astype(np.float32)
        for xs_cam, raw_cam in CAMERAS.items():
            obs = raw[f"observation/{raw_cam}"]
            frames = [decode_image_bit(b) for b in obs["rgb"][:-1]]
            out[f"vision/{xs_cam}/colors"] = np.array(encode_image_bit(frames))
            out[f"vision/{xs_cam}/extrinsics_matrix"] = obs["cam2world_gl"][:-1].astype(np.float32)
            out[f"vision/{xs_cam}/intrinsic_matrix"] = obs["intrinsic_cv"][:-1].astype(np.float32)
            out[f"vision/{xs_cam}/shape"] = np.array(frames[0].shape, dtype=np.int32)
        out.create_dataset("instructions", data=json.dumps(instructions["seen"]),
                           dtype=h5py.string_dtype())
        out.create_dataset("data_format_version", data="v1.0", dtype=h5py.string_dtype())
        out["additional_info/frequency"] = np.int32(15)
    os.replace(tmp, dst)


def open_zip(task, kind):
    f = HfFileSystem().open(ZIP.format(task=task, kind=kind), "rb", block_size=64 << 20)
    z = zipfile.ZipFile(f)
    return f, z, z.namelist()[0].split("/")[0]


def fetch(task, n, config, group):
    def out_dir(i):
        d = ROBOTWIN / "data" / f"{config}{i // group if group else ''}" / task / "aloha_agilex" / "data"
        d.mkdir(parents=True, exist_ok=True)
        return d
    for attempt in range(8):
        try:
            f, z, pre = open_zip(task, "randomized_500")
            with f:
                for i in range(n):
                    dst = out_dir(i) / f"episode_{i:07d}.hdf5"
                    if dst.exists():
                        continue
                    instr = json.loads(z.read(f"{pre}/instructions/episode{i}.json"))
                    to_xspark(z.read(f"{pre}/data/episode{i}.hdf5"), instr, dst)
            count = sum(1 for i in range(n) if (out_dir(i) / f"episode_{i:07d}.hdf5").exists())
            print(f"[fetch] {task}: {count} episodes", flush=True)
            return task, count
        except Exception as e:  # HF rate limiting (error page -> "not a zip file"), EFS errno 5
            print(f"[retry {attempt + 1}/8] {task}: {e}", flush=True)
            time.sleep(60 * (attempt + 1))  # outlast HF's 5-minute rate window
    return task, -1


def verify(task, n):
    """Convert raw clean episodes and compare with the published demo_clean files."""
    f, z, pre = open_zip(task, "clean_50")
    tmpdir = ROBOTWIN / "data" / "_verify_xspark"
    tmpdir.mkdir(parents=True, exist_ok=True)
    with f:
        for i in range(n):
            mine = tmpdir / f"{task}_{i}.hdf5"
            to_xspark(z.read(f"{pre}/data/episode{i}.hdf5"),
                      json.loads(z.read(f"{pre}/instructions/episode{i}.json")), mine)
            ref = ROBOTWIN / "data/demo_clean" / task / "aloha_agilex/data" / f"episode_{i:07d}.hdf5"
            with h5py.File(mine, "r") as a, h5py.File(ref, "r") as b:
                keys = []
                a.visit(lambda k: keys.append(k) if isinstance(a[k], h5py.Dataset) else None)
                other = []
                b.visit(lambda k: other.append(k) if isinstance(b[k], h5py.Dataset) else None)
                assert sorted(keys) == sorted(other), (set(keys) ^ set(other))
                worst = 0.0
                for k in keys:
                    x, y = a[k][()], b[k][()]
                    if k.endswith("/colors"):
                        d = max(np.abs(decode_image_bit(p).astype(int) - decode_image_bit(q).astype(int)).mean()
                                for p, q in zip(x, y))
                        worst = max(worst, d)
                    elif k == "instructions":
                        assert json.loads(x) == json.loads(y), k
                    elif isinstance(x, bytes) or isinstance(y, bytes):
                        assert x == y, k
                    else:
                        assert x.shape == y.shape and np.allclose(x, y, atol=1e-6), (k, x.shape, y.shape)
                print(f"[verify] {task} episode {i}: all {len(keys)} fields match; "
                      f"images re-encoded, worst frame mean |diff| {worst:.2f}/255", flush=True)
            mine.unlink()


parser = argparse.ArgumentParser()
parser.add_argument("tasks", nargs="+")
parser.add_argument("--n", type=int, default=200)
parser.add_argument("--config", default="demo_randomized")
parser.add_argument("--jobs", type=int, default=12)
parser.add_argument("--group", type=int, default=0)
parser.add_argument("--verify", action="store_true")
args = parser.parse_args()

if args.verify:
    for task in args.tasks:
        verify(task, args.n)
    sys.exit(0)
# Processes, not threads: JPEG decode/encode is CPU work that the GIL would serialize.
with ProcessPoolExecutor(args.jobs) as pool:
    results = list(pool.map(partial(fetch, n=args.n, config=args.config, group=args.group), args.tasks))
bad = [t for t, c in results if c != args.n]
print(f"[fetch] done: {len(results) - len(bad)}/{len(results)} tasks complete", flush=True)
if bad:
    sys.exit(f"incomplete: {bad}")
