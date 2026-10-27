"""Live-sim observation check: does the eval observation look like the training demos?

Creates one RoboTwin task env the way scripts/eval_policy_xpolicylab.py does, resets it
with the seed of demo episode 0 (so the scene should match that demo's first frame),
converts get_obs() with the client's robotwin_obs_to_xpolicylab(), and compares its
images and state with frame 0 of the demo: resolution, dtype, value range, mean R/G/B,
pixel difference, joint values.

Run in the RoboTwin venv, from the repo root (only the default `rt` shader works here):
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=7 \
  RoboTwin/.venv/bin/python gr00t/live_obs_check.py --task beat_block_hammer
Outputs go to outputs/open_loop_check/: live_obs_<task>.npz, live_vs_demo_<task>.{png,json}.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ROBOTWIN = ROOT / "RoboTwin"
OUT = ROOT / "outputs/open_loop_check"
PARTS = ("left_arm_joint_states", "left_ee_joint_states", "right_arm_joint_states", "right_ee_joint_states")


def stats(img):
    return {"shape": list(img.shape), "dtype": str(img.dtype), "min": int(img.min()), "max": int(img.max()),
            "mean_rgb": [round(float(img[..., c].mean()), 2) for c in range(3)]}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--task", default="beat_block_hammer")
    parser.add_argument("--task-config", default="demo_clean")
    args_cli = parser.parse_args()
    task = args_cli.task
    OUT.mkdir(parents=True, exist_ok=True)

    os.chdir(ROBOTWIN)  # eval_policy.sh runs the client from RoboTwin/ (relative asset paths)
    spec = importlib.util.spec_from_file_location("eval_policy_xpolicylab", ROBOTWIN / "scripts/eval_policy_xpolicylab.py")
    client = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(client)
    client.add_xpolicylab_paths(client.DEFAULT_XPOLICYLAB_ROOT)
    from XPolicyLab.utils.data_loader import load

    # Same env construction as the eval client's main()/eval_remote_policy().
    usr_args = {"task_name": task, "task_config": args_cli.task_config, "policy_name": "GR00T_N17",
                "ckpt_name": "live_obs_check"}
    args, _ = client.load_task_args(usr_args)
    args["eval_mode"] = True
    args["render_freq"] = 0
    task_dir = ROBOTWIN / f"data/{args_cli.task_config}/{task}/aloha_agilex"
    seed = int((task_dir / "seed.txt").read_text().split()[0])  # seed of demo episode 0
    task_env = client.class_decorator(task)
    task_env.setup_demo(now_ep_num=0, seed=seed, is_test=True, **args)
    try:
        observation = task_env.get_obs()
        xpl = client.robotwin_obs_to_xpolicylab(observation, instruction=task, env_idx=0, task_env=task_env)
    finally:
        task_env.close_env()

    np.savez(OUT / f"live_obs_{task}.npz",
             **{f"img_{k}": v["color"] for k, v in xpl["vision"].items()},
             **{f"state_{k}": np.asarray(v) for k, v in xpl["state"].items()})

    demo = load(str(task_dir / "data/episode_0000000.hdf5"))
    result = {"task": task, "seed": seed, "cameras": {}}
    rows = []
    for cam in ("cam_head", "cam_left_wrist", "cam_right_wrist"):
        live = np.asarray(xpl["vision"][cam]["color"])
        ref = demo["vision"][cam]["colors"][0]
        diff = np.abs(live.astype(np.int16) - ref.astype(np.int16))
        swapped = np.abs(live[..., ::-1].astype(np.int16) - ref.astype(np.int16))
        result["cameras"][cam] = {"live": stats(live), "demo_frame0": stats(ref),
                                  "mean_abs_diff": round(float(diff.mean()), 2),
                                  "mean_abs_diff_if_live_were_BGR": round(float(swapped.mean()), 2)}
        rows.append((cam, live, ref, np.clip(diff * 5, 0, 255).astype(np.uint8)))

    live_state = np.concatenate([np.asarray(xpl["state"][p.replace("_states", "_state")], dtype=np.float32).reshape(-1)
                                 for p in PARTS])
    demo_state = np.concatenate([np.asarray(demo["state"][p][0], dtype=np.float32).reshape(-1) for p in PARTS])
    result["state"] = {"live": live_state.round(4).tolist(), "demo_frame0": demo_state.round(4).tolist(),
                       "max_abs_diff": float(np.abs(live_state - demo_state).max())}

    from PIL import Image, ImageDraw

    h, w = rows[0][1].shape[:2]
    canvas = Image.new("RGB", (3 * w, len(rows) * (h + 18)), "white")
    draw = ImageDraw.Draw(canvas)
    for r, (cam, *imgs) in enumerate(rows):
        for c, (img, name) in enumerate(zip(imgs, ("live sim (eval input)", "demo frame 0", "|diff| x5"))):
            canvas.paste(Image.fromarray(img), (c * w, r * (h + 18) + 18))
            draw.text((c * w + 4, r * (h + 18) + 3), f"{cam}: {name}", fill="black")
    canvas.save(OUT / f"live_vs_demo_{task}.png")
    (OUT / f"live_vs_demo_{task}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
