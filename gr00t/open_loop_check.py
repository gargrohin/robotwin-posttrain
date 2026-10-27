"""Open-loop replay check of the GR00T N1.7 RoboTwin eval pipeline.

Takes recorded demo frames (HDF5) and pushes them through the same code the benchmark
uses: the sim client's conversion functions (eval_policy_xpolicylab.py), the websocket
msgpack codec, and the GR00T adapter `Model` (built from deploy.yml with eval.sh's
overrides). The predicted 16-step action chunk is compared with what the demo actually
did next, and with a "hold still" baseline that repeats the current joint state.
If the eval path fed the model something different from training (cameras, state
order, image format, action layout), the model error would be close to or above the
hold-still error.

Also, for one frame per task, compares the image the model gets at eval with the same
frame decoded from the training LeRobot video (GR00T's own decoder), and checks the
training parquet's state/action against the demo. The main run uses an eval-style
instruction (a "seen" template filled from the episode's scene info, as the eval
client does); each frame is also run with the one instruction that episode was
trained with, to see whether the wording matters.

One Model serves all tasks here; the server builds one per task, but task_name only
feeds default_prompt, which deploy.yml sets and which is unused when an instruction is
present.

Run in the GR00T venv (never plain `uv run`), from the repo root:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=7 \
  HF_TOKEN="$(cat ~/.cache/huggingface/token)" \
  RoboTwin/XPolicyLab/policy/GR00T_N17/gr00t_n17/.venv/bin/python gr00t/open_loop_check.py
Outputs go to outputs/open_loop_check/: summary.json, per_frame.csv, images_<task>.png.
"""

from __future__ import annotations

import argparse
import ast
import csv
import importlib
import importlib.util
import json
import os
import sys
import time
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ROBOTWIN = ROOT / "RoboTwin"
GR00T_DIR = ROBOTWIN / "XPolicyLab/policy/GR00T_N17"
DATASET = ROOT / "data/lerobot/RoboTwin-5task-aloha_agilex-joint"
OUT = ROOT / "outputs/open_loop_check"

TASKS = ["beat_block_hammer", "lift_pot", "handover_block", "stack_blocks_two", "open_laptop"]
CKPT = "RoboTwin-5task_30k-aloha_agilex-joint-0"
HORIZON = 16
# Flat 14-dim layout used by the sim (take_action) and the demos:
# left 6 joints, left gripper, right 6 joints, right gripper.
LEFT_ARM, RIGHT_ARM = np.arange(0, 6), np.arange(7, 13)
ARM, GRIP = np.r_[LEFT_ARM, RIGHT_ARM], np.array([6, 13])
PARTS = ("left_arm_joint_states", "left_ee_joint_states", "right_arm_joint_states", "right_ee_joint_states")

os.environ.setdefault("GR00T_VIDEO_BACKEND", "pyav")  # as in eval.sh / training


def load_model(checkpoint_num: str):
    """Build the adapter Model the way setup_policy_server.py does for gr00t/eval.sh."""
    import yaml

    with open(GR00T_DIR / "deploy.yml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    overrides = {  # eval.sh's --overrides (port/host are unused in-process)
        "port": "0", "host": "localhost", "bench_name": "RoboTwin", "task_name": TASKS[0],
        "ckpt_name": CKPT, "env_cfg_type": "aloha_agilex", "seed": "0",
        "policy_name": "GR00T_N17", "action_type": "joint", "action_dim": "14",
        "checkpoint_num": checkpoint_num, "cosmos_model_path": "nvidia/Cosmos-Reason2-2B",
    }
    for key, value in overrides.items():  # same value parsing as setup_policy_server.py
        try:
            cfg[key] = ast.literal_eval(value)
        except Exception:
            cfg[key] = value
    adapter = importlib.import_module("XPolicyLab.policy.GR00T_N17.model")
    return adapter, adapter.Model(cfg)


def load_eval_client():
    """Import RoboTwin's sim client for its observation/action conversion functions.

    It imports the simulator (`envs`, which needs sapien) at module level, but the
    functions used here are plain numpy, so those two imports are stubbed.
    """
    envs = types.ModuleType("envs")
    envs.CONFIGS_PATH = str(ROBOTWIN / "env_cfg/task_config") + "/"
    envs_utils = types.ModuleType("envs.utils")
    create_actor = types.ModuleType("envs.utils.create_actor")
    create_actor.UnStableError = type("UnStableError", (Exception,), {})
    envs.utils, envs_utils.create_actor = envs_utils, create_actor
    sys.modules.update({"envs": envs, "envs.utils": envs_utils, "envs.utils.create_actor": create_actor})

    path = ROBOTWIN / "scripts/eval_policy_xpolicylab.py"
    spec = importlib.util.spec_from_file_location("eval_policy_xpolicylab", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ws_roundtrip(value):
    """Encode/decode through the websocket msgpack codec, as client<->server do."""
    from client_server.ws.protocol.codec import decode_frame, encode_frame

    return decode_frame(encode_frame({"payload": value}))["payload"]


def load_demo(path: Path):
    """XPolicyLab's HDF5 loader (images decoded with decode_image_bit), retrying EFS errno 5."""
    from XPolicyLab.utils.data_loader import load

    for attempt in range(5):
        try:
            return load(str(path))
        except OSError:
            if attempt == 4:
                raise
            time.sleep(2 * (attempt + 1))


def flat(group: dict) -> np.ndarray:
    return np.concatenate([np.asarray(group[p], dtype=np.float32) for p in PARTS], axis=1)


def sim_observation(data: dict, t: int, sim_cam_of: dict) -> dict:
    """The dict task_env.get_obs() returns, filled from demo frame t."""
    obs = {"observation": {}, "joint_action": {}, "endpose": {}}
    for h5_cam, sim_cam in sim_cam_of.items():
        cam = data["vision"][h5_cam]
        obs["observation"][sim_cam] = {
            "rgb": cam["colors"][t],
            "intrinsic_cv": cam["intrinsic_matrix"][t],
            "extrinsic_cv": cam["extrinsics_matrix"][t],
        }
    state = data["state"]
    vector = []
    for side in ("left", "right"):
        arm = [float(x) for x in state[f"{side}_arm_joint_states"][t]]  # a list, like get_obs()
        gripper = float(state[f"{side}_ee_joint_states"][t][0])
        obs["joint_action"][f"{side}_arm"] = arm
        obs["joint_action"][f"{side}_gripper"] = gripper
        obs["endpose"][f"{side}_endpose"] = np.asarray(state[f"{side}_ee_poses"][t])
        obs["endpose"][f"{side}_gripper"] = gripper
        vector += arm + [gripper]
    obs["joint_action"]["vector"] = np.array(vector)
    return obs


def predict(client, model, xpl_obs: dict, sim_obs: dict) -> np.ndarray:
    """One eval-loop policy query: update_obs + get_action, actions as sent to take_action."""
    model.update_obs(ws_roundtrip(xpl_obs))
    chunk = client.normalize_action_chunk(ws_roundtrip({"result": model.get_action()})["result"])
    return np.stack([
        client.xpolicylab_action_to_robotwin(a, action_type="joint", current_observation=sim_obs)[0]
        for a in chunk
    ])


def lerobot_episode_index(task_i: int, ep: int, state: np.ndarray, lengths: dict):
    """Find the training episode holding this demo (expected task_i*50+ep; verified)."""
    import pandas as pd

    def matches(idx):
        df = pd.read_parquet(DATASET / f"data/chunk-000/episode_{idx:06d}.parquet")
        return len(df) == len(state) and np.allclose(np.stack(df["observation.state"]), state, atol=1e-6)

    guess = task_i * 50 + ep
    if lengths.get(guess) == len(state) and matches(guess):
        return guess
    for idx, n in lengths.items():
        if n == len(state) and matches(idx):
            return idx
    return None


def image_stats(img: np.ndarray) -> dict:
    return {
        "shape": list(img.shape), "dtype": str(img.dtype),
        "min": int(img.min()), "max": int(img.max()),
        "mean_rgb": [round(float(img[..., c].mean()), 2) for c in range(3)],
    }


def image_check(task, adapter, model, client, xpl_obs, sim_obs, t, lerobot_idx, video_keys, repeats):
    """Eval-path images (what Gr00tPolicy receives) vs the training video frame."""
    from PIL import Image, ImageDraw

    from gr00t.utils.video_utils import get_frames_by_indices

    encoded = adapter._encode_observation(ws_roundtrip(xpl_obs), model.default_prompt)
    result, rows = {}, []
    train_frames = {}
    for key, original_key in video_keys.items():
        eval_img = encoded["video"][key][0, 0]
        video = DATASET / f"videos/chunk-000/{original_key}/episode_{lerobot_idx:06d}.mp4"
        train_img = get_frames_by_indices(str(video), [t], video_backend="pyav")[0]
        train_frames[key] = train_img
        diff = np.abs(eval_img.astype(np.int16) - train_img.astype(np.int16))
        swapped = np.abs(eval_img[..., ::-1].astype(np.int16) - train_img.astype(np.int16))
        result[key] = {
            "eval": image_stats(eval_img), "train_video": image_stats(train_img),
            "mean_abs_diff": round(float(diff.mean()), 2),
            "mean_abs_diff_if_eval_were_BGR": round(float(swapped.mean()), 2),
        }
        rows.append((key, eval_img, train_img, np.clip(diff * 5, 0, 255).astype(np.uint8)))

    # Side-by-side PNG: per camera, eval image | training video frame | |diff| x5.
    h, w = rows[0][1].shape[:2]
    label_h = 18
    canvas = Image.new("RGB", (3 * w, len(rows) * (h + label_h)), "white")
    draw = ImageDraw.Draw(canvas)
    for r, (key, *imgs) in enumerate(rows):
        y = r * (h + label_h)
        for c, (img, name) in enumerate(zip(imgs, ("eval input", "training video", "|diff| x5"))):
            canvas.paste(Image.fromarray(img), (c * w, y + label_h))
            draw.text((c * w + 4, y + 3), f"{key}: {name}", fill="black")
    png = OUT / f"images_{task}.png"
    canvas.save(png)
    result["png"] = str(png)

    # Does the image difference (JPEG demo vs AV1 training video) change the actions?
    train_obs = {**xpl_obs, "vision": {k: dict(v) for k, v in xpl_obs["vision"].items()}}
    cam_of_key = {"front": "cam_head", "left_wrist": "cam_left_wrist", "right_wrist": "cam_right_wrist"}
    for key, img in train_frames.items():
        train_obs["vision"][cam_of_key[key]]["color"] = img
    with_eval = np.stack([predict(client, model, xpl_obs, sim_obs) for _ in range(repeats)])
    with_train = np.stack([predict(client, model, train_obs, sim_obs) for _ in range(repeats)])
    gap = np.abs(with_eval.mean(0) - with_train.mean(0))
    result["action_change_from_training_video_images"] = {
        "arm_mean_abs_rad": round(float(gap[:, ARM].mean()), 5),
        "gripper_mean_abs": round(float(gap[:, GRIP].mean()), 5),
        "sampling_spread_arm_rad": round(float(with_eval.std(0)[:, ARM].mean()), 5),
        "sampling_spread_gripper": round(float(with_eval.std(0)[:, GRIP].mean()), 5),
    }
    return result


def errors(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Mean abs error for arm joints (rad) and grippers, over the horizon and at steps 1/16."""
    e = np.abs(pred - gt)
    return {
        "arm": float(e[:, ARM].mean()), "grip": float(e[:, GRIP].mean()),
        "arm_step1": float(e[0, ARM].mean()), "arm_step16": float(e[-1, ARM].mean()),
        "grip_step1": float(e[0, GRIP].mean()), "grip_step16": float(e[-1, GRIP].mean()),
        "left_arm": float(e[:, LEFT_ARM].mean()), "right_arm": float(e[:, RIGHT_ARM].mean()),
        "left_grip": float(e[:, 6].mean()), "right_grip": float(e[:, 13].mean()),
        "per_joint": e.mean(0).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tasks", nargs="+", default=TASKS)
    parser.add_argument("--episodes", nargs="+", type=int, default=[0, 24, 48])
    parser.add_argument("--timesteps", type=int, default=10, help="frames per episode, spread evenly")
    parser.add_argument("--repeats", type=int, default=5, help="calls on one frame to measure sampling noise")
    parser.add_argument("--repeat-frames", type=int, default=2, help="frames per task for the noise check")
    parser.add_argument("--checkpoint-num", default="30000")
    parser.add_argument("--seed", type=int, default=0, help="torch seed, only for reproducible numbers")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    import torch

    torch.manual_seed(args.seed)
    adapter, model = load_model(args.checkpoint_num)  # before the client: it prepends RoboTwin dirs to sys.path
    client = load_eval_client()
    sim_cam_of = {xpl: sim for sim, xpl in client.CAMERA_NAME_MAP.items()}  # cam_head -> head_camera, ...
    modality = json.loads((DATASET / "meta/modality.json").read_text())
    video_keys = {k: v["original_key"] for k, v in modality["video"].items()}
    lengths, train_instruction_of = {}, {}
    with open(DATASET / "meta/episodes.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            lengths[row["episode_index"]] = row["length"]
            train_instruction_of[row["episode_index"]] = row["tasks"][0]

    rows, summary = [], {"checkpoint": f"{CKPT}/checkpoint-{args.checkpoint_num}", "tasks": {}}
    for task in args.tasks:
        task_dir = ROBOTWIN / f"data/demo_clean/{task}/aloha_agilex"
        scene_info = json.loads((task_dir / "scene_info.json").read_text())
        model_errs, train_instr_errs, hold_errs, spreads = [], [], [], []
        checks = {"instructions": []}
        for ep_i, ep in enumerate(args.episodes):
            data = load_demo(task_dir / f"data/episode_{ep:07d}.hdf5")
            state, action = flat(data["state"]), flat(data["action"])
            assert np.allclose(action[:-1], state[1:]), "expected action[t] == state[t+1]"
            lerobot_idx = lerobot_episode_index(TASKS.index(task), ep, state, lengths)

            # Eval-style instruction ("seen" template filled from this episode's scene info),
            # and the one instruction this episode was trained with.
            info = scene_info[f"episode_{ep}"]["info"]
            instruction = client.build_instruction({"task_name": task}, {"info": info}, "seen", 100, ep)
            train_instruction = train_instruction_of[lerobot_idx]
            checks["instructions"].append({
                "episode": ep, "eval_style": instruction, "trained_with": train_instruction,
                "eval_style_in_this_demo's_instruction_list": instruction in data["instructions"],
            })

            if ep_i == 0:
                import pandas as pd

                df = pd.read_parquet(DATASET / f"data/chunk-000/episode_{lerobot_idx:06d}.parquet")
                checks["lerobot_episode_index"] = lerobot_idx
                checks["train_parquet_max_abs_diff"] = {
                    "state": float(np.abs(np.stack(df["observation.state"]) - state).max()),
                    "action": float(np.abs(np.stack(df["action"]) - action).max()),
                }

            model.reset()  # the eval loop resets the policy at the start of each episode
            ts = np.linspace(0, len(state) - HORIZON, args.timesteps).round().astype(int)
            for f_i, t in enumerate(ts):
                sim_obs = sim_observation(data, t, sim_cam_of)
                xpl_obs = client.robotwin_obs_to_xpolicylab(sim_obs, instruction=instruction, env_idx=0)
                gt = action[t:t + HORIZON]
                hold = np.repeat(state[t][None], HORIZON, axis=0)
                pred = predict(client, model, xpl_obs, sim_obs)
                xpl_obs_tr = client.robotwin_obs_to_xpolicylab(sim_obs, instruction=train_instruction, env_idx=0)
                m, h = errors(pred, gt), errors(hold, gt)
                m_tr = errors(predict(client, model, xpl_obs_tr, sim_obs), gt)
                model_errs.append(m)
                train_instr_errs.append(m_tr)
                hold_errs.append(h)
                rows.append({"task": task, "episode": ep, "t": int(t), "T": len(state),
                             **{f"model_{k}": round(v, 5) for k, v in m.items() if k != "per_joint"},
                             "model_train_instr_arm": round(m_tr["arm"], 5),
                             "model_train_instr_grip": round(m_tr["grip"], 5),
                             **{f"hold_{k}": round(v, 5) for k, v in h.items() if k != "per_joint"}})

                if ep_i == 0 and f_i in np.linspace(1, len(ts) - 2, args.repeat_frames).round().astype(int):
                    samples = np.stack([pred] + [predict(client, model, xpl_obs, sim_obs)
                                                 for _ in range(args.repeats - 1)])
                    spreads.append({
                        "t": int(t),
                        "std_arm_rad": float(samples.std(0)[:, ARM].mean()),
                        "std_grip": float(samples.std(0)[:, GRIP].mean()),
                        "max_pairwise_arm_rad": float(max(np.abs(a - b)[:, ARM].mean()
                                                          for a in samples for b in samples)),
                        "model_err_arm_each_sample": [round(errors(s, gt)["arm"], 5) for s in samples],
                    })
                if ep_i == 0 and f_i == len(ts) // 2:
                    checks["image"] = image_check(task, adapter, model, client, xpl_obs, sim_obs, int(t),
                                                  lerobot_idx, video_keys, args.repeats)
                    packed = np.concatenate([v[0, 0] for v in adapter._encode_observation(
                        ws_roundtrip(xpl_obs), model.default_prompt)["state"].values()])
                    checks["adapter_state_vs_train_parquet_max_abs_diff"] = float(
                        np.abs(packed - np.stack(df["observation.state"])[t]).max())
            print(f"{task} ep{ep}: model arm {np.mean([e['arm'] for e in model_errs[-len(ts):]]):.4f} rad, "
                  f"hold arm {np.mean([e['arm'] for e in hold_errs[-len(ts):]]):.4f} rad", flush=True)

        def avg(errs):
            out = {k: float(np.mean([e[k] for e in errs])) for k in errs[0] if k != "per_joint"}
            out["per_joint"] = np.mean([e["per_joint"] for e in errs], axis=0).round(5).tolist()
            return out

        summary["tasks"][task] = {"frames": len(model_errs), "model": avg(model_errs),
                                  "model_with_training_instruction": avg(train_instr_errs),
                                  "hold_still": avg(hold_errs), "sampling_spread": spreads, "checks": checks}

    with open(OUT / "per_frame.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\nMean abs error over the 16-step chunk (arms: rad; grippers: 0=closed..1=open)")
    print("| task | model arm | hold arm | model arm s1/s16 | hold arm s1/s16 | model grip | hold grip "
          "| model L/R arm | arm, train instr | noise std arm |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for task, s in summary["tasks"].items():
        m, h, tr = s["model"], s["hold_still"], s["model_with_training_instruction"]
        noise = np.mean([x["std_arm_rad"] for x in s["sampling_spread"]]) if s["sampling_spread"] else float("nan")
        print(f"| {task} | {m['arm']:.4f} | {h['arm']:.4f} | {m['arm_step1']:.4f}/{m['arm_step16']:.4f} | "
              f"{h['arm_step1']:.4f}/{h['arm_step16']:.4f} | {m['grip']:.4f} | {h['grip']:.4f} | "
              f"{m['left_arm']:.4f}/{m['right_arm']:.4f} | {tr['arm']:.4f} | {noise:.4f} |")
    print(f"\nWrote {OUT}/summary.json, per_frame.csv, images_<task>.png")


if __name__ == "__main__":
    main()
