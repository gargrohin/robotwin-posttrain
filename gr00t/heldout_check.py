"""Memorization check: does a checkpoint predict the demonstrator's actions much better on
demos it trained on than on demos it never saw?

Three demo sets per task, 5 demos each:
  train_randomized  randomized demos the mix200 model trained on (episodes 20, 60, 100, 140, 180)
  train_clean       clean demos every model trained on (episodes 5, 15, 25, 35, 45)
  unseen_randomized randomized demos no model trained on (episodes 200-204, fetched with
                    fetch_randomized.py --start 200 --config demo_randomized_heldout)
At 8 evenly spaced moments per demo, the model predicts the next 16 actions through the
same code path as the benchmark (gr00t/open_loop_check.py's helpers), and the error is the
mean absolute arm-joint difference from what the demonstrator did (radians). "Hold still"
(repeat the current joint state) gives the scale. Every demo gets the first sentence of
its own instruction list, so seen and unseen demos are treated alike.

A model that memorized its training demos has a much lower error on train_randomized than
on unseen_randomized; one that learned the skill has similar errors on both.

Run in the GR00T venv from the repo root, one checkpoint per process:
  --prepare          extract the sampled moments once into outputs/heldout_check/frames.pkl
  --ckpt C --checkpoint-num N   evaluate one checkpoint, writing outputs/heldout_check/<C>_<N>.json
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import open_loop_check as olc  # noqa: E402

OUT = olc.ROOT / "outputs/heldout_check"
FRAMES = OUT / "frames.pkl"
TASKS = ["pick_dual_bottles", "place_object_stand", "move_pillbottle_pad", "place_empty_cup", "place_phone_stand",
         "move_can_pot", "dump_bin_bigbin", "handover_mic", "grab_roller", "click_bell"]
DATA = olc.ROBOTWIN / "data"
SETS = {
    "train_randomized": lambda t, i: DATA / f"demo_randomized{i // 50}/{t}/aloha_agilex/data/episode_{i:07d}.hdf5",
    "train_clean": lambda t, i: DATA / f"demo_clean/{t}/aloha_agilex/data/episode_{i:07d}.hdf5",
    "unseen_randomized": lambda t, i: DATA / f"demo_randomized_heldout/{t}/aloha_agilex/data/episode_{i:07d}.hdf5",
}
EPISODES = {"train_randomized": [20, 60, 100, 140, 180], "train_clean": [5, 15, 25, 35, 45],
            "unseen_randomized": [200, 201, 202, 203, 204]}
MOMENTS = 8


def prepare():
    client = olc.load_eval_client()
    sim_cam_of = {xpl: sim for sim, xpl in client.CAMERA_NAME_MAP.items()}
    demos = []
    for name, path_of in SETS.items():
        for task in TASKS:
            for ep in EPISODES[name]:
                data = olc.load_demo(path_of(task, ep))
                state, action = olc.flat(data["state"]), olc.flat(data["action"])
                instruction = list(data["instructions"])[0]
                ts = np.linspace(0, len(state) - olc.HORIZON, MOMENTS).round().astype(int)
                demos.append({
                    "set": name, "task": task, "episode": ep, "instruction": instruction,
                    "moments": [(olc.sim_observation(data, t, sim_cam_of), action[t:t + olc.HORIZON],
                                 np.repeat(state[t][None], olc.HORIZON, axis=0)) for t in ts],
                })
    OUT.mkdir(parents=True, exist_ok=True)
    with open(FRAMES, "wb") as f:
        pickle.dump(demos, f)
    print(f"{len(demos)} demos x {MOMENTS} moments -> {FRAMES}")


def evaluate(ckpt, num, seed):
    import torch

    torch.manual_seed(seed)
    olc.CKPT = ckpt
    _, model = olc.load_model(num)  # before the client: it prepends RoboTwin dirs to sys.path
    client = olc.load_eval_client()
    with open(FRAMES, "rb") as f:
        demos = pickle.load(f)
    errs = {}  # (set, task) -> [(model arm error, hold-still arm error)]
    for d in demos:
        model.reset()  # as the eval loop does at each episode start
        for sim_obs, gt, hold in d["moments"]:
            xpl_obs = client.robotwin_obs_to_xpolicylab(sim_obs, instruction=d["instruction"], env_idx=0)
            pred = olc.predict(client, model, xpl_obs, sim_obs)
            errs.setdefault((d["set"], d["task"]), []).append((olc.errors(pred, gt)["arm"], olc.errors(hold, gt)["arm"]))
    result = {"ckpt": ckpt, "checkpoint": num, "sets": {}}
    for name in SETS:
        rows = [e for (s, _), v in errs.items() if s == name for e in v]
        m, h = np.mean([r[0] for r in rows]), np.mean([r[1] for r in rows])
        result["sets"][name] = {
            "arm_error": float(m), "hold_still": float(h), "moments": len(rows),
            "per_task": {t: float(np.mean([r[0] for r in v])) for (s, t), v in errs.items() if s == name},
        }
    with open(OUT / f"{ckpt}_{num}.json", "w") as f:
        json.dump(result, f, indent=1)
    print(json.dumps({k: {kk: round(vv, 4) for kk, vv in v.items() if kk != "per_task"}
                      for k, v in result["sets"].items()}))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--ckpt")
    p.add_argument("--checkpoint-num")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    prepare() if a.prepare else evaluate(a.ckpt, a.checkpoint_num, a.seed)
