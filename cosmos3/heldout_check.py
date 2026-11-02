"""Open-loop check of a Cosmos3-Nano checkpoint on demo moments: GR00T's held-out check, same
moments, same metric, same output format.

The moments are gr00t/heldout_check.py's (outputs/heldout_check/frames.pkl): 10 tasks x 15 demos
(5 trained randomized, 5 trained clean, 5 unseen randomized) x 8 moments. At each moment the
policy predicts the next 16 actions through the benchmark's code path (sim client conversion ->
websocket codec -> Cosmos3_Nano adapter Model, built from its deploy.yml as cosmos3/eval.sh does),
and the error is the mean absolute arm-joint difference from what the demonstrator did (rad),
next to "hold still". Compare with GR00T's outputs/heldout_check/<ckpt>_<step>.json (mix200_150k at
150k: ~0.016-0.018 rad on every set, hold still ~0.064).

Run in the Cosmos venv from the repo root, on one free GPU (never the training GPUs):
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<gpu> LD_LIBRARY_PATH= \
  HF_TOKEN="$(cat ~/.cache/huggingface/token)" \
  /opt/dlami/nvme/rohingarg/cosmos3-venv/bin/python cosmos3/heldout_check.py --checkpoint-num 1500
  --stride 5: quick run, the first demo of each (set, task): 240 moments instead of 1,200.
  --num-steps/--guidance/--shift/--actions-per-call: sampling overrides (default: deploy.yml).
Writes outputs/heldout_check/<ckpt>_<iteration>[_<overrides>].json.
"""

import argparse
import importlib
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gr00t"))
import heldout_check as hc  # noqa: E402
import open_loop_check as olc  # noqa: E402

DEPLOY = olc.ROBOTWIN / "XPolicyLab/policy/Cosmos3_Nano/deploy.yml"
RUN_DIR = "/mnt/efs/rohingarg/cri/cosmos3/runs/cosmos3_action_robotwin/action_sft/action_policy_robotwin_nano"


def load_model(ckpt: str, run_dir: str, checkpoint_num: str, overrides: dict):
    """The adapter Model as setup_policy_server.py builds it for cosmos3/eval.sh."""
    cfg = yaml.safe_load(DEPLOY.read_text())
    cfg.update({  # eval.sh's --overrides (port/host unused in-process)
        "port": 0, "host": "localhost", "bench_name": "RoboTwin", "task_name": hc.TASKS[0], "ckpt_name": ckpt,
        "env_cfg_type": "aloha_agilex", "seed": 0, "policy_name": "Cosmos3_Nano", "action_type": "joint",
        "action_dim": 14, "checkpoint_num": checkpoint_num, "run_dir": run_dir, **overrides,
    })
    return importlib.import_module("XPolicyLab.policy.Cosmos3_Nano.model").Model(cfg)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--checkpoint-num", default="last")
    p.add_argument("--ckpt", default="cosmos3_nano_mix200", help="label for the output file")
    p.add_argument("--run-dir", default=RUN_DIR)
    p.add_argument("--stride", type=int, default=1, help="use every Nth demo (5 = one per set and task)")
    p.add_argument("--num-steps", type=int)
    p.add_argument("--guidance", type=float)
    p.add_argument("--shift", type=float)
    p.add_argument("--actions-per-call", type=int)
    a = p.parse_args()
    overrides = {k: v for k, v in {"num_steps": a.num_steps, "guidance": a.guidance, "shift": a.shift,
                                   "actions_per_call": a.actions_per_call}.items() if v is not None}

    model = load_model(a.ckpt, a.run_dir, a.checkpoint_num, overrides)  # before the client edits sys.path
    client = olc.load_eval_client()
    with open(hc.FRAMES, "rb") as f:
        demos = pickle.load(f)[:: a.stride]
    errs, call_s = {}, []  # (set, task) -> [(model arm error, hold-still arm error)]
    for i, d in enumerate(demos):
        model.reset()  # as the eval loop does at each episode start
        for sim_obs, gt, hold in d["moments"]:
            xpl_obs = client.robotwin_obs_to_xpolicylab(sim_obs, instruction=d["instruction"], env_idx=0)
            start = time.monotonic()
            pred = olc.predict(client, model, xpl_obs, sim_obs)
            call_s.append(time.monotonic() - start)
            n = len(pred)  # < 16 only with --actions-per-call
            errs.setdefault((d["set"], d["task"]), []).append(
                (olc.errors(pred, gt[:n])["arm"], olc.errors(hold[:n], gt[:n])["arm"]))
        print(f"{i + 1}/{len(demos)} demos, {np.median(call_s):.2f} s per call", flush=True)

    iteration = int(model.checkpoint.name.removeprefix("iter_"))
    result = {
        "ckpt": a.ckpt, "checkpoint": iteration, "checkpoint_dir": str(model.checkpoint), "stride": a.stride,
        "settings": {"num_steps": model.num_steps, "guidance": model.guidance, "shift": model.shift,
                     "actions_per_call": model.actions_per_call},
        "seconds_per_call": {"median": float(np.median(call_s)), "mean": float(np.mean(call_s))}, "sets": {},
    }
    for name in hc.SETS:
        rows = [e for (s, _), v in errs.items() if s == name for e in v]
        if not rows:
            continue
        result["sets"][name] = {
            "arm_error": float(np.mean([r[0] for r in rows])), "hold_still": float(np.mean([r[1] for r in rows])),
            "moments": len(rows),
            "per_task": {t: float(np.mean([r[0] for r in v])) for (s, t), v in errs.items() if s == name},
        }
    suffix = "".join(f"_{k}{v}" for k, v in overrides.items()) + (f"_stride{a.stride}" if a.stride > 1 else "")
    out = hc.OUT / f"{a.ckpt}_{iteration}{suffix}.json"
    out.write_text(json.dumps(result, indent=1))
    print(json.dumps({k: {kk: round(vv, 4) for kk, vv in v.items() if kk != "per_task"}
                      for k, v in result["sets"].items()}))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
