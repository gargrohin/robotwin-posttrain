"""Open-loop check of a FLUX 3 Action checkpoint on demo moments: GR00T's held-out check, same
moments, same metric, same output format.

The moments are gr00t/heldout_check.py's (outputs/heldout_check/frames.pkl): 10 tasks x 15 demos
(5 trained randomized, 5 trained clean, 5 unseen randomized) x 8 moments. At each moment the
policy predicts its action chunk through the benchmark's code path (sim client conversion ->
websocket codec -> Flux3_Action adapter Model, built from its deploy.yml as flux3/eval.sh does),
and the error is the mean absolute arm-joint difference from what the demonstrator did (rad) over
the first 16 actions (the horizon frames.pkl stores, and the one GR00T's and Cosmos's numbers use;
Flux predicts and executes 32), next to "hold still". Compare with
outputs/heldout_check/<ckpt>_<step>.json of GR00T (mix200_150k at 150k: ~0.016-0.018 rad on every
set, hold still ~0.064) and cosmos3_nano_mix200_*.

Run in the flux-action venv from the repo root, on one free GPU (never the training GPUs):
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<gpu> LD_LIBRARY_PATH= HF_HUB_OFFLINE=1 \
  /opt/dlami/nvme/rohingarg/flux3-venv/bin/python flux3/heldout_check.py --checkpoint-num 1050
  --stride 5: quick run, the first demo of each (set, task): 240 moments instead of 1,200.
  --num-inference-steps/--guidance-scale/--guidance-scale-action/--sampler-shift/--actions-per-call:
  sampling overrides (default: deploy.yml, i.e. the DROID reference settings).
Writes outputs/heldout_check/<ckpt>_<step>[_<overrides>].json.
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

DEPLOY = olc.ROBOTWIN / "XPolicyLab/policy/Flux3_Action/deploy.yml"
EXPORT_ROOT = "/opt/dlami/nvme/rohingarg/flux3-eval/exports/f3a-mix200-b512-2n-v2"
WEIGHTS = "/opt/dlami/nvme/rohingarg/flux3-weights"


def load_model(ckpt: str, export_root: str, weights: str, checkpoint_num: str, overrides: dict):
    """The adapter Model as setup_policy_server.py builds it for flux3/eval.sh."""
    cfg = yaml.safe_load(DEPLOY.read_text())
    cfg.update({  # eval.sh's --overrides (port/host unused in-process)
        "port": 0, "host": "localhost", "bench_name": "RoboTwin", "task_name": hc.TASKS[0], "ckpt_name": ckpt,
        "env_cfg_type": "aloha_agilex", "seed": 0, "policy_name": "Flux3_Action", "action_type": "joint",
        "action_dim": 14, "checkpoint_num": checkpoint_num, "export_root": export_root,
        "video_vae_path": f"{weights}/video_vae.safetensors", "text_encoder_path": f"{weights}/text_encoder",
        **overrides,
    })
    return importlib.import_module("XPolicyLab.policy.Flux3_Action.model").Model(cfg)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--checkpoint-num", default="last")
    p.add_argument("--ckpt", default="flux3_mix200_b512", help="label for the output file")
    p.add_argument("--export-root", default=EXPORT_ROOT)
    p.add_argument("--weights", default=WEIGHTS)
    p.add_argument("--stride", type=int, default=1, help="use every Nth demo (5 = one per set and task)")
    p.add_argument("--num-inference-steps", type=int)
    p.add_argument("--guidance-scale", type=float)
    p.add_argument("--guidance-scale-action", type=float)
    p.add_argument("--sampler-shift", type=float)
    p.add_argument("--actions-per-call", type=int)
    a = p.parse_args()
    overrides = {k: v for k, v in {"num_inference_steps": a.num_inference_steps, "guidance_scale": a.guidance_scale,
                                   "guidance_scale_action": a.guidance_scale_action, "sampler_shift": a.sampler_shift,
                                   "actions_per_call": a.actions_per_call}.items() if v is not None}

    import torch

    model = load_model(a.ckpt, a.export_root, a.weights, a.checkpoint_num, overrides)  # before the client edits sys.path
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
            n = min(len(pred), len(gt))  # the stored horizon (16) unless --actions-per-call is smaller
            errs.setdefault((d["set"], d["task"]), []).append(
                (olc.errors(pred[:n], gt[:n])["arm"], olc.errors(hold[:n], gt[:n])["arm"]))
        print(f"{i + 1}/{len(demos)} demos, {np.median(call_s):.2f} s per call", flush=True)

    c = model.model.config
    result = {
        "ckpt": a.ckpt, "checkpoint": int(model.export.name.removeprefix("step-")), "checkpoint_dir": str(model.export),
        "weight_profile": model.weight_profile, "stride": a.stride, "horizon_scored": int(len(gt)),
        "settings": {"sampler": c.sampler, "num_inference_steps": c.num_inference_steps, "sampler_shift": c.sampler_shift,
                     "guidance_scale": c.guidance_scale, "guidance_scale_action": c.guidance_scale_action,
                     "actions_per_call": model.actions_per_call, "dtype": c.torch_dtype},
        "seconds_per_call": {"median": float(np.median(call_s)), "mean": float(np.mean(call_s))},
        "gpu_peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2), "sets": {},
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
    out = hc.OUT / f"{a.ckpt}_{result['checkpoint']}{suffix}.json"
    out.write_text(json.dumps(result, indent=1))
    print(json.dumps({k: {kk: round(vv, 4) for kk, vv in v.items() if kk != "per_task"}
                      for k, v in result["sets"].items()}))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
