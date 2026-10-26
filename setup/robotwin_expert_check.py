"""Run RoboTwin's scripted expert on a few seeds and compare with upstream.

This is the seed-search phase of scripts/collect_data.py (setup_demo +
play_once + check_success, nothing saved), so it exercises cuRobo planning end
to end. The downloaded demo_clean data includes seed.txt, the seeds where
upstream's expert succeeded; a correctly working planner should succeed on
(nearly) the same seeds.

Run from RoboTwin/, on GPU 6:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 \
    .venv/bin/python ../setup/robotwin_expert_check.py beat_block_hammer 0 12
"""

import importlib
import os
import sys
import time

sys.path.append("./")
task_name = sys.argv[1] if len(sys.argv) > 1 else "beat_block_hammer"
first_seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
num_seeds = int(sys.argv[3]) if len(sys.argv) > 3 else 12

t = time.time()
import curobo
import yaml

from envs._GLOBAL_CONFIGS import CONFIGS_PATH
from envs.utils import UnStableError

env_class = getattr(importlib.import_module(f"envs.{task_name}"), task_name)
print(f"imports: {time.time() - t:.1f}s, curobo from {os.path.dirname(curobo.__file__)}", flush=True)


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# Same args collect_data.py builds for demo_clean + aloha-agilex, seed-search phase.
args = load_yaml(os.path.join(CONFIGS_PATH, "demo_clean.yml"))
robot_file = load_yaml(os.path.join(CONFIGS_PATH, "_embodiment_config.yml"))[args["embodiment"][0]]["file_path"]
robot_cfg = load_yaml(os.path.join(robot_file, "config.yml"))
args.update(
    task_name=task_name,
    task_config="demo_clean",
    left_robot_file=robot_file,
    right_robot_file=robot_file,
    dual_arm_embodied=True,
    left_embodiment_config=robot_cfg,
    right_embodiment_config=robot_cfg,
    need_plan=True,
)

seed_file = f"./data/demo_clean/{task_name}/aloha_agilex/seed.txt"
upstream = set(map(int, open(seed_file).read().split())) if os.path.exists(seed_file) else None

env = env_class()
results = []
for seed in range(first_seed, first_seed + num_seeds):
    t = time.time()
    note = ""
    try:
        env.setup_demo(now_ep_num=0, seed=seed, **args)
        env.play_once()
        ok = bool(env.plan_success and env.check_success() and env.planned_joints_legal()[0])
    except UnStableError:
        ok, note = False, " (unstable objects)"
    except Exception as e:
        ok, note = False, f" ({type(e).__name__}: {e})"
    env.close_env()
    up = None if upstream is None else seed in upstream
    results.append((ok, up))
    print(f"seed {seed}: success={ok} upstream_success={up} ({time.time() - t:.1f}s){note}", flush=True)

agree = sum(ok == up for ok, up in results)
print(f"RESULT {sum(ok for ok, _ in results)}/{len(results)} succeeded; agrees with upstream on {agree}/{len(results)} seeds",
      flush=True)
os._exit(0)  # skip interpreter teardown (cuRobo / SAPIEN cleanup)
