"""cuRobo-only check: build and warm up the motion planner exactly like
RoboTwin's CuroboPlanner (envs/robot/planner.py), with no SAPIEN in the process.

Run from RoboTwin/, on GPU 6:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 \
    .venv/bin/python ../setup/curobo_warmup_check.py
"""

import os
import sys
import time

sys.path.append("./")
t = time.time()
import torch
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig

print(f"imports: {time.time() - t:.1f}s, torch {torch.__version__}, {torch.cuda.get_device_name(0)}", flush=True)

yml = os.path.abspath("./assets/embodiments/aloha-agilex/curobo_left.yml")  # must be absolute
# Same table obstacle as planner.py, with the robot origin at 0 (only the warmup matters here).
world = {"cuboid": {"table": {"dims": [0.7, 2, 0.04], "pose": [0, 0, 0.74, 1, 0, 0, 0]}}}

t = time.time()
cfg = MotionGenConfig.load_from_robot_config(yml, world, interpolation_dt=1 / 250, num_trajopt_seeds=1)
motion_gen = MotionGen(cfg)
print(f"MotionGen built: {time.time() - t:.1f}s", flush=True)

t = time.time()
motion_gen.warmup()
torch.cuda.synchronize()
print(f"RESULT warmup OK: {time.time() - t:.1f}s", flush=True)
