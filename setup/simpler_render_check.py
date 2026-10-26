"""Render one SimplerEnv observation (SAPIEN 2) and save it.

Runs in SimplerEnv's venv. On the H100s this hangs in camera.take_picture() during
env reset; run it under `timeout -s ABRT <secs> python -X faulthandler ...` to get
a stack trace instead of waiting forever.

usage: simpler_render_check.py <out.png>
"""

import os
import sys
import time

T0 = time.time()
import simpler_env  # noqa: E402  (import is slow on EFS; time it)
from PIL import Image  # noqa: E402
from simpler_env.utils.env.observation_utils import (  # noqa: E402
    get_image_from_maniskill2_obs_dict,
)

print(f"imported {time.time() - T0:.0f}s", flush=True)
env = simpler_env.make("google_robot_pick_coke_can")
print(f"made env {time.time() - T0:.0f}s", flush=True)
obs, _ = env.reset(seed=0)
print(f"reset {time.time() - T0:.0f}s", flush=True)
img = get_image_from_maniskill2_obs_dict(env, obs)
Image.fromarray(img).save(sys.argv[1])
print("SimplerEnv obs image", img.shape, "mean", img.mean().round(1), flush=True)
os._exit(0)
