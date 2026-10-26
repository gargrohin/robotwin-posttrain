"""Load one real RoboTwin task the way its eval does, reset it, grab camera images.

Run from RoboTwin/ (its code uses relative paths), on GPU 6:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 \
    .venv/bin/python ../setup/robotwin_task_render_check.py beat_block_hammer [seed] [shader]

Mirrors scripts/eval_policy_xpolicylab.py: demo_clean config, aloha-agilex,
setup_demo(..., is_test=True), then get_obs(). No planning, no policy.
By default it keeps RoboTwin's renderer settings (ray tracing "rt" shader,
set in envs/_base_task.py). Passing a shader (e.g. "default") overrides that
for this process only, without editing RoboTwin.
Saves outputs/render_check/<task>_<shader>_<camera>.png.

--skip-curobo: cuRobo's planner warmup crashes on these H100s ("CUDA error:
an illegal instruction", see curobo_warmup_check.py). Reset + get_obs only
need the planner's plan_grippers (plain numpy), so this flag swaps in a stub
planner for this process to test rendering on its own. Planning (expert
check, play_once, data collection) still needs a working cuRobo.
"""

import importlib
import os
import sys
import time

sys.path.append("./")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../outputs/render_check")
skip_curobo = "--skip-curobo" in sys.argv
argv = [a for a in sys.argv if a != "--skip-curobo"]
task_name = argv[1] if len(argv) > 1 else "beat_block_hammer"
seed = int(argv[2]) if len(argv) > 2 else 100000  # eval's first seed: 100000 * (1 + 0)
shader = argv[3] if len(argv) > 3 else "rt"

t = time.time()
import numpy as np
import sapien
import yaml
from PIL import Image

from envs._GLOBAL_CONFIGS import CONFIGS_PATH
from envs.utils import UnStableError

env_class = getattr(importlib.import_module(f"envs.{task_name}"), task_name)
print(f"imports: {time.time() - t:.1f}s, curobo from {os.path.dirname(sys.modules['curobo'].__file__)}", flush=True)

if shader != "rt":  # _base_task.py hard-codes "rt"; redirect that call for this run only
    _set_shader = sapien.render.set_camera_shader_dir
    sapien.render.set_camera_shader_dir = lambda _: _set_shader(shader)

if skip_curobo:
    from envs.robot.robot import Robot

    class GripperOnlyPlanner:  # same plan_grippers as envs/robot/planner.py
        def plan_grippers(self, now_val, target_val):
            vals = np.linspace(now_val, target_val, 200)
            return {"num_step": 200, "per_step": (target_val - now_val) / 200, "result": vals}

    def set_planner_without_curobo(self, scene=None):
        self.communication_flag = False
        self.left_planner = self.right_planner = GripperOnlyPlanner()

    Robot.set_planner = set_planner_without_curobo


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# Same args the eval script builds (load_task_args) for a single-embodiment robot.
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
    eval_mode=True,
)

env = env_class()
for s in range(seed, seed + 5):  # some seeds spawn unstable objects; eval skips those too
    t = time.time()
    try:
        env.setup_demo(now_ep_num=0, seed=s, is_test=True, **args)
        break
    except UnStableError as e:
        print(f"seed {s} unstable ({e}), trying next", flush=True)
        env.close_env()
print(f"env setup (seed {s}): {time.time() - t:.1f}s", flush=True)
dev = env.scene.render_system.device
print(f"render device: {dev.name} pci={dev.pci_string}, shader={sapien.render.get_camera_shader_dir()}, "
      f"skip_curobo={skip_curobo}", flush=True)

for i in range(2):  # first call includes shader compile / denoiser init
    t = time.time()
    obs = env.get_obs()
    print(f"get_obs {i}: {time.time() - t:.2f}s", flush=True)

os.makedirs(OUT_DIR, exist_ok=True)
for cam_name, data in obs["observation"].items():
    rgb = data["rgb"]  # uint8 [H, W, 3]
    png = os.path.abspath(os.path.join(OUT_DIR, f"{task_name}_{shader}_{cam_name}.png"))
    Image.fromarray(rgb).save(png)
    print(f"{cam_name}: shape={rgb.shape} mean={rgb.mean():.1f} std={rgb.std():.1f} -> {png}", flush=True)
os._exit(0)  # skip interpreter teardown; we only care about the frames
