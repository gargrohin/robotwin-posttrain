"""Minimal SAPIEN 3 offscreen render: red box on a ground plane, one camera.

Run from RoboTwin/ with its venv, on GPU 6:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 \
    .venv/bin/python ../setup/sapien3_render_check.py default   # rasterizer
  ... sapien3_render_check.py rt                                # ray tracing + OIDN (RoboTwin's setting)

Checks that the frame really contains the box (red pixels and segmentation
ids), not just that take_picture() returns. Saves
outputs/render_check/sapien3_<shader>.png.
"""

import os
import sys
import time

import numpy as np
from PIL import Image

OUT_DIR = os.path.join(os.path.dirname(__file__), "../outputs/render_check")
shader = sys.argv[1] if len(sys.argv) > 1 else "default"

t = time.time()
import sapien

print(f"import sapien {sapien.__version__}: {time.time() - t:.1f}s", flush=True)

sapien.render.set_camera_shader_dir(shader)
if shader == "rt":  # same settings as RoboTwin's envs/_base_task.py
    sapien.render.set_ray_tracing_samples_per_pixel(32)
    sapien.render.set_ray_tracing_path_depth(8)
    sapien.render.set_ray_tracing_denoiser("oidn")

# Same construction as RoboTwin: Engine + SapienRenderer + create_scene.
t = time.time()
engine = sapien.Engine()
renderer = sapien.SapienRenderer()
engine.set_renderer(renderer)
scene = engine.create_scene(sapien.SceneConfig())
dev = scene.render_system.device
print(f"render device: {dev.name} pci={dev.pci_string}", flush=True)

scene.add_ground(0)
scene.set_ambient_light([0.5, 0.5, 0.5])
scene.add_directional_light([0, 0.5, -1], [0.5, 0.5, 0.5], shadow=True)
scene.add_point_light([1, 0, 1.8], [1, 1, 1], shadow=True)

builder = scene.create_actor_builder()
builder.add_box_visual(half_size=[0.2, 0.2, 0.2], material=[1, 0, 0])
box = builder.build_kinematic(name="red_box")
box.set_pose(sapien.Pose([0, 0, 0.2]))

# SAPIEN camera convention: x forward, y left, z up. Identity rotation looks along +x.
cam = scene.add_camera("cam", 320, 240, np.deg2rad(60), 0.01, 10)
cam.entity.set_pose(sapien.Pose([-1.2, 0, 0.3]))
print(f"scene setup: {time.time() - t:.1f}s", flush=True)

for i in range(2):  # first frame includes shader compile / OIDN init
    t = time.time()
    scene.step()
    scene.update_render()
    cam.take_picture()
    rgba = cam.get_picture("Color")  # float32 [H, W, 4] in [0, 1]
    print(f"frame {i}: take_picture+get_picture {time.time() - t:.2f}s", flush=True)

seg = cam.get_picture("Segmentation")  # uint32 [H, W, 4]; channel 1 = actor (entity) id
box_px = int((seg[..., 1] == box.per_scene_id).sum())
red_px = int(((rgba[..., 0] > 0.3) & (rgba[..., 1] < 0.15) & (rgba[..., 2] < 0.15)).sum())
print(
    f"RESULT shader={shader}: box seg px={box_px}, red px={red_px}, "
    f"seg ids={np.unique(seg[..., 1]).tolist()}, mean rgb={rgba[..., :3].mean():.3f}",
    flush=True,
)

os.makedirs(OUT_DIR, exist_ok=True)
png = os.path.abspath(os.path.join(OUT_DIR, f"sapien3_{shader}.png"))
Image.fromarray((rgba[..., :3].clip(0, 1) * 255).astype(np.uint8)).save(png)
print(f"saved {png}", flush=True)
os._exit(0)  # skip interpreter teardown; we only care about the frame
