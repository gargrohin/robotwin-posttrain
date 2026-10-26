"""Minimal SAPIEN 2 offscreen render (red box on a ground plane).

Runs in SimplerEnv's venv (sapien 2.2.2). On the H100s this initializes Vulkan
and picks the right GPU, but the frame comes back empty (segmentation all 0).
"""

import os

import sapien.core as sapien

engine = sapien.Engine()
renderer = sapien.SapienRenderer(offscreen_only=True)
engine.set_renderer(renderer)
scene = engine.create_scene()
scene.add_ground(0)
scene.set_ambient_light([0.6, 0.6, 0.6])
scene.add_directional_light([1, 0, -1], [1, 1, 1])
b = scene.create_actor_builder()
b.add_box_visual(half_size=[0.2, 0.2, 0.2], color=[1, 0, 0])
box = b.build_kinematic()
box.set_pose(sapien.Pose([0, 0, 0.5]))
cam = scene.add_camera("c", 640, 480, 1.0, 0.01, 10)
cam.set_pose(sapien.Pose([-1.5, 0, 0.5]))

scene.step()
scene.update_render()
cam.take_picture()
img = cam.get_float_texture("Color")
seg = cam.get_uint32_texture("Segmentation")
red = int(((img[..., 0] > 0.3) & (img[..., 1] < 0.15)).sum())
print(f"SAPIEN2 render: red box px={red}, max segmentation id={seg.max()}", flush=True)
os._exit(0)
