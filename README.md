# robotwin-posttrain

Post-train VLAs (GR00T N1.7 first) on RoboTwin 2.0, benchmark, then hill-climb.

## Setup (`setup/`)

| Script | What it does |
|---|---|
| `install_robotwin.sh` | Clones RoboTwin + XPolicyLab, builds `RoboTwin/.venv`, downloads assets |
| `install_gr00t.sh` | Builds the GR00T N1.7 policy env in XPolicyLab (works around an aarch64 wheel bug) |
| `sapien2_render_check.py` | Minimal SAPIEN 2 render (SimplerEnv venv) |
| `simpler_render_check.py` | One SimplerEnv frame (SimplerEnv venv) |

Note: the current `RoboTwin/.venv` was built from public PyPI while the company pip
mirror token was expired; versions are pinned, so the packages are the same.

## Results so far (2026-10-23, H100)

- RoboTwin env: torch 2.4.1+cu121, sapien 3.0.0b1, mplib (patched), cuRobo,
  pytorch3d all import.
- GR00T env: torch 2.7.1+cu128, flash-attn 2.7.4, gr00t, XPolicyLab import.
- SAPIEN 2 (SimplerEnv) does not render on these GPUs: the minimal scene comes back
  empty and SimplerEnv hangs on its first camera frame. SAPIEN 3 (RoboTwin) is
  not tested yet.
