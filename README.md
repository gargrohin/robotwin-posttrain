# robotwin-posttrain

Post-train VLAs (GR00T N1.7 first) on RoboTwin 2.0, benchmark, then hill-climb.
Current status and results: [LOG.md](LOG.md) (plain-language work log). Technical
notes: [AGENTS.md](AGENTS.md).

## What's ours and what's upstream

This repo holds only our own code. Everything else is upstream code, used as published:

| Code | Source | Where |
|---|---|---|
| Benchmark, simulator, eval client (`scripts/eval_policy_xpolicylab.py`) | [RoboTwin 2.0](https://github.com/RoboTwin-Platform/RoboTwin), unmodified | `RoboTwin/` (cloned by `setup/install_robotwin.sh`, not in this repo) |
| Policy adapters and data converters | [XPolicyLab](https://github.com/XPolicyLab/XPolicyLab) | `RoboTwin/XPolicyLab/` |
| GR00T N1.7 model, training loop, data loading | NVIDIA Isaac GR00T, vendored in XPolicyLab | `RoboTwin/XPolicyLab/policy/GR00T_N17/gr00t_n17/` |
| Motion planner | [cuRobo](https://github.com/NVlabs/curobo) v0.7.8 | `RoboTwin/envs/curobo/` |
| **Our scripts**: data prep, train/eval launchers, job queues, checks | **ours** | [`gr00t/`](gr00t/), [`setup/`](setup/) |
| **Our edits to upstream code**: aloha_agilex robot alias, faster GR00T data loading, cuRobo H100 fix | **ours** | [`patches/`](patches/) (one file per commit) |

Training uses GR00T's own fine-tuning code; `gr00t/train.sh` only sets the knobs
(data, steps, batch size, GPUs). Evaluation uses RoboTwin's official eval script
unchanged, so scores are comparable to the leaderboard.

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
