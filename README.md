# robotwin-posttrain

Post-train robot policies (GR00T N1.7, Cosmos3-Nano, FLUX 3 Action) on RoboTwin 2.0,
benchmark them, then hill-climb.

Best results so far (all 50 tasks, every model on GR00T's exact eval scenes):

| model | randomized scenes | clean scenes |
|---|---|---|
| Cosmos3-Nano, iteration 5,500 | 85.0% | 88.6% |
| FLUX 3 Action, step 3,000 | 71.0% | 71.1% |
| GR00T N1.7, checkpoint-150000 | 45.3% | 45.5% |

- Current state, how things work and the rules for agents: [AGENTS.md](AGENTS.md) (start with
  "Start here").
- Work log: [LOG.md](LOG.md).
- All results: [notes/results.md](notes/results.md).
- Training runs: [notes/runs/README.md](notes/runs/README.md).

## What's ours and what's upstream

This repo holds only our own code. Everything else is upstream code, used as published:

| Code | Source | Where |
|---|---|---|
| Benchmark, simulator, eval client (`scripts/eval_policy_xpolicylab.py`) | [RoboTwin 2.0](https://github.com/RoboTwin-Platform/RoboTwin); one edit: a per-episode record file (`patches/robotwin/0001`) | `RoboTwin/` (cloned by `setup/install_robotwin.sh`, not in this repo) |
| Policy adapters and data converters | [XPolicyLab](https://github.com/XPolicyLab/XPolicyLab) | `RoboTwin/XPolicyLab/` |
| GR00T N1.7 model, training loop, data loading | NVIDIA Isaac GR00T, vendored in XPolicyLab | `RoboTwin/XPolicyLab/policy/GR00T_N17/gr00t_n17/` |
| Motion planner | [cuRobo](https://github.com/NVlabs/curobo) v0.7.8 | `RoboTwin/envs/curobo/` |
| Cosmos3-Nano model and trainer | [NVIDIA cosmos-framework](https://github.com/NVIDIA/cosmos-framework), local branch `robotwin` | outside this repo |
| FLUX 3 Action model and trainer | [BFL flux-action](https://github.com/black-forest-labs/flux-action), local branch `robotwin` | outside this repo |
| **Our scripts**: data prep, train/eval launchers, job queues, checks | **ours** | [`gr00t/`](gr00t/), [`cosmos3/`](cosmos3/), [`flux3/`](flux3/), [`setup/`](setup/) |
| **Our edits to upstream code**: aloha_agilex robot alias, faster GR00T data loading, Cosmos and Flux eval adapters, episode records, cuRobo H100 fix | **ours** | [`patches/`](patches/) (one file per commit) |

Each model trains with its maker's own code (GR00T's fine-tuning code, NVIDIA cosmos-framework,
BFL flux-action); our launchers only set the knobs (data, steps, batch size, GPUs). Evaluation uses RoboTwin's official eval script, whose
scoring is unchanged (our only edit writes a record file per episode), so scores are comparable
to the leaderboard.

## Setup (`setup/`)

| Script | What it does |
|---|---|
| `install_robotwin.sh` | Clones RoboTwin + XPolicyLab, builds `RoboTwin/.venv`, downloads assets |
| `install_gr00t.sh` | Builds the GR00T N1.7 policy env in XPolicyLab (works around an aarch64 wheel bug) |
| `sapien2_render_check.py` | Minimal SAPIEN 2 render (SimplerEnv venv) |
| `simpler_render_check.py` | One SimplerEnv frame (SimplerEnv venv) |

Note: the current `RoboTwin/.venv` was built from public PyPI while the company pip
mirror token was expired; versions are pinned, so the packages are the same.

## Environment checks (2026-10-23, H100)

- RoboTwin env: torch 2.4.1+cu121, sapien 3.0.0b1, mplib (patched), cuRobo,
  pytorch3d all import.
- GR00T env: torch 2.7.1+cu128, flash-attn 2.7.4, gr00t, XPolicyLab import.
- SAPIEN 2 (SimplerEnv) does not render on these GPUs: the minimal scene comes back
  empty and SimplerEnv hangs on its first camera frame. SAPIEN 3 (RoboTwin) is
  not tested yet.
