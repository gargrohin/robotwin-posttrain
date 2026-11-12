# robotwin-posttrain

Post-train robot policies (GR00T N1.7, Cosmos3-Nano, FLUX 3 Action) on RoboTwin 2.0,
benchmark them, then hill-climb.

Best results so far (all 50 tasks, every model on GR00T's exact eval scenes):

| model | training data | randomized scenes | clean scenes |
|---|---|---|---|
| Cosmos3-Nano, iteration 6,000 | mix500 | **87.1%** | 88.4% |
| Cosmos3-Nano, iteration 5,500 | mix200 | 85.0% | **88.6%** |
| FLUX 3 Action, update 9,000 (continuation) | mix200 | 75.6% | 77.8% |
| GR00T N1.7, checkpoint-150000 | mix200 | 45.3% | 45.5% |

- Training data per task: mix200 = 50 clean + 200 randomized demos; mix500 = 50 clean + 495 randomized (close to
  the 50 + 500 most published RoboTwin results use). On the same Cosmos recipe, mix500 adds +2.5 points randomized
  at iteration 6,000 (by task 25 better, 10 worse); clean is unchanged.
- FLUX 3 Action with the head camera at native resolution (no upscaling) matches the original canvas at update 3,000
  (70.8–72.7% / 71.3–74.5% vs 71.0 / 71.1%) and trains faster. A Flux run on mix500 with this canvas is in progress.
- Training speedups (same model math up to rounding): Flux about 2× faster per update, Cosmos 34% less time per update.

- Current state, how things work and the rules for agents: [AGENTS.md](AGENTS.md) (start with
  "Start here").
- Work log: [LOG.md](LOG.md).
- All results: [notes/results.md](notes/results.md).
- Training runs: [notes/runs/README.md](notes/runs/README.md).

## What we added

Each model trains with its maker's own code: GR00T's fine-tuning code,
[NVIDIA cosmos-framework](https://github.com/NVIDIA/cosmos-framework) and
[BFL flux-action](https://github.com/black-forest-labs/flux-action). The benchmark is
[RoboTwin 2.0](https://github.com/RoboTwin-Platform/RoboTwin) with [XPolicyLab](https://github.com/XPolicyLab/XPolicyLab).
On top of that, this repo has:
- **Data:** converters for RoboTwin's demos into each trainer's format, including the randomized demos RoboTwin
  ships only in raw form (`gr00t/fetch_randomized.py`), and the mix200 / mix500 builds with per-episode checks.
- **Eval adapters** for Cosmos3-Nano and FLUX 3 Action in XPolicyLab, each checked to feed the model exactly what it
  saw in training (`cosmos3/check_adapter.py`, `flux3/check_adapter.py`).
- **Train and eval launchers, job queues and analysis scripts:** [`gr00t/`](gr00t/), [`cosmos3/`](cosmos3/),
  [`flux3/`](flux3/), [`analysis/`](analysis/).
- **Our edits to upstream code, one patch per commit:** [`patches/`](patches/).

## How evals are run

- All 50 tasks, robot `aloha_agilex`, in two settings: **randomized** (cluttered tables, random backgrounds, lighting
  and table height, unseen instruction wording; ~20 scenes per task, 1,015 total) and **clean** (10 scenes per task,
  499 total).
- Every model runs on **exactly the same scenes**: GR00T's eval seeds, replayed seed by seed, with the same
  instruction sentence per scene. A seed check confirms the completed set after each eval.
- Scoring uses RoboTwin's official eval script unchanged. Our only edit writes a record file per episode.
- Models are compared scene by scene with sign tests (scenes only one model solves), not just by the averages.
- Evals run on cluster GPUs, one simulator stream per GPU. The simulator's ray-traced rendering, not the policy, is
  the bottleneck: ~75–90 GPU-hours per full eval.
- Results: [notes/results.md](notes/results.md). Failure analysis: [notes/eval_analysis.md](notes/eval_analysis.md).

## Infrastructure fixes

- **cuRobo on H100:** every RoboTwin task crashed in the motion planner's L-BFGS kernel, because of a warp-shuffle
  mask bug. Fixed and validated ([notes/curobo_h100_lbfgs_bug.md](notes/curobo_h100_lbfgs_bug.md)).
- **Rendering:** only RoboTwin's ray-tracing shader works on these GPUs; the raster shaders hang.
- **Dataset conversion:** dropped video frames and silently truncated parts on a shared filesystem. Fixed, with a
  check that decodes every video and compares every episode with its source.
- **GR00T data loading:** CPU per sample went from 114 ms to 32 ms, with bit-identical samples.
- **Cluster training:** two-node save markers that survive re-admission, and checkpoint slots that work without
  delete rights.
- **FLUX 3 Action:** export of checkpoints trained with the compiled model.

## Training speedups

Profiling showed where each update's time went. The fixes keep the model's math the same, up to tiny rounding
differences from compiling:
- **FLUX 3 Action: ~2× faster per update** (52 s → 26 s on one 8-GPU node).
  - The video encoder now runs alongside the model; before, data preparation blocked it.
  - Wasted padding frames are dropped, gradients are sharded to free memory, the encoders and the model are
    compiled, and a data-loader stall every 24 updates is fixed.
  - Keeping the head camera at native resolution saves another ~27% of time per update.
- **Cosmos3-Nano: 34% less time per update** (20.0 s → 13.2 s on one node; 1.48× on a two-node run).
  - The GPU sat idle during video encoding because the CPU couldn't issue work fast enough. A captured CUDA graph of
    the encoder, overlapped with the model, fixes that.
  - The backward pass stops at the frozen text half, and spare memory keeps more activations instead of
    recomputing them.
- Details and measurements: [notes/investigations.md](notes/investigations.md).

## Setup (`setup/`)

| Script | What it does |
|---|---|
| `install_robotwin.sh` | Clones RoboTwin + XPolicyLab, builds `RoboTwin/.venv`, downloads assets |
| `install_gr00t.sh` | Builds the GR00T N1.7 policy env in XPolicyLab (works around an aarch64 wheel bug) |
| `sapien2_render_check.py` | Minimal SAPIEN 2 render (SimplerEnv venv) |
| `simpler_render_check.py` | One SimplerEnv frame (SimplerEnv venv) |

Note: the current `RoboTwin/.venv` was built from public PyPI while the company pip
mirror token was expired; versions are pinned, so the packages are the same.
