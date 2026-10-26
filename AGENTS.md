# robotwin-posttrain: notes for coding agents

Assignment: post-train a VLA (GR00T N1.7 first) on RoboTwin 2.0, benchmark it, then
hill-climb. **The user is doing this to learn.** Explain, guide and review; only write
implementation code or run jobs when the user asks.

## Layout
- `setup/`: our install scripts and renderer checks.
- `gr00t/prepare_data.sh`: demos → LeRobot v3.0 → v2.1 → GR00T dataset
  `data/lerobot/RoboTwin-5task-aloha_agilex-joint`.
- `gr00t/eval.sh`: evaluate a checkpoint on one task (`CKPT=<run dir name> TASK=..
  TEST_NUM=.. TASK_CONFIG=demo_clean|demo_randomized CHECKPOINT_NUM=last|<step>
  POLICY_GPU=.. ENV_GPU=..`). It starts the GR00T policy server and the RoboTwin sim
  client itself, because XPolicyLab's client does `conda activate` (our env is a uv
  venv). It also overrides `cosmos_model_path=nvidia/Cosmos-Reason2-2B`: deploy.yml's
  `shared/Cosmos-Reason2-2B` doesn't exist, and the adapter would treat it as a bogus HF
  repo id. Logs go to `logs/eval_<CKPT>_ckpt-<N>_<TASK>_<CONFIG>.{server,client}.log`.
  RoboTwin ignores `--device_id` in single-task eval, so the launcher sets
  `CUDA_VISIBLE_DEVICES` instead.
- `gr00t/train.sh`: fine-tuning launcher. Use `RUN=<name> GPUS=5,6 MAX_STEPS=..
  GLOBAL_BATCH_SIZE=..`; checkpoints go to
  `RoboTwin/XPolicyLab/policy/GR00T_N17/checkpoints/RoboTwin-<RUN>-aloha_agilex-joint-<SEED>/`
  and the log to `logs/train_<RUN>_seed<SEED>.log`.
- `RoboTwin/`: upstream clone (git-ignored). Avoid editing upstream; any edit goes
  in "Upstream edits" below.
- `data/lerobot/`: converted LeRobot datasets (git-ignored; `HF_LEROBOT_HOME` and
  `GR00T_LEROBOT_HOME` point here).
  - `data/demo_clean/<task>/aloha_agilex/{data,video,instruction}`: downloaded demos (HDF5).
  - `XPolicyLab/policy/GR00T_N17/`: GR00T adapter (`process_data.sh`, `train.sh`,
    `eval.sh`, `configs/`, `model.py`).
- `outputs/`: renderer-check images.

## Environments
- `RoboTwin/.venv` (simulator): py3.10, torch 2.4.1+cu121, sapien 3.0.0b1, mplib
  (patched), cuRobo, pytorch3d. Built from public PyPI while the mirror token was expired.
- `RoboTwin/XPolicyLab/policy/GR00T_N17/gr00t_n17/.venv` (model): py3.10, torch
  2.7.1+cu128, flash-attn 2.7.4, gr00t, XPolicyLab. **Never plain `uv run` here**: it
  re-syncs for aarch64 and fails. Use `.venv/bin/python`, `uv run --no-sync`, or
  `UV_NO_SYNC=1`.
- `.venv-lerobot` (one-time dataset conversion only): py3.10, `lerobot==0.4.4`, CPU
  torch, XPolicyLab (editable). XPolicyLab's v3.0 converter needs ≥0.4.4
  (`streaming_encoding`); GR00T's pinned commit c75455a (0.4.1) lacks it, while 0.4.4
  still has everything `convert_v3_to_v2.py` imports. It's kept out of the GR00T venv
  because lerobot bumps numpy/av/datasets past GR00T's pins.
- Install packages with `uv pip install` (user preference), on uv's default index
  (public PyPI). No mirror token needed.

## Version control
Commit every fine-tuning or bug-fix change so it can be reverted (user request).
- This repo (`main`): our scripts, notes and configs.
- `RoboTwin/XPolicyLab`: local branch `local/aloha-gr00t` holds our adapter edits (not
  pushed). Revert with `git -C RoboTwin/XPolicyLab revert <sha>`, or switch back to
  the detached upstream commit `7074afb`.
- `RoboTwin/envs/curobo`: local branch `h100-fix` (commit `13d961c`) holds the H100
  L-BFGS kernel fix. The built `.so` files are git-ignored; see Status, step 0, for the
  build and revert steps.

## Upstream edits (re-apply if XPolicyLab is updated)
- `XPolicyLab/policy/GR00T_N17/process_data.sh` (commit 8e0e7e4 on
  `local/aloha-gr00t`): in `write_modality_json` and
  `write_modality_config`, `arx_x5)` → `arx_x5|aloha_agilex)` (same 14-dim layout and
  camera keys). `resolve_src_dataset` is deliberately unchanged, so aloha must set
  `GR00T_SRC_DATASET`. Check with `git -C RoboTwin/XPolicyLab diff`.

## Decisions
- Model: GR00T N1.7 (`nvidia/GR00T-N1.7-3B`). It's the only assigned model with an
  XPolicyLab adapter.
- Robot `aloha_agilex`; 50 clean demos per task (leaderboard protocol); eval 100
  episodes per task, clean + randomized.
- Scope (2026-10-26): `beat_block_hammer`, `lift_pot`, `handover_block`,
  `stack_blocks_two`, `open_laptop`. Skipped `place_empty_cup` (RoboTwin issue #83:
  very slow rendering on H-series GPUs). Chosen for skill coverage; difficulty
  headroom not yet verified against per-task scores.

## Data pipeline for GR00T
1. Demos: in `RoboTwin/` with its venv,
   `bash scripts/download_xpolicylab_data.sh <tasks...>` → HDF5.
2. HDF5 → LeRobot v3.0: `XPolicyLab/scripts/transform_lerobot_v30_format.py
   "demo_clean.<task>.aloha_agilex" --repo_id <name>`. Reads `RoboTwin/data` and
   `RoboTwin/env_cfg`; writes to `$HF_LEROBOT_HOME/<repo_id>`. Needs `lerobot`, which is
   in neither venv yet; `process_data.sh` installs a pinned commit into the GR00T venv.
3. GR00T prep: `process_data.sh <bench> <ckpt> aloha_agilex joint` with
   `GR00T_SRC_DATASET=<repo_id>` and `GR00T_LEROBOT_HOME`. It only has templates
   for `arx_x5`; `aloha_agilex` needs its own `meta/modality.json` and
   `configs/aloha_agilex_config.py` (same 14-dim layout: per arm, 6 joints + gripper).
   `model.py` loads `configs/<env_cfg_type>_config.py` at eval time too.

## Gotchas
- pip goes through a company CodeArtifact mirror (`~/.config/pip/pip.conf`), and the token
  expires every 12h. Refresh with `aws codeartifact login --tool pip --domain <DOMAIN>
  --domain-owner <AWS_ACCOUNT_ID> --repository <REPOSITORY> --region us-west-2`. If it's
  expired, ask the user; don't bypass it.
- Shared GPUs. Check `nvidia-smi` first; use `CUDA_DEVICE_ORDER=PCI_BUS_ID`. On
  2026-10-26 GPU 6 was used for step 0 and then another user's vLLM job started on it.
  GPU 5 was freed with the user's OK (after they checked with the team) by
  stopping their idle vLLM container. The container
  was kept; bring it back with restarting that container.
- EFS is slow: cold imports take ~1 min; installs take tens of minutes.
- EFS can throw a one-off `Input/output error` (errno 5) on reads, especially right
  after files are written. Retry before assuming a file is corrupt.
- uv uses the mirror only when `UV_INDEX` is set, e.g.
  `export UV_INDEX="$(pip3 config get global.index-url)"`, so it expires with the pip token.
- Hugging Face: `HF_HOME=/mnt/nvme/.cache/huggingface` (local NVMe, fast, not on EFS)
  holds the model cache but **no token**. The working token is in
  `~/.cache/huggingface/token`, so pass it as `HF_TOKEN` (`gr00t/train.sh` does this).
  Cosmos-Reason2-2B is gated (license accepted by the user). Don't print `env`
  unfiltered: it contains token variables.
- RoboTwin's `_install.sh` keeps going after errors; grep its output for `ERROR`.
- Decode RoboTwin camera bytes only with `decode_image_bit` (no BGR↔RGB swap after it).
- SAPIEN 2 (SimplerEnv) doesn't render on these H100s. SAPIEN 3 (RoboTwin) renders
  with the `rt` shader only (see Status, step 0).
- Robots: `aloha_agilex` is RoboTwin's default and the leaderboard's robot; its URDF is
  `arx5_description_isaac.urdf` (two 6-joint arms, `fl_`/`fr_` joints). `arx_x5` is a
  separate RoboTwin robot (`X5A.urdf`) and the one RoboDojo uses, which is why the GR00T
  adapter ships `arx_x5` templates. Both are 6 joints + gripper per arm.

## Status (2026-10-26)
- [x] Envs installed; RoboTwin assets downloaded (~15 GB).
- [x] Step 0 (2026-10-26, GPU 6): **rendering works** with RoboTwin's default `rt`
  (ray-tracing) shader. First frame ~1s, then ~0.01s; 4 task cameras in ~0.2s. **Raster
  shaders (`default`, `minimal`) hang forever** on these H100s, so never switch to them.
  **cuRobo crash: fixed and applied (2026-10-26, user-approved).** Every task's
  `setup_demo` crashed with `CUDA error: an illegal instruction` in
  `lbfgs_step_cu.forward`.
  - Root cause: in `lbfgs_step_kernel.cu` (`reduce_v0`/`reduce_v1`), all 32 lanes run
    `__shfl_down_sync` with a mask covering only lanes `< elems`. That's undefined
    behaviour; nvcc 12.1 emits `WARPSYNC <reg>`, which H100 traps. It hits every
    RoboTwin trajectory solve (d_opt=168), not just warmup. Matches RoboTwin #452/#234
    and cuRobo #481.
  - Fix: shuffle with `FULL_MASK` and make lanes `≥ elems` contribute 0
    (`setup/curobo_lbfgs_warp_mask.patch`). It's on branch `h100-fix` (commit
    `13d961c`, on v0.7.8) in `RoboTwin/envs/curobo`. Rebuilt in place with
    `CUDA_HOME=/usr/local/cuda-12.1 TORCH_CUDA_ARCH_LIST=9.0 ../../.venv/bin/python
    setup.py build_ext --inplace`; the editable install already points at `src/`.
  - Validated in the real env on GPU 6: warmup OK; compute-sanitizer synccheck 0
    errors; kernel matches PyTorch L-BFGS within 2e-4 (100 cases); `beat_block_hammer`
    sets up and renders; scripted expert 52/68 seeds succeed. Logs:
    `outputs/render_check/real_*.log`.
  - **Don't rebuild with nvcc 12.9 without the patch.** It doesn't crash but returns
    wrong steps (6/88 cases, rel. error up to 0.98).
  - No-rebuild fallback: `use_cuda_kernel: False` in the 6 L-BFGS yml files under
    `envs/curobo/src/curobo/content/configs/task/` (~1.8× slower planning).
  - Revert: in `envs/curobo`, run `git switch --detach v0.7.8`, then
    `cp -p <repo>/scratch/curobo_debug/original_build/*.so src/curobo/curobolib/`
    (SHA256SUMS there), then `rm -rf build`.
  - `scratch/curobo_debug/` (1.2 GB) can be deleted except `original_build/`.
  - Data collection was not run: its default output path would overwrite the
    downloaded demos. Scripts: `setup/sapien3_render_check.py`,
  `setup/robotwin_task_render_check.py` (`--skip-curobo` stubs the planner for that run),
  `setup/curobo_warmup_check.py`; images and logs are in `outputs/render_check/`.
  Full write-up of the cuRobo bug and fix: `notes/curobo_h100_lbfgs_bug.md`.
- [x] Step 2: 50 clean demos per task downloaded and verified (250 HDF5 files, ~2 GB).
  Per episode: `state/` and `action/` hold `{left,right}_arm_joint_states` (6) and
  `{left,right}_ee_joint_states` (1); cameras are `vision/cam_{head,left_wrist,right_wrist}/colors`;
  `instructions` is inside the file. Recorded at 15 Hz (`env_cfg/aloha_agilex.yml`).
- [x] Step 3 (done 2026-10-26): `data/lerobot/RoboTwin-5task-aloha_agilex-joint`: 250
  episodes, 51,352 frames, 0 frame mismatches, no dropped frames, no read retries.
  History, via `gr00t/prepare_data.sh` (separate
  `.venv-lerobot`, approved by the user). The first full run produced a flawed dataset:
  - **Dropped video frames** in 146/249 episodes. LeRobot's streaming encoder waits
    0.1 s on a full queue (default 30 frames) and then drops the frame, leaving videos
    one frame shorter than their episodes.
  - **One missing episode** (`lift_pot/episode_0000038`): an EFS errno-5 read error,
    and the converter skips failed files.
  - Fix: `gr00t/convert_to_lerobot.py` runs the official converter with
    `encoder_queue_maxsize=1000` (> max episode length 354) and retries on `OSError`.
    `gr00t/check_dataset.py` (now step 4 of prep) decodes every video and requires
    frames == steps and 250 episodes. It was confirmed to catch the flawed dataset.
  - Re-run in progress. The flawed copy is kept as `data/lerobot/RoboTwin-smoke-…`
    for a training-pipeline smoke test only (`RUN=smoke`, 200 steps). Delete it, and
    `checkpoints/RoboTwin-smoke-*`, afterwards.
  - Dataset facts: 15 fps, 240×320 per camera, ~205 steps per episode (106–354),
    ~51k frames total; one instruction is sampled per episode (213 distinct strings).
- [x] Step 4: fine-tune (first 30k run done). Plan agreed 2026-10-26 (user away for a few hours):
  - One model on all 5 tasks; GR00T defaults (frozen VLM backbone, train the action
    head, lr 1e-4); GPUs 5+6 (user frees GPU 5; GPU 6 is shared with cuRobo debugging);
    logs local only (no wandb).
  - Sanity run first: 2k steps, global batch 32. If the loss drops and checkpoints
    save, run 30k steps at global batch 64. If not, debug and fix.
  - Cosmos-Reason2-2B license accepted by the user (HF returns 200).
  - Agent may fix routine issues on its own (logged here and committed). It may apply a
    cuRobo fix that's validated end to end, but must never kill others' processes.
  - Pipeline smoke test (`RUN=smoke`, flawed data, 200 steps, batch 32, GPUs 5+6):
    passed. Loss went 1.04 → 0.34 (grad norm 0.1–0.9). Speed was ~2.5 steps/s once warm
    (~13 samples/s overall, including startup). Each checkpoint save takes ~1.5 min
    on EFS. Startup (imports + model load) takes ~4 min. Loss is printed every 10 steps
    as `{'loss': …}` lines (split the log on `\r`). Checkpoints are in
    `checkpoints/RoboTwin-<RUN>-aloha_agilex-joint-0/RoboTwin-<RUN>-aloha_agilex-joint-0/checkpoint-<step>`.
  - **Sanity run passed** (`RUN=5task_sanity`, clean data, 2k steps, batch 32, GPUs
    5+6, 25 min): loss 1.05 → 0.026 (step 1000 ≈ 0.05), checkpoint-1000 and -2000 saved.
    Steady ~2.7 steps/s.
  - **30k run started 2026-10-26 ~02:45 UTC**: `RUN=5task_30k`, 30k steps, global batch
    64, `SAVE_STEPS=2500` (fewer slow EFS saves; the last 5 kept = 20k–30k), GPUs 5+6.
    Log: `logs/train_5task_30k_seed0.log`. Startup took ~15 min (dataloader workers
    "Caching shard…" first; that's normal, not a hang). Then 1.32 steps/s at batch 64
    using ~32 GB/GPU, so the ETA is ~6.3 h (≈09:20 UTC) plus saves. LR warmup is 5%
    (1,500 steps).
  - Sanity checkpoint-2000 eval (10 eps, `beat_block_hammer`, demo_clean, GPU 6 shared
    with the 30k run, ~45 min): **0/10**. But the videos show correct behavior: it
    reaches the hammer, grasps and lifts it, and swings it toward the block, missing
    only on precision (`outputs/eval_sanity/ep*_strip.png`). So the camera mapping,
    state/action layout, gripper and relative-action handling are right, and 2k steps
    (~1.25 epochs) is just undertrained.
  - Learning-curve evals of the 30k run's checkpoint-10000 and -20000 (same 10-ep
    test) are queued by a background watcher. Trainer keeps only the last 5, so 10k
    must be evaluated before it's deleted. The first watcher died at ~05:38 UTC
    (cleanup after a Claude session restart) and was relaunched at 05:41; training
    was unaffected.
  - Progress at 05:40 UTC: step ~10k at ~1.9 steps/s; ETA ~08:35 UTC. Mean loss per
    1k steps: 0.27 (0–1k), 0.045, 0.031, 0.028, 0.023, 0.019, 0.017, 0.016, 0.014 (8–9k).
  - **Throughput diagnosis (05:40 UTC):** GPU SM util averages 64–72% (dips during
    checkpoint saves and eval startup). The machine is 71% CPU-idle and the dataloader
    workers sit idle with shards prefetched, so data loading isn't the bottleneck.
    Each training rank's main thread is ~95% busy (`OMP_NUM_THREADS=1`, set by
    torchrun): the step loop is CPU-bound (Python and kernel-launch overhead and/or
    CPU-side ops). Batch 16→32 per GPU raised throughput 85→~120 samples/s, consistent
    with that. `py-spy` can't attach (`ptrace_scope=1`), so profile with
    `torch.profiler` in a short run. Candidates for the next run: bigger per-GPU batch;
    `OMP_NUM_THREADS` 4–8; checkpoints to local NVMe (`/mnt/nvme`, ~4 TB free;
    EFS saves take 1.5–2.5 min each); `SAVE_ONLY_MODEL=1` in `finetune.sh` (~7 GB vs
    ~35 GB per checkpoint, but no resume). During a save, rank 1 shows 100% SM util:
    that's NCCL busy-waiting, not work.
  - Decision (user, 2026-10-26 ~05:45 UTC): let the 30k run finish unchanged; apply
    the speedups to the next run and measure them.
  - **Learning curve** (`beat_block_hammer`, demo_clean, 10 eps, same seeds 100000 and
    100002–100010, `eval_checkpoints.sh`):

    | checkpoint | success | seeds solved |
    |---|---|---|
    | sanity 2k (batch 32) | 0/10 | none |
    | 30k-run 10k | 0/10 | none |
    | 30k-run 20k | **4/10** | 100000, 100004, 100006, 100010 |
    | 30k-run 30k | queued | |

    With 10 episodes the uncertainty is roughly ±30%; the full benchmark uses 100. A
    successful episode ends early: the 20k eval took ~28 min vs ~45 min for an
    all-fail one.
  - **30k run finished 09:38 UTC**: 6.8 h (1.22 steps/s overall, 78 samples/s), train
    loss 0.021 overall. Mean loss per 5k steps: 0.080, 0.016, 0.011, 0.0078, 0.0055,
    0.0043. Kept checkpoints: 20000, 22500, 25000, 27500, 30000 (~35 GB each).
  - Each checkpoint is ~35 GB (full model plus optimizer state). Trainer keeps the
    last 5 (`save_total_limit 5`), so up to ~175 GB per run on EFS.
- [ ] Step 5: benchmark. Eval pipeline smoke test passed (2026-10-26): the smoke
  checkpoint on `beat_block_hammer`, demo_clean, 2 episodes, on GPU 6 shared with
  training. Server loads, client connects, episodes run, the arm moves (frames in
  `outputs/eval_smoke/`), 0/2 success (expected for 200 steps on flawed data). Results
  go to `RoboTwin/eval_result/<task>/GR00T_N17/<config>/<ckpt>/<timestamp>/`
  (`_result.txt` plus one mp4 per episode).
  **Speed concern:** a failing episode runs the full 400 steps, ~5 min on a shared
  GPU. A full benchmark (5 tasks × 100 eps × clean/randomized) would take days on one
  GPU, so plan for RoboTwin's `--num_workers` and several GPUs, and re-measure on an
  unshared GPU.
  Parallel-eval options found in the code (not tried yet):
  - RoboTwin `scripts/eval_policy_xpolicylab.py --eval_batch true --num_workers N`
    spawns N sim workers that share a seed queue; each opens its own websocket client
    to the same policy server. `gr00t/eval.sh` currently passes `--eval_batch false`.
  - The policy server (`XPolicyLab/setup_policy_server.py`) is a single asyncio
    process, so model calls are likely serialized; the GR00T adapter has
    `update_obs_batch`/`get_action_batch` for batched inference.
  - Next: on an unshared GPU, measure per-step time split (policy vs sim+render), then
    try N workers. Parallelism across GPUs/tasks: `scripts/eval_policy_multitask.py`
    (sets `CUDA_VISIBLE_DEVICES` per job).
- [ ] Step 6: hill-climb.
