# Our edits to upstream code

Every line we changed in upstream code, one `git format-patch` file per commit. The
commits live on local branches of the nested upstream repos, which aren't part of
this repo.

| Directory | Upstream repo, base | Local branch |
|---|---|---|
| `xpolicylab/` | XPolicyLab (`RoboTwin/XPolicyLab`), commit `7074afb` | `local/aloha-gr00t` |
| `curobo/` | cuRobo (`RoboTwin/envs/curobo`), tag `v0.7.8` | `h100-fix` |
| `robotwin/` | RoboTwin (`RoboTwin`), commit `ea8b211` | `local/aloha-gr00t` |
| `cosmos-framework/` | [NVIDIA cosmos-framework](https://github.com/NVIDIA/cosmos-framework), commit `55f0554` | `robotwin` |
| `flux-action/` | [BFL flux-action](https://github.com/black-forest-labs/flux-action), commit `89ad4b6` | `robotwin` |

- `xpolicylab/0001`: let the `aloha_agilex` robot reuse GR00T's `arx_x5` data
  templates (same 14-dim layout).
- `xpolicylab/0002`–`0008`: faster GR00T data loading. They don't change the training
  samples (verified bit-identical with seeded RNGs). See `AGENTS.md`, "Upstream
  edits", for what each one does.
- `xpolicylab/0009`: `finetune.sh` takes `SAVE_TOTAL_LIMIT` from the environment
  (default 5), so a run can keep every checkpoint.
- `xpolicylab/0011`: `policy/Cosmos3_Nano/`, an XPolicyLab adapter that serves a
  cosmos-framework RoboTwin action policy (Cosmos3-Nano) to the RoboTwin eval client.
- `xpolicylab/0012`: `policy/Flux3_Action/`, an XPolicyLab adapter that serves a FLUX 3 Action
  RoboTwin policy (BFL flux-action export) to the RoboTwin eval client.
- `xpolicylab/0010`: `convert_v3_to_v2.py --workers N` (default 1, which is the original
  behavior) writes the per-episode files in parallel. Output is byte-identical; 2,500
  episodes take 2.5 min at 32 workers instead of 22 min.
- `robotwin/0001`: the eval client writes `episode<N>.json` next to each rollout video
  (seed, instruction, outcome, steps, step limit, sim error, checkpoint number), which
  the client log doesn't record. Read by `analysis/build_index.py`.
- `curobo/0001`: fix the illegal-instruction crash of cuRobo's L-BFGS kernel on H100.
  Write-up: [`notes/curobo_h100_lbfgs_bug.md`](../notes/curobo_h100_lbfgs_bug.md).
- `cosmos-framework/0001`–`0007`: Cosmos3-Nano action-policy training on RoboTwin:
  - the `aloha_agilex` dataset and SFT wrapper;
  - the `action_policy_robotwin_nano` experiment (NVIDIA's LIBERO recipe) with its run TOML and
    launcher;
  - mix200 joint-position normalizer stats, and a stats tool;
  - a CPU dataloader smoke test.
- `cosmos-framework/0008`: inference loads DCP checkpoints whose structured model config lacks
  `_auto_camera_fps`. Without it, loading crashes.
- `flux-action/0001`–`0002`: RoboTwin `aloha_agilex` training configs from BFL's DROID recipe:
  the full recipe, and the reduced schedule we ran (batch 512, 3,000 updates).
- `flux-action/0003`: DCP checkpoint planning runs on a gloo group. NCCL crashed while saving on
  2 nodes.
- `flux-action/0004`: the continuation config (3,000 → 9,000 updates). Also a `heads_warmup_start`
  schedule option, so the heads can warm up again after a resume. Its default, 0, keeps the
  recipe's schedule.

Apply to fresh clones with `git -C RoboTwin/XPolicyLab am <repo>/patches/xpolicylab/*.patch`
`git -C RoboTwin/envs/curobo am <repo>/patches/curobo/*.patch`,
`git -C RoboTwin am <repo>/patches/robotwin/*.patch`,
`git -C <cosmos-framework clone> am <repo>/patches/cosmos-framework/*.patch` and
`git -C <flux-action clone> am <repo>/patches/flux-action/*.patch`.
