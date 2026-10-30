# Our edits to upstream code

Every line we changed in upstream code, one `git format-patch` file per commit. The
commits live on local branches of the nested upstream repos, which aren't part of
this repo.

| Directory | Upstream repo, base | Local branch |
|---|---|---|
| `xpolicylab/` | XPolicyLab (`RoboTwin/XPolicyLab`), commit `7074afb` | `local/aloha-gr00t` |
| `curobo/` | cuRobo (`RoboTwin/envs/curobo`), tag `v0.7.8` | `h100-fix` |
| `robotwin/` | RoboTwin (`RoboTwin`), commit `ea8b211` | `local/aloha-gr00t` |

- `xpolicylab/0001`: let the `aloha_agilex` robot reuse GR00T's `arx_x5` data
  templates (same 14-dim layout).
- `xpolicylab/0002`–`0008`: faster GR00T data loading. They don't change the training
  samples (verified bit-identical with seeded RNGs). See `AGENTS.md`, "Upstream
  edits", for what each one does.
- `xpolicylab/0009`: `finetune.sh` takes `SAVE_TOTAL_LIMIT` from the environment
  (default 5), so a run can keep every checkpoint.
- `xpolicylab/0010`: `convert_v3_to_v2.py --workers N` (default 1, which is the original
  behavior) writes the per-episode files in parallel. Output is byte-identical; 2,500
  episodes take 2.5 min at 32 workers instead of 22 min.
- `robotwin/0001`: the eval client writes `episode<N>.json` next to each rollout video
  (seed, instruction, outcome, steps, step limit, sim error, checkpoint number), which
  the client log doesn't record. Read by `analysis/build_index.py`.
- `curobo/0001`: fix the illegal-instruction crash of cuRobo's L-BFGS kernel on H100.
  Write-up: [`notes/curobo_h100_lbfgs_bug.md`](../notes/curobo_h100_lbfgs_bug.md).

Apply to fresh clones with `git -C RoboTwin/XPolicyLab am <repo>/patches/xpolicylab/*.patch`
`git -C RoboTwin/envs/curobo am <repo>/patches/curobo/*.patch` and
`git -C RoboTwin am <repo>/patches/robotwin/*.patch`.
