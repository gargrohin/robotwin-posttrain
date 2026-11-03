# Run card: `cosmos3_nano_mix200`

NVIDIA Cosmos3-Nano (16B "omni" world model, Mixture-of-Transformers) post-trained as a
robot action policy on the same data as GR00T `mix200_150k`, with NVIDIA's own LIBERO
policy recipe kept exactly, so the two models compare on the same data and the same
RoboTwin scenes.

| | |
|---|---|
| Started | 2026-11-02 03:24 EST (first attempt at 03:23 failed instantly: a bug in my supervisor's `export` line, fixed) |
| GPUs | 0, 4, 5, 6, 7 (H100 80 GB), FSDP across 5 |
| Measured (10-iteration test, 03:05–03:22) | peak 59.4–60.8 GB per GPU; 36.3–37.6 s per iteration; ~8 min startup; checkpoint 166 GB, ~2 min to NVMe |
| ETA | 5,000 × ~37 s ≈ 51 h + saves, so ~2026-11-04 07:00 EST |
| Output | `/mnt/efs/rohingarg/cri/cosmos3/runs/cosmos3_action_robotwin/action_sft/action_policy_robotwin_nano/` (`checkpoints/iter_<N>`, `config.yaml`) |
| Logs | trainer `/mnt/efs/rohingarg/cri/cosmos3/runs/logs/action_policy_robotwin_nano_attempt<N>.log`; supervisor `logs/cosmos3_nano_mix200.log` |
| Disk | ~166 GB per checkpoint × 11 (every 500 + final) ≈ 1.8 TB on EFS |

## Command

```
GPUS=0,4,5,6,7 setsid -f nohup bash cosmos3/run_robotwin_nano.sh > logs/cosmos3_nano_mix200.log 2>&1 < /dev/null
```

The supervisor runs cosmos-framework's `examples/launch_sft_action_policy_robotwin_nano.sh`.
If training exits early, it waits for the GPUs to be nearly empty and relaunches; the trainer
resumes from `latest_checkpoint.txt` (up to 3 attempts).

## Code, model, data

| | |
|---|---|
| cosmos-framework | github.com/NVIDIA/cosmos-framework @ 55f0554 + our branch `robotwin` @ `f8a6519` (local clone at `/mnt/efs/rohingarg/cri/cosmos3/cosmos-framework`, not pushed) |
| Environment | `/opt/dlami/nvme/rohingarg/cosmos3-venv` (uv 0.11.29 from the repo's `uv.lock`, `--all-extras --group=cu130-train`); torch 2.10.0+cu130, transformers 4.57.6 |
| Base model | `nvidia/Cosmos3-Nano`, HF snapshot `e59a53c25979a090fa8706c9acc0c254a6e89b92`, converted to DCP (`examples/checkpoints/Cosmos3-Nano`); Wan2.2 VAE (`Wan-AI/Wan2.2-TI2V-5B`/`Wan2.2_VAE.pth`) |
| Data | `data/lerobot/RoboTwin-mix200-aloha_agilex-joint` (same as GR00T `mix200_150k`): 50 tasks × (50 clean + 200 randomized), 12,500 episodes, 2,759,740 frames; index cache `/mnt/efs/rohingarg/cri/cosmos3/robotwin_cache/RoboTwin-mix200-aloha_agilex-joint.index.npz` |
| Normalizer stats | `cosmos_framework/data/generator/action/normalizer_stats/robotwin_mix200_aloha_joint_pos_stats.json` (sha256 19d7ef157f837074…), exact over all frames |

## Recipe

NVIDIA's `action_policy_libero_nano`, unchanged except for what our robot needs:

| | Value | Source |
|---|---|---|
| Training | full fine-tune (generation expert + action heads ≈ 7B of 15.2B params trainable), EMA on | NVIDIA |
| Optimizer | fused AdamW, betas (0.9, 0.99), weight decay 0.05, grad clip 1.0 | NVIDIA |
| Learning rate | 5e-5, 500 warm-up, linear decay over a 16,000-step cycle | NVIDIA |
| Length | 5,000 iterations (NVIDIA's 4-suite LIBERO length); ≈ 10.4M examples vs GR00T's 9.6M | NVIDIA |
| Batch | 32 per GPU × 5 GPUs × 13 accumulation = 2,080 (NVIDIA: 2,048 on 16 GPUs) | adapted |
| Checkpoints | every 500 iterations + final | NVIDIA |
| Actions | 14-D absolute joint positions; current state as the given first row; predict 16 | NVIDIA's DROID joint-position recipe |
| Cameras | head on top, both wrists below at half size (AgiBot layout), 256 tier | NVIDIA's AgiBot layout |
| Timing | 15 fps (RoboTwin's), 16-action chunks ≈ 1.07 s | our data |
| Normalization | quantile (q01/q99), **no clipping** (as the official code) | NVIDIA |
| Embodiment | `robotwin`, domain 17, 14-D (already registered; head slot untrained in the public checkpoint) | NVIDIA |

**Not settled at the end.** With a 16,000-step cycle, the learning rate at iteration 5,000 is
5e-5 × (1 − 4,500 ÷ 15,500) = 3.55e-5. That's NVIDIA's tested setup. A decay-to-0 variant
is our own later experiment.

## Plan

- Evaluate checkpoints 1,500 and 2,000 while training continues; stop early if they're good
  or flat.
- Eval: `cosmos3/eval.sh` (sim) and `cosmos3/heldout_check.py` (open-loop), adapter in
  XPolicyLab `policy/Cosmos3_Nano`.

## Log

- 2026-11-02 03:24 EST: started on GPUs 0, 4, 5, 6, 7, wandb off.
- 08:38 EST: iteration 500 checkpoint saved. Loss ~14.5 at the start, ~0.6–1.07 by 500–507.
  ~37 s per iteration, ~14 s of it VAE encoding of the camera frames.
- 08:43 EST: stopped at 507 and relaunched with wandb on (project `robotwin-posttrain`).
  It resumes from 500. Recipe and run folder unchanged. Earlier log:
  `logs/cosmos3_nano_mix200.part1_iter0-507_nowandb.log`.
- 08:51 EST: resumed at 501, ~35.5 s per iteration. Inference needed a fix on the cosmos-framework
  branch: commit 30bd31a (`_auto_camera_fps` on a structured config); training is still on f8a6519.
- Iteration 500: open-loop (stride 5) arm error 0.046 / 0.038 / 0.045 rad (train randomized /
  train clean / unseen randomized; holding still 0.058 / 0.066 / 0.061), close to GR00T mix200
  at 15k steps (same number of examples seen). Sim: beat_block_hammer clean 1/3.
  `outputs/heldout_check/cosmos3_nano_mix200_500_stride5.json`.
- Iteration 1,000 (saved 15:11 EST): open-loop (stride 5) 0.030 / 0.029 / 0.028 rad (ratio to holding
  still 0.51 / 0.44 / 0.46): about GR00T mix200 at 60k steps, with ~half the examples.
  `outputs/heldout_check/cosmos3_nano_mix200_1000_stride5.json`.
- 16-GPU cluster copy (`cosmos3_nano_mix200_2n`, batch 2,048, NVIDIA's 2-node layout), iteration
  500: open-loop ratio 0.72 / 0.58 / 0.72 vs this run's 0.78 / 0.59 / 0.74, the same within noise.
  `outputs/heldout_check/cosmos3_nano_mix200_2n_500_stride5.json`.
- Iteration 1,500 (local run): open-loop ratio 0.44 / 0.35 / 0.41, between GR00T mix200 60k and
  90k. `outputs/heldout_check/cosmos3_nano_mix200_1500_stride5.json`.
- **Sim, iteration 1,500, 10 tasks randomized (same scenes as GR00T): 66/99 (67%)**, equal to GR00T
  mix200 at 150k (66/99) with ~1/3 of the training examples. `python3 gr00t/paired_compare.py --per-task`.
- **16-GPU copy, sim learning curve (10 tasks randomized, same scenes as GR00T):** 2,500 77/99 (78%),
  3,000 79/99 (80%), 4,000 85/99 (86%), **5,000 86/99 (87%)**; GR00T mix200 150k 66/99 (67%).
  `python3 gr00t/paired_compare.py --per-task`.
