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
- The RoboTwin eval adapter for Cosmos is still to write.
