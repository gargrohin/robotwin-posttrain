# Run card: `mix200_150k_rewarm50k`

**Question.** Does the best model get better with more training after it has settled?
In the decay experiment, extra practice (more passes over the data) accounted for ~35% of
the 60k → 150k gain, so more passes might still help. The user expects it won't, and the
flat 120k → 150k step (62% → 63% randomized, 10 tasks) agrees. This run is cheap, so it
runs on free GPUs, but the dataset build for `mix500` takes priority.

**How.** The model at 150k has its learning rate at ~0. To train further:
1. raise the learning rate again (5% warm-up to 5e-5, half the original peak, so the model
   doesn't drift far);
2. let it settle back to 0 along a cosine curve over 50k steps.

Expect it to get slightly worse during the high-rate stretch and to recover as the rate
falls. The peak of 5e-5 is a judgment call.

| | |
|---|---|
| From | `mix200_150k` checkpoint-150000, **weights only** (optimizer state restarts) |
| Data | `RoboTwin-mix200-aloha_agilex-joint`, unchanged (fingerprints rechecked 2026-11-01) |
| Steps | 50,000; learning rate 5e-5 peak, 5% warm-up (2,500 steps), cosine to 0 |
| Practice | +50,000 × 64 ÷ 2,759,740 = +1.16 passes, 3.48 → 4.64 |
| GPUs | 5, 7 (2-way), global batch 64. GPUs 0 and 4 got another account's 56 GB jobs minutes before launch, so a first dry run on 0, 4, 5, 7 was stopped and moved. |
| Checkpoints | every 10,000 steps |
| Log | `logs/train_mix200_150k_rewarm50k_seed0.log`; facts in the run folder's `run_info.json` |

**Evaluation plan.**
1. The prediction-error check (`gr00t/heldout_check.py`) on its checkpoints, to show the
   dip and the recovery.
2. The simulator only if the final checkpoint's error beats `mix200_150k` checkpoint-150000
   (0.0156 on unseen randomized demos).
