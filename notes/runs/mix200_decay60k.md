# Run card: `mix200_decay60k`

**Question.** Why did `mix200_150k` jump from 16% (checkpoint-60000) to 63% (checkpoint-150000)
on randomized scenes of the 10 tasks? Two candidate reasons:
1. **Learning rate.** It needed the low-learning-rate end of training. At 60k its
   learning rate was still 7.0e-5.
2. **Practice.** It needed more passes over the data: 1.39 passes at 60k, 3.48 at 150k.
   Passes = steps × 64 ÷ 2,759,740 frames.

**Experiment.** Take checkpoint-60000's weights, train 10,000 more steps on the same data
while the learning rate falls from 7.0e-5 to 0, and evaluate on the same scenes.
- This adds little practice: 10,000 × 64 = 0.64M samples = 0.23 passes, so 1.62 in total.
- If success rises near the 150k model, the learning rate was the reason; if it stays
  near 16%, practice was.

| | |
|---|---|
| Started | 2026-11-01 09:44 EST |
| From | `mix200_150k` checkpoint-60000, **weights only** |
| GPUs | 4, 5 (H100), 2-way data parallel, global batch 64 (32 per GPU) |
| Checkpoints | `RoboTwin/XPolicyLab/policy/GR00T_N17/checkpoints/RoboTwin-mix200_decay60k-aloha_agilex-joint-0/RoboTwin-mix200_decay60k-aloha_agilex-joint-0/checkpoint-{5000,10000}` |
| Log | `logs/train_mix200_decay60k_seed0.log` |

## Command

From the repo root, at commit `16501e5`:

```
FROM=<abs path>/RoboTwin/XPolicyLab/policy/GR00T_N17/checkpoints/RoboTwin-mix200_150k-aloha_agilex-joint-0/RoboTwin-mix200_150k-aloha_agilex-joint-0/checkpoint-60000 \
RUN=mix200_decay60k GPUS=4,5 LR=7.0e-5 MAX_STEPS=10000 SAVE_STEPS=5000 setsid -f nohup bash gr00t/train_decay.sh
```

## Settings

Identical to `mix200_150k` (`notes/runs/mix200_150k.md`) except:
- **Weights only.** DeepSpeed refuses to load ZeRO optimizer state saved on 4 GPUs onto 2
  ("Automatic adjustment of ZeRO's optimizer state partitioning with a new world size
  is not currently supported"), so the AdamW state restarts.
- **Learning-rate schedule** (GR00T's own: `--warmup_ratio 0.05`, cosine):
  - warm-up from 0 to 7.0e-5 over the first 500 steps, which absorbs the optimizer
    restart;
  - then a cosine decay to 0 at step 10,000.
  - 7.0e-5 is `mix200_150k`'s learning rate at step 60,000 (7.0086e-05 in its
    `trainer_state.json`).
- **2 GPUs** instead of 4: global batch still 64.
- **Checkpoints:** saved at 5,000 and 10,000 steps.

## Code and data

| | |
|---|---|
| Repo | `16501e5` |
| XPolicyLab, branch `local/aloha-gr00t` | `ab4f0d9` |
| RoboTwin | `7d1c8bc` (eval logging patch) |
| Dataset | `data/lerobot/RoboTwin-mix200-aloha_agilex-joint`; fingerprints rechecked unchanged (`info.json` 2848be48…, `episodes.jsonl` 725e4f80…, `tasks.jsonl` ae6ec590…, `stats.json` 818f2b52…, `modality.json` d7ebf154…) |
| Base model | `nvidia/GR00T-N1.7-3B` + `nvidia/Cosmos-Reason2-2B`, via the checkpoint |

## Dry run (100 steps, GPUs 4 and 5, deleted afterwards)

- **Weights:** loaded with no mismatch error.
- **Loss:** 0.0097 → 0.0071. The main run was at 0.0087 around step 60k, so training
  continued smoothly from the checkpoint, with no jump from the optimizer restart.
- **Learning rate:** ramped to 7.0e-5, then decayed to ~0.
- **Checkpoint:** checkpoint-100 saved (34 GB).

## Evaluation (armed to start when training exits)

- **Simulator** (`gr00t/eval_after.sh`, 4 single-stream workers on each of GPUs 4 and 5):
  checkpoint-10000 on the 10 tasks, clean + randomized, 10 episodes each. These are the
  first 10 of the old model's seeds, so they're the same scenes as all earlier evals.
  Queue `logs/queue_decay60k_10000.txt` (32 jobs, 200 episodes).
- **Prediction-error check** (`gr00t/heldout_check.py`, GPU 4): checkpoints 10000 and
  5000. Log `logs/heldout_decay60k.log`.
- **Compare against:** `mix200_150k` at 60k and at 150k, restricted to the same first 10
  seeds per task.

## Results

**Training:** finished 2026-11-01 11:27 EST, 10,000 steps in 1 h 41 min at 1.66 steps/s on
average.

**Simulator** (finished 15:25 EST). 10 tasks, first 10 old seeds each; every model is
scored on exactly the same seeds (`gr00t/paired_compare.py`):

| model | randomized | clean |
|---|---|---|
| 50task_60k (clean only, 60k, settled) | 17/98 (17%) | 63/98 (64%) |
| mix200_150k at 60k (not settled) | 14/99 (14%) | 22/98 (22%) |
| **mix200_decay60k (60k + 10k settling)** | **46/100 (46%)** | **50/100 (50%)** |
| mix200_150k at 150k | 66/99 (67%) | 62/99 (63%) |

**Prediction error** (`gr00t/heldout_check.py`; radians; trained-on randomized / unseen
randomized / clean):
- after 5k of the 10k settling steps: 0.0250 / 0.0228 / 0.0221
- after all 10k: 0.0218 / 0.0192 / 0.0192
- for reference, mix200_150k at 60k: 0.0295 / 0.0263 / 0.0266; at 150k: 0.0176 / 0.0156 / 0.0159

**Conclusions:**
- **Settling explains most of the 60k → 150k jump:** 60% of the randomized gap
  ((46−14)/(67−14)) and 68% of the clean gap ((50−22)/(63−22)). The remaining ~35% is
  more practice, or longer settling; this run can't separate the two.
- **At roughly equal training cost** (70k vs 60k steps), the new data wins on randomized
  (46% vs 17%) but loses on clean (50% vs 64%; luck range ±10 points at 100 episodes).
  The likely cause is less practice per clean demo: 1.6 vs 7 passes. With 150k steps,
  clean catches up (63% vs 64%).
- **For future runs:** judge only settled checkpoints. More varied data needs
  proportionally more training to keep clean-scene skill.
