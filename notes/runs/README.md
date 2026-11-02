# Training runs: registry and lineage

Every GR00T N1.7 training run, where it started from, what it trained on, and where its
checkpoints are. Update this file whenever a run starts or finishes. Each run's checkpoint
folder also has a `run_info.json` with the same facts.

**Checkpoint paths.**
`RoboTwin/XPolicyLab/policy/GR00T_N17/checkpoints/RoboTwin-<run>-aloha_agilex-joint-0/RoboTwin-<run>-aloha_agilex-joint-0/checkpoint-<step>`
- Each checkpoint is ~34 GB.
- Only the two `model-0000*-of-00002.safetensors` files plus the configs (~7 GB) are
  needed to run the model. The rest is training state.

## Lineage

```
nvidia/GR00T-N1.7-3B (NVIDIA base model)
├── 5task_sanity        2k steps, 5 tasks × 50 clean demos
├── 5task_30k           30k steps, 5 tasks × 50 clean demos
├── 50task_60k          60k steps, 50 tasks × 50 clean demos
└── mix200_150k         150k steps, 50 tasks × (50 clean + 200 randomized demos)
    ├── checkpoint-60000  → mix200_decay60k          10k steps, learning rate 7.0e-5 → 0
    └── checkpoint-150000 → mix200_150k_rewarm50k    planned 50k steps, learning rate 5e-5 → 0; stopped at ~29.4k

nvidia/Cosmos3-Nano (16B, separate framework: NVIDIA cosmos-framework)
├── cosmos3_nano_mix200      5,000 iterations, full fine-tune on mix200, NVIDIA's LIBERO recipe, 5 local GPUs
└── cosmos3_nano_mix200_2n   same recipe from scratch on 16 cluster GPUs (NVIDIA's 2-node layout)
```

## Runs

| run | started from | data | steps, learning rate | GPUs, batch | dates | checkpoints kept | results | details |
|---|---|---|---|---|---|---|---|---|
| `5task_sanity` | base model | `5task` | 2k; 1e-4 cosine, 5% warm-up | 2 (5, 6), 32 | 2026-10-26 | 1000, 2000 | pipeline check: `beat_block_hammer` 0/10 at 2k, but correct behavior | LOG.md, AGENTS.md step 4 |
| `5task_30k` | base model | `5task` | 30k; 1e-4 cosine, 5% warm-up | 2 (5, 6), 64 | 2026-10-26 | 20000–30000 every 2500 | ckpt-30000, 5 tasks: clean 28.3%, randomized 4.7% | `notes/results.md` |
| `50task_60k` | base model | `50task` | 60k; 1e-4 cosine, 5% warm-up | 4, 64 | 2026-10-27 | 40000–60000 every 5000 | ckpt-60000, 50 tasks: clean 42.6%, randomized 9.7% | `notes/results.md` |
| `mix200_150k` | base model | `mix200` | 150k; 1e-4 cosine, 5% warm-up | 4 (3, 5, 0, 7), 64 | 2026-10-30 | 15000–150000 every 15000 | ckpt-150000, 50 tasks: clean 46.0%, randomized 45.2%; 10-task curve; no memorization | `notes/runs/mix200_150k.md` |
| `mix200_decay60k` | `mix200_150k` ckpt-60000 (weights only) | `mix200` | 10k; 7.0e-5, 5% warm-up, cosine to 0 | 2 (4, 5), 64 | 2026-11-01 | 5000, 10000 | ckpt-10000, 10 tasks (first 10 seeds): clean 50%, randomized 46% | `notes/runs/mix200_decay60k.md` |
| `mix200_150k_rewarm50k` | `mix200_150k` ckpt-150000 (weights only) | `mix200` | 50k; 5e-5, 5% warm-up, cosine to 0 | 2 (5, 7), 64 | 2026-11-01 21:00 EST; **stopped early** 2026-11-02 03:04 EST at step ~29,400 of 50,000, to free GPUs 5 and 7 for Cosmos (user OK) | 10000, 20000 | not evaluated; the planned prediction-error check was cancelled with the run | `notes/runs/mix200_150k_rewarm50k.md` |

| `cosmos3_nano_mix200` | `nvidia/Cosmos3-Nano` (Cosmos framework) | `mix200` | 5,000 iterations; 5e-5, 500 warm-up, linear decay over a 16,000 cycle (ends at 3.55e-5) | 5 (0, 4, 5, 6, 7), ~2,080 | 2026-11-02 03:24 EST, ETA ~11-04 07:00 | every 500 (+ final), ~166 GB each, at `/mnt/efs/rohingarg/cri/cosmos3/runs/...` | running | `notes/runs/cosmos3_nano_mix200.md` |
| `cosmos3_nano_mix200_2n` | `nvidia/Cosmos3-Nano` (Cosmos framework) | `mix200` | same as `cosmos3_nano_mix200` | 16 H100s on a company cluster (2 nodes × 8; model split across a node's 8 GPUs, one copy per node), 2,048 | submitted 2026-11-02 16:11 EST; ~10.1 s per iteration in the 2-node test (~14.5 h total) | every 500, kept in cloud storage | queued | `notes/runs/cosmos3_nano_mix200.md` |

## Datasets

All under `data/lerobot/`. A run's `RoboTwin-<run>-aloha_agilex-joint` is a link to the
dataset it used.

| dataset | contents | episodes |
|---|---|---|
| `RoboTwin-5task-aloha_agilex-joint` | 5 tasks × 50 clean demos | 250 |
| `RoboTwin-50task-aloha_agilex-joint` | 50 tasks × 50 clean demos | 2,500 |
| `RoboTwin-mix200-aloha_agilex-joint` | 50 tasks × (50 clean + randomized episodes 0–199) | 12,500 |
| `RoboTwin-mix500-aloha_agilex-joint` (built 2026-11-02, not yet trained on) | 50 tasks × (50 clean + randomized episodes 0–494) | 27,250 |

**Held out from all training** (for the memorization check):
- randomized episodes 200–204 of the 10 tasks (`RoboTwin/data/demo_randomized_heldout`).
  These **are** in `mix500`.
- randomized episodes 495–499 of all 50 tasks (`RoboTwin/data/demo_randomized_heldout500`).

## Continuing from any checkpoint

- **New run from a checkpoint's weights**, with its own learning-rate schedule:

  ```
  FROM=<checkpoint dir> RUN=<new name> GPUS=<ids> LR=<peak> MAX_STEPS=<n> SAVE_STEPS=<n> bash gr00t/train_decay.sh
  ```

  The optimizer state restarts, so the schedule starts with a short warm-up.
- **Resuming an interrupted run** with its full training state:
  - re-run its launcher with the same `RUN` (e.g. `gr00t/run_mix200.sh`), and GR00T
    resumes from the latest checkpoint in the run folder;
  - this only works on the same number of GPUs, because DeepSpeed can't re-split its saved
    optimizer state.
- **Evaluating a checkpoint on the same scenes as earlier evals:** make a queue with
  `gr00t/paired_queue.py`, then run `gr00t/eval_after.sh` or `gr00t/benchmark_worker.sh`
  with `MAX_SHARDS=1`. Compare with `gr00t/paired_compare.py`.
