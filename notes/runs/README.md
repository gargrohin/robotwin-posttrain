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
    └── iter 5000 → cosmos3_nano_mix200_2n_decay1k   1,000 iterations, learning rate 3.55e-5 → 0 (full state resumed)

black-forest-labs/flux-3-action-base (7B, BFL flux-action trainer)
└── flux3_mix200_b512   3,000 updates at batch 512 on mix200, the official DROID recipe scaled to 1/10 (budget)
    └── step 3000 → flux3_mix200_b512_cont6k   +6,000 updates (to 9,000) at batch 512, full state resumed, re-warm-up + cool-down
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

| `cosmos3_nano_mix200` | `nvidia/Cosmos3-Nano` (Cosmos framework) | `mix200` | 5,000 iterations; 5e-5, 500 warm-up, linear decay over a 16,000 cycle (ends at 3.55e-5) | 5 (0, 4, 5, 6, 7), ~2,080 | 2026-11-02 03:24 EST; **stopped** 2026-11-02 22:47 EST at iteration ~1,700 (user OK), after the 16-GPU copy passed it | 500, 1000, 1500, ~166 GB each, at `/mnt/efs/rohingarg/cri/cosmos3/runs/...` | open-loop 500/1000/1500 in the run card; sim eval of 1500 running | `notes/runs/cosmos3_nano_mix200.md` |
| `cosmos3_nano_mix200_2n` | `nvidia/Cosmos3-Nano` (Cosmos framework) | `mix200` | same as `cosmos3_nano_mix200` | 16 H100s on a company cluster (2 nodes × 8; model split across a node's 8 GPUs, one copy per node), 2,048 | first try 16:30 EST, evicted at iteration ~240 (no checkpoint yet); resubmitted 17:28 EST at high priority, started 17:41 EST, **finished 2026-11-03 08:29 EST** (5,000 iterations, ~10.3 s per iteration, no interruptions) | 500–5000 every 500, ~177 GB each, in cloud storage | sim (10 tasks randomized, same scenes as GR00T): 2500 78%, 3000 80%, 4000 86%, **5000 87%** (GR00T mix200 150k: 67%); **50 tasks at 5000: randomized 81.8%, clean 84.2%** (GR00T 45.2%, 45.4%) | `notes/runs/cosmos3_nano_mix200.md` |
| `cosmos3_nano_mix200_2n_decay1k` | `cosmos3_nano_mix200_2n` iter 5000 (full state) | `mix200` | 1,000 iterations, learning rate linear 3.55e-5 → 0 (second LambdaLinear cycle) | 16 H100s on a company cluster, 2,048 | 2026-11-04 09:43–12:51 EST, 10.2–10.4 s per iteration | 5250, 5500, 5750, 6000 (~165 GB each), in cloud storage | 50 tasks: 5500 randomized 85.0% / clean 88.6%, 6000 84.5% / 88.2% (Cosmos 5000: 81.9% / 84.4%; 5500 ≈ 6000); metrics logged to wandb | `notes/results.md` |
| `flux3_mix200_b512` | `black-forest-labs/flux-3-action-base` (BFL flux-action) | `mix200` | 3,000 updates at batch 512; official DROID full fine-tune recipe (lrs, optimizer, loss weights, EMA unchanged), schedule scaled to 1/10 (trunk frozen 100, warmups 200/100, cooldown 2,500→3,000) | 16 H100s on a company cluster (2 nodes), 512 = 16 GPUs × 4 windows × 8 accumulation | started 2026-11-03 16:43 EST, **finished 2026-11-04 16:15 EST** (3,000 updates, ~26.8 s per update) | every 150 updates (20 × ~129 GB), in cloud storage | step 1500: 10-task randomized 74/100 (GR00T 67, Cosmos 5000 87); **step 3000, 50 tasks, GR00T's seeds: randomized 71.0%, clean 71.1%** (Cosmos 5000 81.9% / 84.4%, GR00T 45.3% / 45.5%; `notes/results.md`); metrics logged to wandb | local notes (not in repo) |
| `flux3_mix200_b512_cont6k` | `flux3_mix200_b512` step 3000 (full state: weights, optimizer, both EMAs; BFL's resume path) | `mix200` (same index, checked byte for byte) | 6,000 more updates (3,000 → 9,000) at batch 512; official peak lrs (trunk 1.92e-4, heads 9.6e-4); re-warm-up from 0 over 200 (trunk) / 100 (heads) updates, constant, linear cool-down 8,000 → 9,000; EMAs carried over | 16 H100s (2 nodes) until update 4,375, then 8 H100s (1 node); batch 512 throughout | started 2026-11-05 06:06 EST (~27 s per update); stopped 16:50 EST at update 4,375 to free our team's GPUs for another team; a 1-node job on the shared queue was pre-empted twice before its first save; a 2-node job on another team's quota waited 17.5 h and was cancelled unstarted; resumed from the 4,375 save on 2026-11-06 11:31 EST as 1 node on another team's quota (8 GPUs × 4 windows × 16 accumulation = 512, ~53 s per update, saves every 50); stopped 2026-11-06 12:43 EST at update 4,450 (GPUs freed for another team); resumed 12:48 EST from the 4,450 save as 1 node on the shared borrowed pool (pre-emptible; restarted from the latest save whenever stopped); ETA ~10:00 EST Mon Nov 9 without pre-emptions | full training state every 125 updates (every 50 from update 4,375 on) in two slots that are overwritten in turn (at most 2 in cloud storage); weights-only exports (EMA 0.10, bf16) every 1,000 updates | update 5,000, all 50 tasks on GR00T's exact seeds (step-3000 protocol): randomized 650/1015 = 64.0%, clean 309/499 = 61.9% (step 3000 of the parent: 71.0% / 71.1%; paired by seed, a drop: sign test p < 0.001 on both); the same 50-task eval runs at 6,000, 7,000, 8,000 and 9,000 as each export lands (4,000 exported, not evaluated); metrics logged to wandb | local notes (not in repo) |

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
