# Run card: `flux3_headnative_w41_b512` (side head-native run)

Identical to the main head-native run `flux3_headnative_b512` except for one setting: the action loss
weight is 41.18 instead of 50. BFL set 50 for 2,720 predicted video tokens per window; the head-native
canvas has 2,240, so 50 × 2,240 / 2,720 = 41.18 keeps the baseline's balance between the video and action
losses. Versus `flux3_headnative_b512`, the weight is the only change; versus the baseline, the canvas
and the weight both change (the weight change only restores the baseline's balance).

| | |
|---|---|
| Submitted | 2026-11-07 22:46 EST, on the cluster's shared borrowed pool (pre-emptible) |
| Started | see LOG.md |
| Hardware | 1 node, 8 H100s; batch 512 = 8 GPUs × 4 windows × 16 accumulation |
| Measured in the test (20 updates + resume, 2026-11-07 22:05–22:36; run with the main run's config, the two differ only in the weight) | 37.9–39.8 s per update (baseline layout on 1 node: 52.5–54); data wait ~0.002 s; peak allocated 62.1–62.7 GiB, device memory used 74 GiB of 80; save ~66–71 s; resume check passed |
| Expected length | 3,000 × ~38.6 s + saves and loader restarts ≈ 34 h without pre-emptions (~270 GPU-hours) |
| wandb | `flux3-f3a-headnative-w41-b512-1n` (project `robotwin-posttrain`) |

## Code, model, data

| | |
|---|---|
| Trainer | BFL flux-action, our branch `exp-small-canvas` @ `63cc40f` (from `robotwin` @ `5e8d3df`): new camera layout `droid_native_head`, two configs. Not pushed. |
| Config | `configs/robotwin/train_headnative_w41_b512_s3000.json` (flux-action) |
| Base model | `black-forest-labs/flux-3-action-base`, same files as the baseline |
| Data | `mix200`, the same flux-action index as the baseline (12,500 episodes) |
| Eval adapter | XPolicyLab branch `exp-small-canvas` @ `e0189a1` (from `local/aloha-gr00t` @ `2b435d8`): head-native exports get the simulator's 240×320 frames unchanged; own warm-up request |

## Recipe

Everything as the baseline's `train_b512_s3000.json`: BFL base, fresh 14-D heads, seed 42; 3,000 updates
at batch 512; trunk frozen for 100 updates, warm-ups 200 (trunk) / 100 (heads), linear cool-down
2,500 → 3,000; peak learning rates 1.92e-4 (trunk) / 9.6e-4 (heads); AdamW (0.9, 0.99), eps 1e-8, weight
decay 0.05; **no gradient clipping** (`max_grad_norm: null`, set explicitly; the trainer's default would
be 1.0); timesteps logit-logistic 0.75 / shift 42; action scale 2; loss video 1 / **action 41.18**; caption
dropout 0.1; augmentation on; power EMAs 0.10 and 0.05.

## Differences from the baseline

| what | baseline `flux3_mix200_b512` | this run | measured size / status |
|---|---|---|---|
| canvas | 544×736, DROID layout; latent 17×20 = 340 tokens per frame | 448×640, `droid_native_head`; latent 14×20 = 280 tokens per frame | video tokens per window 3,060 → 2,520 (predicted 2,720 → 2,240) |
| head row | head upscaled 240×320 → 360×640 (stretched 4:3 → 16:9), full width | head at native 240×320, centered; black 160-px bands left and right | the head covers 7.5×10 latent tokens instead of 11.25×20 |
| padding | reflect: right 96 px (dropped from the latent), bottom 4 rows | black side bands; reflect bottom 28 rows | the last latent row holds 4 content rows and 28 padding rows |
| head augmentation | random crop 95% + resize back + color jitter on the upscaled 360×640 frame | the same on the native 240×320 frame | inherent to the change; same crop fraction and jitter ranges |
| wrist path | data loader upscales to 360×640 (ffmpeg), augmentation, then halved to 180×320 | decoded at 240×320, augmentation, then upscaled to 360×640 (torch bilinear) and halved inside the layout | wrist slot differs from the baseline's training pixels by 1.14/255 on average (p99 4/255, max 16/255; 48 frames). The baseline's own eval inputs differed from its training inputs by the same amount (1.16/255). Augmentation now acts before the wrist upscale |
| action loss weight | 50 (set by BFL for 2,720 video tokens per window) | **41.18** | keeps the baseline's per-modality balance (32 × 41.18 / 2,240 = 32 × 50 / 2,720 = 0.588); the overall loss scale is within 2% of the baseline's. The only difference from `flux3_headnative_b512` |
| hardware | 2 nodes, 16 GPUs × 4 windows × 8 accumulation | 1 node, 8 GPUs × 4 windows × 16 accumulation | **assumed not to matter, not tested.** Same batch 512 and same math; which windows share an update differs |
| data order | seed 42, one uninterrupted job | seed 42; the shared pool can pre-empt, and each restart draws a new data order from the run seed (BFL's resume default) | statistically equivalent; restarts are counted in LOG.md |
| logging and saving | metrics every 10 updates; full state every 150 | metrics every update; full state every 50 into two slots overwritten in turn | no effect on training math |
| speedups | none | none at launch | if validated speedups are adopted later, the update where they start and their equivalence check go here |
| eval input | simulator frames resized to 360×640 at eval (torch), training frames resized by ffmpeg | frames pass unchanged; the adapter's canvas is bit-identical to the training canvas for the same frames (CPU check, 150 checks) | eval inputs closer to training |

## Checks before launch

- Unit tests of the layout (pixel placement, wrist slots equal to the DROID layout's for the same upscaled
  cameras, padding, latent size, config validation) and BFL's test suite: every test passes on the devbox
  or in the training image.
- Adapter check (CPU): canvas, state token, caption and position ids bit-identical to training on 9 real
  windows, also through the full simulator → server path; recorded demos within 0.73–1.16/255 (codec);
  recorded actions returned exactly.
- 1-node cluster test: 20 updates, two saves, two exports, resume from the cloud copy (see above).
- Eval-image smoke test on the test's export: 3 tasks × 3 episodes ran end to end (server load ~30 s,
  0.67–0.75 s per policy call).

## Eval plan

- 10-task check at update 1,500: the baseline's 10 tasks and seeds (GR00T's first 10 randomized seeds),
  paired with the baseline's 73/99 and with `flux3_headnative_b512`.
- Full 50-task eval at 3,000: both configs, GR00T's exact seeds (as the baseline: randomized 71.0%, clean
  71.1%). **The 3,000 10-task check is read from this eval's logs** (same seeds, same queue-line layout;
  the user's decision), not run separately; baseline 72/99.
- Held-out arm error at 1,500 and 3,000 (`flux3/heldout_check.py`); baseline at 3,000: 0.029 / 0.021 /
  0.023 rad.

## Results

(pending)
