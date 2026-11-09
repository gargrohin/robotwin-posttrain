# Run card: `flux3_headnative_b512` (main head-native run)

FLUX 3 Action trained like the baseline `flux3_mix200_b512`, with one intended change: the head camera
goes onto the model's canvas at its native 240×320 instead of being upscaled to 360×640 and stretched
from 4:3 to 16:9. The question is how well the pretrained model adapts when the unnecessary upscaling
is removed. A side run, `flux3_headnative_w41_b512`, is identical except for the action loss weight
(see its card).

| | |
|---|---|
| Submitted | 2026-11-07 22:46 EST, on the cluster's shared borrowed pool (pre-emptible) |
| Started | see LOG.md |
| Hardware | 1 node, 8 H100s; batch 512 = 8 GPUs × 4 windows × 16 accumulation |
| Measured in the test (20 updates + resume, 2026-11-07 22:05–22:36) | 37.9–39.8 s per update (baseline layout on 1 node: 52.5–54); data wait ~0.002 s; peak allocated 62.1–62.7 GiB, device memory used 74 GiB of 80; save ~66–71 s; resume check passed |
| Expected length | 3,000 × ~38.6 s + saves and loader restarts ≈ 34 h without pre-emptions (~270 GPU-hours) |
| wandb | `flux3-f3a-headnative-b512-1n` (project `robotwin-posttrain`) |

## Code, model, data

| | |
|---|---|
| Trainer | BFL flux-action, our branch `exp-small-canvas` @ `63cc40f` (from `robotwin` @ `5e8d3df`): new camera layout `droid_native_head`, two configs. Not pushed. |
| Config | `configs/robotwin/train_headnative_b512_s3000.json` (flux-action) |
| Base model | `black-forest-labs/flux-3-action-base`, same files as the baseline |
| Data | `mix200`, the same flux-action index as the baseline (12,500 episodes) |
| Eval adapter | XPolicyLab branch `exp-small-canvas` @ `e0189a1` (from `local/aloha-gr00t` @ `2b435d8`): head-native exports get the simulator's 240×320 frames unchanged; own warm-up request |

## Recipe

Everything as the baseline's `train_b512_s3000.json`: BFL base, fresh 14-D heads, seed 42; 3,000 updates
at batch 512; trunk frozen for 100 updates, warm-ups 200 (trunk) / 100 (heads), linear cool-down
2,500 → 3,000; peak learning rates 1.92e-4 (trunk) / 9.6e-4 (heads); AdamW (0.9, 0.99), eps 1e-8, weight
decay 0.05; **no gradient clipping** (`max_grad_norm: null`, set explicitly; the trainer's default would
be 1.0); timesteps logit-logistic 0.75 / shift 42; action scale 2; loss video 1 / **action 50**; caption
dropout 0.1; augmentation on; power EMAs 0.10 and 0.05.

## Differences from the baseline

| what | baseline `flux3_mix200_b512` | this run | measured size / status |
|---|---|---|---|
| canvas | 544×736, DROID layout; latent 17×20 = 340 tokens per frame | 448×640, `droid_native_head`; latent 14×20 = 280 tokens per frame | video tokens per window 3,060 → 2,520 (predicted 2,720 → 2,240) |
| head row | head upscaled 240×320 → 360×640 (stretched 4:3 → 16:9), full width | head at native 240×320, centered; black 160-px bands left and right | the head covers 7.5×10 latent tokens instead of 11.25×20 |
| padding | reflect: right 96 px (dropped from the latent), bottom 4 rows | black side bands; reflect bottom 28 rows | the last latent row holds 4 content rows and 28 padding rows |
| head augmentation | random crop 95% + resize back + color jitter on the upscaled 360×640 frame | the same on the native 240×320 frame | inherent to the change; same crop fraction and jitter ranges |
| wrist path | data loader upscales to 360×640 (ffmpeg), augmentation, then halved to 180×320 | decoded at 240×320, augmentation, then upscaled to 360×640 (torch bilinear) and halved inside the layout | wrist slot differs from the baseline's training pixels by 1.14/255 on average (p99 4/255, max 16/255; 48 frames). The baseline's own eval inputs differed from its training inputs by the same amount (1.16/255). Augmentation now acts before the wrist upscale |
| action loss weight | 50 (set by BFL for 2,720 video tokens per window) | 50, unchanged | with 2,240 video tokens, actions weigh 1.21× more relative to video than in the baseline (the side run tests 41.18) |
| hardware | 2 nodes, 16 GPUs × 4 windows × 8 accumulation | 1 node, 8 GPUs × 4 windows × 16 accumulation | **assumed not to matter, not tested.** Same batch 512 and same math; which windows share an update differs |
| data order | seed 42, one uninterrupted job | seed 42; the shared pool can pre-empt, and each restart draws a new data order from the run seed (BFL's resume default) | statistically equivalent; restarts are counted in LOG.md |
| logging and saving | metrics every 10 updates; full state every 150 | metrics every update; full state every 50 into two slots overwritten in turn | no effect on training math |
| speedups | none | none, for the whole run | a switch to the exact speedups at 8 per GPU was planned on 2026-11-08 and called off at 11:05 ET: restarting means re-queuing for a whole borrowed node, and none had come free for ~9 h, so the restart could cost more than the ~5–6 h saved. The run finishes on its launch settings. |
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
  paired with the baseline's 73/99.
- Full 50-task eval at 3,000: both configs, GR00T's exact seeds (as the baseline: randomized 71.0%, clean
  71.1%). **The 3,000 10-task check is read from this eval's logs** (same seeds, same queue-line layout;
  the user's decision), not run separately; baseline 72/99.
- Held-out arm error at 1,500 and 3,000 (`flux3/heldout_check.py`); baseline at 3,000: 0.029 / 0.021 /
  0.023 rad.

## Results

### Update 1,500 (EMA 0.10 export; checked 2026-11-08 21:19–22:56 EST)

10-task check, randomized, on the baseline's exact seeds (GR00T's first 10 randomized seeds per task; seed lists
verified 10/10 tasks):

| | `flux3_headnative_b512` (w50) | `flux3_headnative_w41_b512` (w41) | baseline `flux3_mix200_b512` |
|---|---|---|---|
| first 10 seeds per task | **74/100** | **72/100** | 74/100 |
| paired set (decay-run seeds, as `gr00t/paired_compare.py`) | 73/99 | 71/99 | 73/99 |
| seed by seed vs baseline (only this one solves / only the baseline solves) | +13 / −13, sign test p = 1.0 | +12 / −14, p = 0.85 | |

w41 vs w50 seed by seed: +9 / −11, p = 0.82. Per task (w50 / w41 / baseline): pick_dual_bottles 6/6/3,
place_object_stand 7/7/8, move_pillbottle_pad 6/6/8, place_empty_cup 10/10/8, place_phone_stand 8/6/9, move_can_pot
2/1/3, dump_bin_bigbin 8/9/7, handover_mic 7/7/8, grab_roller 10/10/10, click_bell 10/10/10.

**At 1,500, removing the head camera's upscaling changes nothing measurable**, and neither does the loss weight
(41.18 vs 50). Per-task gaps are 1–3 seeds of 10, within noise.

Held-out arm error at 1,500 (train randomized / train clean / unseen randomized; ranks checkpoints within a run but
does not line up across runs, so information only): w50 0.0365 / 0.0290 / 0.0300, w41 0.0368 / 0.0286 / 0.0306,
baseline 0.0366 / 0.0274 / 0.0296. Policy calls 0.83–1.05 s on the devbox GPU (baseline canvas: 1.95 s).

### Update 3,000 (status checked 2026-11-09 11:42 ET)

Training is complete; the EMA 0.10 export manifest is present in cloud storage. All 12 full-eval slices were
submitted by 11:41 ET and are queued. The launcher uses the baseline update-3,000
queues and per-stream seed lists: all 50 tasks, 1,017 planned randomized episodes and 500 clean episodes.
Results are pending; the same logs will provide the 10-task comparison.
