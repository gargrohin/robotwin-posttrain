# Run card: `flux3_mix500_hn_9k` (FLUX 3 Action on mix500, head-native canvas, 9,000 updates)

The FLUX 3 Action recipe of the head-native runs (`flux3_headnative_b512`: head camera at its native
240×320 on a 448×640 canvas, action loss weight 50, no gradient clipping), trained from the BFL base
weights on the larger `mix500` data for 9,000 updates, with the measured speedups switched on.

| | |
|---|---|
| Hardware | 1 node, 8 H100s; batch 512 = 8 GPUs × 16 windows × 4 accumulation |
| Queues | the same run is submitted twice at the same priority (team quota without borrowing, and the shared borrowed pool); the copy that does not start first is cancelled, the other keeps the run's checkpoint folder |
| Expected speed | ~26.4 s per update measured on the larger DROID canvas with the same speedups; the head-native canvas should be faster, not yet measured together |
| wandb | `flux3-f3a-mix500-hn-9k-1n` (project `robotwin-posttrain`), from update 1 |

## Code, model, data

| | |
|---|---|
| Trainer | BFL flux-action, our local branch `mix500-run` @ `ec9d0b3`: `exp-small-canvas` (camera layout `droid_native_head`) merged with `exp-speedup` @ `e85df2d` (`dit_compile`, `loader_spans_epochs` and the other speedups), plus one new option `extra_checkpoint_steps` (additional saves at listed updates). Not pushed. |
| Config | `configs/robotwin/train_mix500_hn_9k.json` |
| Base model | `black-forest-labs/flux-3-action-base`, fresh 14-D heads, seed 42 |
| Data | `mix500`: 50 tasks × (50 clean + randomized 0–494) = 27,250 episodes, 6,019,158 frames, 5,119,908 valid window starts (randomized 495–499 per task held out) |
| Index | built with `flux-action index-lerobot` exactly as the mix200 index (3 cameras, absolute actions, chunk 32, range normalization with clip 6); counts: 27,250 listed = eligible = train episodes, 0 rejected |
| Statistics | `statistics.json` sha256 `79848144…`, `manifest.json` sha256 `1232aeaa…`. Against the mix200 statistics, the largest change of any action or state q01/q99 value is 0.9% of that dimension's mix200 range |
| Eval adapter | XPolicyLab branch `exp-small-canvas` @ `e0189a1`, as for the head-native runs |

## Recipe

BFL's DROID schedule shape, scaled to 9,000 updates (the 30,000-update original scaled by 0.3):

| phase | updates |
|---|---|
| trunk frozen | 300 |
| trunk warm-up | 600 (trunk LR reaches its peak at update 900) |
| heads warm-up | 300 |
| constant peak | to 7,500 |
| linear cool-down | 7,500 → 9,000 |

Peak learning rates 1.92e-4 (trunk) / 9.6e-4 (heads), AdamW (0.9, 0.99), eps 1e-8, weight decay 0.05,
**no gradient clipping** (`max_grad_norm: null`), timesteps logit-logistic 0.75 / shift 42, action scale 2,
loss video 1 / **action 50**, caption dropout 0.1, augmentation on, power EMAs 0.10 and 0.05.

Learning rate used by update `g` (zero-based; computed with the trainer's schedule code):

| update g | trunk | heads |
|---|---|---|
| 0 | 0 | 3.19e-06 |
| 300 | 0 | 9.60e-04 |
| 900 | 1.92e-04 | 9.60e-04 |
| 7,500 | 1.92e-04 | 9.60e-04 |
| 8,250 | 9.60e-05 | 4.80e-04 |
| 9,000 | 0 | 0 |

## Differences from the head-native w50 run (`flux3_headnative_b512`)

A config comparison (`TrainConfig`, after the launcher's overrides) shows only these changes:

| what | head-native | this run |
|---|---|---|
| data, statistics | mix200 (12,500 episodes) | mix500 (27,250 episodes); one epoch = 52 updates |
| length and schedule | 3,000; frozen 100, warm-ups 200 / 100, cool-down 2,500 → 3,000 | 9,000; frozen 300, warm-ups 600 / 300, cool-down 7,500 → 9,000 |
| windows per GPU × accumulation | 4 × 16 | 16 × 4 (same batch 512) |
| speedups | none | `stage_cameras`, `encode_thread`, `vae_graph`, `vae_skip_padding`, `vae_no_autocast`, `vae_compile`, `text_encoder_compile`, `dit_compile`, `inductor_deterministic` (+ `TORCHINDUCTOR_DETERMINISTIC=1`), `reduce_scatter_every_microbatch`, `loader_spans_epochs`, `log_recompiles`. Measured on the larger canvas: 1.98× faster; the compiled parts change the loss by about 1e-4 (rounding level), not bitwise. |
| logging | every update in both | every update |
| saves | full state every 50 updates | full state every 250 updates in two slots overwritten in turn (pre-emption recovery); a permanent copy at update 100 (resume test); exports (EMA 0.10, bf16) at 2,250 / 4,500 / 6,750 / 9,000 |

`loader_spans_epochs` is exact only if an epoch's batch count per GPU is a whole multiple of the 8 loader
workers: 27,250 episodes give 26 rounds × 8 = 208 batches per GPU per epoch, a multiple of 8 and of the
accumulation 4, so it applies. 626 episodes per epoch are left out (different ones each epoch).

## Checks before launch

- CPU tests on the branch: 258 pass; the one deselected test (`test_seek_decode_matches_sequential_decode[ffmpeg]`)
  fails on the devbox's old ffmpeg for the same reason it fails on the base branch.
- Config comparison against the head-native config (above): only the listed differences.
- Index: 27,250 episodes, statistics compared with mix200's; data files in cloud storage match the local
  mix500 folder in name and size (109,008 files).
- No devbox GPU was free for a 1-GPU smoke run of the combination (head-native canvas with the compile stack);
  the first minutes of the cluster run are its first test.
- Resume test, when the permanent update-100 copy exists: resume from it for 20 updates on the borrowed pool, without
  saving; check that step, learning rates, both EMA betas and loss continue and that nothing recompiles after warm-up.

## Eval plan

Full 50-task evals, GR00T's exact seeds (the same protocol and seed lists as every Flux eval), on the exports at
updates 2,250, 4,500, 6,750 and 9,000; the head-native adapter and eval image; compared with the Cosmos mix500 run
and with Flux at update 9,000 on mix200 (75.6% randomized / 77.8% clean).

## Restart at update 2,500 (2026-11-12 03:54 ET)

The first job ran from 2026-11-11 10:42 ET to update about 3,100 at 19.7 s per update. Two problems were found at
update 2,500:
- The in-job export of update 2,250 failed: a checkpoint written with `dit_compile` names the DiT block weights with
  a `_orig_mod` segment, and the exporter looked for the plain names. The failed export kept the older save slot
  reserved, so no save after 2,500 could upload.
- The permanent copy at update 100 failed (server-side copy of objects over 5 GB), so no permanent copy exists.

Fixes: the exporter now loads weights under the names stored in the checkpoint (same results for checkpoints without
the segment; branch `mix500-run` @ `f456b5f`, with a CPU test), and the permanent-copy list is empty. Resume was
checked before cancelling: the model keys the compiled and checkpoint-wrapped DiT produces equal the saved keys
(451 of 451). The run then resumed from the update-2,500 save with the same run name, config and schedule; the
updates after 2,500 of the first job (about 600) were discarded. As on any resume, the data order is redrawn from the
run seed.

## Results

Not yet available.
