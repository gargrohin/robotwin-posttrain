# Investigations: questions, findings, corrected assumptions

This is how the project is run: keep asking small questions, verify every answer, follow anomalies deeper,
and hold no preconceived assumptions. A running record of questions asked about our training and evals, what checking them showed, which earlier
assumptions turned out wrong, and what is still open. Small questions about speed or odd numbers have
repeatedly overturned things we believed about the runs, so every claim here says how it was checked.
Newest topics at the bottom. Times are ET.

Conventions: **Q** = the question asked. **Found** = what checking showed (with how). **Corrected** = an
earlier belief that was wrong. **Open** = not yet answered.

---

## 1. Why are Cosmos and Flux so far ahead of GR00T? (Nov 6)
- **Q:** model size, trainable parameters, or something else?
- **Found** (counted from our checkpoints):
  - GR00T has 3.46B parameters, of which our runs train 1.62B. Its vision-language part (1.83B) stays frozen.
  - Cosmos3-Nano trains ~7B of 15.2B, including its image-generation part.
  - Flux trains all ~7B.
  - Cosmos and Flux also keep predicting future video during fine-tuning.
- **Found:** GR00T's held-out prediction error (0.016) is lower than Flux's (0.023), yet it does far worse in
  the sim. So copying demo actions well is not the same as acting well in closed loop.
- **Open:** size, frozen vs trained vision, and model type are all mixed together across the three models.
  The direct test is GR00T with its vision-language part unfrozen.

## 2. Public RoboTwin results and improvement recipes (Nov 6)
- **Q:** what models and hill-climbing recipes have worked for others?
- **Found** (research, sources in `notes/robotwin_public_results.md`):
  - There are two protocols. The official leaderboard trains on clean demos only. Most papers train on clean
    plus 500 randomized demos. Numbers across the two aren't comparable.
  - Under the papers' protocol, the best models reach 92–96%. Our Cosmos (85.0 / 88.6) is level with Motus
    and LingBot-VLA.
  - No public RoboTwin numbers exist for Cosmos3-Nano or FLUX 3 Action.
  - The closest recipe to our case is value-model sampling on the policy's own rollouts (Q-Planning:
    83 → 91%). Correction data also has support. RL gains are shown only from low starting points.

## 3. Cosmos: does a failed scene fail every time? (Nov 5)
- **Q:** rerun decay 5,500 on the 209 scenes it failed, with the same instruction sentences.
- **Found:**
  - 89 of 209 succeed on a second try (43%): randomized 65/152, clean 24/57. All 209 got the original
    sentence.
  - Checkpoint 6,000 solved 93 of the same 209, so the difference between those two checkpoints was mostly
    sampling randomness.
  - Cosmos seeds its sampling once per policy-server process, not per episode.
- **Corrected:** the instruction sentence depends on how many episodes the eval stream runs, so a naive rerun
  would have changed the sentence for 198 of the 209 scenes. Our seed-list client now takes the original
  count.
- **Found:** the cluster eval package had a startup bug: right after uploading its inputs, a job sometimes
  found a seed list missing (once on the Cosmos rerun, 3 of 12 slices on Flux 5,000). A re-download check
  now guards it. The cause is still unknown.

## 4. Flux continuation: loss going up (Nov 5–6)
- **Q:** the loss went up after the continuation started. Is the learning-rate re-warm to blame?
- **Found:**
  - Yes. The first run ended its cool-down at 0.067. Re-warming to the peak raised it to ~0.0775, then it
    fell to ~0.072.
  - At the same peak learning rate, the first run was at 0.078, so progress continued.
- **Q:** is there precedent for re-warming a cooled-down checkpoint?
- **Found:**
  - Warm restarts (SGDR) bump the loss at each restart, then end lower.
  - Continual pre-training papers (Gupta et al. 2023; Ibrahim et al. 2024) show the same bump and recovery.
  - With a constant learning rate followed by a cool-down (MiniCPM; Hägele et al. 2024), the recommended way
    to extend a run is to continue from before its cool-down.
- **Corrected:** I had recommended deleting the first run's intermediate full training states and didn't
  point out that the states just before its cool-down (2,400–2,850) were the textbook place to extend from.
  That option is now gone. The latest states of every run were kept.

## 5. Flux step 5,000 scored worse than step 3,000 (Nov 7)
- **Q:** why did 5,000 drop to 64.0% randomized / 61.9% clean, from 71.0 / 71.1?
- **Your hypothesis:** the re-warm phase.
- **Found, held-out check** (prediction error on demo moments, training and unseen demos):
  - 5,000's error doubled on the training demos as well as the unseen ones: 0.050 / 0.044 / 0.043, against
    3,000's 0.029 / 0.021 / 0.023. So it wasn't overfitting.
  - **Q:** what is this check and why does it show overfitting? Overfitting would keep the training-demo error
    low while the unseen error rose; here both rose.
- **Found:** step 4,000 equals step 3,000 (0.030 / 0.022 / 0.023). So the damage came between 4,000 and
  5,000.
- **Found:**
  - a loss spike at update 4,790: gradient norm 3.9 (~25× normal), action error 20× normal, recovered within
    ~40 updates;
  - only one update with a gradient norm above 1.0 in ~4,800 updates across both runs;
  - smaller excursions at ~3,370 and ~4,370.
- **Q:** 5,000's loss and gradient norm had already recovered, so how can the spike explain it?
- **Found:**
  - The export is the running average (EMA) of the weights, not the weights the loss is measured on.
  - **Q:** why export a running average? It smooths out the noise of high-learning-rate training. BFL's
    recipe exports it, and near the end of a cool-down it converges to the trained weights.
  - Test at 5,350: the average (0.042 / 0.034 / 0.035) was *worse* than the trained weights it came from
    (0.037 / 0.029 / 0.031). The average carried the spike, with 41% of it still from before the spike.
  - The trained weights were still ~0.007 worse than at 4,000. Lasting damage, high-learning-rate noise, or
    both. This can't be separated, because no pre-spike trained weights survived.
- **Found** (spike investigation):
  - No error, data, video or hardware cause.
  - The exact batches were rebuilt from the seed. Two very hard windows came right before the spike, but
    equally hard ones trained many times without one.
  - Most likely cause: optimizer instability at the peak learning rate with no gradient clipping. Confidence
    moderate.
- **Your decisions:**
  - Keep going as planned. Don't stop the run.
  - No mid-run recipe changes (clipping, learning rate), so the final model maps to one recipe. Fixes go into
    a new run.
  - Spikes get reported, and each eval is read against them.
- **Found** (eval results):
  - 5,500: 66.3 / 67.1.
  - 6,000: 71.8 / 71.3, level with 3,000. Held-out 0.035 / 0.027 / 0.029.
  - A small spike at 6,460 (gradient norm 1.18, action error 4×). About 47% of the 7,000 average comes from
    weights after it.
- **Open:** whether the cool-down (8,000→9,000) lifts the continuation above step 3,000.

## 6. Cluster capacity and moving the run (Nov 5–7)
- **Found:**
  - Borrowed capacity gets pre-empted: two 1-node copies were pre-empted within minutes.
  - 2-node borrowed jobs need two whole idle nodes in one network block, and borrowers can't pre-empt others,
    so they can wait for hours.
  - Our own quota can reclaim capacity from borrowers.
- **Found:** who uses which GPUs (read-only cluster queue query). Several teams leave large quotas unused,
  and that unused quota is what the borrowed pool runs on.
- **Found** (bug):
  - Cluster jobs can't delete in cloud storage. The run's saves therefore overwrite two fixed slots.
  - A restart inside the same 2-node job could reuse stale save markers and corrupt a save. Fixed with one
    marker folder per admission, then tested with a fake store and a real 2-node restart.
  - Loading 1-node saves onto 2 nodes was verified.
- **Q:** a CPU-only machine for the export helper?
- **Found:**
  - The export runs on CPU with byte-identical output.
  - Our CPU queue is full and isn't allowed to borrow.
  - The helper kept its GPU idle while waiting (88 min once). It now goes in ~10 min before each save.
- **Corrected:** I said a process on a devbox GPU couldn't be stopped. We have stopped idle servers several
  times before; this time the process just wasn't visible from the devbox.

## 7. Why Flux trains ~10× slower than Cosmos per example (Nov 7)
- **Q:** why is Flux slower?
- **Found** (from configs and measured speeds):
  - Flux: ~1.2 examples/s/GPU, 4 per GPU at a time. Each example is 33 frames on a 544×736 canvas with 32
    actions.
  - Cosmos: ~12 examples/s/GPU, 32 per GPU at a time. Each example is a 16-action chunk at its 256-pixel
    setting.
  - Both train ~7B parameters, so the difference is the amount of video per example.
- **Q:** why such large frames?
- **Found:**
  - The base model was fine-tuned by BFL on the DROID layout: one 360×640 view over two half-size views.
  - Our cameras are 240×320. The head camera is upscaled to 360×640 and stretched from 4:3 to 16:9.
  - The wrists are shrunk to 180×320, slightly squashed.
  - Most of the canvas's pixels are interpolated, so they carry no information.
- **Q:** are we sending only half of each wrist image?
- **Found:** no. The whole wrist image is there, at lower resolution (180×320 against the native 240×320).
- **Q:** would Cosmos do better at Flux's resolution?
- **Found:**
  - My first answer, "probably not much; its input is already close to the cameras' native size", was wrong.
  - **Corrected (Nov 7, from the code):** Cosmos composes the three cameras into a 360×320 canvas, head on top
    and wrists below. Its 256 setting then *downscales* that to 256×228, adding 28 columns of padding. So the
    head camera reaches the model at ~0.7× native size per side, about half its pixels.
  - Upscaling to Flux's size would still add no information. Native size for Cosmos would mean a larger
    setting than 256, which is untested.
  - Each maker chose its resolution for its own reference data: LIBERO's small images for NVIDIA, DROID's HD
    cameras for BFL.

## 8. Speed experiment findings (Nov 7, before any runs)
- **Corrected:**
  - "Activation recomputation would save memory": it's already on for all 33 transformer blocks (BFL's
    default), checked in the config and the code.
  - "8 examples per GPU doesn't fit": that failure was under an older encoder setting, and 8 was never tried
    with today's.
- **Found:**
  - The largest memory item is a full-precision copy of all gradients kept during accumulation, 27.6 GB per
    GPU. Sharing it out after each micro-batch (ZeRO stage 2, gradient sharding) frees ~24 GB.
  - The video encoder can work through each clip 8 frames at a time with the same math (time-tiling).
  - Tiling by image regions would change its output, because its attention spans the whole canvas.
  - Each 33-frame clip is padded with 12 black frames before encoding. The purpose is unknown, and it's being
    measured.
  - The text encoder runs on almost every micro-batch: 256 cached captions against 11,293 distinct ones.
  - Nothing logged where an update's time goes. Timing logs are being added.
  - The running job peaks at 65 GB of tensors per GPU. The GPU driver reports 99% used, because PyTorch keeps
    freed memory in reserve.
- **Your decisions:**
  - Test time-tiling, gradient sharding, both together, and 8+ examples per GPU.
  - Measure the black padding separately.
  - Add timing logs.
  - No selective recomputation: the gain is too small.

## 9. Small-canvas experiment design (Nov 7)
- **Goal (yours):** remove unnecessary upscaling. Not speed, and no added downscaling.
- **Corrected:** my first option (A) would have shrunk the wrists 3×, a second change. The chosen layout
  keeps the wrists exactly as today and puts the head at its native 240×320 (no upscale, no stretch). That
  gives a 448×640 canvas, with black side bands and mirrored padding at the bottom.
- **Q:** the action loss weight.
- **Found:**
  - BFL's weight of 50 was set for 2,720 predicted video tokens per window.
  - The new canvas has 2,240. Keeping 50 raises the actions' share of the loss from 37% to 42%. 41.18 keeps
    37%.
- **Open:** keep 50 or rescale. A literature review of how BFL chose 50 and how others handle a changed video
  token count is running. Your view: possibly worth its own experiment.

## 8b. Video encoder measurements (Nov 7, 20:50, one devbox GPU, 64 real training windows)
- **Q:** is time-tiling exact, and what do the black padding frames do?
- **Found:** every variant below gives encoder output bitwise identical to today's (max difference exactly 0).

  | variant | time per window | peak memory per window |
  |---|---|---|
  | today's encode | 0.307 s | 8.44 GiB |
  | time-tiled, 8 frames | 0.408 s (+33%) | 3.58 GiB (−58%) |
  | time-tiled, 16 frames | 0.348 s (+14%) | 4.23 GiB (−50%) |
  | no black padding (33 frames) | 0.232 s (−24%) | 5.98 GiB (−29%) |

- **Found (padding):**
  - The encoder's chunked mode only accepts 45 + n×44 frames, so every 33-frame window is padded to 45 to fit
    that rule.
  - The encoder looks only backward in time, so the 12 padding frames can't change the 9 compressed frames kept.
    The measurement confirms it.
  - The code's stated reason is that training and deployment see the same latent distribution.
  - Dropping the padding would cut encoder time by 24% with identical output. Not changed (your decision).
- **Open:** how much of an update the encoder takes. Stage 1's timing logs measure it.

## 9b. How the action loss weight of 50 was chosen (Nov 7, literature review)
- **Q:** what does BFL say about 50, and what do others do when the number of video tokens changes?
- **Found:** the full review, with sources, is in the Flux experiment's local notes (loss_weight_review.md).
  - **BFL's only explanation is a code comment** in packing.py. A base of 100 "undoes the dilution of 32
    action tokens among 2752 (2720/32 = 85, rounded up to tilt toward actions)", then divided by the action
    scale 2, giving 50. No public doc, blog, model card or report explains it or ablates it. The tech report's
    1/s² reweighting doesn't match the code's single division by s.
  - **BFL's own recipes don't rescale it.**
    - Games and drone: a 512×512 canvas (2,048 video tokens) with weight 50 kept, so actions get 0.78 of the
      video weight, against DROID's 0.588.
    - SO-101: per-modality averaging, weights 1.0 / 0.5.
  - **Other models:** none adjusts the weight when resolution or token count changes, and none ablates it. Most
    average each modality separately with fixed weights, so the balance doesn't depend on token count (Cosmos3
    10/10, UVA, Motus, LingBot-VA, UWM, Genie Envisioner). WorldVLA picked its weight because image tokens
    outnumber action tokens, then kept it when the image token count quadrupled.
  - **In our logs,** the action term is only ~5% of the logged loss value (4–8%), because per-token action
    errors are much smaller than video errors.
    - The "37% share" is the share of the weights, not of the loss value.
    - The logged `loss` isn't comparable across canvases. Compare `video_mse` and `action_mse` instead.
  - **Corrected:** the trainer's default is gradient clipping at 1.0. The baseline config explicitly sets it off
    (`max_grad_norm: null`). Any new config must set that explicitly, or it silently clips.
- **Recommendation (review):** 41.18, the cleanest control, but with low confidence that 50 vs 41.18 changes
  results measurably. The spread across BFL's own recipes (0.5–0.78) is wider than the shift from keeping 50.

## 9c. Head-native canvas: decisions (Nov 7)
- **Two runs, one change each:**
  - **Run 1:** head-native canvas, action weight 50. Versus the first run, only the canvas changes.
  - **Run 2:** the same with weight 41.18. Versus run 1, only the weight changes. This is your choice, so we
    learn whether the balance matters even if the review expects no difference.
- **Accepted differences, to be listed exactly in each run card:**
  - **Wrists:** made through the baseline's resize steps inside the new layout. ~1.1/255 mean pixel difference
    from the baseline's training bytes. Crop and jitter now come before the wrist resize.
  - **Head:** augmentation acts on native frames, which is inherent to removing the upscale.
  - **Hardware:** 1 node vs 2. *Assumed* not to change training noticeably; not tested.
- **Speedups:** only ones whose training math is unchanged and that measurably help (exact time-tiling; more
  examples per GPU and gradient sharding, checked equivalent). The same on both runs. Never the padding drop.

- **Found (Nov 7, building the head-native runs):**
  - The cluster launcher's entrypoint forces `log_every` 10 in training mode, whatever the config says. That's
    why the main run logs every 10 updates. Logging every update needs an explicit override at launch.
  - With no resize anywhere, the head-native eval input is bit-identical to training on 9 real windows. The
    baseline's eval differed from training by 1.8–2.0/255 because of its resize.

- **Open (Nov 7, 22:10):** how often do borrowed 1-node jobs get pre-empted? Both experiment test jobs run on the
  borrowed queue by choice. Our quota had 32 GPUs free, but no node was physically idle. Each experiment's notes
  keep a running count of pre-emptions, updates lost and time to re-admission.

## 8c. Where Flux's training time goes (Nov 7, speed test stage 1, reference run, 1 node, 4 examples/GPU)
- **Measured:** 52.4 s per update; peak allocated 65.8 GB, peak reserved 75.3 GB.

  | where | phase | seconds per update |
  |---|---|---|
  | main model (main stream) | forward | 8.5 |
  | main model (main stream) | gap with no main-model work, between forward and backward | 22.4 |
  | main model (main stream) | backward, incl. recomputation | 21.5 |
  | encoders (side stream) | video encoder | 22.1 |
  | encoders (side stream) | text encoder | 2.0 |
  | encoders (side stream) | augmentation + canvas | 1.8 |
  | other | optimizer + averaging + gradient sync | ~0.05 |
  | other | data wait | ~0 |

- **Corrected:** the trainer is designed to run the next micro-batch's encoders on a side stream *during* the
  current backward pass. In practice they don't overlap. The 22.4 s gap matches the video encoder's 22 s, so the
  encoder runs on its own, and it takes ~43% of every update. The profiler agrees: the GPU is busy 46.5 of
  53.2 s, and the summed kernel time (48.5 s) shows the two streams barely overlap. This applies to the main
  run too, since it uses the same code.
- **Implications:**
  - Time-tiling makes the encoder slower (+33%) with nothing to hide it behind, so it's expected to be slower
    overall unless its freed memory buys more examples per GPU.
  - The big levers are outside the approved tests:
    - dropping the padding frames (exact, ~5 s/update less);
    - making the overlap actually work (up to ~22 s/update less if fully hidden);
    - encoding the video once ahead of time (removes the encoder entirely, but changes how augmentation works).
- **Your decisions (Nov 7, 23:00):**
  - Top priority: make the encoder/backward overlap actually work, as a side experiment now, proven to leave
    training unchanged (equivalence run against the reference).
  - Also important: drop the padding frames (exact), measured alone and with the overlap fix.
  - Tiling is adopted only if it pays off, likely only on top of a working overlap.
- **Found (checked in code):** nothing is compiled during Flux training. `compile_model` defaults to False and
  would only compile the video encoder and text encoder; the DiT is never compiled. That fits the ~23 s per
  update of small non-matmul kernels. Compiling the encoders is now a candidate in the overlap work.
- **Your decisions (Nov 7, ~23:20):**
  - Quantify the overlap fix and compiling, alone and together, and prove whether they change training.
  - If they don't, adopt them in the head-native runs (both runs at once, switched right after a save) and in
    future Flux runs.
  - The ongoing main run only with your explicit permission.
- **Idea (yours, Nov 7, ~23:30):** if the speedups are large, train Flux 0→9,000 from scratch with one schedule
  (BFL's DROID recipe scaled to 3/10), instead of today's 0→3,000 + re-warm 3,000→9,000. That gives one recipe
  end to end, and a direct test of longer training without the re-warm. On 2 nodes: ~3 days at today's speed,
  ~1.8 days at an estimated 1.6×. "Within a day" needs ~2.9×. Same canvas and settings as the first run. Decide
  once the speed numbers are measured.

## 8d. Why Flux's video encoder doesn't overlap, and what the speedups do (Nov 7, 23:00)
- **Found (root cause, from a devbox profiler trace):**
  - The step that prepares the next small batch (augmentation, text encoder, video encoder) holds the CPU
    thread ~1.45 s per small batch, about as long as the encoders' GPU work.
  - The main cause: camera frames from the data loader get stacked into a new, non-pinned CPU tensor
    (~274 MB per small batch), so each copy to the GPU blocks until the previous window's encoding is done.
    There are 122 blocking copies plus other syncs.
  - The backward pass can only be queued after that step returns, so the GPU's main queue sits idle meanwhile.
  - Fix that works: run the preparation in a worker thread. The backward is then queued at once, and the gap
    disappears in the trace.
- **Found (correctness, end-to-end on 1 GPU with real data):**
  - Bitwise identical inputs to the main model, and bitwise identical loss and gradient norm: the worker
    thread, staging frames to the GPU first, a CUDA graph for the encoder, dropping the padding, tiling, and
    all of them together.
  - Gradient sharding and tiling at 4 per GPU: the first update's loss bitwise identical, later differences
    within run-to-run noise (loss ~1.5e-5, gradient norm ~1e-4).
  - Compiling the text encoder: within noise.
  - **Compiling the video encoder changes its output** (1.16% relative difference) and moves the loss ~20× the
    noise. By the agreed criterion that changes training, so it needs your decision.
- **Found (speed per window on one GPU):**

  | encoder mode | GPU time per window |
  |---|---|
  | today | 0.322 s |
  | CUDA graph | same |
  | no padding | 0.242 s (−25%, exact) |
  | compiled | 0.167 s (−48%, not exact) |
  | compiled + no padding | 0.128 s (−60%, not exact) |

  BFL's own compile option crashes in training (a CUDA-graph mode that conflicts with repeated encodes). The
  experiment compiles it a different way.
- **Found (whole updates, 1 node, 4 per GPU):**
  - tiling 59.4 s (+14%, slower);
  - gradient sharding 52.4 s (no faster, but frees 22.5 GB per GPU);
  - 16 per GPU with tiling still runs out of memory.
- **Open:** the end-to-end speed of the overlap fix and the full stacks (six jobs submitted at 23:02).
- **Q:** the padding satisfies a shape rule, so how can it be skipped?
- **Found (code):**
  - The encoder itself only needs T ≡ 1 (mod 4) frames (`video_vae.py:892`), and 33 qualifies.
  - The 45 + 44n rule belongs only to BFL's chunked routine for long videos (`min_chunk_enc`, lines 730–767).
    Training pads 33 → 45 so each window is one chunk ("so training and deployment see the same latent
    distribution").
  - The no-padding version calls BFL's own unchunked `encode_task` ("without padding or chunk overlap"). Since
    the encoder is causal in time, the result is bitwise identical, as measured.
- **Open (subagent):** why compiling the encoder changes its output, whether that size of change harms training
  (yardstick: bf16 vs fp32), why BFL doesn't compile in training, and why BFL's compile option crashes.
- **Found (speed test stage 1, 1 node, today's settings, updates 4–20):**
  - 4 per GPU: 52.4 s (its repeat 51.8 s: 1% noise).
  - 8 per GPU: 61.8 s, **18% slower**. Near the memory limit the allocator retried 263 times and the GPU idled
    16.8 s per update. The loss was unchanged.
  - 16 per GPU: out of memory.
  - 8 per GPU with tiling: 54.6 s.
- **Corrected:** "more examples per GPU would spread the small-kernel cost". The GPU does the same total work per
  update at 4 and 8 per GPU (46.1 vs 46.5 s busy, ~23 s of small kernels either way). Bigger batches don't
  reduce that work, so the gains have to come from the encoder's ~22 s.
- **Q (user): does more examples per GPU reduce the number of updates, or show the model more data?** Neither.
  The global batch is GPUs × examples per GPU × accumulation rounds (flux-action `trainer.py:253`). The speed tests
  set accumulation = 512 / (8 × examples per GPU), so every update still uses 512 examples, and the number of
  updates and the total data seen don't change. Only the split changes: 4 per GPU is 16 rounds of 4; 8 per GPU is
  8 rounds of 8. At 4 and 8 per GPU each GPU even sees the same windows in an update (speed NOTES). That's why the
  loss matched.
- **Found (head-native launch, measured):** the smaller canvas runs at ~38.6 s per update on 1 node, vs 52.4 s
  for today's canvas on the same setup, about 26% faster, before any speedups.
- **Found (speed, cluster stage 2): gradient sharding at 16 per GPU is 45.7 s per update, 1.20× the
  reference** (peak 57.6 / 69.2 GB, 0 allocator retries; 32 per GPU runs out of memory). The DiT forward + backward
  drops from 30.0 to 24.0 s per update: 4 rounds per update instead of 16 means fewer parameter all-gathers. The
  VAE stays at ~21 s. The equivalence check passed (update-1 loss bitwise identical; later differences within
  run-to-run noise).
- **Corrected (my stage-1 note above):** "more examples per GPU won't buy much". It was true at 8 per GPU without
  sharding, where the allocator thrashed. Once gradient sharding frees memory, fewer rounds per update mean fewer
  all-gathers, and that saves 6 s per update.
- **Found (overlap fix, cluster, 8 GPUs):**
  - DiT inputs were bitwise identical in all 80 hashed micro-batches.
  - Update-1 loss was bitwise identical. Later loss and grad norm stayed within run-to-run noise.
  - So the overlap fix is exact.
- **Found (local 4-GPU, real DiT, small batch):**
  - Overlap removes the gap but gains only 3.5–4.4%.
  - Dropping the padding gains 8.9%.
  - Overlap + VAE graph + no padding gains 13.5%.
  - Every exact variant had a bitwise-identical update-1 loss.
- **Corrected (my earlier framing):** "the gains have to come from the encoder's ~22 s via overlap". Overlap only
  recovers time the GPU sits idle. When the encoder overlaps the backward, the two share the GPU and the backward
  stretches. On the cluster at 4 per GPU the profiler showed 46.5 s of GPU work in 52.4 s, so overlap alone can
  recover at most ~6 s. Larger gains need less GPU work: no padding (−25% of encoder work) and fewer all-gathers.
- **Open (compile):** the local compiled stack ran 63, 16.5, 8.4, 5.9, 10.1, 5.5 s per update. That's start-up
  compile, then spikes. The speed agent's guess: recompiles from new caption shapes in the text encoder. Not
  verified yet; the cluster's 20-update compile runs and the compile investigation should settle it.
- **Found (speed, cluster equivalence, 8 GPUs, 5 updates):** overlap, overlap + VAE graph and padding drop each
  gave the DiT bitwise-identical inputs in 80/80 hashed micro-batches and a bitwise-identical update-1 loss. Later
  differences (≤2.4e-5 loss, ≤1.4e-4 grad norm) are the size of reference-vs-reference noise. With gradient
  sharding and tiling also passed, all five exact pieces leave training unchanged.
- **Found:** CUDA-graph capture of the VAE can't be combined with tiling. Each tiled block copies a CPU value
  (the window start) to the GPU, which capture forbids ("operation not permitted when stream is capturing").
  The trainer now refuses that combination.
- **Corrected (the speed agent's guess, relayed by me):** "compile spikes come from new caption shapes". The
  verified cause: the worker thread's intra-op thread count differed from the main thread's, where the encoders
  were compiled. Compiled code guards on that count, so everything recompiled once in the worker. That's fixed.
  The 10.1 s at the 4-GPU run's fifth update is still unexplained and being checked.
- **Found (compile agent, 1-GPU mini trainer, TORCH_LOGS=recompiles):** a second recompile, measured 6/6 times, also
  after the thread fix. The warm-up encodes two captions (`trainer.py:748-750`), so the text encoder compiles for
  batch 2 and guards `2 <= hidden_states.size()[0]`. The first micro-batch that encodes exactly one caption fails
  the guard and recompiles every language-model layer: 3.0–4.7 s, once per rank, at whichever early update.
  Adding a one-caption warm-up call removed it (zero recompiles after the loader starts, 2 runs). Caption length
  doesn't trigger recompiles. Whether this explains the 10.1 s fifth update is being verified.
- **Found (compile agent):** the thread-count mismatch has a race behind it. The DataLoader's pin-memory thread
  calls `torch.set_num_threads(1)`. A thread created after that starts with 1 thread, while the main thread keeps 96,
  so the outcome depends on thread start order. Setting the count in the worker's initializer removes the race.
- **Q (user): why does compiling the video encoder change the numbers? Is it harmful? Why doesn't BFL compile
  in training, and why did their compile crash?** (compile investigation, devbox, local report.)
  - **Cause, measured op by op:** Inductor fuses RoPE and the GLU gate and rounds once in fp32, where eager rounds
    every step to bf16 (~27% of those outputs move by one bf16 step). Norms change a few values per million
    from a different summation order. Matmuls, the attention kernel and the conv are bitwise.
  - **Magnification:** the encoder magnifies any rounding change. Moving ~29 of 288 M stage-0 values by one bf16
    step changes the final latents by 0.4%.
  - **Yardstick:** compiled is closer to fp32 (1.00%) than eager bf16 is (1.19%); compiled vs eager is 1.10%.
    The eager encoder is deterministic, so there's no run-to-run noise to compare against. The nearest equal
    is swapping NATTEN's kernel (0.65%).
  - **Harm:** nothing measurable without a training run. In a zero-noise mini trainer, an fp32 encoder moves the
    loss 2× more than compiling. Action-loss changes are ≤3e-5 for all variants.
  - **Eval path:** eval encodes the single observed frame eagerly. That's already 0.59% from training's frame
    0, and compiling training would make it 1.19%.
  - **BFL:** documents `compile_model` as a training-speed option but never enables it. Their training never
    compiles the DiT. No reason is written anywhere.
  - **Crash:** each encoder layer is compiled separately in reduce-overhead mode, and with no step mark CUDA
    Graph Trees free layer i's output when layer i+1 runs. Serving survives because it calls
    `cudagraph_mark_step_begin()` per request. Fixed, it's no faster than default mode.
  - **New exact options:**
    - no autocast around the encode: −8%, bitwise;
    - with no padding and cached RoPE tables: −33% encoder time, bitwise.
    - Compile + no padding: −64% encoder time, 1.10% off eager.
    - `TORCHINDUCTOR_DETERMINISTIC=1` makes compile reproducible across processes.
  - **Side finding:** BFL's `F3_NATTEN_BACKEND` override has no effect on the encoder (NATTEN 0.21 ignores the
    backend passed that way).
- **Corrected (mine):** "VAE compile changes the loss ~20× the noise". That compared a mini-trainer change with the
  cluster runs' noise, two different setups. Measured on the same setup, compile moves the loss less than
  switching to an fp32 encoder does.
- **Found (Cosmos, devbox, isolated encoder):** the Wan2.2 encoder on 32 real-sized clips, one clip per call as
  training does, is 0.54 s of GPU time per micro-batch: 0.49 s of kernels over 23,904 launches. The CPU needs 0.54
  s just to launch them, so the call is launch-bound. Batching all 32 clips into one call is 0.46 s, so one clip
  per call isn't the main cost.
- **Open (Cosmos):** in training the same encoder work logs 0.90 s per micro-batch (16-GPU run) and 1.09 s (local
  run), 1.7–2× the isolated time. Not yet known why: GPU gaps inside the calls, CPU contention slowing the
  launches, or something else. The 1-node profiling run (queued on the borrowed pool) is meant to answer it.
- **Found (speed, 1 GPU, real trainer and encoders, tiny DiT):**
  - no autocast: −8% encoder time; RoPE cache: −0.4%. Both are bitwise (DiT inputs, loss, grad norm).
  - Exact v2 (overlap + graph + no padding + no autocast + RoPE cache): −33% encoder time, bitwise.
  - With both recompile fixes, all 12 compiles happen at warm-up and none after. Steps after the first ran
    1.82–2.01 s, with no spikes.
  - The 4-GPU check is running.
- **Found (cluster):**
  - Tiling on top of gradient sharding at 16 per GPU is slower: 51.9 vs 45.7 s.
  - 32 per GPU still runs out of memory, because at that size the DiT's activations are the limit.
  - So tiling buys nothing once gradient sharding is on.
- **Closed (the 10.1 s spike):** the 4-GPU rerun of the fixed compiled stack (10 updates, recompile logging)
  logged 48 recompiles, all during start-up, and zero in updates 2–10. Steady updates ran ~4.8 s, with one of 5.34 s
  (no recompile, no allocator retries, no data wait: ~10% jitter). Update 2 (8.8 s) was the first real text-encoder
  batches and the memory pools filling. So the earlier spike was the batch-of-one text-encoder recompile, which
  the one-caption warm-up removes.
- **Found (cluster compile equivalence, 8 GPUs, 4 per GPU, 5 updates, 80 micro-batches hashed):**
  - Text-encoder compile: text contexts differ in 80/80, but the loss stays within run-to-run noise (update 1
    9.6e-6; later ≤2.6e-5).
  - Video-encoder compile: latents differ in 80/80, and the loss is off 9.1e-5 at update 1 and ≤1.1e-4 later,
    4–7× the noise (1.5–2.4e-5). It changes training slightly, in line with the 1.1% latent difference the
    compile investigation measured. Whether that matters for the policy needs a training run plus eval.
- **Found (local 4-GPU, real DiT, small batch, mean of updates 4–10):** reference 6.79 s; exact v2 5.72 s
  (−15.8%, bitwise); fixed compiled stack 4.91 s (−27.7%, update-1 loss 2.1e-4 off).
- **Corrected:** "compile adds at most ~7% once the overlap works" (the speed agent's earlier estimate, quoted in
  the compile report). Measured on top of exact v2, compile cuts update time a further 14% locally (5.72 → 4.91 s).
  The cluster numbers (v2exact vs v2cmp) are still to come.
- **Found (cluster, 8 GPUs, 4 per GPU, 20 updates):**
  - Overlap alone: 50.0 s per update (1.047×). That fits the ~6 s idle-time ceiling measured earlier.
  - Overlap + VAE graph without gradient sharding: 52.4 s, no gain. The graph keeps an ~8 GB memory pool between
    updates, which leaves no headroom at 4 per GPU; the allocator retried 38 times (reserved 77.4 GB).
  - So the graph only pays together with gradient sharding (which frees ~22 GB), and they'll be adopted together.
- **Found (all exact pieces at once, cluster, 4 per GPU, 5 updates):** overlap + no padding + gradient sharding +
  tiling against its own reference. DiT inputs bitwise in 80/80 micro-batches; update-1 loss bitwise; later
  differences within noise. Combined, the exact pieces leave training unchanged.
- **Q: does the held-out arm error (open-loop, 1,200 demo moments) predict full-eval success?**

  | update | arm error train rand / train clean / unseen rand | success rand / clean |
  |---|---|---|
  | 3,000 (first run) | 0.0294 / 0.0214 / 0.0234 | 71.0% / 71.1% |
  | 4,000 | 0.0299 / 0.0216 / 0.0231 | not evaluated |
  | 5,000 | 0.0503 / 0.0441 / 0.0433 | 64.0% / 61.9% |
  | 5,500 | 0.0392 / 0.0314 / 0.0323 | 66.3% / 67.1% |
  | 6,000 | 0.0345 / 0.0272 / 0.0289 | 71.8% / 71.3% |
  | 6,500 | 0.0306 / 0.0237 / 0.0259 | 74.1% / 76.0% |
  | 7,000 | 0.0297 / 0.0220 / 0.0245 | eval running |

  - **Found:** within the continuation (5,000 → 6,500), arm error falls and success rises at every step.
  - **Found:** across runs they don't line up. Step 3,000 has lower arm error than 6,000 and 6,500 on all three
    sets, yet the same success as 6,000 and less than 6,500.
  - **So far:** the check orders checkpoints within one run, but it can't stand in for the eval when comparing
    across runs or recipes. That rests on 6 points, and nothing explains the cross-run mismatch yet. 7,000's
    eval adds a point.
- **Found (cluster, 8 GPUs, 20 updates; reference 52.4 s at 4 per GPU):**

  | setup | per GPU | s/update | speedup |
  |---|---|---|---|
  | no padding | 4 | 47.2 | 1.11× |
  | overlap | 4 | 50.0 | 1.05× |
  | overlap + graph | 4 | 52.4 | 1.00× (38 allocator retries) |
  | overlap + graph + no padding | 4 | 44.8 | 1.17× |
  | gradient sharding | 16 | 45.7 | 1.15× |
  | overlap + no padding + gradient sharding + tiling, no graph | 16 | 46.9 | 1.12× |

  - Overlap + graph + no padding: forward 9.3 s, backward 35.4 s, no gap. The encoder's work now runs fully
    alongside the backward, and both stretch while they share the GPU.
  - At 16 per GPU without the graph, the main thread waits 21.6 s per update for the worker thread. The worker
    can't queue the eager encoder's kernels fast enough for 16 windows, so the launches are CPU-bound. The graph
    removes that cost but can't be combined with tiling. The Cosmos encoder is also launch-bound (see above).
- **Corrected (relayed by me):** "gradient sharding at 16 per GPU is 1.20×". That was against the stage-2 job's
  own baseline. Against the stage-1 reference (52.4 s) it's 1.15×.
- **Found (head-native run 1, updates 1–216, from its metrics log):** wall time is 41.5 s per update against
  38.8 s of measured step time. The extra comes from two things:
  - each save (every 50 updates) adds ~95–125 s, about 2.2 s per update;
  - every 24 updates the loader starts a new "epoch" (384 batches per rank ÷ 16 rounds) and refills, which adds
    ~18 s, about 0.75 s per update. Over 3,000 updates that's ~37 min, small but avoidable. Not checked yet
    whether persistent workers or prefetching across the epoch boundary would remove it.
  - The frozen-trunk phase (updates <100) and the unfrozen phase run at the same speed (38.6 vs 39.0 s).
- **Q (user): what extra work would compiled eval need for the head-native canvas?**
  - **Corrected (mine):** "BFL's compiled model only supports the DROID canvas, so head-native needs extra work".
    The restriction is the action head, not the canvas. `_released_inference_config` (flux-action
    policy.py:55-66) requires `action_prediction_droid`, action_dim 8 and chunk 32. Canvas and layout aren't
    checked. The compiled inference DiT (`BF16InferenceDiT`) hard-codes the DROID action stream with 8 channels
    (transformer_inf_bf16.py:34-37).
  - Both our runs (main and head-native) use `action_prediction_aloha` with 14-D actions, so neither can use it.
    `prepare_inference` returns before compiling anything, encoder included (policy.py:276-277). Our adapter
    already notes this (XPolicyLab Flux3_Action/model.py:127).
  - So compiled eval means the same work for both runs. Two options:
    - compile the generic DiT and the encoder in our adapter;
    - port BFL's inference DiT to the 14-D head.
- **Q (user): would compiling the eval policy speed up evals?** No: the simulator dominates. From the 6,500
  eval's logs: 72.9 GPU-h in total.
  - Startup, per-queue-line model loads (26 s each) and idle tails: 7.6 h (10%). Episodes: 65.3 h (90%).
  - Policy calls: 15,833 at a median 786 ms, ~3.5 h, 5.3% of episode time. The rest, ~0.46 s per executed
    action (~62 h), is simulator stepping, rendering and client work. The logs don't split it further.
  - A 2× faster policy call would save ~1.7 GPU-h per eval (~2.4%).
  - Failures run to the step limit: median 252 s vs 83 s for a success. The 25% of episodes that fail take
    ~46% of episode time.
  - **Open:** with one stream per GPU, how busy is the GPU during an episode? Not measured for Flux. If the
    GPU is mostly idle, two streams per GPU could cut eval GPU-hours.
- **Found (v2eq, cluster, 8 GPUs, 4 per GPU, 5 updates, 80 micro-batches hashed; timing from 2 updates, rough):**
  - Exact v2 (overlap + graph + no padding + no autocast + RoPE cache): DiT inputs bitwise 80/80, update-1 loss
    bitwise, later within noise; ~43.8 s (≈1.20×).
  - With gradient sharding added: also exact; ~44.5 s, so at 4 per GPU sharding doesn't help.
  - Compiled stack: loss 6.5e-5 at update 1 and ≤9.1e-5 later, about 4× the noise; ~38.7 s (≈1.35×).
  - Proper 20-update timings come from v2exact and cmpmain.
- **Q (user): without the 4,790 spike, would the main run be much better by now?** Not known; evidence so far:
  - Held-out arm error at 4,000 (before the spike) is the same as at 7,000 (0.0299/0.0216/0.0231 vs
    0.0297/0.0220/0.0245). Within this run the check has tracked eval success (5,000 → 6,500). If that holds for
    4,000, the run was at about today's level then, and spent ~2,500 updates recovering. That's an inference: 4,000
    never had a full eval, and the check rests on 6 points.
  - Direct test: a full eval of the 4,000 export, which exists.
  - Smaller spikes also hit at 3,360, 4,350 and 6,460. The spike investigation's most likely cause (moderate
    confidence): instability at constant peak LR with gradient clipping off.
  - Our configs follow BFL's DROID config (`max_grad_norm: null`). Their SO-101 config clips at 1.0, which is also
    the trainer default. Our ordinary grad norms sit around 0.7–1.5, so clipping at 1.0 would also clip ordinary
    updates. Look at the grad-norm distribution before choosing a threshold for a new run.
  - Schedule (`train_b512_cont6k.json`): constant peak LR until 8,000, then a 1,000-update cooldown to 9,000.
    Cosmos gained most from its decay, so 9,000 is the main run's verdict.

---

## Lessons
- **Check before stating.** Several confident statements about the run were wrong until checked (sections
  5, 6, 8). State which claims are measured and which are assumed.
- **Small questions pay off.** Questions about speed, odd numbers or "why" have found bugs (save markers,
  eval startup), wrong beliefs (recomputation, wrist size, overfitting) and real causes (the spike, EMA
  mixing).
- **Keep one change per experiment,** and one recipe per run, so results can be traced to a cause.

## Checklist before any long training run (Flux, Cosmos, or any other model)
Built from the findings above. Check each item and write down the answer before launching.
1. **Inputs vs cameras:** does the input canvas or resolution upscale or stretch the camera images? How many
   pixels carry real information?
2. **Loss balance:** how is the loss combined (per token or per modality)? How do the weights depend on token
   counts? What share does each part get?
3. **Memory:**
   - What actually uses it: weights, optimizer, full gradient copies during accumulation, activations, encoders.
   - Is activation recomputation on?
   - Is gradient sharding (ZeRO-2) used?
   - What is the encoders' peak?
4. **Speed:**
   - Run a ~20-update benchmark with per-phase timing (encoder, forward, backward, communication, optimizer,
     data wait) on the target hardware.
   - Find the bottleneck before choosing examples per GPU and node count.
   - Note padding or recomputation that does no useful work (e.g. black frames padded before the video encoder).
5. **Stability:**
   - Clip gradients or not.
   - Peak learning rate vs batch size compared with the reference recipe, and the re-warm peak when continuing
     from a cooled-down checkpoint.
   - Log every update.
   - Keep a spare full save when a spike shows up, so it can be replayed.
6. **Exports and evals:**
   - Which weights get evaluated (averaged vs trained).
   - How long the averaging window is, and how a spike inside it shows up in an eval.
7. **Storage and restarts:**
   - What the job can and can't delete.
   - Save slots; a resume test and an in-job restart test on the real cluster topology.
8. **Records:** the exact commit, config, data fingerprint, seeds and topology for every segment of the run.

**Mid-run changes:**
- Recipe changes: never.
- Exact or equivalent computation changes: only if the gain is worth a restart. Then test on a copy, switch
  right after a complete save while keeping the previous one, and check the first updates' loss.
