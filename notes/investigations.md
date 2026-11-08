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
