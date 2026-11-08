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
  - Probably not much. Its input is already close to the cameras' native size, so upscaling adds no
    information.
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
