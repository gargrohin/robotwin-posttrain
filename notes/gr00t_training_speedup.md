# Making GR00T training ~1.5× faster: the data-loading investigation

2026-10-26/27, during the 50-task training run (`RUN=50task_60k`, 4 GPUs). Times are
US Eastern.

**Summary.** The training GPUs kept waiting for data. Preparing training samples on
the CPU was the bottleneck, not the GPUs. After two rounds of fixes:

| | before | after |
|---|---|---|
| time from launch to the first training step | ~30 min | ~5 min |
| steady training speed | ~1.75 steps/s (swinging 1.0–2.6) | **2.63 steps/s** |
| CPU to prepare one training sample | 114 ms | **32 ms** |
| 60k-step run | ~9.3 h | ~6.3 h |

The model still trains on exactly the same samples. Outputs were verified
bit-identical (see "Verification").

## Background: how GR00T feeds the GPUs

- **One training process per GPU** (DeepSpeed ZeRO-2). Each process has its own copy
  of the dataset and its own **dataloader workers**: separate CPU processes, forked
  from it, that build training samples in the background. The global batch is 64,
  i.e. 16 samples per GPU per step.
- **Data path for one sample** (`gr00t_n17/gr00t/data/dataset/`):
  1. **Shards.** `ShardedSingleStepDataset` splits every episode's timesteps into 10
     random subsets (`episode_sampling_rate 0.1`), each in a different shard of ~1,024
     samples. That mixes many episodes into every batch.
  2. **Episode load** (`LeRobotEpisodeLoader`). Read the episode's parquet file (joint
     states and actions) and decode its 3 camera videos (AV1, 240×320, 15 fps).
  3. **`extract_step_data`.** Pick out one timestep: 3 images, state, the next 16
     actions, the instruction.
  4. **Processor.** Relative actions and normalization, image augmentation
     (albumentations colour jitter), conversion to PIL images, chat template.
  5. **Collator, per batch.** The Qwen3-VL image processor (the backbone's own image
     preprocessing) and the tokenizer.
- All 4 GPU processes take a step together. If any one waits for data, all of them
  wait.

## Problem 1: 25 minutes of idle GPUs at startup (ffprobe)

**Symptom.** After launch, the GPUs sat at 0% for ~25 min in "Initializing
datasets". On a shared machine an idle-looking GPU invites other users to take it.

**Cause.** `LeRobotEpisodeLoader._get_synchronized_episode_length` makes sure training
never asks for a video frame that doesn't exist. For each episode it compares the
episode length with each video's frame count and uses the smaller one. It gets the
frame count by running `ffprobe`, a separate program, on the video file:
- once per video (2,500 episodes × 3 cameras = 7,500), in **each** of the 4 GPU
  processes, over the slow network disk;
- and, as it turned out, **again inside every episode load during training**,
  3 ffprobe runs per load (4–12 were running at any moment).

**Fix** (XPolicyLab `88f9e50`, `patches/xpolicylab/0002`). If the dataset has
`meta/video_frames.json` (video path → frame count), the loader reads the count from
it. Otherwise it falls back to ffprobe. `gr00t/check_dataset.py`, which already opens
every video to check it, writes the table from each video's header: the same number
ffprobe reports.

**Result.** 5-task dataset: 69.5 s → 0.1 s with identical episode lengths for all 250
episodes. 50-task dataset init: ~25 min → **4.7 s**.

## Problem 2: training waits for data

**Symptom.** Once training started, speed swung between ~1.0 and 2.6 steps/s and
averaged ~1.8. It ran fast while workers had prepared shards ready, then stalled. The
GPUs used only 21 of 80 GB each, and their compute utilization jumped between ~10%
and 100%. Each GPU's 8 dataloader workers were pegged at ~93% CPU, and the whole
machine was at load 188–256 on 192 cores (other users' jobs plus ours).

**First attempt: more workers (didn't help).** Restarted with 16 workers per GPU
instead of 8 (64 in total, ~9 GB RAM each). Speed went from ~1.9 to ~1.8: no gain.
Each worker now ran at only ~57% of a core. The machine had no spare CPU to give, so
more workers just split the same share.

**A wrong estimate on the way.** I timed decoding one 894-frame video in a single
process: 0.25 s. From that I concluded decoding was only ~2–3% of the cost. That was
wrong. I measured **wall time**, but the AV1 decoder (libdav1d) quietly used many
threads, so its **CPU time** was ~5× higher. When the question is "how much CPU does
this cost", measure CPU time (`time.process_time()`), not wall time.

## Profiling: where the CPU actually went

A background agent rebuilt the training dataset exactly as `launch_finetune.py` does,
in a CPU-only process. It measured CPU time per stage for 2 shards (2,061 samples),
with the same thread settings as the real workers. CPU ms per sample:

| stage | before | after |
|---|---|---|
| **video decode + RGB conversion** | **75.3** | **3.9** |
| parquet load + joint split | 0.5 | 0.6 |
| `extract_step_data` | 1.1 | 1.1 |
| relative actions + normalization | 0.7 | 0.7 |
| albumentations (3 views, incl. colour jitter) | 7.4 | 6.2 |
| PIL conversion + chat template | 3.9 | 0.5 |
| collator: Qwen3-VL image processor | 16.8 | 17.0 |
| collator: tokenizer | 5.3 | 0.5 |
| **total** | **113.8** | **32.2** |

Video decoding was two-thirds of the cost, for two reasons:
1. **Thread explosion.** By default, FFmpeg's libdav1d decoder starts about one thread
   per CPU core, ~192 here. On small 240×320 frames that burns ~5× the CPU of one
   thread for no speed gain. Every one of the 64 workers did it.
2. **Decoding ~10× more frames than used.** Because of the shard split, each shard
   needs only 1/10 of an episode's timesteps. But each load decoded, RGB-converted and
   stacked **every** frame of all 3 videos. Over an epoch, each frame was decoded 10
   times.

## The fixes

One commit each, on XPolicyLab's local branch `local/aloha-gr00t`
(`patches/xpolicylab/0003`–`0008`):

| commit | change | why |
|---|---|---|
| `ca02cce` | PyAV decodes single-threaded and converts to RGB only the frames it returns | stops the thread explosion; skips converting unused frames |
| `0acfe8d` | new `LeRobotEpisodeLoader.load_episode(idx, step_indices, allow_padding)`, used by `get_shard`, decodes only the rows `extract_step_data` reads; other rows hold `None`. `__getitem__` still decodes everything | ~10× fewer frames |
| `53d2c40` | start decoding at the last keyframe before each needed frame (these videos have a keyframe every 2 frames) instead of from the start of the video | skips the frames in between. Fast path only when packet timestamps and keyframes look normal; otherwise full sequential decode as before |
| `9185684` | build PIL images with `cv2.merge` instead of making PIL copy a transposed array | ~20× faster, same pixels |
| `dc36f02` | albumentations: replay the drawn augmentation on the 2nd and 3rd camera without rebuilding the pipeline | same augmentation, less overhead |
| `39385cc` | `train.sh` sets `TOKENIZERS_PARALLELISM=false` | HF tokenizers started a per-core thread pool in every worker: 5.3 ms CPU for 0.2 ms of work |

## Verification: the model trains on exactly the same data

- **Training samples.** Built the dataset through `launch_finetune.py`'s own code path
  with the same arguments and seeded `random`, `numpy` and `torch` per shard. Then
  fingerprinted every processed sample (state, action, action mask, text, a hash of
  each image) and the first collated batches in full (`pixel_values`, `input_ids`,
  `attention_mask`). Before vs after, all bit-identical:
  - with augmentation: 2,061 samples and 8 batches;
  - with augmentation off: 1,013 samples and 4 batches.
- **Video decoding.** Old vs new decode functions gave bit-identical frames on 60
  random videos, each with 4 index sets: all frames, a 10% subset, unsorted with
  repeats, and indices past the end of the video.
- **Augmentation replay.** Identical images over 300 seeds, and identical random-number
  state afterwards.

## Result in the real run

Restarted at 00:51 ET. GR00T resumes from the latest checkpoint automatically
(`trainer.train(resume_from_checkpoint=True)`), so it continued from
checkpoint-5000 and lost only ~130 steps. Measured over 8 minutes (01:02–01:10 ET):
**2.63 steps/s**, with dataloader workers idling at ~11% CPU each (was 57–93%).
Expected finish ≈ 07:00 ET instead of ≈ 09:30.

## Data-prep speedups, same night

- **50-task conversion to LeRobot format** (`gr00t/convert_parallel.py`). Converting
  one task after another would take ~6 h (2–11 min per task). Instead it converts 12
  tasks at a time into separate datasets and merges them with LeRobot's
  `aggregate_datasets`: ~30 min, and ~70 min for the whole prep including merge,
  stats and checks.
- **Dataset check** (`gr00t/check_dataset.py`). Decodes videos with 32 processes:
  7,500 videos in ~70 s.

## What's left

- **GPU compute is now the limit, at ~30–97% utilization (avg ~55%).** With 16
  samples per GPU, each step's fixed CPU-side overhead in the training process
  (Python, kernel launches; its autograd thread ran at ~94%) is a big share of the
  step. A bigger per-GPU batch would amortize it, but batch size is a training
  hyperparameter, so it's a hill-climbing experiment, not a free speedup.
- **Qwen3-VL image processor: 17 ms per sample, half of what's left.** It runs on
  CPU in the collator (float conversion, frame duplication, patchify over ~75 MB per
  batch). Moving it to the GPU would remove most of it, but that changes model and
  inference code.
- **Albumentations: 6 ms.** Moving it to the GPU or batching it would change the
  random draws, i.e. the training data.
- **Gotcha.** GR00T rewrites `meta/relative_stats.json` at every startup, so two runs
  starting at the same time on the same dataset can read it mid-write.

## Files

- Patches: `patches/xpolicylab/0002`–`0008` (this repo).
- Profiling and equivalence scripts with their logs: `scratch/dataloader_profiling/`
  (local only, git-ignored). `run_profile.py` gives the per-stage table;
  `equiv_dump.py` / `equiv_compare.py` do the bit-identical check.
- Training logs: `logs/train_50task_60k_seed0.attempt1.log` (8 workers, old code),
  `.attempt2.log` (16 workers, old code), `train_50task_60k_seed0.log` (fixed code).
  Local only.
