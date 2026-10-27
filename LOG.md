# Work log (RoboTwin / GR00T)

Plain-language, timestamped log of what was done, what the evals returned, and why
each decision was made. Newest entries at the bottom. Times are UTC up to the
"Times switch to ET" entry, New York time (ET) after it. Technical
details for agents live in `AGENTS.md`.

## Results so far

Model **5task_30k** (one GR00T N1.7 model trained on 5 tasks × 50 clean demos,
30k steps), checkpoint-30000. Success rate over 100 episodes per task (4 shards × 25).

| task | demo_clean | demo_randomized |
|---|---|---|
| beat_block_hammer (one arm) | 47% (47/100) | running (GPU 4) |
| lift_pot (two arms) | 18% (18/100) | running (GPU 7) |
| open_laptop (one arm) | running: 13/18 so far | queued |
| stack_blocks_two (two arms) | running: 6/91 so far | queued |
| handover_block (two arms) | running: 1/77 so far | queued |

Pattern: the one-arm tasks score much higher than the two-arm tasks. The open-loop
check (below) will tell whether that's a pipeline bug or the policy being weak.

## 2026-10-27

**01:19** Status check. demo_clean: beat_block_hammer 47/100 and lift_pot 18/100
are done. Running: open_laptop (GPU 1), stack_blocks_two (GPU 6),
handover_block (GPU 5). The 5 demo_randomized jobs are queued.

**01:20** Turned off Claude Code's idle compaction (`"idleCompaction": false` in
`~/.claude/settings.json`), as you asked.

**01:25** Decision (you): five tasks is too few. Scores vary from 1% to 72% by
task, so a 5-task average mostly reflects which tasks were picked. Next model: **one
GR00T model trained on all 50 RoboTwin tasks** (50 clean demos each, the standard
RoboTwin set). Training steps doubled to 60k, because each task gets 10× less
training per step. Note: the current model is also one model on all 5 tasks (the
instruction text tells it which task), not one model per task.

**01:26** Checked that RoboTwin's Hugging Face repo has 50 clean demos for all 50
tasks (23.4 GB total). We download those instead of generating demos with the
scripted expert.

**01:27** Started downloading the 45 new tasks' demos (CPU and network only).
Log: `logs/download_50task.log`.

**01:28** Your answers:
- Training gets priority over the old model's demo_randomized evals. I may stop one
  in-progress randomized job when the 50-task data is ready, and re-queue it.
- GPU 0 can be used while it's free. (At 01:28 another user took GPU
  0, so I'm leaving it alone.)
- The eval-bug check runs while you're away; we review the results together.
- Any GPU that's free can be used, for training (more GPUs = faster), evals, or both.

**01:29** GPUs 4 and 7 were free, so I started two more eval workers on them. They
took the next queued jobs: demo_randomized beat_block_hammer (GPU 4) and lift_pot
(GPU 7). Each takes ~3–8 h. Our GPUs are now 1, 4, 5, 6, 7.

**01:30** Made the scripts work for any task set (`gr00t/prepare_data.sh` takes
`TASKS`/`SRC`/`CKPT_NAME`; `gr00t/train.sh` takes `DATASET`). The defaults are
unchanged, so the 5-task commands still work.

**01:31** Started the **open-loop bug check** in a background agent (GPU 4 or 7,
small memory, next to the evals):
- Feed demo frames through the same code the eval uses, and compare the model's
  predicted next-16 actions with what the demo actually did.
- Compare against a "hold still" baseline. If the model's error is far below that
  baseline on every task, the pipeline is wired correctly, and the low two-arm scores
  come from the policy, not a bug.
- Also compares the images the model sees at eval with the training images
  (resolution, colour order).
- Outputs go to `outputs/open_loop_check/`.

Download progress at 01:31: 38/50 tasks.

**01:33** Download finished: all 50 tasks × 50 clean demos (33 GB, every task has
exactly 50 episodes).

**01:36** Started converting the 50-task data into GR00T's format
(`logs/prepare_50task.log`). The old one-task-at-a-time converter would take ~6 h
here, judging by the 5-task run's 2–11 min per task. So `gr00t/convert_parallel.py`
converts 12 tasks at a time into separate datasets and then merges them with
LeRobot's own `aggregate_datasets` tool. The steps after that are the same as for
5 tasks: downgrade to the format GR00T reads, compute normalization stats, then
`check_dataset.py` decodes every video and checks one frame per step and 2,500
episodes. Output dataset: `data/lerobot/RoboTwin-50task-aloha_agilex-joint`.

**01:40** Freed GPUs for training without killing anything. I moved the 3
not-yet-started demo_randomized jobs (handover_block, stack_blocks_two, open_laptop)
from `logs/queue_30k.txt` to `logs/queue_30k_held.txt`. When the clean evals on
GPUs 5 and 6 finish, their workers find the queue empty and exit, which leaves those
GPUs free for the 50-task training. The held jobs go back in the queue once training
has its GPUs. Still running: GPU 1 open_laptop clean, GPU 4 beat_block_hammer
randomized, GPU 5 handover_block clean, GPU 6 stack_blocks_two clean, GPU 7
lift_pot randomized.

**01:40** Conversion speed: ~3.5–7 min per task with 12 at once; the first 3 tasks
finished at 01:38 with no errors or read retries. While it runs, the machine is very
busy (load ~200 on 192 cores), which may briefly slow the evals.

**02:08** **Bug check result: no bug found.** (Background agent; scripts
`gr00t/open_loop_check.py` and `gr00t/live_obs_check.py`, commit 4cf9226; outputs
in `outputs/open_loop_check/`.)
- Test: 150 demo frames (3 demos × 10 moments × 5 tasks) went through the eval
  client's own conversion code, the websocket encoding and the GR00T adapter. The
  model's next-16 predicted actions were compared with what the demo actually did
  next, and with a "hold still" guess.
- The model is 3–5× closer to the demo than hold-still on every task. The two-arm
  tasks are **not** worse than the one-arm tasks, and the left and right arms are
  about equally accurate. A wiring bug (cameras, state order, colours, action format)
  would push the model's error up to the hold-still level.

  | task | arm error, model / hold still (rad) | gripper error, model / hold still |
  |---|---|---|
  | beat_block_hammer | 0.012 / 0.066 | 0.005 / 0.036 |
  | lift_pot | 0.019 / 0.091 | 0.012 / 0.081 |
  | handover_block | 0.014 / 0.054 | 0.014 / 0.053 |
  | stack_blocks_two | 0.013 / 0.054 | 0.011 / 0.019 |
  | open_laptop | 0.013 / 0.035 | 0.005 / 0.026 |

  (0.01 rad ≈ 0.6°. Gripper: 0 = closed, 1 = open.)
- State fed at eval = training state exactly. Images fed at eval = training video
  frames up to compression noise (same size, same colour order; PNGs
  `outputs/open_loop_check/images_<task>.png`).
- The live simulator, reset to a demo's seed, gives the same image and joints as
  that demo's first frame (`live_vs_demo_*.png`).
- Calling the model 5× on the same frame changes its prediction by only ~0.2°, so
  its built-in randomness is small.
- Where it goes wrong: decision moments in handover_block, e.g. when to let go
  with the left gripper, or when the right arm should start moving. That matches the
  eval videos.
- Conclusion: the low two-arm scores come from the policy (small mistakes at key
  moments add up once the robot acts on its own predictions), not from the eval
  pipeline.
- Caveat: these were training demos, so this checks the wiring, not generalization.
- Side finding (possible hill-climb idea): training used one random instruction
  sentence per demo, so the model saw only ~50 phrasings per task, and at eval it
  gets new phrasings. Feeding the demo's own training sentence lowers arm error
  15–35%. Training with all instruction variants per demo might help a little.

**02:10** Result: **stack_blocks_two demo_clean = 7/100 (7%)**. GPU 6 is free and
waiting for the 50-task training. All 50 tasks converted with no errors; merging
them now. Made `check_dataset.py` decode videos in parallel (5-task check: 9 s, 0
mismatches, same as before).

**02:45** **50-task dataset ready:** `data/lerobot/RoboTwin-50task-aloha_agilex-joint`.
2,500 episodes, 548,893 frames (10.7× the 5-task set), 2,413 distinct instruction
sentences, episodes 74–894 steps long. Every video checked: one frame per step, 0
mismatches. Prep took ~70 min in total (download 6 min, conversion ~30 min, merge
~13 min, the rest stats and checks).

**02:52** Someone else (a root-owned vLLM server, started ~02:31) took **GPU 6**
while it sat idle waiting for the data. It's not ours, so I'm leaving it alone.

**02:55** Training GPUs, per your "training first" OK:
- **GPU 5**, once its handover_block clean eval finishes (every shard is on its
  last episode).
- **GPU 7**, by stopping its lift_pot demo_randomized job, the one approved
  preemption. I picked lift_pot because it had the least progress (~15 episodes,
  0 successes so far). bbh randomized on GPU 4 had 4/23 and keeps running. The
  lift_pot job goes back to the hold file and its partial results are discarded
  (logs kept in `logs/preempted/`).
- A detached watcher (`scratch/start_50task_train.sh`, log
  `logs/start_50task_train.log`) does this the moment GPU 5 frees, so nobody grabs
  GPU 5 in between. If someone else takes GPU 5 first, it leaves GPU 7 alone.
- Training recipe: same as the 5-task run (GR00T defaults, global batch 64, lr 1e-4),
  except **60k steps** instead of 30k, and `OMP_NUM_THREADS=8`. The last run was
  CPU-bound with 1 thread per GPU process; I'll compare steps/s against it.
  Checkpoint every 5k steps; the last 5 are kept.
  Log: `logs/train_50task_60k_seed0.log`.

**03:05** handover_block demo_clean finished: **3/100**. The watcher stopped
lift_pot randomized on GPU 7 (0/17 so far; re-held) and started training on GPUs
5+7.

**03:10** Decision (you): **train on 4 GPUs instead of 2** (~7–8 h instead of ~14 h).
Stopped the 2-GPU training (it was still loading, so nothing lost) and the two
old-model evals on GPUs 1 and 4. Their partial results are kept
(`logs/preempted/partial_counts_0310.txt`, logs and videos in `logs/preempted/` and
`outputs/preempted_eval/`):
- **open_laptop demo_clean: 33/50 = 66%.** This is its result for now (±13%). It
  can be topped up to 100 later with new shards on new seeds.
- beat_block_hammer demo_randomized: 4/30 so far. Re-held, to resume later the
  same way.

**03:12** **Training started on GPUs 1, 4, 5, 7**: `RUN=50task_60k`, one model on all
50 tasks, 60k steps, global batch 64 (16 per GPU), lr 1e-4, checkpoint every 5k
steps, `OMP_NUM_THREADS=8`. Log: `logs/train_50task_60k_seed0.log`. ETA ~10:30–11:30
UTC after ~15 min of startup. All 4 of our GPUs are on training now; the old-model
evals wait until it finishes.

Old-model evals still to do: demo_randomized for all 5 tasks (bbh has 30 episodes,
lift_pot's 17 were discarded) and the open_laptop clean top-up. Queue:
`logs/queue_30k_held.txt`.

**03:30** Why training startup is slow, and a fix for next time:
- Before training, GR00T checks every video's frame count with `ffprobe`, so it
  never reads a frame that doesn't exist. Every GPU process does this separately,
  with no cache: 4 × 7,500 videos, ~20+ min on the shared disk, with the GPUs at 0%.
- Fix (your call): GR00T now reads the counts from a table
  (`meta/video_frames.json`) that `check_dataset.py` writes once during data prep.
  GR00T falls back to ffprobe if the table is missing.
- Tested on the 5-task data: identical episode lengths, 69.5 s → 0.1 s.
- Commits: GR00T side `88f9e50` (XPolicyLab branch `local/aloha-gr00t`); our side
  is `check_dataset.py`. The tables now exist for both datasets.
- The current run started before the fix and keeps its old code. Its data workers
  are forked copies, so they don't reload the file. Future runs get the fast path.

**03:30** Multi-GPU method: GR00T uses **DeepSpeed ZeRO stage 2** automatically
with more than one GPU. Each GPU has the full model and a different slice of each
batch; the optimizer state and gradients are split across the GPUs to save memory.

**03:39 UTC** You asked about someone taking our GPUs. At 03:39 only our 4 training
processes were on GPUs 1/4/5/7. They had looked idle (0%, 1.5 GB) for ~25 min during
GR00T's startup, which is probably why. Lesson saved: on this machine any GPU that
looks idle for a few minutes gets taken, so the goal is ~100% utilization with no
gaps between jobs.

### Times switch to ET (New York, EDT = UTC−4) from here, as you asked.

**23:42 ET (Oct 26)** **Training is stepping.** GPUs 1/4/7 at ~100%, loss 1.05 → 0.35
in the first 300 steps. Speed **2.62 steps/s** on 4 GPUs (the 5-task run on 2 GPUs
averaged 1.22). ETA: 60k steps ≈ 6.3 h plus checkpoint saves, so done ≈ **06:15–06:45
ET**.

**23:45 ET** Armed the hand-off, so the GPUs go straight from training to evals with
no idle gap (`scratch/after_50task_train.sh`, log `logs/after_50task_train.log`):
- When training exits, each of GPUs 1/4/5/7 starts an eval worker (30 s apart).
- First the **50-task screening**: checkpoint-60000, 20 episodes per task (4 shards
  × 5), clean scenes, 50 tasks (`logs/queue_50task.txt`; place_empty_cup last
  because it renders very slowly).
- Then the old 5-task model's remaining evals (`logs/queue_30k.txt`): open_laptop
  clean top-up (+50 eps on new seeds), handover/stack/open_laptop randomized (100
  each), beat_block_hammer randomized top-up (+70), lift_pot randomized top-up (+83).
- If training fails and there's no checkpoint-60000, the workers run only the
  old-model evals, so the GPUs still stay busy.
- Script changes for this: queue lines can set episode count and seed offset
  (top-ups use new seeds, so nothing is double-counted); `summarize_eval.py` also
  counts the stopped shards' partial results (`logs/preempted/`); the per-GPU start
  gap is configurable.

Old model, results so far, including partials: clean mean **28.2%** (beat_block_hammer
47, lift_pot 18, stack_blocks_two 7, handover_block 3, open_laptop 66 on 50 eps);
randomized: beat_block_hammer 4/30, lift_pot 0/17.

**23:47–23:52 ET** Found why training was slower than expected:
- After a fast start (2.6 steps/s on cached data), speed swung between ~1 and 2.6,
  averaging ~1.8, so ~9 h, not the 6.5 h I first said (I had quoted a peak).
- The GPUs had plenty of free memory (21 of 80 GB) but kept **waiting for data**:
  each GPU's 8 data-loading workers were pegged at ~93% CPU, and the shared machine's
  CPUs were ~98% busy (load 188 on 192 cores).
- Video decoding itself turned out to be cheap (an 894-frame video decodes in
  0.25 s; decoding only the needed frames barely helps). Most worker time is
  per-sample image processing, and the CPU share we get on the busy machine.
- ffprobe also ran *during* training (3 per episode load); the table fix removes
  that too.

**23:53 ET** Restarted training (your OK) with **16 data workers per GPU** (64 in
total; RAM ~580 GB of 1.58 TB free) and the ffprobe-free loader. 32 per GPU would
need ~1.15 TB of RAM on a shared machine, too risky. The first attempt's ~15 min of
steps are discarded (no checkpoint yet); its log is
`logs/train_50task_60k_seed0.attempt1.log`. Dataset init took **4.7 s** (first
attempt: ~25 min). Training stepped again at **23:58 ET**, 5 min after the restart.
The hand-off watcher was stopped before the restart and re-armed after it.

**23:59 ET** Another user started a vLLM server on **GPU 1** (30 GB) next to
our training rank (22 GB). Both fit in the 80 GB, so training continues there (your
call). Risk: the 4 training GPUs move in lockstep, so a busy vLLM would slow the
whole run; I'm measuring the speed.

**00:04 ET** **GPU 6 came free and is ours again**: an old-model eval worker
(`logs/queue_30k_gpu6.log`) is on it, starting with the open_laptop clean top-up (+50
episodes, shards 4–7), then the randomized evals from `logs/queue_30k.txt`.

**00:10 ET** Speed after the restart: **~1.8 steps/s, no better** than the first
attempt (~1.9). The machine's CPUs are oversubscribed (load 200–256 on 192 cores),
so doubling our workers only got ~24% more CPU time; each worker now runs at ~57% of
a core. Each training sample costs ~0.34 CPU-seconds to prepare, and video decoding
is only ~2–3% of that. ETA at 1.8 steps/s: **~09:15 ET**.
Started a background agent to profile GR00T's per-sample processing (CPU only, no
GPU, training untouched) and make it cheaper without changing what the model trains
on. If it finds a big win, I'll ask you before restarting from checkpoint-5000
(due ~00:50 ET), so almost nothing is lost.
