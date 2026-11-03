# Work log (RoboTwin / GR00T)

Plain-language, timestamped log of what was done, what the evals returned, and why
each decision was made. Newest entries at the bottom. Times are UTC up to the
"Times switch to ET" entry, New York time (ET) after it. Technical
details for agents live in `AGENTS.md`.

## Results so far

**Full, current per-task results for both models: [`notes/results.md`](notes/results.md).**
The table below is the 5-task model only, from early on.

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
`gr00t/open_loop_check.py` and `gr00t/live_obs_check.py`, commit 72beb1d; outputs
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

**00:44 ET** **Profiling result (background agent): the real CPU hog was video
decoding**, contrary to my earlier estimate:
- Each data worker's AV1 decoder started ~192 threads (one per core); with 64
  workers that flooded the machine.
- Workers decoded ~10× more frames than training uses.
- Fixed in 6 commits on our local GR00T branch (`ca02cce`, `0acfe8d`, `53d2c40`,
  `9185684`, `dc36f02`, `39385cc`): CPU per training sample **114 → 32 ms (3.5×
  cheaper)**; video decode 75 → 4 ms.
- Verified the training samples are **bit-for-bit identical** before vs after
  (2,061 samples + 8 batches, seeded, with and without augmentation).
- Training was at step 5,130 (~1.75 steps/s, load ~250) with checkpoint-5000 saved.

**00:51 ET** Restart (your OK). My first restart command killed only itself (the
process search matched its own command line, the known `pkill` pitfall); nothing
else was affected. Second attempt: training relaunched at 00:51 ET on GPUs 1/4/5/7.

**00:53 ET** GPU 0 became completely free while GPU 1 had the other user's vLLM growing to 45 GB,
so (your suggestion) moved training to **GPUs 0, 4, 5, 7**. It was still loading,
so nothing was lost. GR00T resumes automatically from the latest checkpoint
(checkpoint-5000). The hand-off watcher was updated to GPUs 0/4/5/7 and re-armed.
GPU 1 is back to the other user alone.

**01:10 ET** **The fix worked.** Training resumed from checkpoint-5000 (stepping from
01:00 ET) and measured **2.63 steps/s** over 8 minutes (was ~1.75). Data workers now
idle at ~11% CPU each (were 57–93%), so training is no longer waiting for data.
Remaining 53.4k steps ≈ 5.6 h plus checkpoint saves, so done ≈ **07:00 ET**. GPU
compute utilization is ~30–97% (avg ~55%): the per-GPU batch of 16 is small and each
step has fixed CPU-side overhead. A bigger per-GPU batch would push that up, but it
changes the training recipe, so it's an option for the next run, not this one.

**01:45 ET** Put this repo on GitHub, as you asked:
**https://github.com/gargrohin/robotwin-posttrain** (public), pushed with the gargrohin
account only.
- Before pushing: scanned every file and the whole git history. No secrets
  (tokens/keys) anywhere. Work-internal details were replaced with placeholders
  throughout the history: the company pip-mirror account details, other users'
  names, shared-machine notes.
- All commits are re-authored as gargrohin's GitHub noreply address, so no work email
  shows.
- Your MIT LICENSE commit was merged in, not overwritten.
- `patches/` now holds every line we changed in upstream code (XPolicyLab/GR00T: 8
  commits; cuRobo: 1). The README has a "what's ours vs upstream" table. Our own code
  is `gr00t/` and `setup/`; everything else (RoboTwin, XPolicyLab, GR00T, cuRobo) is
  upstream.
- Commit hashes in this repo changed with the rewrite (e.g. the open-loop check is
  now `72beb1d`).

**02:00 ET** Wrote up the training-speed investigation in depth:
[`notes/gr00t_training_speedup.md`](notes/gr00t_training_speedup.md). It covers the
ffprobe startup problem, why more workers didn't help, the profiling table, each of
the 6 data-loading fixes, how they were verified, and what's left. The cuRobo H100
crash is written up in [`notes/curobo_h100_lbfgs_bug.md`](notes/curobo_h100_lbfgs_bug.md).
The profiling scripts are preserved locally in `scratch/dataloader_profiling/`.

## 2026-10-27 (morning)

**07:36 ET** **50-task training finished**: 60k steps in 6 h 37 min (2.52 steps/s on
average, including checkpoint saves), final train loss 0.0096 (run average). Kept
checkpoints 40k/45k/50k/55k/60k. The hand-off watcher started eval workers on
GPUs 0/4/5/7 within seconds, so there was no idle gap.

**02:42 ET (overnight)** Old model, **open_laptop demo_clean finished: 68/102 = 67%**
(the 50 earlier episodes plus the 52-episode top-up). Old-model clean mean over the 5
tasks: **28.3%**. GPU 6 then started handover_block demo_randomized.

**09:35 ET** **50-task screening** (checkpoint-60000, 20 episodes per task, clean):
- Done (6 tasks): adjust_bottle 95%, beat_block_hammer 45%, click_alarmclock 100%,
  click_bell 95%, dump_bin_bigbin 75%, grab_roller 90%.
- Running: blocks_ranking_rgb 3/14, blocks_ranking_size 1/12, handover_mic 4/4,
  handover_block (just started).
- 40 tasks still queued. Some tasks allow much longer episodes (blocks_ranking 1,200
  steps, put_bottles_dustbin 1,700, vs 400 for most), and failures run to the limit,
  so those take hours.
- beat_block_hammer is 45% vs 47% for the 5-task model, so far no sign that
  training on 50 tasks hurt it (20 vs 100 episodes, so ±22% vs ±10%).

Old model, handover_block demo_randomized (GPU 6): **0/68** so far.

**09:50 ET** Queued **randomized-scene evals for the 50-task model** (your OK): 50 tasks
× 20 episodes, appended to `logs/queue_50task.txt` after the remaining clean ones.
GPUs 0/4/5/7 run them before the old model's leftovers; GPU 6 keeps doing the old
model's list. Expect another ~12+ h after the clean screening, because failures run
to the step limit and randomized scenes fail more.

**10:20 ET** GPU 6 (old model) still has ~30 h of randomized evals queued: handover
~3.5 h left, then stack_blocks_two, open_laptop, the beat_block_hammer and lift_pot
top-ups, all mostly failures running to 400–800 steps. Fixed a gap: when that list
empties, GPU 6's worker would have exited and left the GPU idle. A chain now moves
GPU 6 straight into the new model's queue (`logs/queue_chain_gpu6.log`). Suggested
trimming the old model's randomized evals to ~20 episodes per task (pending your
call).

**10:30 ET** Decision (you): **old model randomized evals trimmed to ~20 episodes per
task** (old queue saved as `logs/queue_30k.before_trim.txt`). What's left on GPU 6:
handover_block (running, already 68 episodes), stack_blocks_two 20, open_laptop 20,
lift_pot +4 (17 → 21). beat_block_hammer is dropped because it already has 30. GPU 6
should finish the old model around 19:00–20:00 ET instead of tomorrow afternoon, then
join the new model's evals.

**10:20–10:40 ET** Stopping the old model's handover randomized eval (your OK) went
wrong, but no harm was done:
- My first stop command silently did nothing. The agent's shell is zsh, which doesn't
  split a variable of process IDs into words. Its file-moving part did run, though,
  and moved the eval's output folder while it was running.
- At 10:30 ET all 4 handover streams finished their 25th episode and crashed only when
  writing the final summary into the moved folder. All **100 episodes were recorded:
  0/100**.
- GPU 6's worker moved on to stack_blocks_two randomized (20 episodes) at 10:30 ET.
  My second stop attempt (10:38) found the old processes already gone and touched
  nothing.
- GPU 7 was never affected: its blocks_ranking_size job (new model) happened to finish
  normally at 10:21 ET (1/20), and it started lift_pot.
- Lessons added to AGENTS.md "Gotchas": use bash for PID-list loops; never move a
  running eval's output folder.

Old model, randomized so far: beat_block_hammer 4/30 (13%), handover_block 0/100,
lift_pot 0/17.

**10:50 ET** **Freed GPU 6 for someone who needed a GPU** (your OK). Stopped GPU 6's
chain into the new model's queue, then its old-model worker (stack_blocks_two
randomized; no episodes recorded yet, so ~25 min of work lost). Checked it was gone:
GPU 6 at 8 MB. GPUs 0/4/5/7 untouched. The old model's 3 unfinished randomized jobs
(stack_blocks_two 20, open_laptop 20, lift_pot +4) are back in `logs/queue_30k.txt`, so
GPUs 0/4/5/7 run them automatically after the new model's evals. The new model's evals
now have 4 GPUs instead of 5: clean screening ≈ tonight 8–10 pm ET, randomized ≈
tomorrow early afternoon.

**15:00 ET** Other users started big jobs on our eval GPUs: ~46 GB on GPU 7 (since
~13:17 ET) and ~42 GB on GPU 4 (since ~14:58). The simulator then couldn't get GPU
memory ("cannot create buffer"), so streams aborted. GPU 7's place_bread_skillet and
GPU 4's place_cans_plasticbox were broken, and place_a2b_left (earlier) came out with
10/20 episodes. GPU 5 is shared with a small 1.4 GB job and is fine; GPU 0 is ours
alone.

**15:28 ET** Decision (you): restarted the eval workers on **GPUs 4 and 7 with 2
streams each** instead of 4 (~26 GB, fits next to the other jobs). Checked the old
workers were gone first. place_a2b_left (+10), place_bread_skillet (20) and
place_cans_plasticbox (20) were requeued at the front, on new seeds; neither of the
last two had recorded any episodes. GPUs 0/5 keep 4 streams. New ETA: clean
screening ~11 pm–1 am ET, randomized tomorrow.

**18:51 ET** Status: clean screening 34/50 tasks done. Other users' jobs on GPUs 0
(two × 32 GB) and 5 (33 GB) crashed some of our streams; place_cans_plasticbox (10/20)
and place_object_basket (15/20) came out short.

**18:58 ET** Decision (you): **eval workers on all 8 GPUs, each sizing itself to free
memory** (`gr00t/benchmark_worker.sh`).
- Before each task a worker checks its GPU's free memory and starts as many streams
  as fit (14 GB each, 4 GB headroom, max 4). If none fit, it waits and re-checks
  every minute. No more out-of-memory crashes, and any spare memory on any GPU gets
  used, including GPUs other users are on.
- Switched to new queue files (`logs/queue_50task_v2.txt`, `logs/queue_30k_v2.txt`).
  The old workers on GPUs 4/7 finish their current task, find their old queues empty
  and exit; the new workers there wait for that.
- Stopped the old workers on GPUs 0 and 5 (checked gone).
- Requeued on new seeds: place_phone_stand (20), place_cans_plasticbox (+10),
  place_object_basket (+5).
- At 18:58: GPU 5 (now free) 4 streams, GPU 3 3, GPUs 2 and 6 1 each, GPUs 0/1 waiting
  for memory, GPUs 4/7 finishing old tasks.
- Randomized evals (50 tasks × 20) stay queued (your call: keep evaluating, pause if
  the GPUs are needed for hill-climbing).

**19:16 ET** Status: 35/50 clean tasks done (mean 43.4% so far), 17 of our eval streams
on all 8 GPUs. Problem on GPU 0: there was room for 3 streams when its task started,
then the other user's jobs grew back and our streams ran out of memory. A one-time
free-memory check isn't enough on a GPU whose other users' memory moves.

**19:30 ET** Improved the worker (`gr00t/benchmark_worker.sh`) without stopping any
running task:
- Free memory is the *lowest* of 6 samples over ~1 min, with a 10 GB margin (instead
  of 4) when other users are on the GPU.
- After each task it counts the episodes actually recorded. If streams crashed, it
  requeues the shortfall on new seeds, at most twice per task.
- It re-execs itself between tasks, so future edits apply without restarts.
- Tested the counting and other-user detection on real logs, then swapped each GPU's
  worker at its next task boundary. The current tasks kept running; only the waiting
  worker loops were stopped.
- Tasks started by the old workers (e.g. put_bottles_dustbin on GPU 0) aren't
  auto-healed; I'll top up any short ones by hand when the clean pass finishes.

**22:10 ET** Stopped the stuck **put_bottles_dustbin** clean eval on GPU 0 (your OK).
It had started at 19:04 during GPU 0's out-of-memory episode: its motion planner never
initialized, so every scene failed setup (~170,000 seeds in 3 h, 0 episodes). Checked
it was gone. Requeued it at the front as 4 chunks of 5 episodes on separate seed
ranges, so several GPUs can share this long task (1,700-step limit). Status: clean
screening 47/50 tasks, mean 46.1%; randomized 2 running, 48 queued.

## 2026-10-28

**10:23 ET** Status: **clean screening complete**: 50 tasks × 20 episodes, mean **43.0%**.
Randomized: 41/50 tasks done, mean **11.1%** (a big robustness gap). 8 running, 3
queued (2 of them requeued automatically after crashing at startup on a crowded GPU
1). The old model's 3 trimmed leftovers come after.

**10:26 ET** Stopped a **hung stream** (your OK): place_dual_shoes randomized, stream 2
on GPU 3, silent since ~04:25 ET (its simulator spun one CPU core at 100%, GPU at
0%). Checked it was gone. The worker recorded 14/20 episodes and requeued the
missing 6 on new seeds by itself, then moved on.

**10:35 ET** Added **hang detection** to the worker (your OK):
- Every 5 min it checks each running stream's client log; if a stream has been silent
  for 60+ min, it stops that stream, and the self-healing requeues its episodes.
- Healthy streams were measured to log at least every ~7.5 min, even on
  1,700-step episodes, so 60 min leaves a wide margin.
- Tested on a dummy job: it stopped exactly the silent stream and nothing else.
- Each GPU picks it up at its next task.

**11:10 ET** **Freed 4 GPUs for other people** (your call): GPUs **2, 4, 7** (ones we had
to ourselves) and shared GPU **1**. Checked none of our processes are left. We kept
**GPU 5** (ours alone; put_bottles_dustbin randomized at 17/20) and shared GPUs 0, 3,
6. The stopped streams' finished episodes are kept (logs in `logs/preempted/`,
counted by `summarize_eval.py`): stack_bowls_two 13, move_can_pot 1. The rest is
requeued on fresh seeds for the 4 remaining workers: place_dual_shoes 6,
move_can_pot 19, stack_bowls_two 7, blocks_ranking_rgb 20. (A first attempt at the
requeue hit a script bug and did nothing; the queue had already been empty, so no
harm.)

**12:21 ET** At your request, stopped **another user's idle vLLM server on GPU 1** (the
owner is out of office). Checked first: 0% GPU over 30 s, no client connections, up
~1.5 days. Stopped it gracefully (SIGTERM to `vllm serve`). Its exact launch command
is saved locally (git-ignored) in `scratch/stopped_servers/` so it can be restarted.
GPU 1 freed (6 MB).

**12:22 ET** Started our adaptive eval worker on **GPU 1**: 4 streams,
stack_bowls_two randomized (+7 episodes). Another user started a small 1.9 GB job
there the same minute; both fit. Our GPUs now: 0, 1, 3, 5, 6.

**14:10 ET** **Correction:** my task counts since last night were one too high (the
summary's header row was counted as a task). Real status: clean 49/50, randomized
49/50.
- **put_object_cabinet clean** had never completed. It crashed out of memory at 19:11
  ET yesterday on a crowded GPU under the old worker, which had no self-healing.
  The "clean complete" at 10:23 above was wrong. It's now requeued as 4 chunks of 5
  episodes (fresh seeds 40–55); the first chunk is running on GPU 3, where I
  restarted our worker after it had exited on an empty queue.
- **blocks_ranking_rgb randomized** is running on GPU 0 (9 episodes so far).
- New model so far: **clean 43.0%** (49 tasks), **randomized 9.9%** (49 tasks).
- Old model randomized: beat_block_hammer 4/30, handover_block 0/100, lift_pot 0/21;
  stack_blocks_two (GPU 5) and open_laptop (GPU 6) still running.
- Our workers on GPUs 1 and 3 had exited at ~13:30 when the queues emptied; GPU 1 is
  now used by others.

**17:10 ET** **All evals finished.** Every task in every setting has its full 20+
episodes; the workers exited on empty queues and released their GPUs.
- **50-task model:** clean **42.6%** (50/50 tasks), randomized **9.7%** (50/50).
  - Clean: 10 tasks ≥80%, 8 at 50–80%, 19 at 20–50%, 13 under 20%.
  - Randomized: 41 of 50 tasks under 20%; best are shake_bottle 55% and
    shake_bottle_horizontally 50%.
- **5-task model:** clean 28.3%, randomized 4.7%.
- On those 5 tasks the 50-task model is about the same or better (stack_blocks_two
  clean 7% → 20%; open_laptop 67% → 50% is within noise at 20 episodes).
- Full tables: `notes/results.md`.

## 2026-10-30

**01:00–01:25 ET** Next experiment (your call): **one model trained on clean +
randomized demos**, to attack the clean 42.6% vs randomized 9.7% gap.
- **Data:** 50 clean + **200 randomized** demos per task (12,500 episodes). RoboTwin
  publishes randomized demos only in its raw format (500 per task, 391 GB), which
  XPolicyLab can't read. `gr00t/fetch_randomized.py` reads the zips remotely, pulls just
  the 200 episodes needed (~50 GB) and converts them to XPolicyLab format. The mapping
  (state = raw step t, action = raw step t+1, same 4 cameras, "seen" instructions) was
  derived from, and checked against, the clean demos RoboTwin publishes in both
  formats: 6 episodes in 3 tasks reproduce the official files field-for-field (images
  differ only by JPEG re-encoding, ~1/255).
- **Training:** same recipe as the 50-task model, **150k steps**, checkpoint every 15k,
  all kept (~350 GB) for a learning curve later (`SAVE_TOTAL_LIMIT`, XPolicyLab commit
  47929bf). GPUs **3, 0, 4, 7**, fixed. GPU 3: another user's idle vLLM server will be
  stopped by hand (your OK). GPUs 0/4/7 each have a stable ~17.5 GB job of another user,
  so training starts only when each has ≥ 45 GB free (our ~22 GB + ≥ 23 GB spare).
- **No bottlenecks this time:** all of last time's dataloader fixes are in GR00T's code.
  Data prep is split into 200 small conversion jobs (4 groups of 50 episodes per task),
  24 at a time; the machine is quiet (load ~53 on 192 cores).
- **Pipeline** `gr00t/run_mix200.sh`, detached so it survives the Claude session ending
  (log `logs/run_mix200.log`): fetch → convert + merge → checks → train, with
  auto-resume from the last checkpoint after a crash.
- A test caught an argument bug (`--extra_parts` swallowed the task list) before the
  real run reached it; fixed. A subagent is testing the whole dataset chain on tiny
  data in parallel with the fetch.

**01:30 ET** The fetch was capped by Python's interpreter lock (24 threads in one
process, ~360% CPU on a 192-core machine): 294 episodes/min. Switched it to 24
processes and restarted the pipeline (it resumes where it was): **687 episodes/min**,
so the fetch finishes ~01:45 ET. Then convert + merge + checks (~1–1.5 h), then
training.

**01:40 ET** Corrected the data-prep estimate. My "~1–1.5 h" was wrong; it's ~4 h with
the current steps, scaled from last time's measured timings (12,500 episodes vs
2,500): fetch ~25 min, LeRobot conversion ~40 min (24 jobs at a time), merge ~60 min,
v3→v2 downgrade ~2 h, stats + check ~10 min. The merge and the downgrade are
single-threaded, and together they're a round trip: merge glues the videos into big
files, then the downgrade cuts them back into per-episode files. A subagent is making
the downgrade parallel and verifying identical output before the pipeline reaches it
(~2 h → ~20 min), so the data should be ready ~04:00 ET. For the next dataset, skip the
round trip (downgrade each part in parallel, then merge per-episode files).

**01:50 ET** Fetch problem and fix:
- After the switch to 24 processes, ~30 tasks failed with "File is not a zip file":
  we were sending ~12 range requests/s, very likely over Hugging Face's rate limit
  (a throttled request returns an error page, not file bytes).
- Stopped the fetch (no GPU; 3,558 episodes kept) and relaunched it with 64 MB read
  blocks instead of 8 MB (the data files sit back to back in each zip, so ~8× fewer
  requests), 12 processes, and retries that wait out the 5-minute rate window.
- Now **308 episodes/min with zero errors**; fetch done ~02:20 ET.

**01:55 ET** The subagent's test of the dataset chain (convert → merge → v3→v2 →
stats → check → GR00T loader) on tiny data **passed**: merged episode counts,
instructions from both sources, bit-identical values after the merge, and every video
checked. It found and fixed a real bug (commit `ca6b248`): the check wrote the "dataset
ready" marker before deciding pass or fail, so a re-run after a failed check would
have trained on bad data. I added a retry for one-off network-disk read errors in that
check (commit `909bee1`). A second subagent is making the v3→v2 step parallel.

**02:05 ET** Skipping the round trip for this dataset too (your call). The old
pipeline merged all pieces into big video files and then cut them back apart (~3 h,
single-threaded). New build (`gr00t/build_mix200.sh`, detached, log
`logs/build_mix200.log`): wait for the fetch → convert the 200 randomized pieces → copy
the 50 clean pieces → downgrade each piece to GR00T's format on its own, 24 at a time
(~10 min) → merge the per-episode files directly (`gr00t/merge_v2.py`, ~10 min) →
GR00T stats → check every video. A subagent is writing `merge_v2.py` and validating it
against the dataset built the official way last time (same 50 clean pieces → same
episodes, values and videos); the build waits for that validation before merging.
Data should be ready ~03:15 ET. The old pipeline's later steps were stopped; the
fetch kept running (7,000 / 10,000 episodes at 02:05).

**02:04 ET** Both subagents finished; the fetch finished at 01:59 ET (10,000 randomized
episodes) and the build is converting the randomized pieces now.
- **Direct merge validated** (`gr00t/merge_v2.py`, commit `c5ddd1c`). It rebuilt the
  50-task clean dataset from its 50 pieces and matched the official build: same 2,500
  episodes and lengths, same instructions, identical values in every column, every
  video with identical frames (pixel-identical on 150 decoded spot checks), same
  normalization stats for state and action. Two harmless differences: state/action
  keep a fixed-length-14 type (as in the 5-task dataset GR00T already trained on), and
  the bookkeeping index stats are now correct (the official build kept stale ones).
  The new path takes ~4.5 min vs ~3 h. I created `scratch/merge_v2_validated`, so
  the build will merge as soon as the pieces are downgraded.
- **Parallel v3→v2 downgrade** (XPolicyLab commit `ab4f0d9`): `--workers N`, default 1
  (unchanged behavior); byte-identical output, 22 min → 2.5 min for 2,500 episodes at 32
  workers. Our build doesn't use it (it runs 24 one-worker downgrades at once instead).
- Exported XPolicyLab's two newest commits to `patches/xpolicylab/0009`–`0010` and
  committed the fetch's rate-limit fixes (`7987f5c`).

**02:15 ET** Data prep, two changes:
- **Faster conversion.** The 200 randomized pieces were converting at ~2.2/min (end
  ~03:30 ET) with the box 60% idle and our 24 converters using ~1 core each, so the
  limit was how many run at once. I started a second, detached converter on the 50
  `demo_randomized3` pieces, all at once (log `logs/convert_randomized3.log`). The
  main converter skips any piece that already has `meta/info.json`, which LeRobot
  writes when a conversion starts, and it won't reach `demo_randomized3` for ~45 min,
  so the two never work on the same piece. Saves ~25 min.
- **Data check ready** (`gr00t/verify_dataset.py`): checks every merged episode against
  its source file (same length, identical state and action, the instruction GR00T reads
  is one of the file's sentences), compares one video frame per camera per piece with
  the source image, counts episodes per task and setting, and checks the stats for
  NaN/inf. On the existing 50-task dataset: 0 problems, frames within 1.4/255 of the
  source; with the sources deliberately shuffled it fails, as it should.

**03:45 ET** The dataset build **failed its final check**: 11,961 of 12,500 episodes (every
video in them was fine). 17 of the 200 randomized pieces were short (1–49 of 50
episodes): 14 from the extra `demo_randomized3` converter I started at 02:14, and 3
from the main one while both were running.
- **Cause:** on the network disk (EFS) the converter sometimes can't delete its temp
  folder (`Errno 39 Directory not empty`: a file still open when deleted leaves a
  hidden `.nfs…` placeholder). After that error the converter's writer is broken: every
  later episode of the piece fails, and the script still exits 0. Running 74
  conversions at once made this race common; at 24 it's rare. **My extra converter
  caused it** and cost ~40 min instead of saving 25.
- **Fix:** `build_mix200.sh` now counts each piece's episodes after conversion; short
  pieces are deleted and converted again (up to 3 attempts), so this can't slip through
  silently. I deleted the failed merged dataset and re-ran the build (log
  `logs/build_mix200.log`; the failed run is in `logs/build_mix200.attempt1.log`). It
  redoes only the 17 pieces (17 at once), then downgrade, merge, stats and the check;
  data ready ~04:30 ET.

**05:09 ET** **Training started: `mix200_150k`** (50 clean + 200 randomized demos per task,
all 50 tasks, 150k steps, a checkpoint every 15k, all kept). Full details for
reproducing and analysing it: [`notes/runs/mix200_150k.md`](notes/runs/mix200_150k.md).
- **Data** (`data/lerobot/RoboTwin-mix200-aloha_agilex-joint`): the rebuilt dataset
  passed every check: 12,500 episodes (per task: 50 clean + 200 randomized), every
  video one frame per step, and every episode's state, action and instruction identical
  to its source file (`gr00t/verify_dataset.py`, 0 problems; frames within 1.6/255 of
  the source images). One finding: 169 randomized episodes (1.7%, none clean) have a
  joint wound past a full turn (up to 8.5 rad). These are smooth motions as RoboTwin
  recorded them, and GR00T scales by the 1st/99th percentiles (unchanged by them), so I
  kept them.
- **GPU 3:** still the same idle vLLM server (up 1 day 21 h, 0% use, no connections), so
  I stopped it just before the dry run, as agreed; its launch command is saved in
  `scratch/stopped_servers/`. GPU 5 was empty.
- **Dry run** (300 steps, GPUs 3, 5, 0, 7): data loaded, loss 1.04 → 0.10, checkpoint
  saved (34 GB, complete; then deleted), ~2.3 steps/s with the GPUs ~40–60% busy. That
  looked data-bound, so the real run uses 32 workers per GPU instead of 16 (your earlier
  OK); it made no difference (~2.35 steps/s), so the limit is elsewhere in loading.
- **Launch hiccup:** in the ~2 min between the dry run and the launch, another user's
  jobs started on GPUs 0, 3 and 7. On 0 and 7 they were 49 GB each, leaving ~10 GB
  spare with ours, below the margin you asked for. I tried to move the just-started run
  to GPUs 1 and 2 (~40 GB spare), but stopping our running job is blocked without your
  OK, so it stayed. By the first training step those jobs had ended; all four GPUs are
  now ours alone (~21.7 GB used, ~59.7 GB free each).
- **Settings:** identical to `50task_60k` except the dataset, 150k steps, saving every
  15k with all 10 checkpoints kept, and 32 workers (diffed the saved configs). Commits:
  this repo `8dacfd9`, XPolicyLab `ab4f0d9`. Base model `nvidia/GR00T-N1.7-3B`
  (snapshot `2fc962b`). The learning rate follows one cosine schedule over 150k steps,
  so the 15k, 30k, … checkpoints are snapshots of one run.
- **ETA ~22:45 ET tonight** (~2.35 steps/s). On a crash it waits for room on the same
  GPUs and resumes from the last checkpoint (up to 3 times).

**12:25 ET** Started the **eval explorer** (`analysis/`, own uv venv, read-only on
logs and videos). First step, an episode index (`analysis/build_index.py`, ~50 s cold):
- 2,766 episodes from 413 client logs, 2,626 paired with their video. Every pairing
  passes a frame check (video frames = steps, +1 on success), and the totals match
  `summarize_eval.py` / `notes/results.md` exactly.
- Every policy failure runs to the step limit (2,063/2,063), so "timeout" tells us
  nothing; failure types have to come from watching the videos.
- 140 episodes have no video: 5task_30k randomized handover_block (folders moved
  mid-run on 10-27, see below), lift_pot (videos discarded), and 23 all-timeout
  episodes whose folder can't be told apart from a sibling shard's.
- Data problem: 20 of 5task_30k's handover_block randomized "failures" are not policy
  failures. Its folders were moved to `outputs/preempted_eval/` while running, so
  episodes 20–24 of each shard died on a broken video pipe after 1 step. Real result:
  0/80, not 0/100.

**12:45 ET** Eval explorer, first working version (your picks: no re-encoding, eval
logging change, `.gitignore`):
- **Eval logging (your OK):** the eval client now writes `episode<N>.json` next to each
  video: seed, instruction, outcome, steps, checkpoint number. It's on RoboTwin's local
  branch `local/aloha-gr00t`, exported as `patches/robotwin/0001`. Tested on a stub, not
  yet in a real eval; the index cross-checks every record against the log.
- **App:** `analysis/run.sh` (Streamlit, CPU only, localhost:8611; running detached now
  as PID 3986960). Pick model / checkpoint / setting / task → success with a 95%
  range → click an episode → video with frame stepping. Videos play as recorded
  (H.264); nothing is re-encoded. Every model/checkpoint/setting view tested headless.
- `.gitignore`: `analysis/cache/` (index, app log) and `analysis/tags/` (your tags).

**19:11 ET** **Eval of the new model's checkpoints: plan agreed and armed.** It starts by
itself on GPUs 0, 3, 5 and 7 the moment training finishes (~23:00 ET). Full plan and
commands are in the run card, `notes/runs/mix200_150k.md`, under "Evaluation plan".
- **What runs, in order:**
  1. Checkpoints 150000 and 60000 on 10 tasks (`pick_dual_bottles`,
     `place_object_stand`, `move_pillbottle_pad`, `place_empty_cup`,
     `place_phone_stand`, `move_can_pot`, `dump_bin_bigbin`, `handover_mic`,
     `grab_roller`, `click_bell`), clean + randomized, ~20 episodes each.
  2. Checkpoint 150000 on the other 40 tasks: randomized ~20 episodes, clean 10.
  3. Checkpoints 120000, 90000 and 30000 on the same 10 tasks, randomized only.
  - Total 2,623 episodes, estimated 112.6 GPU-hours (~28 h on 4 GPUs), against ~1,100
    for every checkpoint on everything.
- **Same scenes as the old model.** Every job replays the exact seeds the old model was
  scored on, so results compare scene by scene.
  - How: `gr00t/paired_queue.py` copies the old eval's shard layout (seed index and
    episode count per shard) from its logs, one queue line per shard. The workers run
    one stream per job (4 workers per GPU), so no job gets re-split onto different seeds.
  - Crash retries now use seed index +1000 (`benchmark_worker.sh`), so they can't
    overwrite another job's log.
- **Smoke test passed** (19:01–19:10 ET, GPU 5): one episode of `click_bell` on the new
  model's checkpoint-60000 ran through the same worker setup and succeeded (199 of 400
  steps). The new `episode0.json` recorded the checkpoint number and the instruction. I
  deleted the test's outputs so they don't mix into the results.
- **Caveat for reading the 60k comparison.** The new model's checkpoint-60000 was taken at
  learning rate 7.0e-5. The old model's checkpoint-60000 was at 7.6e-14, the end of its
  schedule. Both values are from each checkpoint's `trainer_state.json`. Equal steps
  isn't like-for-like.
- **Launcher:** `gr00t/eval_mix200.sh`, detached (pid 1192637). It waits for the training
  supervisor (pid 3231349) to exit and checks that checkpoint-150000 saved completely.
  Log `logs/eval_mix200.log`; worker logs `logs/worker_mix200_gpu<G>_<1-4>.log`.

**23:18 ET** **Training finished, evals running.**
- **Training:** `mix200_150k` finished at 23:11 ET: 150,000 steps in 18 h 12 min, 2.29
  steps/s on average, no restarts.
  - Loss: mean training loss over the run 0.0127; last logged 0.0043.
  - All 10 checkpoints are kept (15000 … 150000).
- **Evals:** they started by themselves at 23:12 ET as planned: 16 single-stream workers, 4
  each on GPUs 0, 3, 5 and 7.
  - First up: checkpoint 150000 on the 10 tasks, using the old model's exact seed layout
    (e.g. `click_bell` randomized as 4 shards × 5 episodes at seed indexes 0–3).
  - The 150k vs 60k comparison on those 10 tasks is expected ~04:00–05:00 ET.

**03:59 ET** **Evals now on 5 GPUs.** GPU 4 was completely empty, so with your OK I
added 4 more workers on it, giving 20 streams in total: 4 each on GPUs 0, 3, 4, 5 and 7.
- **Setup:** same queue list and one stream per job as the others. I checked this from
  each worker's environment.
- **Command:**

  ```
  GPU=4 MAX_SHARDS=1 QUEUES="<same list as gr00t/eval_mix200.sh>" setsid -f nohup bash gr00t/benchmark_worker.sh > logs/worker_mix200_gpu4_<i>.log
  ```

  ×4, started 20 s apart.
- **Speed so far:** the 61 jobs finished by 03:45 ET took 16% longer than the old
  model's eval of the same tasks: 52.0 vs 45.0 stream-hours.
- **ETA:** ~458 stream-hours were left at 03:45. Over 20 streams instead of 16, that
  moves the finish from ~08:30 ET to ~03:00 ET Sunday (Nov 1), give or take a few hours. (Corrected: Nov 1 is a Sunday.)

**11:15 ET (Sat Oct 31)** **Results: the 10 tasks, three models, same scenes.**

Successes / episodes. "Old" = the 50-clean-demo model at checkpoint 60000; "new" = the
clean + randomized model.

| task | clean: old / new 60k / new 150k | randomized: old / new 60k / new 150k |
|---|---|---|
| click_bell | 19/20 / 11/20 / 19/20 | 8/20 / 15/20 / 20/20 |
| dump_bin_bigbin | 15/20 / 10/20 / 17/20 | 2/20 / 5/20 / 12/20 |
| grab_roller | 18/20 / 11/20 / 18/20 | 8/21 / 4/21 / 19/21 |
| handover_mic | 8/20 / 2/20 / 8/20 | 1/20 / 0/20 / 3/20 |
| move_can_pot | 6/20 / 0/20 / 6/20 | 1/20 / 0/20 / 10/20 |
| move_pillbottle_pad | 9/20 / 1/20 / 12/20 | 1/20 / 0/20 / 11/20 |
| pick_dual_bottles | 15/20 / 1/20 / 12/20 | 3/20 / 0/20 / 16/20 |
| place_empty_cup | 15/20 / 3/20 / 16/20 | 4/20 / 4/20 / 15/20 |
| place_object_stand | 10/20 / 5/20 / 15/20 | 1/20 / 3/20 / 11/20 |
| place_phone_stand | 8/20 / 1/20 / 7/20 | 2/20 / 2/20 / 10/20 |
| **total** | **123/200 (62%) / 45/200 (22%) / 130/200 (65%)** | **31/201 (15%) / 33/201 (16%) / 127/201 (63%)** |

- **The final new model (150k):** randomized 15% → 63%, clean unchanged (62% → 65%,
  within luck for 200 episodes).
- **The new model at 60k:** no better than the old model on randomized (16%) and much
  worse on clean (22%). Same number of steps, but it wasn't a finished model yet. Two
  likely reasons, which these numbers can't separate:
  - its learning rate was still high (7.0e-5 vs ~0);
  - it had seen each demo far fewer times. 60,000 steps × 64 samples = 3.84M samples:
    1.4 passes over its 2,759,740 frames, against 7.0 passes over the old model's
    548,893.
  So the gain needed the longer training; randomized data at an equal step count did
  not show it.
- **One stream crashed:** a policy server on GPU 3 ran out of GPU memory while loading
  (06:42 ET). Two of our streams were loading at once, plus another user's 2.3 GB
  process. The automatic retry re-ran its 7 `put_bottles_dustbin` randomized episodes on
  fresh seeds, so those 7 aren't the old model's scenes.
- **ETA** for the rest (other 40 tasks at 150k, then 120k/90k/30k on the 10 tasks): about
  07:00–12:00 ET Sunday (Nov 1).

**18:40 ET (Sat Oct 31)** **Memorization check: no sign of memorization.**

`gr00t/heldout_check.py`, run on GPU 4 next to the evals, 12:03–18:29 ET.
- **Method:** 10 tasks × 5 demos per set × 8 moments per demo = 400 predictions per set.
  At each moment the model predicts the next 16 actions through the eval code path.
- **Metric:** mean absolute arm-joint error in radians, against what the demonstrator did.
  "Hold still" (repeat the current joint state) scores ~0.064 on every set.
- **Sets:**
  - randomized demos the new model trained on: episodes 20/60/100/140/180;
  - randomized demos no model saw: episodes 200–204, fetched for this check;
  - clean demos both models trained on.

| model | randomized, trained on | randomized, never seen | clean, trained on |
|---|---|---|---|
| old model, 60k | 0.0357 | 0.0316 | 0.0175 |
| new model, 15k | 0.0479 | 0.0447 | 0.0453 |
| new model, 30k | 0.0389 | 0.0370 | 0.0361 |
| new model, 45k | 0.0349 | 0.0325 | 0.0323 |
| new model, 60k | 0.0295 | 0.0263 | 0.0266 |
| new model, 75k | 0.0280 | 0.0256 | 0.0253 |
| new model, 90k | 0.0228 | 0.0200 | 0.0194 |
| new model, 105k | 0.0217 | 0.0198 | 0.0200 |
| new model, 120k | 0.0188 | 0.0167 | 0.0171 |
| new model, 135k | 0.0181 | 0.0160 | 0.0164 |
| new model, 150k | 0.0176 | 0.0156 | 0.0159 |

- **No memorization.** At every checkpoint the unseen demos score as well as or slightly
  better than the trained-on ones (ratio 1.05–1.14). The old model, which never trained
  on either randomized set, shows the same ratio (1.13), so that small difference comes
  from the scenes, not from training. Both errors fall together through 150k.
- **The new model passed the old one:** on clean demos from ~120k (0.0171 vs 0.0175); on
  randomized demos from ~45k–60k. Final 150k: 0.0156–0.0176 on every set, against the old
  model's 0.0316–0.0357 on randomized.
- **Results:** `outputs/heldout_check/<model>_<checkpoint>.json`.

**06:30 ET (Sun Nov 1)** **All evals of the new model finished** (last job 04:10 ET). All
2,621 planned episodes were recorded. Full tables: `notes/results.md`, top section.
- **50-task benchmark, checkpoint 150000 vs the old model** (average of per-task rates):
  - randomized: **9.7% → 45.2%**;
  - clean: **42.6% → 46.0%**.

  The clean gain is small: 40 of the 50 tasks got 10 clean episodes, so per-task clean
  numbers are rough. Randomized went up on 44 of 50 tasks. Six tasks stay at or near 0%
  on both: `put_bottles_dustbin`, `stack_blocks_three`, `handover_block`, `hanging_mug`,
  `open_microwave`, `place_dual_shoes`. `open_microwave` randomized dropped 3/20 → 0/20.
- **The 10 tasks across checkpoints** (randomized; same scenes):

  | model | rate | learning rate |
  |---|---|---|
  | old | 15% | ~0 |
  | 30k | 11% | 9.4e-05 |
  | 60k | 16% | 7.0e-05 |
  | 90k | 37% | 3.8e-05 |
  | 120k | 62% | 1.1e-05 |
  | 150k | 63% | ~0 |

  Most of the gain came between 60k and 120k, while the learning rate fell from 7.0e-5
  to 1.1e-5; the 120k → 150k rise is small. Over the same stretch the model also went
  from 1.4 to 2.8 passes over its data. So this curve alone can't separate "lower
  learning rate" from "more practice". The short decay-from-60k experiment would.
- **GPUs:** our eval GPUs are free. Others took 0, 4 and 7 after the evals; 3 and 5 sat
  empty from ~04:10 ET.

**09:45 EST (Sun Nov 1)** **Started the learning-rate decay experiment** (`mix200_decay60k`, run
card `notes/runs/mix200_decay60k.md`) on GPUs 4 and 5, which you said were free.
- **Question:** did the main run's jump from 16% (60k) to 63% (150k) come from the
  low-learning-rate end of training, or from more passes over the data?
- **Setup:** checkpoint-60000's weights, trained 10,000 more steps on the same data. The
  learning rate ramps to 7.0e-5 over the first 500 steps, then falls to 0. That adds only
  0.23 passes over the data.
- **Two differences from the plan as discussed:**
  - the optimizer state restarts (DeepSpeed can't load it on 2 GPUs when it was saved
    on 4), hence the short ramp;
  - the drop is GR00T's cosine curve, not a straight line.
- **Dry run** (100 steps): weights loaded, loss continued at ~0.008 (the main run was at
  0.0087 near step 60k), learning rate followed the plan, checkpoint saved. Deleted it
  afterwards.
- **Mistake on my side:** I launched the real run while the dry run's processes were
  still shutting down, on the same communication port. It started cleanly anyway (the
  old processes exited seconds later), but I won't chain launches that way again.
- **Armed to start when training exits:**
  - the simulator eval of checkpoint-10000: the 10 tasks, clean + randomized, the first
    10 old seeds each, on GPUs 4 and 5;
  - the prediction-error check of checkpoints 10000 and 5000.

**20:30 EST (Sun Nov 1)** **Learning-rate decay experiment: done.** Full table:
`notes/runs/mix200_decay60k.md`.
- **Results** (10 tasks, identical seeds, ~100 episodes per cell):

  | model | randomized | clean |
  |---|---|---|
  | old | 17% | 64% |
  | new 60k | 14% | 22% |
  | **new 60k + 10k settling** | **46%** | **50%** |
  | new 150k | 67% | 63% |

- **Settling** explains ~60% (randomized) and ~68% (clean) of the 60k → 150k jump; more
  practice the rest.
- **At about equal training cost,** randomized demos win big on randomized scenes but cost
  clean performance (50% vs 64%) until the model trains longer.
- **Correction:** my 5-task early read (75% of the gap) overstated it; with all 10 tasks
  it's 60%.
- **GPUs:** 0, 4, 5, 7 free since ~15:25 EST; nothing queued.

**20:55 EST (Sun Nov 1)** **Run registry, more-data prep, and continued training.**
- **Registry:** `notes/runs/README.md` lists every training run with its lineage (which
  checkpoint it started from), data, schedule, GPUs, checkpoints kept, results and run
  card.
  - Each run's checkpoint folder now has a `run_info.json`. Written after the fact for the
    5 existing runs; future runs write it at launch (`gr00t/write_run_info.py`, called by
    `train.sh` and `train_decay.sh`).
- **Data prep for `mix500`** (priority): 50 clean + randomized episodes 0–494 per task,
  27,250 episodes.
  - Fetching randomized episodes 200–494 for all 50 tasks (14,750) since 20:45, ~40 min;
    log `logs/fetch_mix500.log`.
  - Episodes 495–499 per task are fetched separately and kept out of training, for the
    memorization check.
  - `gr00t/build_mix500.sh` (detached, log `logs/build_mix500.log`) then converts only
    the new groups, reuses mix200's pieces, merges 550 pieces, computes stats, checks
    every video, and compares every episode with its source file.
- **Continued training of the best model:** `mix200_150k_rewarm50k` (run card
  `notes/runs/mix200_150k_rewarm50k.md`), 50k steps from checkpoint-150000, learning rate
  back up to 5e-5 then down to 0.
  - **GPU mix-up:** another account ("ubuntu") started 56 GB jobs on GPUs 0 and 4 minutes
    before my dry run. Only ~3 GB would have been left over, so I stopped my dry run
    (1 minute old, no progress) and moved it to GPUs 5 and 7.

**02:05 EST (Mon Nov 2)** **Started: NVIDIA Cosmos 3 Nano as a policy on our data** (your top
priority). Goal: post-train Cosmos3-Nano on the same data as GR00T `mix200_150k`
(`RoboTwin-mix200-aloha_agilex-joint`) and compare on the same RoboTwin scenes.
- **What Cosmos 3 is.** NVIDIA's mid-2026 "omni" world model: one Mixture-of-Transformers
  model for reasoning, video generation and robot actions; Nano = 16B parameters.
  - Policy post-training comes from NVIDIA's `cosmos-framework` (github.com/NVIDIA/cosmos-framework),
    which ships recipes for LIBERO and DROID: full fine-tuning, 2×8 GPUs, global batch 2048.
  - No RoboTwin or XPolicyLab adapter exists (checked XPolicyLab upstream too).
  - Our robot isn't one of its built-in embodiments (14-D joint positions, 3 cameras, 15 fps),
    so it needs a RoboTwin dataset/experiment config and an eval adapter.
- **Hugging Face:** every Cosmos 3 repo is ungated and our token has access; nothing to accept.
- **Done so far:**
  - downloaded `nvidia/Cosmos3-Nano` (33 GB, HF cache on NVMe) and the Wan2.2 VAE (2.8 GB);
  - started installing `cosmos-framework`'s CUDA-13 training env (its own `.venv`), in
    `/mnt/efs/rohingarg/cri/cosmos3/` (outside this repo, like `RoboTwin/`);
  - a subagent is studying NVIDIA's LIBERO recipe and writing the RoboTwin version on a
    local `robotwin` branch of the clone (dataset class, experiment config, normalization
    stats, dryrun and data-loader test). No GPU use yet.

**02:50 EST (Mon Nov 2)** **Cosmos3-Nano recipe decided: NVIDIA's own, kept exactly.**
- **Settings** (NVIDIA's LIBERO policy recipe, `action_policy_libero_nano.py` + its TOML):
  - full fine-tuning of all 16B parameters;
  - fused AdamW, betas (0.9, 0.99), weight decay 0.05;
  - learning rate 5e-5, 500-step warm-up, then linear decay over a 16,000-step cycle;
  - global batch ~2,048 (per-GPU batch × GPUs × gradient accumulation; 2,080 on 5 GPUs);
  - checkpoint every 500 iterations.
- **Length:** 5,000 iterations, NVIDIA's own length for its 4-suite LIBERO run. It also
  roughly matches the number of examples GR00T saw: 5,000 × 2,048 = 10.2M vs 150,000 × 64
  = 9.6M.
- **The learning rate won't be settled at the end.** NVIDIA stops at 2,000 / 5,000
  iterations of a 16,000 cycle, so at iteration 5,000 it is still 5e-5 × (1 − 4,500 ÷
  15,500) = 3.55e-5. That is their tested setup, so it is the baseline. Letting it decay
  to 0 is our own variant, run separately later (a short decay from a saved checkpoint,
  as with GR00T's `mix200_decay60k`).
- **GPUs:** whatever is free at launch (up to 5–6); per-GPU memory will show whether 4 would
  suffice.
- **Speed is unknown:** my very rough guess is ~50 s per iteration on 5 GPUs. I'll measure
  it in the first minutes before committing to the full length.

**02:47 EST (Mon Nov 2)** **Cosmos env install fixed: 1.5 min instead of 1+ hour.**
- **Cause:** uv's download cache is on the network disk (EFS). Two attempts spent ~1 hour
  unpacking NVIDIA's CUDA libraries there without installing a single package.
- **Fix:** installed with uv's cache and the environment on the machine's local NVMe drive.
  It finished in ~1.5 min: 899 packages, PyTorch 2.10.0 + CUDA 13.0.
- **Where:** `/opt/dlami/nvme/rohingarg/cosmos3-venv`, linked as `cosmos-framework/.venv`.
  NVMe is wiped if the machine restarts; the env rebuilds from the lock file in minutes.
- **Next:** the subagent can now compute the stats, test the data loading and dry-run the
  config. Gotcha added to AGENTS.md.

**03:06 EST (Mon Nov 2)** **Cosmos3-Nano: recipe ready, GPU test running.**
- **Subagent finished the RoboTwin recipe** (branch `robotwin` of the cosmos-framework clone,
  7 commits). The config dry-run passed, and all 30 CPU data-loading checks passed (images,
  actions, instructions).
- **Choices** (each following an official NVIDIA recipe):
  - 14-D absolute joint positions, with the current state as the first, given row (NVIDIA's
    DROID joint-position recipe);
  - head camera on top, both wrists below at half size (NVIDIA's AgiBot layout);
  - 15 fps, 16-action chunks, quantile normalization with our own stats;
  - no clipping, as the official code does: values above 1 in magnitude appear in 15.3% of
    frames, above 10 in 0.12%;
  - RoboTwin already has a slot in Cosmos's action heads (domain 17, 14-D); it starts
    untrained in the public checkpoint.
- **Subagent's estimates (not measured):** ~60–73 GB per GPU on 5 GPUs; ~26–42 s per
  iteration, so ~36–58 h for 5,000; ~180 GB per checkpoint, ~2 TB in all.
- **GPUs:** stopped the continued-training run at step ~29,400 (its 10k and 20k
  checkpoints kept; registry updated) and cancelled its follow-up check, to free GPUs 5
  and 7. GPU 3 had been taken by another user.
- **GPU test:** 10 iterations on GPUs 0, 4, 5, 6, 7 (batch 32 × 5 × 13 = 2,080) since 03:05
  EST, output to NVMe scratch. It measures peak memory and seconds per iteration; the real
  launch follows if it passes.
- **Still to do:** the RoboTwin eval adapter for Cosmos (training takes ~2 days, so there's
  time).

**03:30 EST (Mon Nov 2)** **Cosmos3-Nano training launched** (run card
`notes/runs/cosmos3_nano_mix200.md`).
- **GPU test** (10 iterations, 5 GPUs) passed: peak 59.4–60.8 GB per GPU, 36.3–37.6 s per
  iteration, ~8 min startup, a 166 GB checkpoint saved in ~2 min to NVMe.
- **4 GPUs would probably fit** (my estimate from the subagent's ~206 GB of fixed training
  state): ~206 ÷ 4 + 20 GB working memory ≈ 71 GB per GPU, at ~46 s per iteration.
- **Real run** on GPUs 0, 4, 5, 6, 7 since 03:24 EST. The first attempt at 03:23 did
  nothing: my supervisor's `export` line swallowed the launch command; fixed and relaunched.
- **ETA:** 5,000 × ~37 s ≈ 51 h, so ~07:00 EST Wednesday (Nov 4). It resumes after crashes
  from the latest checkpoint. Checkpoints every 500 iterations (~166 GB each, ~1.8 TB in
  all, on EFS).
- **Plan:** evaluate checkpoints 1,500 and 2,000 while training continues, and stop early if
  they're good or flat. Needs the Cosmos RoboTwin eval adapter.

**04:15 EST (Mon Nov 2)** **Cosmos eval adapter ready** (subagent; CPU-verified, not yet
run on a GPU).
- **What it is:** the XPolicyLab adapter `policy/Cosmos3_Nano/` (XPolicyLab `92c0a8c`, patch
  `patches/xpolicylab/0011`), plus `cosmos3/eval.sh` (one eval stream, same logs as GR00T's),
  `cosmos3/check_adapter.py` and `cosmos3/heldout_check.py` (the same open-loop prediction
  check as GR00T's, comparable numbers).
- **Verified, CPU only:** 321 checks.
  - On real training moments, the model input the adapter builds is bit-identical to
    training's: image composite, prompt, normalized state.
  - Output decoding returns the exact 16 actions.
  - The real policy server and the RoboTwin client code exchange a chunk correctly.
- **Settings follow NVIDIA's LIBERO server:** 30 sampling steps, guidance 1.0, shift 5.0, all
  16 actions executed per query (as our GR00T eval).
- **Expensive to evaluate (estimates):** ~45 GB per stream, so one stream per GPU vs four for
  GR00T, at ~3–9 episodes per GPU-hour. Sim evals of Cosmos checkpoints need a budget
  decision from you; the open-loop check is cheap (~20–60 min per checkpoint).
- **Next:** when checkpoint 500 lands (~08:30), restart training with wandb on, then run a quick
  open-loop check on it if a GPU is free, to test loading, memory and speed.

**04:55 EST (Mon Nov 2)** **Company-cluster packaging for Cosmos is ready; submitting
it needs you.**
- **Built and validated locally, no GPUs** (subagent): a Docker image (27.1 GB, CUDA 12.8
  build), job configs for 1 and 2 nodes, an in-container script and a submit script. The
  in-container script:
  - downloads inputs from S3;
  - resumes from the run's S3 checkpoint;
  - uploads each checkpoint as it's saved;
  - logs to the company wandb;
  - coordinates 2 nodes.
- **Kept local:** all files plain, uncommitted, in the private local folder, as you asked.
  Nothing pushed anywhere.
- **Inputs uploaded to S3:** 50,020 files, 70.2 GB (data, index, base model, VAE).
- **My test-job submission was blocked** by Claude Code's permission system ("shared cluster
  mutation"); I didn't work around it. To run the 1-node test yourself (~30–40 min, ~5
  GPU-hours, borrowing pool, 2-hour limit), run from the private local cluster folder (see my handoff):
  `./submit.sh --mode test --nodes 1 --run-name nano-st-test-1n --deadline-hours 2 --borrow`
  - Add `--dry-run` first to review what gets submitted (I did: correct).
  - The test prints the cluster's driver version, the GPUs, and the time per iteration.
- **Estimates if it works:** 1 node, ~23 s per iteration (~32 h for 5,000); 2 nodes ~12–14 s
  (~18–20 h, cross-node cost unmeasured). A full run could resume from a local checkpoint
  staged in S3; the commands are in the subagent's report, summarized in my handoff.
- **Behaviour to know:** after a resume on the cluster, the data order is reshuffled (seed 42
  + resume iteration), so repeated preemptions don't keep replaying the start of the data.
  `RESHUFFLE_ON_RESUME=0` turns that off. The wandb key ends up visible in the job's
  command, as with SkyRL's launcher.

**08:44 EST (Mon Nov 2)** **Cosmos run restarted with wandb on (you OK'd it).**
- **Iteration 500 checkpoint saved** at 08:38 EST (166 GB). The loss fell from ~14.5 at the
  start to ~0.6–1.07 around iterations 500–507, at ~37 s per iteration.
- **Restart:** at iteration 507 I stopped the run and relaunched it on the same 5 GPUs
  (0, 4, 5, 6, 7). It resumes from iteration 500, so ~4 minutes of training are redone.
  - wandb project `robotwin-posttrain`, logged in from `~/.netrc`.
  - The run folder and every recipe setting are unchanged.
  - The log before the restart is `logs/cosmos3_nano_mix200.part1_iter0-507_nowandb.log`.
- **Speed observation:** VAE encoding of the camera frames takes ~14 s of each ~37 s
  iteration (~38%). The framework encodes every training sample on the fly. Encoding the
  data once ahead of time could cut each iteration to ~23 s, but that changes NVIDIA's
  pipeline. It's an option for later runs, not this one.
- **mix500 dataset finished** at 03:51 EST with 0 problems: 50 tasks × (50 clean +
  randomized episodes 0–494).

**09:10 EST (Mon Nov 2)** **Cosmos: wandb works, and the eval now runs end to end. First
numbers at iteration 500.**
- **Training resumed cleanly** from iteration 500 at 08:51. At 522 the loss was 0.76–1.19 and
  each iteration took ~35.5 s. wandb run `action_policy_robotwin_nano` in project
  `robotwin-posttrain`, logging every 10 iterations.
- **Bug found and fixed in NVIDIA's inference code.**
  - Loading a training checkpoint crashed: the code adds a setting used only by NVIDIA's
    self-driving checkpoints (`_auto_camera_fps`) to a config that rejects new keys.
  - Fix: two lines on our local cosmos-framework branch (commit 30bd31a, not pushed). It
    doesn't affect training.
- **Open-loop check, iteration 500** (quick version: 80 moments per set instead of 400).
  - Error is the mean arm-joint error in rad. Ratio is error ÷ the error of holding still;
    lower is better.

  | model | examples seen | train randomized | train clean | unseen randomized |
  |---|---|---|---|---|
  | Cosmos, iteration 500 | 1.04M | 0.046 (0.78) | 0.038 (0.59) | 0.045 (0.74) |
  | GR00T mix200, step 15k | 0.96M | 0.048 (0.76) | 0.045 (0.69) | 0.045 (0.69) |
  | GR00T mix200, step 30k | 1.92M | 0.039 (0.61) | 0.036 (0.55) | 0.037 (0.57) |
  | GR00T mix200, step 150k | 9.6M | 0.018 (0.28) | 0.016 (0.24) | 0.016 (0.24) |

  - **Reading:** after the same number of training examples, Cosmos is about where GR00T was.
    This is early: Cosmos's warm-up only just ended, so its learning rate only now reached
    the peak.
  - Memorization: the unseen set is no worse than the trained randomized set.
- **Sim test, iteration 500**, beat_block_hammer clean: **1/3 succeeded**.
  - Each policy call takes ~1.4 s for 16 actions.
  - Policy plus simulator use ~38.5 GB, so one stream per GPU.
  - 3 episodes took ~12 min after a ~80 s model load: ~4 min per episode, ~15 episodes per
    GPU-hour. That's better than my earlier estimate of 3–9, but it's from only 3 episodes.
- **Next:** evaluate checkpoints 1,500 and 2,000 once they're saved (~18:45 and
  ~23:40 EST; the run should finish ~Nov 4 06:00 EST). The sim eval size needs your budget decision. The open-loop check is cheap
  (~15 min per checkpoint), and I'll run it when a GPU is free.

**10:25 EST (Mon Nov 2)** **You submitted the 1-node cluster test at 10:10 EST. It's waiting
in the queue (borrowing pool, 2-hour limit, 20 iterations).** I'm watching its status. When it
runs, the time per iteration shows up in wandb (`robotwin-posttrain`) next to the local run.

**11:15 EST (Mon Nov 2)** **Cluster test passed: 8 H100s on one node train Cosmos at ~19.9 s
per iteration, against ~35.5 s on our 5 local GPUs (1.8× faster).**
- Timeline: queued 10:10–10:55 EST (shared pool), then everything ran in 15 min.
  - inputs downloaded in 66 s;
  - first iteration 99 s (warm-up);
  - iterations 2–19: 19.5–20.3 s each, batch 2,048 (32 per GPU × 8 GPUs × 8 accumulation);
  - checkpoint at 20: saved in ~100 s, uploaded in 49 s (166 GB);
  - exited cleanly.
- The loss started at ~14–15.6, as the local run's did.
- **wandb:** the test landed in the team workspace, not next to the local run, because the
  cluster job used a different wandb key than the local login. To fix before a full run.
- **Projection:**
  - The local run reaches 5,000 iterations ~Nov 4 06:00 EST.
  - Moving it to one cluster node from the local iteration-1,000 checkpoint (~13:55 today)
    would mean ~4,000 × 19.9 s ≈ 22 h of training, plus queueing and copying. It would finish
    ~Nov 3 afternoon and free our 5 GPUs for evals.
  - Untested: resuming a 5-GPU checkpoint on 8 GPUs.
  - The batch would be 2,048 vs 2,080 locally, unless set to 26 per GPU × 8 × 10 = 2,080.
- Your call: move or not.

**11:30 EST (Mon Nov 2)** **Cluster: prepared a 2-node (16-GPU) run, as you asked.** All
changes are in the private local folder; nothing committed.
- **wandb:** cluster jobs now use your own key from `~/.netrc`, the same login the local run
  uses, so their runs land next to it.
- **2-node layout:** on 2 nodes, the cluster script now uses NVIDIA's own layout from their
  LIBERO recipe (2 nodes × 8 GPUs): each model copy is split across one node's 8 GPUs, and the
  two nodes exchange only gradients.
  - Batch: 32 per GPU × 16 GPUs × 4 accumulation = 2,048, NVIDIA's.
  - The trainer's CPU config check resolved these settings correctly.
  - The submit dry run looks right.
- **Next:** you submit a 2-node test (20 iterations, ending with a checkpoint that both nodes
  upload). If it passes, a full 5,000-iteration run from scratch on 2 nodes.

**11:40 EST (Mon Nov 2)** You submitted the 2-node test at 11:29 EST; it's queued. I'm watching it.

**12:45 EST (Mon Nov 2)** **The local Cosmos run has slowed to ~71 s per iteration (from 35.5 s).**
- Cause: another user's training job started on GPU 7 at ~11:35 EST. Our 5 GPUs step in
  lockstep, so the whole run waits on the shared one. Not ours to stop.
- At this speed the local run would need ~80 h more, so the cluster run matters more.
- The 2-node test has been queued since 11:29 (shared pool). If it hasn't started by ~13:00,
  I'll cancel it and resubmit it on our team's own quota (no borrowing).

**13:15 EST (Mon Nov 2)** **2-node test resubmitted at high priority** (your suggestion), on
our team's own quota.
- The two normal-priority copies are still queued: one in the shared pool, one I submitted
  at 13:02 on our team's quota.
- Claude Code's permission system blocked me from cancelling them, so they need you (commands
  in my message). If they start anyway, each just runs the same 15-min test (~4 GPU-hours).
- 13:20 EST: with your permission, cancelled the two normal-priority copies. Only the
  high-priority 2-node test is left in the queue.
- 13:30 EST: also submitted the test in the shared pool (borrowing) at high priority, as you
  want for tests. I'll cancel whichever of the two high-priority tests hasn't started when the
  other does. Only the full run uses our team's own quota.

**14:05 EST (Mon Nov 2)** **The 2-node test failed after ~1 min of training setup. Cause
found and fixed; resubmitted.**
- Everything up to data loading worked: both nodes connected, all 16 GPUs joined one
  communication group, NVIDIA's 8 × 2 layout was applied, and the model initialized.
- **Then every process aborted.** The data loader starts its worker processes with fork().
  The library that carries traffic between nodes (AWS EFA) treats that as unsafe unless told
  otherwise, and kills the job. Its own message says to set `FI_EFA_FORK_SAFE=1`.
- This only affects multi-node runs: the 1-node test doesn't use that library.
- **Fix:** the cluster script now sets `FI_EFA_FORK_SAFE=1` (and `RDMAV_FORK_SAFE=1`) on
  multi-node runs. It's a local file, uncommitted.
- **Resubmitted the test** at high priority as two copies, one in the shared pool and one on
  our team's quota. I'll cancel whichever doesn't start first. Fresh run names, so the failed
  run's leftover files can't confuse the new ones.
- **Local run:** another user's job is still on GPU 7, so it runs at ~71 s per iteration.

**15:25 EST (Mon Nov 2)** The two fixed 2-node tests have been queued since 14:01.
- As you suggested, I also submitted one under a second team, at high priority on
  that team's own quota. The shared borrowing pool is the same for every team, so only the
  team's own quota adds a new chance to start.
- The submit script got a `--team` option for this (local file).
- Whichever of the three tests starts first, I'll cancel the other two.
- The local run is back to ~36 s per iteration (the other user's job left GPU 7). It saved
  iteration 1,000 at 15:11, and the open-loop check of that checkpoint is running on GPU 3.

**15:35 EST (Mon Nov 2)** **Open-loop check, Cosmos iteration 1,000:** it now learns faster
per training example than GR00T.

Mean arm-joint error in rad; in brackets, error ÷ the error of holding still (lower is better):

| model | examples seen | train randomized | train clean | unseen randomized |
|---|---|---|---|---|
| Cosmos, iteration 500 | 1.04M | 0.046 (0.78) | 0.038 (0.59) | 0.045 (0.74) |
| **Cosmos, iteration 1,000** | 2.08M | **0.030 (0.51)** | **0.029 (0.44)** | **0.028 (0.46)** |
| GR00T mix200, step 30k | 1.92M | 0.039 (0.61) | 0.036 (0.55) | 0.037 (0.57) |
| GR00T mix200, step 60k | 3.84M | 0.030 (0.47) | 0.027 (0.41) | 0.026 (0.41) |

- Cosmos at 1,000 is about where GR00T was at 60k, with ~half the examples.
- The unseen demos score no worse than the trained ones, so no sign of memorization.
- Caveats:
  - This is the quick check (80 moments per set).
  - Lower prediction error doesn't guarantee more sim successes. At GR00T 60k, the 10-task
    randomized sim success was 16%.
- Model load took 696 s this time (vs ~90 s at iteration 500), probably because the shared
  drive was slow.

**15:35 EST (Mon Nov 2)** **First look at what goes well and badly for mix200_150k checkpoint-150000**
(`analysis/task_report.py`, CPU only; index rebuilt: 5,590 episodes, every 150k episode has
its video and its instruction record, all matching the logs).
- **Task length matters most.** Tasks whose clean demos take ≤150 steps (median) score 56%
  randomized; >350 steps, 6% (6 tasks). Rank correlation with demo length: −0.44.
- **Two-arm tasks score lower (32% vs 56%) but are also longer** (median demo 249 vs 147
  steps). Among short tasks (≤150 steps) both kinds score 56%. 50 tasks can't fully
  separate the two.
- **Not the step limit:** the limit is a median 2.8× the demo length, and successes take a
  median 0.96× the demo length.
- **Clean barely moved** vs 50task_60k on the same 581 scenes (93 newly solved, 71 lost);
  randomized gained a lot (382 newly solved, 23 lost on 984 scenes).

**16:12 EST (Mon Nov 2)** **2-node test passed; full 16-GPU Cosmos run launched** (as you
authorized).
- **Test** (fixed version, shared pool, high priority): started 15:55. I cancelled the two
  other queued copies.
  - **~10.1 s per iteration** on 16 H100s (9.8–10.35 s), vs 19.9 s on 8: almost exactly 2×.
  - Checkpoint at 20: saved in ~2 min. One copy was uploaded (166 GB), since the second node
    holds an identical copy. Marked as the latest.
  - Both nodes exited cleanly. The run landed in your own wandb (`rohin-garg/robotwin-posttrain`).
- **Full run `nano-mix200-2n`**, submitted 16:11:
  - from scratch: NVIDIA's base model, NVIDIA's LIBERO recipe, and their own 2-node × 8-GPU
    layout;
  - batch 2,048 = 32 per GPU × 16 GPUs × 4 accumulation;
  - 5,000 iterations, checkpoint every 500;
  - our team's own quota (no borrowing), normal priority, 24-hour limit.
  - Expected ~14.5 h of running after it starts.
- **Still untested on 2 nodes:** resuming from a checkpoint after an interruption. Unlikely to
  be needed without borrowing.
- **The local 5-GPU run continues**, as a cross-check: same recipe, batch 2,080 vs 2,048, so
  the loss curves should match. I'll stop it only with your OK once the cluster run passes it.

**16:30 EST (Mon Nov 2)** **The full 16-GPU Cosmos run (`nano-mix200-2n`) started**, after
19 min in the queue. At ~10.1 s per iteration it should finish ~Nov 3 07:00 EST.
- **Deleting the two test checkpoints (354 GB) failed:** this machine's storage role can
  write but not delete. Nothing was removed; it needs you or a role allowed to delete.
- **Checkpoint size:** 177 GB each.
  - model: 121 GB, the 15.2B weights at full precision plus their running average (EMA);
  - optimizer: 56 GB, two running averages per trainable weight.
- **What the training loss measures:** `train/loss` = 10 × (video-prediction loss + action
  loss), e.g. 0.848 = 10 × (0.080 + 0.0047) at iteration ~1,100 locally.
  - So the main loss curve mostly tracks future-frame prediction.
  - For actions, watch `train@2_detail/flow_matching_loss_action`, or our open-loop check.
- 16:35 EST: deleted the two test checkpoints (354 GB) with your OK, using a storage role
  that's allowed to delete. Their logs and wandb ids are kept.

**17:20 EST (Mon Nov 2)** **The 16-GPU cluster run was stopped at iteration ~240 and put back
in the queue.**
- It last logged at 17:16 (iteration 240, ~10.2–10.4 s per iteration, loss 1.18). By 17:19
  its status was "queued" again.
- No logs reached storage, so it was cut off abruptly, not a crash inside training. That
  points to being evicted (a higher-priority job takes the GPUs) or a node failure.
- No checkpoint existed yet (the first is at 500), so ~45 min on 16 GPUs is lost.
- When it gets nodes again, it starts over from the base model by itself, with a new wandb
  run.
- 17:28 EST: resubmitted the 16-GPU run at **high priority** (your call), so normal-priority
  jobs can't evict it.
  - Same run name and settings: from scratch, 5,000 iterations, checkpoint every 500, no
    borrowing, 24-hour limit.
  - I cancelled the requeued normal-priority job first. It had no checkpoint, so nothing was
    lost.
  - The local run keeps going as insurance until the cluster run has passed it and saved a
    checkpoint.
- 17:41 EST: the high-priority 16-GPU run started after 13 min in the queue. At ~10.2 s per
  iteration: first checkpoint (500) ~19:15 EST, finish ~Nov 3 08:30 EST.

**19:01 EST (Tue Nov 3)** **Eval viewer: training demo next to the rollout** (your request). Picking a
task now also plays one of its training demos from the mix200 dataset (clean demos for
clean, randomized for randomized; choose demo 0–49 or 0–199). Both players start at 10 steps
per second; the demo files are 15 frames per second and the rollouts 10, so side by side
they'd otherwise run at different speeds. Checked: all 12,500 demos map to the right task
(4 spot checks against known episodes, frame count = step count), and the app runs headless
in every view.
- 19:15 EST: the 16-GPU run's **first checkpoint (iteration 500) reached cloud storage**.
  - It's complete (177 GB), both nodes reported done, and it's marked as the latest.
  - No slowdown in training.
  - This is the first real-run test of the 2-node save and upload, and it worked.
  - Now checking that it loads in our eval code and gives an open-loop error similar to the
    local run's iteration 500 (on GPU 3, ~30 min including the download).
- 19:35 EST: **the 16-GPU run's iteration-500 checkpoint works in our eval code**, and matches
  the local run's iteration 500 on the quick open-loop check.
  - Ratio to holding still: cluster 0.72 / 0.58 / 0.72 vs local 0.78 / 0.59 / 0.74 (train
    randomized / train clean / unseen randomized). That's within the check's noise.
  - It loads in 99 s and takes 1.2 s per call.
  - Download of the model part only (114 GB): 6 min. The temporary copy is deleted.
  - So the cluster run can replace the local one once it has passed it.
- 20:42 EST: 16-GPU run iteration 1,000: checkpoint in cloud storage, loss ~1.0, ~10.3 s per
  iteration. The local run saved iteration 1,500; its open-loop check is running on GPU 3.

**21:05 EST (Mon Nov 2)** **Open-loop check, Cosmos (local run) iteration 1,500.** Ratio of
arm error to the error of holding still; lower is better:

| model | examples seen | train randomized | train clean | unseen randomized |
|---|---|---|---|---|
| Cosmos 500 | 1.04M | 0.78 | 0.59 | 0.74 |
| Cosmos 1,000 | 2.08M | 0.51 | 0.44 | 0.46 |
| **Cosmos 1,500** | 3.12M | **0.44** | **0.35** | **0.41** |
| GR00T mix200 60k | 3.84M | 0.47 | 0.41 | 0.41 |
| GR00T mix200 90k | 5.76M | 0.36 | 0.30 | 0.31 |
| GR00T mix200 150k | 9.6M | 0.28 | 0.24 | 0.24 |

- Still at least as good as GR00T per example seen: Cosmos 1,500 (3.1M) is between GR00T 60k
  and 90k.
- The gain is slowing: 0.51 → 0.44 from 1,000 to 1,500, after 0.78 → 0.51 from 500 to
  1,000.
- No sign of memorization (unseen ≈ trained).
- Model load took 882 s, slow again on the shared drive.

**22:50 EST (Mon Nov 2)** **Local Cosmos run stopped (your OK); its GPUs now run the first
Cosmos sim eval.**
- The 16-GPU cluster run passed the local run at 22:45 (iteration 1,700 vs 1,698), after
  saving three checkpoints and passing the load-and-compare check. I stopped the local run at
  22:47. Its checkpoints 500, 1,000 and 1,500 stay on disk.
- **Sim eval, Cosmos iteration 1,500** (local checkpoint): the 10 tasks, randomized, each
  task's first 10 scenes, the same scenes as GR00T's 10-task evals (`gr00t/paired_queue.py`).
  - 13 jobs, 100 episodes, `logs/queue_cosmos1500_rand10.txt`.
  - One stream per GPU on GPUs 0, 4, 5, 6, 7 (`benchmark_worker.sh` with
    `EVAL_SH=cosmos3/eval.sh`).
  - ~1.5–2 h.
  - Compare with GR00T mix200 on the same scenes: 60k, 150k, and decay60k.

**23:55 EST (Mon Nov 2)** **First Cosmos sim eval: iteration 1,500 ties GR00T's best model on
the same randomized scenes, with a third of the training examples.**
- 10 tasks, the same randomized scenes for every model (`gr00t/paired_compare.py --per-task`,
  which now includes Cosmos):

| model | training examples | randomized, same scenes |
|---|---|---|
| GR00T 50task_60k (clean demos only) | 3.84M | 17/98 (17%) |
| GR00T mix200 at 60k | 3.84M | 14/99 (14%) |
| GR00T mix200 60k + 10k decay | 4.5M | 46/100 (46%) |
| GR00T mix200 at 150k | 9.6M | 66/99 (67%) |
| **Cosmos mix200, iteration 1,500** | 3.1M | **66/99 (67%)** |

- Per task, Cosmos vs GR00T 150k:
  - handover_mic 7 vs 2;
  - place_object_stand 10 vs 5;
  - place_empty_cup 9 vs 7;
  - pick_dual_bottles 3 vs 7;
  - place_phone_stand 5 vs 7;
  - grab_roller 6 vs 9;
  - equal or within 1 on the rest.
  - With 10 scenes per task, single-task gaps are noisy; the 10-task total is the number to
    trust.
- Caveats:
  - randomized scenes only (clean not run);
  - the checkpoint is from the local 5-GPU run (batch 2,080), not the 16-GPU run;
  - 1,500 of 5,000 iterations.
- ~1 h of wall time on 5 GPUs, ~4.5 min per episode. GPUs 0, 4, 5, 6, 7 are released.

**01:00 EST (Tue Nov 3)** **Started a subagent on Flux 3 Action** (your request).
- Plan:
  1. find the model and its official training code;
  2. adapt it to the same mix200 data with the official recipe;
  3. package it for the cluster, modeled on the Cosmos package;
  4. run a test job (high priority);
  5. if the test works, launch the full run (no borrowing, normal priority; high if it gets
     evicted).
- Its cluster files stay local and uncommitted.
- It reports back before any run larger than 2 nodes or ~24 h, and stops at license gates or
  blocked permissions.
- The 16-GPU Cosmos run is at iteration ~2,450, healthy.

**01:20 EST (Tue Nov 3)** **The Flux 3 Action subagent stopped at step 1: the weights are
gated behind a non-commercial license that you'd have to accept.** Nothing was installed,
built or submitted.
- **The model** (from the subagent's research):
  - Black Forest Labs, released 2026-09-22.
  - A 7B "world action model": camera frames + robot state + instruction → the next 32
    actions plus future video, like Cosmos.
  - Open training code (Apache-2.0): `github.com/black-forest-labs/flux-action`.
  - Weights: `huggingface.co/black-forest-labs/flux-3-action-base`.
- **License:** FLUX Kommunity License v1.0, non-commercial.
  - A company with ≥ $5M revenue may use it only for testing, evaluation, or non-commercial
    R&D in a non-production environment.
  - Whoever accepts it for an employer states they can bind that employer.
  - Your HF account currently gets 403 on the weight files.
- **Compute (estimate from Black Forest Labs' published speeds, not measured):**
  - The official recipe (30k updates at batch 2,048) would be ~1,100 h on 1 node or ~550 h on
    2 nodes.
  - Even the short 3k-update schedule is ~55 h on 2 nodes.
  - 2 nodes × 24 h fits e.g. batch 512 × ~5.4k updates.
  - We'd need a measured speed, then a reduced schedule.
- Notes: `/mnt/efs/rohingarg/cri/flux3/NOTES.md`; official repo cloned (read-only) under
  `/mnt/efs/rohingarg/cri/flux3/flux-action`.
- 01:30 EST: **your decisions for Flux:**
  - the assignment fits the license terms;
  - run it on another team's quota, not ours;
  - train for 24–36 h of wall time (2 nodes) instead of the full recipe.
  - The subagent resumed: license access check → test job (high priority, measures speed) →
    reduced schedule sized to ~30 h, official hyperparameters otherwise → full run (no
    borrowing, normal priority; high if evicted).

**01:10 EST (Tue Nov 3)** **Sim evals of the 16-GPU Cosmos run's checkpoints, as they land**
(your request).
- For each of iterations 2,500, 3,000, 4,000 and 5,000:
  1. wait for the checkpoint in cloud storage;
  2. download the model part (~114 GB, ~6 min) to local NVMe;
  3. run the same 10 tasks × first 10 randomized scenes as the GR00T and Cosmos-1,500 evals,
     one eval stream per GPU on GPUs 3–7 (each waits for free memory).
- Results: `python3 gr00t/paired_compare.py --per-task` (labels `cosmos16 <iteration>`).
- The download-and-queue script is local (it contains the storage path).
- Expected: 2,500 eval ~01:15–02:30 EST; 5,000 lands ~08:30.

**02:10 EST (Tue Nov 3)** **Sim eval of the 16-GPU Cosmos run at iteration 2,500: 77/99 (78%)
randomized, ahead of GR00T's best (67%) and of Cosmos at 1,500 (67%), on the same 10-task
scenes.**
- `python3 gr00t/paired_compare.py --per-task`: see the table in my message.
- Cosmos 2,500 vs GR00T 150k, per task:
  - ahead: handover_mic 10 vs 2, place_object_stand 9 vs 5, place_empty_cup 10 vs 7,
    dump_bin_bigbin 9 vs 7;
  - behind: pick_dual_bottles 3 vs 7, move_can_pot 5 vs 6;
  - equal or within 1: the others.
- One eval without repeat runs. A ~10-point gap on ~100 scenes is larger than the usual
  noise, but repeat runs haven't measured it.
- Checkpoint downloaded from cloud storage to NVMe in 73 s.
- Next: iteration 3,000 (~02:30 EST) is evaluated automatically, then 4,000 and 5,000.

**03:35 EST (Tue Nov 3)** **Cosmos 16-GPU run at iteration 3,000: 79/99 (80%) randomized** on
the same 10-task scenes.
- Learning curve so far: 1,500 → 67%, 2,500 → 78%, 3,000 → 80%. GR00T's best is 67%.
- The gain from 2,500 to 3,000 (+2 episodes) is within noise; it may be flattening.
- Per task vs 2,500:
  - grab_roller 10 (was 9), move_pillbottle_pad 7 (6), move_can_pot 6 (5),
    pick_dual_bottles 4 (3);
  - handover_mic 9 (10), place_phone_stand 5 (6);
  - the rest the same.
- pick_dual_bottles is still Cosmos's weakest task vs GR00T (4 vs 7).
- Local copy of the checkpoint deleted after the eval. 4,000 (~05:20 EST) and 5,000
  (~08:30) are next, evaluated automatically.

**04:45 EST (Tue Nov 3)** **Flux 3 Action: package built and a 1-node test trained, but the
full run isn't launched. Needs your decision on how to queue it.**
- **Recipe:** the official full fine-tune for the DROID robot dataset (BFL's
  `configs/droid/train.json`), adapted only for our robot and data:
  - new 14-D action heads;
  - the official LeRobot indexer;
  - DROID camera layout, with our 240×320 images stretched to 360×640;
  - batch and length for the budget.
  - Optimizer, learning rates, loss weights, EMA etc. unchanged. Code is on a local branch.
- **1-node test** (high priority, borrowed pool):
  - trained, saved a 129 GB checkpoint, uploaded;
  - 69 s per update at batch 512, max 2 windows per GPU (72.8/79 GB);
  - **resume ran out of memory**: the fix is written but not yet tested on the cluster;
  - the wandb logging bug is fixed locally.
- **Draft full run:** 2 nodes, batch 512, 3,000 updates (official schedule scaled to 1/10),
  checkpoint every 150 updates (~1.5 h), ~30 h. Estimated from the 1-node speed.
- **Queue problem:** the second team's quota is full, with a long high-priority backlog. A
  normal-priority run would wait behind all of it. Our team's quota is also nearly full.
- **Still queued:** a 2-node test and a 1-node test of the resume fix (shared pool, high
  priority).
- **Asked you:** queue option (a–e), and whether to delete the 129 GB test checkpoint.
- Local notes: `/mnt/efs/rohingarg/cri/flux3/NOTES.md`.
