# Work log (RoboTwin / GR00T)

Plain-language, timestamped log of what was done, what the evals returned, and why
each decision was made. Newest entries at the bottom. Times are UTC. Technical
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
