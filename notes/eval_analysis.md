# Eval analysis

Started 2026-11-02; kept up to date as the analysis goes. What goes well and what goes badly for
our models, to decide what to improve next. Part 1 is the current best model, Cosmos3-Nano;
part 2 is the earlier GR00T analysis; part 3 is the FLUX 3 Action continuation.

## Conclusions so far (updated 2026-11-09)

- **Head-native canvas (head camera not upscaled) matches the DROID canvas at Flux update 3,000** (Nov 10): 70.8 / 74.5
  (weight 50) and 72.7 / 71.3 (weight 41.18) vs 71.0 / 71.1; no significant difference (p ≥ 0.16), while training ~28%
  faster per update. Cosmos mix500 finished training; its evals are being submitted (no scores yet).
  have been established in this status audit, so the measured rankings below are unchanged.

- **Cosmos3-Nano after its learning-rate decay (iteration 5,500) is the best model:** 85.0%
  randomized, 88.6% clean, on GR00T's exact scenes. GR00T 150k: 45.3% / 45.5%. Both trained on
  the same data (`mix200`).
- **FLUX 3 Action's continuation ended at 75.6% randomized, 77.8% clean (update 9,000)**, +4.5 / +6.6 over
  step 3,000 (p ≈ 0.005). It plateaued from 6,500, and the cool-down added nothing measurable. The run dipped at 5,000 after a loss spike and recovered by 6,000
  (part 3). It's still 6.6 points behind Cosmos 5,000 on both settings.
- **No Cosmos task is at 0%.** 14 tasks are under 80% on randomized; the lowest is hanging_mug
  (30%).
- **For Cosmos, task length matters little; using both arms is the weak spot** (80% vs 89%
  randomized). For GR00T, long tasks failed most (part 2).
- **Within a task, Cosmos fails the same way each time.** The most common failure: the first
  grasp knocks the object over (gripper not open enough, or off center). Then placement near
  misses, then no recovery after a mistake.
- **43% of Cosmos's failed scenes succeed on a second try** with the same checkpoint, scene and
  instruction sentence (89 of 209). Counting a scene as solved if either try worked: randomized
  91.4%, clean 93.4%. The repeat failures concentrate in hanging_mug and handover_block.
- **82 failed scenes (5.4% of 1,514) failed all three tries** (5,500 twice, 6,000 once); 16 of
  them are hanging_mug. The other 127 failed scenes succeeded at least once, so picking better
  among the model's own attempts could add up to ~8 points. The 82 need behavior the model doesn't
  show yet.
- **Some failures come from the success check or the simulator:**
  place_mouse_pad's check allows 1.5 × 1.2 cm and one direction; 30 of the 50 tasks require
  both grippers open at the end; stack_bowls_three seed 100007 looks like a physics glitch.
- **Open question:** in two tasks without a handover, one arm tries to take the object from
  the other. Guess (untested): behavior from the handover tasks leaks into other two-arm tasks.

## Part 1: Cosmos3-Nano, decay checkpoint 5,500 (current best)

### 1.1 The eval

- **Model:** Cosmos3-Nano (16B), trained on `mix200` (50 clean + 200 randomized demos per task)
  for 5,000 iterations on 16 cluster GPUs, then 1,000 more iterations with the learning rate
  falling to 0 (run `cosmos3_nano_mix200_2n_decay1k` in `notes/runs/README.md`; logs use the
  label `cosmos3_nano_mix200_2n_decay`). Iteration 5,500 is halfway through that decay; 6,000
  scores the same.
- **Scenes:** the same as GR00T 150k's. Randomized: all of GR00T's seeds, ~20 per task (1,015
  episodes). Clean: GR00T's first 10 per task (499).
- **Result:** randomized 863/1,015 = 85.0%, clean 442/499 = 88.6%.
- **Same tasks as in training, new scenes.** Test seeds start at 100,000; the clean training
  demos used seeds 0 to about 200. Randomized tests use instruction sentences that never
  appeared in training ("unseen").

### 1.2 Per task

- **Randomized:** 36 tasks at 80% or more, 13 at 41–79%, 1 at 30% (hanging_mug), none at 10% or
  less. 13 tasks are at 100%: adjust_bottle, beat_block_hammer, click_alarmclock, click_bell,
  grab_roller, open_laptop, place_burger_fries, place_container_plate, place_empty_cup,
  shake_bottle, shake_bottle_horizontally, stack_blocks_two, stack_bowls_two.
- **The 14 tasks under 80% on randomized** (successes / episodes, iterations 5,500 and 6,000):

  | task | randomized 5,500 | randomized 6,000 | clean 5,500 | clean 6,000 | arms | demo steps |
  |---|---|---|---|---|---|---|
  | hanging_mug | 6/20 | 8/20 | 1/10 | 2/10 | both | 335 |
  | handover_block | 9/20 | 9/20 | 6/10 | 3/10 | both | 279 |
  | put_bottles_dustbin | 11/20 | 11/20 | 9/10 | 9/10 | both | 656 |
  | pick_diverse_bottles | 12/21 | 14/21 | 7/10 | 6/10 | both | 121 |
  | place_mouse_pad | 12/20 | 14/20 | 8/10 | 7/10 | one | 148 |
  | scan_object | 13/20 | 17/20 | 9/10 | 8/10 | both | 169 |
  | turn_switch | 13/20 | 15/20 | 9/10 | 8/10 | one | 96 |
  | rotate_qrcode | 13/20 | 13/20 | 8/10 | 9/10 | one | 153 |
  | place_dual_shoes | 15/22 | 15/22 | 6/10 | 10/10 | both | 222 |
  | place_fan | 14/20 | 15/20 | 9/10 | 8/10 | one | 145 |
  | move_can_pot | 15/20 | 14/20 | 10/10 | 10/10 | one | 151 |
  | put_object_cabinet | 15/20 | 14/20 | 8/10 | 8/10 | both | 269 |
  | move_stapler_pad | 16/21 | 14/21 | 8/10 | 8/10 | one | 156 |
  | stack_bowls_three | 16/21 | 18/21 | 9/10 | 9/10 | both | 471 |

- **Single tasks are noisy:** 5,500 and 6,000 score the same overall, yet some tasks differ a lot
  between them, e.g. place_can_basket randomized 17/20 vs 8/20.

### 1.3 What it's good at and bad at

- **Length matters little.** Randomized success by the task's demo length (median steps): 88%
  (150 or fewer), 87% (151–250), 77% (251–350), 80% (over 350). Rank correlation with length:
  −0.18 (GR00T: −0.44). Long tasks GR00T couldn't do: stack_blocks_three 0/20 → 17/20,
  open_microwave 0/20 → 18/19, blocks_ranking_rgb 2/21 → 19/21. Both trained on the same data, so
  GR00T's trouble with long tasks came from the model or its training setup.
- **Two-arm tasks are weaker:** randomized 80% vs 89% for one-arm tasks, clean 83% vs 93%, and
  lower in every length group (83 vs 89%, 84 vs 89%, 73 vs 100%, 78 vs 95%). 8 of the 14 tasks
  under 80% use both arms.
- **Scene by scene against GR00T 150k** (1,514 scenes both ran): Cosmos solves 654 scenes GR00T
  failed; GR00T solves 35 scenes Cosmos failed.

### 1.4 Do failures repeat?

Checkpoints 5,500 and 6,000 on the 1,514 scenes both ran:

| | 6,000 succeeded | 6,000 failed |
|---|---|---|
| **5,500 succeeded** (1,305) | 1,205 | 100 |
| **5,500 failed** (209) | 93 | 116 |

- A success usually stays a success (100 of 1,305 flip, 7.7%). A failure often doesn't (93 of
  209 flip, 44%). So about half of the failures are scenes Cosmos can do, but not every time.
- Scenes both failed, most per task: hanging_mug 17 of 30, handover_block 11, pick_diverse_bottles
  9, rotate_qrcode 7, turn_switch 6.
- **Rerun of 5,500 itself on its 209 failed scenes** (2026-11-05, cluster, ~30 GPU-hours; label
  `cosmos3_nano_mix200_2n_decay_rerun1`): every scene got the same instruction sentence as in
  the original episode (checked against each episode record).
  - **89 of 209 succeed (43%):** randomized 65/152, clean 24/57. Checkpoint 6,000 solved 93 of the
    same 209 (44%), so the difference between the two checkpoints mattered about as much as a
    plain retry.
  - **Why a retry can differ:** the Cosmos policy draws fresh random noise for every chunk of
    actions; its random generator is seeded once per policy-server process, not per episode.
  - **Counting a scene as solved if either try worked:** randomized 91.4%, clean 93.4% (one try:
    85.0% / 88.6%).
  - **Failures that repeated** (failed again / failed scenes): hanging_mug clean 9/9 and
    randomized 11/14, handover_block randomized 8/11, put_bottles_dustbin randomized 7/9,
    place_mouse_pad randomized 6/8, place_fan randomized 5/6, move_stapler_pad 6/7,
    put_object_cabinet 6/7.
  - Source: LOG.md 2026-11-05 20:45 EST; the cluster eval package's `per_scene.csv`.
- **Three tries per failed scene** (2026-11-09): 5,500's original try, its rerun, and checkpoint
  6,000. 6,000 is a different checkpoint, but the rerun showed the two differ about as much as a
  plain retry.
  - Of the 209 scenes 5,500 failed, **82 failed all three tries** (clean 24, randomized 58), 72
    succeeded once, 55 succeeded twice.
  - The 82 are 5.4% of the 1,514 scenes, spread over 24 tasks: hanging_mug 16, handover_block 7,
    put_object_cabinet, pick_diverse_bottles and rotate_qrcode 5 each.
  - Counting a scene as solved if any of the three tries worked: 1,432 of 1,514 = 94.6%. That is
    the most that choosing the best of the three attempts could reach.
  - Reproduce: `analysis/.venv/bin/python analysis/retry_split.py cosmos3_nano_mix200_2n_decay:5500
    cosmos3_nano_mix200_2n_decay_rerun1:5500 cosmos3_nano_mix200_2n_decay:6000`

### 1.5 What the failures look like

Two reviews: 13 failures I looked at frame by frame (one randomized failure per weak task,
2026-11-05), and your 27 notes in the viewer (22 on 5,500, 5 on Cosmos 5,000; viewer
"notes" view, file `analysis/tags/notes.sqlite`).

- **The first grasp knocks the object over** (12 of your notes): the gripper isn't open wide
  enough, or not centered, so it hits the object and flips or pushes it. handover_block,
  pick_diverse_bottles, pick_dual_bottles.
- **Placement near misses:** the handover block lands half on the pad and falls; the mug misses
  the hook by a bit; the mouse is off center or not turned; a bowl is dropped from too high.
- **No recovery after a mistake,** often with the gripper opening and closing over and over.
  When an object falls over, Cosmos never picks it up again (5 of my 13).
- **One arm takes the object from the other arm,** in tasks without a handover
  (pick_diverse_bottles seed 100024, pick_dual_bottles seed 100003).
- **Wrong order or target, rarely:** blocks_ranking_size (order), place_fan (went toward a red
  clutter object; a guess), stack_bowls_three seed 100012 (couldn't pick the base bowl).
- **Cosmos doesn't freeze:** in 90% of the 209 failures the robot still moves in the last 5% of
  the episode, like in successes (`analysis/motion.py`).
- Within a task, the same mistake recurs.

### 1.6 Success checks and the simulator

- **place_mouse_pad** (`RoboTwin/envs/place_mouse_pad.py:118`): the mouse must end within
  1.5 × 1.2 cm of the pad's center, turned to one particular direction, with both grippers open.
  "Not perfectly centered" and "not turned another 90 degrees" fail under it.
- **30 of the 50 tasks require both grippers open at the end,** so "placed but never let go"
  fails (move_stapler_pad seed 100005).
- **stack_bowls_three seed 100007:** at step 340 a bowl is set down tilted on the edge of the
  stack; by step 360 all three bowls are gone from view, faster than any arm moves. Most likely
  the physics engine threw the overlapping bowls.

### 1.7 Open questions

- Does handover behavior leak into other two-arm tasks? Count how often one arm takes the
  object from the other.
- Do the scripted demos open the gripper wider, or approach more centered, than Cosmos? Compare
  with the training-demo panel in the viewer.
- How wide does Cosmos actually open the gripper? Videos can't measure it; the eval records only
  the head-camera video. Recording full rollouts (all cameras, joint positions, actions) would,
  and every hill-climbing method needs that data anyway.

### 1.8 Next steps for hill climbing (updated 2026-11-09)

**Goal:** make success less random on scenes Cosmos can already do, and teach it to recover after
a mistake. Nothing below is running; most of it waits for GPUs.

**Recording Cosmos's runs:** to train on its own runs, they must be saved in the same form as the
training demos: at every step, the images from all three cameras Cosmos sees (head and both
wrists), the robot's joint positions, and the moves Cosmos chose. Today each episode saves only
the head-camera video and a one-line summary (success, steps, seed, instruction).
- Option 1 needs this added to the eval.
- Experiment 2b needs it, or a saved simulator state, to restart from the middle of a Cosmos run.
- RL code records this itself while it runs.
- Experiment 2a doesn't need it.

**1. Training on its own successes** (you want to try this; blocked on GPUs)
- What: run Cosmos many times on scenes outside the test set (e.g. training seeds), keep the
  episodes that succeeded, and train it a bit more on those.
- Why: on one scene Cosmos acts differently from try to try, because each plan starts from random
  numbers. Training on the tries that worked makes those moves more likely, so the scene succeeds
  more often.
- Aims at: the 127 failed scenes that succeeded in at least one of three tries (1.4). At most ~8
  points, if all of them became reliable.
- Can't do: teach behavior Cosmos never shows (the 82 scenes that failed every try, recovery).
- Rough cost (estimate, not measured): 1,000 episodes × ~4 GPU-minutes ≈ 67 GPU-hours of runs,
  plus a short fine-tune from decay 5,500.
- Reading: "Advantage-weighted & filtered BC" on Climbing the Nines.

**2. Recovery data** (experiments worth doing; not started)
- Why: Cosmos never recovers after a mistake, and its training demos contain no mistakes. They
  come from RoboTwin's scripted program, which reads exact object positions from the simulator.
- Plan: let Cosmos run; when it goes wrong, give control to the scripted program from that exact
  simulator state; record how it finishes; train Cosmos on those recordings; repeat.
- Experiments, in order:
  - a. Can the scripted program finish a task from a saved mid-episode state? Read its code,
    then test from saved states, task by task. If it can't, the plan stops here.
  - b. Hand over at many points: in a failed Cosmos episode, give control to the scripted program
    at regular steps (e.g. every 20) and keep the handovers that succeed. This avoids having to
    find the exact moment the episode went wrong, at the cost of more sim runs.
  - c. Open question: there's no deterministic way yet to find when an episode goes wrong. The
    simulator knows exact object positions, so a single failure kind can be checked by a rule
    (e.g. "the bottle tilted more than 45°"), but each kind needs its own rule.
- Reading: "Human corrections & DAgger" on Climbing the Nines; SimPT (arXiv 2610.29161), which
  did this on RoboTwin with a scripted expert (67% → 82%) and had to add per-task code to
  restart the expert mid-task.

**3. RL** (blocked on GPUs; uncertain)
- What: start from the current Cosmos and adjust it using whether each run succeeded.
- Why: the only option that might discover recovery on its own. Uncertain, because it can only
  reward behavior that shows up in its runs, and Cosmos has shown no recovery so far.
- Cost: far more runs than option 1.
- Reading: "On-policy policy gradients" on Climbing the Nines, with its notes on flow and
  diffusion policies (Cosmos builds each plan from random numbers in 30 refining rounds).

**Set aside, with your reasons:**
- Changing Cosmos's settings at test time (how many moves it carries out before planning again,
  how many refining rounds): it departs from training, with no failure it targets.
- Sampling several plans at each moment and picking one with a scoring model: the model's own
  randomness stays, and training the scoring model is a hard problem of its own.
- Making mistakes on purpose in the scripted demos (knock or shift the object partway): easy to
  describe, but there's no reliable way to make such mistakes match Cosmos's real ones.

**Also:** the same mistake every time on a few tasks (hanging_mug, handover_block,
pick_diverse_bottles) may mean a skill is missing there; more demos for those tasks may help.
mix500 (500 randomized demos per task, training now) adds data for every task.

**Reproduce part 1** (after `analysis/.venv/bin/python analysis/build_index.py`):
- `analysis/.venv/bin/python analysis/task_report.py cosmos3_nano_mix200_2n_decay 5500`
- `analysis/.venv/bin/python analysis/length_spread.py cosmos3_nano_mix200_2n_decay 5500`
- `analysis/.venv/bin/python analysis/repeat_check.py cosmos3_nano_mix200_2n_decay:5500 cosmos3_nano_mix200_2n_decay:6000`
- `analysis/.venv/bin/python analysis/repeat_check.py mix200_150k:150000 cosmos3_nano_mix200_2n_decay:5500`

## Part 2: GR00T mix200_150k, checkpoint 150000 (earlier analysis)

### 2.0 The eval in brief

- **Model:** GR00T N1.7 trained for 150k steps on 50 tasks, with 50 clean + 200 randomized
  demos per task (`mix200_150k`), checkpoint 150000.
- **Result:** clean 46.0%, randomized 45.2% (the average of the 50 per-task success rates).
- **Test episodes per task:**
  - randomized: about 20, for every task;
  - clean: 20 for 10 tasks, 10 for the other 40.
- **Same setups as the older model.** A number called the seed fixes each episode's
  starting setup: which objects, where they sit, the instruction sentence, and in randomized
  episodes also the clutter, background, lights and table height. This model was tested with
  the same seeds as the earlier `50task_60k` model.
- **Data:** `analysis/cache/episodes.parquet`, rebuilt with
  `analysis/.venv/bin/python analysis/build_index.py`.

### 2.1 Longer tasks succeed less

**Task length** here means how many steps the task's training demos take. One step is one
action; the demos record 15 steps per second. Each task's length is the median over its 50
clean demos.

| demo length (median steps) | tasks | randomized success | clean success | randomized success, lowest to highest task |
|---|---|---|---|---|
| 150 or fewer | 20 | 56% | 57% | 5% (lift_pot) to 100% (click_bell) |
| 151-250 | 17 | 50% | 50% | 5% (place_dual_shoes) to 100% (shake_bottle) |
| 251-350 | 7 | 35% | 36% | 0% (handover_block) to 90% (stack_bowls_two) |
| over 350 | 6 | 6% | 12% | 0% (open_microwave) to 19% (stack_bowls_three) |

Success is the average of the tasks' rates in each group. Randomized success falls from 56%
to 50%, 35% and 6% as tasks get longer. A task's demos barely vary in length, so its median
is a fair summary (checked in appendix A).

#### What this doesn't show

- **Length is not the only reason.** Short tasks fail too: of the 20 tasks of 150 steps or
  fewer, 6 score 35% or less on randomized: lift_pot 5%, place_fan 15%, place_mouse_pad 15%,
  place_object_scale 20%, stamp_seal 33%, turn_switch 35%.
- **Long tasks mostly use both arms** (5 of the 6 tasks over 350 steps). So this can't tell
  whether the length itself or the two-arm coordination is what's hard.
- **Single-task numbers are rough.** Each task has about 20 randomized episodes and 10 or
  20 clean ones. Group averages are steadier than any one task's rate.

**Reproduce:** `analysis/.venv/bin/python analysis/length_spread.py` prints these tables
(and the ones in the appendix).

### 2.2 Clean success didn't improve with 200 more demos per task

- **Clean:** 42.6% (`50task_60k`) → 46.0% (150k). On the 10 tasks with 20 clean episodes:
  62% → 65%. Over the 581 clean scenes both models ran, 93 are solved only by the new model
  and 71 only by the old one.
- **Randomized:** 9.7% → 45.2%. Clean and randomized now score about the same.
- **The new model trained more overall, but less on the clean demos** (assuming training
  picks frames evenly, as the run card's pass counts do):

  | | 50task_60k | mix200_150k |
  |---|---|---|
  | training samples | 60,000 steps × 64 = 3.84M | 150,000 steps × 64 = 9.6M |
  | training frames | 548,893, all clean | 2,759,740, of which 548,893 (20%) clean |
  | passes over the clean demos | 3.84M / 548,893 = 7.0 | 9.6M × 20% / 548,893 = 3.5 |

- **The randomized demos almost never look clean:** every table is cluttered, and the
  background is plain in only 2% of them (`clean_background_rate: 0.02` in
  `RoboTwin/env_cfg/task_config/demo_randomized.yml`).

**Reading:** the extra demos taught the model to handle clutter, but not to do the tasks
better. Seeing the scene no longer looks like the main problem; doing the task does.

#### Possible causes

We focus on the first two for now.

1. **The parts of GR00T we don't train.** Our runs train the action model (the part that
   turns features into actions), the projectors, and a small normalization layer on the
   VLM's output. The VLM itself, i.e. the vision encoder that reads the camera images and
   the language model, stays as NVIDIA released it (`tune_visual: false`, `tune_llm: false`,
   `tune_top_llm_layers: 0` in the checkpoint's `experiment_cfg/final_model_config.json`).
   If what the VLM passes on doesn't carry what the action needs, for example exactly where
   a handle is, the action model can't make up for it.
   - Idea: train the VLM too, all of it or only its top layers.
   - Idea: before paying for that, look at what the action model receives from the VLM, to
     see whether the VLM is the problem. Possible ways, not tried:
     - from the VLM's outputs on demo frames, train a small predictor of something the
       action needs (e.g. where the target object is). If it can't, that information isn't
       getting through;
     - swap in the VLM output of a different frame and measure how much the predicted
       actions change. If they hardly change, the action model barely uses the images.
2. **No recoveries in the demos.** RoboTwin's demo recorder keeps a run only if its scripted
   robot succeeded (`RoboTwin/scripts/collect_data.py:179–191`), and the downloaded demos'
   `seed.txt` skips numbers (5, 12, 17, …), which fits failed runs being thrown away. So the
   model has never seen a failed attempt, and as far as we know none where a mistake gets
   fixed. Once it drifts a little from the demos, it may not know how to get back. This
   would also fit section 2.1: longer tasks give more chances to slip.
   - Check: in the videos, look for failures where a small mistake (a bump, a drop, a grasp
     slightly off) is never corrected.
   - Possible fixes, later: demos that include recoveries, or RL (training from the robot's
     own attempts).
3. **Luck (not measured).** The model's own randomness can make the same scene succeed once
   and fail the next time. We have never run one model twice on the same scenes, because
   evals are expensive, so we can't say how much of the scene-by-scene difference between
   the two models is luck.

### 2.3 What the clean failures look like (visual review, first pass)

Watched in the eval viewer (2026-11-02): clean failures of checkpoint 150000 on 3 tasks with
20 clean episodes each and middling success, next to its successful runs. Randomized
failures and the training demos haven't been watched yet.

**The gripper can do these tasks.** Before every eval episode, RoboTwin's scripted robot
runs the same scene with the same gripper, and the scene is used only if it succeeds
(`RoboTwin/scripts/eval_policy_xpolicylab.py:788`; on in our evals). So every object below
was picked up and placed successfully by this gripper. The failures come from how the policy
uses it: how wide it opens, where it grabs, how fast it moves.

#### move_can_pot (6/20): put the sauce can next to the pot

7 failures watched:
1. Grabbed the pot instead of the sauce can. It couldn't hold the pot, pushed it further
   away and tipped it over.
2. The gripper wasn't open wide enough for the can, so small pushes tipped it over.
3. The same as 2.
4. Opened wide enough and picked the can up, but placed it badly. It tried to fix that but
   couldn't pick the can up again.
5. Placed the can not close enough to the pot. The can wobbled; it tried to steady it and
   couldn't. Placing it too fast also tips it over.
6. Kept trying to grab the can at a spot where there was no can (seed 300004).
7. Couldn't lift the slippery can; the gripper wasn't placed well on it.

Most failures are about picking up the can: 5 of 7 are grip or placement problems (2, 3, 4,
5, 7), 1 grabbed the wrong object (1), and 1 grabbed at an empty spot (6). When it does pick
the can up properly, the rest is straightforward. It tried to fix a mistake twice (4, 5),
and both fixes failed.

#### place_phone_stand (7/20)

5 to 7 failures watched, all much alike. The phone is thin and slippery, and there are 2
ways to hold it, both slippery. The policy often can't pick up a thin phone; when it does,
the phone often slips, and placing it on the stand is hard. No good attempts to recover.

#### handover_mic (8/20)

Grabbing the mic with the first hand works. The handover to the second hand fails:
- The two grippers get in each other's way when they're close.
- The second hand grabs the mic too low: several times it went for a spot even lower than
  the first hand, which was already holding the mic low.
- Or the first hand takes up too much of the mic, leaving no room for the second.

In the successes, the first hand held the mic very low, leaving room at the top for the
second hand. When that doesn't happen, there's no attempt to recover. It looks as if the
policy has no good sense of where the other arm and the object are at the same time.

#### What this suggests so far

- **Most failures are near misses:** grip width, where it grabs, how it places. That fits
  cause 1 in section 2.2 (precision).
- **Mistakes almost never get fixed.** Usually no fix is tried, and the two that were tried
  failed. That fits cause 2 (no recoveries in the demos).
- **handover_mic adds a third kind:** the two arms don't work together on one object.
- **Not checked yet:** how the scripted demos grab the same objects (how wide the gripper
  opens, where it grabs, how slowly it approaches). The viewer now plays a training demo next
  to each rollout.

#### Next

1. Randomized failures of the same 3 tasks (move_can_pot 10/20, place_phone_stand 10/20,
   handover_mic 3/20): are they the same kinds as on clean, or new ones caused by the
   clutter, like grabbing a clutter object?
2. The short tasks that score low (section 2.1), where length can't be the reason: lift_pot,
   place_fan, place_mouse_pad, place_object_scale, stamp_seal, turn_switch.

## Part 3: FLUX 3 Action, continuation from step 3,000 (in progress)

Same 50 tasks and exact seeds as the other models (1,015 randomized, 499 clean). Full evals every 1,000 updates
plus the 500s; the run is going to 9,000.

| checkpoint | randomized | clean |
|---|---|---|
| step 3,000 (first run) | 71.0% | 71.1% |
| 4,000 (before the spike) | 72.1% | 72.9% |
| 5,000 | 64.0% | 61.9% |
| 5,500 | 66.3% | 67.1% |
| 6,000 | 71.8% | 71.3% |
| 6,500 | 74.1% | 76.0% |
| 7,000 | 74.8% | 77.0% |
| 7,500 | 75.4% | 74.5% |
| 8,000 | 75.0% | 76.8% |
| 8,500 | 74.9% | 78.8% |
| **9,000 (end of cool-down)** | **75.6%** | **77.8%** |
| Cosmos 5,000 (same eval) | 81.9% | 84.4% |

Paired by seed ("only A solves" vs "only B solves"):
- 6,500 vs 6,000: randomized 110 vs 87 (p = 0.12); clean 61 vs 38 (p = 0.027).
- 6,500 vs 3,000: randomized 132 vs 101 (p = 0.049; by task 24 better, 13 worse); clean 69 vs 45 (p = 0.031;
  by task 24 better, 12 worse).
- 6,500 vs Cosmos 5,000: Cosmos ahead by 8.0 points randomized (p = 3e-7) and 8.4 clean (p = 4e-4).
- 7,000 vs 6,500: randomized 108 vs 101 (p = 0.68); clean 47 vs 42 (p = 0.67). No significant difference.
- 7,000 vs 3,000: randomized 135 vs 97 (p = 0.015; by task 25 better, 15 worse); clean 77 vs 48 (p = 0.012; by task
  26 better, 11 worse).
- 7,000 vs Cosmos 5,000: Cosmos ahead by 7.3 points randomized (p = 3e-6) and 7.4 clean (p = 0.001).
- 7,500 vs 7,000: randomized 105 vs 99 (p = 0.73); clean 40 vs 52 (p = 0.25). No significant difference.
- 7,500 vs 3,000: randomized 136 vs 92 (p = 0.004; by task 26 better, 15 worse); clean 69 vs 52 (p = 0.15).
- 7,500 vs Cosmos 5,000: Cosmos ahead by 6.7 points randomized (p = 1e-5) and 9.8 clean (p = 2e-5).
- 8,000 vs 7,000: randomized 104 vs 102 (p = 0.94); clean 47 vs 48 (p = 1.0). Level.
- 8,000 vs 3,000: randomized 132 vs 92 (p = 0.009; by task 26 better, 12 worse); clean 77 vs 49 (p = 0.016; by task
  26 better, 7 worse).
- 8,000 vs Cosmos 5,000: Cosmos ahead by 7.2 points randomized (p = 4e-6) and 7.6 clean (p = 0.001).
- 8,500 vs 8,000: randomized 83 vs 84 (p = 1.0); clean 49 vs 39 (p = 0.34). Level.
- 8,500 vs 3,000: randomized 134 vs 95 (p = 0.012; by task 24 better, 12 worse); clean 76 vs 38 (p = 5e-4; by task
  26 better, 8 worse).
- 8,500 vs Cosmos 5,000: Cosmos ahead by 7.2 points randomized (p = 4e-6) and 5.6 clean (p = 0.013).
- **9,000 (final) vs 3,000:** randomized 149 vs 103 (p = 0.004; 29 tasks better, 13 worse); clean 81 vs 48
  (p = 0.005; 27 better, 9 worse). The continuation adds +4.5 / +6.6 points.
- 9,000 vs Cosmos 5,000: Cosmos ahead by 6.6 points on both settings (randomized p = 2e-5, Cosmos better on 33 tasks,
  Flux on 8; clean p = 0.002).
- Per task: the biggest gains over 3,000 are place_cans_plasticbox (+11 of 30) and put_bottles_dustbin (+8). The
  losses are open_microwave (−8 of 29), stack_bowls_three (−7) and turn_switch (−5). Flux leads Cosmos on 11 tasks,
  mostly by 1–3 (pick_dual_bottles 25 vs 19). The per-task table is in notes/results.md.
- Held-out arm error at 9,000: 0.0285 / 0.0204 / 0.0233, the lowest of all checkpoints.

Reading:
- The dip at 5,000 followed a loss spike near 4,790 that the averaged (EMA) weights carried into the export.
  The spike investigation is in notes/investigations.md.
- At 6,500 the weights from before that spike are ~9% of the EMA, and those after a smaller spike at 6,460
  are ~5%.
- **6,500 through 9,000 sit on a plateau** (randomized 74.1–75.6%, clean 74.5–78.8%); the differences between
  them are within noise. The cool-down (8,000 → 9,000) added no measurable gain.
- **The gain over step 3,000 holds at 7,000** (p ≈ 0.01 on both settings, two checkpoints in a row above
  3,000). 6,500 → 7,000 is flat, within noise. Several checkpoints have been compared against 3,000, so read
  single p-values with that in mind.
- Held-out arm error at 7,000 is back at the 3,000–4,000 level (0.0297 / 0.0220 / 0.0245).
- **Update 4,000 (before the spike) was still at the step-3,000 level:** randomized 72.1% (127 vs 116 seeds
  against 3,000, p = 0.52), clean 72.9% (57 vs 48, p = 0.44). Against 9,000 it is lower on both (randomized 105
  vs 140, p = 0.03; clean 42 vs 66, p = 0.03); against 6,500 and 7,000 the gaps (2–4 points) are not
  significant (p 0.09–0.22). So the gain over 3,000 arrived between 4,000 and 6,500, after the spike, not before
  it. Same seeds and the same instruction sentence per episode as the 6,500 and 7,000 evals.
- The 4,000 eval ran two eval streams per GPU: each stream ran 2.0x slower (0.96 vs 0.48 s per executed action;
  policy calls 2.4 s vs 0.79 s), so GPU time per action was unchanged (0.54 s) and the eval cost 75.7 GPU-hours
  vs 72.6–72.9 for the 1-stream evals. Packing two streams per GPU does not save GPU time.

### Part 3b: head-native canvas at update 3,000 (Nov 10)

Same recipe and data as Flux step 3,000; the head camera stays at its native 240×320 (canvas 448×640) instead of being
upscaled ~3× into the DROID canvas. Two loss weights (50, 41.18). Full 50-task evals, same seeds.

| model | randomized | clean |
|---|---|---|
| Flux step 3,000 (DROID canvas) | 71.0% | 71.1% |
| head-native, weight 50 | 70.8% | 74.5% |
| head-native, weight 41.18 | 72.7% | 71.3% |

- No significant difference in any pairing (episode sign tests p = 0.16–0.95). At 1,500 the 10-task checks also tied.
- So the upscaling buys nothing measurable, and dropping it makes training ~28% faster per update. A candidate default
  for the next Flux run (mix500), together with the compile speedups (combined speed not yet measured).
- Tables and sign tests: `notes/results.md`.

## Appendix A: is the median a fair summary of a task's length?

Yes, for 45 of the 50 tasks: their demos barely vary in length.
- **Typical spread** (standard deviation divided by mean, median over the 50 tasks): 3.7% for
  the clean demos, 4.3% for the randomized demos. For example, click_bell's 50 clean demos
  take 74 to 86 steps.
- **5 tasks vary by 10% or more:**

  | task | spread | shortest to longest demo | middle half of demos | median | effect on its group |
  |---|---|---|---|---|---|
  | dump_bin_bigbin | 34% | 157 to 349 | 169 to 338 | 176 | could be 151-250 or 251-350 |
  | open_microwave | 26% | 413 to 894 | 427 to 463 | 431 | over 350 either way |
  | place_bread_basket | 22% | 157 to 329 | 225 to 290 | 229 | could be 151-250 or 251-350 |
  | open_laptop | 20% | 146 to 354 | 182 to 227 | 222 | middle half stays in 151-250 |
  | put_bottles_dustbin | 12% | 404 to 824 | 530 to 664 | 656 | over 350 either way |

  Moving dump_bin_bigbin and place_bread_basket into 251-350 changes randomized success of
  the two middle groups from 50% and 35% to 50% and 39%. The pattern stays.
- **Other ways of measuring length give the same answer.**
  - Using the median of the 200 randomized demos instead moves 1 task to another group:
    place_mouse_pad (147.5 vs 151.5 steps, right at the 150 edge).
  - Ranking the 50 tasks by length and by randomized success, the two rankings have a
    rank correlation of −0.44, whether length is the median of the clean demos, the median
    of the randomized demos, or the mean of the clean demos.

## Appendix B: every task (GR00T 150k)

Demo length in steps, over each task's 50 clean demos. Spread is the standard deviation
divided by the mean. Success is checkpoint 150000's.

| task | median | middle half | min to max | spread | randomized | clean |
|---|---|---|---|---|---|---|
| click_bell | 77 | 75 to 78 | 74 to 86 | 3% | 100% | 95% |
| click_alarmclock | 83 | 79.25 to 88.5 | 76 to 102 | 8% | 95% | 100% |
| grab_roller | 93 | 90.25 to 96 | 87 to 107 | 5% | 90% | 90% |
| turn_switch | 96 | 93 to 97 | 90 to 107 | 4% | 35% | 40% |
| lift_pot | 111 | 108 to 113 | 106 to 118 | 3% | 5% | 10% |
| beat_block_hammer | 113.5 | 111.25 to 116 | 108 to 125 | 3% | 55% | 40% |
| move_playingcard_away | 116 | 115 to 118 | 114 to 131 | 3% | 55% | 60% |
| press_stapler | 117.5 | 115 to 122.75 | 112 to 130 | 4% | 67% | 90% |
| pick_diverse_bottles | 121 | 118 to 124 | 110 to 132 | 4% | 48% | 50% |
| pick_dual_bottles | 123 | 119 to 126 | 112 to 135 | 4% | 80% | 60% |
| place_phone_stand | 126 | 126 to 128 | 123 to 139 | 2% | 50% | 35% |
| place_object_stand | 139 | 137 to 139 | 134 to 149 | 2% | 55% | 75% |
| adjust_bottle | 143 | 139 to 146.75 | 133 to 156 | 4% | 90% | 100% |
| move_pillbottle_pad | 143.5 | 139 to 155 | 132 to 171 | 7% | 55% | 60% |
| place_object_scale | 144 | 141 to 148 | 135 to 164 | 4% | 20% | 10% |
| stamp_seal | 144 | 139 to 150.75 | 133 to 159 | 5% | 33% | 30% |
| place_fan | 145 | 137.25 to 157.75 | 130 to 172 | 8% | 15% | 10% |
| place_a2b_right | 146.5 | 142.25 to 150.75 | 138 to 168 | 4% | 75% | 70% |
| place_mouse_pad | 147.5 | 142 to 154.75 | 136 to 174 | 6% | 15% | 30% |
| place_a2b_left | 149 | 144 to 152.75 | 138 to 165 | 4% | 80% | 80% |
| move_can_pot | 150.5 | 148.25 to 155 | 141 to 171 | 4% | 50% | 30% |
| rotate_qrcode | 153 | 151 to 156 | 148 to 178 | 4% | 30% | 20% |
| move_stapler_pad | 155.5 | 146.25 to 162.75 | 138 to 179 | 6% | 10% | 10% |
| place_container_plate | 157 | 154 to 159 | 152 to 180 | 4% | 85% | 100% |
| place_bread_skillet | 165 | 162 to 168 | 153 to 185 | 4% | 45% | 60% |
| scan_object | 168.5 | 166.25 to 173 | 152 to 188 | 4% | 20% | 30% |
| place_empty_cup | 172 | 168.25 to 174.75 | 164 to 189 | 3% | 75% | 80% |
| dump_bin_bigbin | 175.5 | 169.25 to 337.5 | 157 to 349 | 34% | 60% | 85% |
| place_shoe | 176.5 | 172 to 182 | 161 to 203 | 5% | 40% | 40% |
| open_laptop | 221.5 | 182.25 to 226.75 | 146 to 354 | 20% | 81% | 50% |
| place_dual_shoes | 222 | 218 to 225.75 | 213 to 245 | 4% | 5% | 10% |
| handover_mic | 222 | 221 to 223 | 216 to 225 | 1% | 15% | 40% |
| place_bread_basket | 229 | 225 to 289.75 | 157 to 329 | 22% | 43% | 60% |
| place_burger_fries | 241 | 240 to 242 | 238 to 243 | 1% | 90% | 80% |
| place_object_basket | 245 | 243 to 249 | 227 to 292 | 4% | 43% | 10% |
| shake_bottle | 246.5 | 244 to 254 | 237 to 262 | 3% | 100% | 100% |
| place_can_basket | 249 | 247 to 255.75 | 239 to 272 | 3% | 65% | 40% |
| put_object_cabinet | 269 | 265.25 to 271.75 | 260 to 284 | 2% | 5% | 10% |
| shake_bottle_horizontally | 275.5 | 273 to 283.75 | 264 to 291 | 2% | 90% | 100% |
| handover_block | 279 | 276.25 to 286 | 272 to 300 | 2% | 0% | 10% |
| place_cans_plasticbox | 287 | 286 to 290 | 283 to 292 | 1% | 40% | 40% |
| stack_bowls_two | 311 | 305.5 to 316.75 | 288 to 348 | 4% | 90% | 60% |
| stack_blocks_two | 313 | 306.5 to 320.75 | 290 to 330 | 3% | 20% | 30% |
| hanging_mug | 335 | 332.25 to 340 | 325 to 357 | 2% | 0% | 0% |
| open_microwave | 431 | 427.25 to 462.75 | 413 to 894 | 26% | 0% | 10% |
| blocks_ranking_rgb | 462 | 453.25 to 467 | 440 to 484 | 2% | 10% | 20% |
| blocks_ranking_size | 463.5 | 454.25 to 471 | 439 to 482 | 2% | 10% | 10% |
| stack_bowls_three | 470.5 | 458.25 to 484.75 | 438 to 505 | 4% | 19% | 30% |
| stack_blocks_three | 472 | 462.75 to 480.75 | 453 to 493 | 2% | 0% | 0% |
| put_bottles_dustbin | 655.5 | 530.25 to 663.75 | 404 to 824 | 12% | 0% | 0% |
