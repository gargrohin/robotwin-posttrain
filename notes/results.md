# Benchmark results: GR00T N1.7 on RoboTwin 2.0

Success rate per task; episodes in brackets. Clean = scenes like the training demos;
randomized = cluttered tables, random backgrounds/lighting/table height, unseen
instruction wording. 20 episodes per task ≈ ±20% per task; the 50-task mean pools
1,000 episodes (≈ ±3%).

## Model mix200_150k (one model, all 50 tasks × (50 clean + 200 randomized demos), 150k steps), checkpoint-150000

**Mean: clean 46.0%, randomized 45.2%** (50 / 50 tasks). The old model (50task_60k, below)
scored clean 42.6%, randomized 9.7%. Run card: `notes/runs/mix200_150k.md`.

- **Same scenes as the 50task_60k eval.** Every job replays that eval's seeds
  (`gr00t/paired_queue.py`).
- **Episodes:**
  - randomized: the same ~20 per task;
  - clean: 20 on the 10 tasks `pick_dual_bottles`, `place_object_stand`,
    `move_pillbottle_pad`, `place_empty_cup`, `place_phone_stand`, `move_can_pot`,
    `dump_bin_bigbin`, `handover_mic`, `grab_roller`, `click_bell`, and the first 10 of
    the old seeds on the other 40 tasks. With 10 episodes, a per-task rate can move by
    about ±30% from luck alone.
- **One exception:** 7 of `put_bottles_dustbin`'s 21 randomized episodes ran on fresh
  seeds, after a GPU memory error on the first try.
- Sorted by the new model's randomized rate.

| task | clean: old | clean: new | randomized: old | randomized: new |
|---|---|---|---|---|
| click_bell | 95% (19/20) | 95% (19/20) | 40% (8/20) | 100% (20/20) |
| shake_bottle | 100% (20/20) | 100% (10/10) | 55% (11/20) | 100% (20/20) |
| click_alarmclock | 100% (20/20) | 100% (10/10) | 24% (5/21) | 95% (20/21) |
| adjust_bottle | 95% (19/20) | 100% (10/10) | 5% (1/21) | 90% (19/21) |
| grab_roller | 90% (18/20) | 90% (18/20) | 38% (8/21) | 90% (19/21) |
| stack_bowls_two | 86% (18/21) | 60% (6/10) | 10% (2/21) | 90% (19/21) |
| place_burger_fries | 85% (17/20) | 80% (8/10) | 10% (2/20) | 90% (18/20) |
| shake_bottle_horizontally | 100% (20/20) | 100% (10/10) | 50% (10/20) | 90% (18/20) |
| place_container_plate | 100% (20/20) | 100% (10/10) | 30% (6/20) | 85% (17/20) |
| open_laptop | 50% (10/20) | 50% (5/10) | 14% (3/21) | 81% (17/21) |
| pick_dual_bottles | 75% (15/20) | 60% (12/20) | 15% (3/20) | 80% (16/20) |
| place_a2b_left | 50% (10/20) | 80% (8/10) | 10% (2/20) | 80% (16/20) |
| place_a2b_right | 45% (9/20) | 70% (7/10) | 0% (0/20) | 75% (15/20) |
| place_empty_cup | 75% (15/20) | 80% (16/20) | 20% (4/20) | 75% (15/20) |
| press_stapler | 70% (14/20) | 90% (9/10) | 14% (3/21) | 67% (14/21) |
| place_can_basket | 10% (2/20) | 40% (4/10) | 0% (0/20) | 65% (13/20) |
| dump_bin_bigbin | 75% (15/20) | 85% (17/20) | 10% (2/20) | 60% (12/20) |
| beat_block_hammer | 45% (9/20) | 40% (4/10) | 20% (4/20) | 55% (11/20) |
| move_pillbottle_pad | 45% (9/20) | 60% (12/20) | 5% (1/20) | 55% (11/20) |
| move_playingcard_away | 80% (16/20) | 60% (6/10) | 5% (1/20) | 55% (11/20) |
| place_object_stand | 50% (10/20) | 75% (15/20) | 5% (1/20) | 55% (11/20) |
| move_can_pot | 30% (6/20) | 30% (6/20) | 5% (1/20) | 50% (10/20) |
| place_phone_stand | 40% (8/20) | 35% (7/20) | 10% (2/20) | 50% (10/20) |
| pick_diverse_bottles | 30% (6/20) | 50% (5/10) | 29% (6/21) | 48% (10/21) |
| place_bread_skillet | 35% (7/20) | 60% (6/10) | 5% (1/20) | 45% (9/20) |
| place_bread_basket | 45% (9/20) | 60% (6/10) | 0% (0/21) | 43% (9/21) |
| place_object_basket | 40% (8/20) | 10% (1/10) | 0% (0/21) | 43% (9/21) |
| place_cans_plasticbox | 32% (7/22) | 40% (4/10) | 5% (1/20) | 40% (8/20) |
| place_shoe | 20% (4/20) | 40% (4/10) | 10% (2/20) | 40% (8/20) |
| turn_switch | 45% (9/20) | 40% (4/10) | 5% (1/20) | 35% (7/20) |
| stamp_seal | 10% (2/20) | 30% (3/10) | 0% (0/21) | 33% (7/21) |
| rotate_qrcode | 20% (4/20) | 20% (2/10) | 0% (0/20) | 30% (6/20) |
| place_object_scale | 25% (5/20) | 10% (1/10) | 5% (1/20) | 20% (4/20) |
| scan_object | 50% (10/20) | 30% (3/10) | 0% (0/20) | 20% (4/20) |
| stack_blocks_two | 20% (4/20) | 30% (3/10) | 0% (0/20) | 20% (4/20) |
| stack_bowls_three | 0% (0/20) | 30% (3/10) | 0% (0/21) | 19% (4/21) |
| handover_mic | 40% (8/20) | 40% (8/20) | 5% (1/20) | 15% (3/20) |
| place_fan | 15% (3/20) | 10% (1/10) | 0% (0/20) | 15% (3/20) |
| place_mouse_pad | 10% (2/20) | 30% (3/10) | 5% (1/20) | 15% (3/20) |
| blocks_ranking_rgb | 15% (3/20) | 20% (2/10) | 0% (0/21) | 10% (2/21) |
| blocks_ranking_size | 5% (1/20) | 10% (1/10) | 0% (0/21) | 10% (2/21) |
| move_stapler_pad | 20% (4/20) | 10% (1/10) | 0% (0/21) | 10% (2/21) |
| lift_pot | 20% (4/20) | 10% (1/10) | 0% (0/20) | 5% (1/20) |
| put_object_cabinet | 23% (6/26) | 10% (1/10) | 5% (1/20) | 5% (1/20) |
| place_dual_shoes | 5% (1/20) | 10% (1/10) | 0% (0/22) | 5% (1/22) |
| handover_block | 0% (0/20) | 10% (1/10) | 0% (0/20) | 0% (0/20) |
| hanging_mug | 10% (2/20) | 0% (0/10) | 0% (0/20) | 0% (0/20) |
| open_microwave | 5% (1/20) | 10% (1/10) | 15% (3/20) | 0% (0/20) |
| put_bottles_dustbin | 0% (0/24) | 0% (0/10) | 0% (0/21) | 0% (0/21) |
| stack_blocks_three | 0% (0/20) | 0% (0/10) | 0% (0/20) | 0% (0/20) |

### The 10 tasks above, at 5 checkpoints of this run (randomized, same scenes)

| model | successes / episodes | rate | learning rate at that checkpoint |
|---|---|---|---|
| 50task_60k, checkpoint-60000 | 31/201 | 15% | 7.6e-14 (end of its run) |
| mix200_150k, checkpoint-30000 | 23/201 | 11% | 9.4e-05 |
| mix200_150k, checkpoint-60000 | 33/201 | 16% | 7.0e-05 |
| mix200_150k, checkpoint-90000 | 75/201 | 37% | 3.8e-05 |
| mix200_150k, checkpoint-120000 | 124/201 | 62% | 1.1e-05 |
| mix200_150k, checkpoint-150000 | 127/201 | 63% | 1.2e-14 |

Clean, same 10 tasks: 50task_60k 123/200 (62%), mix200_150k at 60k 45/200 (22%), at 150k
130/200 (65%). Learning rates are from each checkpoint's `trainer_state.json`.

## Model 50task_60k (one model, all 50 tasks × 50 clean demos, 60k steps), checkpoint-60000

**Mean: clean 42.6%, randomized 9.7%** (50 / 50 tasks)

| task | clean | randomized |
|---|---|---|
| click_alarmclock | 100% (20/20) | 24% (5/21) |
| place_container_plate | 100% (20/20) | 30% (6/20) |
| shake_bottle | 100% (20/20) | 55% (11/20) |
| shake_bottle_horizontally | 100% (20/20) | 50% (10/20) |
| adjust_bottle | 95% (19/20) | 5% (1/21) |
| click_bell | 95% (19/20) | 40% (8/20) |
| grab_roller | 90% (18/20) | 38% (8/21) |
| stack_bowls_two | 86% (18/21) | 10% (2/21) |
| place_burger_fries | 85% (17/20) | 10% (2/20) |
| move_playingcard_away | 80% (16/20) | 5% (1/20) |
| dump_bin_bigbin | 75% (15/20) | 10% (2/20) |
| pick_dual_bottles | 75% (15/20) | 15% (3/20) |
| place_empty_cup | 75% (15/20) | 20% (4/20) |
| press_stapler | 70% (14/20) | 14% (3/21) |
| open_laptop | 50% (10/20) | 14% (3/21) |
| place_a2b_left | 50% (10/20) | 10% (2/20) |
| place_object_stand | 50% (10/20) | 5% (1/20) |
| scan_object | 50% (10/20) | 0% (0/20) |
| beat_block_hammer | 45% (9/20) | 20% (4/20) |
| move_pillbottle_pad | 45% (9/20) | 5% (1/20) |
| place_a2b_right | 45% (9/20) | 0% (0/20) |
| place_bread_basket | 45% (9/20) | 0% (0/21) |
| turn_switch | 45% (9/20) | 5% (1/20) |
| handover_mic | 40% (8/20) | 5% (1/20) |
| place_object_basket | 40% (8/20) | 0% (0/21) |
| place_phone_stand | 40% (8/20) | 10% (2/20) |
| place_bread_skillet | 35% (7/20) | 5% (1/20) |
| place_cans_plasticbox | 32% (7/22) | 5% (1/20) |
| move_can_pot | 30% (6/20) | 5% (1/20) |
| pick_diverse_bottles | 30% (6/20) | 29% (6/21) |
| place_object_scale | 25% (5/20) | 5% (1/20) |
| put_object_cabinet | 23% (6/26) | 5% (1/20) |
| lift_pot | 20% (4/20) | 0% (0/20) |
| move_stapler_pad | 20% (4/20) | 0% (0/21) |
| place_shoe | 20% (4/20) | 10% (2/20) |
| rotate_qrcode | 20% (4/20) | 0% (0/20) |
| stack_blocks_two | 20% (4/20) | 0% (0/20) |
| blocks_ranking_rgb | 15% (3/20) | 0% (0/21) |
| place_fan | 15% (3/20) | 0% (0/20) |
| hanging_mug | 10% (2/20) | 0% (0/20) |
| place_can_basket | 10% (2/20) | 0% (0/20) |
| place_mouse_pad | 10% (2/20) | 5% (1/20) |
| stamp_seal | 10% (2/20) | 0% (0/21) |
| blocks_ranking_size | 5% (1/20) | 0% (0/21) |
| open_microwave | 5% (1/20) | 15% (3/20) |
| place_dual_shoes | 5% (1/20) | 0% (0/22) |
| handover_block | 0% (0/20) | 0% (0/20) |
| put_bottles_dustbin | 0% (0/24) | 0% (0/21) |
| stack_blocks_three | 0% (0/20) | 0% (0/20) |
| stack_bowls_three | 0% (0/20) | 0% (0/21) |

## Model 5task_30k (one model, 5 tasks × 50 clean demos, 30k steps), checkpoint-30000

**Mean: clean 28.3%, randomized 4.7%**

| task | clean | randomized | 50-task model, clean | 50-task model, randomized |
|---|---|---|---|---|
| beat_block_hammer | 47% (47/100) | 13% (4/30) | 45% (9/20) | 20% (4/20) |
| handover_block | 3% (3/100) | 0% (0/100) | 0% (0/20) | 0% (0/20) |
| lift_pot | 18% (18/100) | 0% (0/21) | 20% (4/20) | 0% (0/20) |
| open_laptop | 67% (68/102) | 10% (2/20) | 50% (10/20) | 14% (3/21) |
| stack_blocks_two | 7% (7/100) | 0% (0/21) | 20% (4/20) | 0% (0/20) |
