# Benchmark results: GR00T N1.7 on RoboTwin 2.0

Success rate per task; episodes in brackets. Clean = scenes like the training demos;
randomized = cluttered tables, random backgrounds/lighting/table height, unseen
instruction wording. 20 episodes per task ≈ ±20% per task; the 50-task mean pools
1,000 episodes (≈ ±3%).

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
