# Benchmark results: GR00T N1.7 on RoboTwin 2.0

Success rate per task; episodes in brackets. Clean = scenes like the training demos;
randomized = cluttered tables, random backgrounds/lighting/table height, unseen
instruction wording. 20 episodes per task ≈ ±20% per task; the 50-task mean pools
1,000 episodes (≈ ±3%).

## FLUX 3 Action (flux3_mix200_b512), step 3,000, vs Cosmos 5,000 and GR00T mix200_150k (same seeds)

BFL's `flux-3-action-base` (7B), fine-tuned with BFL's flux-action trainer on mix200 (same data as
GR00T mix200_150k and Cosmos): 3,000 updates at batch 512, the official DROID recipe with its schedule
scaled to 1/10. Evaluated 2026-11-04 on cluster GPUs, one stream per GPU, exact GR00T seed lists. EMA
weights (0.10) exported in bf16; sampling uses BFL's reference DROID inference settings, 32 actions
per call.

| | Flux 3,000 | Cosmos 5,000 | GR00T mix200_150k |
|---|---|---|---|
| randomized, GR00T's first 10 seeds per task | 344/498 = **69.1%** | 410/498 = 82.3% | 234/498 = 47.0% |
| randomized, all GR00T seeds | 720/1014 = **71.0%** | 830/1014 = 81.9% | 459/1014 = 45.3% |
| clean, GR00T's first 10 seeds per task | 355/499 = **71.1%** | 421/499 = 84.4% | 227/499 = 45.5% |

(Seeds all three models ran. Cosmos decay 5,500, on the same seeds: randomized 85.0%, clean 88.6%.)

- Seed by seed (sign test on seeds where exactly one model succeeds):
  - vs GR00T: randomized 335 Flux-only vs 73 GR00T-only (p ≈ 3e-41); clean 163 vs 35 (p ≈ 6e-21).
  - vs Cosmos 5,000: randomized 79 Flux-only vs 192 Cosmos-only (p ≈ 5e-12); clean 37 vs 103 (p ≈ 2e-8).
  - vs Cosmos decay 5,500: randomized 64 vs 206 (p ≈ 1e-18); clean 24 vs 111 (p ≈ 1e-14).
- Learning curve, 10 tasks × GR00T's first 10 randomized seeds (`gr00t/paired_compare.py`, 99 paired):
  step 1,500 73/99, step 3,000 72/99. The second half of training added nothing measurable there.
- Held-out open-loop arm error (`flux3/heldout_check.py`, 1,200 moments; train_randomized / train_clean
  / unseen_randomized, rad): step 1,500 0.037 / 0.027 / 0.030; step 3,000 0.029 / 0.021 / 0.023.
  GR00T 150k 0.018 / 0.016 / 0.016; hold still ~0.064.
- Lost seeds: RoboTwin's expert check raised on the cluster for open_microwave 300004,
  put_bottles_dustbin 300001 (randomized) and place_object_basket 100006 (clean), so those are unpaired.
- Largest gaps to Cosmos 5,000 (randomized, all seeds): place_cans_plasticbox 9 vs 20,
  move_stapler_pad 6 vs 15, move_can_pot 7 vs 15, place_mouse_pad 5 vs 12, place_object_basket
  11 vs 18. Flux is ahead on stack_bowls_three (18 vs 12) and place_object_stand (18 vs 16).
- Eval cost: ~77 GPU-hours (516k sim steps, 0.48 s per step including rendering).

| task | rand 10 Flux | rand 10 Cosmos | rand 10 GR00T | rand all Flux | rand all Cosmos | rand all GR00T | clean 10 Flux | clean 10 Cosmos | clean 10 GR00T |
|---|---|---|---|---|---|---|---|---|---|
| adjust_bottle | 10/10 | 10/10 | 10/10 | 21/21 | 21/21 | 19/21 | 10/10 | 10/10 | 10/10 |
| beat_block_hammer | 10/10 | 10/10 | 9/10 | 20/20 | 19/20 | 11/20 | 10/10 | 10/10 | 4/10 |
| blocks_ranking_rgb | 9/10 | 9/10 | 1/10 | 20/21 | 19/21 | 2/21 | 9/10 | 9/10 | 2/10 |
| blocks_ranking_size | 6/10 | 8/10 | 0/10 | 14/21 | 17/21 | 2/21 | 7/10 | 7/10 | 1/10 |
| click_alarmclock | 10/10 | 9/10 | 10/10 | 21/21 | 20/21 | 20/21 | 10/10 | 10/10 | 10/10 |
| click_bell | 10/10 | 10/10 | 10/10 | 20/20 | 20/20 | 20/20 | 10/10 | 10/10 | 10/10 |
| dump_bin_bigbin | 7/10 | 9/10 | 7/10 | 14/20 | 18/20 | 12/20 | 8/10 | 10/10 | 8/10 |
| grab_roller | 10/10 | 10/10 | 9/10 | 21/21 | 21/21 | 19/21 | 10/10 | 10/10 | 8/10 |
| handover_block | 4/10 | 3/10 | 0/10 | 5/20 | 4/20 | 0/20 | 3/10 | 6/10 | 1/10 |
| handover_mic | 9/10 | 8/10 | 2/10 | 18/20 | 18/20 | 3/20 | 9/10 | 10/10 | 2/10 |
| hanging_mug | 0/10 | 0/10 | 0/10 | 2/20 | 1/20 | 0/20 | 1/10 | 3/10 | 0/10 |
| lift_pot | 8/10 | 7/10 | 1/10 | 14/20 | 15/20 | 1/20 | 7/10 | 10/10 | 1/10 |
| move_can_pot | 3/10 | 7/10 | 6/10 | 7/20 | 15/20 | 10/20 | 5/10 | 8/10 | 3/10 |
| move_pillbottle_pad | 8/10 | 9/10 | 7/10 | 16/20 | 16/20 | 11/20 | 7/10 | 7/10 | 6/10 |
| move_playingcard_away | 9/10 | 10/10 | 5/10 | 16/20 | 20/20 | 11/20 | 7/10 | 10/10 | 6/10 |
| move_stapler_pad | 2/10 | 8/10 | 1/10 | 6/21 | 15/21 | 2/21 | 2/10 | 7/10 | 1/10 |
| open_laptop | 9/10 | 10/10 | 8/10 | 19/21 | 20/21 | 17/21 | 9/10 | 7/10 | 5/10 |
| open_microwave | 7/10 | 10/10 | 0/10 | 14/19 | 14/19 | 0/19 | 10/10 | 7/10 | 1/10 |
| pick_diverse_bottles | 2/10 | 7/10 | 6/10 | 9/21 | 11/21 | 10/21 | 2/10 | 6/10 | 5/10 |
| pick_dual_bottles | 2/10 | 6/10 | 7/10 | 11/20 | 14/20 | 16/20 | 7/10 | 5/10 | 6/10 |
| place_a2b_left | 9/10 | 10/10 | 8/10 | 18/20 | 20/20 | 16/20 | 8/10 | 9/10 | 8/10 |
| place_a2b_right | 10/10 | 8/10 | 8/10 | 19/20 | 18/20 | 15/20 | 8/10 | 9/10 | 7/10 |
| place_bread_basket | 6/10 | 9/10 | 2/10 | 15/21 | 20/21 | 9/21 | 7/10 | 10/10 | 6/10 |
| place_bread_skillet | 9/10 | 8/10 | 5/10 | 16/20 | 17/20 | 9/20 | 6/10 | 9/10 | 6/10 |
| place_burger_fries | 8/10 | 10/10 | 8/10 | 17/20 | 20/20 | 18/20 | 9/10 | 10/10 | 8/10 |
| place_can_basket | 3/10 | 8/10 | 6/10 | 7/20 | 12/20 | 13/20 | 4/10 | 7/10 | 4/10 |
| place_cans_plasticbox | 5/10 | 10/10 | 3/10 | 9/20 | 20/20 | 8/20 | 4/10 | 10/10 | 4/10 |
| place_container_plate | 9/10 | 10/10 | 9/10 | 17/20 | 20/20 | 17/20 | 9/10 | 10/10 | 10/10 |
| place_dual_shoes | 5/10 | 7/10 | 0/10 | 15/22 | 17/22 | 1/22 | 7/10 | 7/10 | 1/10 |
| place_empty_cup | 9/10 | 10/10 | 7/10 | 18/20 | 20/20 | 15/20 | 10/10 | 10/10 | 9/10 |
| place_fan | 4/10 | 7/10 | 2/10 | 10/20 | 14/20 | 3/20 | 8/10 | 10/10 | 1/10 |
| place_mouse_pad | 2/10 | 5/10 | 2/10 | 5/20 | 12/20 | 3/20 | 3/10 | 7/10 | 3/10 |
| place_object_basket | 4/10 | 7/10 | 4/10 | 11/21 | 18/21 | 9/21 | 6/9 | 8/9 | 1/9 |
| place_object_scale | 6/10 | 7/10 | 3/10 | 10/20 | 14/20 | 4/20 | 5/10 | 9/10 | 1/10 |
| place_object_stand | 9/10 | 7/10 | 5/10 | 18/20 | 16/20 | 11/20 | 10/10 | 8/10 | 8/10 |
| place_phone_stand | 6/10 | 8/10 | 7/10 | 13/20 | 16/20 | 10/20 | 8/10 | 9/10 | 2/10 |
| place_shoe | 9/10 | 10/10 | 4/10 | 16/20 | 19/20 | 8/20 | 9/10 | 9/10 | 4/10 |
| press_stapler | 10/10 | 10/10 | 6/10 | 20/21 | 21/21 | 14/21 | 9/10 | 10/10 | 9/10 |
| put_bottles_dustbin | 2/8 | 5/8 | 0/8 | 6/19 | 11/19 | 0/19 | 3/10 | 9/10 | 0/10 |
| put_object_cabinet | 5/10 | 7/10 | 1/10 | 11/20 | 13/20 | 1/20 | 7/10 | 6/10 | 1/10 |
| rotate_qrcode | 6/10 | 8/10 | 4/10 | 15/20 | 17/20 | 6/20 | 6/10 | 8/10 | 2/10 |
| scan_object | 8/10 | 10/10 | 3/10 | 13/20 | 17/20 | 4/20 | 8/10 | 7/10 | 3/10 |
| shake_bottle | 10/10 | 10/10 | 10/10 | 20/20 | 20/20 | 20/20 | 10/10 | 10/10 | 10/10 |
| shake_bottle_horizontally | 10/10 | 10/10 | 9/10 | 20/20 | 20/20 | 18/20 | 10/10 | 9/10 | 10/10 |
| stack_blocks_three | 5/10 | 8/10 | 0/10 | 11/20 | 17/20 | 0/20 | 7/10 | 9/10 | 0/10 |
| stack_blocks_two | 9/10 | 10/10 | 1/10 | 18/20 | 20/20 | 4/20 | 9/10 | 10/10 | 3/10 |
| stack_bowls_three | 8/10 | 8/10 | 1/10 | 18/21 | 12/21 | 4/21 | 6/10 | 9/10 | 3/10 |
| stack_bowls_two | 9/10 | 10/10 | 10/10 | 19/21 | 21/21 | 19/21 | 10/10 | 8/10 | 6/10 |
| stamp_seal | 8/10 | 9/10 | 5/10 | 14/21 | 18/21 | 7/21 | 2/10 | 6/10 | 3/10 |
| turn_switch | 6/10 | 4/10 | 2/10 | 13/20 | 12/20 | 7/20 | 4/10 | 7/10 | 4/10 |
| **total** | **344/498 = 69.1%** | **410/498 = 82.3%** | **234/498 = 47.0%** | **720/1014 = 71.0%** | **830/1014 = 81.9%** | **459/1014 = 45.3%** | **355/499 = 71.1%** | **421/499 = 84.4%** | **227/499 = 45.5%** |

## Cosmos3-Nano learning-rate decay, iteration 6,000 (end of decay), vs Cosmos 5,000 and GR00T (same seeds)

| | decay 6,000 | decay 5,500 | Cosmos 5,000 | GR00T mix200_150k |
|---|---|---|---|---|
| randomized, first 10 per task | 413/498 = 82.9% | 422/498 = 84.7% | 410/498 = 82.3% | 234/498 = 47.0% |
| randomized, all GR00T seeds | 857/1014 = **84.5%** | 862/1014 = 85.0% | 830/1014 = 81.9% | 459/1014 = 45.3% |
| clean, first 10 per task | 440/499 = **88.2%** | 442/499 = 88.6% | 421/499 = 84.4% | 227/499 = 45.5% |

- 6,000 vs Cosmos 5,000 (sign test on seeds where they differ): rand first 10: seeds where new and C5000 differ: new-only successes 41, C5000-only 38; two-sided sign test p = 0.822; rand all: seeds where new and C5000 differ: new-only successes 94, C5000-only 67; two-sided sign test p = 0.040; clean first 10: seeds where new and C5000 differ: new-only successes 52, C5000-only 33; two-sided sign test p = 0.050
- 6,000 vs 5,500, seed by seed: randomized 858 vs 863 of 1015 (66 vs 71 seeds only one solves, p = 0.73);
  clean 440 vs 442 of 499 (27 vs 29, p = 0.89). The second half of the decay added nothing measurable.
- Same 3 seeds not run as at 5,500.

| task | rand first 10: new | C5000 | GR00T | rand all: new | C5000 | GR00T | clean first 10: new | C5000 | GR00T |
|---|---|---|---|---|---|---|---|---|---|
| adjust_bottle | 10/10 | 10/10 | 10/10 | 21/21 | 21/21 | 19/21 | 10/10 | 10/10 | 10/10 |
| beat_block_hammer | 10/10 | 10/10 | 9/10 | 20/20 | 19/20 | 11/20 | 10/10 | 10/10 | 4/10 |
| blocks_ranking_rgb | 8/10 | 9/10 | 1/10 | 18/21 | 19/21 | 2/21 | 10/10 | 9/10 | 2/10 |
| blocks_ranking_size | 8/10 | 8/10 | 0/10 | 19/21 | 17/21 | 2/21 | 7/10 | 7/10 | 1/10 |
| click_alarmclock | 10/10 | 9/10 | 10/10 | 21/21 | 20/21 | 20/21 | 10/10 | 10/10 | 10/10 |
| click_bell | 10/10 | 10/10 | 10/10 | 20/20 | 20/20 | 20/20 | 10/10 | 10/10 | 10/10 |
| dump_bin_bigbin | 10/10 | 9/10 | 7/10 | 20/20 | 18/20 | 12/20 | 10/10 | 10/10 | 8/10 |
| grab_roller | 10/10 | 10/10 | 9/10 | 21/21 | 21/21 | 19/21 | 10/10 | 10/10 | 8/10 |
| handover_block | 4/10 | 3/10 | 0/10 | 9/20 | 4/20 | 0/20 | 3/10 | 6/10 | 1/10 |
| handover_mic | 9/10 | 8/10 | 2/10 | 19/20 | 18/20 | 3/20 | 10/10 | 10/10 | 2/10 |
| hanging_mug | 2/10 | 0/10 | 0/10 | 8/20 | 1/20 | 0/20 | 2/10 | 3/10 | 0/10 |
| lift_pot | 8/10 | 7/10 | 1/10 | 17/20 | 15/20 | 1/20 | 10/10 | 10/10 | 1/10 |
| move_can_pot | 7/10 | 7/10 | 6/10 | 14/20 | 15/20 | 10/20 | 10/10 | 8/10 | 3/10 |
| move_pillbottle_pad | 8/10 | 9/10 | 7/10 | 17/20 | 16/20 | 11/20 | 8/10 | 7/10 | 6/10 |
| move_playingcard_away | 9/10 | 10/10 | 5/10 | 19/20 | 20/20 | 11/20 | 9/10 | 10/10 | 6/10 |
| move_stapler_pad | 5/10 | 8/10 | 1/10 | 14/21 | 15/21 | 2/21 | 8/10 | 7/10 | 1/10 |
| open_laptop | 10/10 | 10/10 | 8/10 | 21/21 | 20/21 | 17/21 | 9/10 | 7/10 | 5/10 |
| open_microwave | 10/10 | 10/10 | 0/10 | 19/19 | 14/19 | 0/19 | 9/10 | 7/10 | 1/10 |
| pick_diverse_bottles | 7/10 | 7/10 | 6/10 | 14/21 | 11/21 | 10/21 | 6/10 | 6/10 | 5/10 |
| pick_dual_bottles | 7/10 | 6/10 | 7/10 | 14/20 | 14/20 | 16/20 | 8/10 | 5/10 | 6/10 |
| place_a2b_left | 9/10 | 10/10 | 8/10 | 18/20 | 20/20 | 16/20 | 9/10 | 9/10 | 8/10 |
| place_a2b_right | 8/10 | 8/10 | 8/10 | 18/20 | 18/20 | 15/20 | 10/10 | 9/10 | 7/10 |
| place_bread_basket | 9/10 | 9/10 | 2/10 | 19/21 | 20/21 | 9/21 | 10/10 | 10/10 | 6/10 |
| place_bread_skillet | 10/10 | 8/10 | 5/10 | 18/20 | 17/20 | 9/20 | 9/10 | 9/10 | 6/10 |
| place_burger_fries | 9/10 | 10/10 | 8/10 | 19/20 | 20/20 | 18/20 | 9/10 | 10/10 | 8/10 |
| place_can_basket | 4/10 | 8/10 | 6/10 | 8/20 | 12/20 | 13/20 | 7/10 | 7/10 | 4/10 |
| place_cans_plasticbox | 8/10 | 10/10 | 3/10 | 17/20 | 20/20 | 8/20 | 10/10 | 10/10 | 4/10 |
| place_container_plate | 10/10 | 10/10 | 9/10 | 20/20 | 20/20 | 17/20 | 10/10 | 10/10 | 10/10 |
| place_dual_shoes | 8/10 | 7/10 | 0/10 | 15/22 | 17/22 | 1/22 | 10/10 | 7/10 | 1/10 |
| place_empty_cup | 10/10 | 10/10 | 7/10 | 20/20 | 20/20 | 15/20 | 9/10 | 10/10 | 9/10 |
| place_fan | 6/10 | 7/10 | 2/10 | 15/20 | 14/20 | 3/20 | 8/10 | 10/10 | 1/10 |
| place_mouse_pad | 7/10 | 5/10 | 2/10 | 14/20 | 12/20 | 3/20 | 7/10 | 7/10 | 3/10 |
| place_object_basket | 7/10 | 7/10 | 4/10 | 16/21 | 18/21 | 9/21 | 8/9 | 8/9 | 1/9 |
| place_object_scale | 8/10 | 7/10 | 3/10 | 16/20 | 14/20 | 4/20 | 9/10 | 9/10 | 1/10 |
| place_object_stand | 9/10 | 7/10 | 5/10 | 19/20 | 16/20 | 11/20 | 10/10 | 8/10 | 8/10 |
| place_phone_stand | 8/10 | 8/10 | 7/10 | 14/20 | 16/20 | 10/20 | 8/10 | 9/10 | 2/10 |
| place_shoe | 9/10 | 10/10 | 4/10 | 19/20 | 19/20 | 8/20 | 9/10 | 9/10 | 4/10 |
| press_stapler | 10/10 | 10/10 | 6/10 | 21/21 | 21/21 | 14/21 | 10/10 | 10/10 | 9/10 |
| put_bottles_dustbin | 3/8 | 5/8 | 0/8 | 10/19 | 11/19 | 0/19 | 9/10 | 9/10 | 0/10 |
| put_object_cabinet | 9/10 | 7/10 | 1/10 | 14/20 | 13/20 | 1/20 | 8/10 | 6/10 | 1/10 |
| rotate_qrcode | 6/10 | 8/10 | 4/10 | 13/20 | 17/20 | 6/20 | 9/10 | 8/10 | 2/10 |
| scan_object | 10/10 | 10/10 | 3/10 | 17/20 | 17/20 | 4/20 | 8/10 | 7/10 | 3/10 |
| shake_bottle | 10/10 | 10/10 | 10/10 | 20/20 | 20/20 | 20/20 | 10/10 | 10/10 | 10/10 |
| shake_bottle_horizontally | 9/10 | 10/10 | 9/10 | 19/20 | 20/20 | 18/20 | 10/10 | 9/10 | 10/10 |
| stack_blocks_three | 9/10 | 8/10 | 0/10 | 19/20 | 17/20 | 0/20 | 10/10 | 9/10 | 0/10 |
| stack_blocks_two | 10/10 | 10/10 | 1/10 | 20/20 | 20/20 | 4/20 | 10/10 | 10/10 | 3/10 |
| stack_bowls_three | 9/10 | 8/10 | 1/10 | 18/21 | 12/21 | 4/21 | 9/10 | 9/10 | 3/10 |
| stack_bowls_two | 10/10 | 10/10 | 10/10 | 21/21 | 21/21 | 19/21 | 10/10 | 8/10 | 6/10 |
| stamp_seal | 10/10 | 9/10 | 5/10 | 20/21 | 18/21 | 7/21 | 8/10 | 6/10 | 3/10 |
| turn_switch | 7/10 | 4/10 | 2/10 | 15/20 | 12/20 | 7/20 | 8/10 | 7/10 | 4/10 |
| **total** | **413/498 = 82.9%** | **410/498 = 82.3%** | **234/498 = 47.0%** | **857/1014 = 84.5%** | **830/1014 = 81.9%** | **459/1014 = 45.3%** | **440/499 = 88.2%** | **421/499 = 84.4%** | **227/499 = 45.5%** |

## Cosmos3-Nano learning-rate decay, iteration 5,500, vs Cosmos 5,000 and GR00T mix200_150k (same seeds)

Continued from 5,000 with the learning rate going linearly 3.55e-5 → 0 over 1,000 iterations (checkpoint
5,500 = halfway). Evaluated 2026-11-04 on cluster GPUs with exact seed lists, EMA weights as before.

| | decay 5,500 | Cosmos 5,000 | GR00T mix200_150k |
|---|---|---|---|
| randomized, first 10 per task | 422/498 = **84.7%** | 410/498 = 82.3% | 234/498 = 47.0% |
| randomized, all GR00T seeds | 862/1014 = **85.0%** | 830/1014 = 81.9% | 459/1014 = 45.3% |
| clean, first 10 per task | 442/499 = **88.6%** | 421/499 = 84.4% | 227/499 = 45.5% |

Paired by seed against Cosmos 5,000 (two-sided sign test on seeds where they differ):
rand first 10: seeds where new and C5000 differ: new-only successes 47, C5000-only 35; two-sided sign test p = 0.224
rand all: seeds where new and C5000 differ: new-only successes 95, C5000-only 63; two-sided sign test p = 0.013
clean first 10: seeds where new and C5000 differ: new-only successes 45, C5000-only 24; two-sided sign test p = 0.015

Seeds not run (RoboTwin's expert check raises on the cluster): randomized open_microwave 300004 and
put_bottles_dustbin 300001; clean place_object_basket 100006.

| task | rand first 10: new | C5000 | GR00T | rand all: new | C5000 | GR00T | clean first 10: new | C5000 | GR00T |
|---|---|---|---|---|---|---|---|---|---|
| adjust_bottle | 10/10 | 10/10 | 10/10 | 21/21 | 21/21 | 19/21 | 10/10 | 10/10 | 10/10 |
| beat_block_hammer | 10/10 | 10/10 | 9/10 | 20/20 | 19/20 | 11/20 | 10/10 | 10/10 | 4/10 |
| blocks_ranking_rgb | 8/10 | 9/10 | 1/10 | 19/21 | 19/21 | 2/21 | 9/10 | 9/10 | 2/10 |
| blocks_ranking_size | 7/10 | 8/10 | 0/10 | 17/21 | 17/21 | 2/21 | 4/10 | 7/10 | 1/10 |
| click_alarmclock | 10/10 | 9/10 | 10/10 | 21/21 | 20/21 | 20/21 | 10/10 | 10/10 | 10/10 |
| click_bell | 10/10 | 10/10 | 10/10 | 20/20 | 20/20 | 20/20 | 10/10 | 10/10 | 10/10 |
| dump_bin_bigbin | 9/10 | 9/10 | 7/10 | 18/20 | 18/20 | 12/20 | 10/10 | 10/10 | 8/10 |
| grab_roller | 10/10 | 10/10 | 9/10 | 21/21 | 21/21 | 19/21 | 10/10 | 10/10 | 8/10 |
| handover_block | 5/10 | 3/10 | 0/10 | 9/20 | 4/20 | 0/20 | 6/10 | 6/10 | 1/10 |
| handover_mic | 9/10 | 8/10 | 2/10 | 19/20 | 18/20 | 3/20 | 8/10 | 10/10 | 2/10 |
| hanging_mug | 3/10 | 0/10 | 0/10 | 6/20 | 1/20 | 0/20 | 1/10 | 3/10 | 0/10 |
| lift_pot | 8/10 | 7/10 | 1/10 | 16/20 | 15/20 | 1/20 | 10/10 | 10/10 | 1/10 |
| move_can_pot | 8/10 | 7/10 | 6/10 | 15/20 | 15/20 | 10/20 | 10/10 | 8/10 | 3/10 |
| move_pillbottle_pad | 9/10 | 9/10 | 7/10 | 18/20 | 16/20 | 11/20 | 8/10 | 7/10 | 6/10 |
| move_playingcard_away | 9/10 | 10/10 | 5/10 | 19/20 | 20/20 | 11/20 | 10/10 | 10/10 | 6/10 |
| move_stapler_pad | 8/10 | 8/10 | 1/10 | 16/21 | 15/21 | 2/21 | 8/10 | 7/10 | 1/10 |
| open_laptop | 10/10 | 10/10 | 8/10 | 21/21 | 20/21 | 17/21 | 9/10 | 7/10 | 5/10 |
| open_microwave | 10/10 | 10/10 | 0/10 | 18/19 | 14/19 | 0/19 | 8/10 | 7/10 | 1/10 |
| pick_diverse_bottles | 7/10 | 7/10 | 6/10 | 12/21 | 11/21 | 10/21 | 7/10 | 6/10 | 5/10 |
| pick_dual_bottles | 9/10 | 6/10 | 7/10 | 19/20 | 14/20 | 16/20 | 9/10 | 5/10 | 6/10 |
| place_a2b_left | 9/10 | 10/10 | 8/10 | 19/20 | 20/20 | 16/20 | 10/10 | 9/10 | 8/10 |
| place_a2b_right | 9/10 | 8/10 | 8/10 | 19/20 | 18/20 | 15/20 | 10/10 | 9/10 | 7/10 |
| place_bread_basket | 8/10 | 9/10 | 2/10 | 19/21 | 20/21 | 9/21 | 10/10 | 10/10 | 6/10 |
| place_bread_skillet | 10/10 | 8/10 | 5/10 | 16/20 | 17/20 | 9/20 | 9/10 | 9/10 | 6/10 |
| place_burger_fries | 10/10 | 10/10 | 8/10 | 20/20 | 20/20 | 18/20 | 10/10 | 10/10 | 8/10 |
| place_can_basket | 9/10 | 8/10 | 6/10 | 17/20 | 12/20 | 13/20 | 7/10 | 7/10 | 4/10 |
| place_cans_plasticbox | 8/10 | 10/10 | 3/10 | 18/20 | 20/20 | 8/20 | 10/10 | 10/10 | 4/10 |
| place_container_plate | 10/10 | 10/10 | 9/10 | 20/20 | 20/20 | 17/20 | 10/10 | 10/10 | 10/10 |
| place_dual_shoes | 7/10 | 7/10 | 0/10 | 15/22 | 17/22 | 1/22 | 6/10 | 7/10 | 1/10 |
| place_empty_cup | 10/10 | 10/10 | 7/10 | 20/20 | 20/20 | 15/20 | 10/10 | 10/10 | 9/10 |
| place_fan | 7/10 | 7/10 | 2/10 | 14/20 | 14/20 | 3/20 | 9/10 | 10/10 | 1/10 |
| place_mouse_pad | 6/10 | 5/10 | 2/10 | 12/20 | 12/20 | 3/20 | 8/10 | 7/10 | 3/10 |
| place_object_basket | 10/10 | 7/10 | 4/10 | 19/21 | 18/21 | 9/21 | 9/9 | 8/9 | 1/9 |
| place_object_scale | 9/10 | 7/10 | 3/10 | 19/20 | 14/20 | 4/20 | 9/10 | 9/10 | 1/10 |
| place_object_stand | 8/10 | 7/10 | 5/10 | 17/20 | 16/20 | 11/20 | 10/10 | 8/10 | 8/10 |
| place_phone_stand | 8/10 | 8/10 | 7/10 | 16/20 | 16/20 | 10/20 | 9/10 | 9/10 | 2/10 |
| place_shoe | 10/10 | 10/10 | 4/10 | 19/20 | 19/20 | 8/20 | 9/10 | 9/10 | 4/10 |
| press_stapler | 9/10 | 10/10 | 6/10 | 20/21 | 21/21 | 14/21 | 10/10 | 10/10 | 9/10 |
| put_bottles_dustbin | 4/8 | 5/8 | 0/8 | 10/19 | 11/19 | 0/19 | 9/10 | 9/10 | 0/10 |
| put_object_cabinet | 8/10 | 7/10 | 1/10 | 15/20 | 13/20 | 1/20 | 8/10 | 6/10 | 1/10 |
| rotate_qrcode | 7/10 | 8/10 | 4/10 | 13/20 | 17/20 | 6/20 | 8/10 | 8/10 | 2/10 |
| scan_object | 7/10 | 10/10 | 3/10 | 13/20 | 17/20 | 4/20 | 9/10 | 7/10 | 3/10 |
| shake_bottle | 10/10 | 10/10 | 10/10 | 20/20 | 20/20 | 20/20 | 10/10 | 10/10 | 10/10 |
| shake_bottle_horizontally | 10/10 | 10/10 | 9/10 | 20/20 | 20/20 | 18/20 | 10/10 | 9/10 | 10/10 |
| stack_blocks_three | 7/10 | 8/10 | 0/10 | 17/20 | 17/20 | 0/20 | 10/10 | 9/10 | 0/10 |
| stack_blocks_two | 10/10 | 10/10 | 1/10 | 20/20 | 20/20 | 4/20 | 10/10 | 10/10 | 3/10 |
| stack_bowls_three | 7/10 | 8/10 | 1/10 | 16/21 | 12/21 | 4/21 | 9/10 | 9/10 | 3/10 |
| stack_bowls_two | 10/10 | 10/10 | 10/10 | 21/21 | 21/21 | 19/21 | 10/10 | 8/10 | 6/10 |
| stamp_seal | 10/10 | 9/10 | 5/10 | 20/21 | 18/21 | 7/21 | 8/10 | 6/10 | 3/10 |
| turn_switch | 6/10 | 4/10 | 2/10 | 13/20 | 12/20 | 7/20 | 9/10 | 7/10 | 4/10 |
| **total** | **422/498 = 84.7%** | **410/498 = 82.3%** | **234/498 = 47.0%** | **862/1014 = 85.0%** | **830/1014 = 81.9%** | **459/1014 = 45.3%** | **442/499 = 88.6%** | **421/499 = 84.4%** | **227/499 = 45.5%** |

## Cosmos3-Nano, 16-GPU run, iteration 5,000, vs GR00T mix200_150k checkpoint-150000 (same seeds)

Evaluated 2026-11-03 on cluster GPUs, one eval stream per GPU, seed by seed against GR00T's logs.

| | Cosmos 5,000 | GR00T mix200_150k |
|---|---|---|
| randomized, GR00T's first 10 seeds per task | 410/499 = **82.2%** | 234/499 = 46.9% |
| randomized, all of GR00T's seeds (20–22 per task) | 830/1015 = **81.8%** | 459/1015 = 45.2% |
| clean, GR00T's first 10 seeds per task | 421/500 = **84.2%** | 227/500 = 45.4% |

- 1015 of GR00T's 1017 randomized seeds were run. RoboTwin's scripted expert check crashes on
  put_bottles_dustbin 100017 and open_microwave 300004 on the cluster, so those two are unpaired.
- 22 Cosmos episodes landed on seeds GR00T never ran and are excluded from the comparison. The
  expert check that decides whether a seed is usable isn't deterministic, so "start + count"
  queues can skip a different seed than GR00T's did. Missed GR00T seeds were rerun from exact
  seed lists.
- Run-to-run check: the same 10 tasks × 10 seeds, run on the devbox and on the cluster: 87/100 vs
  84/100, same outcome on 93 of 100 seeds.

| task | rand 10 Cosmos | rand 10 GR00T | rand all Cosmos | rand all GR00T | clean 10 Cosmos | clean 10 GR00T |
|---|---|---|---|---|---|---|
| adjust_bottle | 10/10 | 10/10 | 21/21 | 19/21 | 10/10 | 10/10 |
| beat_block_hammer | 10/10 | 9/10 | 19/20 | 11/20 | 10/10 | 4/10 |
| blocks_ranking_rgb | 9/10 | 1/10 | 19/21 | 2/21 | 9/10 | 2/10 |
| blocks_ranking_size | 8/10 | 0/10 | 17/21 | 2/21 | 7/10 | 1/10 |
| click_alarmclock | 9/10 | 10/10 | 20/21 | 20/21 | 10/10 | 10/10 |
| click_bell | 10/10 | 10/10 | 20/20 | 20/20 | 10/10 | 10/10 |
| dump_bin_bigbin | 9/10 | 7/10 | 18/20 | 12/20 | 10/10 | 8/10 |
| grab_roller | 10/10 | 9/10 | 21/21 | 19/21 | 10/10 | 8/10 |
| handover_block | 3/10 | 0/10 | 4/20 | 0/20 | 6/10 | 1/10 |
| handover_mic | 8/10 | 2/10 | 18/20 | 3/20 | 10/10 | 2/10 |
| hanging_mug | 0/10 | 0/10 | 1/20 | 0/20 | 3/10 | 0/10 |
| lift_pot | 7/10 | 1/10 | 15/20 | 1/20 | 10/10 | 1/10 |
| move_can_pot | 7/10 | 6/10 | 15/20 | 10/20 | 8/10 | 3/10 |
| move_pillbottle_pad | 9/10 | 7/10 | 16/20 | 11/20 | 7/10 | 6/10 |
| move_playingcard_away | 10/10 | 5/10 | 20/20 | 11/20 | 10/10 | 6/10 |
| move_stapler_pad | 8/10 | 1/10 | 15/21 | 2/21 | 7/10 | 1/10 |
| open_laptop | 10/10 | 8/10 | 20/21 | 17/21 | 7/10 | 5/10 |
| open_microwave | 10/10 | 0/10 | 14/19 | 0/19 | 7/10 | 1/10 |
| pick_diverse_bottles | 7/10 | 6/10 | 11/21 | 10/21 | 6/10 | 5/10 |
| pick_dual_bottles | 6/10 | 7/10 | 14/20 | 16/20 | 5/10 | 6/10 |
| place_a2b_left | 10/10 | 8/10 | 20/20 | 16/20 | 9/10 | 8/10 |
| place_a2b_right | 8/10 | 8/10 | 18/20 | 15/20 | 9/10 | 7/10 |
| place_bread_basket | 9/10 | 2/10 | 20/21 | 9/21 | 10/10 | 6/10 |
| place_bread_skillet | 8/10 | 5/10 | 17/20 | 9/20 | 9/10 | 6/10 |
| place_burger_fries | 10/10 | 8/10 | 20/20 | 18/20 | 10/10 | 8/10 |
| place_can_basket | 8/10 | 6/10 | 12/20 | 13/20 | 7/10 | 4/10 |
| place_cans_plasticbox | 10/10 | 3/10 | 20/20 | 8/20 | 10/10 | 4/10 |
| place_container_plate | 10/10 | 9/10 | 20/20 | 17/20 | 10/10 | 10/10 |
| place_dual_shoes | 7/10 | 0/10 | 17/22 | 1/22 | 7/10 | 1/10 |
| place_empty_cup | 10/10 | 7/10 | 20/20 | 15/20 | 10/10 | 9/10 |
| place_fan | 7/10 | 2/10 | 14/20 | 3/20 | 10/10 | 1/10 |
| place_mouse_pad | 5/10 | 2/10 | 12/20 | 3/20 | 7/10 | 3/10 |
| place_object_basket | 7/10 | 4/10 | 18/21 | 9/21 | 8/10 | 1/10 |
| place_object_scale | 7/10 | 3/10 | 14/20 | 4/20 | 9/10 | 1/10 |
| place_object_stand | 7/10 | 5/10 | 16/20 | 11/20 | 8/10 | 8/10 |
| place_phone_stand | 8/10 | 7/10 | 16/20 | 10/20 | 9/10 | 2/10 |
| place_shoe | 10/10 | 4/10 | 19/20 | 8/20 | 9/10 | 4/10 |
| press_stapler | 10/10 | 6/10 | 21/21 | 14/21 | 10/10 | 9/10 |
| put_bottles_dustbin | 5/9 | 0/9 | 11/20 | 0/20 | 9/10 | 0/10 |
| put_object_cabinet | 7/10 | 1/10 | 13/20 | 1/20 | 6/10 | 1/10 |
| rotate_qrcode | 8/10 | 4/10 | 17/20 | 6/20 | 8/10 | 2/10 |
| scan_object | 10/10 | 3/10 | 17/20 | 4/20 | 7/10 | 3/10 |
| shake_bottle | 10/10 | 10/10 | 20/20 | 20/20 | 10/10 | 10/10 |
| shake_bottle_horizontally | 10/10 | 9/10 | 20/20 | 18/20 | 9/10 | 10/10 |
| stack_blocks_three | 8/10 | 0/10 | 17/20 | 0/20 | 9/10 | 0/10 |
| stack_blocks_two | 10/10 | 1/10 | 20/20 | 4/20 | 10/10 | 3/10 |
| stack_bowls_three | 8/10 | 1/10 | 12/21 | 4/21 | 9/10 | 3/10 |
| stack_bowls_two | 10/10 | 10/10 | 21/21 | 19/21 | 8/10 | 6/10 |
| stamp_seal | 9/10 | 5/10 | 18/21 | 7/21 | 6/10 | 3/10 |
| turn_switch | 4/10 | 2/10 | 12/20 | 7/20 | 7/10 | 4/10 |
| **total** | **410/499 = 82.2%** | **234/499 = 46.9%** | **830/1015 = 81.8%** | **459/1015 = 45.2%** | **421/500 = 84.2%** | **227/500 = 45.4%** |

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
