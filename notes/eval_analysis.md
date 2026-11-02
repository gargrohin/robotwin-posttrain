# Eval analysis: GR00T mix200_150k, checkpoint 150000

Started 2026-11-02. What goes well and what goes badly for our best model so far, to decide
what to improve next.

## The eval in brief

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

## 1. Longer tasks succeed less

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

### What this doesn't show

- **Length is not the only reason.** Short tasks fail too: of the 20 tasks of 150 steps or
  fewer, 6 score 35% or less on randomized: lift_pot 5%, place_fan 15%, place_mouse_pad 15%,
  place_object_scale 20%, stamp_seal 33%, turn_switch 35%.
- **Long tasks mostly use both arms** (5 of the 6 tasks over 350 steps). So this can't tell
  whether the length itself or the two-arm coordination is what's hard.
- **Single-task numbers are rough.** Each task has about 20 randomized episodes and 10 or
  20 clean ones. Group averages are steadier than any one task's rate.

**Reproduce:** `analysis/.venv/bin/python analysis/length_spread.py` prints these tables
(and the ones in the appendix).

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

## Appendix B: every task

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
