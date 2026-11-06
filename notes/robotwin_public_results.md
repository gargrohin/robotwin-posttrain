# RoboTwin 2.0: public results and hill-climbing recipes

Web research done 2026-11-05. Every number has a link. "Secondhand" means the number comes from
a paper quoting another paper, and the original was not checked. Success rates are percentages,
written **randomized / clean**, the same order as our own results.

Three training protocols come up again and again:

- **Protocol A (official leaderboard):** 50 clean demos per task. Evaluated with 100 trials per
  task in clean scenes ("Easy") and randomized scenes ("Hard"). Entries are either one model for
  all 50 tasks (co-train) or one model per task (single).
- **Protocol B (the "Motus protocol"):** 50 clean + 500 randomized demos per task, one model for
  all 50 tasks, usually 100 trials per task in each setting. Most papers from 2026 use it.
- **Ours:** 50 clean + 200 randomized demos per task, one model for all 50 tasks. Evaluated on
  about 20 randomized + 10 clean scenes per task. This is much closer to B than to A.

## 1. Summary

- **Cosmos3-Nano and FLUX 3 Action have no public RoboTwin 2.0 numbers.** NVIDIA's Cosmos 3
  report and BFL's FLUX 3 Action page report robot results only on RoboLab, RoboArena,
  MolmoSpaces and real robots. The only Cosmos-backbone policy with RoboTwin 2.0 numbers is
  AW-0 (Cosmos-Predict2.5-2B): 20.6 / 80.8 under protocol A.
- **The official leaderboard trains on clean demos only.** It was updated 2026-10-30 and has 33
  entries. The best are GO-3 Nano Preview (GE-Act 2.0, AgiBot) at 81.5 / 88.5 and PrismWAM*
  (THU & NVIDIA, uses depth) at 75.5 / 91.8. The paper's original single-task baselines (ACT, DP,
  DP3, RDT, π0) score 0.6–16.3 randomized and 28–55 clean.
- **Leaderboard numbers can't be compared with ours.** Because protocol A never trains on
  randomized scenes, the same model scores very differently under A and B. Fast-WAM gets
  1.9 / 77.8 on the leaderboard and 91.8 / 91.9 in its own protocol-B paper.
- **Under protocol B the best published models reach 92–96%.** MotuBrain 96.1 / 95.8;
  PatchWAM 96.2 / 96.0 (with 2,100 extra demos); Qwen-RobotManip-Context 94.0 / 93.7; LingBot-VA
  91.6 / 92.9. Our Cosmos3-Nano (85.0 / 88.6, with 200 instead of 500 randomized demos per task)
  is at the level of Motus (87.0 / 88.7) and LingBot-VLA (86.7 / 88.6), about 7–11 points below
  the top.
- **GR00T has few public numbers, and they agree with ours.**
  - GR00T N1.7 under protocol A: 20.7 / 43.6, from the Qwen-RobotManip report. Our clean-only
    GR00T run: 9.7 / 42.6.
  - GR00T N1.7 trained per task on 50 randomized demos: 55.4 randomized, from LM-X.
  - No GR00T number under protocol B was found.
- **Baseline numbers are unreliable across papers.** π0.5 under the "same" protocol B is reported
  as 43.8 / 43.0 (Motus), 76.8 / 82.7 (LingBot-VA, LingBot-VLA) and 87.0 / 86.8 (InternVLA-A1).
  Many papers copy baselines instead of re-running them.
- **The closest recipe to our starting point is Q-Planning.** It took a frozen Fast-WAM from 83.2
  to 91.4 on 47 RoboTwin tasks. Only a small Q-function (a learned value model) was trained, on
  the policy's own rollouts, and it was used to pick among 32 sampled action chunks. This fits our
  finding that about half of failed scenes succeed on a retry.
- **Correction and recovery data is the best-supported fix for our failure modes:**
  - SimPT (HG-DAgger with a simulated expert): π0.5 67.3 → 82.1 on 7 tasks.
  - CARE (corrective demos on 7 hard two-arm tasks): π0 39.6 → 60.7. About 9 points of that come
    from the data alone; the rest needs a GPT-4.1 planner and a 3D monitor.
- **RL fine-tuning gives large gains only from low starting points.** SimpleVLA-RL, RLinf,
  MART-GRPO and dVLA-RL take single-task models from 20–60% to 69–92% on 3–12 tasks. No RL result
  covers all 50 tasks with one model, starts above 80%, or uses a video-model policy.
- **Hardest tasks match ours.** Hanging Mug is lowest or near-lowest in almost every paper. Turn
  Switch, Move Stapler Pad, Put Object Cabinet, Rotate QRcode, Place Mouse Pad and Handover Block
  are also common. 10 of our 14 tasks under 80% are among the 15 lowest on the leaderboard's top
  models. Open Microwave is among the hardest elsewhere; our Cosmos solves it (18/19).

## 2. Results table

Robot is aloha_agilex (Aloha-AgileX, two 6-DoF arms + grippers) unless noted. "Eps" is
evaluation episodes per task per setting.

### 2a. Official leaderboard (protocol A)

Source: https://robotwin-platform.github.io/leaderboard, with data from
https://robotwin-platform.github.io/data/robotwin_leaderboard.json (updated 2026-10-30).
- Training: 50 demo_clean per task.
- Eval: 100 trials per task per setting.
- All entries train one model for all 50 tasks (co-train) unless marked "single".
- The leaderboard ranks by the mean of clean and randomized.

| model (contributor) | randomized | clean | listed |
|---|---|---|---|
| GO-3 Nano Preview = GE-Act 2.0 (AgiBot) | 81.48 | 88.54 | 2026-10-25 |
| PrismWAM* (THU & NVIDIA; depth at inference; Wan2.1-14B) | 75.50 | 91.84 | 2026-10-23 |
| PICA-WAM (Yude Zou) | 74.70 | 90.68 | 2026-10-30 |
| CloudRobo-WM | 75.00 | 88.90 | 2026-10-19 |
| MWA-Phoenix (Anyverse & OpenPhoenix) | 72.42 | 86.54 | 2026-10-21 |
| PatchWAM | 66.72 | 91.56 | 2026-10-02 |
| ME-Dex-1.0 (Li Auto) | 68.12 | 89.58 | 2026-09-24 |
| EWAM | 72.08 | 82.16 | 2026-10-29 |
| ViGAR (PKU & AgiBot) | 67.02 | 82.00 | 2026-10-11 |
| UniWAM (OLA-HKUSTGZ) | 67.56 | 75.12 | 2026-09-01 |
| PrismWAM (depth; Wan2.2-5B) | 53.16 | 87.98 | 2026-10-23 |
| GigaBrain-0.7 (GigaAI) | 67.92 | 66.80 | 2026-08-15 |
| WorldLA (ISCAS) | 46.74 | 82.18 | 2026-10-24 |
| WorldScape Policy 2.0 (ManiFold AI) | 38.50 | 84.92 | 2026-09-24 |
| 4D-WAM (OpenHelix) | 41.80 | 81.50 | 2026-08-18 |
| π0.5 (RoboTwin team run) | 46.00 | 70.66 | 2026-08-10 |
| OLA-Geo | 33.64 | 82.96 | 2026-09-01 |
| Spatial Forcing | 26.74 | 77.18 | 2026-08-17 |
| X-WAM | 25.84 | 69.98 | 2026-08-10 |
| X-VLA | 20.95 | 67.96 | 2026-08-10 |
| ABot-M0 | 30.36 | 57.40 | 2026-08-17 |
| EventVLA | 15.74 | 65.58 | 2026-08-10 |
| Xiaomi Robotics-0 | 18.22 | 62.86 | 2026-08-10 |
| FastWAM | 1.92 | 77.76 | 2026-08-10 |
| GalaxeaVLA | 12.72 | 62.72 | 2026-08-17 |
| Discrete Forcing | 11.20 | 59.32 | 2026-10-10 |
| AHA-WAM | 3.20 | 64.32 | 2026-08-10 |
| starVLA | 3.16 | 46.52 | 2026-08-17 |
| π0 (single) | 16.34 | 46.42 | 2025-08-04 |
| DP3 (single; point cloud) | 4.96 | 55.24 | 2025-08-04 |
| RDT (single) | 13.72 | 34.50 | 2025-08-04 |
| ACT (single) | 1.74 | 29.74 | 2025-08-04 |
| DP (single) | 0.64 | 28.04 | 2025-08-04 |

The single-task rows are the RoboTwin 2.0 paper's Table 5: https://arxiv.org/abs/2506.18088
(v1 2025-06-22, v2 2025-08-27). Pretrained models (RDT, π0) lose 21–30 points from clean to
randomized; models without pretraining fall to near 0.

### 2b. Other protocol-A results (clean-only training) from papers

| model | protocol details | randomized | clean | source, date |
|---|---|---|---|---|
| GR00T N1.7 | protocol A; how GR00T was trained is not stated | 20.7 | 43.6 | Qwen-RobotManip Table 5, https://arxiv.org/abs/2606.17846 (v2 2026-06-17). Also quoted in GE-Act 2.0 and 4D-WAM (https://arxiv.org/abs/2608.08023) |
| π0.5 (Qwen team run) | protocol A | 47.9 | 73.1 | same Table 5 |
| ABot-M0 (Qwen team run) | protocol A | 36.0 | 70.7 | same |
| StarVLA (Qwen team run) | protocol A | 10.6 | 58.1 | same; the leaderboard has 3.16 / 46.52 |
| Qwen-RobotManip-Context (joint) | protocol A | 69.4 | 84.7 | same |
| X-VLA (own paper) | "leaderboard settings", 50 demos per task | 39.0 | 70.0 | https://arxiv.org/abs/2510.10274 (2025-10-11); leaderboard re-run: 20.95 / 67.96 |
| AW-0, Agentic WAM (video expert from Cosmos-Predict2.5-2B) | 50 clean demos per task, **50 eps** | 20.6 | 80.8 | https://arxiv.org/abs/2610.30136 (2026-10-21) |
| π0.5 (AW-0 paper run) | same, 50 eps | 27.7 | 54.9 | same |
| WorldScape Policy 2.0 | protocol A | 38.50 | 84.92 | https://arxiv.org/abs/2607.18840 (v3 2026-10-20) |
| PatchWAM | protocol A plus photometric, AdaIN and FFT augmentation of the clean demos | 66.72 | 91.56 | https://arxiv.org/abs/2609.25961 (2026-09-22) |
| **Ours: GR00T N1.7** (`50task_60k`) | 50 clean demos per task; ~20 randomized + 10 clean eps | 9.7 | 42.6 | `AGENTS.md` |

### 2c. Protocol B (50 clean + 500 randomized per task, multi-task)

Most use 100 trials per task; exceptions are noted.

| model | type | randomized | clean | source, date |
|---|---|---|---|---|
| MotuBrain (ShengShu, Vidu-based) | WAM | 96.1 | 95.8 | https://arxiv.org/abs/2604.27792 (v5 2026-07-15); eps not stated |
| PatchWAM + 2,100 extra demos on 7 weak tasks | WAM | 96.20 | 96.04 | https://arxiv.org/abs/2609.25961 (2026-09-22); without the extra demos 94.46 / 94.68 |
| Long-WAM, IDM variant (NVIDIA/MIT; LongLive backbone) | WAM | 94.2 | 94.7 | https://arxiv.org/abs/2610.10528 (2026-10-07); training protocol not stated in the parts read |
| WorldScape Policy 2.0 | WAM | 94.2 | 94.3 | https://arxiv.org/abs/2607.18840 |
| Qwen-RobotManip-Context | VLA | 94.0 | 93.7 | https://arxiv.org/abs/2606.17846 |
| Riemann-1.0 | WAM | 94.0 | 94.6 | https://arxiv.org/abs/2608.27033 |
| LingBot-VA 2.0 | WAM | 93.4 | 93.8 | https://arxiv.org/abs/2607.08639 (v2 2026-07-16) |
| ReWAM (frozen DINOv3, no video pretraining) | WAM | 93.6 | 93.6 | https://arxiv.org/abs/2609.38163 (2026-09-29) |
| Rolling-WAM | WAM | 93.0 | 93.5 | https://arxiv.org/abs/2609.30247 (v2 2026-10-05) |
| AHA-WAM | WAM | 92.2 | 93.4 | https://arxiv.org/abs/2606.09811 (2026-06-09); leaderboard (A): 3.20 / 64.32 |
| Faster-WAM | WAM | 92.3 | 92.8 | https://arxiv.org/abs/2608.04404 (2026-08-05) |
| LingBot-VA (Ant/Robbyant) | WAM | 91.55 | 92.93 | https://arxiv.org/abs/2601.21998 (v1 2026-01-29) |
| Fast-WAM (Tsinghua/Galaxea) | WAM | 91.78 | 91.88 | https://arxiv.org/abs/2603.16666 (v2 2026-03-23); leaderboard (A): 1.92 / 77.76 |
| ACE-Ego-0 | VLA | 90.62 | 91.12 | https://arxiv.org/abs/2606.17200 (2026-06-15) |
| Hy-Embodied-0.5-VLA (Tencent) | VLA | 90.1 | 90.9 | https://huggingface.co/tencent/Hy-Embodied-0.5-VLA-RoboTwin; paper https://arxiv.org/abs/2606.14409 (2026-06). 32 GPUs, 150K steps, batch 128 |
| X-WAM | WAM | 90.7 | 89.8 | https://arxiv.org/abs/2604.26694 (v2 2026-05-07); leaderboard (A): 25.84 / 69.98 |
| InternVLA-A1-3B | VLA | 89.64 | 89.40 | https://huggingface.co/InternRobotics/InternVLA-A1-3B-RoboTwin; paper https://arxiv.org/abs/2601.02456 |
| Being-H0.7 | latent WAM | 89.6 | 90.2 | https://arxiv.org/abs/2605.00078 (2026-04-30) |
| JoyAI-RA 0.1 | VLA | 89.28 | 90.48 | https://arxiv.org/abs/2604.20100 (2026-04-22) |
| LiLa-WAM (0.5B) | WAM | 89.04 | 90.48 | https://arxiv.org/abs/2608.03701 (2026-08-04); **50 eps**; ~110 GPU-hours on one RTX 5090 |
| ZR-0 | VLA | 87.98 | 88.70 | https://arxiv.org/abs/2606.30552 (v2 2026-07-01) |
| Motus (THU/ShengShu) | WAM | 87.02 | 88.66 | https://arxiv.org/abs/2512.13030 (v2 2025-12-25) |
| LingBot-VLA, with / without depth | VLA | 86.68 / 85.34 | 88.56 / 86.50 | https://arxiv.org/abs/2601.18692 (v4 2026-06-15) |
| GigaWorld-Policy (Wan2.2-5B) | WAM | 85 | 87 | https://arxiv.org/abs/2603.17240 (2026-03-18); the appendix gives 86 clean |
| ABot-M0 | VLA | 85.08 | 86.06 | https://arxiv.org/abs/2602.11236 (v2 2026-04-14) |
| X-Tokenizer + Wall-OSS | VLA | 80.88 | 84.70 | https://arxiv.org/abs/2606.14752 (2026-06-28). Numbers from a search snippet of the PDF; the HTML has them only in a figure, so unverified |
| π0.5, three different reported runs under B | VLA | 43.84 / 76.76 / 86.96 | 42.98 / 82.74 / 86.76 | Motus run (40k steps) / LingBot-VLA run (copied by most later papers) / InternVLA-A1 card |
| X-VLA (Motus run) | VLA | 72.84 | 72.80 | Motus |
| π0, two reported runs | VLA | 58.4 / 79.50 | 65.9 / 79.98 | LingBot-VA / InternVLA-A1 card |
| DINO+MLP probe, 0.09B, no language | baseline | 59.4 | 60.4 | https://arxiv.org/abs/2606.04233 (2026-06-02) |

Secondhand only: PatchWAM's Table 1 also lists AIM 92.1 / 94.0, G0.5 92.8 / 93.7, Next Forcing
93.5 / 94.1, ABot-M0.5 94.2 / 94.0, HoloBrain-0 90.8 / 91.3, MECo-WAM 91.98 / 93.26 and ImageWAM
93.56 / 93.2. ZimaBlue reports 94.3 / 94.7 on "a mixture of clean and randomized demos", counts not
stated (https://arxiv.org/abs/2609.00188).

### 2d. Other setups that matter for us

| model | protocol | randomized | clean | source, date |
|---|---|---|---|---|
| GR00T N1.7 (public July 2026 code) | 50 *randomized* demos per task, post-trained per task, 100 eps, randomized eval only | 55.4 | n/a | LM-X, https://arxiv.org/abs/2608.25757 (v5 2026-10-08) |
| LM-X (Cosmos-Reason2-2B backbone) | same | 74.1 | n/a | same |
| GR00T N1.5, VLM frozen | one model for 50 tasks, 50 demos per task (clean or randomized not stated), 100 eps, eval setting not stated | 61.32 (setting unknown) | | PVI, https://arxiv.org/abs/2603.12772 (2026-03-13) |
| Fast-WAM, frozen | protocol-B weights, 47 tasks, 20 eps, setting not stated | 83.2 (setting unknown) | | Q-Planning, https://arxiv.org/abs/2608.21204 (2026-08-21) |
| **Ours: Cosmos3-Nano** (decay 5,500) | 50 clean + 200 randomized, multi-task, ~20 randomized + 10 clean eps, GR00T's seeds | **85.0** | **88.6** | `notes/eval_analysis.md` |
| **Ours: FLUX 3 Action** (step 3,000) | same | **71.0** | **71.1** | `notes/results.md` |
| **Ours: GR00T N1.7** (`mix200_150k`) | same | **45.3** | **45.5** | `notes/results.md` |

### 2e. Cosmos and FLUX: what is public

- **Cosmos 3 report:** https://arxiv.org/abs/2606.02800 (v4 2026-06-23). Policy results only on
  RoboLab, RoboArena and MolmoSpaces. No RoboTwin.
- **FLUX 3 Action:** https://bfl.ai/models/flux-3-action (September 2026). RoboLab-120 results:
  FLUX 3 Action 42.92, Cosmos3-Nano-Policy 36.8, π0.5 28.0, GR00T N1.6 7.2. On a real robot
  (Positronic), FLUX 28/30, Cosmos3-Nano 27/30, π0.5 13/30. The page does not mention RoboTwin.
  The model card (https://huggingface.co/black-forest-labs/flux-3-action-base) has no benchmark
  numbers.
- **Cosmos Policy:** https://arxiv.org/abs/2601.16163 (2026-01-22). LIBERO, RoboCasa and real
  ALOHA. No RoboTwin.
- **Community Cosmos3 RoboTwin checkpoints, no scores reported:**
  - https://huggingface.co/GM717/cosmos3-edge-robotwin-aloha-agilex (Cosmos3-Edge, clean +
    randomized data)
  - https://huggingface.co/Myungkyu/cosmos3_mot_l36_d1024_robotwin_b64_60k_ema (Cosmos3-Nano with
    an action DiT)
- **EWAM (Astronex):** https://arxiv.org/abs/2606.12690. Built on a frozen
  Cosmos3-Nano-Policy-DROID; RoboLab only. This is a different model from the "EWAM" on the
  leaderboard.

## 3. Recipes table

"Protocol C" here means a custom setup: a few tasks, one model per task, about 1,000 demos per
task, randomized training and eval (the SimpleVLA-RL / RLinf family). Gains are before → after.

| method | base model | gain | cost | code |
|---|---|---|---|---|
| **Q-Planning:** off-policy Q-function picks among 32 sampled chunks; 10 rounds of self-improvement on own rollouts; policy frozen ([2608.21204](https://arxiv.org/abs/2608.21204), 2026-08-21) | Fast-WAM, protocol-B weights | 47 tasks, 20 eps: 83.2 → 83.8 (selection only) → **91.4** (after 10 rounds) | ~1B-parameter Q-net (DINOv2 + T5 + 500M decoder); 45k steps on 4×H200; per round, 100 episodes per task + 200 Q-only gradient steps (~47k sim episodes total, my arithmetic); 32 samples per planning step at test time | project page only, no repo found |
| **SimPT:** HG-DAgger with a CuRobo expert restarted from saved progress snapshots ([2610.29161](https://arxiv.org/abs/2610.29161), 2026-10-21) | π0.5, 50 demos per task | 7 tasks, 200 held-out seeds each: SFT 67.29 → HG-DAgger **82.14**; RECAP 76.50; FlowPRO 71.86; DPO+SFT 63.14 (worse) | 240 collection seeds per task; 945 recovery attempts with the progress graph (3,781 without); 10k training steps per method; GPU count not reported | none |
| **CARE:** corrective demos from fitted failure distributions + stage monitor at inference ([2609.24118](https://arxiv.org/abs/2609.24118), 2026-09-21) | π0, π0-FAST; 150 demos per task, demo_randomized | 7 hard two-arm tasks: π0 39.6 → 60.7; π0-FAST 17.4 → 33.1. Data alone (5-task mix including one RoboFactory task): 31.6 → 40.8 | 100 preliminary rollouts per task; 50 corrective trajectories per error type; 40k steps at batch 32 on 2×A800. Inference uses a GPT-4.1 planner + SAM 3 + Depth Anything 3 monitor | https://github.com/xiaojunlan/care |
| **RePO-VLA:** failure-recovery data (FRBench: 46 tasks, 23,453 episodes, 6,392 recovery) + value token ([2605.09410](https://arxiv.org/abs/2605.09410), 2026-05-10) | π0.5 | 10 tasks, 50 eps: nominal 33.6 / 27.4 → 44.0 / 44.6; with injected failures 15.4 / 15.0 → 43.0 / 37.0 | dataset build; not stated | none found |
| **SimpleVLA-RL:** GRPO, binary success reward ([2509.09674](https://arxiv.org/abs/2509.09674), 2025-09-11) | OpenVLA-OFT, one model per task, 1,000 demos per task (protocol C) | 12 tasks, randomized: 38.3 → 68.8. 0 demos: 0 → 0; 100 demos: 7.3 → 25.4 (5 tasks). Learned to push objects into place instead of grasping ("pushcut") | 8×A800; 1,000 RL scenes per task; GPU-hours not reported | https://github.com/PRIME-RL/SimpleVLA-RL |
| **RLinf-VLA:** GRPO / PPO ([2510.06710](https://arxiv.org/abs/2510.06710); docs https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/robotwin.html) | OpenVLA-OFT; π0 / π0.5; LingBot-VLA | OFT, 6 tasks: 24.48 → 84.63 (SFT scored greedy, RL sampled, success-at-any-time). Docs: OFT 7 tasks 28.79 → 86.16; π0 adjust_bottle 76.56 → 98.44; LingBot-VLA click_bell 96.88 → 98.75, place_shoe 93.75 → 98.44 | 128 parallel envs, 8–16 GPUs; hours not stated; configs for 10 tasks only | https://github.com/RLinf/RLinf |
| **MART-GRPO** ([2610.32431](https://arxiv.org/abs/2610.32431), 2026-10-23) | OpenVLA-OFT, one model per task | 7 tasks: SFT 40.68 → trajectory GRPO 65.01 → 68.64 | 8×H100; 40 updates × 512 rollouts (~20k rollouts per task) | none |
| **dVLA-RL:** PPO via RLinf ([2606.23623](https://arxiv.org/abs/2606.23623), 2026-06-22) | MM-ACT (discrete diffusion VLA), multi-task, 1,000 demos per task | 8 tasks: 61.4 → 92.0. RL number is best online success during training (64 episodes) | 512 trajectories per step; GPUs not stated | none |
| **Multi-agent RFT:** per-arm filtered BC on positive-advantage trajectories + human demos where the policy keeps failing + DSRL online ([2609.36588](https://arxiv.org/abs/2609.36588), 2026-09-29) | π0, π0.5 | 4 tasks: +23.1 average; π0.5 Lift Pot 24 → 50, Handover Mic 43 → 73 | not stated | anonymous repo link in paper |
| **TACO:** pseudo-count verifier, best of 50 sampled chunks ([2512.02834](https://arxiv.org/abs/2512.02834), 2025-12-02) | π0.5 | 41 tasks, 100 seeds, apparently clean: 59.3 → 64.0 (Handover Block 24 → 36, Move Can Pot 42 → 57) | no retraining; 50 samples per step | https://github.com/breez3young/TACO |
| **PACE:** chooses how many steps of each chunk to execute ([2606.00537](https://arxiv.org/abs/2606.00537), v4 2026-11-03) | π0.5, one model per task, 50 clean demos | 50 tasks, clean, 900 eps each: best fixed horizon (H=25) 57.8 → 64.2; worse than the best fixed horizon on 19 of 50 tasks | training-free; negligible selector cost | via https://niejunnan25.github.io/pace-project/ |
| **GPC:** combine two policies' denoising scores ([2510.01068](https://arxiv.org/abs/2510.01068)) | RDT + DP3 | 6 tasks: best single 0.65 → 0.72; Hanging Mug 0.21 → 0.36 | per-task weight search | not checked |
| **Video co-training** during policy training (Fast-WAM) | Fast-WAM | protocol B: 84.80 / 82.76 → 91.78 / 91.88 | part of training | see Fast-WAM paper |
| **Embodied / video pretraining** | Motus; MotuBrain; LingBot-VA | Motus no pretraining 77.00 / 72.8 → 87.02 / 88.66; MotuBrain 91.3 / 91.5 → 96.1 / 95.8; LingBot-VA without joint pretraining 80.6 clean → 92.9 | Motus ~8,000 + 10,000 + 400 GPU-hours over its 3 stages | MotuBrain: github.com/shengshu-ai/Motubrain |
| **Extra demos on the weakest tasks** (PatchWAM) | PatchWAM | 94.46 / 94.68 → 96.20 / 96.04 with 2,100 extra demos on 7 tasks (how they were made is not stated) | 110k updates on 32 GPUs | not checked |
| **Randomized training data** | Fast-WAM; RDT / π0 | Fast-WAM protocol A → B: 1.92 / 77.76 → 91.78 / 91.88 (different runs, not a controlled ablation). RoboTwin paper, 5 unseen tasks, randomized eval: π0 pretrained on clean 24.9 → on randomized 29.1 | data is free; RoboTwin ships 500 randomized per task | n/a |
| **Cosmos3-Nano-Sim-Transfer augmentation** of demos ([model card](https://huggingface.co/wjwjwjwj/gr00t-n1.7-robotwin-adjust_bottle-cosmos-aug)) | GR00T N1.7, VLM frozen | adjust_bottle only, 100 eps: 52 / 73 → 97 / 99 (50 clean + 138 generated scenes) | 2×B300 | https://github.com/nota-github/xpu-gr00t-aug-recipe |
| **Cross-embodiment co-training** (X-Tokenizer) | Wall-OSS | 64.0 / 70.9 → 74.4 / 77.9 (joint vs single-embodiment) | not stated | not checked |
| **Adapter on a frozen WAM** (CoRe-WAM, [2609.27314](https://arxiv.org/abs/2609.27314)) | Motus | +2.58 / +3.56 | 1.59M parameters, 5k updates on 8 H100s | not checked |
| **Asynchronous execution** (harmful unless the model is built for it) | LingBot-VA; Fast-WAM | LingBot-VA naive async: 92.9 → 74.3 clean. In Long-WAM's test: Fast-WAM 91.8 → 76.4, LingBot-VA 92.2 → 46.7, Long-WAM 94.4 → 94.2 | n/a | Long-WAM: github.com/NVlabs/LongLive |
| **Cosmos Policy best-of-N planning** with world model + value head (no RoboTwin result) | Cosmos Policy | +12.5 on 2 real ALOHA tasks | planner fine-tuned on 648 rollouts; ~5 s per action chunk | via https://research.nvidia.com/labs/dir/cosmos-policy/ |

**Searched, no RoboTwin 2.0 result found:**
- **RL methods:** πRL (LIBERO, ManiSkill), VLA-RL, TGRPO, ReinFlow, FPO, ARFM, SRPO, PLD, GR-RL,
  π*0.6/RECAP (real robots only).
- **World-model RL:** WMPO, World-Env, VLA-RFT, WAM-RL (LIBERO only).
- **Recovery and test-time:** FailSafe, Do What You Say, FORGE-Tree, RoboMonkey, Beyond Success.
- **Other gaps:** no RTC or temporal-ensembling ablation on RoboTwin 2.0, no controlled
  clean-only vs clean+randomized ablation on one model, and no study of 200 vs 500 randomized
  demos.

## 4. Failure modes and hardest tasks reported elsewhere

**Hardest tasks, by source:**

- **Official leaderboard, mean of the 9 entries averaging ≥74%** (my computation from the
  leaderboard JSON):
  - Randomized, lowest 15: open_microwave 28.1, hanging_mug 33.8, move_stapler_pad 36.0,
    put_object_cabinet 42.1, rotate_qrcode 49.0, blocks_ranking_size 50.7, move_can_pot 50.9,
    place_mouse_pad 56.3, stamp_seal 56.7, place_can_basket 57.8, handover_block 60.2,
    place_dual_shoes 60.6, turn_switch 60.6, beat_block_hammer 62.2, place_fan 62.3.
  - Clean, lowest: hanging_mug 43.3, put_object_cabinet 59.7, open_microwave 63.9.
- **Protocol B papers** (randomized / clean):
  - Motus: Hanging Mug 38 / 38, Move Can Pot 34 clean, Handover Block 73 / 86.
  - LingBot-VA: Hanging Mug 28 / 40, Turn Switch 45 / 44.
  - Fast-WAM: Hanging Mug 62 / 58, Turn Switch 59 / 61, Open Microwave 45 / 62.
  - MotuBrain: Hanging Mug 43 / 55, Turn Switch 84 / 82, Put Bottles Dustbin 87 / 87.
  - JoyAI-RA: Hanging Mug 28 / 31, Click Alarmclock 56 / 64, Stack Blocks Three 62 / 60.
  - ACE-Ego-0: Hanging Mug 31 / 29, Turn Switch 57 / 59, Click Alarmclock 38 / 52.
  - LingBot-VLA: Hanging Mug 27–53% across variants; π0.5 17 / 18.
  - GigaWorld-Policy: Hanging Mug 0.12 / 0.16, Scan Object 0.64 / 0.60.
- **GR00T family:**
  - GR00T N1.7 (LM-X): handover_block 0, hanging_mug 0, place_phone_stand 0,
    put_bottles_dustbin 1, open_microwave 13, place_fan 13.
  - GR00T N1.5, single-task (PVI): pick_diverse_bottles 3, scan_object 4.
- **RoboTwin 2.0 paper:** Blocks Ranking RGB / Size near 0 for most single-task baselines, even
  in clean scenes.
- **Robustness study** (https://arxiv.org/abs/2603.22078): camera viewpoint and robot initial
  state are the hardest extra perturbations.

**Comparison with ours:**

- **Shared hard tasks:** 10 of our 14 tasks under 80% randomized are among the leaderboard's 15
  lowest: hanging_mug, handover_block, place_mouse_pad, turn_switch, rotate_qrcode,
  place_dual_shoes, place_fan, move_can_pot, put_object_cabinet, move_stapler_pad. Hanging Mug is
  the hardest task for nearly everyone. The best reported are AHA-WAM (78 randomized) and
  Rolling-WAM (58 / 70), and most top models are at 30–60.
- **Where we differ:**
  - open_microwave: ours 18/19 randomized; most others find it hard.
  - put_bottles_dustbin and pick_diverse_bottles: rarely in others' bottom lists. GR00T-family
    models struggle with them (GR00T N1.7 1% on put_bottles_dustbin; GR00T N1.5 3% on
    pick_diverse_bottles).
- **Failure types reported elsewhere:**
  - **Grasp errors.** RePO-VLA lists premature gripper close, grasp slip, grasp position offset
    and grasp orientation mismatch. These are close to our "first grasp knocks the object over
    (gripper not open enough, or off center)".
  - **Arm conflicts.** FocusVLA (https://arxiv.org/abs/2603.28740, 7 tasks) reports failures
    where "both arms approach the target object simultaneously". It says a larger action chunk
    helps, with no numbers. It also reports objects pushed out of reach. This may relate to our
    open question about one arm reaching for the other arm's object.
  - **Pushing instead of grasping.** SimpleVLA-RL reports RL learning to push objects into place
    ("pushcut").
  - **Lack of recovery.** This is the stated motivation of SimPT, CARE and RePO-VLA; none
    quantifies it for a strong base model.
- **Not found elsewhere:**
  - No paper gives a failure-type breakdown for a top model like ours. They report per-task
    success only.
  - No quantified one-arm vs two-arm comparison was found. CARE and SimPT choose hard two-arm
    tasks as targets, which is indirect support.
- **Retry noise:**
  - Overall, reruns of the same model differ by about 1 point: ARPS's rerun of Fast-WAM gives
    90.4 / 91.1 vs 91.9 / 91.8 in its own paper (https://arxiv.org/abs/2609.23369).
  - PatchWAM says differences under 1 point may be noise.
  - No public number matches our "half of failed scenes succeed on a retry", which is
    per-scene. The large per-scene variance is consistent with the gains from sampling-plus-
    selection methods (Q-Planning, TACO).

## 5. Comparability notes

- **Training data decides the randomized score.** Protocol A (clean only) gives randomized
  scores far below clean (Fast-WAM 1.9 vs 77.8). Protocols with randomized data give
  randomized ≈ clean.
  - Our 50 + 200 sits between A and B, much closer to B. Compare our numbers to protocol-B
    papers. They may be lower than what the same model would get with 500 randomized demos per
    task.
  - No paper isolates the effect of 200 vs 500.
  - RoboTwin's download includes 500 randomized demos per task, already used by our `mix500`
    build (`AGENTS.md`).
- **Eval size.** The official protocol uses 100 episodes per task per setting (5,000 per
  setting). Ours uses 1,015 randomized and 499 clean.
  - 95% intervals: ours 85.0 ± 2.2 randomized and 88.6 ± 2.8 clean; 100 eps per task gives
    about ± 1.0.
  - Per task, 20 episodes gives about ± 17 points at 80%, so per-task comparisons with papers
    are weak.
  - Some papers use 50 eps (AW-0, LiLa-WAM), 20 eps (Q-Planning), or 300 randomized eps
    (Spatial Forcing, FocusVLA).
- **Seeds.** The official eval script (`scripts/eval_policy_xpolicylab.py`) starts at seed
  100000 × (1 + seed) and skips seeds where the scripted expert fails ("expert check").
  - Our test seeds also start at 100000 and come from GR00T's eval logs, so they are probably
    the first ~20 / 10 of the same seed stream. Unverified; it depends on the seed argument used.
  - Papers rarely state their seeds.
- **Instructions.** In the current RoboTwin config, clean eval uses seen instructions
  (`demo_clean.yml: eval_instruction: seen`) and randomized eval uses unseen ones
  (`demo_randomized.yml: eval_instruction: unseen`). Our setup matches.
- **Success checks.** LeRobot's RoboTwin page (https://huggingface.co/docs/lerobot/robotwin)
  says open_laptop's success check breaks when the expert check is skipped, and drops the task
  from its sweep. RoboTwin's own script runs the expert check by default. Our notes flag tight
  checks (place_mouse_pad) and the "both grippers open at the end" rule on 30 tasks; these apply
  to everyone using the same code.
- **Who ran the baselines.**
  - Many tables copy numbers: ABot-M0 takes π0.5 and X-VLA from Motus; ZR-0, JoyAI-RA and
    ACE-Ego-0 copy LingBot-VA / LingBot-VLA.
  - Reruns of one model under one protocol disagree widely: π0.5 under B ranges 43–87; StarVLA
    under A is 10.6 / 58.1 (Qwen team) vs 3.2 / 46.5 (leaderboard); X-WAM under A is 74.4 / 82.5
    (Acting in Meters) vs 25.8 / 70.0 (leaderboard).
  - Our own GR00T numbers are a fair internal comparison, since all three models use the same
    data and seeds. They are not evidence about GR00T's ceiling.
- **Single vs multi-task.** The leaderboard mixes both (single entries are marked). All RL
  recipes above are single-task on subsets of 3–12 tasks; PACE is single-task on all 50. Ours is
  one model for all 50.
- **Robot.** Everything above uses aloha_agilex except possibly SimpleVLA-RL, which says "Agilex
  Piper" without saying which RoboTwin embodiment.
- **Depth.** The leaderboard flags models that need depth at inference (PrismWAM, DP3).
  LingBot-VLA's best number uses depth. Ours uses RGB only.
- **Floor for protocol B.** A 0.09B DINO+MLP probe with no language input reaches 59.4 / 60.4.
  Protocol B is easier to score well on than A.
- **RoboTwin 1.0 vs 2.0.** SimpleVLA-RL and TACO also report RoboTwin 1.0 numbers (39.8 → 70.4
  and 32.2 → 41.3). Those are excluded from the tables above. The CVPR 2025 MEIS challenge report
  (https://arxiv.org/abs/2506.23351) used 17 tasks on an earlier RoboTwin setup and was not
  counted.

## 6. Recommendations for Cosmos3-Nano

**1. Sampling with a learned value model, then self-improvement on our own rollouts (Q-Planning
style).**
- **Why it fits:** about half of our failed scenes succeed on a retry, so the policy often has a
  good action available and doesn't pick it. Q-Planning is the only published case that starts
  from a strong policy (Fast-WAM, 83.2%, protocol-B weights) on RoboTwin, which is our situation.
  It reached 91.4 without changing the policy weights. TACO's verifier-based best-of-50 gave +4.7
  on π0.5 (59.3 → 64.0).
- **What we would build:**
  - Train a ~1B Q-function on Cosmos rollouts (successes and failures). Correction
    (2026-11-06): our evals don't record what this needs yet. Each episode saves only the
    head-camera video and a one-line summary (`episode<N>.json`), without per-step joint
    positions, actions or wrist cameras, so recording full rollouts comes first.
  - At each step, sample several action chunks from Cosmos and execute the one the Q-function
    rates highest.
  - Repeat rollouts → Q update → eval.
- **Cost:**
  - Q-function training: 4×H200 for 45k steps in the paper.
  - About 47k sim episodes over 10 rounds, at our ~4 min per episode (`AGENTS.md`).
  - Our 16B policy would need to sample N chunks per step (32 in the paper), which multiplies
    inference cost. Batched sampling helps but needs measuring.
- **Code:** none public.
- **Precedent:** Cosmos Policy uses the same idea (world model + value, best-of-N) and gained
  +12.5 on real ALOHA tasks. NVIDIA's code for it is public.

**2. Recovery data from simulator snapshots (HG-DAgger with RoboTwin's scripted expert),
focused on the 14 weak tasks.**
- **Why it fits:** it targets "first grasp knocks the object over", "placement just misses" and
  "no recovery after a mistake". Our model has never seen a failed attempt being corrected.
  - SimPT: π0.5 67.3 → 82.1 on 7 tasks, 50 demos per task. HG-DAgger beat RECAP, FlowPRO and
    DPO there; DPO made things worse.
  - CARE's corrective data alone: 31.6 → 40.8 on hard two-arm tasks. These are the tasks where
    we are weakest (80% vs 89%).
- **What we would build:**
  - Run Cosmos on training seeds of the weak tasks.
  - At failure points, restore a sim snapshot and let the CuRobo expert finish from there.
  - Fine-tune on demos + recovery segments, continuing from decay 5,500 with a short schedule.
- **Cost:** SimPT used 240 seeds per task and about 950 recovery attempts in total for 7 tasks,
  plus one fine-tune. The expert must be able to plan from mid-episode states, which RoboTwin's
  task scripts don't do out of the box; SimPT built a progress graph for this. This is the main
  engineering risk.
- **Code:** SimPT has none. CARE's code is public: https://github.com/xiaojunlan/care

**3. Move to 500 randomized demos per task (protocol B), plus extra demos on the weakest tasks.**
- **Why it fits:** randomized data is the largest known lever on the randomized score. It is
  free (already downloaded; `mix500` exists for GR00T). It also puts our numbers on the same
  protocol as nearly every 2026 paper, so the comparison in section 2c becomes direct.
  - PatchWAM's 2,100 extra demos on 7 weak tasks gave +1.7 randomized / +1.4 clean from an
    already high ~94.5, with other changes made in the same run.
  - No published ablation of 200 vs 500 exists, so the size of the gain for us is unknown.
- **Cost:** one more Cosmos fine-tune; how long depends on our run settings. Plus one full eval.
- **Code:** none needed.

**Cheap checks to run alongside:**
- **Execution horizon.** Try how many steps of each chunk are executed before replanning, on the
  10-task set. PACE showed fixed horizons of 5, 25 and 50 differ by up to 9 points on π0.5, and
  FocusVLA says longer chunks reduce arm conflicts.
- **Noise.** Repeat one eval to measure noise before comparing small gains.
- **Lower priority:** per-task RL (SimpleVLA-RL / RLinf GRPO). The gains are large but only shown
  from low starting points, one model per task, on OpenVLA-OFT-size models. Nobody has shown RL on
  a 16B video-model policy on RoboTwin. Expect it to be the most expensive option per point
  gained.
