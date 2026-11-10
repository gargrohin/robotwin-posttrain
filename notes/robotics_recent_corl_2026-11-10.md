# Robotics releases and CoRL research: October 27–November 10, 2026

Checked against public primary sources on November 10, 2026. This is a broad, prioritized survey, not an exhaustive list of every robotics paper. Dates with a known timestamp are converted to New York time (ET). A dated blog without a timestamp retains its publisher's date. An arXiv date means the first preprint submission, not necessarily the first public disclosure or the day code became available. Repository creation and last-modified dates are not release dates.

## What matters most for this project

1. **Meituan-Robotics-0 is a new candidate for local post-training.** Code, weights and RoboTwin integration are public. Its reported 93.88% randomized / 93.22% clean uses a different training/evaluation protocol from ours. Its paired correction-data method, Rollback, is also relevant.
2. **Dream Labs released reusable tools around its hosted Dream-1 model.** Its SDK and November hackathon data-collection kit are Apache-2.0; public model weights and a local trainer were not verified. Its original announcement is dated only October, while SDK updates are verified within this window.
3. **LeRobot 0.7.0 integrates Cosmos 3 and FLUX 3 Action.** ToyBlocks, DexSuite, CaPtain, ResMerge and Human-1M-100K also provide code, models or data. Open-HumanWAM/Open-HumanScale provides weights, with an incomplete broader release.
4. **Official leaderboards and new paper results differ.** The RoboTwin board's latest update is October 30, adding PICA-WAM; GO-3 Nano Preview still leads its default ranking. MR0 and HumanWAM's new paper scores are absent from that snapshot. RoboChallenge has current public feeds, but their leading entries do not establish a new improvement during this fortnight.
5. **New evidence qualifies the plan to train on successful rollouts.** The November 4 generalist-improvement paper finds that this can improve familiar tasks while hurting performance on new objects. Value-based reinforcement learning transfers better in several comparisons; its strongest implementation is still pending.
6. **GR00T 2 predicts future video and actions jointly.** Its technical report studies successful-rollout training and recovery. **Code is announced for December.** No GR00T 2 weights were verified, and the 95% five-demo headline uses tasks selected for high performance after adaptation.
7. **SimuGuard identifies a RoboTwin/SAPIEN contact bug worth investigating.** Our simulator package version matches its validated adapter; whether any of our failures have this cause remains unmeasured.

No experiments, model migrations or benchmark changes were performed for this survey. The hill-climbing experiments remain paused.

## CoRL timing

The [official CoRL site](https://www.corl.org/) lists **Austin, November 9–12, 2026**: main conference November 9–11 and workshops November 12. The conference is still underway at this survey's cutoff. The site dates its demo list October 27, late-breaking results October 31, and livestream availability November 9; accepted papers were listed October 19, outside this window.

[NVIDIA's official conference page](https://www.nvidia.com/en-us/events/corl/) schedules Jim Fan's world-action-model and agentic-robotics keynote for November 9 at **2:50 PM ET**. Do not describe November 11–12 presentations or workshop outcomes as completed yet.

## Models, data and software

Terms used below: a **policy** maps observations to robot actions; a **VLA** takes vision and language to predict actions; a **world-action model (WAM)** predicts future observations and actions together. **SFT** means supervised fine-tuning on demonstrations; **RL** means reinforcement learning from rewards. Public source code, downloadable model weights and hosted API access are recorded separately.

| Development | Verified new event in the window | What changed | Available now and limits |
|---|---|---|---|
| **GR00T 2 / GR00T 2 Turbo** | November research page and whitepaper, around the November 9 keynote; the page does not give an exact publication day | Joint video/action prediction, human-to-robot transfer, fast adaptation, RL and agent-assisted recovery | [Paper and demos](https://research.nvidia.com/labs/gear/gr00t-2/) public. Page explicitly says **“Code Available in December.”** No GR00T 2 code/weights/license verified. Existing Isaac-GR00T N1.7 releases are older. |
| **Dream Labs: Dream-1, SDK and hackathon kit** | SDK packages updated October 29 and November 6 ET; kit supports November 7–8 event. Original model post only dates itself October 2026 | Hosted WAM for two-arm robots; zero-shot embodiment transfer, EEF control and hosted fine-tuning; Quest-based demonstration collection | [SDK](https://github.com/dreamlabs-co/dreamlabs) and [kit](https://github.com/dreamlabs-co/physical-ai-hack-2026) Apache-2.0, with component exceptions. **No public Dream-1 weights or local model trainer verified.** [Research post](https://dreamlabs.ai/research/dream-1/) includes limitations and benchmark claims; independent benchmark source could not be accessed. |
| **Human-1M-100K** | Dataset repository created November 4; publicly linked by the November GR00T 2 page | 100,000-hour subset of a larger egocentric human-data collection, with language and head/hand motion | [Dataset card, samples and full-data link](https://huggingface.co/datasets/nvidia/PhysicalAI-Human-1M-100K), CC BY 4.0. The card points to a separate public storage bucket for the full release. Samples/metadata were inspected; the full corpus was not downloaded or counted. These are human observations/motion, not ready-made ALOHA joint-action demonstrations. |
| **GEAR data-collection hardware** | Repository created November 2; November GR00T 2 release links it | DexUMI wearable hand, UMI handheld gripper, active three-axis camera neck | [Designs, bill of materials and guides](https://github.com/NVlabs/GR00T-DataHardwarePrototypes) available. Hardware/docs CC BY 4.0; software primarily Apache-2.0, with documented third-party exceptions. Collection infrastructure, not a policy benchmark. |
| **Cosmos 3 interactive simulation checkpoints** | [Official announcement November 5](https://huggingface.co/blog/nvidia/cosmos-interactive-world-simulation) | Sim-Bimanual predicts observations from actions; Sim-Transfer renders observations from simulator controls; Sim-Auto and Transfer-Auto serve driving | [Weights](https://huggingface.co/collections/nvidia/cosmos3-examples) and [training/inference recipes](https://github.com/NVIDIA/cosmos-framework) available; weights use OpenMDW 1.1. This is a dated release announcement; at least one model repository already existed in August. These checkpoints are environment models, not newly benchmarked RoboTwin action policies. |
| **LeRobot v0.7.0** | [Release November 5, 12:13 PM ET](https://github.com/huggingface/lerobot/releases/tag/v0.7.0) | Cosmos 3, FLUX 3 Action, LaWAM and Long-WAM integrations; G0.5, LingBot-VLA 2.0, DM0.5 and Wall-OSS-0.5 integrations; data streaming; new distributed trainer; interactive language control | [Code and release guide](https://huggingface.co/blog/lerobot/lerobot-release-v070) available. These are **new integrations**, not evidence that each underlying model first launched this fortnight. Models retain their own licenses; G0.5 is noncommercial. |
| **Meituan-Robotics-0 (MR0)** | [Preprint November 5](https://arxiv.org/abs/2611.08577); code/model repositories created that day | Generalist desktop VLA; three-stage training; Rollback correction collection; event memory | [Code](https://github.com/Meituan-Robotics/Meituan-Robotics-0/tree/opensource), [RoboTwin weights](https://huggingface.co/MeituanRobotics/Meituan-Robotics-0-RoboTwin) and other checkpoints available. RoboTwin weights: Apache-2.0. Code LICENSE is MIT text with added upstream-attribution/rebase conditions; GitHub labels it “Other.” [RoboDesk data](https://huggingface.co/datasets/MeituanRobotics/RoboDesk): CC BY-NC-SA 4.0. |
| **Open-HumanWAM / Open-HumanScale** | [First preprint November 4](https://arxiv.org/abs/2611.06817); repository news dates technical-report release November 5 | 50,000-hour egocentric scaling study; HumanWAM-Vanilla and visual-goal-conditioned HumanWAM-Goal | [Repository](https://github.com/DAGroup-PKU/HumanNet/tree/Open-HumanWAM) and actual [Goal RoboTwin weight files](https://huggingface.co/DAGroup-PKU/HumanWAM-Goal-RoboTwin) exist. README says corpus uploading, some code/reproduction work in progress and several checkpoints pending. Dataset is gated; unauthenticated API does not expose its files. Code declares Apache-2.0; inspected weight cards do not specify their own license. Naming differs between abstract, HTML and repository branches. |
| **ToyBlocks** | [Official release November 6](https://huggingface.co/blog/allenai/toyblocks) | Controlled block world for studying generalization, planning, physical execution and learned simulators; automatic task/skill demonstration generation | [Code](https://github.com/allenai/ToyBlocks), Molmo2-4B and ManiFlow task models: Apache-2.0. [Data](https://huggingface.co/datasets/allenai/toyblocks-data): ODC-BY. Authors report 85,336 long-horizon demonstrations, nearly 1,600 hours and 144M frames. Integrated RL training is **explicitly not in the current package**. |
| **DexSuite** | [Preprint October 27](https://arxiv.org/abs/2610.37799); code repository created October 30 | Genesis-based manipulation framework combining arms/hands, teleoperation, tactile sensing and evaluation | [Code](https://github.com/anashoussaini/Dexsuite) MIT; [dataset organization](https://huggingface.co/dexsuite) linked. Paper: 30 tasks, 10 arms, 12 end-effectors, 4,700 successful episodes. Current README: 31 tasks and 11 end-effectors, with explicit compatibility restrictions. Inspect the current compatibility table rather than assuming every theoretical pairing works. |
| **CaPtain** | [Preprint October 29](https://arxiv.org/abs/2610.40799); release news says October, not a precise day | Trains a 4B model to write robot-control code, first supervised then with RL and a simulator-state critic | [Training/harness code](https://github.com/waltstephen/CaPtain) MIT; [model](https://huggingface.co/YijiaFan/CaPtain-4B) Apache-2.0; data linked. Code and model repositories predate October 27, so only the paper date is established inside the window. Extra depth/calibration/contact observations mean its transfer scores are not our policy protocol. |
| **ResMerge** | [Preprint October 30](https://arxiv.org/abs/2610.41663) | Learn small residual corrections with RL, then merge compatible corrections while retaining a frozen base | [Code](https://github.com/zhucoffee/resmerge-official) Apache-2.0 with component-specific exceptions. Released simulation covers ten OpenVLA-OFT/LIBERO task experts; some small-model experiments and sequential-SFT baselines are absent. Not a Cosmos integration. |
| **Movens-Zero / Movens-5M** | [Preprint October 30](https://arxiv.org/abs/2610.41877) | Humanoid motion generation from text, speech, music and video; 5M execution-checked motions and an intent benchmark | [Code, model links and training/evaluation guides](https://github.com/wendell0218/Movens-Zero) public. README says project-specific code/checkpoint licenses are not specified; backbone has a research license. Dataset names differ between project and repo links. Relevant to whole-body control; not an ALOHA manipulation replacement. |

### Dream Labs: what is open, and what its results mean

The organization is [Dream Labs at dreamlabs.ai](https://dreamlabs.ai/), whose [Dream-1 research post](https://dreamlabs.ai/research/dream-1/) identifies its team's earlier work as DreamZero. This is a verified identity, rather than assuming another project with “Dream” in its name is the intended one.

**Product.** Dream-1 is a world-action model offered through a hosted API and a robot playground. It takes a head camera, two wrist cameras, robot state and a task. End-effector pose outputs provide a shared interface for unfamiliar two-arm robots with parallel grippers; the robot's own inverse-kinematics solver converts those poses into joints. Native joint output and hosted fine-tuning are also supported. The company's stated prices are $3 per model-use hour for Dream-1 and $2 for Dream-1-flash, based on 2,250 calls returning 1.6 seconds of actions each. These are published service prices, not a measured total robot operating cost.

**Actual open releases.** The [Python client and control-loop SDK](https://github.com/dreamlabs-co/dreamlabs) are Apache-2.0, with MIT and asset-specific exceptions. [PyPI history](https://pypi.org/pypi/dreamlabs/json) dates version 1.0.23 to October 29 at 4:50 PM ET and version 1.0.37 to November 6 at 7:14 PM ET. The [lighter client](https://pypi.org/pypi/dreamlabs-client/json) has parallel updates, most recently 1.1.3 on November 6 ET. The [Physical AI Hackathon kit](https://github.com/dreamlabs-co/physical-ai-hack-2026), for November 7–8 in San Francisco, records Meta Quest teleoperation into LeRobot v2.1 and submits demonstrations to hosted fine-tuning. It supports YAM and two MakerMods arm families. These are reusable collection/integration tools; they do not contain the Dream-1 model weights or its local training implementation.

**Evidence and limits.** The research page reports zero-shot evaluation on 27 unfamiliar simulated embodiments, with 300 rollouts per embodiment: 150 language-following trials over 50 tasks and 150 unseen-task trials. Its real-robot comparisons use four robots and earlier checkpoints matched against their own YAM reference. Videos are selected examples, not the whole trial population. The authors identify IK quality, obstacle handling, joint limits and infeasible predicted poses as limitations. These tests are not our 50-task RoboTwin evaluation.

**Leaderboard claims remain unverified here.** Dream Labs cites an October 18 RoboDojo-RC Tier 1 evaluation by [Robocurve](https://robocurve.org/) and claims favorable score/cost/speed relative to frontier agents. Robocurve returned HTTP 403 during this survey, so those quantitative comparisons are only the company's summary. RoboDojo-RC is not interchangeable with the standard RoboDojo leaderboard. The October-only announcement date and pre-window benchmark also mean Dream-1's initial launch cannot be securely dated inside October 27–November 10 from the checked sources.

For our project, Dream-1 could eventually be an external-service comparison if a compatible protocol is established. Its open data-collection tooling is useful independently. It is not currently a verified open-weight candidate for our local post-training experiments.

### GR00T 2: what the evidence supports

Sources: [project page](https://research.nvidia.com/labs/gear/gr00t-2/) and [whitepaper](https://research.nvidia.com/labs/gear/gr00t-2/assets/gr00t2_whitepaper.pdf), especially sections 2, 3, A.3, F.4 and K.

**Architecture.** The whitepaper describes a Cosmos-Predict2.5-2B video-transformer backbone with video, robot-state and action tokens; a frozen video encoder compresses the observations. It takes recent camera history, state and language, predicts future video plus an action chunk, and repeats after execution. This brings GR00T closer to the world-action-model approach we are already studying with Cosmos and Flux. Turbo emphasizes compact video representations and onboard execution. Do not infer total deployed model size from the diffusion-transformer parameter count alone.

**Human data.** Reported pretraining mixture: 402,730 hours of egocentric human video, 10,184 hours of UMI data and 23,500 hours of robot teleoperation. The open Human-1M subset is 100,000 hours; it is not the full pretraining corpus.

**Five-demo adaptation.** Reported average success rises from 40% zero-shot to 71% with one demo and 95% with five demos; adaptation takes about 30 minutes per task with LoRA on one RTX 5090. However, the whitepaper says the 50 tasks were **selected among tasks achieving at least 80% after five-demo adaptation**. The 95% headline is therefore conditional on that selection, not an estimate for 50 arbitrary new tasks.

**RL followed by successful-rollout training.** The whitepaper evaluates four contact-rich tasks, with 50 trials per method/task:

| Task | Supervised starting policy | Direct GRPO RL | Then train on successful RL rollouts |
|---|---:|---:|---:|
| Fixed-pipe fittings | 66% | 86% | 88% |
| Finger placement | 58% | 72% | 98% |
| Cable insertion | 76% | 92% | 94% |
| Charger insertion | 40% | 94% | 96% |

GRPO is reinforcement learning that compares rewards among sampled trajectories. The final stage is ordinary supervised self-imitation using its successful trajectories. The report finds the direct-RL gains statistically conclusive for only two of the four tasks at this sample size. These are author-run real-robot experiments, not our measurements, and they do not show that filtering rollouts from an unchanged policy yields the same benefit.

**Recovery.** An external reasoning agent combines the learned policy with code-based recovery skills. It completes 292/300 continuous toy assemblies across three tasks. This measures the combined system, with progress checks and recovery; it should not be compared to a standalone policy's single-attempt success rate. The work provides concrete evidence that explicit recovery can matter, but no released GR00T 2 recovery implementation was verified.

**Our next-model decision.** Wait for actual code/weights and a license before planning a GR00T 2 benchmark. The most reusable immediate idea is the division between collecting successful experience, learning recovery and evaluating the combined system separately.

### LeRobot 0.7: useful changes and migration cost

The [release notes](https://github.com/huggingface/lerobot/releases/tag/v0.7.0) and [announcement](https://huggingface.co/blog/lerobot/lerobot-release-v070) verify:

- Training now uses FSDP2/HSDP and distributed checkpoints; warmup/stable/decay schedules, EMA, mixed precision and fused optimizers are supported.
- Datasets can stream video byte ranges; Lance storage and GPU image augmentation are supported. Published speedups use specific models/hardware and are not predictions for our runs.
- Cosmos Nano/Edge and Flux integration reduce future adapter work. Flux supports inference and LoRA adaptation; new-embodiment training still points to the upstream Flux trainer.
- Remote inference and interactive goal/subtask control replace the old asynchronous-inference module.
- **Compatibility:** old FSDP training states cannot resume directly, Weights & Biases is replaced by Trackio, and default Linux dependencies move to CUDA 13 builds. These affect our established environment and logging requirements. Treat migration as a separate, measured engineering change.

### Two new model comparisons worth retaining

**MR0.** [Paper section 5.1](https://arxiv.org/html/2611.08577v1) describes all 50 RoboTwin tasks, 50 clean plus 500 randomized demonstrations per task, and 100 evaluation episodes per task per setting. The [repository](https://github.com/Meituan-Robotics/Meituan-Robotics-0/tree/opensource) reports 93.88% randomized / 93.22% clean. The result is the standalone VLA, without its agent or RL post-training. That makes it a useful future comparison under a harmonized protocol, but it is not a measured gain over our mix200 or mix500 runs on matched seeds.

**HumanWAM-Goal.** The reported 84.60% randomized / 92.70% clean uses 50 clean downstream demos/task and visual-goal conditioning. [Full text](https://arxiv.org/html/2611.06817v1) adds refined augmentation and a generated-goal pipeline: most training goals come from GPT Image 2.5, reviewed using ChatGPT; evaluation supplies a generated goal per scene. This is additional conditioning and a partly closed upstream generation process. Vanilla reports 68.88% randomized / 92.08% clean. The README also notes a 50-versus-100-trials discrepancy between the released Vanilla evaluation and the report. Neither result is a direct substitute for our protocol.

## New policy-improvement and failure-analysis research

All gains below are author-reported unless stated otherwise. Most papers do not use our model, all 50 RoboTwin tasks and our exact seeds simultaneously.

| Work; first preprint date ET | Main finding | Availability and relevance |
|---|---|---|
| **[Generalist robot policy improvement](https://arxiv.org/abs/2611.06905), Nov 4** | π0.5 can improve jointly across up to 144 MolmoSpaces / 64 LIBERO tasks. Value-based methods generalize better to unseen scenes/objects than filtered supervised training in several comparisons. Three training seeds and confidence intervals. | [MIT repo](https://github.com/lasgroup/generalist-improvement) currently implements filtered SFT and best-of-N selection. README says **OGPO is coming**. The strongest result is not a complete released recipe yet. Strongest new reading for our eventual RL stage. |
| **[GUILD](https://arxiv.org/abs/2611.05368), Nov 3** | Uses a vision-language model to propose demonstrations, then estimates which collected trajectories actually help deployment; feeds this back into collection. Tests simulation and real robots under matched data budgets. | [CoRL project](https://robin-lab.cs.utexas.edu/guild2026/) verified. Its code link is a placeholder; no usable code release verified. Relevant to allocating extra demos to weak tasks/scenes. |
| **[ScheduleGen](https://arxiv.org/abs/2610.39429), Oct 29** | Schedules multi-arm contact skills and collision-free connecting motions together; 28 simulation variants, 79.6% data-generation success and 65.5% learned-policy success. | [Project and paper](https://njawale42.github.io/ScheduleGen/) available; no implementation link verified. Especially relevant to synchronized two-arm demonstrations, though not an existing RoboTwin recovery adapter. |
| **[OopsieRecover](https://arxiv.org/abs/2611.08462), Nov 5** | Converts a handful of corrective demonstrations and explanations into spatial costs that steer a frozen policy; studies transfer to new layouts/scenes. | [Project](https://robin-lab.cs.utexas.edu/oopsierecover) verified; no released code verified. CoffeeMug: 84% after eight corrections on 25 held-out layouts; real-robot result is task progress, not complete-task success. Includes inference-time steering, which we previously set aside. |
| **[VLA-MOPD](https://arxiv.org/abs/2611.12286), Nov 9** | Consolidates task specialists into one flow-matching policy by querying the relevant teacher on student-visited states and intermediate denoising states. Reports RoboTwin mean 73.0%→89.3%. | Paper verified; no code/license verified. Need task count, training mixture and matched protocol before treating its aggregate as comparable. Potential route to consolidating specialized recovery policies. |
| **[RACE](https://arxiv.org/abs/2611.05838), Nov 3, 9:26 PM** | Ranks checkpoints using held-out successful trajectories, penalizing deviations with a calibrated cumulative score. Reports simulation rank correlation 0.83 and 1,800 real trials. | [Project](https://changyeon.site/race) explicitly says code coming soon. Scores are rankings, not predicted success probabilities. Could later screen checkpoints; does not replace full benchmark evaluations. |
| **[Streaming Failure Forecasting](https://arxiv.org/abs/2611.11688), Nov 8** | Updates only a monitor's linear readout from episode success/failure after deployment drift; studies 434 recorded real-robot runs. | CoRL workshop version noted in paper. [Project](https://sff-research.org) failed TLS access; code not verified. Monitoring performance on replayed runs is not a demonstration of successful online recovery. |
| **[Weightsmith / closed-form instruction distillation](https://arxiv.org/abs/2611.11894), Nov 9** | A known-good rewording can be distilled into policy weights; fixes one diagnosed π0.5 instruction-binding failure on held-out RoboLab seeds. | [MIT code and benchmark](https://github.com/adeeb10abbas/weightsmith) verified. Authors explicitly report failures on contact/perception repairs and substantial unchanged-policy rerun variation. Useful diagnostic lesson; not evidence that weight editing fixes our grasp errors. |
| **[Which Hand Was Late?](https://arxiv.org/abs/2610.37533), Oct 27** | Bimanual failure diagnosis may require both arms' signals; some causes cannot be identified from passive observations. VLM answers can fail to expose information present internally. | Paper verified; no code release verified. Supports recording synchronized cameras, state and actions from both arms. Narrow simulated tool-use study. |
| **[SPARK](https://arxiv.org/abs/2611.12102), Nov 9** | Adapts visual foundation representations using temporal abstraction and patch alignment while preserving semantics. | CoRL paper verified; no author code link verified. Representation learning, not a demonstrated Cosmos/Flux recovery recipe. |

The generalist-improvement paper is an especially useful qualification to “just train on our successes.” Its filtered-SFT baseline can improve familiar tasks while damaging performance on novel objects. It compares full-policy versus action-expert-only updates and critic/collection choices. The public README recommends four GPUs with approximately 100 GB memory each for its own implementation; this is not an estimate for our Cosmos experiment. A Cosmos port and its GPU-hour cost remain unmeasured.

### SimuGuard: a concrete benchmark issue to investigate

[Paper, October 30](https://arxiv.org/abs/2610.41857) · [repository](https://github.com/ALEXander22568/simuguard).

The authors trace some RoboTwin object ejections to a PhysX collision-detection threshold that discards valid small triangles. Their example can fail under both a learned policy and the scripted expert. The current repository validates **one adapter, RoboTwin/SAPIEN 3**, and pins SAPIEN 3.0.0b1, which matches this project's installed package version. The broader four-benchmark scope in the abstract should not be read as four validated adapters.

Important limits from the implementation documentation:

- A flagged contact is a candidate for investigation, not proof of a simulator bug.
- Exact replay requires matching machine/build and recorded substep actuation from the episode start. In-contact snapshot restore can drift.
- Monitoring overhead is reported at 7–12 ms per substep, compared with roughly 0.5 ms physics; it is not free.
- README and mechanism-document numerical claims differ. This survey does not promote their strongest patch-success headline as independently established.
- No project license was verified. No package installation, physics patch or new evaluation was performed here.

For this project, first inspect existing failures for the documented ejection signature and check scripted-expert behavior on the same scene. Any patched-engine result must be labeled as a different evaluation environment and must not be silently mixed into our existing benchmark. We have not established that our knocked-over objects are caused by this bug.

## Official leaderboard snapshot, checked November 10

### RoboTwin 2.0

The [official board](https://robotwin-platform.github.io/leaderboard) and its [public data](https://robotwin-platform.github.io/data/robotwin_leaderboard.json) contain 33 entries and say **last updated October 30**. The stated training protocol is **50 clean demos per task**, with 100 evaluation trials per task in each setting. Its default ranking averages clean and randomized success.

| Entry | Randomized | Clean | Listed | Observation detail |
|---|---:|---:|---|---|
| GO-3 Nano Preview | 81.48% | 88.54% | October 25 | No depth |
| PrismWAM* | 75.50% | 91.84% | October 23 | Uses depth |
| PICA-WAM | 74.70% | 90.68% | **October 30** | No depth |
| CloudRobo-WM | 75.00% | 88.90% | October 19 | No depth |

These are the first four entries under the default ranking. The dated additions within this survey window are **EWAM on October 29 (72.08% randomized / 82.16% clean)** and **PICA-WAM on October 30**. The retrieved snapshot is unchanged from our November 5 survey. It does not contain the new MR0 or HumanWAM paper results.

MR0 trains on an additional 500 randomized demos per task; HumanWAM-Goal receives generated goal images. Our runs also use randomized training data and a different evaluation count. Comparing these numbers requires retaining those protocol differences. The snapshot establishes current listed entries, but cannot reconstruct every historical rank change.

### RoboChallenge Table30

The [official leaderboard](https://robochallenge.ai/leaderboard) exposes separate [v1](https://robochallenge.ai/api/v1/leaderboard/leaderboard_all.json) and [v2](https://robochallenge.ai/api/v2/leaderboard/leaderboard_table30_v2.json) feeds. Both returned a November 10, **02:00:36 ET** last-modified header. That is a file update time; it does not date each model result. The site sorts by success rate, then by its separate task score.

| v1 entry | Success rate | Task score | Submission type |
|---|---:|---:|---|
| Era0 | 64.33% | 76.34 | Separate model per task |
| DM0 | 62.00% | 72.25 | Separate model per task |
| GigaBrain-0.1 | 51.67% | 68.34 | Separate model per task |
| Spirit-v1.5 | 51.00% | 67.19 | Separate model per task |
| Lira_generalist | 45.00% | 59.83 | One model across tasks |
| π0.5 baseline | 42.67% | 61.84 | Separate model per task |

The v1 feed contains 22 entries. Its code/checkpoint links are empty and public-availability flags are false for all retrieved submissions. Those fields do not establish whether the underlying model families have releases elsewhere.

The v2 feed contains 64 entries. Its top three listed generalist submissions are **LoopWAM (43.33%)**, **VLA-DM0.5 (40.67%)** and **ambase (31.67%)**. Their task coverage differs: **25, 30 and 29 tasks**, respectively. Their latest submissions are dated September 24, September 15 and September 29; ambase's latest completion is October 23. These records do not establish a newly achieved top score within October 27–November 10. Different coverage and benchmark versions also prevent pooling the v1/v2 rankings. Row timestamps have no timezone, so only their recorded calendar dates are quoted.

Dream Labs' RoboDojo-RC claim is covered above; the independent Robocurve source was inaccessible. No independent Dream-1 leaderboard ranking was verified.

## New benchmarks and agent-generated skills

- **[RoboARCade](https://arxiv.org/abs/2611.08209), November 5:** tests agentic control on long-horizon tasks across three embodiments in MuJoCo. It separates poor manipulation strategy and imprecise execution from perception errors, and reports frequent false claims of subtask success. Paper says code and traces are public; repository availability/license were not independently verified in this pass.
- **[CRONOS](https://arxiv.org/abs/2611.05080), November 3:** [project](https://embodiedai-ntu.github.io/cronos/) describes 16 tasks across four shared scenes, OpenVLA/SpatialVLA and PPO, with scene resets every 1, 4 or 32 tasks. Removing resets exposes object-placement and arm-state drift. Project code control did not yield a verified repository.
- **[HolidayFactory 1.0 / 24/7-Bench](https://arxiv.org/abs/2611.11789), November 8 ET:** industrial simulation and seven persistent-work tasks. The abstract explicitly says policies have **not yet been compared over persistent runs**. Treat as infrastructure/research proposal, not a demonstrated deployed-policy ranking.
- **[RoboMoCap](https://arxiv.org/abs/2611.10994), November 7:** over 200 hours combining egocentric cameras with 120 Hz optical whole-body motion capture, including locomotion and manipulation. [Project](https://bitrobot-foundation.github.io/robomocap/) linked by authors; dataset download completeness and license not verified. More relevant to humanoid work than RoboTwin arms.
- **[A Controlled Study of Models, Skills, Feedback, and Curriculum in Agentic Robotics](https://arxiv.org/abs/2611.09030), November 6:** nine manipulation tasks, 631 experimental configurations and 28,000 coding-agent rollouts. Authors find interpreting feedback into the right edit is often harder than collecting more feedback. No implementation release verified.
- **[ReSkill](https://arxiv.org/abs/2611.10510), November 7:** edits reusable robot skill programs and retains revisions after paired full-task tests. The abstract separates development-set gains from robustness/model-selection evaluations and real-robot results. This is improvement of code-based skills, not retraining a video-action network.
- **[Flip Anything](https://arxiv.org/abs/2611.11941), November 9:** CoRL late-breaking project in which a coding agent develops flipping strategies in simulation, including handovers, toppling and recovery. [Videos/project](https://berkeleyautomation.github.io/flipanything/) linked. Reported 24.7%→95.0% is the authors' progressive skill-development result; code availability not verified.
- **[PrimeSim](https://arxiv.org/abs/2611.04870), November 3:** action-conditioned multiview simulator trained with 30,110 hours from 22 platforms; authors report 10.1 FPS and policy-RL use. The abstract promises weights/training/inference code, but no actual artifact release was verified. Relevant comparison for Cosmos Sim-Bimanual, not evidence that either can replace physics in our exact benchmark.
- **[Trusting Imagined Futures](https://arxiv.org/abs/2611.04873), November 3:** audits imagined rewards against real execution before choosing supervision. Reports +3 points over the starting data and +1.47 points for audit gating at fixed ranking/margin settings on LIBERO. This is evidence for checking world-model reward errors rather than accepting a plausible generated success as ground truth. No code verified.

ToyBlocks makes a related point experimentally: it deliberately balances successful and failed interaction data when testing learned simulators, then measures both motion and outcome prediction. Its reported transfer result, 19%→78% after adding 100 generated trajectories, is a particular spatial-generalization experiment, not a general expected gain from synthetic data.

## Older work discussed at CoRL or in new integrations

Do not count these as newly launched in this two-week window without separate dated release evidence:

- **π0.7:** the [official Physical Intelligence blog](https://www.pi.website/blog) dates it April 16, 2026; its API beta is October 13. Both are outside the window, even though π0.7 appears in the CoRL program.
- **SimFoundry:** [paper](https://arxiv.org/abs/2606.28276) is from June 2026; its use in GR00T 2 is new context, not a November invention.
- **SimPT:** [paper](https://arxiv.org/abs/2610.29161) first submitted October 21. Directly relevant background: saved simulator progress states, CuRobo recovery and fixed-interval versus progress-graph snapshot selection. Seven separate π0.5 task policies, not our single 50-task Cosmos policy. No released implementation verified.
- **LaWAM, Long-WAM, LingBot-VLA 2.0, FineART, FLUX 3 Action and original Cosmos 3:** their new LeRobot integrations are verified November news; their original reports predate this period.
- **CARE, Q-Planning, Fast-WAM and PatchWAM:** existing project-reading leads; this survey did not establish a new code or model release inside the requested window.

## Suggested reading order and decision points

1. **SimuGuard:** determine whether the described artifact exists in our recorded failures. Preserve the exact benchmark protocol.
2. **GR00T 2 sections A.3 and F.4, plus generalist-improvement sections 3–4:** successful data, recovery and RL; pay particular attention to task selection, exploration and retained performance on other tasks.
3. **MR0 Rollback, GUILD and ScheduleGen:** how to collect useful corrections or coordinated two-arm demonstrations rather than simply accumulating more initial-state demonstrations.
4. **LeRobot 0.7.0:** decide whether its integrations meaningfully reduce maintenance cost; measure compatibility and performance in isolation before a migration.
5. **MR0 and HumanWAM-Goal:** consider future matched-seed benchmarks only after adapter, observation, data and license differences are resolved. The new public numbers do not change our measured ranking by themselves.
6. **ToyBlocks, DexSuite, CaPtain and ResMerge:** reusable infrastructure and alternate methods for follow-on research. Each would broaden scope beyond the current Cosmos/Flux hill climb.

Research conclusions and proposed priorities are inferences from the cited work. Public scores, runtimes, dataset sizes and hardware requirements are authors' measurements or release claims; none was reproduced locally for this survey.

## Source-access and completeness notes

The survey used official conference pages, arXiv abstracts/full text, author project pages, GitHub repository/release metadata, PyPI release timestamps and Hugging Face model/dataset cards. Search snippets were used only to discover sources. Google required JavaScript; DuckDuckGo presented a challenge; neither restriction was bypassed. Some targeted search results were irrelevant. The Streaming Failure Forecasting project failed TLS access, and Robocurve returned HTTP 403. An inability to verify code is reported as such, not as proof that no code exists.

Several source discrepancies materially affect reproducibility: GR00T 2's selected five-demo tasks and three-versus-four-task summaries; HumanWAM/Open-HumanScale naming and partial releases; DexSuite task/hand counts and compatibility; generalist-improvement's pending OGPO implementation; SimuGuard's validated adapter scope and inconsistent numerical summaries. These qualifications should accompany any downstream comparison or experiment plan.
