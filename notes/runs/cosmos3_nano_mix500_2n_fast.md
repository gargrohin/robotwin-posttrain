# Run card: `cosmos3_nano_mix500_2n_fast`

Cosmos3-Nano trained on **mix500** with our measured training speedups. Everything else is our best Cosmos
recipe (`cosmos3_nano_mix200_2n` + `cosmos3_nano_mix200_2n_decay1k`), now in one schedule of 6,000
iterations. One change of data, plus the speedups; nothing else.

| | |
|---|---|
| Purpose | does 2.18× the total demo count (randomized 495 instead of 200 per task; clean stays 50) improve on Cosmos mix200 decay 5,500 (85.0% randomized / 88.6% clean)? |
| Base model | `nvidia/Cosmos3-Nano`, the same DCP conversion as the mix200 runs |
| GPUs | 16 H100 on 2 nodes, the training runs' layout: model sharded over a node's 8 GPUs, one copy per node |
| Batch | 32 per GPU × 16 GPUs × 4 accumulation = 2,048 |
| Length | 6,000 iterations ≈ 12.3M examples |
| Speed (expected) | ~6.6 s per iteration from the 1-node benchmark (13.21 s per 8 micro-batches); about 11 h plus saves |
| wandb | project `robotwin-posttrain`, run `nano-mix500-2n-fast`, from iteration 1 |

## Data

| | |
|---|---|
| Dataset | `data/lerobot/RoboTwin-mix500-aloha_agilex-joint` (LeRobot v2.1), built by `gr00t/build_mix500.sh` |
| Content | 50 tasks × (50 clean + randomized episodes 0–494) = **27,250 episodes**, 6,019,158 frames (checked: `meta/info.json`, `meta/episodes.jsonl` and the index built from every episode's parquet all agree) |
| Held out | randomized episodes 495–499 of every task are not in the dataset (the build leaves them out for the memorization check) |
| Normalizer stats | `robotwin_mix500_aloha_joint_pos_stats.json` (sha256 `8282865a880f61fc…`), exact over all frames, same tool and method as mix200 (quantile q01/q99, no clipping) |
| Stats vs mix200 | q01/q99 within 0.036/0.014 per dim; mean/std within 0.009; only the extreme min/max differ more (more data, longer tails) |

## Code

cosmos-framework branch `mix500-fast` @ `10ebb73` = our `robotwin` branch (30bd31a) + the speed-investigation
branches:
- the speedup options (`EncodeSpeedups`, `freeze_und`, `skip_last_n_layers`), all opt-in;
- the timing callback (not enabled in this run);
- the mix500 stats file;
- `ExtraSaves` (one extra save at iteration 100 for the resume test).

## Recipe (unchanged from the best run except data and speedups)

| | Value |
|---|---|
| Training | full fine-tune of the generation expert + action heads (≈ 7B of 15.2B), EMA on |
| Optimizer | fused AdamW, betas (0.9, 0.99), weight decay 0.05, grad clip 1.0, 5× LR on the action heads |
| Actions, cameras, timing | 14-D absolute joint positions, 16-action chunks, head + both wrists (256 tier), 15 fps, as mix200 |
| Loss | 10× vision + 10× action flow matching, as before |
| Seed / data order | 42; episode order seeded 42 (a resume reseeds it as before) |

### Learning-rate schedule, in one run

The earlier mix200 model completed 5,000 updates plus a 1,000-update continuation, reaching 6,000 total.
This run matches that total, batch 2,048 and learning-rate schedule: 12,288,000 sampled training examples.
With 27,250 demos instead of 12,500, it repeats the larger dataset less. Whether mix500 benefits from more
updates remains untested. The original configuration's 16,000 is the learning-rate cycle length; its
initial training limit was 5,000.

Two LambdaLinear cycles, `cycle_lengths=[5000,1000]`, `warm_up_steps=[500,0]`, `f_start=[1e-6, F]`,
`f_max=[1.0, F]`, `f_min=[F, 0.0]`, with F = (16,000 − 5,000) / (16,000 − 500) = 0.70968.

Checked on CPU with the trainer's own scheduler builder:
- updates 1–5,000 match the original 16,000-cycle schedule exactly (max difference 6.8e-21);
- updates 5,001–6,000 decrease linearly to 0, the same values as the decay run.

| update | learning rate (action heads ×5) |
|---|---|
| 1 (iteration 0) | 5.0e-11 |
| 500 | 4.99e-5 |
| 501 | 5.0e-5 |
| 5,000 | 3.5487e-5 |
| 5,001 | 3.5484e-5 |
| 5,500 | 1.7777e-5 |
| 6,000 (last) | 3.5e-8 (0 after it) |

### Speedups (measured on 1 node, investigations section 10)

| flag | effect | exactness |
|---|---|---|
| encoder CUDA graph | −15.6% | bitwise identical latents and losses |
| encoder overlap (with the graph) | a further −5% | bitwise identical |
| `freeze_und` (no backward through the frozen text half) | a further −12% | same math; rounding-level differences under compile (bitwise with compile off) |
| no recomputation in the last 20 layers | a further −6%, +22 GB | same math; rounding level under compile |
| together | 20.0 → 13.2 s per 1-node update (1.52×) | |

## Saves

- Full training state every 250 iterations, written alternately into two fixed overwrite slots. The newest
  complete one is what a resume reads; cluster jobs can overwrite but not delete.
- Permanent full copies of iterations 100 (for the resume test), 4,000, 5,000, 5,500 and 6,000 (for the evals).
- One extra save at iteration 100, which leaves the 250-iteration schedule unchanged.

## Resume test (separate, short)
Save/resume/export with the speedup flags was untested before this run. A separate short job on the shared pool
resumes the iteration-100 copy with the same flags and trains ~20 updates. It writes nothing into this run. It
checks:
- iteration and learning-rate continuity, and that EMA and optimizer state were loaded;
- loss level against the main run's iterations 101–120;
- recompiles and the encoder graph after a resume.

## Eval plan
Full 50-task evals at 4,000, 5,000, 5,500 and 6,000, on GR00T's exact seeds (the protocol of every Cosmos eval:
randomized all seeds, clean first 10 per task), triggered as the saves land.
- The eval code needs the mix500 stats file next to the run's config.

## Log
- 2026-11-08/09: data checked; stats and index built; data uploaded for the cluster (109,008 files, 81.8 GB, counts and bytes equal to the local copy); config pre-flight on CPU (the
  only differences from the mix200 config are the intended ones: data, stats, schedule, iterations, save interval,
  2-node layout, speedup flags); launcher save-slot and multi-node logic tested locally.
- 2026-11-08 22:24 ET: two copies of the same run (same name, config and save location) submitted at once: one to
  our team quota (priority high, no borrowing), one to the shared borrowed pool (priority high). Rule: the first to
  start runs and the other is cancelled while still queued, so only one copy ever trains (one checkpoint set).
- 22:40 ET: the **team-quota copy started**; the shared-pool copy was still queued and was cancelled at once (it never
  ran or wrote anything). The run is on our own quota, so it is not pre-empted by our priority.
- Eval preparation: the eval launcher now ships the mix500 stats file with the eval inputs (the eval image predates
  it); eval queues for 4,000/5,000/5,500/6,000 are the decay run's exact seed lists.
- Measured on the cluster:
  - 6.83 s per iteration (median, iterations 52–210), 6.85 s at ~680, against 10.0 s for the mix200 2-node run at the
    same iterations;
  - full saves 113 s, two uploads of 49 s each;
  - peak GPU memory at 200: 55.4 GB allocated, 64.8 GB reserved, 71.2 GB in use (mix200: 34.8 / 42.2 / 47.5), no
    out-of-memory errors.
- 2026-11-08 23:36 ET: eval path check on the iteration-100 copy passed (stats file loaded, 9/9 test episodes ran).
- 2026-11-09 ~02:20 ET: the job was cancelled at iteration ~1,748 (no reason given; our tools did not cancel it). The
  last full state is 1,500.
- 2026-11-09: resubmitted with the user's approval to our quota and the borrowed pool. At 11:34 ET both copies
  remain queued, with the admission watcher alive. The cloud pointers and completion markers confirm the latest
  complete state is still 1,500. There is no measured admission time or revised finish time yet.
- 2026-11-09 11:40 ET: detached milestone watcher started after guard checks and a live read-only check. It
  submits all 12 borrowed eval slices per complete milestone (4,000 / 5,000 / 5,500 / 6,000), using the exact
  reference inputs. Submission ambiguity stops the watcher for inspection. It does not collect results.
- User clarified that one duplicate may be cancelled even if both training copies start between watcher checks.

- 2026-11-09 20:16 ET: the replacement quota copy started on two nodes; its borrowed duplicate was cancelled
  while still queued. The run continued from the iteration-1,500 save without changing the layout.
- 2026-11-10 05:27 ET: the cloud pointers and completion marker reached iteration 6,000. Completion was first
  observed by the status logger at 05:31 ET. The proposed one-node continuation is no longer needed.
- The separate resume test failed before container startup: the GPU allocator reported a device requiring
  reset. This was not its expected no-save exit after iteration 120; the dedicated resume check remains open.
- The eval watcher stopped on a service HTTP 503 at Nov 9 23:04 ET, before any milestone eval was submitted.
  All four permanent model metadata objects were checked on Nov 10; recovery submission started at 12:49 ET.
  Read failures now retry up to five attempts; ambiguous submission outcomes still stop. Scores remain pending.
- 2026-11-10 12:52 ET: the recovery build stopped before submitting its first eval job: the image registry
  returned HTTP 403 for the Docker base image. No milestone eval has been launched. Resolve registry access,
  then recheck live jobs and clear the verified pre-submission failure from watcher bookkeeping before retry.
- 2026-11-10 12:59 ET: refreshed the registry token with the existing AWS credentials and verified the exact
  base-image manifest. Confirmed no matching eval jobs, preserved the failure log/state, and cleared only the
  failed build's inflight marker under the watcher lock. Detached recovery submissions resumed.
