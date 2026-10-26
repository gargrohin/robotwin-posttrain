# cuRobo crashes on H100: a warp-shuffle mask bug in the L-BFGS kernel

Found and fixed 2026-10-26 (investigation by a Claude subagent; evidence in
`outputs/render_check/real_*.log` and the scripts in `setup/`).

## Symptom
Every RoboTwin task died during setup, before any camera image existed:

```
setup_demo → load_robot → CuroboPlanner → MotionGen.warmup()
RuntimeError: CUDA error: an illegal instruction was encountered
```

CUDA errors are reported asynchronously, so the Python line in the traceback is
usually not the culprit. Re-running with `CUDA_LAUNCH_BLOCKING=1` makes every kernel
launch synchronous, and that pointed at `lbfgs_step_cu.forward`
(`curobolib/opt.py:58`): cuRobo's custom CUDA kernel for the L-BFGS optimizer it uses
to solve for trajectories.

## Background: warps and shuffles
- A GPU runs threads in groups of 32 called **warps**. Threads in a warp ("lanes") can
  exchange registers directly with **shuffle** instructions, e.g.
  `__shfl_down_sync(mask, val, k)` returns `val` from the lane `k` positions higher.
  That's the standard way to sum 32 numbers quickly (a "tree reduction": add the value
  16 lanes away, then 8, 4, 2, 1).
- The `mask` says which lanes take part. **Rule:** every lane that executes the
  shuffle must be in the mask, and every lane in the mask must execute it. Breaking
  this is *undefined behaviour*: the compiler and hardware may do anything.
- Since Volta, lanes of a warp can be scheduled independently, so the compiler
  enforces the mask with a `WARPSYNC <mask>` instruction. On Hopper (H100), a lane
  that executes it without being in the mask is a hardware fault: "illegal
  instruction".

## The bug
`reduce_v0`/`reduce_v1` in `src/curobo/curobolib/cpp/lbfgs_step_kernel.cu` sum `m`
values in two stages:
1. Each warp reduces its own 32 values and writes one partial sum to shared memory
   `data[]`. That gives `elems = ceil(m / 32)` partial sums.
2. Warp 0 sums those `elems` partial sums:

```cuda
unsigned mask2 = __ballot_sync(FULL_MASK, threadIdx.x < elems);  // only lanes < elems
if (threadIdx.x / 32 == 0) {                 // ...but ALL 32 lanes of warp 0 enter
  psum_t val2 = data[threadIdx.x % 32];
  for (int i = elems - 1; i > 0; i /= 2) {
    val2 += __shfl_down_sync(mask2, val2, shift);   // lanes >= elems: not in mask2
    shift *= 2;
  }
```

All 32 lanes of warp 0 run the shuffle, but `mask2` only names lanes `< elems`.
That's exactly the undefined behaviour above. nvcc 12.1 compiled it to
`WARPSYNC <reg>` (12,282 of them in the binary), and H100 traps.

**Why it hit every plan, not just warmup:** stage 2 only runs with more than one
warp, i.e. `m > 32`. RoboTwin's planner problem has 168 variables (`d_opt=168`), so
every trajectory solve goes through this path.

**Evidence:**
- `compute-sanitizer --tool memcheck` reported "Illegal instruction" at
  `lbfgs_update_buffer_and_step_v1_compile_m<…,15>+0x480`.
- `--tool synccheck` reported "Barrier error … Invalid arguments".
- It matches RoboTwin issues #452 and #234 and cuRobo #481 ("H100 needs CUDA ≥ 12.6").

## The tempting wrong fix
The GitHub advice is to rebuild with a newer CUDA (we tried nvcc 12.9, no code
change). The crash goes away, **but the answers become wrong**: 6 of 88 test cases
returned wrong L-BFGS steps, with relative error up to 0.98.

Why: once lanes `≥ elems` take part in the shuffle, they add whatever is in
`data[lane]`. Those slots were never written in this call, so they hold stale
values from earlier work. Lane 0 adds values from up to the next power of two above
`elems`, so whenever `elems` isn't a power of two, garbage gets summed in.

**Lesson: "it stopped crashing" ≠ "it's correct".** Always check a numeric kernel
against a reference.

## The fix (`setup/curobo_lbfgs_warp_mask.patch`)
Let all 32 lanes take part (`FULL_MASK`, so no undefined behaviour), and make the
extra lanes contribute zero (so no garbage):

```diff
-        unsigned mask2 = __ballot_sync(FULL_MASK, threadIdx.x < elems);
         if (threadIdx.x / 32 == 0)
         {
           psum_t val2  = data[threadIdx.x % 32];
+          if (threadIdx.x >= elems) val2 = 0;
           ...
-            val2  += __shfl_down_sync(mask2, val2, shift);
+            val2  += __shfl_down_sync(FULL_MASK, val2, shift);
```

The same change goes in both `reduce_v0` and `reduce_v1`. Register usage is unchanged.

## Validation (real env, GPU 6)
| Check | Result |
|---|---|
| `MotionGen.warmup()` (`setup/curobo_warmup_check.py`) | OK, 8.8 s |
| `compute-sanitizer --tool synccheck` | 0 errors |
| CUDA kernel vs cuRobo's PyTorch L-BFGS, 100 cases (`setup/curobo_lbfgs_kernel_check.py`) | max relative error 2.1e-4 |
| `beat_block_hammer` setup + cameras (`setup/robotwin_task_render_check.py`) | OK |
| Scripted expert, seeds 0–67 (`setup/robotwin_expert_check.py`) | 52/68 succeed (PyTorch L-BFGS: 56/68; differences look like numerical noise; identical on rerun) |

## Where it lives / how to undo
- Source: `RoboTwin/envs/curobo`, local branch `h100-fix`, commit `13d961c` (on v0.7.8).
- Build: from `RoboTwin/envs/curobo`, run
  `CUDA_HOME=/usr/local/cuda-12.1 PATH=/usr/local/cuda-12.1/bin:$PATH TORCH_CUDA_ARCH_LIST=9.0 ../../.venv/bin/python setup.py build_ext --inplace`
  (~7 min). The editable install already points at `src/`, so no reinstall.
- Revert: run `git switch --detach v0.7.8`, then
  `cp -p ../../../scratch/curobo_debug/original_build/*.so src/curobo/curobolib/`
  (checksums in `SHA256SUMS`), then `rm -rf build`.
- No-rebuild fallback: `use_cuda_kernel: False` in the 6 L-BFGS yml files under
  `src/curobo/content/configs/task/` uses the PyTorch version (~1.8× slower planning).

## Debugging techniques worth remembering
1. `CUDA_LAUNCH_BLOCKING=1`: find which kernel actually failed.
2. `compute-sanitizer` (`memcheck`, `synccheck`): name the instruction and the kind
   of violation.
3. Differential testing: compare the CUDA kernel against a slow reference
   implementation on many random inputs.
4. Test any "fix" for correctness, not just for the absence of the crash.

## Possible follow-up
Upstream (NVlabs/curobo) still has this code as of v0.7.8. The patch is small and
self-contained; it could be offered as a PR with the 6/88 wrong-result finding.
