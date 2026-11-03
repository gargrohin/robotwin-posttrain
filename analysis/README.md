# Eval explorer

Browse RoboTwin eval results episode by episode: pick a model, checkpoint, setting and
task, see its success rate, click an episode, and watch (or step through) its rollout
video. Next to it plays a training demo of the same task (clean or randomized,
matching the setting), so you can compare what the model does with what it was taught.
Both players start at 10 steps per second, so they move in step. The view
"two models, same scenes" puts two models' rollouts of the same scene (seed) side by side,
lists the scenes both ran with each one's result, and sorts tasks by the gap between them. CPU only; read-only on `logs/`, `RoboTwin/eval_result/` and the datasets.

## Start

```bash
analysis/run.sh          # serves http://localhost:8611 (another port: analysis/run.sh 8612)
```

Run it in a VS Code terminal (connected with Remote-SSH). VS Code forwards the port and
offers "Open in Browser". If it doesn't, open the **Ports** tab, add `8611`, and click
the globe icon. Any browser plays the rollout videos. If one won't play, open the same
URL in VS Code's own browser (command palette: "Simple Browser: Show"). Ctrl+C stops it.

## After new evals: refresh the index

```bash
analysis/.venv/bin/python analysis/build_index.py   # ~30 s; then reload the page
```

New models and checkpoints show up by themselves; they come from the eval log names
(`logs/**/eval_<model>_ckpt-<step>_<task>_<setting>[_seed<k>].client.log`).

## What the index holds

`analysis/cache/episodes.parquet`, one row per episode: model, checkpoint, task, setting,
shard (parallel stream; its seeds start at 100000·(1+shard)), episode, seed, success,
steps, step limit, sim crash, instruction (evals from 2026-10-30 on), video path and a
frame check. `analysis/cache/streams.parquet` has one row per eval stream, including
streams and video folders that couldn't be paired. How the pairing works and how it is
checked: the docstring of `build_index.py`.

Things to know when reading the numbers:
- Every failure runs to the step limit; RoboTwin never ends a failed episode early.
- "Spot checks" (logs in `logs/learning_curve/`) are early 10-episode evals that reuse
  seeds of the full benchmark. They are left out of the totals unless you tick the box.
- A few old episodes have no video (deleted, or moved mid-run); the episode table says
  "missing".

## Setup (once)

```bash
uv venv analysis/.venv --python 3.12
uv pip install --python analysis/.venv/bin/python pandas pyarrow av streamlit
```
