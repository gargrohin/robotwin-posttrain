"""Per-task success of one checkpoint next to facts about each task's training demos.

Demo facts come from the 2,500 clean demos of the 50task dataset (episode i belongs to
the (i // 50)-th task in sorted order): median length in steps, and whether both arms
move (each arm's 6 joints travel more than 1 radian in total over the episode). Results
come from analysis/cache/episodes.parquet (build_index.py), without spot checks.

usage: analysis/.venv/bin/python analysis/task_report.py [model] [ckpt] [reference model]
  defaults: mix200_150k 150000 50task_60k (the reference is compared scene by scene)
"""

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
DEMOS = ROOT / "data/lerobot/RoboTwin-50task-aloha_agilex-joint"
CACHE = ROOT / "analysis/cache"


def demo_facts():
    cache = CACHE / "demo_facts.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    tasks = sorted(p.name for p in (ROOT / "RoboTwin/data/demo_clean").iterdir())

    def one(i):
        path = DEMOS / f"data/chunk-{i // 1000:03d}/episode_{i:06d}.parquet"
        s = np.stack(pq.read_table(path, columns=["observation.state"]).column(0).to_pylist())
        d = np.abs(np.diff(s, axis=0))  # state: left 6 joints + gripper, right 6 + gripper
        return {"task": tasks[i // 50], "length": len(s),
                "both_arms": d[:, 0:6].sum() > 1.0 and d[:, 7:13].sum() > 1.0}

    with ThreadPoolExecutor(32) as pool:
        df = pd.DataFrame(pool.map(one, range(50 * len(tasks))))
    df.to_parquet(cache)
    return df


def main():
    model, ckpt, ref = (sys.argv[1:] + ["mix200_150k", "150000", "50task_60k"][len(sys.argv[1:]):])[:3]
    pd.set_option("display.width", 200)
    demos = demo_facts()
    tf = demos.groupby("task").agg(demo_len=("length", "median"), both_arm_share=("both_arms", "mean"))
    e = pd.read_parquet(CACHE / "episodes.parquet")
    e = e[~e.spot_check]
    m = e[(e.model == model) & (e.ckpt == int(ckpt))]
    rate = m.pivot_table(index="task", columns="setting", values="success", aggfunc="mean")
    tf = tf.join(rate.rename(columns={"demo_clean": "clean", "demo_randomized": "rand"}), how="inner")
    tf["limit"] = m.groupby("task").step_limit.max()
    tf["succ_steps"] = m[m.success].groupby("task").steps.median()
    tf["arms"] = np.where(tf.both_arm_share >= 0.5, "both arms", "one arm")
    tf["demo_len_bin"] = pd.cut(tf.demo_len, [0, 150, 250, 350, 10_000],
                                labels=["<=150", "151-250", "251-350", ">350"])

    print(f"{model} @ {ckpt}: mean of per-task rates, clean {tf.clean.mean():.1%}, "
          f"randomized {tf.rand.mean():.1%} ({len(tf)} tasks)\n")
    bands = pd.cut(tf.rand, [-0.01, 0.10, 0.40, 0.799, 1.0],
                   labels=["0-10%", "11-40%", "41-79%", "80-100%"])
    for band, g in tf.groupby(bands, observed=True):
        print(f"randomized {band}: {len(g)} tasks: "
              + ", ".join(f"{t} {r:.0%}" for t, r in g.rand.sort_values().items()))
    for col in ["arms", "demo_len_bin"]:
        print("\n" + tf.groupby(col, observed=True).agg(
            tasks=("rand", "size"), randomized=("rand", "mean"), clean=("clean", "mean"),
            demo_len=("demo_len", "median")).round(2).to_string())
    print("\nrandomized rate by demo length and arms (tasks in brackets):")
    mean = pd.crosstab(tf.demo_len_bin, tf.arms, values=tf.rand, aggfunc="mean")
    count = pd.crosstab(tf.demo_len_bin, tf.arms)
    print((mean.map(lambda x: f"{x:.0%}") + count.map(lambda n: f" ({n})")).to_string())
    print(f"\nstep limit / demo length, median over tasks: {(tf.limit / tf.demo_len).median():.2f}")
    print(f"success steps / demo length, median over tasks: {(tf.succ_steps / tf.demo_len).median():.2f}")

    key = ["task", "setting", "seed"]
    j = pd.concat({"ref": e[e.model == ref].drop_duplicates(key).set_index(key).success,
                   "new": m.drop_duplicates(key).set_index(key).success}, axis=1, join="inner")
    for setting, g in j.groupby(level="setting"):
        print(f"\n{setting}, {len(g)} scenes run by both {ref} and {model}: both solve "
              f"{(g.ref & g.new).sum()}, only {model} {(~g.ref & g.new).sum()}, only {ref} "
              f"{(g.ref & ~g.new).sum()}, neither {(~g.ref & ~g.new).sum()}")


if __name__ == "__main__":
    main()
