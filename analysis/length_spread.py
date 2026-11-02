"""Task length (steps in the training demos) vs success, with the spread of demo length per task.

Lengths come from the mix200 dataset's meta/episodes.jsonl: episodes 0-2499 are the clean
demos (tasks in sorted order, 50 each), then 4 groups of 2,500 randomized demos (same
order, 50 per task each), i.e. 50 clean + 200 randomized demos per task. Success comes from
analysis/cache/episodes.parquet (build_index.py). Prints markdown for notes/eval_analysis.md.

usage: analysis/.venv/bin/python analysis/length_spread.py [model] [ckpt]   (default mix200_150k 150000)
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EDGES = [0, 150, 250, 350, 10_000]
LABELS = ["150 or fewer", "151-250", "251-350", "over 350"]


def main():
    model, ckpt = (sys.argv[1:] + ["mix200_150k", "150000"][len(sys.argv[1:]):])[:2]
    tasks = sorted(p.name for p in (ROOT / "RoboTwin/data/demo_clean").iterdir())
    lines = (ROOT / "data/lerobot/RoboTwin-mix200-aloha_agilex-joint/meta/episodes.jsonl").open()
    d = pd.DataFrame([{"task": tasks[i % 2500 // 50], "kind": "clean" if i < 2500 else "randomized",
                       "length": json.loads(line)["length"]} for i, line in enumerate(lines)])
    assert (d.groupby(["task", "kind"]).size().unstack() == [50, 200]).all().all()
    g = d.groupby(["task", "kind"]).length
    s = pd.DataFrame({"median": g.median(), "p25": g.quantile(0.25), "p75": g.quantile(0.75),
                      "min": g.min(), "max": g.max(), "cv": g.std(ddof=0) / g.mean(),
                      "mean": g.mean()}).unstack("kind")
    e = pd.read_parquet(ROOT / "analysis/cache/episodes.parquet")
    m = e[(e.model == model) & (e.ckpt == int(ckpt)) & ~e.spot_check]
    rate = m.pivot_table(index="task", columns="setting", values="success", aggfunc="mean")
    t = pd.DataFrame({"med": s[("median", "clean")], "rmed": s[("median", "randomized")],
                      "cv": s[("cv", "clean")], "rand": rate.demo_randomized,
                      "clean": rate.demo_clean}).dropna()
    t["group"] = pd.cut(t.med, EDGES, labels=LABELS)
    t["rgroup"] = pd.cut(t.rmed, EDGES, labels=LABELS)

    print(f"### Groups ({model} @ {ckpt})\n")
    print("| demo length (median steps) | tasks | randomized success | clean success | "
          "randomized success, lowest to highest task |")
    print("|---|---|---|---|---|")
    for label, x in t.groupby("group", observed=True):
        lo, hi = x.rand.idxmin(), x.rand.idxmax()
        print(f"| {label} | {len(x)} | {x.rand.mean():.0%} | {x.clean.mean():.0%} | "
              f"{x.rand.min():.0%} ({lo}) to {x.rand.max():.0%} ({hi}) |")

    cv_c, cv_r = s[("cv", "clean")], s[("cv", "randomized")]
    wide = cv_c[cv_c >= 0.10].sort_values(ascending=False)
    print("\n### Spread of demo length within each task\n")
    print(f"- Typical spread: standard deviation / mean, median over the 50 tasks: "
          f"{cv_c.median():.1%} (clean demos), {cv_r.median():.1%} (randomized demos).")
    print(f"- Tasks with spread of 10% or more (clean demos): {len(wide)}:")
    for task in wide.index:
        print(f"  - {task}: {cv_c[task]:.0%}; lengths {s.loc[task, ('min', 'clean')]:.0f} to "
              f"{s.loc[task, ('max', 'clean')]:.0f}, middle half {s.loc[task, ('p25', 'clean')]:.0f} "
              f"to {s.loc[task, ('p75', 'clean')]:.0f}, median {s.loc[task, ('median', 'clean')]:.0f}")
    moved = t.index[t.group != t.rgroup]
    print(f"- Tasks that change group if the randomized demos' median is used instead: "
          f"{len(moved)} ({', '.join(f'{x}: {t.med[x]:.1f} vs {t.rmed[x]:.1f}' for x in moved)})")
    for name, col in [("median of clean demos", t.med), ("median of randomized demos", t.rmed),
                      ("mean of clean demos", s[("mean", "clean")].reindex(t.index))]:
        print(f"- Rank correlation of task length ({name}) with randomized success: "
              f"{pd.concat([col, t.rand], axis=1).corr('spearman').iloc[0, 1]:.2f}")

    print("\n### Every task\n")
    print("| task | median | middle half | min to max | spread | randomized | clean |")
    print("|---|---|---|---|---|---|---|")
    for task, r in t.sort_values("med").iterrows():
        print(f"| {task} | {r.med:g} | {s.loc[task, ('p25', 'clean')]:g} to "
              f"{s.loc[task, ('p75', 'clean')]:g} | {s.loc[task, ('min', 'clean')]:g} to "
              f"{s.loc[task, ('max', 'clean')]:g} | {r.cv:.0%} | {r.rand:.0%} | {r.clean:.0%} |")


if __name__ == "__main__":
    main()
