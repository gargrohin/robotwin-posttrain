"""Do failures repeat? Two checkpoints on the scenes (task, setting, seed) both ran.

Prints a 2x2 table (A succeeded/failed x B succeeded/failed) per setting and overall, and
per task how many scenes both failed and how many only one failed. Reads
analysis/cache/episodes.parquet (build_index.py), without spot checks. Where one checkpoint
ran a scene twice, the cluster run is kept (its folder path sorts first).

usage: analysis/.venv/bin/python analysis/repeat_check.py <model>:<ckpt> <model>:<ckpt>
  e.g. cosmos3_nano_mix200_2n_decay:5500 cosmos3_nano_mix200_2n_decay:6000
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
KEY = ["task", "setting", "seed"]


def runs(e, spec):
    model, ckpt = spec.rsplit(":", 1)
    d = e[(e.model == model) & (e.ckpt == int(ckpt)) & ~e.spot_check]
    return d.sort_values("folder").drop_duplicates(KEY).set_index(KEY).success


def main():
    a_spec, b_spec = sys.argv[1:3]
    e = pd.read_parquet(ROOT / "analysis/cache/episodes.parquet")
    j = pd.concat({"a": runs(e, a_spec), "b": runs(e, b_spec)}, axis=1, join="inner")
    print(f"A = {a_spec}, B = {b_spec}: {len(j)} scenes run by both\n")
    for name, g in [*j.groupby(level="setting"), ("all", j)]:
        a, b = g.a.astype(bool), g.b.astype(bool)
        print(f"{name}: A succeeded {a.sum()} (B also {(a & b).sum()}, B failed {(a & ~b).sum()}); "
              f"A failed {(~a).sum()} (B succeeded {(~a & b).sum()}, B also failed {(~a & ~b).sum()})")
    a, b = j.a.astype(bool), j.b.astype(bool)
    t = pd.DataFrame({"both failed": ~a & ~b, "only one failed": a ^ b}).groupby(level="task").sum()
    t["scenes"] = j.groupby(level="task").size()
    t = t[(t["both failed"] + t["only one failed"]) > 0].sort_values(
        ["both failed", "only one failed"], ascending=False)
    print("\nper task:\n" + t.to_string())


if __name__ == "__main__":
    main()
