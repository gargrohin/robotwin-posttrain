"""Which failures ever succeed? A base checkpoint's failed scenes, retried by other runs.

For every scene (task, setting, seed) the base run failed and every retry run also ran, counts
how many retries succeeded (0 = failed every try). Prints the counts per setting and, for
scenes that failed every try, per task. Same episode selection as repeat_check.py.

usage: analysis/.venv/bin/python analysis/retry_split.py <model>:<ckpt> <model>:<ckpt> [...]
  e.g. cosmos3_nano_mix200_2n_decay:5500 cosmos3_nano_mix200_2n_decay_rerun1:5500 \
       cosmos3_nano_mix200_2n_decay:6000
"""

import sys

import pandas as pd

from repeat_check import ROOT, runs


def main():
    base, *retries = sys.argv[1:]
    e = pd.read_parquet(ROOT / "analysis/cache/episodes.parquet")
    b = runs(e, base).astype(bool)
    j = pd.concat({s: runs(e, s) for s in retries}, axis=1, join="inner")
    j = j[j.index.isin(b[~b].index)].astype(bool)
    print(f"base {base}: {len(b)} scenes, {(~b).sum()} failed; {len(j)} of those retried by all of "
          f"{', '.join(retries)}\n")
    n = j.sum(axis=1).rename("retries that succeeded")
    print(pd.crosstab(n, n.index.get_level_values("setting"), margins=True).to_string())
    never = n[n == 0].groupby(level="task").size().sort_values(ascending=False)
    print(f"\nfailed every try, per task ({never.sum()} scenes):\n" + never.to_string())


if __name__ == "__main__":
    main()
