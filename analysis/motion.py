"""When does the robot stop moving? A cheap signal from each rollout video.

For each episode of one checkpoint: decode the head-camera video in grayscale at 1/4 size,
take the mean absolute change between consecutive frames (0-255 scale), and find the last
frame where it exceeds a threshold. The camera is fixed, so change comes from the arms and
the objects they move. Writes analysis/cache/motion_<model>_<ckpt>.parquet, one row per
episode: last_move (fraction of the episode at the last movement) and still_share (fraction
of frames without movement).

usage: analysis/.venv/bin/python analysis/motion.py <model> <ckpt> [threshold]
"""

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import av
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "analysis/cache"


def motion(path):
    with av.open(str(ROOT / path)) as c:
        frames = [f.to_ndarray(format="gray")[::4, ::4].astype(np.float32)
                  for f in c.decode(video=0)]
    return np.abs(np.diff(np.stack(frames), axis=0)).mean(axis=(1, 2))


def main():
    model, ckpt = sys.argv[1], int(sys.argv[2])
    threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
    e = pd.read_parquet(CACHE / "episodes.parquet")
    e = e[(e.model == model) & (e.ckpt == ckpt) & e.video.notna()].reset_index(drop=True)
    with ThreadPoolExecutor(32) as pool:
        curves = list(pool.map(motion, e.video))
    moving = [np.convolve(m, np.ones(5) / 5, mode="same") > threshold for m in curves]
    e["last_move"] = [(np.nonzero(mv)[0].max() + 1) / len(mv) if mv.any() else 0.0 for mv in moving]
    e["still_share"] = [1 - mv.mean() for mv in moving]
    e["motion_curve"] = curves
    out = CACHE / f"motion_{model}_{ckpt}.parquet"
    e[["task", "setting", "seed", "success", "steps", "step_limit", "video", "last_move",
       "still_share", "motion_curve"]].to_parquet(out)
    print(f"{len(e)} episodes -> {out}")


if __name__ == "__main__":
    main()
