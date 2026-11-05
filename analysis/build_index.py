"""Build the episode index for the eval explorer. Read-only on logs and videos.

Pairs every eval client log (logs/**/eval_*.client.log) with its rollout-video folder
and writes analysis/cache/episodes.parquet (one row per episode) and
analysis/cache/streams.parquet (one row per eval stream, i.e. one client process).

How the pairing works (verified against the RoboTwin eval client):
- A finished stream's log ends with "Data has been saved to <folder>/_result.txt". The
  folder is under RoboTwin/eval_result, or, for evals run on the cluster, under
  ../cluster_eval*/results/<job>/eval_result (logs copied into logs/, videos left there).
- A stream that was stopped early has no such line. Its folder is found among the
  unclaimed folders by the episode fingerprint below plus modification times (the
  newest video and the log are both last written when the stream stops).
- Episode N of a stream is video episode<N>.mp4 (N counts only episodes that ran,
  so skipped seeds don't use up a number). Each action step writes one frame, and a
  success writes one extra final frame, so frames == steps (+1 on success). Every
  pairing is checked this way (column frames_ok).
- Evals run after 2026-10-30 also write episode<N>.json next to each video (RoboTwin
  patch 0001) with the instruction and checkpoint number. They fill the instruction
  column, are cross-checked against the log (record_ok), and stand in for the log
  when it is missing (e.g. overwritten by a rerun of the same stream).
- spot_check marks the early 10-episode evals in logs/learning_curve/. They reuse
  seeds of the full benchmark, so aggregates should leave them out.

usage: analysis/.venv/bin/python analysis/build_index.py
"""

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import av
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"
LOCAL_RESULTS = ROOT / "RoboTwin/eval_result"
# evals run on the cluster, copied back one folder per job (their logs are copied into logs/)
CLUSTER_RESULTS = sorted(ROOT.parent.glob("cluster_eval*/results/*/eval_result"))
VIDEO_ROOTS = [LOCAL_RESULTS, ROOT / "outputs/preempted_eval", *CLUSTER_RESULTS]
CACHE = ROOT / "analysis/cache"
MAX_MTIME_GAP_S = 15 * 60  # stopped stream: newest video vs log last write

ANSI = re.compile(r"\x1b\[[0-9;]*m")
NAME = re.compile(
    r"eval_(?P<model_full>.+?)_ckpt-(?P<ckpt>[^_]+)_(?P<task>.+)_"
    r"(?P<setting>demo_clean|demo_randomized)(?:_seed(?P<shard>\d+))?\.client\.log$"
)
EVENTS = re.compile(
    r"step: (?P<step>\d+) / (?P<limit>\d+)"
    r"|(?P<outcome>Success!|Fail!)"
    r"|(?P<crash>Policy rollout error)"
    r"|(?P<not_counted>env error before first action)"
    r"|Success rate: (?P<suc>\d+)/(?P<n>\d+) => [^,]*, current seed: (?P<seed>\d+)"
    r"|Data has been saved to (?P<saved>\S+)/_result\.txt"
    r"|Final success rate: (?P<final_suc>\d+)/(?P<final_n>\d+)"
    r"|Eval Instruction: (?P<instruction_type>\w+)"
)
VIDEO = re.compile(r"episode(\d+)\.mp4$")
RECORD = re.compile(r"episode(\d+)\.json$")
FOLDER_TS = re.compile(r"\d{4}-\d\d-\d\d_\d\d-\d\d-\d\d_\d{6}$")


def short_model(model_full):
    return re.sub(r"^RoboTwin-|-aloha_agilex-joint-\d+$", "", model_full)


def parse_log(path):
    """Stream metadata plus one dict per finished episode."""
    meta = NAME.search(path.name).groupdict()
    text = ANSI.sub("", path.read_text(errors="replace"))
    episodes, step, limit, outcome, crashed, suc_before = [], 0, None, None, False, 0
    stream = {"saved": None, "final_suc": None, "final_n": None, "instruction_type": None}
    for m in EVENTS.finditer(text):
        g = m.lastgroup
        if g in ("step", "limit"):
            step, limit = int(m["step"]), int(m["limit"])
        elif g == "outcome":
            outcome = m["outcome"]
        elif g == "crash":
            crashed = True
        elif g == "not_counted":
            step, outcome, crashed = 0, None, False
        elif g in ("suc", "n", "seed"):
            suc = int(m["suc"])
            episodes.append({
                "episode": int(m["n"]) - 1,
                "seed": int(m["seed"]),
                # the running counter is authoritative; outcome line is a cross-check
                "success": suc > suc_before,
                "outcome_line": outcome,
                "steps": step,
                "step_limit": limit,
                "sim_error": crashed,
            })
            step, outcome, crashed, suc_before = 0, None, False, suc
        elif g in ("final_suc", "final_n"):
            stream["final_suc"], stream["final_n"] = int(m["final_suc"]), int(m["final_n"])
        else:
            stream[g] = m[g]
    st = path.stat()
    stream.update(
        log=str(path.relative_to(ROOT)),
        model_full=meta["model_full"],
        model=short_model(meta["model_full"]),
        ckpt=None if meta["ckpt"] == "last" else int(meta["ckpt"]),
        task=meta["task"],
        setting=meta["setting"],
        shard=int(meta["shard"] or 0),
        n_episodes=len(episodes),
        n_success=sum(e["success"] for e in episodes),
        in_progress_steps=step,  # steps of an episode cut off by a stop, else 0
        finished=stream["final_n"] is not None,
        log_mtime=st.st_mtime,
        spot_check=path.parent.name == "learning_curve",
    )
    return stream, episodes


def scan_folders():
    """Every folder holding rollout videos, with per-video size and mtime."""
    folders = {}
    for root in VIDEO_ROOTS:
        for dirpath, _, filenames in os.walk(root):
            d = Path(dirpath)
            if not FOLDER_TS.search(d.name):
                continue
            rel = Path(os.path.relpath(d, ROOT))  # ../cluster_eval/... for cluster copies
            parts = d.relative_to(root).parts  # <task>/<policy>/<setting>/<model>/<ts>
            known = root.name == "eval_result" and len(parts) == 5
            videos, records = {}, {}
            for f in filenames:
                if v := VIDEO.match(f):
                    st = (d / f).stat()
                    videos[int(v[1])] = (str(rel / f), st.st_size, st.st_mtime)
                elif r := RECORD.match(f):
                    records[int(r[1])] = d / f
            folders[str(rel)] = {
                "folder": str(rel),
                "task": parts[0] if known else None,
                "setting": parts[2] if known else None,
                "model_full": parts[3] if known else None,
                "started": datetime.strptime(d.name, "%Y-%m-%d_%H-%M-%S_%f")
                .replace(tzinfo=timezone.utc),
                "has_result": "_result.txt" in filenames,
                "videos": videos,
                "records": records,
                # how the eval client's log names this folder ("Data has been saved to ...")
                "saved_as": str(Path("eval_result") / d.relative_to(root)) if known else None,
                "local": root == LOCAL_RESULTS,
            }
    return folders


def read_records(folders):
    """Replace each folder's episode<N>.json paths by their contents."""
    paths = [p for f in folders.values() for p in f["records"].values()]
    with ThreadPoolExecutor(32) as pool:
        loaded = dict(zip(paths, pool.map(lambda p: json.loads(p.read_text()), paths)))
    for f in folders.values():
        f["records"] = {n: loaded[p] for n, p in f["records"].items()}


def probe_frames(folders):
    """Frame count per video from its header, cached by (path, size, mtime)."""
    cache_file = CACHE / "video_probe.parquet"
    cached = {}
    if cache_file.exists():
        for r in pd.read_parquet(cache_file).itertuples():
            cached[(r.video, r.size, r.mtime)] = r.frames
    keys = [v for f in folders.values() for v in f["videos"].values()]
    todo = [k for k in keys if k not in cached]

    def probe(key):
        try:
            with av.open(str(ROOT / key[0])) as c:
                return key, c.streams.video[0].frames
        except Exception:  # e.g. a stopped stream's half-written mp4
            return key, -1

    with ThreadPoolExecutor(32) as pool:
        cached.update(pool.map(probe, todo))
    rows = [{"video": k[0], "size": k[1], "mtime": k[2], "frames": cached[k]} for k in keys]
    pd.DataFrame(rows).to_parquet(cache_file, index=False)
    print(f"probed {len(todo)} new videos ({len(keys)} total)")
    return {k[0]: cached[k] for k in keys}


def fingerprint_ok(ep, frames):
    if frames is None or frames < 0:
        return False
    if ep["sim_error"]:  # crash can land before or after the frame write
        return frames in (ep["steps"], ep["steps"] + 1)
    return frames == ep["steps"] + int(ep["success"])


def match_score(stream, episodes, folder, frames):
    """Fraction of the log's episodes whose video fingerprint matches, or None."""
    for k in ("task", "setting", "model_full"):
        if folder[k] is not None and folder[k] != stream[k]:
            return None
    vids = folder["videos"]
    if not vids or len(vids) not in (len(episodes), len(episodes) + 1):
        return None
    newest = max(v[2] for v in vids.values())
    if abs(newest - stream["log_mtime"]) > MAX_MTIME_GAP_S:
        return None
    if not episodes:
        return 0.0
    hits = sum(fingerprint_ok(e, frames.get(vids[e["episode"]][0])) for e in episodes
               if e["episode"] in vids)
    return hits / len(episodes)


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    logs = sorted(LOGS.rglob("eval_*.client.log"))
    parsed = [parse_log(p) for p in logs if NAME.search(p.name)]
    folders = scan_folders()
    read_records(folders)
    frames = probe_frames(folders)

    by_saved = {}  # log's name for a folder -> folders with that name, this machine's first
    for f in folders.values():
        if f["saved_as"]:
            by_saved.setdefault(f["saved_as"], []).append(f["folder"])
    claimed = {}  # folder -> log
    for stream, _ in parsed:
        if stream["saved"] and by_saved.get(stream["saved"]):
            folder = by_saved[stream["saved"]][0]
            stream["folder"], stream["match"] = folder, "saved_path"
            claimed[folder] = stream["log"]
    same_name = sum(len(v) > 1 for v in by_saved.values())
    if same_name:
        print(f"{same_name} folder names exist in more than one place; used the first")
    for stream, episodes in parsed:
        if stream.get("folder"):
            continue
        scored = []
        for f in folders.values():
            if f["folder"] in claimed:
                continue
            s = match_score(stream, episodes, f, frames)
            if s is not None:
                scored.append((s, f["folder"]))
        scored.sort(reverse=True)
        # accept only a clear winner: full fingerprint match, unique
        if scored and scored[0][0] == 1.0 and (len(scored) == 1 or scored[1][0] < 1.0):
            stream["folder"], stream["match"] = scored[0][1], "fingerprint"
            claimed[scored[0][1]] = stream["log"]
        else:
            stream["folder"] = None
            stream["match"] = f"none ({len(scored)} candidates)" if scored else "none"

    for f in folders.values():  # no log, but episode records: index from those
        # this machine's results only: the cluster job folders include test and retry jobs,
        # so there only folders a copied log names are used
        if f["folder"] in claimed or not f["records"] or not f["model_full"] or not f["local"]:
            continue
        recs = [f["records"][n] for n in sorted(f["records"])]
        ckpt = recs[0].get("checkpoint_num")
        episodes = [{"episode": r["episode"], "seed": r["seed"], "success": r["success"],
                     "outcome_line": None, "steps": r["steps"], "step_limit": r["step_limit"],
                     "sim_error": r["sim_error"]} for r in recs]
        parsed.append(({
            "log": None, "model_full": f["model_full"], "model": short_model(f["model_full"]),
            "ckpt": None if ckpt in (None, "last") else int(ckpt), "task": f["task"],
            "setting": f["setting"], "shard": min(r["seed"] for r in recs) // 100000 - 1,
            "instruction_type": recs[0].get("instruction_type"), "folder": f["folder"],
            "finished": f["has_result"], "spot_check": False, "match": "records only",
            "n_episodes": len(episodes), "n_success": sum(e["success"] for e in episodes),
        }, episodes))
        claimed[f["folder"]] = None

    ep_rows, stream_rows = [], []
    for stream, episodes in parsed:
        folder = folders.get(stream["folder"]) if stream["folder"] else None
        stream["started"] = folder["started"] if folder else None
        stream["n_videos"] = len(folder["videos"]) if folder else 0
        stream_rows.append(stream)
        for e in episodes:
            video = folder["videos"].get(e["episode"]) if folder else None
            rec = folder["records"].get(e["episode"]) if folder else None
            n = frames.get(video[0]) if video else None
            ep_rows.append({
                **{k: stream[k] for k in ("model", "ckpt", "task", "setting", "shard",
                                          "model_full", "instruction_type", "log", "folder",
                                          "started", "finished", "spot_check")},
                **e,
                "timed_out": not e["success"] and e["steps"] == e["step_limit"],
                "instruction": rec["instruction"] if rec else None,
                "record_ok": all(rec[k] == e[k] for k in ("seed", "success", "steps"))
                if rec and stream["log"] else None,
                "video": video[0] if video else None,
                "video_frames": n,
                "frames_ok": fingerprint_ok(e, n) if video else None,
            })
    for f in folders.values():  # folders no log claims
        if f["folder"] not in claimed:
            stream_rows.append({
                "folder": f["folder"], "task": f["task"], "setting": f["setting"],
                "model_full": f["model_full"],
                "model": short_model(f["model_full"]) if f["model_full"] else None,
                "started": f["started"], "n_videos": len(f["videos"]), "match": "no log",
            })

    episodes_df = pd.DataFrame(ep_rows)
    streams_df = pd.DataFrame(stream_rows)
    for df in (episodes_df, streams_df):
        df["ckpt"] = df["ckpt"].astype("Int64")
    episodes_df.to_parquet(CACHE / "episodes.parquet", index=False)
    streams_df.to_parquet(CACHE / "streams.parquet", index=False)
    print(f"{len(episodes_df)} episodes from {len(parsed)} streams; "
          f"{len(folders)} video folders, {len(folders) - len(claimed)} without a log")


if __name__ == "__main__":
    main()
