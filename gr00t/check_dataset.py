"""Check a LeRobot v2.1 dataset: every camera video has exactly one frame per step.

If the check passes, also writes meta/video_frames.json (each video's header frame
count, what ffprobe reports), which our patched GR00T loader reads instead of running
ffprobe on every video in every training rank. On failure it removes that file, since
gr00t/run_mix200.sh takes the file to mean the dataset is ready.

usage: check_dataset.py <dataset_dir> [expected_episodes]
"""

import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import av

root = Path(sys.argv[1])
expected = int(sys.argv[2]) if len(sys.argv) > 2 else None
info = json.loads((root / "meta/info.json").read_text())
episodes = [json.loads(line) for line in (root / "meta/episodes.jsonl").open()]
video_keys = [k for k, f in info["features"].items() if f["dtype"] == "video"]


def check(job):
    ep, key = job
    i = ep["episode_index"]
    rel = str(Path(info["video_path"].format(
        episode_chunk=i // info["chunks_size"], video_key=key, episode_index=i
    )))
    for attempt in range(5):  # EFS can throw a one-off errno 5 right after files are written
        try:
            with av.open(str(root / rel)) as container:
                header_frames = container.streams.video[0].frames
                n = sum(1 for _ in container.decode(video=0))
            break
        except OSError:
            if attempt == 4:
                raise
            time.sleep(2 * (attempt + 1))
    if n != ep["length"]:
        return rel, header_frames, f"episode {i} {key}: {n} frames, {ep['length']} steps"
    return rel, header_frames, None


if __name__ == "__main__":
    jobs = [(ep, key) for ep in episodes for key in video_keys]
    with Pool(32) as pool:
        results = list(pool.imap(check, jobs, chunksize=8))
    bad = [msg for _, _, msg in results if msg]
    frames = {rel: n for rel, n, _ in results}
    frames_path = root / "meta/video_frames.json"

    print(f"{len(episodes)} episodes, {len(video_keys)} cameras, {len(bad)} mismatches")
    for line in bad[:20]:
        print("  " + line)
    if bad or (expected is not None and len(episodes) != expected):
        frames_path.unlink(missing_ok=True)
        sys.exit(f"FAILED (expected {expected} episodes)" if expected else "FAILED")
    frames_path.write_text(json.dumps(frames, indent=0))
