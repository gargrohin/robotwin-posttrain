"""Write <run folder>/run_info.json: where a training run started from, its data, settings,
code versions and launch time, so every checkpoint folder says where it came from (the
registry of all runs is notes/runs/README.md).

usage: write_run_info.py <run folder> key=value [key=value ...]
"""

import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def commit(repo):
    return subprocess.run(["git", "-C", str(ROOT / repo), "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True).stdout.strip()


out = Path(sys.argv[1]) / "run_info.json"
now = dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
if out.exists():  # a resume: keep the original record, note the relaunch
    info = json.loads(out.read_text())
    info.setdefault("relaunched", []).append(now)
else:
    info = dict(arg.split("=", 1) for arg in sys.argv[2:])
    info["commits"] = {"robotwin-posttrain": commit("."), "XPolicyLab": commit("RoboTwin/XPolicyLab"),
                       "RoboTwin": commit("RoboTwin")}
    info["launched"] = now
    info["registry"] = "notes/runs/README.md"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(info, indent=1) + "\n")
