"""Run XPolicyLab's LeRobot v3.0 converter with two fixes, without editing it.

1. LeRobot's streaming video encoder waits only 0.1 s on a full queue (default 30
   frames) and then drops the frame, which happened in 146/249 episodes here. A queue
   longer than any episode (max 354 frames) makes drops impossible.
2. EFS sometimes fails a read with errno 5; the converter then skips the episode.
   Retry the HDF5 load instead.

Same arguments as transform_lerobot_v30_format.py.
"""

import functools
import importlib.util
import sys
import time
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset

CONVERTER = (
    Path(__file__).resolve().parents[1]
    / "RoboTwin/XPolicyLab/scripts/transform_lerobot_v30_format.py"
)
ENCODER_QUEUE_MAXSIZE = 1000
LOAD_RETRIES = 5

spec = importlib.util.spec_from_file_location("transform_lerobot_v30_format", CONVERTER)
converter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(converter)

_create = LeRobotDataset.create


@functools.wraps(_create)
def create_with_big_queue(*args, **kwargs):
    kwargs.setdefault("encoder_queue_maxsize", ENCODER_QUEUE_MAXSIZE)
    return _create(*args, **kwargs)


LeRobotDataset.create = create_with_big_queue

_load = converter.load


@functools.wraps(_load)
def load_with_retry(*args, **kwargs):
    for attempt in range(1, LOAD_RETRIES + 1):
        try:
            return _load(*args, **kwargs)
        except OSError as e:
            if attempt == LOAD_RETRIES:
                raise
            print(f"[retry {attempt}/{LOAD_RETRIES}] {args[0]}: {e}", flush=True)
            time.sleep(2 * attempt)


converter.load = load_with_retry

if __name__ == "__main__":
    sys.argv[0] = str(CONVERTER)
    converter.main()
