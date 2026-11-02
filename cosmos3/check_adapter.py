"""CPU check of the Cosmos3_Nano XPolicyLab adapter (no GPU, no checkpoint needed).

A. Training parity. For real windows of RoboTwin-mix200, builds the training sample exactly as
   the run does (get_action_robotwin_sft_dataset with the run's config.yaml settings; caption
   dropout off, idle frames 0) and the adapter's sample from the same camera frames (decoded
   from the training videos), state and instruction. The adapter side goes through the eval
   path: a RoboTwin get_obs()-style dict -> the sim client's robotwin_obs_to_xpolicylab ->
   websocket codec -> the server's decode_obs_images -> adapter. Requires a bit-identical
   conditioning frame, an identical JSON prompt, an identical normalized state row and
   identical metadata (sequence plan, fps, domain, image size, action width).
B. Eval-image parity. Recorded demos (HDF5, the simulator's own renders) through the same eval
   path vs the training sample of the same moment: the conditioning frame may differ only by
   codec noise (HDF5 JPEG vs training AV1); the state row and prompt must be identical.
C. Output decoding, on synthetic model outputs, through the adapter's real get_action with a
   stand-in model: drop the state row, keep 14 of the padded channels, denormalize (q01/q99),
   return the GR00T-style per-step dicts; then through the sim client's normalize_action_chunk
   and xpolicylab_action_to_robotwin, which must yield the original absolute 14-D actions in
   the simulator's order. Also checks the sampling arguments and the batch layout.
D. Settings: normalizer stats file and fingerprint; how often training windows have 0 idle
   frames (the adapter always says 0, as NVIDIA's LIBERO server does).

Run in the Cosmos venv from the repo root (CPU only):
  CUDA_VISIBLE_DEVICES= /opt/dlami/nvme/rohingarg/cosmos3-venv/bin/python cosmos3/check_adapter.py
Writes outputs/cosmos3_adapter_check/{summary.json,conditioning_frame.png}; exits 1 on a failure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = Path("/mnt/efs/rohingarg/cri/cosmos3/runs/cosmos3_action_robotwin/action_sft/action_policy_robotwin_nano")
OUT = ROOT / "outputs/cosmos3_adapter_check"
DEMOS = [("beat_block_hammer", 5), ("handover_block", 15), ("open_laptop", 25)]  # demo_clean, all in mix200
STATS_SHA256_PREFIX = "19d7ef157f837074"  # notes/runs/cosmos3_nano_mix200.md

from XPolicyLab.policy.Cosmos3_Nano import model as adapter  # noqa: E402  (before the client edits sys.path)
from XPolicyLab.utils.process_data import decode_obs_images  # noqa: E402

sys.path.insert(0, str(ROOT / "gr00t"))
import open_loop_check as olc  # noqa: E402  (GR00T check: eval-client loader, codec round trip, demo loader)

failures: list[str] = []


def check(ok: bool, what: str) -> bool:
    print(f"  {'PASS' if ok else 'FAIL'}  {what}")
    if not ok:
        failures.append(what)
    return ok


def sim_obs_from(images: list[np.ndarray], state: np.ndarray) -> dict:
    """What RoboTwin's task_env.get_obs() returns, for these images and joint state."""
    cams = ("head_camera", "left_camera", "right_camera")
    return {
        "observation": {cam: {"rgb": img} for cam, img in zip(cams, images)},
        "joint_action": {
            "left_arm": [float(x) for x in state[0:6]], "left_gripper": float(state[6]),
            "right_arm": [float(x) for x in state[7:13]], "right_gripper": float(state[13]),
            "vector": state.astype(np.float64),
        },
        "endpose": {},
    }


def eval_path_obs(client, sim_obs: dict, instruction: str) -> dict:
    """sim obs -> what the policy server hands Model.update_obs."""
    xpl = client.robotwin_obs_to_xpolicylab(sim_obs, instruction=instruction, env_idx=0)
    return decode_obs_images(olc.ws_roundtrip(xpl))


def caption(sample: dict) -> str:
    c = sample["ai_caption"]
    return json.dumps(c) if isinstance(c, dict) else c


def compare(train: dict, mine: dict, *, exact_frame: bool) -> dict:
    """Training sample vs adapter sample; returns frame-difference stats."""
    tv, mv = train["video"], mine["video"]
    check(tv.shape == mv.shape and tv.dtype == mv.dtype == torch.uint8, f"video {tuple(mv.shape)} uint8, as training")
    diff = (tv[:, 0].int() - mv[:, 0].int()).abs()
    if exact_frame:
        check(int(diff.max()) == 0, "conditioning frame (frame 0) bit-identical to training")
    check(bool((mv[:, 1:] == 0).all()), "frames 1.. black (generated, never encoded)")
    check(torch.equal(train["image_size"], mine["image_size"]), f"image_size {mine['image_size'].tolist()}")
    check(caption(train) == caption(mine), "JSON prompt identical")
    ta, ma = train["action"], mine["action"]
    check(ta.shape == ma.shape, f"action {tuple(ma.shape)} (state row + 16, padded to max_action_dim)")
    check(torch.equal(ta[0], ma[0]), "normalized state row identical (all 64 channels)")
    check(bool((ma[1:] == 0).all()), "rows 1.. zero (replaced by noise when sampling)")
    check(train["sequence_plan"].as_dict() == mine["sequence_plan"].as_dict(),
          f"sequence plan {mine['sequence_plan'].as_dict()}")
    for key in ("conditioning_fps", "domain_id", "raw_action_dim", "idle_frames"):
        check(torch.equal(train[key], mine[key]), f"{key} = {mine[key].item()}")
    for key in ("mode", "viewpoint"):
        check(train[key] == mine[key], f"{key} = {mine[key]}")
    tr, mr = train["action_processing_record"], mine["action_processing_record"]
    check(tr.raw_action_dim == mr.raw_action_dim == 14 and tr.action_normalizer is mr.action_normalizer is None,
          "action record: raw_action_dim 14, no normalizer inside the model (denormalized by the adapter)")
    check(set(train) == set(mine), f"same sample keys ({len(mine)})")
    return {"frame0_mean_abs_diff": float(diff.float().mean()), "frame0_max_abs_diff": int(diff.max())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR)
    parser.add_argument("--idle-windows", type=int, default=5000)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    summary: dict = {"run_dir": str(args.run_dir)}

    from cosmos_framework.data.generator.action.action_normalization import normalize_action
    from cosmos_framework.data.generator.action.datasets.action_sft_dataset import get_action_robotwin_sft_dataset

    settings = adapter.load_run_settings(args.run_dir)
    ds_cfg = settings["dataset"]
    builder = adapter.InputBuilder(ds_cfg)
    layout = adapter.action_layout("aloha_agilex")
    client = olc.load_eval_client()

    # The training dataset exactly as the run builds it, minus caption dropout and tokenization.
    kwargs = {k: v for k, v in ds_cfg.items() if k != "_target_"}
    kwargs.update(tokenizer_config=None, cfg_dropout_rate=0.0, iterable_shuffle=False)
    sft = get_action_robotwin_sft_dataset(**kwargs)
    data = sft._dataset
    true_idle = data._compute_idle_frames
    data._compute_idle_frames = lambda action: 0  # the adapter can't know future idling; it says 0

    def train_sample(pos: int, start: int) -> tuple[dict, int]:
        prev = int(data._valid_cum[pos - 1]) if pos > 0 else 0
        idx = prev + start
        assert data.locate(idx) == (pos, start)
        row = int(data._ep_start[pos]) + start
        raw = np.concatenate([data._state[row:row + 1], data._action[row:row + data._chunk_length]])
        return sft[idx], int(true_idle(torch.from_numpy(raw)))

    print("A. training parity (frames from the training videos, through the eval path)")
    rows_a = []
    n_ep = len(data._ep_index)
    for pos in (0, n_ep // 2, n_ep - 1):
        length = int(data._ep_length[pos])
        for start in (0, (length - data._chunk_length) // 2, length - data._chunk_length - 1):
            train, idle = train_sample(pos, start)
            frames = data._episode_frames(pos)
            images = [f[start].permute(1, 2, 0).numpy() for f in frames]
            state = data._state[int(data._ep_start[pos]) + start]
            obs = eval_path_obs(client, sim_obs_from(images, state), data._instructions[pos])
            print(f" episode {int(data._ep_index[pos])} frame {start}/{length} (true idle frames: {idle})")
            mine = builder.build(*adapter.parse_observation(obs, layout, ""))
            rows_a.append({"episode": int(data._ep_index[pos]), "frame": start, "true_idle_frames": idle,
                           **compare(train, mine, exact_frame=True)})
    summary["A_training_parity"] = rows_a
    summary["prompt_example"] = caption(mine)
    print(f" prompt example: {caption(mine)}")

    print("B. eval-image parity (recorded demos, the simulator's renders, through the eval path)")
    rows_b = []
    sim_cam_of = {xpl: sim for sim, xpl in client.CAMERA_NAME_MAP.items()}
    for task, ep in DEMOS:
        demo = olc.load_demo(olc.ROBOTWIN / f"data/demo_clean/{task}/aloha_agilex/data/episode_{ep:07d}.hdf5")
        state = olc.flat(demo["state"])
        matches = [p for p in np.flatnonzero(data._ep_length == len(state))
                   if np.allclose(data._state[data._ep_start[p]:data._ep_start[p] + len(state)], state, atol=1e-6)]
        if not check(len(matches) == 1, f"{task} demo {ep}: exactly one matching training episode"):
            continue
        pos = int(matches[0])
        for t in (0, len(state) // 2, len(state) - data._chunk_length - 1):
            train, _ = train_sample(pos, t)
            obs = eval_path_obs(client, olc.sim_observation(demo, t, sim_cam_of), data._instructions[pos])
            print(f" {task} demo {ep} = training episode {int(data._ep_index[pos])}, frame {t}")
            mine = builder.build(*adapter.parse_observation(obs, layout, ""))
            stats = compare(train, mine, exact_frame=False)
            check(stats["frame0_mean_abs_diff"] < 4.0,
                  f"conditioning frame within codec noise of training (mean |diff| {stats['frame0_mean_abs_diff']:.2f})")
            rows_b.append({"task": task, "demo": ep, "episode": int(data._ep_index[pos]), "frame": t, **stats})
        if task == DEMOS[0][0]:
            from PIL import Image

            panels = [train["video"][:, 0], mine["video"][:, 0],
                      ((train["video"][:, 0].int() - mine["video"][:, 0].int()).abs() * 5).clamp(0, 255).to(torch.uint8)]
            Image.fromarray(torch.cat(panels, dim=-1).permute(1, 2, 0).numpy()).save(OUT / "conditioning_frame.png")
    summary["B_eval_image_parity"] = rows_b

    print("C. output decoding (synthetic model outputs, adapter get_action with a stand-in model)")
    rng = np.random.default_rng(0)
    q01, q99 = data._load_norm_stats()["q01"].numpy(), data._load_norm_stats()["q99"].numpy()
    target = (q01 + (q99 - q01) * rng.uniform(-0.2, 1.2, size=(16, 14))).astype(np.float32)  # incl. outside q01..q99
    state = target[0] - 0.01
    norm = normalize_action(torch.from_numpy(np.concatenate([state[None], target])), "quantile", data._load_norm_stats())

    class StandIn:
        def __init__(self, rows):
            self.rows, self.calls = rows, []

        def generate_samples_from_batch(self, batch, **kw):
            self.calls.append((batch, kw))
            return {"action": [self.rows], "vision": [None]}

    sim_obs = sim_obs_from([np.full((240, 320, 3), v, np.uint8) for v in (50, 100, 150)], state)
    for width in (14, 64):  # the model returns 14 channels (unpadded by its record); 64 is handled too
        rows = torch.cat([norm, torch.full((17, width - 14), 7.0)], dim=1) if width > 14 else norm
        stand_in = StandIn(rows)
        policy = adapter.Model.__new__(adapter.Model)
        policy._setup({"env_cfg_type": "aloha_agilex", "seed": 3, "task_name": "beat_block_hammer"}, settings, stand_in)
        policy.reset()
        policy.update_obs(eval_path_obs(client, sim_obs, "Pick up the hammer and beat the block."))
        reply = olc.ws_roundtrip({"result": policy.get_action()})["result"]  # what the client receives
        chunk = client.normalize_action_chunk(reply)
        flat = [client.xpolicylab_action_to_robotwin(a, action_type="joint", current_observation=sim_obs) for a in chunk]
        got = np.stack([f for f, _ in flat])
        check(len(chunk) == 16 and all(kind == "qpos" for _, kind in flat), f"[{width} ch] 16 joint (qpos) actions")
        check(set(reply[0]) == {"left_arm_joint_state", "left_ee_joint_state", "right_arm_joint_state",
                                "right_ee_joint_state"} and reply[0]["left_arm_joint_state"].shape == (6,),
              f"[{width} ch] GR00T_N17 per-step dict format")
        err = float(np.abs(got - target).max())
        check(err < 1e-5, f"[{width} ch] sim receives the original absolute actions (max |err| {err:.1e})")
    batch, kw = stand_in.calls[0]
    check(kw["num_steps"] == 30 and kw["guidance"] == 1.0 and kw["shift"] == 5.0 and len(kw["seed"]) == 1,
          f"sampling args {dict(kw, seed=kw['seed'])}")
    check(batch["video"][0][0].shape[:2] == (3, 17) and batch["action"][0][0].shape == (17, 64)
          and isinstance(batch["ai_caption"][0], str) and batch["image_size"][0].shape == (1, 4)
          and batch["domain_id"][0].tolist() == [17] and batch["conditioning_fps"][0].tolist() == [15],
          "batch layout: video [[3,17,H,W]], action [[17,64]], prompt str, image_size [1,4], domain 17, fps 15")
    policy._setup({"env_cfg_type": "aloha_agilex", "seed": 3, "actions_per_call": 8}, settings, StandIn(norm))
    policy.update_obs(eval_path_obs(client, sim_obs, "x"))
    check(len(policy.get_action()) == 8, "actions_per_call=8 returns the first 8")
    seeds = lambda s: np.random.default_rng(s).integers(0, 2**31, size=3).tolist()  # noqa: E731
    summary["C_seed_streams"] = {"shard0": seeds(0), "shard1": seeds(1)}

    print("D. settings")
    stats_path = Path(ds_cfg["action_stats_path"])
    digest = hashlib.sha256(stats_path.read_bytes()).hexdigest()
    check(digest.startswith(STATS_SHA256_PREFIX), f"normalizer stats {stats_path.name} sha256 {digest[:16]} (run card)")
    check(Path(data._stats_file) == stats_path, "training dataset and adapter use the same stats file")
    idx = rng.integers(0, len(data), size=args.idle_windows)
    idle = []
    for i in idx:
        pos, start = data.locate(int(i))
        row = int(data._ep_start[pos]) + start
        raw = np.concatenate([data._state[row:row + 1], data._action[row:row + data._chunk_length]])
        idle.append(int(true_idle(torch.from_numpy(raw))))
    hist = np.bincount(idle, minlength=data._chunk_length + 2)
    summary["D_idle_frames_histogram"] = hist.tolist()
    print(f"  training windows with 0 idle frames: {hist[0] / len(idle):.1%} of {len(idle)} random windows "
          f"(histogram 0..17: {hist.tolist()})")
    summary["failures"] = failures
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"\n{'ALL CHECKS PASSED' if not failures else f'{len(failures)} FAILED'}; wrote {OUT}/summary.json")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
