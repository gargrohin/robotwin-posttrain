"""CPU check of the Flux3_Action XPolicyLab adapter (no GPU, no weights loaded: the DiT is built on
the meta device, and the video VAE, text encoder and sampler are stand-ins that record their inputs).

A. Training parity. For real windows of the RoboTwin-mix200 index, runs the policy's own training
   path (FluxActionPolicy.prepare, in eval mode: no caption dropout, no augmentation) on the
   training window, and the adapter's inference path on the same moment.
   A1: the window's frames as the trainer decodes them (360x640) given to the adapter's batch
       builder: the canvas the VAE receives (bf16, 544x736), the state token, the caption and the
       position ids must be bit-identical to the training window's frame 0.
   A2: the same frames at their native 240x320 through the eval path (RoboTwin get_obs()-style dict
       -> the sim client's robotwin_obs_to_xpolicylab -> websocket codec -> the server's
       decode_obs_images -> adapter Model): state token and caption identical; the canvas differs
       only by the camera resize (the trainer's decoder rescaled 240x320 -> 360x640 with libswscale
       bilinear, the adapter with the policy's own bilinear resize); its difference is reported.
       With --trainer-frames (frames decoded by the training image's ffmpeg 6.1.1, see
       flux3/NOTES or the run notes) the resize alone is compared with the trainer's exact bytes.
B. Eval-image parity. Recorded demos (HDF5, the simulator's own renders) through the same eval
   path vs the training window of the same moment: canvas within codec noise, state token and
   caption identical.
C. Output decoding. The stand-in sampler returns a training window's own normalized targets; the
   adapter's get_action -> websocket codec -> the sim client's normalize_action_chunk and
   xpolicylab_action_to_robotwin must give back the window's recorded absolute actions (14-D, the
   simulator's order). Also: 32 actions per call, GR00T dict format, actions_per_call, the noise
   seed per query, the sampling settings.
D. Settings: the export's normalization equals the index statistics; its camera keys, chunk, fps
   and parameterization are the training run's; the export's sha256 manifest.

Run in the flux-action venv from the repo root (CPU only):
  CUDA_VISIBLE_DEVICES= /opt/dlami/nvme/rohingarg/flux3-venv/bin/python flux3/check_adapter.py
Writes outputs/flux3_adapter_check/{summary.json,canvas.png}; exits 1 on a failure.
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
OUT = ROOT / "outputs/flux3_adapter_check"
SOURCE = ROOT / "data/lerobot/RoboTwin-mix200-aloha_agilex-joint"
INDEX = Path("/opt/dlami/nvme/rohingarg/flux3-data/index")
EXPORT = Path("/opt/dlami/nvme/rohingarg/flux3-eval/exports/f3a-mix200-b512-2n-v2/step-1050")
DEMOS = [("beat_block_hammer", 5), ("handover_block", 15), ("open_laptop", 25)]  # demo_clean, all in mix200

from flux_action.config import DROID_INFERENCE_SETTINGS, PolicyConfig  # noqa: E402
from flux_action.policy import FluxActionPolicy  # noqa: E402
from flux_action.processing import packing  # noqa: E402
from flux_action.training.data import WindowDataset, collate_windows, load_manifest  # noqa: E402
from XPolicyLab.policy.Flux3_Action import model as adapter  # noqa: E402  (before the client edits sys.path)
from XPolicyLab.utils.process_data import decode_obs_images  # noqa: E402

sys.path.insert(0, str(ROOT / "gr00t"))
import open_loop_check as olc  # noqa: E402  (GR00T check: eval-client loader, codec round trip, demo loader)

failures: list[str] = []


def check(ok: bool, what: str) -> bool:
    print(f"  {'PASS' if ok else 'FAIL'}  {what}")
    if not ok:
        failures.append(what)
    return ok


class StandInVAE:
    """Records what the policy hands the VAE; returns zero latents of the right shape."""

    def __init__(self):
        self.videos, self.frames = [], []

    def encode(self, videos):  # training: (B, 3, T, Hc, Wc) bf16
        self.videos.append(videos.detach().clone())
        b, _, t, h, w = videos.shape
        return torch.zeros(b, packing.LATENT_CHANNELS, packing.latent_frames(t), h // 32, w // 32, dtype=torch.bfloat16)

    def encode_frame(self, frame):  # inference: (1, 3, Hc, Wc) bf16
        self.frames.append(frame.detach().clone())
        _, _, h, w = frame.shape
        return torch.zeros(1, packing.LATENT_CHANNELS, 1, h // 32, w // 32, dtype=torch.bfloat16)


class CPUPolicy(FluxActionPolicy):
    """The real policy class with the DiT on the meta device; tensors stay on the CPU."""

    @property
    def device(self):
        return torch.device("cpu")


def make_policy(config: PolicyConfig):
    vae = StandInVAE()
    policy = CPUPolicy(config, video_vae=vae, text_encoder=object(), _restore=True).eval()
    captions: list[str] = []

    def contexts(caps, device, *, fixed_length=None):  # records captions instead of encoding them
        captions.extend(caps)
        return [(torch.zeros(1, 8, 4), torch.zeros(1, 8, 4, dtype=torch.long)) for _ in caps]

    policy._contexts = contexts
    sampled: list[dict] = []

    def sample(cond, caption, seed, prefix=None):  # stand-in sampler: returns the queued targets
        sampled.append({"cond": cond, "caption": caption, "seed": seed})
        return policy.next_targets.clone()

    policy._sample = sample
    policy.next_targets = torch.zeros(config.chunk_size, config.action_dim)
    return policy, vae, captions, sampled


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
    xpl = client.robotwin_obs_to_xpolicylab(sim_obs, instruction=instruction, env_idx=0)
    return decode_obs_images(olc.ws_roundtrip(xpl))


def frame0(item: dict, key: str) -> np.ndarray:
    return item[key][0].permute(1, 2, 0).numpy()  # (3, H, W) uint8 -> HxWx3


def training_reference(policy, vae, captions, item) -> dict:
    """FluxActionPolicy.prepare on one training window (eval mode): canvas frame 0, state, caption."""
    vae.videos.clear(), captions.clear()
    prepared = policy.prepare(collate_windows([item]))
    cfg = policy.config
    action = packing.pack_actions(prepared.state[:, None], prepared.actions,
                                  packing.default_action_times(1, cfg.chunk_size, cfg.fps), policy.modality,
                                  scale=cfg.action_scale)
    ck = f"x_{policy.modality}_cond"
    return {"canvas": vae.videos[-1][0, :, 0], "caption": captions[-1], "state_token": action[ck],
            "state_ids": action[f"{ck}_ids"], "targets": prepared.actions[0],
            "video_cond_ids": packing.pack_video(prepared.latents, fps=cfg.fps)["x_video_cond_ids"]}


def compare(ref: dict, vae, sampled, policy, *, exact_canvas: bool) -> dict:
    mine = vae.frames[-1][0]
    ck = f"x_{policy.modality}_cond"
    cond = sampled[-1]["cond"]
    check(mine.shape == ref["canvas"].shape == (3, 544, 736) and mine.dtype == ref["canvas"].dtype == torch.bfloat16,
          f"canvas {tuple(mine.shape)} bf16, as training")
    diff = (mine.float() - ref["canvas"].float()).abs() * 127.5  # canvas is in [-1, 1]: x 127.5 = /255 units
    if exact_canvas:
        check(bool(torch.equal(mine, ref["canvas"])), "canvas (VAE input) bit-identical to training frame 0")
    check(torch.equal(cond[ck].float(), ref["state_token"].float()) and torch.equal(cond[f"{ck}_ids"], ref["state_ids"]),
          "state token (normalized x action_scale) and its position ids identical")
    check(sampled[-1]["caption"] == ref["caption"], f"caption identical: {ref['caption']!r}")
    check(torch.equal(cond["x_video_cond_ids"], ref["video_cond_ids"]),
          "conditioning-frame token position ids as training (latent frame 0)")
    return {"canvas_mean_abs_diff_255": float(diff.mean()), "canvas_max_abs_diff_255": float(diff.max())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--export", type=Path, default=EXPORT)
    parser.add_argument("--trainer-frames", type=Path,
                        default=Path("/opt/dlami/nvme/rohingarg/flux3-eval/scratch/frames/train_frames.npz"))
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    summary: dict = {"export": str(args.export)}

    config = PolicyConfig(**json.loads((args.export / "config.json").read_text()))
    config = PolicyConfig(**{**config.to_dict(), **DROID_INFERENCE_SETTINGS})  # as load_policy applies them
    policy, vae, captions, sampled = make_policy(config)
    model_cfg = {"env_cfg_type": "aloha_agilex", "seed": 3, "task_name": "beat_block_hammer"}
    model = adapter.Model.__new__(adapter.Model)
    model._setup(model_cfg, policy, device="cpu")
    client = olc.load_eval_client()

    manifest = load_manifest(INDEX / "manifest.json")
    ds360 = WindowDataset(manifest, SOURCE, INDEX / "rows.f32.npy", seed=0, decoder="pyav", frame_hw=(360, 640))
    ds240 = WindowDataset(manifest, SOURCE, INDEX / "rows.f32.npy", seed=0, decoder="pyav", frame_hw=(240, 320))
    keys = [key for key, _ in adapter.CAMERAS]

    print("A. training parity")
    rows_a = []
    n_ep = len(ds360.episodes)
    for pos in (0, n_ep // 2, n_ep - 1):
        starts = ds360.valid_starts(pos)
        for start in (starts[0], starts[len(starts) // 2], starts[-1]):
            item = ds360.window_at(pos, start)
            ref = training_reference(policy, vae, captions, item)
            state = item["state"].numpy()
            print(f" episode {item['episode_index']} frame {start}")
            # A1: the trainer's 360x640 frames straight into the adapter's batch builder.
            batch = adapter.build_batch([frame0(item, k) for k in keys], state, item["task"], "cpu")
            with torch.inference_mode():
                policy.predict_action_chunk(batch)
            a1 = compare(ref, vae, sampled, policy, exact_canvas=True)
            # A2: native 240x320 frames through the whole eval path.
            native = ds240.window_at(pos, start)
            obs = eval_path_obs(client, sim_obs_from([frame0(native, k) for k in keys], state), item["task"])
            model.update_obs(obs)
            model.get_action()
            a2 = compare(ref, vae, sampled, policy, exact_canvas=False)
            rows_a.append({"episode": item["episode_index"], "frame": start, "A1": a1, "A2_eval_path": a2})
    summary["A_training_parity"] = rows_a
    summary["A2_canvas_mean_abs_diff_255"] = float(np.mean([r["A2_eval_path"]["canvas_mean_abs_diff_255"] for r in rows_a]))
    print(f"  A2 canvas vs training (resize only): mean |diff| {summary['A2_canvas_mean_abs_diff_255']:.2f}/255")
    if args.trainer_frames.is_file():
        d = np.load(args.trainer_frames)
        diffs = []
        for name in d.files:
            if name.endswith("_native"):
                mine = adapter.camera_tensor(d[name])[0].permute(1, 2, 0).numpy().astype(int)
                diffs.append(np.abs(mine - d[name.replace("_native", "_train")].astype(int)))
        diffs = np.stack(diffs)
        summary["A2_resize_vs_trainer_ffmpeg"] = {"frames": len(diffs), "mean": float(diffs.mean()),
                                                   "p99": float(np.percentile(diffs, 99)), "max": int(diffs.max())}
        print(f"  adapter resize vs the trainer's ffmpeg 6.1.1 frames ({len(diffs)} frames): "
              f"mean |diff| {diffs.mean():.2f}/255, p99 {np.percentile(diffs, 99):.0f}, max {diffs.max()}")

    print("B. eval-image parity (recorded demos, the simulator's renders, through the eval path)")
    rows_b = []
    sim_cam_of = {xpl: sim for sim, xpl in client.CAMERA_NAME_MAP.items()}
    rows = np.load(INDEX / "rows.f32.npy", mmap_mode="r")
    for task, ep in DEMOS:
        demo = olc.load_demo(olc.ROBOTWIN / f"data/demo_clean/{task}/aloha_agilex/data/episode_{ep:07d}.hdf5")
        state = olc.flat(demo["state"])
        matches = [p for p, e in enumerate(ds360.episodes) if e["n_frames"] == len(state)
                   and np.allclose(rows[e["from_index"]:e["from_index"] + len(state), :14], state, atol=1e-6)]
        if not check(len(matches) == 1, f"{task} demo {ep}: exactly one matching training episode"):
            continue
        pos = matches[0]
        starts = ds360.valid_starts(pos)
        for t in (starts[0], starts[len(starts) // 2], starts[-1]):
            item = ds360.window_at(pos, t)
            ref = training_reference(policy, vae, captions, item)
            model.update_obs(eval_path_obs(client, olc.sim_observation(demo, t, sim_cam_of), item["task"]))
            model.get_action()
            print(f" {task} demo {ep} = training episode {item['episode_index']}, frame {t}")
            stats = compare(ref, vae, sampled, policy, exact_canvas=False)
            check(stats["canvas_mean_abs_diff_255"] < 4.0,
                  f"canvas within codec noise of training (mean |diff| {stats['canvas_mean_abs_diff_255']:.2f}/255)")
            rows_b.append({"task": task, "demo": ep, "episode": item["episode_index"], "frame": t, **stats})
        if task == DEMOS[0][0]:
            from PIL import Image

            to_u8 = lambda c: ((c.float() + 1) * 127.5).round().clamp(0, 255).to(torch.uint8)  # noqa: E731
            a, b = to_u8(ref["canvas"]), to_u8(vae.frames[-1][0])
            panels = [a, b, ((a.int() - b.int()).abs() * 5).clamp(0, 255).to(torch.uint8)]
            Image.fromarray(torch.cat(panels, dim=-1).permute(1, 2, 0).numpy()).save(OUT / "canvas.png")
    summary["B_eval_image_parity"] = rows_b

    print("C. output decoding (stand-in sampler returns a training window's own targets)")
    item = ds360.window_at(n_ep // 3, ds360.valid_starts(n_ep // 3)[10])
    ref = training_reference(policy, vae, captions, item)
    recorded = item["action"].numpy()  # (32, 14) absolute commands of the window
    policy.next_targets = ref["targets"].clone()
    state = item["state"].numpy()
    sim_obs = sim_obs_from([frame0(ds240.window_at(n_ep // 3, ds360.valid_starts(n_ep // 3)[10]), k) for k in keys], state)
    model._setup(model_cfg, policy, device="cpu")
    model.update_obs(eval_path_obs(client, sim_obs, item["task"]))
    reply = olc.ws_roundtrip({"result": model.get_action()})["result"]  # what the client receives
    chunk = client.normalize_action_chunk(reply)
    flat = [client.xpolicylab_action_to_robotwin(a, action_type="joint", current_observation=sim_obs) for a in chunk]
    got = np.stack([f for f, _ in flat])
    check(len(chunk) == 32 and all(kind == "qpos" for _, kind in flat), "32 joint (qpos) actions per call")
    check(set(reply[0]) == set(adapter.STATE_KEYS) and reply[0]["left_arm_joint_state"].shape == (6,),
          "GR00T_N17 per-step dict format")
    err = float(np.abs(got - recorded).max())
    check(err < 1e-5, f"sim receives the window's recorded absolute actions (max |err| {err:.1e} rad)")
    check(sampled[-1]["seed"] == 3_000_000 + 1, f"first query's noise seed = seed_base + 1 ({sampled[-1]['seed']})")
    model.get_action()
    check(sampled[-1]["seed"] == 3_000_000 + 2, "second query's noise seed = seed_base + 2")
    c = policy.config
    settings = {k: getattr(c, k) for k in DROID_INFERENCE_SETTINGS}
    check(settings == DROID_INFERENCE_SETTINGS, f"sampling settings = DROID_INFERENCE_SETTINGS {settings}")
    model._setup({**model_cfg, "actions_per_call": 8}, policy, device="cpu")
    model.update_obs(eval_path_obs(client, sim_obs, "x"))
    check(len(model.get_action()) == 8, "actions_per_call=8 returns the first 8")
    summary["C_max_abs_err_rad"] = err

    print("D. settings")
    stats = json.loads((INDEX / "statistics.json").read_text())
    for name, key in (("action_normalization", "action"), ("state_normalization", "state")):
        got = getattr(c, name)
        check(np.allclose(got["q01"], stats[key]["q01"], rtol=0, atol=0) and np.allclose(got["q99"], stats[key]["q99"], rtol=0, atol=0),
              f"export {name} = index statistics.json {key} q01/q99")
    check((c.chunk_size, c.n_action_steps, c.fps, c.action_dim, c.action_parameterization, c.normalization_clip)
          == (32, 32, 15.0, 14, "absolute", stats["clip"]) and stats["action"]["parameterization"] == "absolute",
          "chunk 32, n_action_steps 32, 15 fps, 14-D absolute, clip 6 (as the index)")
    check(tuple(c.camera_keys) == tuple(f"images.{k}" for k in manifest["camera_order"]),
          f"camera keys {c.camera_keys} = index camera order {manifest['camera_order']} "
          f"({[manifest['cameras'][k] for k in manifest['camera_order']]})")
    mf = json.loads((args.export / "manifest.json").read_text())
    digest = hashlib.sha256((args.export / "config.json").read_bytes()).hexdigest()
    check(mf["sha256"]["config.json"] == digest and mf["weight_profile"] == "ema_0p10" and c.torch_dtype == "bfloat16",
          f"export manifest: config sha256 {digest[:16]}, weights {mf['weight_profile']}, {c.torch_dtype}")
    summary["failures"] = failures
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"\n{'ALL CHECKS PASSED' if not failures else f'{len(failures)} FAILED'}; wrote {OUT}/summary.json")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
