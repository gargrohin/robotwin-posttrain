"""Eval explorer: browse RoboTwin eval episodes by model, checkpoint, setting and task.

Start with analysis/run.sh (see analysis/README.md). Reads analysis/cache/episodes.parquet;
rebuild it after new evals with: analysis/.venv/bin/python analysis/build_index.py
"""

import base64
import math
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "analysis/cache/episodes.parquet"
EVAL_FPS = 10  # RoboTwin rollout videos: 10 fps, one frame per action step
SETTINGS = {"demo_clean": "clean", "demo_randomized": "randomized"}
ET = "America/New_York"

PLAYER = """
<video id="v" src="data:video/mp4;base64,__B64__" controls muted playsinline
       style="width:100%;max-width:640px;image-rendering:pixelated;background:#000"></video>
<div style="font:14px sans-serif;margin-top:4px">
  <button onclick="step(-10)">&minus;10</button>
  <button onclick="step(-1)">&#9664; frame</button>
  <button onclick="toggle()">play / pause</button>
  <button onclick="step(1)">frame &#9654;</button>
  <button onclick="step(10)">+10</button>
  speed <select onchange="v.playbackRate = this.value">
    <option>0.25</option><option>0.5</option><option selected>1</option><option>2</option><option>4</option>
  </select>
  <b id="label" style="margin-left:8px"></b>
  <div style="color:#888;font-size:12px">click the video, then: &larr; &rarr; one frame,
    shift + &larr; &rarr; ten frames, space play / pause</div>
</div>
<script>
const v = document.getElementById("v"), label = document.getElementById("label");
const fps = __FPS__, n = __FRAMES__;
const frame = () => Math.min(n - 1, Math.floor(v.currentTime * fps + 1e-6));
function step(d) {
  v.pause();
  v.currentTime = (Math.max(0, Math.min(n - 1, frame() + d)) + 0.5) / fps;  // middle of the frame
}
function toggle() { v.paused ? v.play() : v.pause(); }
(function tick() { label.textContent = `frame ${frame()} / ${n - 1}`; requestAnimationFrame(tick); })();
document.addEventListener("keydown", e => {
  const d = {ArrowRight: 1, ArrowLeft: -1}[e.key];
  if (d) { step(e.shiftKey ? 10 * d : d); e.preventDefault(); }
  else if (e.key === " ") { toggle(); e.preventDefault(); }
}, true);
</script>
"""


@st.cache_data
def load_index(mtime):  # mtime: reload after a rebuild
    df = pd.read_parquet(INDEX)
    df["ckpt_label"] = df.ckpt.astype("string").fillna("last")
    return df


@st.cache_data(max_entries=32)
def video_b64(path):
    return base64.b64encode((ROOT / path).read_bytes()).decode()


def wilson(k, n, z=1.96):
    """95% confidence interval for a success rate of k out of n."""
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def player(path, frames):
    html = (PLAYER.replace("__B64__", video_b64(path)).replace("__FPS__", str(EVAL_FPS))
            .replace("__FRAMES__", str(frames)))
    st.iframe(html, height=560)


st.set_page_config(page_title="Eval explorer", layout="wide")
if not INDEX.exists():
    st.error("No index yet. Build it: `analysis/.venv/bin/python analysis/build_index.py`")
    st.stop()
df = load_index(INDEX.stat().st_mtime)

with st.sidebar:
    model = st.selectbox("model", sorted(df.model.unique()))
    d = df[df.model == model]
    ckpt = st.selectbox("checkpoint", sorted(d.ckpt_label.unique(),
                                             key=lambda c: -int(c) if c.isdigit() else 1))
    d = d[d.ckpt_label == ckpt]
    setting = st.radio("setting", [s for s in SETTINGS if s in set(d.setting)],
                       format_func=SETTINGS.get, horizontal=True)
    d = d[d.setting == setting]
    if d.spot_check.any() and not d.spot_check.all() and not st.checkbox(
            "include spot checks",
            help="Early 10-episode evals (logs/learning_curve/). They reuse seeds of the "
                 "full benchmark, so they are left out of its numbers by default."):
        d = d[~d.spot_check]
    per_task = d.groupby("task").success.agg(["sum", "size"])
    task = st.selectbox("task", per_task.index, format_func=lambda t: (
        f"{t}  ({per_task.loc[t, 'sum']}/{per_task.loc[t, 'size']})"))
    show = st.radio("episodes", ["all", "failures", "successes"], horizontal=True)
    built = pd.Timestamp(INDEX.stat().st_mtime, unit="s", tz="UTC").tz_convert(ET)
    st.caption(f"Index built {built:%b %d %H:%M} ET. After new evals run "
               "`analysis/.venv/bin/python analysis/build_index.py`, then reload this page.")

t = d[d.task == task].sort_values(["shard", "episode"])
k, n = int(t.success.sum()), len(t)
lo, hi = wilson(k, n)
limit = int(t.step_limit.max())
done = t[t.success].steps
st.subheader(f"{task} · {SETTINGS[setting]} · {model} @ {ckpt}")
c = st.columns(4)
c[0].metric("success", f"{k}/{n} = {k / n:.0%}")
c[1].metric("95% range", f"{lo:.0%} – {hi:.0%}",
            help="With this few episodes, the true success rate is likely somewhere in here.")
c[2].metric("median steps of successes", f"{int(done.median())} / {limit}" if len(done) else "–",
            help="Failures always run to the step limit.")
c[3].metric("episodes with video", f"{t.video.notna().sum()}/{n}")

view = t if show == "all" else t[t.success == (show == "successes")]
if view.empty:
    st.info("No episodes in this view.")
    st.stop()
table = pd.DataFrame({
    "shard": view.shard, "episode": view.episode, "seed": view.seed,
    "result": view.success.map({True: "✅ success", False: "❌ fail"})
    .where(~view.sim_error, "⚠️ sim crash"),
    "steps": view.steps,
    "video": view.video.notna().map({True: "", False: "missing"}),
})
if view.instruction.notna().any():
    table["instruction"] = view.instruction
event = st.dataframe(
    table, hide_index=True, height=300, on_select="rerun", selection_mode="single-row",
    key=f"episodes|{model}|{ckpt}|{setting}|{task}|{show}",
    column_config={"steps": st.column_config.ProgressColumn(
        "steps", format="%d", min_value=0, max_value=limit)},
)
ep = view.iloc[event.selection.rows[0] if event.selection.rows else 0]

left, right = st.columns([3, 2])
with left:
    if pd.notna(ep.video):
        player(ep.video, int(ep.video_frames))
    else:
        st.warning("No video for this episode: its folder was deleted, moved mid-run, or "
                   "can't be told apart from a sibling stream's (see build_index.py).")
with right:
    result = "sim crash (not a policy failure)" if ep.sim_error else (
        "success" if ep.success else "fail")
    st.markdown(f"**shard {ep.shard}, episode {ep.episode}** — {result}")
    st.markdown(f"- seed `{ep.seed}`\n- steps {ep.steps} / {ep.step_limit}")
    st.markdown("- instruction: " + (f"“{ep.instruction}”" if pd.notna(ep.instruction) else
                "_not recorded (evals before 2026-10-30 didn't log it)_"))
    if pd.notna(ep.started):
        st.markdown(f"- stream started {ep.started.tz_convert(ET):%b %d %H:%M} ET")
    st.markdown(f"- video: `{ep.video}`" if pd.notna(ep.video) else "- video: none")
    st.markdown(f"- log: `{ep.log}`" if pd.notna(ep.log) else "- log: none (indexed from records)")
