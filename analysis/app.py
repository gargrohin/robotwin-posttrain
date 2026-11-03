"""Eval explorer: browse RoboTwin eval episodes by model, checkpoint, setting and task.

Start with analysis/run.sh (see analysis/README.md). Reads analysis/cache/episodes.parquet;
rebuild it after new evals with: analysis/.venv/bin/python analysis/build_index.py
"""

import base64
import json
import math
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "analysis/cache/episodes.parquet"
DEMOS = ROOT / "data/lerobot/RoboTwin-mix200-aloha_agilex-joint"  # training demos of mix200_150k
EVAL_FPS = 10  # RoboTwin rollout videos: 10 fps, one frame per action step
DEMO_FPS = 15  # training demo videos: 15 fps, one frame per step
SETTINGS = {"demo_clean": "clean", "demo_randomized": "randomized"}
ET = "America/New_York"
# the local Cosmos run (stopped at ~1,700 iterations); only the cluster run
# cosmos3_nano_mix200_2n is used (user, 2026-11-03)
HIDDEN_MODELS = {"cosmos3_nano_mix200"}

PLAYER = """
<video id="v" src="data:video/mp4;base64,__B64__" controls muted playsinline
       style="width:100%;max-width:640px;image-rendering:pixelated;background:#000"></video>
<div style="font:14px sans-serif;margin-top:4px">
  <button onclick="step(-10)">&minus;10</button>
  <button onclick="step(-1)">&#9664; frame</button>
  <button onclick="toggle()">play / pause</button>
  <button onclick="step(1)">frame &#9654;</button>
  <button onclick="step(10)">+10</button>
  steps per second <select onchange="v.playbackRate = this.value / fps">
    <option>2.5</option><option>5</option><option selected>10</option><option>20</option><option>40</option>
  </select>
  <b id="label" style="margin-left:8px"></b>
  <div style="color:#888;font-size:12px">click the video, then: &larr; &rarr; one frame,
    shift + &larr; &rarr; ten frames, space play / pause</div>
</div>
<script>
const v = document.getElementById("v"), label = document.getElementById("label");
const fps = __FPS__, n = __FRAMES__;
v.defaultPlaybackRate = v.playbackRate = 10 / fps;  // every player starts at 10 steps per second
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


@st.cache_data
def load_demos():
    """One row per training demo: 50 clean + 200 randomized per task.

    Dataset order (run card notes/runs/mix200_150k.md, "Merge order"): episodes 0-2499 are
    the clean demos, tasks sorted, 50 each; then 4 groups of 2,500 randomized demos in the
    same order, holding randomized demos 0-49, 50-99, 100-149 and 150-199 of each task.
    """
    tasks = sorted(p.name for p in (ROOT / "RoboTwin/data/demo_clean").iterdir())
    info = json.loads((DEMOS / "meta/info.json").read_text())
    rows = []
    for i, line in enumerate((DEMOS / "meta/episodes.jsonl").open()):
        ep, k = json.loads(line), i % 2500
        video = info["video_path"].format(episode_chunk=i // info["chunks_size"], episode_index=i,
                                          video_key="observation.images.cam_high")
        rows.append({"task": tasks[k // 50],
                     "setting": "demo_clean" if i < 2500 else "demo_randomized",
                     "number": k % 50 + 50 * max(0, i // 2500 - 1),
                     "length": ep["length"], "instruction": ep["tasks"][0],
                     "video": str((DEMOS / video).relative_to(ROOT))})
    return pd.DataFrame(rows)


@st.cache_data(max_entries=32)
def video_b64(path):
    return base64.b64encode((ROOT / path).read_bytes()).decode()


def wilson(k, n, z=1.96):
    """95% confidence interval for a success rate of k out of n."""
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def player(path, frames, fps):
    html = (PLAYER.replace("__B64__", video_b64(path)).replace("__FPS__", str(fps))
            .replace("__FRAMES__", str(frames)))
    st.iframe(html, height=560)


def pick_model(label, df, default):
    """Model and checkpoint selectors for side `label`; returns its name and episodes."""
    models = sorted(df.model.unique())
    model = st.selectbox(f"model {label}", models, key=f"model {label}",
                         index=models.index(default) if default in models else 0)
    d = df[df.model == model]
    ckpt = st.selectbox(f"checkpoint {label}", key=f"checkpoint {label}", options=sorted(
        d.ckpt_label.unique(), key=lambda c: -int(c) if c.isdigit() else 1))
    return f"{model} @ {ckpt}", d[(d.ckpt_label == ckpt) & ~d.spot_check]


def result_text(success, sim_error):
    return "⚠️ sim crash" if sim_error else ("✅ success" if success else "❌ fail")


def compare_view(df):
    """Two models on the scenes (task, setting, seed) both ran, rollouts side by side."""
    key = ["task", "setting", "seed"]
    with st.sidebar:
        name_a, a = pick_model("A", df, "mix200_150k")
        name_b, b = pick_model("B", df, "cosmos3_nano_mix200_2n")
        j = a.drop_duplicates(key).merge(b.drop_duplicates(key), on=key, suffixes=("_a", "_b"))
        if j.empty:
            st.warning("These two have no scenes in common.")
            st.stop()
        setting = st.radio("setting", [s for s in SETTINGS if s in set(j.setting)],
                           format_func=SETTINGS.get, horizontal=True, key="compare setting")
        j = j[j.setting == setting]
        per_task = j.groupby("task").agg(a=("success_a", "sum"), b=("success_b", "sum"),
                                         n=("seed", "size"))
        task = st.selectbox(
            "task (biggest B minus A first)", key="compare task",
            options=(per_task.b - per_task.a).sort_values(ascending=False).index,
            format_func=lambda t: f"{t}  (A {per_task.a[t]}, B {per_task.b[t]} of {per_task.n[t]})")
        show = st.radio("scenes", ["all", "only B succeeds", "only A succeeds", "both fail",
                                   "both succeed"], key="compare show")
        st.caption(f"A = {name_a}\n\nB = {name_b}\n\nOver all {len(j)} shared {SETTINGS[setting]} "
                   f"scenes: A {int(j.success_a.sum())}, B {int(j.success_b.sum())}.")

    t = j[j.task == task].sort_values("seed")
    sa, sb = t.success_a, t.success_b
    st.subheader(f"{task} · {SETTINGS[setting]}: A = {name_a}, B = {name_b}")
    c = st.columns(6)
    for col, (label, value) in zip(c, [("A succeeds", sa.sum()), ("B succeeds", sb.sum()),
                                      ("both", (sa & sb).sum()), ("only A", (sa & ~sb).sum()),
                                      ("only B", (~sa & sb).sum()), ("neither", (~sa & ~sb).sum())]):
        col.metric(label, f"{int(value)}/{len(t)}")
    view = t[{"all": sa | ~sa, "only B succeeds": ~sa & sb, "only A succeeds": sa & ~sb,
              "both fail": ~sa & ~sb, "both succeed": sa & sb}[show]]
    if view.empty:
        st.info("No scenes in this view.")
        st.stop()
    table = pd.DataFrame({
        "seed": view.seed,
        "A": [result_text(s, e) for s, e in zip(view.success_a, view.sim_error_a)],
        "B": [result_text(s, e) for s, e in zip(view.success_b, view.sim_error_b)],
        "A steps": view.steps_a, "B steps": view.steps_b, "step limit": view.step_limit_a,
        "same instruction": (view.instruction_a == view.instruction_b).map({True: "yes", False: "no"}),
    })
    event = st.dataframe(table, hide_index=True, height=300, on_select="rerun",
                         selection_mode="single-row",
                         key=f"compare|{name_a}|{name_b}|{setting}|{task}|{show}")
    row = view.iloc[event.selection.rows[0] if event.selection.rows else 0]
    for col, side, name in zip(st.columns(2), "ab", [name_a, name_b]):
        with col:
            st.markdown(f"**{side.upper()}: {name}**: "
                        f"{result_text(row[f'success_{side}'], row[f'sim_error_{side}'])}, "
                        f"{row[f'steps_{side}']} / {row[f'step_limit_{side}']} steps")
            if pd.notna(row[f"video_{side}"]):
                player(row[f"video_{side}"], int(row[f"video_frames_{side}"]), EVAL_FPS)
            else:
                st.warning("No video for this episode.")
            instr = row[f"instruction_{side}"]
            st.markdown("- instruction: " + (f"“{instr}”" if pd.notna(instr) else "_not recorded_"))
    st.caption(f"seed `{row.seed}`. Same seed means the same starting scene. The instruction "
               "sentence can still differ between the two: RoboTwin picks it from a list as "
               "long as the eval stream's episode count, so streams of different sizes pick "
               "different sentences.")


st.set_page_config(page_title="Eval explorer", layout="wide")
if not INDEX.exists():
    st.error("No index yet. Build it: `analysis/.venv/bin/python analysis/build_index.py`")
    st.stop()
df = load_index(INDEX.stat().st_mtime)
df = df[~df.model.isin(HIDDEN_MODELS)]

with st.sidebar:
    mode = st.radio("view", ["one model", "two models, same scenes"], horizontal=True)
if mode != "one model":
    compare_view(df)
    st.stop()

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

left, right = st.columns(2)
with right:
    st.markdown("**Training demo of this task**", help=(
        "Recorded by RoboTwin's scripted robot, which knows where every object is and follows "
        "a planned motion; only successful runs were kept. These are mix200_150k's training "
        "demos (the 50 clean ones are also 50task_60k's)."))
    demos = load_demos()
    dm = demos[(demos.task == task) & (demos.setting == setting)].set_index("number")
    num = st.number_input(f"{SETTINGS[setting]} demo (0 to {len(dm) - 1})", 0, len(dm) - 1, 0,
                          key=f"demo|{task}|{setting}")
    demo = dm.loc[num]
    player(demo.video, int(demo.length), DEMO_FPS)
    st.markdown(f"- {demo.length} steps\n- instruction: “{demo.instruction}”")
with left:
    st.markdown("**Rollout**")
    if pd.notna(ep.video):
        player(ep.video, int(ep.video_frames), EVAL_FPS)
    else:
        st.warning("No video for this episode: its folder was deleted, moved mid-run, or "
                   "can't be told apart from a sibling stream's (see build_index.py).")
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
