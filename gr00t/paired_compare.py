"""Compare evals of several models on identical scenes: for each of the 10 tasks, take the
seeds the decay run (mix200_decay60k, checkpoint-10000) was evaluated on and count successes
of every model on exactly those seeds (from each eval's client logs, seed by seed).

usage: python3 gr00t/paired_compare.py [--per-task]   (reads logs/queue_decay60k_10000.txt)
"""
import re, glob, collections, sys
def episodes(ck, num, task, cfg):  # {seed: success} from all of that eval's client logs
    out = {}
    for f in glob.glob(f"logs/eval_{ck}_ckpt-{num}_{task}_{cfg}*.client.log"):
        if not re.fullmatch(rf"logs/eval_{re.escape(ck)}_ckpt-{num}_{task}_{cfg}(_seed\d+)?\.client\.log", f): continue
        t = re.sub(r"\x1b\[[0-9;]*m", "", open(f, errors="replace").read())
        for res, seed in re.findall(r"(Success!|Fail!)\s.*?current seed: (\d+)", t, re.S):
            out[int(seed)] = res == "Success!"
    return out
NEW = "RoboTwin-mix200_150k-aloha_agilex-joint-0"; DEC = "RoboTwin-mix200_decay60k-aloha_agilex-joint-0"
OLD = "RoboTwin-50task_60k-aloha_agilex-joint-0"
MODELS = (("old", OLD, 60000), ("new 60k", NEW, 60000), ("decayed 60k", DEC, 10000), ("new 150k", NEW, 150000),
          ("cosmos 1500", "cosmos3_nano_mix200", 1500)) + tuple(
    (f"cosmos16 {n}", "cosmos3_nano_mix200_2n", n) for n in (2500, 3000, 4000, 5000)
    if glob.glob(f"logs/eval_cosmos3_nano_mix200_2n_ckpt-{n}_*.client.log")) + tuple(  # 16-GPU run, once evaluated
    (f"flux {n}", "flux3_mix200_b512", n) for n in (1500, 3000, 6000, 9000)  # 6000/9000: the continuation run
    if glob.glob(f"logs/eval_flux3_mix200_b512_ckpt-{n}_*.client.log"))  # FLUX 3 Action, once evaluated
TEN = ["pick_dual_bottles", "place_object_stand", "move_pillbottle_pad", "place_empty_cup", "place_phone_stand",
       "move_can_pot", "dump_bin_bigbin", "handover_mic", "grab_roller", "click_bell"]
plan = collections.Counter((l.split()[0], l.split()[1]) for l in open("logs/queue_decay60k_10000.txt"))
for cfg in ("demo_randomized", "demo_clean"):
    tot = collections.Counter(); done = []
    for t in TEN:
        d = episodes(DEC, 10000, t, cfg)
        if len(d) < 10: continue
        done.append(t); seeds = set(d)
        row = []
        for name, ck, num in MODELS:
            e = episodes(ck, num, t, cfg); common = [s for s in seeds if s in e]
            tot[name + " s"] += sum(e[s] for s in common); tot[name + " n"] += len(common)
            row.append(f"{name} {sum(e[s] for s in common)}/{len(common)}")
        if "--per-task" in sys.argv:
            print(f"  {cfg} {t}: " + ", ".join(row))
    if done:
        print(f"{cfg}: {len(done)} tasks finished: " + ", ".join(f"{n} {tot[n+' s']}/{tot[n+' n']} ({tot[n+' s']/max(tot[n+' n'], 1):.0%})" for n, _, _ in MODELS))
    else:
        print(f"{cfg}: no task finished yet")
