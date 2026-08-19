"""DEV enlargement: leave-one-out CV over all 19 train vessels, not a fixed 5.

WHY.  The 3091-trial search selected on a **5-vessel DEV**.  That is enormous selection
pressure on a tiny set, and its -0.0114 is not a held-out number in any useful sense.  DEV
had already shown it could not resolve anything: an exact 5-way tie earlier in the phase.
This replaces it with leave-one-out CV over all 19 -- every vessel is held out once, trained
on the other 18 -- which is the largest honest evaluation available without opening SEALED.

WHAT IS STILL CONTAMINATED, STATED PLAINLY.  The *configuration* (model class, feature
group, hyperparameters) was chosen using these same 19 vessels, so CV removes the "selected
on those particular 5" problem but not configuration selection.  Read the result as an upper
bound on held-out performance, not an unbiased estimate.  SEALED remains the only clean test
and is not opened here.

METHODS, all scored on ``growth_l1``:
    physics            the shipped zero-parameter ODE                    (backbone)
    + AP closure       the closure shipped today, C re-checked by CV     0 learned params
    global_affine      shift + scale on the physics onset                2 params
    node_mlp(gate)     the search winner                                 97 params
    COUNT FLOOR        best achievable on this committed set             the ceiling

    python scripts/eval_onset_cv.py --minutes 55
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.config import BiochemConfig  # noqa: E402
from src.core_physics.ap_closure import ApClosure  # noqa: E402
from src.core_physics.growth_count_metrics import count_optimal_onset, growth_error  # noqa: E402
from src.differentiable_wall_model.onset_residual import build, growth_l1_soft  # noqa: E402

OUT = Path("outputs/onset_search")


def _search():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "aos", str(REPO / "scripts" / "auto_onset_search.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def train_eval(aos, cfg, vessels, tr, te, device, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    cols = aos.cols_for(cfg["groups"])
    X = torch.cat([vessels[n]["X"][:, cols] for n in tr])
    mu, sd = X.mean(0), X.std(0).clamp(min=1e-6)
    for v in vessels.values():
        v["Xn"] = (v["X"][:, cols] - mu) / sd
    model = build(cfg["model"], len(cols), hidden=cfg["hidden"], layers=cfg["layers"],
                  rounds=cfg["layers"], cap=cfg["cap"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    for _ in range(cfg["epochs"]):
        model.train()
        order = list(tr)
        np.random.shuffle(order)
        for n in order:
            v = vessels[n]
            o = model(v["ode"], v["Xn"], v["edges"])
            if not cfg["allow_never"]:
                o = o.clamp(0.0, 1.0)
            loss = growth_l1_soft(o, v["gt_curve"], v["grid"], v["n_gt"], cfg["tau"])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    return aos.eval_hard(model, vessels, te, cols, cfg["allow_never"])[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=55.0)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    t0 = time.time()
    budget = args.minutes * 60.0
    aos = _search()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bio = BiochemConfig(phase="biochem")
    prot = json.load(open("outputs/ap_closure/protocol_gt_meanovertime.json"))
    names = prot["fit"] + prot["dev"]
    lb = json.load(open(OUT / "leaderboard.json"))
    best_cfg = lb["top"][0]["cfg"]
    print("device=%s   %d vessels, leave-one-out CV   SEALED NOT OPENED" % (device, len(names)))
    print("winner config: %s / %s / %d-dim / %d epochs\n"
          % (best_cfg["model"], ",".join(best_cfg["groups"]), best_cfg["hidden"],
             best_cfg["epochs"]))

    vessels = aos.load_vessels(names, bio, float(prot["best_cl"]["C"]), device)
    names = [n for n in names if n in vessels]
    assert not (set(vessels) & set(prot["sealed"]))

    # ---------------------------------------------------------- zero-parameter arms
    from src.core_physics.onset_features import committed_set
    from src.core_physics.physics_wall_model import first_crossing, integrate_mat_trajectory
    from src.core_physics.ap_closure import make_rollout_hook

    M_TO_CM = 100.0
    lss, sgt = float(bio.lss), float(bio.sgt) / M_TO_CM
    coef = float(bio.L_char) * M_TO_CM / float(bio.gamma_m)
    per = {}

    def closure_arm(C):
        res = {}
        for n in names:
            z = np.load(Path("outputs/wall_species_cache") / f"{n}.npz")
            gate = (z["dsrx0"] < sgt) * coef * np.abs(z["dsrx0"]) + (z["sr0"] < lss)
            S = committed_set(gate, z["sr0"], z["wall_edges"])
            nt = len(z["t"])
            hook = None if C == 0 else make_rollout_hook(
                ApClosure(C=C, q=1.0, kernel="static"), bio, z["sr0"])
            traj, _ = integrate_mat_trajectory(aos.WallShim(z), bio, gate, da_scale=40.0,
                                               ap_closure=hook)
            idx = first_crossing(traj, float(bio.viscosity_mat_crit))
            cr = idx >= 0
            med = int(np.median(idx[cr])) if cr.any() else 0
            on = np.where(S, np.where(idx >= 0, idx, med), -1)
            res[n] = growth_error(on, z["gt_onset"], nt)["growth_l1"]
            if C == 0:
                per.setdefault("floor", {})[n] = growth_error(
                    count_optimal_onset(S, z["gt_onset"], nt), z["gt_onset"], nt)["growth_l1"]
        return res

    per["physics"] = closure_arm(0.0)
    print("[closure] re-checking C on the growth metric over all 19 vessels")
    sweep = {}
    for C in (20.0, 40.0, 62.42, 100.0, 160.0, 250.0):
        sweep[C] = closure_arm(C)
        print("   C=%-7.2f growth_l1 %.4f" % (C, np.mean(list(sweep[C].values()))))
    C_best = min(sweep, key=lambda c: float(np.mean(list(sweep[c].values()))))
    per["ap closure (shipped C=62.42)"] = sweep[62.42]
    per["ap closure (CV-best C=%.0f)" % C_best] = sweep[C_best]

    # ------------------------------------------------------------------ learned arms
    arms = {"global_affine (2 par)": dict(best_cfg, model="global_affine", groups=["gate"]),
            "node_mlp gate (97 par)": best_cfg}
    for tag, cfg in arms.items():
        if time.time() - t0 > budget * 0.9:
            print("[budget] skipping %s" % tag)
            continue
        res = {}
        for n in names:
            tr = [m for m in names if m != n]
            vals = [train_eval(aos, cfg, vessels, tr, [n], device, s)
                    for s in range(args.seeds)]
            res[n] = float(np.mean(vals))
        per[tag] = res
        print("[cv] %-26s done  %.0fs elapsed" % (tag, time.time() - t0))

    # ----------------------------------------------------------------------- report
    print("\n" + "=" * 88)
    print("LEAVE-ONE-OUT CV over %d vessels   growth_l1 (0 = perfect)" % len(names))
    print("=" * 88)
    base = np.array([per["physics"][n] for n in names])
    floor = np.array([per["floor"][n] for n in names])
    print("%-30s %9s %9s %10s %14s" % ("arm", "mean", "median", "vs physics", "95% CI"))
    print("%-30s %9.4f %9.4f %10s" % ("physics (backbone)", base.mean(), np.median(base), "-"))
    rows = {}
    for tag in per:
        if tag in ("physics", "floor"):
            continue
        v = np.array([per[tag][n] for n in names])
        d = v - base
        se = d.std(ddof=1) / np.sqrt(len(d))
        rows[tag] = dict(mean=float(v.mean()), delta=float(d.mean()),
                         lo=float(d.mean() - 1.96 * se), hi=float(d.mean() + 1.96 * se),
                         better=int((d < -1e-9).sum()), worse=int((d > 1e-9).sum()))
        print("%-30s %9.4f %9.4f %+10.4f  [%+.4f,%+.4f] %2d/%2d better"
              % (tag, v.mean(), np.median(v), d.mean(), rows[tag]["lo"], rows[tag]["hi"],
                 rows[tag]["better"], len(names)))
    print("%-30s %9.4f %9.4f %+10.4f   <- the ceiling"
          % ("COUNT FLOOR (mask)", floor.mean(), np.median(floor), floor.mean() - base.mean()))
    prize = base.mean() - floor.mean()
    print("\n   prize for any timing model: %+.4f" % -prize)
    for tag, r in sorted(rows.items(), key=lambda kv: kv[1]["delta"]):
        print("   %-30s recovers %5.1f%% of it%s"
              % (tag, 100.0 * (-r["delta"]) / prize,
                 "" if r["hi"] < 0 else "   (CI includes zero)"))

    (OUT / "cv_result.json").write_text(json.dumps(
        dict(per_vessel=per, rows=rows, names=names, prize=float(prize),
             C_best=float(C_best)), indent=2, default=float), encoding="utf-8")
    print("\nwrote %s   (%.0f min)" % (OUT / "cv_result.json", (time.time() - t0) / 60))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
