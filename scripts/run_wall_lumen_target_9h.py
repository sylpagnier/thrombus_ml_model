"""9h autonomous ladder: WC_v7 wall + GT lumen fire without spray.

Phases:
  A  Probe Prec8h under WALL_ONLY (cheap baseline)
  B  Spray-gated precision retrain (Prec8h init, teachers + spray negatives)
  C  Pivot if needed (stronger FP / unfreeze / teacher-only)
  D  Final orig10 gates; optional polish if budget remains

Usage:
  python -u scripts/run_wall_lumen_target_9h.py
  python -u scripts/run_wall_lumen_target_9h.py --deadline-hours 9 --fresh
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

ORIG10 = (
    "patient001,patient002,patient003,patient004,patient005,"
    "patient006,patient007,patient008,patient010,patient011"
)
TEACHERS = "patient001,patient006,patient007,patient010"
SPRAY_NEGS = "patient002,patient008"
TRAIN_B = f"{TEACHERS},{SPRAY_NEGS}"
from src.evaluation.compound_deploy_gates import (  # noqa: E402
    DEFAULT_FOCUS_ANCHORS as FOCUS,
    gate_compound_eval,
)

WALL = "outputs/biochem/biochem_gnn/locked/species_gnn_best.pth"
PREC8H = "outputs/biochem/offwall_model/wc_v7_frontier_ge2_prec_8h/growth_frontier_ge2_prec/best.pth"
D_ORIG = "outputs/biochem/offwall_model/wc_v7_open001_6h/growth_D_Orig10_Band/best.pth"


def _now() -> datetime:
    return datetime.now()


def _write_state(path: Path, obj: dict) -> None:
    obj = dict(obj)
    obj["updated"] = _now().isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _run(label: str, args: list[str], *, env: dict[str, str] | None = None) -> int:
    full_env = dict(os.environ)
    if env:
        full_env.update(env)
    print(f"\n[RUN] {label}\n  {' '.join(args)}", flush=True)
    rc = subprocess.call([sys.executable, "-u", *args], cwd=str(REPO), env=full_env)
    print(f"[i] {label} exit={rc}", flush=True)
    return int(rc)


def _budget_ok(deadline: datetime, *, need_min: float = 5.0) -> bool:
    left = (deadline - _now()).total_seconds() / 60.0
    print(f"[i] budget remaining ~{left:.0f}m", flush=True)
    return left >= need_min


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _gate_from_eval(path: Path, *, a_mean_f1: float | None = None) -> dict:
    rep = _load_json(path)
    simple = rep.get("simple") or rep
    per = simple.get("per_anchor") or {}
    mean = simple.get("mean") or {}
    gate = gate_compound_eval(
        per,
        mean,
        wall_floor_f1=a_mean_f1,
        focus_anchors=FOCUS,
    )
    gate["path"] = str(path)
    return gate


def _probe(
    *,
    out: Path,
    growth: str,
    anchors: str,
    wall: str,
    label: str,
    route: str = "wall",
    frontier_hops: int = 2,
) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    env = {
        "SPECIES_CONTINUOUS_VEL_DECAY": "1",
        "SPECIES_CONTINUOUS_VEL_DECAY_WALL_ONLY": "1",
    }
    args = [
        str(REPO / "scripts" / "eval_mat_growth_simple.py"),
        "--ckpt",
        wall,
        "--mat-leg",
        "WC_v7_clot_phi_mse",
        "--no-baseline",
        "--out",
        str(out),
        "--anchors",
        anchors,
        "--offwall-ckpt",
        growth,
        "--two-model-route",
        route,
        "--two-model-frontier-hops",
        str(int(frontier_hops)),
    ]
    rc = _run(f"probe {label}", args, env=env)
    if rc != 0:
        raise RuntimeError(f"probe {label} failed rc={rc}")
    return out


def _probe_wall_only(*, out: Path, anchors: str, wall: str, label: str) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    env = {
        "SPECIES_CONTINUOUS_VEL_DECAY": "1",
        "SPECIES_CONTINUOUS_VEL_DECAY_WALL_ONLY": "1",
    }
    args = [
        str(REPO / "scripts" / "eval_mat_growth_simple.py"),
        "--ckpt",
        wall,
        "--mat-leg",
        "WC_v7_clot_phi_mse",
        "--no-baseline",
        "--out",
        str(out),
        "--anchors",
        anchors,
    ]
    rc = _run(f"probe_wall {label}", args, env=env)
    if rc != 0:
        raise RuntimeError(f"wall probe {label} failed rc={rc}")
    return out


def _train(
    *,
    out_ckpt: Path,
    init: str,
    anchors: str,
    val_anchor: str,
    spray_anchors: str,
    epochs: int,
    early_stop: int,
    max_windows: int,
    fn_w: str,
    fp_w: str,
    underpred: str,
    freeze: bool,
    wall_floor_delta: float,
    label: str,
    spray_max_ge2: float = 3.0,
    spray_penalty: float = 0.03,
) -> Path:
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    env = {
        "SPECIES_CONTINUOUS_VEL_DECAY": "1",
        "SPECIES_CONTINUOUS_VEL_DECAY_WALL_ONLY": "1",
        "SPECIES_LUMEN_SHAPE_FN_W": fn_w,
        "SPECIES_LUMEN_SHAPE_FP_W": fp_w,
        "SPECIES_CONTINUOUS_UNDERPRED_WEIGHT": underpred,
    }
    args = [
        "-m",
        "src.training.train_offwall_growth",
        "--val-anchor",
        val_anchor,
        "--anchors",
        anchors,
        "--epochs",
        str(epochs),
        "--early-stop",
        str(early_stop),
        "--max-windows",
        str(max_windows),
        "--hops-k",
        "5",
        "--supervise-mode",
        "frontier_ge2",
        "--frontier-hops",
        "2",
        "--loss-mode",
        "loss_lumen_shape",
        "--lumen-shape-weight",
        "4.0",
        "--ckpt-metric",
        "hop_ge2_target",
        "--train-feat-source",
        "band",
        "--mat-leg",
        "WC_v7_clot_phi_mse",
        "--init",
        init,
        "--out",
        str(out_ckpt),
        "--compound-val",
        "--wall-ckpt",
        WALL,
        "--wall-clot-floor-delta",
        str(wall_floor_delta),
        "--compound-val-every",
        "2",
        "--spray-val-anchors",
        spray_anchors,
        "--spray-val-max-ge2",
        str(spray_max_ge2),
        "--spray-score-penalty",
        str(spray_penalty),
    ]
    if freeze:
        args.append("--freeze-backbone")
    rc = _run(f"train {label}", args, env=env)
    if rc != 0 or not out_ckpt.is_file():
        raise RuntimeError(f"train {label} failed rc={rc} ckpt={out_ckpt}")
    return out_ckpt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deadline-hours", type=float, default=9.0)
    ap.add_argument("--run-root", default="outputs/biochem/offwall_model/wc_v7_wall_lumen_target_9h")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--skip-phase-a", action="store_true")
    ap.add_argument("--skip-phase-b", action="store_true")
    ap.add_argument(
        "--start-phase",
        default="",
        help="Jump to C or D after loading existing Phase A/B artifacts",
    )
    args = ap.parse_args()

    started = _now()
    deadline = started + timedelta(hours=float(args.deadline_hours))
    root = REPO / args.run_root
    root.mkdir(parents=True, exist_ok=True)
    state_path = root / "state.json"
    state: dict = {
        "started": started.isoformat(),
        "deadline": deadline.isoformat(),
        "phase": "start",
        "target_hit": False,
        "best_growth": None,
        "notes": [],
        "phases": {},
    }
    if state_path.is_file() and not args.fresh:
        try:
            state.update(_load_json(state_path))
        except Exception:
            pass
    if args.fresh:
        for p in root.glob("probe_*.json"):
            p.unlink(missing_ok=True)
        for p in (root / "growth_B").glob("*"):
            if p.is_file():
                p.unlink(missing_ok=True)
        for p in (root / "growth_C").glob("*"):
            if p.is_file():
                p.unlink(missing_ok=True)

    print("=" * 72, flush=True)
    print("WALL+LUMEN TARGET 9H (WALL_ONLY contract)", flush=True)
    print(f"[i] start={started.isoformat()} deadline={deadline.isoformat()}", flush=True)
    print(f"[i] out={root}", flush=True)
    print("=" * 72, flush=True)

    # Contract
    os.environ["SPECIES_CONTINUOUS_VEL_DECAY"] = "1"
    os.environ["SPECIES_CONTINUOUS_VEL_DECAY_WALL_ONLY"] = "1"

    wall_ckpt = str((REPO / WALL).resolve())
    prec8h = str((REPO / PREC8H).resolve())
    if not Path(wall_ckpt).is_file():
        raise FileNotFoundError(wall_ckpt)
    if not Path(prec8h).is_file():
        raise FileNotFoundError(prec8h)

    a_mean = None

    # ----- Phase A: Prec8h under WALL_ONLY + wall-alone baseline -----
    skip_a = bool(args.skip_phase_a) or str(args.start_phase or "").strip().upper() in ("C", "D", "E")
    if (not skip_a) and not args.skip_phase_a and _budget_ok(deadline, need_min=40):
        state["phase"] = "A_probe"
        _write_state(state_path, state)
        probe_a = root / "probe_A_wall_alone.json"
        probe_p = root / "probe_A_prec8h_wall_only.json"
        if args.fresh or not probe_a.is_file():
            _probe_wall_only(out=probe_a, anchors=ORIG10, wall=wall_ckpt, label="A_wall")
        if args.fresh or not probe_p.is_file():
            _probe(out=probe_p, growth=prec8h, anchors=ORIG10, wall=wall_ckpt, label="A_prec8h")
        gate_a = _gate_from_eval(probe_a)
        gate_p = _gate_from_eval(probe_p, a_mean_f1=gate_a["mean_f1"])
        a_mean = gate_a["mean_f1"]
        state["phases"]["A"] = {"wall": gate_a, "prec8h": gate_p}
        state["notes"].append(
            f"A: wall F1={gate_a['mean_f1']:.3f} 001={gate_a['focus']['patient001']['ge2_pred']:.0f}; "
            f"prec8h F1={gate_p['mean_f1']:.3f} 001={gate_p['focus']['patient001']['ge2_pred']:.0f} "
            f"007={gate_p['focus']['patient007']['ge2_pred']:.0f} spray002={gate_p['focus']['patient002']['ge2_pred']:.0f}"
        )
        _write_state(state_path, state)
        print(f"[i] Phase A wall target_hit={gate_a['target_hit']} prec8h target_hit={gate_p['target_hit']}", flush=True)
        if gate_p["target_hit"]:
            state["target_hit"] = True
            state["best_growth"] = prec8h
            state["notes"].append("Prec8h under WALL_ONLY already hits target — skip retrain")
            _write_state(state_path, state)
            print("[OK] TARGET HIT on Phase A Prec8h", flush=True)
            return 0
    elif (root / "probe_A_wall_alone.json").is_file():
        a_mean = _gate_from_eval(root / "probe_A_wall_alone.json")["mean_f1"]

    # ----- Phase B: spray-gated precision train -----
    growth_b = root / "growth_B" / "best.pth"
    skip_b = bool(args.skip_phase_b) or str(args.start_phase or "").strip().upper() in ("C", "D", "E")
    if (not skip_b) and _budget_ok(deadline, need_min=90):
        state["phase"] = "B_train_spray_gate"
        _write_state(state_path, state)
        # Budget-aware epochs
        left_h = (deadline - _now()).total_seconds() / 3600.0
        if left_h > 6:
            epochs, es, wins = 10, 4, 36
        elif left_h > 3.5:
            epochs, es, wins = 8, 3, 28
        else:
            epochs, es, wins = 5, 3, 20
        if args.fresh or not growth_b.is_file():
            _train(
                out_ckpt=growth_b,
                init=prec8h,
                anchors=TRAIN_B,
                val_anchor="patient001",
                spray_anchors=SPRAY_NEGS,
                epochs=epochs,
                early_stop=es,
                max_windows=wins,
                fn_w="4",
                fp_w="5",
                underpred="2.5",
                freeze=True,
                wall_floor_delta=0.10,
                label="B_spray_gate",
                spray_max_ge2=3.0,
                spray_penalty=0.03,
            )
        probe_b = root / "probe_B_spray_gate.json"
        if _budget_ok(deadline, need_min=35) and (
            args.fresh or not probe_b.is_file() or "B" not in (state.get("phases") or {})
        ):
            _probe(out=probe_b, growth=str(growth_b), anchors=ORIG10, wall=wall_ckpt, label="B")
            gate_b = _gate_from_eval(probe_b, a_mean_f1=a_mean)
            state["phases"]["B"] = gate_b
            state["notes"].append(
                f"B: F1={gate_b['mean_f1']:.3f} 001={gate_b['focus']['patient001']['ge2_pred']:.0f} "
                f"007={gate_b['focus']['patient007']['ge2_pred']:.0f} "
                f"spray002={gate_b['focus']['patient002']['ge2_pred']:.0f} "
                f"target={gate_b['target_hit']}"
            )
            if gate_b["target_hit"]:
                state["target_hit"] = True
                state["best_growth"] = str(growth_b)
                _write_state(state_path, state)
                print("[OK] TARGET HIT on Phase B", flush=True)
                return 0
            state["best_growth"] = str(growth_b)
            _write_state(state_path, state)
        elif probe_b.is_file() and "B" not in (state.get("phases") or {}):
            gate_b = _gate_from_eval(probe_b, a_mean_f1=a_mean)
            state["phases"]["B"] = gate_b
            state["best_growth"] = str(growth_b) if growth_b.is_file() else state.get("best_growth")
            _write_state(state_path, state)

    # ----- Phase C pivots -----
    # EDA: Prec8h under WALL_ONLY sprays (legacy decay was FP filter). Phase B hard
    # spray_max=3 rejected all saves. Pivot: extreme FP + soft spray curriculum from Prec8h.
    growth_c = root / "growth_C" / "best.pth"
    gate_b = (state.get("phases") or {}).get("B") or {}
    if not gate_b and (root / "probe_B_spray_gate.json").is_file():
        gate_b = _gate_from_eval(root / "probe_B_spray_gate.json", a_mean_f1=a_mean)
        state["phases"]["B"] = gate_b
    need_007 = not bool((gate_b.get("gates") or {}).get("lumen_teachers_open", False))
    need_spray = not bool(gate_b.get("spray_clean", False))
    need_001 = not bool((gate_b.get("focus") or {}).get("patient001", {}).get("ge2_pred", 0) > 0.5)
    b_no_valid_ckpt = float((gate_b or {}).get("mean_f1") or 0) < 0.70 or need_spray

    start_phase = str(args.start_phase or "").strip().upper()
    run_c = (not start_phase or start_phase == "C") and _budget_ok(deadline, need_min=80) and not state.get(
        "target_hit"
    )
    if run_c and start_phase != "E" and start_phase != "D":
        state["phase"] = "C_pivot"
        _write_state(state_path, state)
        # Prefer Prec8h if B never saved a spray-clean / wall-floor-valid ckpt
        init_c = prec8h if b_no_valid_ckpt else (str(growth_b) if growth_b.is_file() else prec8h)
        # Pivot logic from evidence
        if need_spray and not need_001:
            # Opened lumen but sprays — extreme FP; soft spray gate so ckpt can save
            fn_w, fp_w, under, freeze, anchors, val = "2", "12", "1.5", True, TRAIN_B, "patient001"
            spray_max, spray_pen = 40.0, 0.05
            note = "C: extreme FP + soft spray curriculum (EDA: WALL_ONLY lost decay FP filter)"
        elif need_001 and need_007:
            fn_w, fp_w, under, freeze, anchors, val = "6", "4", "3.0", True, TRAIN_B, "patient007"
            spray_max, spray_pen = 25.0, 0.04
            note = "C: recall tilt on 007 val"
        elif need_007 and not need_001:
            fn_w, fp_w, under, freeze, anchors, val = "5", "8", "2.5", False, TRAIN_B, "patient007"
            spray_max, spray_pen = 30.0, 0.04
            note = "C: unfreeze for 007 transfer + FP"
        else:
            fn_w, fp_w, under, freeze, anchors, val = "3", "10", "2.0", True, TRAIN_B, "patient001"
            spray_max, spray_pen = 20.0, 0.05
            note = "C: default FP polish"
        state["notes"].append(note)
        left_h = (deadline - _now()).total_seconds() / 3600.0
        epochs = 8 if left_h > 3 else 5
        if args.fresh or not growth_c.is_file():
            _train(
                out_ckpt=growth_c,
                init=init_c,
                anchors=anchors,
                val_anchor=val,
                spray_anchors=SPRAY_NEGS,
                epochs=epochs,
                early_stop=max(3, epochs // 2),
                max_windows=28 if left_h > 3 else 20,
                fn_w=fn_w,
                fp_w=fp_w,
                underpred=under,
                freeze=freeze,
                wall_floor_delta=0.10,
                label="C_pivot",
                spray_max_ge2=spray_max,
                spray_penalty=spray_pen,
            )
        if _budget_ok(deadline, need_min=35):
            probe_c = root / "probe_C_pivot.json"
            _probe(out=probe_c, growth=str(growth_c), anchors=ORIG10, wall=wall_ckpt, label="C")
            gate_c = _gate_from_eval(probe_c, a_mean_f1=a_mean)
            state["phases"]["C"] = gate_c
            state["notes"].append(
                f"C: F1={gate_c['mean_f1']:.3f} 001={gate_c['focus']['patient001']['ge2_pred']:.0f} "
                f"007={gate_c['focus']['patient007']['ge2_pred']:.0f} "
                f"spray002={gate_c['focus']['patient002']['ge2_pred']:.0f} "
                f"target={gate_c['target_hit']}"
            )
            if gate_c["target_hit"] or gate_c["mean_f1"] > float((gate_b or {}).get("mean_f1") or 0):
                state["best_growth"] = str(growth_c)
            if gate_c["target_hit"]:
                state["target_hit"] = True
                _write_state(state_path, state)
                print("[OK] TARGET HIT on Phase C", flush=True)
                return 0
            _write_state(state_path, state)

    # ----- Phase D: if still failing and budget, EDA-informed last try (D_Orig10 init + heavy FP) -----
    growth_d = root / "growth_D" / "best.pth"
    if _budget_ok(deadline, need_min=70) and not state.get("target_hit"):
        state["phase"] = "D_eda_pivot"
        d_init = str((REPO / D_ORIG).resolve())
        if Path(d_init).is_file():
            state["notes"].append("D: D_Orig10 capacity + extreme FP spray gate (EDA: wipe was FP filter)")
            _write_state(state_path, state)
            if args.fresh or not growth_d.is_file():
                _train(
                    out_ckpt=growth_d,
                    init=d_init,
                    anchors=TRAIN_B,
                    val_anchor="patient001",
                    spray_anchors=SPRAY_NEGS,
                    epochs=6,
                    early_stop=3,
                    max_windows=24,
                    fn_w="2",
                    fp_w="16",
                    underpred="1.5",
                    freeze=True,
                    wall_floor_delta=0.12,
                    label="D_eda",
                    spray_max_ge2=25.0,
                    spray_penalty=0.06,
                )
            if _budget_ok(deadline, need_min=35):
                probe_d = root / "probe_D_eda.json"
                _probe(out=probe_d, growth=str(growth_d), anchors=ORIG10, wall=wall_ckpt, label="D")
                gate_d = _gate_from_eval(probe_d, a_mean_f1=a_mean)
                state["phases"]["D"] = gate_d
                if gate_d["target_hit"]:
                    state["target_hit"] = True
                    state["best_growth"] = str(growth_d)
                elif gate_d["mean_f1"] > float(((state.get("phases") or {}).get("C") or {}).get("mean_f1") or 0):
                    state["best_growth"] = str(growth_d)
                state["notes"].append(
                    f"D: F1={gate_d['mean_f1']:.3f} target={gate_d['target_hit']}"
                )
                _write_state(state_path, state)

    # ----- Phase E: frontier-route deploy (EDA: speed cannot separate 002 spray from 001 GT;
    # wall-route gives specialist all lumen -> spray. Frontier limits growth to committed nbhd.) -----
    if _budget_ok(deadline, need_min=40) and not state.get("target_hit"):
        state["phase"] = "E_frontier_route"
        best = state.get("best_growth") or (str(growth_c) if growth_c.is_file() else prec8h)
        state["notes"].append(
            "E: frontier-route re-eval (EDA: spray idle speed overlaps 001 GT; "
            "wall-route specialist owns all lumen)"
        )
        _write_state(state_path, state)
        probe_e = root / "probe_E_frontier.json"
        if args.fresh or not probe_e.is_file():
            _probe(
                out=probe_e,
                growth=str(best),
                anchors=ORIG10,
                wall=wall_ckpt,
                label="E_frontier",
                route="frontier",
                frontier_hops=2,
            )
        gate_e = _gate_from_eval(probe_e, a_mean_f1=a_mean)
        state["phases"]["E"] = gate_e
        state["notes"].append(
            f"E frontier: F1={gate_e['mean_f1']:.3f} 001={gate_e['focus']['patient001']['ge2_pred']:.0f} "
            f"007={gate_e['focus']['patient007']['ge2_pred']:.0f} "
            f"spray002={gate_e['focus']['patient002']['ge2_pred']:.0f} "
            f"target={gate_e['target_hit']}"
        )
        if gate_e["target_hit"] or (
            gate_e["spray_clean"]
            and gate_e["mean_f1"] >= float(((state.get("phases") or {}).get("C") or {}).get("mean_f1") or 0)
        ):
            state["best_growth"] = str(best)
            state["best_route"] = "frontier"
        if gate_e["target_hit"]:
            state["target_hit"] = True
            _write_state(state_path, state)
            print("[OK] TARGET HIT on Phase E frontier route", flush=True)
            return 0
        _write_state(state_path, state)

        # If frontier cleans spray but loses lumen, short harden from C with soft FP keep
        if (
            gate_e.get("spray_clean")
            and not gate_e["target_hit"]
            and _budget_ok(deadline, need_min=90)
        ):
            growth_e = root / "growth_E_frontier" / "best.pth"
            state["notes"].append("E2: short FP polish under wall-route train; deploy will prefer frontier")
            if args.fresh or not growth_e.is_file():
                _train(
                    out_ckpt=growth_e,
                    init=str(best),
                    anchors=TRAIN_B,
                    val_anchor="patient001",
                    spray_anchors=SPRAY_NEGS,
                    epochs=5,
                    early_stop=3,
                    max_windows=20,
                    fn_w="3",
                    fp_w="14",
                    underpred="1.5",
                    freeze=True,
                    wall_floor_delta=0.10,
                    label="E2_frontier_polish",
                    spray_max_ge2=15.0,
                    spray_penalty=0.06,
                )
            probe_e2 = root / "probe_E2_frontier.json"
            _probe(
                out=probe_e2,
                growth=str(growth_e),
                anchors=ORIG10,
                wall=wall_ckpt,
                label="E2_frontier",
                route="frontier",
                frontier_hops=2,
            )
            gate_e2 = _gate_from_eval(probe_e2, a_mean_f1=a_mean)
            state["phases"]["E2"] = gate_e2
            state["notes"].append(
                f"E2 frontier: F1={gate_e2['mean_f1']:.3f} target={gate_e2['target_hit']}"
            )
            if gate_e2["target_hit"]:
                state["target_hit"] = True
                state["best_growth"] = str(growth_e)
                state["best_route"] = "frontier"
                _write_state(state_path, state)
                print("[OK] TARGET HIT on Phase E2", flush=True)
                return 0
            if gate_e2["mean_f1"] > float(gate_e.get("mean_f1") or 0):
                state["best_growth"] = str(growth_e)
                state["best_route"] = "frontier"
            _write_state(state_path, state)

    # ----- Final summary -----
    state["phase"] = "done"
    state["elapsed_h"] = (_now() - started).total_seconds() / 3600.0
    best = state.get("best_growth")
    summary = {
        "target_hit": state.get("target_hit"),
        "best_growth": best,
        "best_route": state.get("best_route", "wall"),
        "phases": state.get("phases"),
        "notes": state.get("notes"),
        "elapsed_h": state.get("elapsed_h"),
    }
    (root / "final_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_state(state_path, state)
    print("=" * 72, flush=True)
    print(f"[i] DONE target_hit={state.get('target_hit')} best={best} route={state.get('best_route', 'wall')}", flush=True)
    for n in state.get("notes") or []:
        print(f"  - {n}", flush=True)
    print(f"[save] {root / 'final_summary.json'}", flush=True)
    return 0 if state.get("target_hit") else 2


if __name__ == "__main__":
    raise SystemExit(main())
