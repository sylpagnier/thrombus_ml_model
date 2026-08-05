"""WC v8 compound improvement sweeps (post-promote polish / retrain axes).

Axes wired from MAT_GROWTH next-levers:
  1. hops_sweep     - eval-time frontier hops 0 / 0.5 / 1 / 2 (+ per-vessel map)
  2. fp_polish_010  - short FP polish, frontier-h1 compound-val, compound_primary ckpt
  3. frontier_h1    - retrain with compound-val matching deploy (frontier route hops=1)
  4. recall_007     - 007 val anchor, spray-gate 010/006 (no 010 over-fire)
  5. unfreeze       - 1-2 epoch partial backbone unfreeze with strong wall floor

Usage:
  python -u scripts/run_wc_v8_improvement_sweeps.py
  python -u scripts/run_wc_v8_improvement_sweeps.py --only hops_sweep,fp_polish_010
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
WALL = "outputs/biochem/biochem_gnn/locked/species_gnn_best.pth"
GROWTH_V8 = "outputs/biochem/biochem_gnn/locked/compound_growth_best.pth"
GROWTH_C = "outputs/biochem/offwall_model/wc_v7_wall_lumen_target_9h/growth_C/best.pth"
WALL_FLOOR_JSON = "outputs/biochem/offwall_model/wc_v7_wall_lumen_target_9h/probe_A_wall_alone.json"

TEACHERS_NO_010 = "patient001,patient006,patient007"
SPRAY_ALL = "patient002,patient006,patient008,patient010"
SPRAY_OVERFIRE = "patient006,patient010"

from src.evaluation.compound_deploy_gates import (  # noqa: E402
    DEFAULT_FOCUS_ANCHORS as FOCUS,
    format_gate_summary,
    gate_compound_eval,
)


def _now() -> datetime:
    return datetime.now()


def _write_state(path: Path, obj: dict) -> None:
    obj = dict(obj)
    obj["updated"] = _now().isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _run(label: str, args: list[str], *, env: dict[str, str] | None = None) -> int:
    full_env = dict(os.environ)
    full_env["SPECIES_CONTINUOUS_VEL_DECAY"] = "1"
    full_env["SPECIES_CONTINUOUS_VEL_DECAY_WALL_ONLY"] = "1"
    if env:
        full_env.update(env)
    print(f"\n[RUN] {label}\n  {' '.join(args)}", flush=True)
    rc = subprocess.call([sys.executable, "-u", *args], cwd=str(REPO), env=full_env)
    print(f"[i] {label} exit={rc}", flush=True)
    return int(rc)


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
    gate["compound_gates"] = rep.get("compound_gates")
    return gate


def _probe(
    *,
    out: Path,
    growth: str,
    wall: str,
    label: str,
    route: str = "frontier",
    frontier_hops: float = 1.0,
    hops_map: str = "",
) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
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
        ORIG10,
        "--offwall-ckpt",
        growth,
        "--two-model-route",
        route,
        "--two-model-frontier-hops",
        str(frontier_hops),
        "--wall-floor-json",
        str(REPO / WALL_FLOOR_JSON),
    ]
    if hops_map.strip():
        args.extend(["--two-model-frontier-hops-map", hops_map.strip()])
    rc = _run(f"probe {label}", args)
    if rc != 0:
        raise RuntimeError(f"probe {label} failed rc={rc}")
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
    ckpt_metric: str = "compound_primary_spray",
    compound_val_route: str = "frontier",
    compound_val_frontier_hops: float = 1.0,
    spray_max_ge2: float = 8.0,
    spray_penalty: float = 0.04,
    backbone_unfreeze_after: int = 0,
    backbone_unfreeze_epochs: int = 0,
) -> Path:
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    env = {
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
        ckpt_metric,
        "--train-feat-source",
        "band",
        "--mat-leg",
        "WC_v7_clot_phi_mse",
        "--init",
        init,
        "--out",
        str(out_ckpt),
        "--compound-val",
        "--compound-val-route",
        compound_val_route,
        "--compound-val-frontier-hops",
        str(compound_val_frontier_hops),
        "--wall-ckpt",
        WALL,
        "--wall-clot-floor-delta",
        str(wall_floor_delta),
        "--compound-val-every",
        "1",
        "--spray-val-anchors",
        spray_anchors,
        "--spray-val-max-ge2",
        str(spray_max_ge2),
        "--spray-score-penalty",
        str(spray_penalty),
    ]
    if freeze:
        args.append("--freeze-backbone")
    if backbone_unfreeze_after > 0 and backbone_unfreeze_epochs > 0:
        args.extend(
            [
                "--backbone-unfreeze-after",
                str(backbone_unfreeze_after),
                "--backbone-unfreeze-epochs",
                str(backbone_unfreeze_epochs),
            ]
        )
    rc = _run(f"train {label}", args, env=env)
    if rc != 0 or not out_ckpt.is_file():
        raise RuntimeError(f"train {label} failed rc={rc} ckpt={out_ckpt}")
    return out_ckpt


def _budget_ok(deadline: datetime, *, need_min: float = 5.0) -> bool:
    left = (deadline - _now()).total_seconds() / 60.0
    print(f"[i] budget remaining ~{left:.0f}m", flush=True)
    return left >= need_min


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deadline-hours", type=float, default=12.0)
    ap.add_argument(
        "--run-root",
        default="outputs/biochem/offwall_model/wc_v8_improvement_sweeps",
    )
    ap.add_argument("--init-growth", default=GROWTH_V8)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument(
        "--only",
        default="",
        help="Comma subset: hops_sweep,fp_polish_010,frontier_h1,recall_007,unfreeze",
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
        "best_growth": None,
        "notes": [],
        "phases": {},
    }
    if state_path.is_file() and not args.fresh:
        try:
            state.update(_load_json(state_path))
        except Exception:
            pass

    only = {x.strip() for x in args.only.split(",") if x.strip()}
    run_all = not only

    wall_ckpt = str((REPO / WALL).resolve())
    init_growth = str((REPO / args.init_growth).resolve())
    if not Path(init_growth).is_file():
        init_growth = str((REPO / GROWTH_C).resolve())
    a_mean = None
    if (REPO / WALL_FLOOR_JSON).is_file():
        a_mean = _gate_from_eval(REPO / WALL_FLOOR_JSON)["mean_f1"]

    print("=" * 72, flush=True)
    print("WC V8 IMPROVEMENT SWEEPS", flush=True)
    print(f"[i] init_growth={init_growth}", flush=True)
    print(f"[i] out={root}", flush=True)
    print("=" * 72, flush=True)

    best_growth = state.get("best_growth") or init_growth
    best_score = float(((state.get("phases") or {}).get("baseline") or {}).get("mean_score") or 0.0)

    # Baseline probe (frontier-h1 deploy contract)
    if run_all or "baseline" in only:
        probe_base = root / "probe_baseline_v8_frontier_h1.json"
        if args.fresh or not probe_base.is_file():
            _probe(
                out=probe_base,
                growth=init_growth,
                wall=wall_ckpt,
                label="baseline_v8",
                frontier_hops=1.0,
            )
        gate_base = _gate_from_eval(probe_base, a_mean_f1=a_mean)
        state["phases"]["baseline"] = gate_base
        best_score = float(gate_base.get("mean_score") or best_score)
        state["notes"].append(
            f"baseline: F1={gate_base['mean_f1']:.3f} score={gate_base['mean_score']:.3f} "
            f"relaxed={gate_base['mean_offwall_relaxed_f1']:.3f} "
            f"010_ge2={gate_base['focus']['patient010']['ge2_pred']:.0f}/"
            f"{gate_base['focus']['patient010']['ge2_gt']:.0f}"
        )
        _write_state(state_path, state)

    # 1. Hops sweep (eval only)
    if (run_all or "hops_sweep" in only) and _budget_ok(deadline, need_min=25):
        state["phase"] = "hops_sweep"
        _write_state(state_path, state)
        hops_dir = root / "hops_sweep"
        rc = _run(
            "hops_sweep",
            [
                str(REPO / "scripts" / "sweep_frontier_hops.py"),
                "--growth",
                best_growth,
                "--out-dir",
                str(hops_dir),
            ],
        )
        if rc == 0 and (hops_dir / "hops_sweep_summary.json").is_file():
            summary = _load_json(hops_dir / "hops_sweep_summary.json")
            state["phases"]["hops_sweep"] = summary
            state["notes"].append("[OK] hops sweep complete")
            _write_state(state_path, state)

    # 2. FP polish (010 focus) under frontier-h1 deploy contract
    if (run_all or "fp_polish_010" in only) and _budget_ok(deadline, need_min=90):
        state["phase"] = "fp_polish_010"
        _write_state(state_path, state)
        ckpt = root / "growth_fp_polish_010" / "best.pth"
        if args.fresh or not ckpt.is_file():
            _train(
                out_ckpt=ckpt,
                init=best_growth,
                anchors=f"{TEACHERS_NO_010},patient010,{SPRAY_ALL}",
                val_anchor="patient001",
                spray_anchors=SPRAY_OVERFIRE,
                epochs=4,
                early_stop=2,
                max_windows=20,
                fn_w="3",
                fp_w="14",
                underpred="1.5",
                freeze=True,
                wall_floor_delta=0.12,
                label="fp_polish_010",
                ckpt_metric="compound_primary_spray",
                compound_val_route="frontier",
                compound_val_frontier_hops=1.0,
                spray_max_ge2=15.0,
                spray_penalty=0.05,
            )
        probe = root / "probe_fp_polish_010.json"
        _probe(out=probe, growth=str(ckpt), wall=wall_ckpt, label="fp_polish_010")
        gate = _gate_from_eval(probe, a_mean_f1=a_mean)
        state["phases"]["fp_polish_010"] = gate
        if float(gate.get("mean_score") or 0) >= best_score:
            best_growth = str(ckpt)
            best_score = float(gate.get("mean_score") or best_score)
            state["best_growth"] = best_growth
        state["notes"].append(
            f"fp_polish_010: F1={gate['mean_f1']:.3f} score={gate['mean_score']:.3f} "
            f"010_ge2={gate['focus']['patient010']['ge2_pred']:.0f} "
            f"spray_clean={gate['spray_clean']}"
        )
        _write_state(state_path, state)

    # 3. Train under frontier-h1 deploy contract (compound-val matches deploy)
    if (run_all or "frontier_h1" in only) and _budget_ok(deadline, need_min=120):
        state["phase"] = "frontier_h1_retrain"
        _write_state(state_path, state)
        ckpt = root / "growth_frontier_h1_retrain" / "best.pth"
        if args.fresh or not ckpt.is_file():
            _train(
                out_ckpt=ckpt,
                init=best_growth,
                anchors=f"{TEACHERS_NO_010},patient010,{SPRAY_ALL}",
                val_anchor="patient001",
                spray_anchors=SPRAY_OVERFIRE,
                epochs=8,
                early_stop=3,
                max_windows=28,
                fn_w="4",
                fp_w="10",
                underpred="2.0",
                freeze=True,
                wall_floor_delta=0.12,
                label="frontier_h1_retrain",
                ckpt_metric="compound_primary_spray",
                compound_val_route="frontier",
                compound_val_frontier_hops=1.0,
                spray_max_ge2=12.0,
                spray_penalty=0.04,
            )
        probe = root / "probe_frontier_h1_retrain.json"
        _probe(out=probe, growth=str(ckpt), wall=wall_ckpt, label="frontier_h1_retrain")
        gate = _gate_from_eval(probe, a_mean_f1=a_mean)
        state["phases"]["frontier_h1_retrain"] = gate
        if float(gate.get("mean_score") or 0) >= best_score:
            best_growth = str(ckpt)
            best_score = float(gate.get("mean_score") or best_score)
            state["best_growth"] = best_growth
        state["notes"].append(
            f"frontier_h1_retrain: F1={gate['mean_f1']:.3f} score={gate['mean_score']:.3f} "
            f"relaxed={gate['mean_offwall_relaxed_f1']:.3f}"
        )
        _write_state(state_path, state)

    # 4. 007 recall without 010 spray (separate val anchor + spray-gate 010/006)
    if (run_all or "recall_007" in only) and _budget_ok(deadline, need_min=90):
        state["phase"] = "recall_007"
        _write_state(state_path, state)
        ckpt = root / "growth_recall_007" / "best.pth"
        if args.fresh or not ckpt.is_file():
            _train(
                out_ckpt=ckpt,
                init=best_growth,
                anchors=f"{TEACHERS_NO_010},patient002,patient008",
                val_anchor="patient007",
                spray_anchors=SPRAY_OVERFIRE,
                epochs=6,
                early_stop=3,
                max_windows=24,
                fn_w="6",
                fp_w="6",
                underpred="3.5",
                freeze=True,
                wall_floor_delta=0.12,
                label="recall_007",
                ckpt_metric="compound_primary_spray",
                compound_val_route="frontier",
                compound_val_frontier_hops=1.0,
                spray_max_ge2=10.0,
                spray_penalty=0.05,
            )
        probe = root / "probe_recall_007.json"
        _probe(out=probe, growth=str(ckpt), wall=wall_ckpt, label="recall_007")
        gate = _gate_from_eval(probe, a_mean_f1=a_mean)
        state["phases"]["recall_007"] = gate
        ge2_007 = gate["focus"]["patient007"]["ge2_pred"]
        ge2_gt_007 = gate["focus"]["patient007"]["ge2_gt"]
        if float(gate.get("mean_score") or 0) >= best_score:
            best_growth = str(ckpt)
            best_score = float(gate.get("mean_score") or best_score)
            state["best_growth"] = best_growth
        state["notes"].append(
            f"recall_007: F1={gate['mean_f1']:.3f} 007_ge2={ge2_007:.0f}/{ge2_gt_007:.0f} "
            f"010_ge2={gate['focus']['patient010']['ge2_pred']:.0f}"
        )
        _write_state(state_path, state)

    # 5. Partial backbone unfreeze (strong wall floor)
    if (run_all or "unfreeze" in only) and _budget_ok(deadline, need_min=90):
        state["phase"] = "unfreeze_polish"
        _write_state(state_path, state)
        ckpt = root / "growth_unfreeze_polish" / "best.pth"
        if args.fresh or not ckpt.is_file():
            _train(
                out_ckpt=ckpt,
                init=best_growth,
                anchors=f"{TEACHERS_NO_010},patient010,{SPRAY_ALL}",
                val_anchor="patient001",
                spray_anchors=SPRAY_OVERFIRE,
                epochs=6,
                early_stop=3,
                max_windows=24,
                fn_w="4",
                fp_w="8",
                underpred="2.5",
                freeze=True,
                wall_floor_delta=0.12,
                label="unfreeze_polish",
                ckpt_metric="compound_primary_spray",
                compound_val_route="frontier",
                compound_val_frontier_hops=1.0,
                spray_max_ge2=12.0,
                spray_penalty=0.04,
                backbone_unfreeze_after=3,
                backbone_unfreeze_epochs=2,
            )
        probe = root / "probe_unfreeze_polish.json"
        _probe(out=probe, growth=str(ckpt), wall=wall_ckpt, label="unfreeze_polish")
        gate = _gate_from_eval(probe, a_mean_f1=a_mean)
        state["phases"]["unfreeze_polish"] = gate
        if float(gate.get("mean_score") or 0) >= best_score:
            best_growth = str(ckpt)
            best_score = float(gate.get("mean_score") or best_score)
            state["best_growth"] = best_growth
        state["notes"].append(
            f"unfreeze_polish: F1={gate['mean_f1']:.3f} score={gate['mean_score']:.3f}"
        )
        _write_state(state_path, state)

    state["phase"] = "done"
    state["elapsed_h"] = (_now() - started).total_seconds() / 3600.0
    state["best_growth"] = best_growth
    summary = {
        "best_growth": best_growth,
        "best_score": best_score,
        "phases": state.get("phases"),
        "notes": state.get("notes"),
        "elapsed_h": state.get("elapsed_h"),
    }
    (root / "final_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_state(state_path, state)

    print("=" * 72, flush=True)
    print(f"[i] DONE best_growth={best_growth} best_score={best_score:.3f}", flush=True)
    for n in state.get("notes") or []:
        print(f"  - {n}", flush=True)
    print(f"[save] {root / 'final_summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
