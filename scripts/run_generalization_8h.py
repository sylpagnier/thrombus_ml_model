"""8h clean generalization run: cold growth specialist, sealed challenge 009/032.

Design (see plan 8h Gen Clean Run):
  - Growth: --no-init (never wall / C0 growth warm-start)
  - Wall partner: locked WC_v7 at compound eval only
  - Train includes 007 (shape teacher); val disjoint; challenge sealed
  - Loss: frontier_ge2 + loss_lumen_shape + underpred tilt
  - Ckpt: compound_primary_spray under frontier_offwall h0.5

Usage:
  python -u scripts/run_generalization_8h.py
  python -u scripts/run_generalization_8h.py --smoke
  python -u scripts/run_generalization_8h.py --fresh
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

WALL = "outputs/biochem/biochem_gnn/locked/species_gnn_best.pth"
GROWTH_BASELINE = "outputs/biochem/biochem_gnn/locked/compound_growth_best.pth"

# Sealed split (train ∩ val ∩ challenge = empty)
TRAIN = [
    "patient001",
    "patient002",
    "patient005",
    "patient006",
    "patient007",
    "patient008",
    "patient010",
    "patient011",
    "patient013",
    "patient014",
    "patient016",
    "patient020",
    "patient024",
    "patient025",
    "patient028",
    "patient029",
]
VAL = [
    "patient004",
    "patient015",
    "patient018",
    "patient019",
    "patient021",
    "patient031",
    "patient035",
    "patient036",
]
CHALLENGE = ["patient009", "patient032"]
SPRAY = ["patient002", "patient008"]
PRIMARY_VAL = "patient015"


def _now() -> datetime:
    return datetime.now()


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _assert_disjoint(train: list[str], val: list[str], challenge: list[str]) -> None:
    t, v, c = set(train), set(val), set(challenge)
    bad = []
    if t & v:
        bad.append(f"train∩val={sorted(t & v)}")
    if t & c:
        bad.append(f"train∩challenge={sorted(t & c)}")
    if v & c:
        bad.append(f"val∩challenge={sorted(v & c)}")
    if bad:
        raise RuntimeError("split leakage: " + "; ".join(bad))


def _run(label: str, args: list[str], *, env: dict[str, str] | None = None) -> int:
    full_env = dict(os.environ)
    full_env["SPECIES_CONTINUOUS_VEL_DECAY"] = "1"
    full_env["SPECIES_CONTINUOUS_VEL_DECAY_WALL_ONLY"] = "1"
    if env:
        full_env.update(env)
    print(f"\n[RUN] {label}\n  {' '.join(args)}", flush=True)
    rc = subprocess.call([sys.executable, "-u", *args], cwd=str(REPO), env=full_env)
    print(f"[i] {label} rc={rc}", flush=True)
    return int(rc)


def _train_cmd(
    *,
    out_ckpt: Path,
    train_csv: str,
    val_csv: str,
    spray_csv: str,
    wall_ckpt: str,
    epochs: int,
    early_stop: int,
    max_windows: int,
    lr: float | None,
    no_init: bool,
    init: str | None,
    compound_val_every: int,
) -> list[str]:
    a = [
        "-m",
        "src.training.train_offwall_growth",
        "--anchors",
        train_csv,
        "--val-anchors",
        val_csv,
        "--val-anchor",
        PRIMARY_VAL,
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
        "compound_primary_spray",
        "--train-feat-source",
        "band",
        "--mat-leg",
        "WC_v7_clot_phi_mse",
        "--compound-val",
        "--compound-val-route",
        "frontier_offwall",
        "--compound-val-frontier-hops",
        "0.5",
        "--wall-ckpt",
        wall_ckpt,
        "--wall-clot-floor-delta",
        "0.10",
        "--compound-val-every",
        str(compound_val_every),
        "--spray-val-anchors",
        spray_csv,
        "--spray-val-max-ge2",
        "8",
        "--spray-score-penalty",
        "0.05",
        "--out",
        str(out_ckpt),
    ]
    if no_init:
        a.append("--no-init")
    elif init:
        a.extend(["--init", init])
    if lr is not None:
        a.extend(["--lr", str(lr)])
    return a


def _eval_cmd(
    *,
    out: Path,
    anchors: str,
    wall: str,
    growth: str,
) -> list[str]:
    return [
        "scripts/eval_mat_growth_simple.py",
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
        "frontier_offwall",
        "--two-model-frontier-hops",
        "0.5",
    ]


def _mean_from_eval(path: Path) -> dict:
    if not path.is_file():
        return {}
    rep = json.loads(path.read_text(encoding="utf-8"))
    simple = rep.get("simple") or rep
    return {
        "mean": simple.get("mean") or {},
        "per_anchor": simple.get("per_anchor") or {},
        "path": str(path),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="8h clean generalization growth retrain")
    ap.add_argument(
        "--run-root",
        default="outputs/biochem/offwall_model/generalization_8h",
    )
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="1-epoch tiny smoke (2 train / 2 val) then exit",
    )
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--deadline-hours", type=float, default=8.0)
    ap.add_argument(
        "--init-mode",
        choices=("none", "wall"),
        default="none",
        help="none=cold growth; wall=warm-start from locked wall backbone (pivot)",
    )
    ap.add_argument(
        "--val-fast",
        action="store_true",
        help="Use 4-anchor compound-val subset for faster ckpt (full VAL still used at final eval)",
    )
    ap.add_argument("--stage1-epochs", type=int, default=14)
    ap.add_argument("--stage2-epochs", type=int, default=4)
    ap.add_argument("--max-windows", type=int, default=40)
    ap.add_argument("--fn-w", default="5")
    ap.add_argument("--fp-w", default="2.5")
    ap.add_argument("--underpred", default="3.0")
    ap.add_argument(
        "--tag",
        default="",
        help="Optional run tag for stage dirs (e.g. pivot_wall)",
    )
    args = ap.parse_args()

    train_list = list(TRAIN)
    val_list = list(VAL)
    if args.val_fast:
        # Higher-signal / faster compound-val subset (still disjoint from challenge).
        val_list = ["patient015", "patient021", "patient035", "patient004"]
    _assert_disjoint(train_list, val_list, CHALLENGE)
    for s in SPRAY:
        if s not in train_list:
            raise RuntimeError(f"spray probe {s} must be in TRAIN")

    root = REPO / args.run_root
    root.mkdir(parents=True, exist_ok=True)
    started = _now()
    deadline = started + timedelta(hours=float(args.deadline_hours))
    state_path = root / "state.json"
    tag = (args.tag or args.init_mode).strip() or "run"

    splits = {
        "train_anchors": train_list,
        "val_anchors": val_list,
        "val_full_for_final_eval": VAL,
        "challenge_anchors": CHALLENGE,
        "spray_anchors": SPRAY,
        "primary_val_anchor": PRIMARY_VAL,
        "init_mode": args.init_mode,
        "route": "frontier_offwall",
        "frontier_hops": 0.5,
        "tag": tag,
        "notes": (
            f"init_mode={args.init_mode}; locked wall is eval partner. "
            "007 in train; 009/032 sealed challenge."
        ),
    }
    _write_json(root / "splits.json", splits)

    wall_ckpt = str((REPO / WALL).resolve())
    baseline_growth = str((REPO / GROWTH_BASELINE).resolve())
    if not Path(wall_ckpt).is_file():
        raise FileNotFoundError(wall_ckpt)

    common_env = {
        "SPECIES_LUMEN_SHAPE_FN_W": str(args.fn_w),
        "SPECIES_LUMEN_SHAPE_FP_W": str(args.fp_w),
        "SPECIES_CONTINUOUS_UNDERPRED_WEIGHT": str(args.underpred),
    }

    state = {
        "started": started.isoformat(),
        "deadline": deadline.isoformat(),
        "phase": "start",
        "notes": [],
        "init_mode": args.init_mode,
        "tag": tag,
    }
    if state_path.is_file() and not args.fresh:
        try:
            state.update(json.loads(state_path.read_text(encoding="utf-8")))
        except Exception:
            pass
    _write_json(state_path, state)

    print("=" * 72, flush=True)
    print(
        f"GENERALIZATION 8H (init={args.init_mode}, sealed 009/032, tag={tag})",
        flush=True,
    )
    print(f"[i] out={root}", flush=True)
    print(
        f"[i] train={len(train_list)} val={len(val_list)} challenge={CHALLENGE}",
        flush=True,
    )
    print("=" * 72, flush=True)

    if args.smoke:
        smoke_dir = root / "smoke"
        smoke_ckpt = smoke_dir / "best.pth"
        state["phase"] = "smoke"
        _write_json(state_path, state)
        rc = _run(
            "smoke_train",
            _train_cmd(
                out_ckpt=smoke_ckpt,
                train_csv="patient007,patient002",
                val_csv="patient015,patient004",
                spray_csv="patient002",
                wall_ckpt=wall_ckpt,
                epochs=1,
                early_stop=1,
                max_windows=4,
                lr=3e-4,
                no_init=(args.init_mode == "none"),
                init=None if args.init_mode == "none" else wall_ckpt,
                compound_val_every=1,
            ),
            env=common_env,
        )
        state["notes"].append(f"smoke rc={rc} ckpt={smoke_ckpt.is_file()}")
        _write_json(state_path, state)
        return rc

    stage1 = root / f"growth_stage1_{tag}" / "best.pth"
    stage2 = root / f"growth_stage2_{tag}" / "best.pth"
    train_csv = ",".join(train_list)
    val_csv = ",".join(val_list)
    spray_csv = ",".join(SPRAY)
    no_init = args.init_mode == "none"
    stage1_init = None if no_init else wall_ckpt

    if not args.skip_train:
        if args.fresh or not stage1.is_file():
            state["phase"] = "stage1"
            _write_json(state_path, state)
            rc = _run(
                "stage1",
                _train_cmd(
                    out_ckpt=stage1,
                    train_csv=train_csv,
                    val_csv=val_csv,
                    spray_csv=spray_csv,
                    wall_ckpt=wall_ckpt,
                    epochs=int(args.stage1_epochs),
                    early_stop=max(3, int(args.stage1_epochs) // 2),
                    max_windows=int(args.max_windows),
                    lr=3e-4,
                    no_init=no_init,
                    init=stage1_init,
                    compound_val_every=2,
                ),
                env=common_env,
            )
            if rc != 0 or not stage1.is_file():
                state["notes"].append(f"stage1 failed rc={rc}")
                _write_json(state_path, state)
                return rc if rc != 0 else 1
            state["notes"].append(f"stage1 ok {stage1}")
            _write_json(state_path, state)

        if args.fresh or not stage2.is_file():
            left_h = (deadline - _now()).total_seconds() / 3600.0
            if left_h < 0.8 and stage1.is_file():
                print(f"[WARN] budget low ({left_h:.1f}h); skip stage2 polish", flush=True)
                state["notes"].append("skipped stage2 (budget)")
            else:
                state["phase"] = "stage2"
                _write_json(state_path, state)
                rc = _run(
                    "stage2",
                    _train_cmd(
                        out_ckpt=stage2,
                        train_csv=train_csv,
                        val_csv=val_csv,
                        spray_csv=spray_csv,
                        wall_ckpt=wall_ckpt,
                        epochs=int(args.stage2_epochs),
                        early_stop=max(2, int(args.stage2_epochs) - 1),
                        max_windows=int(args.max_windows),
                        lr=1e-4,
                        no_init=False,
                        init=str(stage1),
                        compound_val_every=2,
                    ),
                    env=common_env,
                )
                if rc != 0 or not stage2.is_file():
                    state["notes"].append(f"stage2 failed rc={rc}; using stage1")
                    _write_json(state_path, state)
                else:
                    state["notes"].append(f"stage2 ok {stage2}")
                    _write_json(state_path, state)

    best = stage2 if stage2.is_file() else stage1
    if not best.is_file():
        print("[ERR] no growth checkpoint", flush=True)
        return 1
    state["best_growth"] = str(best)

    if not args.skip_eval:
        state["phase"] = "eval"
        _write_json(state_path, state)

        # Val cohort (full sealed val set, even if training used --val-fast subset)
        eval_val = root / "eval_val.json"
        eval_val_csv = ",".join(VAL)
        if args.fresh or not eval_val.is_file():
            rc = _run(
                "eval_val",
                _eval_cmd(out=eval_val, anchors=eval_val_csv, wall=wall_ckpt, growth=str(best)),
            )
            if rc != 0:
                return rc

        # Challenge sealed
        chal_csv = ",".join(CHALLENGE)
        eval_chal = root / "eval_challenge.json"
        if args.fresh or not eval_chal.is_file():
            rc = _run(
                "eval_challenge",
                _eval_cmd(out=eval_chal, anchors=chal_csv, wall=wall_ckpt, growth=str(best)),
            )
            if rc != 0:
                return rc

        eval_base = root / "eval_baseline_c0_challenge.json"
        if args.fresh or not eval_base.is_file():
            rc = _run(
                "eval_baseline_c0_challenge",
                _eval_cmd(
                    out=eval_base,
                    anchors=chal_csv,
                    wall=wall_ckpt,
                    growth=baseline_growth,
                ),
            )
            if rc != 0:
                return rc

        # Train-sanity 007 (not a gen claim)
        eval_007 = root / "eval_train_sanity_007.json"
        if args.fresh or not eval_007.is_file():
            _run(
                "eval_007_sanity",
                _eval_cmd(
                    out=eval_007,
                    anchors="patient007",
                    wall=wall_ckpt,
                    growth=str(best),
                ),
            )

        val_m = _mean_from_eval(eval_val)
        chal_m = _mean_from_eval(eval_chal)
        base_m = _mean_from_eval(eval_base)
        s007 = _mean_from_eval(eval_007)

        def _score(block: dict, anc: str) -> float:
            per = block.get("per_anchor") or {}
            row = per.get(anc) or {}
            return float(row.get("deploy_clot_score", 0.0) or 0.0)

        summary = {
            "best_growth": str(best),
            "elapsed_h": (_now() - started).total_seconds() / 3600.0,
            "splits": splits,
            "val_mean": val_m.get("mean"),
            "challenge_mean": chal_m.get("mean"),
            "baseline_challenge_mean": base_m.get("mean"),
            "patient007_train_sanity": (s007.get("per_anchor") or {}).get("patient007"),
            "challenge_scores": {
                "patient032_new": _score(chal_m, "patient032"),
                "patient032_c0": _score(base_m, "patient032"),
                "patient009_new": _score(chal_m, "patient009"),
                "patient009_c0": _score(base_m, "patient009"),
            },
            "notes": state.get("notes"),
        }
        # Path-finding flags
        s32_new = summary["challenge_scores"]["patient032_new"]
        s32_c0 = summary["challenge_scores"]["patient032_c0"]
        summary["challenge_032_beats_c0"] = bool(s32_new > s32_c0 + 0.02)
        _write_json(root / "final_summary.json", summary)
        state["phase"] = "done"
        state["summary"] = summary
        _write_json(state_path, state)

        print("=" * 72, flush=True)
        print(f"[i] DONE best={best}", flush=True)
        print(
            f"[i] challenge 032 score new={s32_new:.3f} c0={s32_c0:.3f} "
            f"beats={summary['challenge_032_beats_c0']}",
            flush=True,
        )
        print(f"[save] {root / 'final_summary.json'}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
