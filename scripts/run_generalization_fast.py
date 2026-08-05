"""Fast two-stage clot generalization pilot using deploy_clot_score-oriented checkpoints."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

WALL = "outputs/biochem/biochem_gnn/locked/species_gnn_best.pth"
# Baseline for end-of-run A/B only — NOT used as growth warm-start (orig10 leakage).
GROWTH_BASELINE = "outputs/biochem/biochem_gnn/locked/compound_growth_best.pth"


def run(label: str, args: list[str], env: dict[str, str] | None = None) -> int:
    full_env = dict(os.environ)
    full_env["SPECIES_CONTINUOUS_VEL_DECAY"] = "1"
    full_env["SPECIES_CONTINUOUS_VEL_DECAY_WALL_ONLY"] = "1"
    if env:
        full_env.update(env)
    print(f"[RUN] {label}\n  {' '.join(args)}", flush=True)
    rc = subprocess.call([sys.executable, "-u", *args], cwd=str(REPO), env=full_env)
    print(f"[i] {label} rc={rc}", flush=True)
    return int(rc)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="outputs/biochem/offwall_model/generalization_fast/splits.json")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--run-root", default="outputs/biochem/offwall_model/generalization_fast/fold0")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--challenge", default="patient009,patient032,patient013")
    ap.add_argument(
        "--init-mode",
        choices=("wall", "none"),
        default="wall",
        help=(
            "wall=warm-start from locked wall backbone (no growth-specialist leakage); "
            "none=random growth init. Never warm-start from C0 growth for this pilot."
        ),
    )
    args = ap.parse_args()

    split_path = Path(args.splits)
    if not split_path.is_absolute():
        split_path = REPO / split_path
    if not split_path.is_file():
        raise FileNotFoundError(f"missing splits json: {split_path}")

    splits = load_json(split_path)
    folds = splits.get("folds") or []
    fold = next((f for f in folds if int(f.get("fold", -1)) == int(args.fold)), None)
    if not fold:
        raise ValueError(f"fold {args.fold} missing in {split_path}")

    train_anchors = list(fold.get("train_anchors") or [])
    primary_val = str(fold.get("primary_val_anchor") or "").strip()
    val_anchors = list(fold.get("val_anchors") or [])
    if not val_anchors and primary_val:
        val_anchors = [primary_val]

    run_root = Path(args.run_root)
    if not run_root.is_absolute():
        run_root = REPO / run_root
    run_root.mkdir(parents=True, exist_ok=True)

    wall_ckpt = str((REPO / WALL).resolve())
    baseline_growth = str((REPO / GROWTH_BASELINE).resolve())

    stage1 = run_root / "growth_stage1_frozen" / "best.pth"
    stage2 = run_root / "growth_stage2_unfreeze" / "best.pth"

    train_csv = ",".join(train_anchors)
    val_csv = ",".join(val_anchors)
    # Prefer spray probes from train (idle / low clot) if present.
    spray_candidates = [a for a in ("patient002", "patient008", "patient006", "patient010") if a in train_anchors]
    spray_csv = ",".join(spray_candidates[:2]) if spray_candidates else ""

    print(f"[i] fold={args.fold} train={len(train_anchors)} val={len(val_anchors)}", flush=True)
    print(f"[i] val set: {val_csv}", flush=True)
    print(f"[i] init_mode={args.init_mode} (no C0 growth warm-start)", flush=True)

    common_env = {
        "SPECIES_LUMEN_SHAPE_FN_W": "6",
        "SPECIES_LUMEN_SHAPE_FP_W": "4",
        "SPECIES_CONTINUOUS_UNDERPRED_WEIGHT": "4.0",
    }

    def train_args(
        out_ckpt: Path,
        *,
        init: str | None,
        epochs: int,
        freeze: bool,
        no_init: bool = False,
    ) -> list[str]:
        a = [
            "-m", "src.training.train_offwall_growth",
            "--anchors", train_csv,
            "--val-anchors", val_csv,
            "--val-anchor", primary_val or val_anchors[0],
            "--epochs", str(epochs),
            "--early-stop", "2",
            "--max-windows", "16",
            "--hops-k", "5",
            "--supervise-mode", "frontier_ge2",
            "--frontier-hops", "2",
            "--loss-mode", "loss_lumen_shape",
            "--lumen-shape-weight", "4.0",
            "--ckpt-metric", "compound_primary_spray",
            "--train-feat-source", "band",
            "--mat-leg", "WC_v7_clot_phi_mse",
            "--compound-val",
            "--compound-val-route", "frontier_offwall",
            "--compound-val-frontier-hops", "0.5",
            "--wall-ckpt", wall_ckpt,
            "--wall-clot-floor-delta", "0.10",
            # Full val set is now cheap enough (wall loaded once); keep every epoch.
            "--compound-val-every", "1",
            "--spray-val-anchors", spray_csv,
            "--spray-val-max-ge2", "8",
            "--spray-score-penalty", "0.05",
            "--out", str(out_ckpt),
        ]
        if no_init:
            a.append("--no-init")
        elif init:
            a.extend(["--init", init])
        if freeze:
            a.append("--freeze-backbone")
        return a

    if args.fresh or not stage1.is_file():
        stage1_no_init = args.init_mode == "none"
        stage1_init = None if stage1_no_init else wall_ckpt
        rc = run(
            "stage1_frozen",
            train_args(stage1, init=stage1_init, epochs=3, freeze=True, no_init=stage1_no_init),
            env=common_env,
        )
        if rc != 0:
            return rc

    if args.fresh or not stage2.is_file():
        rc = run(
            "stage2_unfreeze",
            train_args(stage2, init=str(stage1), epochs=2, freeze=False),
            env=common_env,
        )
        if rc != 0:
            return rc

    # Challenge anchors held out from this fold's train when possible.
    eval_anchors = sorted(set(val_anchors + [a.strip() for a in args.challenge.split(",") if a.strip()]))
    eval_csv = ",".join(eval_anchors)
    ckpt_for_eval = stage2 if stage2.is_file() else stage1

    eval_a = run_root / "eval_stage2_compound.json"
    rc = run(
        "eval_stage2_compound",
        [
            "scripts/eval_mat_growth_simple.py",
            "--ckpt", wall_ckpt,
            "--mat-leg", "WC_v7_clot_phi_mse",
            "--no-baseline",
            "--out", str(eval_a),
            "--anchors", eval_csv,
            "--offwall-ckpt", str(ckpt_for_eval),
            "--two-model-route", "frontier_offwall",
            "--two-model-frontier-hops", "0.5",
        ],
    )
    if rc != 0:
        return rc

    eval_b = run_root / "eval_baseline_c0_compound.json"
    rc = run(
        "eval_baseline_c0_compound",
        [
            "scripts/eval_mat_growth_simple.py",
            "--ckpt", wall_ckpt,
            "--mat-leg", "WC_v7_clot_phi_mse",
            "--no-baseline",
            "--out", str(eval_b),
            "--anchors", eval_csv,
            "--offwall-ckpt", baseline_growth,
            "--two-model-route", "frontier_offwall",
            "--two-model-frontier-hops", "0.5",
        ],
    )
    if rc != 0:
        return rc

    summary = {
        "fold": int(args.fold),
        "train_anchors": train_anchors,
        "val_anchors": val_anchors,
        "primary_val_anchor": primary_val,
        "eval_anchors": eval_anchors,
        "stage1_ckpt": str(stage1),
        "stage2_ckpt": str(stage2),
        "eval_stage2": str(eval_a),
        "eval_baseline": str(eval_b),
        "compound_route": "frontier_offwall",
        "frontier_hops": 0.5,
    }
    (run_root / "pilot_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[save] {run_root / 'pilot_summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
