"""6-fold LOAO train: hold out one biochem anchor per fold (s34 recipe).

Usage::

    python scripts/run_species_gnn_loao_train.py
    python scripts/run_species_gnn_loao_train.py --holdouts patient004,patient006 --epochs 35
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.core_physics.species_pushforward_continuous import BIOCHEM_ANCHORS_6  # noqa: E402
from src.utils.paths import get_project_root  # noqa: E402

DEFAULT_OUT_ROOT = "outputs/biochem/species_gnn_loao"
DEFAULT_INIT = "outputs/biochem/species_snapshot_s34/best.pth"

S34_ENV: dict[str, str] = {
    "SPECIES_CONTINUOUS_GROWTH_ONLY_LOSS": "1",
    "SPECIES_CONTINUOUS_DUAL_HEAD": "1",
    "SPECIES_CONTINUOUS_PHYSICS_READOUT": "0",
    "SPECIES_KIN_PER_VESSEL_NORM": "1",
    "SPECIES_CONTINUOUS_SATURATION_GATE": "1",
    "SPECIES_CONTINUOUS_MATURE_FP_EXEMPT": "1",
    "SPECIES_CONTINUOUS_MATURE_FRAC": "0.95",
    "SPECIES_CONTINUOUS_SATURATION_SCALE": "80",
    "SPECIES_CONTINUOUS_TEMPORAL_GATE": "1",
    "SPECIES_CONTINUOUS_TEMPORAL_LAMBDA_MIN": "0.5",
    "SPECIES_CONTINUOUS_TEMPORAL_LAMBDA_MAX": "1.5",
    "SPECIES_CONTINUOUS_VEL_DECAY": "1",
    "SPECIES_CONTINUOUS_TEACHER_NOISE": "0.02",
    "SPECIES_CONTINUOUS_TEACHER_FP_FRAC": "0.08",
    "SPECIES_CONTINUOUS_TEACHER_BLUR": "0.25",
    "SPECIES_CONTINUOUS_TBPTT_TAIL": "5",
    "SPECIES_CONTINUOUS_CURRICULUM_UNROLL": "1",
    "SPECIES_CONTINUOUS_CLOSED_LOOP_INIT": "0.45",
    "SPECIES_CONTINUOUS_FINAL_STATE_WEIGHT": "0.35",
    "SPECIES_CONTINUOUS_FINAL_STATE_ALL_BAND": "1",
    "SPECIES_CONTINUOUS_SPEED_FP_WEIGHT": "4.0",
    "SPECIES_CONTINUOUS_DEPLOY_HORIZON": "53",
    "SPECIES_PUSHFORWARD_UNROLL": "10",
    "SPECIES_PUSHFORWARD_MAX_UNROLL": "53",
    "SPECIES_PUSHFORWARD_TRAIN_T0_MAX": "35",
}


def _run_fold(
    holdout: str,
    *,
    epochs: int,
    early_stop: int,
    init_ckpt: Path,
    out_root: Path,
    skip_existing: bool,
) -> dict:
    out_dir = out_root / f"holdout_{holdout}"
    ckpt = out_dir / "best.pth"
    if skip_existing and ckpt.is_file():
        print(f"[skip] {holdout} ckpt exists: {ckpt}", flush=True)
        return {"holdout": holdout, "ckpt": str(ckpt), "skipped": True}

    train_anchors = [a for a in BIOCHEM_ANCHORS_6 if a != holdout]
    out_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(S34_ENV)
    env["PYTHONUNBUFFERED"] = "1"

    cmd = [
        sys.executable,
        "-m",
        "src.training.train_species_pushforward_continuous",
        "--phase",
        "s34",
        "--anchors",
        ",".join(train_anchors),
        "--val-anchor",
        holdout,
        "--exclude-val-from-train",
        "--epochs",
        str(epochs),
        "--early-stop",
        str(early_stop),
        "--init-s26",
        str(init_ckpt),
        "--out",
        str(ckpt),
    ]
    print(f"[NEW] LOAO fold holdout={holdout} train={train_anchors}", flush=True)
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(REPO), env=env)
    elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"fold {holdout} failed rc={proc.returncode}")
    row = {"holdout": holdout, "ckpt": str(ckpt), "elapsed_s": elapsed, "skipped": False}
    log_path = out_dir / "train_log.jsonl"
    if log_path.is_file():
        last = None
        for line in log_path.read_text(encoding="utf-8").strip().splitlines():
            if line.strip():
                last = json.loads(line)
        if last:
            row["best_epoch"] = last.get("best_epoch", last.get("epoch"))
            row["deploy_mat_f1_t53"] = last.get("deploy_mat_f1_t53")
            row["best_score"] = last.get("best_score")
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description="6-fold LOAO species GNN (s34) training")
    ap.add_argument("--holdouts", default="", help="Comma list; default all 6 anchors")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--early-stop", type=int, default=18)
    ap.add_argument("--init", default=DEFAULT_INIT)
    ap.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    root = get_project_root()
    holdouts = (
        [h.strip() for h in args.holdouts.split(",") if h.strip()]
        if args.holdouts.strip()
        else list(BIOCHEM_ANCHORS_6)
    )
    init_ckpt = Path(args.init)
    if not init_ckpt.is_absolute():
        init_ckpt = root / init_ckpt
    out_root = Path(args.out_root)
    if not out_root.is_absolute():
        out_root = root / out_root
    out_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for h in holdouts:
        rows.append(
            _run_fold(
                h,
                epochs=int(args.epochs),
                early_stop=int(args.early_stop),
                init_ckpt=init_ckpt,
                out_root=out_root,
                skip_existing=bool(args.skip_existing),
            )
        )

    index = {
        "phase": "species_gnn_loao_s34",
        "holdouts": holdouts,
        "out_root": str(out_root),
        "init_ckpt": str(init_ckpt),
        "folds": rows,
    }
    index_path = out_root / "loao_index.json"
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"[OK] wrote {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
