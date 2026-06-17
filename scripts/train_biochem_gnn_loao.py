"""6-fold LOAO train for biochem GNN baseline."""

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

from src.biochem_gnn.config import (  # noqa: E402
    PHASE_LOAO_INDEX,
    PHASE_TRAIN,
    STACK_NAME,
    apply_train_recipe_env,
    global_ckpt_path,
    loao_root_path,
    rel_path,
)
from src.core_physics.species_pushforward_continuous import discover_biochem_anchors  # noqa: E402
from src.utils.paths import get_project_root  # noqa: E402


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

    train_anchors = [a for a in discover_biochem_anchors(REPO) if a != holdout]
    out_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    apply_train_recipe_env()
    for key, val in os.environ.items():
        if key.startswith("SPECIES_") or key.startswith("BIOCHEM_"):
            env[key] = val
    env["PYTHONUNBUFFERED"] = "1"

    cmd = [
        sys.executable,
        "-m",
        "src.training.train_species_pushforward_continuous",
        "--phase",
        PHASE_TRAIN,
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
            row["deploy_clot_f1_t53"] = last.get("deploy_clot_f1_t53")
            row["best_score"] = last.get("best_score")
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description="LOAO train for biochem GNN")
    ap.add_argument("--holdouts", default="", help="Comma list; default all anchors on disk")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--early-stop", type=int, default=18)
    ap.add_argument("--init", default="")
    ap.add_argument("--out-root", default="")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    root = get_project_root()
    holdouts = (
        [h.strip() for h in args.holdouts.split(",") if h.strip()]
        if args.holdouts.strip()
        else discover_biochem_anchors(root)
    )
    init_ckpt = Path(args.init) if args.init.strip() else global_ckpt_path()
    if not init_ckpt.is_absolute():
        init_ckpt = root / init_ckpt
    out_root = Path(args.out_root) if args.out_root.strip() else loao_root_path()
    if not out_root.is_absolute():
        out_root = root / out_root
    out_root.mkdir(parents=True, exist_ok=True)

    rows = [
        _run_fold(
            h,
            epochs=int(args.epochs),
            early_stop=int(args.early_stop),
            init_ckpt=init_ckpt,
            out_root=out_root,
            skip_existing=bool(args.skip_existing),
        )
        for h in holdouts
    ]

    index = {
        "phase": PHASE_LOAO_INDEX,
        "stack": STACK_NAME,
        "holdouts": holdouts,
        "out_root": str(out_root),
        "init_ckpt": str(init_ckpt),
        "folds": rows,
    }
    index_path = out_root / "loao_index.json"
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"[OK] wrote {rel_path(index_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
