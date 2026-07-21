"""Canonical biochem deploy baseline trainer (GINO-DEQ + species GNN + physics clot).

Trains the multi-component deploy baseline:
  1. species_gnn           - wall-band FI/Mat pushforward GNN (learned)
  2. gelation_beta         - global Mat gelation scale @ deploy horizon (learned)
  3. loao                  - optional per-anchor holdout folds
  4. gino_deq_kine         - external GINO-DEQ ckpt (not trained here)
  5. clot_trigger_physics  - gelation + nucleation (not trained; eval only)
  6. flow_coupling         - NOT YET (inference-time mu->kine feedback)

This is the **deploy** biochem stack. It does **not** replace
the retired GNODE ``train_biochem_corrector`` teacher/mu ladder (removed 2026-06).

Usage::

    python -m src.training.train_biochem_gnn --step species
    python -m src.bin.main train biochem-gnn -- --step all
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from src.biochem_gnn.config import (
    BETA_CKPT,
    GLOBAL_CKPT,
    INIT_WARMSTART,
    LOAO_DIR,
    apply_train_recipe_env,
    global_ckpt_path,
    loao_root_path,
    rel_path,
)
from src.core_physics.species_pushforward_continuous import BIOCHEM_ANCHORS_6
from src.utils.paths import get_project_root


def _repo() -> Path:
    return get_project_root()


def _run(cmd: list[str], *, label: str) -> None:
    print(f"[NEW] {label}", flush=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.run(cmd, cwd=str(_repo()), env=env)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def train_species(
    *,
    val_anchor: str,
    all_anchors: bool,
    anchors: str,
    epochs: int,
    early_stop: int,
    unroll: int,
    init_ckpt: Path,
    out_ckpt: Path,
    fresh: bool,
    lr: float | None = None,
    arch: str = "sage",
) -> None:
    apply_train_recipe_env()
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    if fresh and out_ckpt.is_file():
        out_ckpt.unlink(missing_ok=True)
        for side in ("best.json", "train_log.jsonl"):
            p = out_ckpt.parent / side
            if p.is_file():
                p.unlink()
    cmd = [
        sys.executable,
        "-m",
        "src.training.train_species_pushforward_continuous",
        "--phase",
        "biochem_gnn",
        "--val-anchor",
        val_anchor,
        "--epochs",
        str(epochs),
        "--unroll",
        str(unroll),
        "--early-stop",
        str(early_stop),
        "--init",
        str(init_ckpt),
        "--out",
        str(out_ckpt),
        "--arch",
        str(arch).strip().lower(),
    ]
    if lr is not None:
        cmd.extend(["--lr", str(lr)])
    if all_anchors:
        cmd.append("--all-anchors")
    elif anchors.strip():
        cmd.extend(["--anchors", anchors])
    else:
        cmd.append("--all-anchors")
    _run(cmd, label="biochem_gnn species_gnn")


def _p007_deploy_time_index(root: Path) -> int:
    import torch

    from src.core_physics.species_pushforward_continuous import deploy_eval_time_index

    p = root / "data/processed/graphs_biochem_anchors/patient007.pt"
    data = torch.load(p, map_location="cpu", weights_only=False)
    return deploy_eval_time_index(int(data.y.shape[0]))


def train_viscosity_beta(
    *,
    species_ckpt: Path,
    out_beta: Path,
    time_index: int | None = None,
) -> None:
    out_beta.parent.mkdir(parents=True, exist_ok=True)
    t_idx = int(time_index) if time_index is not None else _p007_deploy_time_index(_repo())
    _run(
        [
            sys.executable,
            "scripts/train_clot_phi_calibration.py",
            "--gnn-ckpt",
            rel_path(species_ckpt),
            "--time-index",
            str(t_idx),
            "--beta-init",
            "1.5",
            "--epochs",
            "300",
            "--lr",
            "0.08",
            "--growth-weight",
            "12.0",
            "--all-anchors",
            "--out",
            rel_path(out_beta),
        ],
        label="biochem_gnn gelation_beta",
    )


def train_loao(
    *,
    epochs: int,
    early_stop: int,
    init_ckpt: Path,
    out_root: Path,
) -> None:
    _run(
        [
            sys.executable,
            "scripts/train_biochem_gnn_loao.py",
            "--epochs",
            str(epochs),
            "--early-stop",
            str(early_stop),
            "--init",
            rel_path(init_ckpt),
            "--out-root",
            rel_path(out_root),
        ],
        label="biochem_gnn loao",
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Train biochem GNN deploy baseline")
    ap.add_argument(
        "--step",
        choices=("species", "viscosity", "loao", "deploy", "all"),
        default="all",
        help="species=species GNN; viscosity=beta; deploy=species+viscosity; loao=6-fold; all=species+viscosity+loao",
    )
    ap.add_argument("--val-anchor", default="patient007")
    ap.add_argument("--all-anchors", action="store_true")
    ap.add_argument("--anchors", default="")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--loao-epochs", type=int, default=40)
    ap.add_argument("--early-stop", type=int, default=35)
    ap.add_argument("--loao-early-stop", type=int, default=18)
    ap.add_argument("--unroll", type=int, default=10)
    ap.add_argument("--lr", type=float, default=None, help="species GNN Adam lr (default recipe)")
    ap.add_argument("--init", default="")
    ap.add_argument("--species-out", default="")
    ap.add_argument("--beta-out", default="")
    ap.add_argument("--loao-root", default="")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument(
        "--arch",
        choices=("sage", "gnode"),
        default="sage",
        help="Pushforward trunk for species GNN (sage=GraphSAGE, gnode=GINO derivative)",
    )
    args = ap.parse_args()

    root = _repo()
    species_out = Path(args.species_out) if args.species_out.strip() else (root / GLOBAL_CKPT)
    beta_out = Path(args.beta_out) if args.beta_out.strip() else (root / BETA_CKPT)
    loao_root = Path(args.loao_root) if args.loao_root.strip() else (root / LOAO_DIR)
    init_ckpt = Path(args.init) if args.init.strip() else (root / INIT_WARMSTART)
    if not init_ckpt.is_absolute():
        init_ckpt = root / init_ckpt

    steps = (
        ["species", "viscosity"]
        if args.step == "deploy"
        else ([args.step] if args.step != "all" else ["species", "viscosity", "loao"])
    )
    if "species" in steps:
        train_species(
            val_anchor=args.val_anchor,
            all_anchors=bool(args.all_anchors),
            anchors=args.anchors,
            epochs=int(args.epochs),
            early_stop=int(args.early_stop),
            unroll=int(args.unroll),
            init_ckpt=init_ckpt,
            out_ckpt=species_out,
            fresh=bool(args.fresh),
            lr=float(args.lr) if args.lr is not None else None,
            arch=str(args.arch).strip().lower(),
        )
        init_ckpt = species_out

    if "viscosity" in steps:
        species_ckpt = species_out if species_out.is_file() else global_ckpt_path()
        train_viscosity_beta(species_ckpt=species_ckpt, out_beta=beta_out)

    if "loao" in steps:
        species_ckpt = species_out if species_out.is_file() else global_ckpt_path()
        train_loao(
            epochs=int(args.loao_epochs),
            early_stop=int(args.loao_early_stop),
            init_ckpt=species_ckpt,
            out_root=loao_root,
        )

    print(f"[OK] biochem_gnn step={args.step}", flush=True)
    print(f"[i] species={rel_path(species_out)} beta={rel_path(beta_out)} loao={rel_path(loao_root)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
