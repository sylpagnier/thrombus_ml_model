"""4h curated sweep: s34 species GNN tuning + architecture variants.

Each leg: warm-start from locked baseline GNN, train 6 anchors, deploy eval vs baseline.

Usage::

    python scripts/sweep_species_gnn_s34_arch.py
    python scripts/sweep_species_gnn_s34_arch.py --legs fp_wall12,arch_delta_res --epochs 22
    python scripts/sweep_species_gnn_s34_arch.py --skip-train --legs ref_baseline
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.species_pushforward_continuous import BIOCHEM_ANCHORS_6  # noqa: E402
from src.core_physics.t0_device import require_cuda_device  # noqa: E402
from src.core_physics.t0_rung4_ladder import eval_rung4_step_clot  # noqa: E402
from src.inference.species_gnn_deploy_env import (  # noqa: E402
    baseline_dir,
    load_deploy_manifest,
    species_gnn_deploy_env,
)
from src.utils.paths import get_project_root  # noqa: E402

SWEEP_ROOT = "outputs/biochem/sweep_species_gnn_s34_arch"
BASELINE_GNN = "outputs/biochem/species_gnn_deploy_baseline/species_gnn_best.pth"
BASELINE_MANIFEST = "data/reference/species_gnn_deploy_baseline.json"

S34_BASE: dict[str, str] = {
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
    "SPECIES_VISCOSITY_CALIB": "1",
}

LEG_CATALOG: dict[str, dict] = {
    "ref_baseline": {
        "hypothesis": "Eval locked baseline only (no train)",
        "train": False,
        "ckpt": BASELINE_GNN,
    },
    "fp_wall12": {
        "hypothesis": "Stronger FP / speed penalty vs wall carpet (p004)",
        "env": {
            "SPECIES_CONTINUOUS_FP_WEIGHT": "12.0",
            "SPECIES_CONTINUOUS_SPEED_FP_WEIGHT": "6.0",
            "SPECIES_CONTINUOUS_SPATIAL_LOSS_WEIGHT": "1.5",
        },
    },
    "temporal_wide": {
        "hypothesis": "Wider temporal lambda range for late growth",
        "env": {
            "SPECIES_CONTINUOUS_TEMPORAL_LAMBDA_MIN": "0.30",
            "SPECIES_CONTINUOUS_TEMPORAL_LAMBDA_MAX": "2.0",
        },
    },
    "closed_hard": {
        "hypothesis": "Harder closed-loop + teacher aug",
        "env": {
            "SPECIES_CONTINUOUS_CLOSED_LOOP_INIT": "0.60",
            "SPECIES_CONTINUOUS_TEACHER_FP_FRAC": "0.12",
            "SPECIES_CONTINUOUS_TEACHER_BLUR": "0.35",
        },
    },
    "sat_tight": {
        "hypothesis": "Tighter saturation headroom",
        "env": {
            "SPECIES_CONTINUOUS_SATURATION_SCALE": "120",
            "SPECIES_CONTINUOUS_MATURE_FRAC": "0.90",
            "SPECIES_CONTINUOUS_MAX_SAT_LOG_MAT": "0.0015",
        },
    },
    "arch_delta_res": {
        "hypothesis": "Zero-init delta residual on fused features",
        "env": {
            "SPECIES_CONTINUOUS_DELTA_RESIDUAL": "1",
            "SPECIES_CONTINUOUS_DELTA_RESIDUAL_ALPHA": "0.35",
        },
    },
    "arch_temp_offset": {
        "hypothesis": "Global mass -> per-channel temporal offset",
        "env": {
            "SPECIES_CONTINUOUS_TEMPORAL_OFFSET": "1",
            "SPECIES_CONTINUOUS_TEMPORAL_OFFSET_SCALE": "0.15",
        },
    },
    "arch_combo_res": {
        "hypothesis": "Delta residual + temporal offset",
        "env": {
            "SPECIES_CONTINUOUS_DELTA_RESIDUAL": "1",
            "SPECIES_CONTINUOUS_TEMPORAL_OFFSET": "1",
        },
    },
    "clot_score_w60": {
        "hypothesis": "Checkpoint pick by deploy clot F1 (60%)",
        "env": {
            "SPECIES_CONTINUOUS_SCORE_CLOUT_W": "0.60",
        },
    },
    "hidden96": {
        "hypothesis": "Wider hidden (96) capacity",
        "env": {"SPECIES_SNAPSHOT_HIDDEN": "96"},
        "hidden": 96,
    },
}

DEFAULT_LEG_ORDER = [
    "ref_baseline",
    "fp_wall12",
    "temporal_wide",
    "closed_hard",
    "arch_delta_res",
    "arch_temp_offset",
    "arch_combo_res",
    "clot_score_w60",
]


def _leg_env(overrides: dict[str, str]) -> dict[str, str]:
    env = os.environ.copy()
    env.update(S34_BASE)
    for k, v in overrides.items():
        env[k] = str(v)
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _train_leg(
    leg_id: str,
    leg_cfg: dict,
    *,
    epochs: int,
    early_stop: int,
    init_ckpt: Path,
    skip_train: bool,
) -> Path:
    root = get_project_root()
    out = root / SWEEP_ROOT / leg_id / "best.pth"
    if leg_cfg.get("train") is False:
        ckpt = root / str(leg_cfg.get("ckpt", BASELINE_GNN))
        if not ckpt.is_file():
            raise FileNotFoundError(f"missing ref ckpt: {ckpt}")
        return ckpt
    if skip_train and out.is_file():
        print(f"[skip] train {leg_id} ckpt exists", flush=True)
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.is_file():
        out.unlink()
    cmd = [
        sys.executable,
        "-m",
        "src.training.train_species_pushforward_continuous",
        "--phase",
        "s34",
        "--all-anchors",
        "--val-anchor",
        "patient007",
        "--epochs",
        str(epochs),
        "--early-stop",
        str(early_stop),
        "--init-s26",
        str(init_ckpt.relative_to(root)).replace("\\", "/"),
        "--out",
        str(out.relative_to(root)).replace("\\", "/"),
    ]
    hidden = leg_cfg.get("hidden")
    if hidden is not None:
        cmd.extend(["--hidden", str(int(hidden))])
    env = _leg_env(dict(leg_cfg.get("env") or {}))
    print(f"[NEW] train leg={leg_id}", flush=True)
    subprocess.run(cmd, cwd=root, env=env, check=True)
    return out


@torch.no_grad()
def _eval_leg(
    leg_id: str,
    ckpt: Path,
    *,
    device: torch.device,
    flow: str = "gt",
) -> dict:
    root = get_project_root()
    manifest = load_deploy_manifest(BASELINE_MANIFEST)
    manifest = dict(manifest)
    try:
        ckpt_rel = str(ckpt.relative_to(root)).replace("\\", "/")
    except ValueError:
        ckpt_rel = str(ckpt)
    manifest["species_gnn_ckpt"] = ckpt_rel
    # Single global ckpt for sweep compare (no LOAO per leg in 4h budget).
    manifest["loao_preferred"] = []
    manifest["ckpt_overrides"] = {a: manifest["species_gnn_ckpt"] for a in BIOCHEM_ANCHORS_6}

    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    rows: list[dict] = []
    t0 = time.perf_counter()
    for anc in BIOCHEM_ANCHORS_6:
        graph = root / "data/processed/graphs_biochem_anchors" / f"{anc}.pt"
        data = torch.load(graph, map_location=device, weights_only=False)
        n_steps = int(data.y.shape[0])
        t_eval = min(53, n_steps - 1)
        with species_gnn_deploy_env(
            manifest,
            overrides={"T0_R4_FLOW_SOURCE": flow},
            anchor=anc,
            prefer_loao=False,
        ):
            report = eval_rung4_step_clot(
                data, phys, bio, device, step="species_gnn", times=[t_eval],
            )
            s0 = eval_rung4_step_clot(data, phys, bio, device, step="s0", times=[t_eval])
        f1 = float(report["clot"][-1].get("clot_f1", 0.0))
        f1_s0 = float(s0["clot"][-1].get("clot_f1", 0.0))
        health = bool(report.get("rollout_health", {}).get("health_pass", False))
        rows.append({
            "anchor": anc,
            "time_eval": int(t_eval),
            "clot_f1_t53": f1,
            "s0_f1_t53": f1_s0,
            "delta_vs_s0": f1 - f1_s0,
            "health_pass": health,
        })
    holdout = [r for r in rows if r["anchor"] != "patient007"]
    p007 = next(r for r in rows if r["anchor"] == "patient007")
    mean_holdout = sum(r["clot_f1_t53"] for r in holdout) / max(len(holdout), 1)
    p004 = next(r for r in rows if r["anchor"] == "patient004")
    p006 = next(r for r in rows if r["anchor"] == "patient006")
    score = (
        0.45 * mean_holdout
        + 0.35 * p007["clot_f1_t53"]
        + 0.10 * p004["clot_f1_t53"]
        + 0.10 * p006["clot_f1_t53"]
    )
    unhealthy = sum(1 for r in rows if not r["health_pass"])
    score -= 0.02 * unhealthy
    return {
        "leg": leg_id,
        "ckpt": str(ckpt),
        "flow": flow,
        "elapsed_s": time.perf_counter() - t0,
        "mean_holdout_f1": mean_holdout,
        "patient007_f1": p007["clot_f1_t53"],
        "patient004_f1": p004["clot_f1_t53"],
        "patient006_f1": p006["clot_f1_t53"],
        "score": score,
        "per_anchor": rows,
    }


def _load_baseline_score() -> float:
    p = baseline_dir() / "eval_summary.json"
    if p.is_file():
        raw = json.loads(p.read_text(encoding="utf-8"))
        p007 = raw.get("patient007_clot_f1_t53")
        mean_h = raw.get("mean_holdout_clot_f1_t53_gt")
        if p007 is not None and mean_h is not None:
            return 0.45 * float(mean_h) + 0.55 * float(p007)
    return 0.55 * 0.523 + 0.45 * 0.701  # fallback


def main() -> int:
    ap = argparse.ArgumentParser(description="Species GNN s34 architecture sweep")
    ap.add_argument("--legs", default="", help="Comma-separated leg ids (default: 4h catalog)")
    ap.add_argument("--epochs", type=int, default=22)
    ap.add_argument("--early-stop", type=int, default=12)
    ap.add_argument("--init", default=BASELINE_GNN)
    ap.add_argument("--flow", default="gt", choices=("gt", "kinematics"))
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-completed", action="store_true")
    args = ap.parse_args()

    device = require_cuda_device()
    root = get_project_root()
    sweep_dir = root / SWEEP_ROOT
    sweep_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = sweep_dir / "manifest.jsonl"
    init_ckpt = root / args.init
    if not init_ckpt.is_file():
        init_ckpt = root / "outputs/biochem/species_snapshot_s34/best.pth"

    leg_ids = [x.strip() for x in args.legs.split(",") if x.strip()] or DEFAULT_LEG_ORDER
    baseline_score = _load_baseline_score()
    print(f"[i] baseline composite score ~{baseline_score:.3f}", flush=True)

    results: list[dict] = []
    for leg_id in leg_ids:
        if leg_id not in LEG_CATALOG:
            print(f"[WARN] unknown leg {leg_id}", flush=True)
            continue
        leg_cfg = LEG_CATALOG[leg_id]
        eval_path = sweep_dir / leg_id / "eval.json"
        if args.skip_completed and eval_path.is_file():
            row = json.loads(eval_path.read_text(encoding="utf-8"))
            results.append(row)
            print(f"[skip] {leg_id} score={row.get('score', 0):.3f}", flush=True)
            continue
        t_leg = time.perf_counter()
        try:
            ckpt = _train_leg(
                leg_id,
                leg_cfg,
                epochs=int(args.epochs),
                early_stop=int(args.early_stop),
                init_ckpt=init_ckpt,
                skip_train=bool(args.skip_train),
            )
            ev = _eval_leg(leg_id, ckpt, device=device, flow=args.flow)
            ev["hypothesis"] = leg_cfg.get("hypothesis", "")
            ev["env"] = leg_cfg.get("env") or {}
            ev["train_s"] = time.perf_counter() - t_leg
            ev["delta_vs_baseline"] = ev["score"] - baseline_score
            eval_path.parent.mkdir(parents=True, exist_ok=True)
            eval_path.write_text(json.dumps(ev, indent=2), encoding="utf-8")
            results.append(ev)
            line = json.dumps({"leg": leg_id, "status": "OK", **ev}) + "\n"
            with manifest_path.open("a", encoding="utf-8") as f:
                f.write(line)
            print(
                f"[OK] {leg_id} score={ev['score']:.3f} "
                f"p007={ev['patient007_f1']:.3f} mean_h={ev['mean_holdout_f1']:.3f} "
                f"delta_base={ev['delta_vs_baseline']:+.3f}",
                flush=True,
            )
        except Exception as exc:
            err = {"leg": leg_id, "status": "ERR", "error": str(exc)}
            with manifest_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(err) + "\n")
            print(f"[ERR] {leg_id}: {exc}", flush=True)

    results.sort(key=lambda r: r.get("score", 0.0), reverse=True)
    payload = {
        "baseline_composite_score": baseline_score,
        "winner": results[0] if results else None,
        "results": results,
    }
    out_json = sweep_dir / "sweep_results.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[save] {out_json}", flush=True)
    if results:
        w = results[0]
        print(
            f"[OK] winner={w['leg']} score={w['score']:.3f} "
            f"p007={w['patient007_f1']:.3f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
