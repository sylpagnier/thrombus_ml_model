"""~30 min V3.2 sweep: trajectory metrics + ranker vs Euler + viz."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.training.clot_ml_device import resolve_clot_ml_eval_device, resolve_clot_ml_training_device  # noqa: E402
from src.training.clot_ml_step0_coef import discover_anchor_paths  # noqa: E402
from src.training.clot_ml_v2_growth_gnn import (  # noqa: E402
    ClotGrowthRateGNN,
    apply_step3_v3_env,
    load_v3_checkpoint,
    teacher_phi_by_t_from_step1,
)
from src.training.clot_ml_v32_growth_ranker import (  # noqa: E402
    V32SweepLegConfig,
    apply_v32_env,
    build_model_for_leg,
    default_v32_sweep_dir,
    eval_leg_on_anchor,
    load_leg_checkpoint,
    resolve_rule_cfg,
    save_leg_checkpoint,
    train_one_graph_leg,
)
from src.training.clot_growth_eval import eval_phi_trajectory_on_anchor  # noqa: E402
from src.training.clot_ml_v2_growth_gnn import rollout_v3_growth_gnn  # noqa: E402
from src.training.train_clot_phi_simple import _split_train_val  # noqa: E402
from src.core_physics.clot_temporal_growth_rules import reset_temporal_kinematics_cache  # noqa: E402


def _sanitize(obj: object) -> object:
    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, float):
        return None if not math.isfinite(obj) else obj
    return obj


def _mean(rows: list[dict], key: str) -> float:
    if not rows:
        return float("nan")
    return float(sum(float(r.get(key, 0.0)) for r in rows) / len(rows))


def default_legs(
    *,
    epochs: int = 6,
    v3_ckpt: str,
    v31_ckpt: str,
) -> list[V32SweepLegConfig]:
    return [
        V32SweepLegConfig(name="replay_v3", arch="euler", epochs=0, init_ckpt=v3_ckpt),
        V32SweepLegConfig(name="replay_v31", arch="euler", epochs=0, init_ckpt=v31_ckpt),
        V32SweepLegConfig(
            name="v32_ranker",
            arch="ranker",
            epochs=epochs,
            onset_weight=1.0,
            ring_weight=0.4,
            temporal_equal=True,
            init_ckpt=v31_ckpt,
            teacher_weight=0.05,
        ),
        V32SweepLegConfig(
            name="v32_ranker_onset2x",
            arch="ranker",
            epochs=epochs,
            onset_weight=2.0,
            ring_weight=0.5,
            temporal_equal=True,
            init_ckpt=v31_ckpt,
            teacher_weight=0.05,
        ),
        V32SweepLegConfig(
            name="v32_euler_onset",
            arch="euler",
            epochs=epochs,
            onset_weight=1.5,
            ring_weight=0.5,
            temporal_equal=True,
            init_ckpt=v31_ckpt,
            teacher_weight=0.05,
        ),
    ]


def _init_model_from_ckpt(
    leg: V32SweepLegConfig,
    *,
    device: torch.device,
) -> torch.nn.Module:
    model = build_model_for_leg(leg, device=device)
    ckpt = leg.init_ckpt.strip()
    if not ckpt or not Path(REPO / ckpt).exists():
        return model
    src, _ = load_v3_checkpoint(REPO / ckpt, device=device, v31="v31" in leg.name or leg.arch == "euler")
    if leg.arch == "ranker":
        for key in ("conv1", "conv2"):
            if hasattr(model, key) and hasattr(src, key):
                getattr(model, key).load_state_dict(getattr(src, key).state_dict())
    else:
        model.load_state_dict(src.state_dict(), strict=True)
    return model


def _eval_replay_trajectory(
    model: ClotGrowthRateGNN,
    rule_cfg,
    paths: list[str],
    *,
    device: torch.device,
    phys: PhysicsConfig,
    bio: BiochemConfig,
    v31: bool,
    tag: str,
) -> list[dict]:
    rows: list[dict] = []
    apply_step3_v3_env(v31=v31)
    for p in paths:
        gp = Path(p)
        reset_temporal_kinematics_cache()
        data = torch.load(gp, map_location=device, weights_only=False)
        phi_by_t = rollout_v3_growth_gnn(
            model, data, rule_cfg, device=device, phys_cfg=phys, bio_cfg=bio
        )
        row = eval_phi_trajectory_on_anchor(
            phi_by_t,
            data,
            anchor=gp.stem,
            device=device,
            phys_cfg=phys,
            bio_cfg=bio,
            rule_tag=tag,
        )
        rows.append(row)
    return rows


def train_leg(
    leg: V32SweepLegConfig,
    *,
    train_paths: list[str],
    val_paths: list[str],
    rule_cfg,
    device: torch.device,
    phys: PhysicsConfig,
    bio: BiochemConfig,
    out_dir: Path,
    teacher_ckpt: str,
) -> Path:
    if leg.epochs <= 0:
        raise ValueError(f"train_leg called on replay leg {leg.name}")

    if leg.arch == "ranker":
        apply_v32_env()
    else:
        apply_step3_v3_env(v31=True)
        os.environ["CLOT_V31_HARD_COMMIT"] = "1"

    model = _init_model_from_ckpt(leg, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=float(leg.lr))
    ckpt_path = out_dir / f"{leg.name}_best.pth"
    best_val = -1.0

    print(f"[NEW] leg={leg.name} arch={leg.arch} epochs={leg.epochs}", flush=True)
    for ep in range(1, int(leg.epochs) + 1):
        model.train()
        loss_sum = 0.0
        for p in train_paths:
            data = torch.load(p, map_location=device, weights_only=False)
            teacher_phi = None
            if leg.teacher_weight > 0 and Path(REPO / teacher_ckpt).exists():
                teacher_phi = teacher_phi_by_t_from_step1(
                    data,
                    rule_cfg,
                    device=device,
                    phys_cfg=phys,
                    bio_cfg=bio,
                    teacher_ckpt=REPO / teacher_ckpt,
                )
            opt.zero_grad(set_to_none=True)
            loss = train_one_graph_leg(
                model,
                data,
                rule_cfg,
                leg=leg,
                device=device,
                phys_cfg=phys,
                bio_cfg=bio,
                teacher_phi_by_t=teacher_phi,
            )
            loss.backward()
            opt.step()
            loss_sum += float(loss.item())
        loss_sum /= max(len(train_paths), 1)

        model.eval()
        val_rows = [
            eval_leg_on_anchor(
                model,
                rule_cfg,
                graph_path=Path(p),
                leg=leg,
                device=device,
                phys_cfg=phys,
                bio_cfg=bio,
            )
            for p in val_paths
        ]
        val_traj = _mean(val_rows, "trajectory_score")
        print(
            f"[i] {leg.name} ep={ep} loss={loss_sum:.4f} traj={val_traj:.3f} "
            f"ring={_mean(val_rows, 'tfinal_wall_ring_frac'):.3f}",
            flush=True,
        )
        if val_traj > best_val:
            best_val = val_traj
            meta = {
                "leg": leg.name,
                "arch": leg.arch,
                "epochs": leg.epochs,
                "onset_weight": leg.onset_weight,
                "ring_weight": leg.ring_weight,
                "temporal_equal": leg.temporal_equal,
                "best_trajectory_score": best_val,
                "init_ckpt": leg.init_ckpt,
            }
            save_leg_checkpoint(ckpt_path, model=model, meta=meta)

    if not ckpt_path.exists():
        meta = {"leg": leg.name, "arch": leg.arch, "best_trajectory_score": best_val}
        save_leg_checkpoint(ckpt_path, model=model, meta=meta)
    return ckpt_path


def eval_leg_all(
    leg: V32SweepLegConfig,
    ckpt_path: Path,
    paths: list[str],
    *,
    rule_cfg,
    device: torch.device,
    phys: PhysicsConfig,
    bio: BiochemConfig,
) -> list[dict]:
    if int(leg.epochs) <= 0:
        model, _ = load_v3_checkpoint(
            ckpt_path,
            device=device,
            v31=leg.name.endswith("v31") or "v31" in leg.name,
        )
        v31 = "v31" in leg.name
        tag = "replay_v31" if v31 else "replay_v3"
        return _eval_replay_trajectory(
            model,
            rule_cfg,
            paths,
            device=device,
            phys=phys,
            bio=bio,
            v31=v31,
            tag=tag,
        )

    model, _ = load_leg_checkpoint(ckpt_path, leg, device=device)
    return [
        eval_leg_on_anchor(
            model,
            rule_cfg,
            graph_path=Path(p),
            leg=leg,
            device=device,
            phys_cfg=phys,
            bio_cfg=bio,
        )
        for p in paths
    ]


def run_viz(
    leg_name: str,
    ckpt_path: Path,
    *,
    anchor: str,
    anchor_dir: str,
    step0_json: str,
    arch: str,
    max_frames: int,
    viz_dir: Path,
) -> Path:
    out_png = viz_dir / f"{leg_name}_{anchor}.png"
    cmd = [
        sys.executable,
        str(REPO / "scripts" / "viz_clot_ml_v32_sweep.py"),
        "--anchor",
        anchor,
        "--anchor-dir",
        anchor_dir,
        "--step0-json",
        step0_json,
        "--leg",
        leg_name,
        "--arch",
        arch,
        "--ckpt",
        str(ckpt_path.relative_to(REPO) if ckpt_path.is_relative_to(REPO) else ckpt_path),
        "--max-frames",
        str(max_frames),
        "--out",
        str(out_png.relative_to(REPO) if out_png.is_relative_to(REPO) else out_png),
    ]
    subprocess.run(cmd, cwd=str(REPO), check=True)
    return out_png


def main() -> int:
    ap = argparse.ArgumentParser(description="V3.2 ~30m growth sweep")
    ap.add_argument("--anchor-dir", default="data/processed/graphs_biochem_anchors")
    ap.add_argument("--step0-json", default="outputs/biochem/clot_ml_ladder/step0_coef/best_coef.json")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--val", default="patient007")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--v3-ckpt", default="outputs/biochem/clot_ml_ladder_v2/v3_growth_gnn/clot_ml_v3_growth_gnn_best.pth")
    ap.add_argument("--v31-ckpt", default="outputs/biochem/clot_ml_ladder_v2/v31_growth_gnn/clot_ml_v31_growth_gnn_best.pth")
    ap.add_argument("--teacher-ckpt", default="outputs/biochem/sweep_clot_ml_physics_6h/step1_a35/clot_ml_step1_best.pth")
    ap.add_argument("--viz-anchors", default="patient007,patient002")
    ap.add_argument("--max-frames", type=int, default=10)
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-viz", action="store_true")
    ap.add_argument("--legs", default="", help="comma-separated leg names (default: all)")
    args = ap.parse_args()

    t0 = time.time()
    out_dir = Path(args.out_dir) if args.out_dir else default_v32_sweep_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = REPO / "outputs/biochem/viz/clot_v2"
    viz_dir.mkdir(parents=True, exist_ok=True)

    device_train = resolve_clot_ml_training_device()
    device_eval = resolve_clot_ml_eval_device()
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    rule_cfg = resolve_rule_cfg(REPO / args.step0_json)

    paths = [str(p) for p in discover_anchor_paths(REPO / args.anchor_dir)]
    train_paths, val_paths = _split_train_val(paths, args.val)
    viz_anchors = [a.strip() for a in args.viz_anchors.split(",") if a.strip()]

    legs = default_legs(epochs=int(args.epochs), v3_ckpt=args.v3_ckpt, v31_ckpt=args.v31_ckpt)
    if args.legs.strip():
        want = {x.strip() for x in args.legs.split(",") if x.strip()}
        legs = [lg for lg in legs if lg.name in want]

    results: list[dict] = []
    for leg in legs:
        leg_dir = out_dir / leg.name
        leg_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = leg_dir / f"{leg.name}_best.pth"

        if int(leg.epochs) <= 0:
            src = REPO / (leg.init_ckpt or args.v31_ckpt)
            if not src.exists():
                print(f"[WARN] skip {leg.name}: missing {src}", flush=True)
                continue
            ckpt_path = src
        elif not args.skip_train:
            ckpt_path = train_leg(
                leg,
                train_paths=train_paths,
                val_paths=val_paths,
                rule_cfg=rule_cfg,
                device=device_train,
                phys=phys,
                bio=bio,
                out_dir=leg_dir,
                teacher_ckpt=args.teacher_ckpt,
            )
        elif not ckpt_path.exists():
            print(f"[WARN] skip {leg.name}: no ckpt and --skip-train", flush=True)
            continue

        per_anchor = eval_leg_all(
            leg,
            ckpt_path,
            paths,
            rule_cfg=rule_cfg,
            device=device_eval,
            phys=phys,
            bio=bio,
        )
        val_row = next((r for r in per_anchor if r["anchor"] == args.val), per_anchor[0] if per_anchor else {})
        summary = {
            "leg": leg.name,
            "arch": leg.arch,
            "ckpt": str(ckpt_path.relative_to(REPO) if ckpt_path.is_relative_to(REPO) else ckpt_path),
            "epochs": leg.epochs,
            "mean_trajectory_score": _mean(per_anchor, "trajectory_score"),
            "mean_band_f1": _mean(per_anchor, "mean_band_f1"),
            "mean_clot_shape": _mean(per_anchor, "mean_clot_shape"),
            "mean_early_recall": _mean(per_anchor, "early_recall_cov"),
            "mean_wall_ring": _mean(per_anchor, "tfinal_wall_ring_frac"),
            "val_trajectory_score": float(val_row.get("trajectory_score", float("nan"))),
            "val_anchor": args.val,
            "per_anchor": per_anchor,
        }
        leg_json = leg_dir / "eval.json"
        leg_json.write_text(json.dumps(_sanitize(summary), indent=2), encoding="utf-8")
        print(
            f"[OK] {leg.name} traj={summary['mean_trajectory_score']:.3f} "
            f"ring={summary['mean_wall_ring']:.3f} early={summary['mean_early_recall']:.3f}",
            flush=True,
        )

        viz_paths: list[str] = []
        if not args.skip_viz:
            for anc in viz_anchors:
                try:
                    png = run_viz(
                        leg.name,
                        ckpt_path,
                        anchor=anc,
                        anchor_dir=args.anchor_dir,
                        step0_json=args.step0_json,
                        arch=leg.arch,
                        max_frames=int(args.max_frames),
                        viz_dir=viz_dir,
                    )
                    viz_paths.append(str(png))
                except Exception as exc:
                    print(f"[WARN] viz {leg.name} {anc}: {exc}", flush=True)
        summary["viz"] = viz_paths
        results.append(summary)

    results.sort(key=lambda r: float(r.get("mean_trajectory_score", 0.0)), reverse=True)
    payload = {
        "sweep": "v32_growth_30m",
        "elapsed_sec": time.time() - t0,
        "val": args.val,
        "epochs_per_train_leg": int(args.epochs),
        "ranked": results,
        "best_leg": results[0]["leg"] if results else "",
    }
    summary_path = out_dir / "sweep_summary.json"
    summary_path.write_text(json.dumps(_sanitize(payload), indent=2), encoding="utf-8")
    print(f"[save] {summary_path}", flush=True)
    if results:
        best = results[0]
        print(
            f"[OK] best={best['leg']} traj={best['mean_trajectory_score']:.3f} "
            f"(val {best['val_trajectory_score']:.3f})",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
