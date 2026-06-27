"""Eval Mat-growth-simple ckpt vs triangle6_wall3hop baseline (analytical gelation clot).

Metrics per anchor @ deploy horizon (pred kine):
  * deploy_mat_f1   - closed-loop Mat active F1 on wall+3hop band
  * deploy_clot_*   - analytical mu1(Mat) gelation + nucleation trigger

Usage::

    python scripts/eval_mat_growth_simple.py
    python scripts/eval_mat_growth_simple.py --ckpt outputs/biochem/biochem_gnn/mat_growth_simple/best.pth
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.biochem_gnn.mat_growth_simple import (  # noqa: E402
    BASELINE_COMPARE_ID,
)
from src.biochem_gnn.config import apply_deploy_env, global_ckpt_path  # noqa: E402
from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.species_deploy_rollout import reset_species_rollout_flow_cache  # noqa: E402
from src.core_physics.species_pushforward_continuous import (  # noqa: E402
    discover_biochem_anchors,
    deploy_eval_time_index,
    eval_deploy_clot_f1,
    eval_full_rollout_fimat_f1,
    load_continuous_bundle,
    train_deploy_eval_flow_source,
)
from src.core_physics.species_pushforward_gnn import build_band_base_features  # noqa: E402
from src.core_physics.t0_device import require_cuda_device  # noqa: E402
from src.utils.kinematics_inference import load_kinematics_predictor, resolve_kinematics_checkpoint  # noqa: E402
from src.utils.paths import get_project_root  # noqa: E402

ANCHOR_DIR = get_project_root() / "data/processed/graphs_biochem_anchors"
DEFAULT_BASELINE_JSON = (
    get_project_root()
    / "outputs/biochem/biochem_gnn/baselines"
    / BASELINE_COMPARE_ID
    / "baseline.json"
)


def _load_static(data, device, kine_model, wall_hops: int) -> dict:
    return build_band_base_features(data, kine_model, device, wall_hops=wall_hops)


def _apply_ckpt_recipe(meta: dict, *, label: str) -> None:
    """Match eval env to how the checkpoint was trained (do not force mat on fi_mat ckpts)."""
    scope = meta.get("pushforward_species_scope") or meta.get("species_scope")
    if scope:
        os.environ["BIOCHEM_PUSHFORWARD_SPECIES_SCOPE"] = str(scope)
    dual = meta.get("dual_head")
    if dual is not None:
        os.environ["SPECIES_CONTINUOUS_DUAL_HEAD"] = "1" if bool(dual) else "0"
    if label == "mat_growth_simple" or scope == "mat":
        from src.biochem_gnn.mat_growth_simple import apply_mat_growth_simple_recipe_env

        apply_mat_growth_simple_recipe_env(force=True)
    else:
        from src.biochem_gnn.config import apply_train_recipe_env

        apply_train_recipe_env(force=True)

    # Restore input/architecture-shaping flags LAST (after the recipe env, which would otherwise
    # clobber them). These change base_feats width (geom) or the spatial-gate input dim (neighbor
    # gate); a mismatch silently drops the trained heads in the partial loader.
    if meta.get("geom_feats") is not None:
        os.environ["SPECIES_GEOM_FEATS"] = "1" if bool(meta.get("geom_feats")) else "0"
    if meta.get("geom_feats_rich") is not None:
        os.environ["SPECIES_GEOM_FEATS_RICH"] = "1" if bool(meta.get("geom_feats_rich")) else "0"
    if meta.get("flow_feats") is not None:
        os.environ["SPECIES_FLOW_FEATS"] = "1" if bool(meta.get("flow_feats")) else "0"
    if meta.get("neighbor_commit_gate") is not None:
        os.environ["SPECIES_CONTINUOUS_NEIGHBOR_COMMIT_GATE"] = (
            "1" if bool(meta.get("neighbor_commit_gate")) else "0"
        )
    if meta.get("neighbor_commit_alpha") is not None:
        os.environ["SPECIES_CONTINUOUS_NEIGHBOR_COMMIT_ALPHA"] = str(meta.get("neighbor_commit_alpha"))
    if meta.get("gate_temp") is not None:
        os.environ["SPECIES_CONTINUOUS_GATE_TEMP"] = str(meta.get("gate_temp"))
    if meta.get("frontier_hops") is not None:
        os.environ["SPECIES_CONTINUOUS_FRONTIER_HOPS"] = str(meta.get("frontier_hops"))
    if meta.get("nucleation_topk") is not None:
        os.environ["SPECIES_CONTINUOUS_NUCLEATION_TOPK"] = str(meta.get("nucleation_topk"))


def _eval_ckpt(
    ckpt_path: Path,
    anchors: list[str],
    device: torch.device,
    *,
    label: str,
) -> dict:
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    meta = dict(payload.get("meta") or {})
    _apply_ckpt_recipe(meta, label=label)
    bundle = load_continuous_bundle(ckpt_path, device=device, quiet=True)
    if bundle is None:
        raise FileNotFoundError(f"could not load continuous bundle: {ckpt_path}")
    model = bundle.model
    wall_hops = int(meta.get("wall_hops", 3))
    kine = load_kinematics_predictor(
        str(resolve_kinematics_checkpoint()),
        device,
        phys_cfg=PhysicsConfig(phase="kinematics"),
    )
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    flow_eval = train_deploy_eval_flow_source()
    per: dict[str, dict] = {}
    for anc in anchors:
        reset_species_rollout_flow_cache()
        data = torch.load(ANCHOR_DIR / f"{anc}.pt", map_location=device, weights_only=False)
        static = _load_static(data, device, kine, wall_hops)
        static["n_times"] = int(data.y.shape[0])
        t_eval = deploy_eval_time_index(int(data.y.shape[0]))
        mat_m = eval_full_rollout_fimat_f1(
            model, data, static, device, time_index=t_eval
        )
        env_snap = {k: os.environ.get(k) for k in ("T0_R4_FLOW_SOURCE", "SPECIES_ROLLOUT_VEL_SOURCE")}
        apply_deploy_env(overrides={"T0_R4_FLOW_SOURCE": flow_eval})
        clot_m = eval_deploy_clot_f1(
            model,
            data,
            static,
            phys,
            bio,
            device,
            time_index=t_eval,
            flow_source=flow_eval,
        )
        for k, v in env_snap.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        per[anc] = {
            "t_eval": int(t_eval),
            "deploy_mat_f1": float(mat_m["deploy_mat_f1"]),
            "deploy_fi_f1": float(mat_m.get("deploy_fi_f1", 0.0)),
            "mat_seed_prec": float(mat_m.get("mat_seed_prec", 0.0)),
            "mat_seed_count": float(mat_m.get("mat_seed_count", 0.0)),
            "mat_front_prec": float(mat_m.get("mat_front_prec", 0.0)),
            "mat_front_speed_ratio": float(mat_m.get("mat_front_speed_ratio", 0.0)),
            "mat_overpaint_frac": float(mat_m.get("mat_overpaint_frac", 0.0)),
            "mat_overpaint_per_gt": float(mat_m.get("mat_overpaint_per_gt", 0.0)),
            "deploy_clot_f1": float(clot_m["deploy_clot_f1"]),
            "deploy_clot_score": float(clot_m.get("deploy_clot_score", 0.0)),
            "deploy_clot_relaxed_prec": float(clot_m.get("deploy_clot_relaxed_prec", 0.0)),
            "deploy_clot_relaxed_rec": float(clot_m.get("deploy_clot_relaxed_rec", 0.0)),
        }
    keys = (
        "deploy_mat_f1",
        "deploy_clot_f1",
        "deploy_clot_score",
        "mat_seed_prec",
        "mat_seed_count",
        "mat_front_prec",
        "mat_front_speed_ratio",
        "mat_overpaint_frac",
        "mat_overpaint_per_gt",
    )
    mean = {k: sum(per[a][k] for a in anchors) / max(len(anchors), 1) for k in keys}
    return {"label": label, "ckpt": str(ckpt_path), "per_anchor": per, "mean": mean, "meta": meta}


def main() -> int:
    ap = argparse.ArgumentParser(description="Mat-growth-simple vs triangle6 baseline eval")
    ap.add_argument("--ckpt", default="", help="Mat-only simple ckpt (default: mat_growth_simple/best.pth)")
    ap.add_argument("--baseline-ckpt", default="", help="Baseline fi_mat ckpt (default: species/best.pth)")
    ap.add_argument("--baseline-json", default=str(DEFAULT_BASELINE_JSON))
    ap.add_argument("--anchors", default="", help="Comma list (default: all anchors on disk)")
    ap.add_argument("--out", default="outputs/biochem/biochem_gnn/mat_growth_simple/compare.json")
    args = ap.parse_args()

    root = get_project_root()
    device = require_cuda_device()
    anchors = (
        [a.strip() for a in args.anchors.split(",") if a.strip()]
        if args.anchors.strip()
        else discover_biochem_anchors(ANCHOR_DIR)
    )
    simple_ckpt = Path(args.ckpt) if args.ckpt.strip() else root / "outputs/biochem/biochem_gnn/mat_growth_simple/best.pth"
    if not simple_ckpt.is_absolute():
        simple_ckpt = root / simple_ckpt
    baseline_ckpt = Path(args.baseline_ckpt) if args.baseline_ckpt.strip() else global_ckpt_path()

    report: dict = {
        "anchors": anchors,
        "baseline_id": BASELINE_COMPARE_ID,
        "baseline_json": str(args.baseline_json),
    }
    if Path(args.baseline_json).is_file():
        report["baseline_recorded"] = json.loads(Path(args.baseline_json).read_text(encoding="utf-8"))

    print(f"[i] eval simple: {simple_ckpt}", flush=True)
    report["simple"] = _eval_ckpt(
        simple_ckpt,
        anchors,
        device,
        label="mat_growth_simple",
    )

    print(f"[i] eval baseline: {baseline_ckpt}", flush=True)
    report["baseline"] = _eval_ckpt(
        baseline_ckpt,
        anchors,
        device,
        label=BASELINE_COMPARE_ID,
    )

    delta = {
        k: report["simple"]["mean"][k] - report["baseline"]["mean"][k]
        for k in report["simple"]["mean"]
    }
    report["delta_simple_minus_baseline"] = delta

    out = Path(args.out)
    if not out.is_absolute():
        out = root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n==================== MAT GROWTH SIMPLE vs BASELINE ====================", flush=True)
    print(
        f"{'metric':<22} {'baseline':>10} {'simple':>10} {'delta':>10}",
        flush=True,
    )
    for k in ("deploy_mat_f1", "deploy_clot_f1", "deploy_clot_score"):
        b = report["baseline"]["mean"][k]
        s = report["simple"]["mean"][k]
        d = delta[k]
        print(f"{k:<22} {b:10.3f} {s:10.3f} {d:+10.3f}", flush=True)
    print(f"[save] {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
