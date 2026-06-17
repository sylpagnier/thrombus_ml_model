"""Grid-search per-anchor gelation beta for deploy clot F1 @ t=53.

Usage::

    python scripts/tune_species_gnn_beta_clot.py --anchors patient004,patient006
    python scripts/tune_species_gnn_beta_clot.py --write-manifest
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.species_pushforward_continuous import BIOCHEM_ANCHORS_6  # noqa: E402
from src.core_physics.t0_device import require_cuda_device  # noqa: E402
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402
from src.core_physics.t0_rung4_ladder import eval_rung4_step_clot  # noqa: E402
from src.inference.species_gnn_deploy_env import (  # noqa: E402
    DEFAULT_MANIFEST,
    load_deploy_manifest,
    species_gnn_deploy_env,
)
from src.evaluation.clot_relaxed_metrics import legacy_clot_f1_metrics as _clot_metrics  # noqa: E402
from src.utils.paths import get_project_root  # noqa: E402


def _f1_at_beta(
    anchor: str,
    beta: float,
    *,
    device: torch.device,
    manifest: dict,
    flow: str,
    t_last: int = 53,
) -> float:
    with species_gnn_deploy_env(
        manifest,
        overrides={
            "T0_R4_FLOW_SOURCE": flow,
            "SPECIES_GELATION_BETA_OVERRIDE": str(beta),
        },
        anchor=anchor,
        prefer_loao=True,
    ):
        graph = get_project_root() / "data/processed/graphs_biochem_anchors" / f"{anchor}.pt"
        data = torch.load(graph, map_location=device, weights_only=False)
        phys = PhysicsConfig(phase="biochem")
        bio = BiochemConfig(phase="biochem")
        report = eval_rung4_step_clot(
            data, phys, bio, device, step="species_gnn", times=[t_last],
        )
        clot = report["clot"]
        return float(clot[-1].get("clot_f1", 0.0)) if clot else 0.0


def tune_anchor(
    anchor: str,
    *,
    device: torch.device,
    manifest: dict,
    flow: str,
    betas: list[float],
    s0_target: float,
) -> dict:
    best_f1 = -1.0
    best_beta = None
    curve: list[dict] = []
    for b in betas:
        f1 = _f1_at_beta(anchor, b, device=device, manifest=manifest, flow=flow)
        curve.append({"beta": b, "clot_f1": f1})
        if f1 > best_f1:
            best_f1 = f1
            best_beta = b
        print(f"  beta={b:.3f} F1={f1:.3f}", flush=True)
    default_beta = 0.689
    f1_default = _f1_at_beta(anchor, default_beta, device=device, manifest=manifest, flow=flow)
    use_beta = best_beta if best_f1 >= f1_default else default_beta
    use_f1 = best_f1 if best_f1 >= f1_default else f1_default
    return {
        "anchor": anchor,
        "best_beta": float(use_beta),
        "best_f1": float(use_f1),
        "default_f1": float(f1_default),
        "beats_s0_target": use_f1 >= s0_target,
        "curve": curve,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Tune per-anchor gelation beta for clot F1")
    ap.add_argument("--anchors", default="patient004,patient006")
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--flow", default="gt", choices=("gt", "kinematics"))
    ap.add_argument("--s0-f1", type=float, default=0.408)
    ap.add_argument("--beta-min", type=float, default=0.15)
    ap.add_argument("--beta-max", type=float, default=2.0)
    ap.add_argument("--beta-steps", type=int, default=25)
    ap.add_argument("--write-manifest", action="store_true")
    ap.add_argument("--out", default="outputs/biochem/species_gnn_deploy/beta_tune.json")
    args = ap.parse_args()

    device = require_cuda_device()
    root = get_project_root()
    manifest = load_deploy_manifest(args.manifest.strip() or None)
    anchors = [a.strip() for a in args.anchors.split(",") if a.strip()]
    betas = [
        args.beta_min + i * (args.beta_max - args.beta_min) / max(args.beta_steps - 1, 1)
        for i in range(args.beta_steps)
    ]

    rows: list[dict] = []
    beta_overrides = dict(manifest.get("beta_overrides") or {})
    for anc in anchors:
        print(f"[i] tune {anc} ({args.flow}) ...", flush=True)
        row = tune_anchor(
            anc,
            device=device,
            manifest=manifest,
            flow=args.flow,
            betas=betas,
            s0_target=float(args.s0_f1),
        )
        rows.append(row)
        if row["best_f1"] > row["default_f1"] + 1e-4:
            beta_overrides[anc] = str(row["best_beta"])
        print(
            f"[OK] {anc} beta={row['best_beta']:.3f} F1={row['best_f1']:.3f} "
            f"(default={row['default_f1']:.3f}) beat_{args.s0_f1:.2f}={row['beats_s0_target']}",
            flush=True,
        )

    out = Path(args.out)
    if not out.is_absolute():
        out = root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"anchors": anchors, "flow": args.flow, "rows": rows}
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if args.write_manifest:
        manifest["beta_overrides"] = beta_overrides
        mpath = Path(args.manifest or DEFAULT_MANIFEST)
        if not mpath.is_absolute():
            mpath = root / mpath
        mpath.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"[save] manifest beta_overrides -> {mpath}", flush=True)

    print(f"[save] {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
