"""Rung 1 (GT gamma) and Rung 2 (proxy gamma) oracle eval.

Rung 1: GT u,v,p + GT species + COMSOL spf.sr
Rung 2: GT u,v,p + GT species + proxy gamma (no spf.sr sidecar)

Usage::

    python scripts/eval_t0_rung12.py --anchor patient007
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

from src.core_physics.t0_clot_predictor import t0_gt_baseline_env
from src.core_physics.t0_rung_config import (  # noqa: E402
    RUNG2_GAMMA_MODE,
    RUNG2_GAMMA_SCALE,
    RUNG2_POISEUILLE_SCALE,
)
from src.core_physics.t0_mu_physics import (  # noqa: E402
    eval_anchor_t0_mu,
    gt_clot_phi_at_time,
    predict_clot_phi_at_time,
    rollout_t0_clot_phi,
    t0_physics_env,
)
from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.training.train_clot_phi_simple import _clot_metrics  # noqa: E402
from src.utils.paths import get_project_root  # noqa: E402

def _clot_timeline(
    data,
    anchor: str,
    phys: PhysicsConfig,
    bio: BiochemConfig,
    device: torch.device,
    times: list[int],
    *,
    gamma_mode: str,
    gamma_scale: float = 1.0,
    poiseuille_scale: float | None = None,
    nucleation: bool = False,
) -> list[dict]:
    rows: list[dict] = []
    mask = torch.ones(int(data.num_nodes), dtype=torch.bool, device=device)
    ctx = (
        t0_physics_env(anchor, gamma_mode="comsol_sr")
        if gamma_mode in ("comsol_sr", "spf_sr", "spf.sr")
        else t0_gt_baseline_env(
            gamma_mode=gamma_mode,
            gamma_scale=gamma_scale,
            poiseuille_scale=poiseuille_scale,
        )
    )
    with ctx:
        traj = (
            rollout_t0_clot_phi(
                data, phys, bio, device, gamma_mode=gamma_mode, nucleation=True, nucleation_hops=1
            )
            if nucleation
            else None
        )
        for t in times:
            phi_gt = gt_clot_phi_at_time(data, t, phys, device)
            if nucleation and traj is not None:
                phi_pred = traj[t]["phi"]
            else:
                phi_pred, _ = predict_clot_phi_at_time(
                    data, t, phys, bio, device, gamma_mode=gamma_mode
                )
            m = _clot_metrics(phi_pred.reshape(-1), phi_gt.reshape(-1), mask)
            rows.append({"time": int(t), **m})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Rung 1/2 oracle eval")
    ap.add_argument("--anchor", default="patient007")
    ap.add_argument("--times", default="0,27,53")
    ap.add_argument("--out", default="outputs/biochem/clot_trigger/t0_rung12_eval.json")
    args = ap.parse_args()

    root = get_project_root()
    graph = root / "data/processed/graphs_biochem_anchors" / f"{args.anchor}.pt"
    if not graph.is_file():
        print(f"[ERR] missing {graph}", file=sys.stderr)
        return 1

    times = [int(x.strip()) for x in args.times.split(",") if x.strip()]
    device = torch.device("cpu")
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    data = torch.load(graph, map_location="cpu", weights_only=False)

    with t0_physics_env(args.anchor, gamma_mode="comsol_sr") as env1:
        rung1_mu = eval_anchor_t0_mu(graph, times=times, gamma_mode="comsol_sr")
    with t0_gt_baseline_env(
        gamma_mode=RUNG2_GAMMA_MODE,
        gamma_scale=RUNG2_GAMMA_SCALE,
        poiseuille_scale=RUNG2_POISEUILLE_SCALE,
    ):
        rung2_mu = eval_anchor_t0_mu(graph, times=times, gamma_mode=RUNG2_GAMMA_MODE)

    clot1 = _clot_timeline(
        data, args.anchor, phys, bio, device, times, gamma_mode="comsol_sr", nucleation=True
    )
    clot2 = _clot_timeline(
        data,
        args.anchor,
        phys,
        bio,
        device,
        times,
        gamma_mode=RUNG2_GAMMA_MODE,
        gamma_scale=RUNG2_GAMMA_SCALE,
        poiseuille_scale=RUNG2_POISEUILLE_SCALE,
        nucleation=True,
    )

    t0_1 = next(r for r in rung1_mu.times if r["time"] == 0)
    t53_1 = next(r for r in rung1_mu.times if r["time"] == times[-1])
    t0_2 = next(r for r in rung2_mu.times if r["time"] == 0)
    t53_2 = next(r for r in rung2_mu.times if r["time"] == times[-1])
    c53_1 = next(r for r in clot1 if r["time"] == times[-1])
    c53_2 = next(r for r in clot2 if r["time"] == times[-1])

    gates = {
        "rung1_bulk_t0": 0.98 <= float(t0_1["ratio_median_bulk"]) <= 1.02,
        "rung1_growth_t_last": 0.95 <= float(t53_1["ratio_median_growth"]) <= 1.05,
        "rung1_pearson_growth_t_last": float(t53_1["pearson_growth"]) >= 0.95,
        "rung2_bulk_t0": 0.95 <= float(t0_2["ratio_median_bulk"]) <= 1.05,
        "rung2_clot_f1_matches_rung1": abs(float(c53_1["clot_f1"]) - float(c53_2["clot_f1"])) < 0.01,
        "rung1_clot_f1_nuc_t_last": float(c53_1["clot_f1"]) >= 0.85,
    }

    payload = {
        "anchor": args.anchor,
        "times": times,
        "rung1": {
            "label": "GT flow + GT species + COMSOL spf.sr",
            "physics_env": env1,
            "mu": rung1_mu.to_dict(),
            "clot_nucleation": clot1,
        },
        "rung2": {
            "label": "GT flow + GT species + proxy gamma (no spf.sr)",
            "gamma_mode": RUNG2_GAMMA_MODE,
            "gamma_scale": RUNG2_GAMMA_SCALE,
            "poiseuille_scale": RUNG2_POISEUILLE_SCALE,
            "mu": rung2_mu.to_dict(),
            "clot_nucleation": clot2,
        },
        "gates": gates,
        "all_gates_pass": all(gates.values()),
    }

    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"[OK] {args.anchor} -> {out}")
    print(f"[i] rung1 t=0 bulk_ratio={t0_1['ratio_median_bulk']:.4f}")
    print(
        f"[i] rung1 t={times[-1]} growth_ratio={t53_1['ratio_median_growth']:.4f} "
        f"r_growth={t53_1['pearson_growth']:.4f}"
    )
    print(f"[i] rung2 t=0 bulk_ratio={t0_2['ratio_median_bulk']:.4f}")
    print(
        f"[i] rung2 t={times[-1]} growth_ratio={t53_2['ratio_median_growth']:.4f} "
        f"r_growth={t53_2['pearson_growth']:.4f}"
    )
    print(
        f"[i] clot F1 nuc t={times[-1]} rung1={c53_1['clot_f1']:.3f} "
        f"rung2={c53_2['clot_f1']:.3f}"
    )
    print(f"[i] gates={gates} all_pass={payload['all_gates_pass']}")
    return 0 if payload["all_gates_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
