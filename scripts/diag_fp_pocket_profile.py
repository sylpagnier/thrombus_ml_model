"""What IS the distant false-positive pocket? One rollout, full node-level dump + profile.

`diag_fp_geography` showed 97.1% of patient020's FPs sit a median 56 hops from any GT
clot -- a second, wrong pocket, not an adjacent halo. That refutes the premise
WALL_MODEL_PLAN s5 used to park `WG_prec_physfp` ("FPs are the low-speed halo ... a
speed-based FP penalty cannot separate them").

The open question is whether the wrong pocket is *physically distinguishable* from the
right one. If the FP pocket is high-flow spray, a speed/shear gate separates it cheaply.
If it is an equally stagnant pocket, then the model is choosing correctly on local
physics and the discriminator has to be something non-local -- which is a much harder
architectural problem.

Dumps every per-node field to .npz so follow-up analysis is free.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.eda_clot_burden import hops_from_wall  # noqa: E402
from scripts.eval_mat_growth_simple import _apply_ckpt_recipe, _load_static  # noqa: E402
from src.config import BiochemConfig, PhysicsConfig  # noqa: E402
from src.core_physics.species_deploy_rollout import reset_species_rollout_flow_cache  # noqa: E402
from src.core_physics.species_pushforward_continuous import (  # noqa: E402
    clear_offwall_model_cache,
    deploy_clot_phi_fields,
    deploy_species_rollout_series,
    load_continuous_bundle,
)
from src.core_physics.t0_device import require_cuda_device  # noqa: E402
from src.utils import species_channels as sc  # noqa: E402
from src.utils.kinematics_inference import (  # noqa: E402
    load_kinematics_predictor,
    resolve_kinematics_checkpoint,
)
from src.utils.paths import get_project_root  # noqa: E402

ANCHOR_DIR = get_project_root() / "data/processed/graphs_biochem_anchors"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--anchor", default="patient020")
    ap.add_argument("--out", default="outputs/biochem/eda/fp_geo/p020_nodes.npz")
    args = ap.parse_args()

    root = get_project_root()
    device = require_cuda_device()
    ckpt = Path(args.ckpt)
    if not ckpt.is_absolute():
        ckpt = root / ckpt

    clear_offwall_model_cache()
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    meta = dict(payload.get("meta") or {})
    _apply_ckpt_recipe(meta, label="mat_growth_simple", ckpt_path=ckpt)
    bundle = load_continuous_bundle(ckpt, device=device, quiet=True)
    model = bundle.model
    wall_hops = int(meta.get("wall_hops", 3))
    kine = load_kinematics_predictor(
        str(resolve_kinematics_checkpoint()), device, phys_cfg=PhysicsConfig(phase="kinematics")
    )
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")

    anc = args.anchor
    reset_species_rollout_flow_cache()
    data = torch.load(ANCHOR_DIR / f"{anc}.pt", map_location=device, weights_only=False)
    static = _load_static(data, device, kine, wall_hops, anc)
    static["n_times"] = int(data.y.shape[0])
    print("rolling out...", flush=True)
    series, data = deploy_species_rollout_series(
        model, data, static, phys, bio, device, flow_source="kinematics"
    )
    phi_pred, phi_gt, wall_mask, t_eval = deploy_clot_phi_fields(
        data, series, static, phys, bio, device, flow_source="kinematics"
    )

    n = int(data.num_nodes)
    wall = data.mask_wall.reshape(-1).bool().cpu()
    hops = hops_from_wall(data.edge_index.cpu(), wall, n, 6)
    sp0 = (data.y[0, :, 0] ** 2 + data.y[0, :, 1] ** 2).sqrt()
    spT = (data.y[t_eval, :, 0] ** 2 + data.y[t_eval, :, 1] ** 2).sqrt()
    u0p = getattr(data, "u0_pred", None)
    v0p = getattr(data, "v0_pred", None)
    sp_kine = (
        (u0p.to(device) ** 2 + v0p.to(device) ** 2).sqrt() if u0p is not None else torch.zeros(n, device=device)
    )
    mat = series[t_eval][:, sc.y_index("Mat")]

    out = dict(
        pred=(phi_pred > 0.5).cpu().numpy(), gt=(phi_gt > 0.5).cpu().numpy(),
        pos=data.x[:, :2].cpu().numpy(), wall=wall.numpy(), hops=hops.numpy(),
        sp0=sp0.cpu().numpy(), spT=spT.cpu().numpy(), sp_kine=sp_kine.cpu().numpy(),
        mat=mat.cpu().numpy(), t_eval=np.array([t_eval]),
    )
    op = Path(args.out)
    if not op.is_absolute():
        op = root / op
    op.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(op, **out)
    print(f"[save] {op}")

    pr, gtm = out["pred"], out["gt"]
    tp, fp = pr & gtm, pr & ~gtm
    print(f"\nTP={tp.sum()} FP={fp.sum()}")
    print(f"{'field':>10} {'TP mean':>12} {'FP mean':>12} {'all-band mean':>14}")
    band = out["hops"] <= 3
    for k in ("sp0", "spT", "sp_kine", "mat"):
        v = out[k]
        print(f"{k:>10} {v[tp].mean():12.6f} {v[fp].mean():12.6f} {v[band].mean():14.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
