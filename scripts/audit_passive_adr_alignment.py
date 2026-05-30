"""GT-only ADR residual audit: masks and formulation modes on anchor graphs.

Usage:
  python scripts/audit_passive_adr_alignment.py --anchor patient007
  python scripts/audit_passive_adr_alignment.py --anchor patient007 --all-formulations
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import torch

from src.config import BiochemConfig, PhysicsConfig, VesselConfig
from src.core_physics.biochem_physics_kernels import BiochemPhysicsKernels
from src.core_physics.physics_kernels import PhysicsKernels
from src.training.biochem_supervision_masks import (
    resolve_adr_node_mask,
    resolve_data_bio_supervision_mask,
    supervision_mask_times_mode,
)
from src.architecture.gnode_biochem import biochem_truth_node_mask

_FORMULATIONS = (
    "convective_nd",
    "log",
    "relative_nd",
    "transport_only",
    "reaction_only",
)


def _load_anchor(stem: str):
    root = _REPO
    anchor_dir = root / VesselConfig(phase="biochem_anchors").graph_output_dir
    path = anchor_dir / f"{stem}.pt"
    if not path.is_file():
        return None, path
    return torch.load(path, map_location="cpu", weights_only=False), path


def _props(data, core: PhysicsKernels) -> dict:
    props = core._get_geometric_props(data)
    n = int(data.num_nodes)
    u_ref = data.u_ref if torch.is_tensor(data.u_ref) else torch.tensor([float(data.u_ref)])
    d_bar = data.d_bar if torch.is_tensor(data.d_bar) else torch.tensor([float(data.d_bar)])
    if u_ref.numel() == 1:
        u_ref = u_ref.view(1).expand(n)
    if d_bar.numel() == 1:
        d_bar = d_bar.view(1).expand(n)
    props["u_ref"] = u_ref
    props["d_bar"] = d_bar
    return props


def _eval(
    kernels: BiochemPhysicsKernels,
    *,
    bio,
    vel,
    props,
    data,
    dC,
    mask,
    fast_transient: bool,
    residual_mode: str,
    species_scope: str,
) -> tuple[float, float, int]:
    lf, ls = kernels.biochem_adr_residual(
        bio,
        vel,
        props,
        data,
        d_pred_dt=dC,
        node_mask=mask,
        fast_transient=fast_transient,
        residual_mode=residual_mode,
        species_scope=species_scope,
    )
    n_m = int(mask.sum().item()) if mask is not None else int(data.num_nodes)
    return float(lf.item()), float(ls.item()), n_m


def audit_masks(data, *, bio_cfg: BiochemConfig, residual_mode: str) -> dict:
    phys = PhysicsConfig(phase="biochem")
    core = PhysicsKernels(phys_cfg=phys)
    kernels = BiochemPhysicsKernels(bio_cfg, core)
    props = _props(data, core)
    device = torch.device("cpu")

    y0 = data.y[0].detach()
    y1 = data.y[1].detach()
    dt = float((data.t[1] - data.t[0]).item()) if hasattr(data, "t") else 1.0
    dt = max(dt, 1e-9)
    d_dt = (y1 - y0) / dt

    vel = y1[:, 0:2]
    bio = y1[:, 4:13]
    dC = d_dt[:, 4:13]
    target_series = data.y[:2].detach()

    truth = biochem_truth_node_mask(data, int(data.num_nodes), device)
    os.environ.setdefault("BIOCHEM_DATA_BIO_MASK_MODE", "clot_band")

    out: dict = {}
    for label, env_mode in (("global", "global"), ("match_nowall", "match_data_bio")):
        os.environ["BIOCHEM_ADR_MASK_MODE"] = env_mode
        if label == "match_nowall":
            os.environ["BIOCHEM_ADR_EXCLUDE_WALL"] = "1"
        else:
            os.environ.pop("BIOCHEM_ADR_EXCLUDE_WALL", None)
        mask = resolve_adr_node_mask(
            data=data,
            device=device,
            truth_mask=truth,
            target_series=target_series,
            bio_cfg=bio_cfg,
            kernels=kernels,
        )
        lf, ls, n_m = _eval(
            kernels,
            bio=bio,
            vel=vel,
            props=props,
            data=data,
            dC=dC,
            mask=mask,
            fast_transient=False,
            residual_mode=residual_mode,
            species_scope="all",
        )
        out[label] = {"L_ADR_F": lf, "L_ADR_S": ls, "mask_n": n_m}
    g = out["global"]["L_ADR_S"]
    m = out["match_nowall"]["L_ADR_S"]
    out["match_to_global_slow"] = m / max(g, 1e-30)
    return out


def audit_formulations(data, *, bio_cfg: BiochemConfig) -> dict:
    phys = PhysicsConfig(phase="biochem")
    core = PhysicsKernels(phys_cfg=phys)
    kernels = BiochemPhysicsKernels(bio_cfg, core)
    props = _props(data, core)
    device = torch.device("cpu")

    y1 = data.y[1].detach()
    y0 = data.y[0].detach()
    dt = max(float((data.t[1] - data.t[0]).item()) if hasattr(data, "t") else 1.0, 1e-9)
    d_dt = (y1 - y0) / dt
    vel = y1[:, 0:2]
    bio = y1[:, 4:13]
    dC = d_dt[:, 4:13]
    target_series = data.y[:2].detach()
    truth = biochem_truth_node_mask(data, int(data.num_nodes), device)

    os.environ["BIOCHEM_ADR_MASK_MODE"] = "match_data_bio"
    os.environ["BIOCHEM_ADR_EXCLUDE_WALL"] = "1"
    mask = resolve_adr_node_mask(
        data=data,
        device=device,
        truth_mask=truth,
        target_series=target_series,
        bio_cfg=bio_cfg,
        kernels=kernels,
    )

    out: dict = {}
    for mode in _FORMULATIONS:
        lf, ls, n_m = _eval(
            kernels,
            bio=bio,
            vel=vel,
            props=props,
            data=data,
            dC=dC,
            mask=mask,
            fast_transient=False,
            residual_mode=mode,
            species_scope="all",
        )
        out[mode] = {"L_ADR_S": ls, "L_ADR_F": lf, "mask_n": n_m}
    base = out["convective_nd"]["L_ADR_S"]
    for mode in _FORMULATIONS:
        out[mode]["vs_convective_nd"] = out[mode]["L_ADR_S"] / max(base, 1e-30)
    return out


def audit_mask_times(data, *, bio_cfg: BiochemConfig) -> dict:
    """Compare supervision mask node counts: last timestep vs union over full ``data.y``."""
    device = torch.device("cpu")
    n = int(data.num_nodes)
    truth = biochem_truth_node_mask(data, n, device)
    full_y = data.y.detach()
    os.environ.setdefault("BIOCHEM_DATA_BIO_MASK_MODE", "clot_band")
    os.environ["BIOCHEM_ADR_MASK_MODE"] = "match_data_bio"
    os.environ["BIOCHEM_ADR_EXCLUDE_WALL"] = "1"

    phys = PhysicsConfig(phase="biochem")
    core = PhysicsKernels(phys_cfg=phys)
    kernels = BiochemPhysicsKernels(bio_cfg, core)

    out: dict = {}
    for label, times in (("last", "last"), ("union", "union")):
        os.environ["BIOCHEM_SUPERVISION_MASK_TIMES"] = times
        sup = resolve_data_bio_supervision_mask(
            data=data,
            device=device,
            truth_mask=truth,
            target_series=full_y,
            bio_cfg=bio_cfg,
            kernels=kernels,
        )
        adr = resolve_adr_node_mask(
            data=data,
            device=device,
            truth_mask=truth,
            target_series=full_y,
            bio_cfg=bio_cfg,
            kernels=kernels,
        )
        out[label] = {
            "data_bio_mask_n": int(sup.sum().item()),
            "adr_mask_n": int(adr.sum().item()) if adr is not None else n,
            "times_mode": supervision_mask_times_mode(),
        }
    if out["last"]["data_bio_mask_n"] > 0:
        out["union_over_last"] = out["union"]["data_bio_mask_n"] / float(out["last"]["data_bio_mask_n"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor", default="patient007")
    ap.add_argument("--fast-transient", action="store_true")
    ap.add_argument("--all-formulations", action="store_true")
    ap.add_argument(
        "--compare-mask-times",
        action="store_true",
        help="Print data_bio / ADR mask_n for SUPERVISION_MASK_TIMES=last vs union (full trajectory)",
    )
    args = ap.parse_args()

    os.environ.setdefault("BIOCHEM_DATA_BIO_MASK_MODE", "clot_band")
    data, path = _load_anchor(args.anchor.strip())
    if data is None:
        print(f"[WARN] anchor not found: {path}")
        return 0

    bio_cfg = BiochemConfig(phase="biochem")
    print(f"[i] GT ADR audit: {path.stem}")

    masks = audit_masks(data, bio_cfg=bio_cfg, residual_mode="convective_nd")
    print("  [masks] convective_nd on GT:")
    for k, v in masks.items():
        if isinstance(v, dict):
            print(f"    {k}: L_ADR_S={v['L_ADR_S']:.4e} n={v['mask_n']}")
        else:
            print(f"    match/global slow ratio: {v:.4f}")

    if args.all_formulations:
        forms = audit_formulations(data, bio_cfg=bio_cfg)
        print("  [formulations] match_nowall mask on GT (vs convective_nd):")
        for mode, v in forms.items():
            if mode in _FORMULATIONS:
                print(
                    f"    {mode:16s} L_ADR_S={v['L_ADR_S']:.4e}  "
                    f"ratio={v.get('vs_convective_nd', 1.0):.4f}"
                )

    if args.compare_mask_times:
        mt = audit_mask_times(data, bio_cfg=bio_cfg)
        print("  [mask_times] clot_band + match_data_bio + exclude_wall (full trajectory):")
        for label in ("last", "union"):
            v = mt[label]
            print(
                f"    {label:5s} data_bio_n={v['data_bio_mask_n']} adr_n={v['adr_mask_n']}"
            )
        if "union_over_last" in mt:
            print(f"    union/last data_bio_n ratio: {mt['union_over_last']:.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
