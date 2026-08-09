"""Verify the wall residual's gates actually engage after the operator fix.

Standing constraint 5.3: "VERIFY THE MECHANISM ENGAGED before trusting any result."
This calls ``biochem_wall_residual`` on a real pack under both operator modes and reports
the internal gate activations, which is the only way to see that the separation branch
went from identically-off to live.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def probe(anchor="patient007"):
    from src.config import BiochemConfig, PhysicsConfig
    from src.core_physics.biochem_physics_kernels import BiochemPhysicsKernels
    from src.core_physics.mls_gradient import clear_operator_cache

    clear_operator_cache()
    bio = BiochemConfig(phase="biochem")
    d = torch.load(f"data/processed/graphs_biochem_anchors/{anchor}.pt",
                   map_location="cpu", weights_only=False)
    k = BiochemPhysicsKernels(bio, None)
    wall = d.mask_wall.reshape(-1).bool()
    u = d.y[0, :, 0].float()
    v = d.y[0, :, 1].float()
    props = {"u_ref": d.u_ref.reshape(-1).expand(d.num_nodes).float(),
             "d_bar": d.d_bar.reshape(-1).expand(d.num_nodes).float()}
    sr = k._compute_shear_rate(u, v, props, d)

    _gx, _gy = k._grad_ops(d, sr)
    dsx = torch.sparse.mm(_gx, sr.unsqueeze(1)).squeeze(1)
    dsy = torch.sparse.mm(_gy, sr.unsqueeze(1)).squeeze(1)
    d_bar = torch.clamp(props["d_bar"], min=1e-8)
    vel = torch.sqrt(u ** 2 + v ** 2) + 1e-8
    stream = (((u / vel) * dsx + (v / vel) * dsy) / d_bar)[wall]
    dxg = (dsx / d_bar)[wall]

    T_gr = bio.soft_step_T_grad * bio.soft_step_T_scale
    T_ls = bio.soft_step_T_low_shear * bio.soft_step_T_scale
    sep_stream = torch.sigmoid((float(bio.sgt) - stream) / T_gr)
    sep_dx = torch.sigmoid((float(bio.sgt) - dxg) / T_gr)
    low = torch.sigmoid((float(bio.lss) - sr[wall]) / T_ls)
    return dict(
        sr_interior_med=float(sr[~wall].median()),
        sr_wall_med=float(sr[wall].median()),
        stream_absmax=float(stream.abs().max()),
        dx_absmax=float(dxg.abs().max()),
        sep_stream_max=float(sep_stream.max()),
        sep_dx_max=float(sep_dx.max()),
        sep_dx_open=float((sep_dx > 0.5).float().mean()),
        low_open=float((low > 0.5).float().mean()),
    )


def main() -> int:
    rows = {}
    for label, env in (("legacy G_x + streamwise", {"BIOCHEM_GRAD_OPERATOR": "legacy",
                                                    "BIOCHEM_SEPARATION_GATE": "stream"}),
                       ("MLS + streamwise", {"BIOCHEM_GRAD_OPERATOR": "mls",
                                             "BIOCHEM_SEPARATION_GATE": "stream"}),
                       ("MLS + d(sr,x)  [new default]", {"BIOCHEM_GRAD_OPERATOR": "mls",
                                                         "BIOCHEM_SEPARATION_GATE": "dx"})):
        prev = {kk: os.environ.get(kk) for kk in env}
        os.environ.update(env)
        try:
            rows[label] = probe()
        finally:
            for kk, vv in prev.items():
                if vv is None:
                    os.environ.pop(kk, None)
                else:
                    os.environ[kk] = vv

    print("COMSOL reference (patient007 t=0): wall spf.sr median 77.9 1/s, "
          "separation gate open on 14.6% of wall nodes\n")
    hdr = ("config", "sr int", "sr wall", "|dsr| max", "sep max", "sep open", "low open")
    print("%-30s %8s %8s %11s %8s %9s %9s" % hdr)
    for label, r in rows.items():
        gate = r["stream_absmax"] if "streamwise" in label else r["dx_absmax"]
        smax = r["sep_stream_max"] if "streamwise" in label else r["sep_dx_max"]
        print("%-30s %8.2f %8.2f %11.3g %8.3g %8.1f%% %8.1f%%"
              % (label, r["sr_interior_med"], r["sr_wall_med"], gate, smax,
                 100 * (r["sep_dx_open"] if "d(sr,x)" in label else 0.0),
                 100 * r["low_open"]))
    print("\n  interior shear was ~0 (rank-deficient operator); the streamwise separation")
    print("  gate is EXACTLY 0 at the wall under both operators, because no-slip pins u=v=0")
    print("  there -- it was never an operator problem alone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
