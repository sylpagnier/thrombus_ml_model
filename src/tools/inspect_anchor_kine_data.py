"""Inspect biochem anchor kine priors vs COMSOL t=0 GT (18ch ``data.x``).

Example:
    python -m src.tools.inspect_anchor_kine_data
    python -m src.tools.inspect_anchor_kine_data --stem patient007
    python -m src.tools.inspect_anchor_kine_data --summary
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.config import NodeFeat, VesselConfig
from src.data_gen.lib.node_feature_assembly import kinematics_uv_prior_max
from src.utils.channel_schema import infer_missing_schema
from src.utils.kinematics_paths import resolve_kinematics_anchor_graph


def _prior_stats(data) -> dict:
    x = data.x
    out = {
        "x_ch": int(x.shape[1]),
        "x_schema": str(getattr(data, "x_schema", "?")),
        "uv_prior_max": float(x[:, NodeFeat.UV_PRIOR].abs().max().item()),
        "mu_prior_mean": float(x[:, NodeFeat.MU_PRIOR].mean().item()),
        "rheo_mean": float(x[:, 10:11].mean().item()),
        "width_max": float(x[:, NodeFeat.WIDTH_ND].max().item()),
    }
    if hasattr(data, "y") and data.y is not None:
        if data.y.dim() == 3:
            gt = data.y[0, :, :4]
        else:
            gt = data.y[:, :4]
        pred_uv = x[:, NodeFeat.UV_PRIOR]
        rel_uv = float((pred_uv - gt[:, :2]).norm() / (gt[:, :2].norm() + 1e-8))
        rel_mu = float((x[:, NodeFeat.MU_PRIOR].reshape(-1) - gt[:, 3]).norm() / (gt[:, 3].norm() + 1e-8))
        out["rel_uv_prior_vs_gt"] = rel_uv
        out["rel_mu_prior_vs_gt"] = rel_mu
    if hasattr(data, "d_bar"):
        out["d_bar_mm"] = float(data.d_bar.reshape(-1)[0].item()) * 1000.0
    if hasattr(data, "re_actual"):
        out["re"] = float(data.re_actual.reshape(-1)[0].item())
    if hasattr(data, "centerline_source"):
        out["centerline"] = str(getattr(data, "centerline_source", ""))
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stem", type=str, default="", help="Single patient stem.")
    p.add_argument("--summary", action="store_true", help="Table only (no per-field dump).")
    p.add_argument("--rheology", type=str, default="carreau", choices=("carreau", "newtonian"))
    args = p.parse_args()

    anchor_dir = Path(VesselConfig(phase="biochem_anchors").graph_output_dir)
    stems = [args.stem.strip()] if args.stem.strip() else sorted(p.stem for p in anchor_dir.glob("patient*.pt"))
    if not stems:
        print(f"[ERR] no patient*.pt under {anchor_dir}")
        return 1

    print(f"[i] biochem anchors: {anchor_dir}")
    print(f"[i] prefer kine sidecar rheology={args.rheology}")
    print()

    for stem in stems:
        bio_p = anchor_dir / f"{stem}.pt"
        if not bio_p.is_file():
            print(f"[WARN] missing {bio_p.name}")
            continue
        kpath = resolve_kinematics_anchor_graph(stem, rheology=args.rheology)
        path = kpath if kpath is not None else bio_p
        tag = "kine_anchor" if kpath is not None else "biochem_only"
        data = torch.load(path, map_location="cpu", weights_only=False)
        data = infer_missing_schema(data, phase_hint="biochem")
        st = _prior_stats(data)
        line = (
            f"{stem:12s} [{tag:11s}] x={st['x_ch']}ch rheo={st['rheo_mean']:.2f} "
            f"uv_max={st['uv_prior_max']:.3f} mu_mean={st['mu_prior_mean']:.3f} "
            f"width={st['width_max']:.2f}"
        )
        if "d_bar_mm" in st:
            line += f" d_bar={st['d_bar_mm']:.1f}mm Re={st.get('re', 0):.0f}"
        if "rel_uv_prior_vs_gt" in st:
            line += f" rel_uv={st['rel_uv_prior_vs_gt']:.4f} rel_mu={st['rel_mu_prior_vs_gt']:.4f}"
        if "centerline" in st and st["centerline"]:
            line += f" cl={st['centerline']}"
        print(line)
        if not args.summary:
            print(f"    path={path}")
            print(f"    schema={st['x_schema']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
