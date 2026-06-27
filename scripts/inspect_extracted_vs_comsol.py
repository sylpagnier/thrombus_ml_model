"""Data-inspection plots of EXTRACTED graph fields, decoded to COMSOL native units.

Renders the directly-extracted state fields (the ones our pipeline supervises) plus
velocity-derived shear quantities at a few snapshot times so the spatial pattern +
magnitude can be eyeballed against the COMSOL Surface plots:

  * ``FI`` / ``T`` / ``Mat`` / ``FG``          -- decoded species channels
  * ``mu_b*(mu2(FI)+mu1(Mat))``                -- effective viscosity (Pa*s)
  * ``spf.sr``                                 -- shear rate (1/s), from grad(u,v)
  * ``d(spf.sr,x)``                            -- x-derivative of shear rate (1/(m*s))

Shear quantities are reconstructed from the stored velocity with the graph gradient
operators ``G_x``/``G_y`` (same path COMSOL uses for ``d(.,x)``), dimensionalised by
``u_ref``/``d_bar``. They are most meaningful at ``t=0`` (baseline flow), which is why
the default time set leads with 0 s.

Each field gets its OWN interactive matplotlib window (zoom/pan toolbar), with the
requested times as columns; a PNG is also saved per field. Use ``--fields`` to plot
only the ones you flag (default: all).

Note: ``kpa_chem(Omega(...))`` is a COMSOL-derived analytic expression that is NOT
stored in the extract, so it is not plotted (only persisted state channels +
velocity-derived shear can be checked against COMSOL directly).

Examples:
  # viz everything (one window per field)
  python scripts/inspect_extracted_vs_comsol.py --patient patient007
  # only the flagged fields
  python scripts/inspect_extracted_vs_comsol.py --fields dsrx sr --times 0 15000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import torch

from src.config import STATE_CHANNEL_MU_EFF_ND, BiochemConfig, PhysicsConfig
from src.core_physics.clot_kinematics_fields import compute_clot_kinematics_fields
from src.core_physics.clot_phi_simple import species_log1p_nd_to_si
from src.training.biochem_species_scope import (
    FIBRINOGEN_CHANNEL,
    FI_CHANNEL,
    MAT_CHANNEL,
    THROMBIN_CHANNEL,
)

SPECIES_OFFSET = 4  # y[:, 4:16] are the 12 log1p species channels

# Short --fields key -> COMSOL label (also the field order in figures / filenames).
KEY_TO_LABEL = {
    "fi": "FI",
    "t": "T (thrombin)",
    "mat": "Mat",
    "fg": "FG",
    "mu": "mu_b*(mu2(FI)+mu1(Mat))",
    "sr": "spf.sr",
    "dsrx": "d(spf.sr,x)",
}
LABEL_TO_KEY = {v: k for k, v in KEY_TO_LABEL.items()}


def _props(data) -> dict:
    return {
        "u_ref": torch.as_tensor(data.u_ref, dtype=torch.float32).reshape(-1),
        "d_bar": torch.as_tensor(data.d_bar, dtype=torch.float32).reshape(-1),
    }


def _decode_fields(data, fidx: int, bio: BiochemConfig, phys: PhysicsConfig, props: dict) -> dict:
    """Decode one stored frame into COMSOL-native units, keyed by COMSOL label.

    Returns label -> (values, units, cmap, fixed (vmin, vmax) | None, symmetric bool).
    """
    y = data.y[fidx].float()
    sp = species_log1p_nd_to_si(y[:, SPECIES_OFFSET : SPECIES_OFFSET + 12], bio)
    fg = torch.expm1(y[:, SPECIES_OFFSET + FIBRINOGEN_CHANNEL].clamp(-10, 8)) * bio.c_Fg0 * 1e3
    th = torch.expm1(y[:, SPECIES_OFFSET + THROMBIN_CHANNEL].clamp(-10, 8)) * bio.c_pT0 * 1e3
    mu = phys.viscosity_nd_to_si(y[:, STATE_CHANNEL_MU_EFF_ND])

    kf = compute_clot_kinematics_fields(data, y[:, 0], y[:, 1], bio, props)

    return {
        "FI": (sp[:, FI_CHANNEL], "uM", "jet", None, False),
        "T (thrombin)": (th, "uM", "jet", None, False),
        "Mat": (sp[:, MAT_CHANNEL], "plt (crit 2e7)", "jet", None, False),
        "FG": (fg, "uM", "jet", None, False),
        "mu_b*(mu2(FI)+mu1(Mat))": (mu, "Pa*s", "bwr", (0.04, 0.1), False),
        "spf.sr": (kf.gamma_si, "1/s", "jet", None, False),
        # jet + symmetric range reproduces COMSOL RainbowClassic (green-centred at 0).
        "d(spf.sr,x)": (kf.dgamma_dx_phys, "1/(m*s)", "jet", None, True),
    }


def _build_triangulation(pos: np.ndarray, edge_index: torch.Tensor,
                         max_edge_factor: float = 2.5) -> mtri.Triangulation:
    """Delaunay triangulation with long triangles masked off.

    We have mesh *edges* (``edge_index``) but not faces, so we Delaunay-triangulate the
    node cloud and drop any triangle whose longest edge greatly exceeds the median mesh
    edge length. That removes spurious triangles that would otherwise bridge the lumen
    or cut across the concave bend, giving a COMSOL-like filled domain.
    """
    x, y = pos[:, 0], pos[:, 1]
    tri = mtri.Triangulation(x, y)
    ei = edge_index.cpu().numpy()
    elen = np.hypot(x[ei[0]] - x[ei[1]], y[ei[0]] - y[ei[1]])
    l_med = float(np.median(elen)) if elen.size else 0.0
    t = tri.triangles
    e0 = np.hypot(x[t[:, 0]] - x[t[:, 1]], y[t[:, 0]] - y[t[:, 1]])
    e1 = np.hypot(x[t[:, 1]] - x[t[:, 2]], y[t[:, 1]] - y[t[:, 2]])
    e2 = np.hypot(x[t[:, 2]] - x[t[:, 0]], y[t[:, 2]] - y[t[:, 0]])
    max_edge = np.maximum(np.maximum(e0, e1), e2)
    if l_med > 0.0:
        tri.set_mask(max_edge > max_edge_factor * l_med)
    return tri


def _nearest_frames(t_vec: torch.Tensor, times: list[float]) -> list[tuple[float, int, float]]:
    out = []
    for tq in times:
        idx = int(torch.argmin((t_vec - tq).abs()).item())
        out.append((tq, idx, float(t_vec[idx].item())))
    return out


def inspect_patient(graph_path: Path, times: list[float], out_dir: Path,
                    labels: list[str], clip_pct: float = 100.0,
                    smooth: bool = True) -> list[Path]:
    """One figure PER field (time snapshots as columns). Returns saved PNG paths."""
    bio, phys = BiochemConfig(), PhysicsConfig()
    d = torch.load(graph_path, map_location="cpu", weights_only=False)
    props = _props(d)
    pos = d.x[:, :2].float().cpu().numpy()
    t_vec = torch.as_tensor(d.t).reshape(-1).float()
    frames = _nearest_frames(t_vec, times)
    marker = max(0.4, min(6.0, 40000.0 / max(pos.shape[0], 1)))
    tri = _build_triangulation(pos, d.edge_index) if smooth else None

    # Decode each requested frame once (kinematics is the costly part), then slice
    # per field so each gets its own uncluttered figure.
    per_frame = [_decode_fields(d, fidx, bio, phys, props) for (_, fidx, _) in frames]

    out_dir.mkdir(parents=True, exist_ok=True)
    out_paths: list[Path] = []
    n_cols = len(frames)
    for label in labels:
        units, cmap = per_frame[0][label][1], per_frame[0][label][2]
        fig, axes = plt.subplots(1, n_cols, figsize=(4.8 * n_cols, 4.8), squeeze=False)
        for c, (tq, fidx, tactual) in enumerate(frames):
            ax = axes[0][c]
            vals, _units, _cmap, fixed, symmetric = per_frame[c][label]
            v = vals.detach().cpu().numpy()
            if fixed is not None:
                vmin, vmax = fixed
            elif symmetric:
                # symmetric range (green-centred jet, COMSOL-style). Default uses the
                # raw max so the colorbar matches COMSOL (~+/-800); --clip-pct < 100
                # trims extreme wall nodes to expose mid-range bulk structure.
                if clip_pct >= 100.0:
                    m = float(max(abs(float(v.min())), abs(float(v.max())), 1e-12))
                else:
                    m = float(max(torch.quantile(vals.abs(), clip_pct / 100.0).item(), 1e-12))
                vmin, vmax = -m, m
            else:
                vmin, vmax = float(v.min()), float(max(v.max(), v.min() + 1e-12))

            if smooth:
                levels = np.linspace(vmin, vmax, 80)
                if levels[0] == levels[-1]:
                    levels = np.linspace(vmin - 1e-9, vmax + 1e-9, 80)
                mappable = ax.tricontourf(tri, v, levels=levels, cmap=cmap, extend="both")
            else:
                mappable = ax.scatter(pos[:, 0], pos[:, 1], c=v, s=marker, cmap=cmap,
                                      vmin=vmin, vmax=vmax, linewidths=0, marker="s")
            ax.set_aspect("equal")
            ax.tick_params(labelsize=8)
            fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.02)
            rng = f"[{float(v.min()):.3g}, {float(v.max()):.3g}]"
            ax.set_title(f"Time={tactual:.0f} s\n{rng}", fontsize=9)

        fig.suptitle(f"{graph_path.stem}  --  {label}  [{units}]", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        out_path = out_dir / f"{graph_path.stem}_{LABEL_TO_KEY[label]}.png"
        fig.savefig(out_path, dpi=130)
        out_paths.append(out_path)
    return out_paths


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--patient", nargs="+", default=["patient007"],
                    help="Graph stem(s), e.g. patient007 patient002")
    ap.add_argument("--graph-dir", default="data/processed/graphs_biochem_anchors")
    ap.add_argument("--times", nargs="+", type=float, default=[0, 8000, 15000, 28000],
                    help="Physical times [s] to snapshot (nearest stored frame used)")
    ap.add_argument("--out", default="outputs/biochem/viz/data_inspection")
    ap.add_argument("--fields", nargs="+", default=["all"],
                    help=f"Fields to plot (default all). Keys: {', '.join(KEY_TO_LABEL)}")
    ap.add_argument("--clip-pct", type=float, default=100.0,
                    help="Symmetric-field colorbar percentile (100=raw max, matches "
                         "COMSOL; e.g. 98 trims wall outliers to show bulk structure)")
    ap.add_argument("--scatter", action="store_true",
                    help="Render raw per-node scatter instead of smooth tricontourf")
    ap.add_argument("--no-show", action="store_true",
                    help="Skip the interactive windows (just save PNGs)")
    args = ap.parse_args()

    if args.no_show:
        matplotlib.use("Agg")

    keys = list(KEY_TO_LABEL) if "all" in [k.lower() for k in args.fields] \
        else [k.lower() for k in args.fields]
    bad = [k for k in keys if k not in KEY_TO_LABEL]
    if bad:
        ap.error(f"unknown --fields {bad}; choose from {list(KEY_TO_LABEL)} or 'all'")
    labels = [KEY_TO_LABEL[k] for k in keys]

    graph_dir = Path(args.graph_dir)
    out_dir = Path(args.out)
    any_fig = False
    for stem in args.patient:
        gp = graph_dir / f"{stem}.pt"
        if not gp.exists():
            print(f"[WARN] missing graph: {gp}")
            continue
        for out_path in inspect_patient(gp, list(args.times), out_dir, labels,
                                        args.clip_pct, smooth=not args.scatter):
            any_fig = True
            print(f"[OK] {out_path}")

    if any_fig and not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
