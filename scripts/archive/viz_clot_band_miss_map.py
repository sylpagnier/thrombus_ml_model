"""Viz: GT viscosity growth map + wall+1hop band misses (unreachable / hop>=2 clots).

Run:
  python scripts/viz_clot_band_miss_map.py --anchor patient007
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from src.config import PhysicsConfig, STATE_CHANNEL_MU_EFF_ND, VesselConfig
from src.core_physics.clot_growth_masks import gt_growth_commit_mask_at_time
from src.core_physics.clot_phi_simple import (
    cap_mu_eff_si,
    clot_phi_thresh_si,
    gt_mu_anchor_cap_si,
)
from src.data_gen.lib.centerline_utils import resolve_anchor_mesh_path
from src.data_gen.lib.mesh_triangle6_edges import edge_index_from_mesh_path


def bfs_hop_from_wall(edge_index: torch.Tensor, wall: torch.Tensor, n: int) -> torch.Tensor:
    dist = torch.full((n,), -1, dtype=torch.long)
    dist[wall] = 0
    row, col = edge_index[0], edge_index[1]
    frontier = wall.clone()
    h = 0
    while bool(frontier.any()):
        h += 1
        nxt = torch.zeros(n, dtype=torch.bool)
        nxt[col[frontier[row]]] = True
        nxt[row[frontier[col]]] = True
        nxt &= dist < 0
        dist[nxt] = h
        frontier = nxt
        if h > 10000:
            break
    return dist


def _load_anchor(anchor: str, root: Path) -> object:
    graph_path = root / VesselConfig(phase="biochem_anchors").graph_output_dir / f"{anchor}.pt"
    if not graph_path.is_file():
        raise FileNotFoundError(graph_path)
    return torch.load(graph_path, map_location="cpu", weights_only=False)


def _classify_clot_nodes(
    clot: torch.Tensor,
    dist: torch.Tensor,
    *,
    band_hops: int,
) -> dict[str, np.ndarray]:
    c = clot.detach().cpu().numpy().astype(bool)
    d = dist.detach().cpu().numpy()
    in_band = c & (d >= 0) & (d <= int(band_hops))
    miss_unreach = c & (d < 0)
    miss_deep = c & (d > int(band_hops))
    miss_any = c & ~in_band
    return {
        "clot": c,
        "in_band": in_band,
        "miss_unreach": miss_unreach,
        "miss_deep": miss_deep,
        "miss_any": miss_any,
        "dist": d,
    }


def _scatter_growth_base(ax, pos: np.ndarray, growth: np.ndarray, *, s: float, vmax: float) -> None:
    bulk = growth <= 1e-6
    if bulk.any():
        ax.scatter(
            pos[bulk, 0],
            pos[bulk, 1],
            c="#d9d9d9",
            s=s * 0.55,
            linewidths=0,
            alpha=0.35,
            zorder=1,
        )
    pos_g = growth > 1e-6
    if pos_g.any():
        sc = ax.scatter(
            pos[pos_g, 0],
            pos[pos_g, 1],
            c=growth[pos_g],
            s=s,
            cmap="YlOrRd",
            vmin=0.0,
            vmax=vmax,
            linewidths=0,
            alpha=0.92,
            zorder=2,
        )
        plt.colorbar(sc, ax=ax, fraction=0.046, label="mu growth [Pa*s]")


def _scatter_band(ax, pos: np.ndarray, wall: np.ndarray, band: np.ndarray, *, s: float) -> None:
    off = ~(wall | band)
    if off.any():
        ax.scatter(pos[off, 0], pos[off, 1], c="#ececec", s=s * 0.4, linewidths=0, alpha=0.25, zorder=1)
    if band.any():
        ax.scatter(
            pos[band, 0],
            pos[band, 1],
            c="#9ecae1",
            s=s * 0.75,
            linewidths=0,
            alpha=0.55,
            zorder=2,
        )
    if wall.any():
        ax.scatter(
            pos[wall, 0],
            pos[wall, 1],
            c="#08519c",
            s=s * 0.9,
            linewidths=0,
            alpha=0.85,
            zorder=3,
        )


def _scatter_miss_overlay(
    ax,
    pos: np.ndarray,
    growth: np.ndarray,
    cls: dict[str, np.ndarray],
    *,
    s: float,
    vmax: float,
) -> None:
    _scatter_growth_base(ax, pos, growth, s=s, vmax=vmax)
    in_band = cls["in_band"]
    if in_band.any():
        ax.scatter(
            pos[in_band, 0],
            pos[in_band, 1],
            facecolors="none",
            edgecolors="#2ca02c",
            s=s * 2.2,
            linewidths=0.6,
            alpha=0.9,
            zorder=4,
        )
    miss_deep = cls["miss_deep"]
    if miss_deep.any():
        ax.scatter(
            pos[miss_deep, 0],
            pos[miss_deep, 1],
            marker="D",
            c="#ff7f0e",
            s=s * 3.5,
            linewidths=0.8,
            edgecolors="black",
            alpha=1.0,
            zorder=6,
        )
    miss_unreach = cls["miss_unreach"]
    if miss_unreach.any():
        ax.scatter(
            pos[miss_unreach, 0],
            pos[miss_unreach, 1],
            marker="X",
            c="#d62728",
            s=s * 4.5,
            linewidths=1.2,
            alpha=1.0,
            zorder=7,
        )


def _maybe_inset(ax, pos: np.ndarray, miss_mask: np.ndarray, growth: np.ndarray, *, s: float, vmax: float) -> None:
    if not bool(miss_mask.any()):
        return
    idx = np.where(miss_mask)[0]
    cx = float(pos[idx, 0].mean())
    cy = float(pos[idx, 1].mean())
    span = max(float(pos[idx, 0].ptp()), float(pos[idx, 1].ptp()), 1e-3)
    pad = span * 0.8
    axins = ax.inset_axes([0.02, 0.02, 0.36, 0.36])
    axins.set_xlim(cx - pad, cx + pad)
    axins.set_ylim(cy - pad, cy + pad)
    gmax = max(float(vmax), 0.10)
    axins.scatter(pos[:, 0], pos[:, 1], c="#e0e0e0", s=s * 0.15, linewidths=0, alpha=0.3)
    axins.scatter(
        pos[idx, 0],
        pos[idx, 1],
        c=growth[idx],
        s=s * 1.2,
        cmap="YlOrRd",
        vmin=0.0,
        vmax=gmax,
        linewidths=0,
        zorder=3,
    )
    axins.scatter(pos[idx, 0], pos[idx, 1], marker="X", c="#d62728", s=s * 2.0, linewidths=0.8, zorder=5)
    axins.set_xticks([])
    axins.set_yticks([])
    axins.set_title("miss zoom", fontsize=8)


def main() -> int:
    ap = argparse.ArgumentParser(description="GT mu growth map with wall+1hop band misses")
    ap.add_argument("--anchor", default="patient007")
    ap.add_argument("--band-hops", type=int, default=1, help="Wall + N hop supervision band (default 1)")
    ap.add_argument("--time-index", type=int, default=-1, help="Macro time index (-1 = final)")
    ap.add_argument("--scatter-size", type=float, default=2.5)
    ap.add_argument(
        "--triangle6-edges",
        action="store_true",
        help="Rebuild edge_index from anchor .nas/.msh (full P2 connectivity)",
    )
    ap.add_argument("--out", default="", help="Output PNG (default auto under reports/figures)")
    args = ap.parse_args()

    root = get_project_root()
    data = _load_anchor(args.anchor, root)
    phys = PhysicsConfig()
    dev = torch.device("cpu")
    n = int(data.num_nodes)
    t_final = int(data.y.shape[0]) - 1
    ti = int(args.time_index)
    if ti < 0:
        ti = t_final + 1 + ti
    ti = max(0, min(ti, t_final))

    wall = data.mask_wall.view(-1).bool() if data.mask_wall is not None else torch.zeros(n, dtype=torch.bool)
    edge_index = data.edge_index
    edge_tag = "corner"
    if args.triangle6_edges:
        mesh_path = resolve_anchor_mesh_path(root / VesselConfig(phase="biochem_anchors").mesh_input_dir, args.anchor)
        if mesh_path is None:
            raise FileNotFoundError(f"no .nas/.msh for {args.anchor}")
        edge_index = edge_index_from_mesh_path(mesh_path)
        edge_tag = "triangle6"
    dist = bfs_hop_from_wall(edge_index, wall, n)
    clot = gt_growth_commit_mask_at_time(data, ti, phys, dev)

    y = data.y[ti].to(dtype=torch.float32)
    mu = cap_mu_eff_si(phys.viscosity_nd_to_si(y[:, STATE_CHANNEL_MU_EFF_ND])).reshape(-1)
    anchor = gt_mu_anchor_cap_si(data, phys, dev).reshape(-1)
    growth = (mu - anchor).clamp(min=0.0).detach().cpu().numpy()
    thresh = float(clot_phi_thresh_si(phys))
    vmax = max(float(growth[clot].max()) if bool(clot.any()) else thresh, thresh * 1.5, 0.10)

    band = (dist >= 0) & (dist <= int(args.band_hops))
    cls = _classify_clot_nodes(clot, dist, band_hops=int(args.band_hops))

    pos = data.x[:, :2].detach().cpu().numpy()
    wall_np = wall.detach().cpu().numpy()
    band_np = band.detach().cpu().numpy()

    n_clot = int(clot.sum())
    n_in = int(cls["in_band"].sum())
    n_unr = int(cls["miss_unreach"].sum())
    n_deep = int(cls["miss_deep"].sum())
    rec = n_in / max(n_clot, 1)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    s = float(args.scatter_size)

    ax = axes[0]
    _scatter_growth_base(ax, pos, growth, s=s, vmax=vmax)
    ax.set_title(f"GT mu growth (t={ti})\nrelu(mu - mu_t0)", fontsize=10)
    ax.set_aspect("equal")
    ax.axis("off")

    ax = axes[1]
    _scatter_band(ax, pos, wall_np, band_np, s=s)
    ax.set_title(f"wall + {args.band_hops} hop band\n(n={int(band_np.sum())})", fontsize=10)
    ax.set_aspect("equal")
    ax.axis("off")

    ax = axes[2]
    _scatter_miss_overlay(ax, pos, growth, cls, s=s, vmax=vmax)
    _maybe_inset(ax, pos, cls["miss_any"], growth, s=s, vmax=vmax)
    ax.set_title(
        f"GT clots vs band (growth >= {thresh:.3f})\n"
        f"in-band={n_in}  miss={n_unr + n_deep}  rec={rec:.3f}",
        fontsize=10,
    )
    ax.set_aspect("equal")
    ax.axis("off")

    legend = [
        Patch(facecolor="#08519c", label="wall"),
        Patch(facecolor="#9ecae1", label=f"wall+{args.band_hops}hop band"),
        Patch(facecolor="none", edgecolor="#2ca02c", label=f"clot in band ({n_in})"),
        Patch(facecolor="#d62728", label=f"miss unreachable ({n_unr})"),
        Patch(facecolor="#ff7f0e", label=f"miss hop>{args.band_hops} ({n_deep})"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=5, fontsize=9, frameon=False)
    fig.suptitle(
        f"{args.anchor} -- growth-only GT clots outside wall+{args.band_hops}hop supervision",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.98])

    out = Path(args.out) if args.out.strip() else (
        reports_dir()
        / "figures"
        / "clot_band_miss"
        / f"{args.anchor}_wall{args.band_hops}hop_miss_{edge_tag}.png"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out}")
    print(
        f"[i] edges={edge_tag}  t={ti}  N_clot={n_clot}  in_band={n_in}  miss_unreach={n_unr}  "
        f"miss_deep={n_deep}  rec@wall+{args.band_hops}hop={rec:.3f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
