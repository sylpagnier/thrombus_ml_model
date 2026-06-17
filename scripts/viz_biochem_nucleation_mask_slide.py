"""Presentation viz: full vessel graph -> wall-band -> nucleation mask."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.config import (  # noqa: E402
    BiochemConfig,
    PhysicsConfig,
    STATE_CHANNEL_MU_EFF_ND,
    VesselConfig,
)
from src.core_physics.clot_growth_masks import resolve_t0_dgamma_wall_mask  # noqa: E402
from src.core_physics.clot_nucleation_mask import (  # noqa: E402
    resolve_commits_at_time,
    resolve_nucleation_eligibility,
    resolve_wall_mask,
)
from src.core_physics.clot_phi_simple import cap_mu_eff_si, clot_phi_mu_cap_si  # noqa: E402
from src.core_physics.species_snapshot_gnn import wall_band_mask  # noqa: E402
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402
from src.utils.channel_schema import BIO_Y_SCHEMA, assert_graph_schema, infer_missing_schema  # noqa: E402
from src.utils.paths import get_project_root  # noqa: E402

NAVY = "#1a2a4a"
MUTED = "#5a6478"
OFF = "#e8e8e8"


def _triangulation(pos: np.ndarray) -> mtri.Triangulation:
    triang = mtri.Triangulation(pos[:, 0], pos[:, 1])
    tri_pts = pos[triang.triangles]
    d1 = np.sum((tri_pts[:, 0, :] - tri_pts[:, 1, :]) ** 2, axis=1)
    d2 = np.sum((tri_pts[:, 1, :] - tri_pts[:, 2, :]) ** 2, axis=1)
    d3 = np.sum((tri_pts[:, 2, :] - tri_pts[:, 0, :]) ** 2, axis=1)
    max_edge_sq = np.max(np.vstack([d1, d2, d3]), axis=0)
    triang.set_mask(max_edge_sq > (np.median(max_edge_sq) * 10.0))
    return triang


def _draw_mesh(ax, pos: np.ndarray, vals: np.ndarray, *, cmap: str, vmin: float, vmax: float, title: str) -> None:
    triang = _triangulation(pos)
    tc = ax.tripcolor(triang, vals.reshape(-1), cmap=cmap, vmin=vmin, vmax=vmax, shading="gouraud")
    plt.colorbar(tc, ax=ax, fraction=0.046)
    ax.set_title(title, fontsize=11, color=NAVY, pad=8)
    ax.set_aspect("equal")
    ax.axis("off")


def _draw_graph_edges(
    ax,
    pos: np.ndarray,
    edge_index: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    color: str = "#c8c8c8",
    alpha: float = 0.25,
    lw: float = 0.35,
    max_edges: int = 25000,
) -> None:
    row, col = edge_index
    if mask is not None:
        m = mask.reshape(-1).astype(bool)
        keep = m[row] & m[col]
        row, col = row[keep], col[keep]
    n_e = int(row.shape[0])
    if n_e > max_edges:
        step = max(1, n_e // max_edges)
        row, col = row[::step], col[::step]
    p = pos
    segs = np.stack([p[row], p[col]], axis=1)
    for i in range(segs.shape[0]):
        ax.plot(segs[i, :, 0], segs[i, :, 1], color=color, alpha=alpha, lw=lw, zorder=1)


def _draw_graph_nodes(
    ax,
    pos: np.ndarray,
    active: np.ndarray,
    *,
    title: str,
    active_color: str = "#2d8f4e",
    inactive_color: str = OFF,
    s_active: float = 7.0,
    s_inactive: float = 2.5,
    show_colorbar: bool = False,
) -> None:
    act = active.reshape(-1).astype(bool)
    if (~act).any():
        ax.scatter(
            pos[~act, 0],
            pos[~act, 1],
            c=inactive_color,
            s=s_inactive,
            linewidths=0,
            alpha=0.55,
            zorder=2,
        )
    if act.any():
        ax.scatter(
            pos[act, 0],
            pos[act, 1],
            c=active_color,
            s=s_active,
            linewidths=0,
            alpha=0.95,
            zorder=3,
        )
        if show_colorbar:
            sm = plt.cm.ScalarMappable(cmap=plt.cm.Greens, norm=plt.Normalize(0, 1))
            plt.colorbar(sm, ax=ax, fraction=0.046, ticks=[0, 1])
    ax.set_title(title, fontsize=11, color=NAVY, pad=8)
    ax.set_aspect("equal")
    ax.axis("off")


def render_slide(
    *,
    anchor: str,
    time_index: int,
    pos: np.ndarray,
    n_all: int,
    wall: np.ndarray,
    band: np.ndarray,
    elig: np.ndarray,
    edge_index: np.ndarray,
    out_path: Path,
    style: str,
) -> None:
    n_wall = int(wall.sum())
    n_band = int(band.sum())
    n_elig = int(elig.sum())

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.82, bottom=0.06, wspace=0.22)

    if style == "mesh":
        all_vals = np.ones(n_all, dtype=float) * 0.15
        _draw_mesh(axes[0], pos, all_vals, cmap="Blues", vmin=0, vmax=1, title=f"Full mesh graph\nN={n_all} nodes")
        _draw_mesh(
            axes[1],
            pos,
            band.astype(float),
            cmap="Greens",
            vmin=0,
            vmax=1,
            title=f"Wall-band subgraph\nN={n_band} nodes (GraphSAGE domain)",
        )
        _draw_mesh(
            axes[2],
            pos,
            elig.astype(float),
            cmap="Greens",
            vmin=0,
            vmax=1,
            title=f"Nucleation mask\nN={n_elig} active nodes",
        )
    else:
        _draw_graph_edges(axes[0], pos, edge_index, color="#b0b8c8", alpha=0.18)
        _draw_graph_nodes(
            axes[0],
            pos,
            np.ones(n_all, dtype=bool),
            title=f"Full vessel graph\nN={n_all} nodes",
            active_color="#6b8cae",
            s_active=3.0,
        )
        _draw_graph_edges(axes[1], pos, edge_index, mask=band, color="#7fbf9b", alpha=0.35)
        _draw_graph_nodes(
            axes[1],
            pos,
            band,
            title=f"Wall-band subgraph\nN={n_band} nodes",
        )
        _draw_graph_edges(axes[2], pos, edge_index, mask=elig, color="#2d8f4e", alpha=0.55, lw=0.5)
        _draw_graph_nodes(
            axes[2],
            pos,
            elig,
            title=f"Restricted nucleation mask\nN={n_elig} nodes (wall | 1-hop commits)",
            s_active=10.0,
        )

    fig.suptitle(
        f"Biochem deploy graph restriction -- {anchor} t={time_index}",
        fontsize=12,
        color=NAVY,
        y=0.98,
    )
    fig.text(
        0.5,
        0.90,
        f"N={n_all} nodes  ->  wall-band N={n_band} (GraphSAGE)  ->  nucleation N={n_elig}",
        ha="center",
        fontsize=10,
        color=MUTED,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_diag(
    *,
    anchor: str,
    time_index: int,
    model_label: str,
    pos: np.ndarray,
    n_all: int,
    wall: np.ndarray,
    band: np.ndarray,
    t0_seed: np.ndarray,
    elig: np.ndarray,
    phi_gt: np.ndarray,
    mu_cap: np.ndarray,
    out_path: Path,
) -> None:
    """Six-panel mesh layout (clot_phi_masks style) for biochem deploy GNN."""
    n_wall = int(wall.sum())
    n_band = int(band.sum())
    n_t0 = int(t0_seed.sum())
    n_elig = int(elig.sum())
    cap = float(clot_phi_mu_cap_si())
    gt_pos = float(phi_gt[elig > 0.5].mean()) if (elig > 0.5).any() else 0.0

    fig = plt.figure(figsize=(14, 9))
    fig.patch.set_facecolor("white")
    plot_i = [1]
    panels = [
        (np.ones(n_all, dtype=float) * 0.15, f"full mesh\nN={n_all}", "Blues", 0.0, 1.0),
        (wall.astype(float), "mask_wall", "Blues", 0.0, 1.0),
        (band.astype(float), f"wall-band (GraphSAGE) n={n_band}", "Greens", 0.0, 1.0),
        (t0_seed.astype(float), f"t=0 dgamma seed n={n_t0}", "Greens", 0.0, 1.0),
        (elig.astype(float), f"nucleation E(t) n={n_elig}", "Greens", 0.0, 1.0),
        (mu_cap, f"GT mu cap {cap:.3f} Pa*s", "bwr", 0.0, cap),
    ]
    for vals, title, cmap, vmin, vmax in panels:
        ax = fig.add_subplot(2, 3, plot_i[0])
        plot_i[0] += 1
        _draw_mesh(ax, pos, vals, cmap=cmap, vmin=vmin, vmax=vmax, title=title)

    fig.suptitle(
        f"Biochem deploy masks -- {anchor} t={time_index} ({model_label})",
        fontsize=11,
        color=NAVY,
        y=0.98,
    )
    fig.text(
        0.5,
        0.93,
        f"wall={n_wall}  band={n_band}  nucleation={n_elig}  gt+ in E={gt_pos:.3f}",
        ha="center",
        fontsize=9.5,
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="Slide viz: full graph -> nucleation mask")
    ap.add_argument("--anchor", default="patient004")
    ap.add_argument("--time-index", type=int, default=62)
    ap.add_argument(
        "--layout",
        choices=("slide", "diag"),
        default="slide",
        help="slide=3-panel graph story; diag=6-panel mesh like clot_phi_masks",
    )
    ap.add_argument("--style", choices=("mesh", "graph"), default="mesh", help="slide layout only")
    ap.add_argument(
        "--model-label",
        default="global_guiding_5h GraphSAGE deploy",
        help="subtitle tag for the best GNN baseline",
    )
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    root = get_project_root()
    device = torch.device("cpu")
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")

    graph_path = root / VesselConfig(phase="biochem_anchors").graph_output_dir / f"{args.anchor}.pt"
    if not graph_path.is_file():
        raise FileNotFoundError(graph_path)

    data = torch.load(graph_path, map_location=device, weights_only=False)
    data = infer_missing_schema(data, phase_hint="biochem")
    assert_graph_schema(data, expected_y_schema=(BIO_Y_SCHEMA,))

    ti = max(0, min(int(args.time_index), int(data.y.shape[0]) - 1))
    pos = data.x[:, :2].detach().cpu().numpy()
    n_all = int(data.num_nodes)
    ei = data.edge_index.detach().cpu().numpy()

    wall = resolve_wall_mask(data, device).cpu().numpy()
    band = wall_band_mask(data, device).cpu().numpy()
    t0_seed = resolve_t0_dgamma_wall_mask(data, device, bio).cpu().numpy()
    commits_prev = resolve_commits_at_time(
        data, max(ti - 1, 0), device=device, phys_cfg=phys, growth_seed="gt"
    )
    elig = (
        resolve_nucleation_eligibility(
            data,
            ti,
            device,
            phys,
            bio,
            commits_prev=commits_prev,
            growth_seed="gt",
            use_dgamma_wall_seed=True,
        )
        .cpu()
        .numpy()
    )

    if args.out.strip():
        out = Path(args.out)
    elif args.layout == "diag":
        out = root / "outputs" / "biochem" / "viz" / f"biochem_deploy_masks_{args.anchor}_t{ti}.png"
    else:
        suffix = "" if args.style == "mesh" else f"_{args.style}"
        out = root / "outputs" / "biochem" / "viz" / f"biochem_nucleation_mask_{args.anchor}_t{ti}{suffix}.png"
    if not out.is_absolute():
        out = root / out

    if args.layout == "diag":
        mu_nd = data.y[ti, :, STATE_CHANNEL_MU_EFF_ND]
        mu_cap = cap_mu_eff_si(phys.viscosity_nd_to_si(mu_nd)).detach().cpu().numpy()
        phi_gt = gt_clot_phi_at_time(data, ti, phys, device).detach().cpu().numpy()
        render_diag(
            anchor=args.anchor,
            time_index=ti,
            model_label=args.model_label.strip(),
            pos=pos,
            n_all=n_all,
            wall=wall,
            band=band,
            t0_seed=t0_seed,
            elig=elig,
            phi_gt=phi_gt,
            mu_cap=mu_cap,
            out_path=out,
        )
    else:
        render_slide(
            anchor=args.anchor,
            time_index=ti,
            pos=pos,
            n_all=n_all,
            wall=wall,
            band=band,
            elig=elig,
            edge_index=ei,
            out_path=out,
            style=args.style,
        )
    print(f"[save] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
