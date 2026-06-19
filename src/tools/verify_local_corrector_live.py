"""In-vivo sanity check for the local kinematic corrector against the GINO-DEQ base flow.

Loads a real patient vessel graph, queries the frozen GINO-DEQ kine model for the base
flow ``[u0, v0]`` (non-dimensional), injects a dummy high-viscosity clot on a wall node,
extracts the k-hop subgraph, applies the corrector, and renders a side-by-side quiver of
base vs. corrected flow so you can eyeball that flow bends smoothly around the obstacle.

Everything is kept in the GINO-DEQ ND convention (positions by ``d_bar``, velocity by
``u_ref``, viscosity by ``PhysicsConfig.viscosity_si_to_nd``) and the feature tensor is
built with the *same* ``assemble_local_corrector_features`` used in training and deploy.

CLI:
    python -m src.tools.verify_local_corrector_live \
        --graph data/processed/graphs_biochem_anchors/patient007.pt \
        --corrector outputs/kinematics/local_corrector/local_kinematic_corrector_best.pth
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch_geometric.utils import k_hop_subgraph

from src.config import PhysicsConfig
from src.core_physics.clot_phi_simple import sdf_nd_from_data
from src.core_physics.coupled_shear_gnn import (
    LocalKinematicCorrector,
    assemble_local_corrector_features,
    load_local_corrector,
)
from src.utils.kinematics_inference import (
    load_kinematics_predictor,
    predict_kinematics,
    resolve_kinematics_checkpoint,
)
from src.utils.paths import data_root, get_project_root, reports_dir


DEFAULT_GRAPH = data_root() / "processed" / "graphs_biochem_anchors" / "patient007.pt"
DEFAULT_CORRECTOR = (
    get_project_root() / "outputs" / "kinematics" / "local_corrector"
    / "local_kinematic_corrector_best.pth"
)


def _pick_wall_seed(data, device: torch.device) -> int:
    """An interior, *connected* wall node (nearest to median wall x).

    The graph connectivity is sparse/banded -- many wall nodes are isolated in
    ``edge_index`` -- so we restrict to nodes with degree >= 1, otherwise the k-hop
    subgraph collapses to the seed alone.
    """
    n = int(data.num_nodes)
    ei = data.edge_index
    deg = torch.zeros(n, device=device)
    deg.index_add_(0, ei[0].to(device), torch.ones(ei.shape[1], device=device))

    mask = getattr(data, "mask_wall", None)
    if mask is not None:
        wall_idx = torch.where(mask.reshape(-1).to(device).bool() & (deg > 0))[0]
    else:  # fall back to connected near-wall nodes by SDF
        sdf = sdf_nd_from_data(data, device, n)
        wall_idx = torch.where((sdf < sdf.median()) & (deg > 0))[0]
    if wall_idx.numel() == 0:
        raise RuntimeError("No connected wall nodes found on the graph.")
    x = data.x[wall_idx, 0].to(device)
    target_x = x.median()
    return int(wall_idx[(x - target_x).abs().argmin()].item())


def run(
    graph_path: Path | str = DEFAULT_GRAPH,
    corrector_path: Path | str = DEFAULT_CORRECTOR,
    *,
    kine_ckpt: Path | str | None = None,
    num_hops: int = 4,
    clot_mu_si: float = 1.5,
    clot_seed_hops: int = 1,
    out_png: Path | str | None = None,
    gui: bool = False,
    device: torch.device | str | None = None,
) -> Path:
    dev = torch.device(device) if device is not None else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    phys = PhysicsConfig(phase="kinematics")

    data = torch.load(graph_path, map_location=dev, weights_only=False)
    data = data.to(dev)
    n = int(data.num_nodes)

    # 1. Frozen base flow from GINO-DEQ (ND u, v)
    kine = load_kinematics_predictor(resolve_kinematics_checkpoint(kine_ckpt), dev)
    with torch.no_grad():
        pred = predict_kinematics(kine, data)
    u0 = pred[:, 0].contiguous()
    v0 = pred[:, 1].contiguous()

    # 2. Inject a dummy wall clot (seed + 1-hop neighbours -> ~3-node footprint)
    seed = _pick_wall_seed(data, dev)
    clot_subset, _, _, _ = k_hop_subgraph(
        torch.tensor([seed], device=dev), num_hops=int(clot_seed_hops),
        edge_index=data.edge_index, relabel_nodes=False, num_nodes=n,
    )
    clot_nodes = clot_subset[:3] if clot_subset.numel() >= 3 else clot_subset
    mu_inf = float(phys.mu_inf)
    delta_mu_si = torch.zeros(n, device=dev)
    delta_mu_si[clot_nodes] = float(clot_mu_si) - mu_inf

    # 3. k-hop subgraph around the clot (full mesh connectivity)
    subset, sub_edge_index, _, _ = k_hop_subgraph(
        clot_nodes, num_hops=int(num_hops), edge_index=data.edge_index,
        relabel_nodes=True, num_nodes=n,
    )

    # 4. Assemble features (same convention as train/deploy)
    pos_nd = data.x[:, 0:2].to(dtype=torch.float32)
    sdf_nd = sdf_nd_from_data(data, dev, n).reshape(-1)
    delta_mu_nd = phys.viscosity_si_to_nd(delta_mu_si)
    x_sub = assemble_local_corrector_features(
        pos_nd, sdf_nd, u0, v0, delta_mu_nd, clot_nodes, subset
    )

    # 5. Load corrector (fall back to a fresh model just to exercise the plumbing)
    cpath = Path(corrector_path)
    if cpath.is_file():
        corrector = load_local_corrector(cpath, dev)
    else:
        print(f"[WARN] corrector checkpoint not found ({cpath}); using untrained model "
              "(diversion will be ~zero -- plumbing check only).")
        corrector = LocalKinematicCorrector().to(dev).eval()

    # 6. Predict + patch
    with torch.no_grad():
        delta_uv = corrector(x_sub, sub_edge_index.to(dev))
    u_final = u0.clone(); v_final = v0.clone()
    u_final[subset] = u_final[subset] + delta_uv[:, 0]
    v_final[subset] = v_final[subset] + delta_uv[:, 1]

    patch_max = float(delta_uv.abs().max().item())
    print(f"[i] graph={Path(graph_path).name} N={n} | subset={int(subset.numel())} nodes | "
          f"clot={int(clot_nodes.numel())} nodes | max|dUV_nd|={patch_max:.4e}")

    out = _plot_diversion(
        data, subset, clot_nodes, u0, v0, u_final, v_final, phys,
        out_png=out_png, gui=gui,
    )
    print(f"[save] {out}")
    return out


def _plot_diversion(
    data, subset, clot_nodes, u0, v0, uf, vf, phys: PhysicsConfig,
    *, out_png: Path | str | None, gui: bool,
) -> Path:
    import matplotlib
    if not gui:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    d_bar = float(data.d_bar.reshape(-1)[0].item()) if hasattr(data, "d_bar") else 1.0
    um = d_bar * 1e6  # ND position -> micrometers
    pos = data.x[subset, 0:2].detach().cpu().numpy() * um
    cpos = data.x[clot_nodes, 0:2].detach().cpu().numpy() * um
    u0n = u0[subset].detach().cpu().numpy(); v0n = v0[subset].detach().cpu().numpy()
    ufn = uf[subset].detach().cpu().numpy(); vfn = vf[subset].detach().cpu().numpy()

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 5), sharex=True, sharey=True)
    ax1.quiver(pos[:, 0], pos[:, 1], u0n, v0n, color="tab:blue", alpha=0.6)
    ax1.scatter(cpos[:, 0], cpos[:, 1], color="black", s=30, zorder=5, label="clot")
    ax1.set_title("GINO-DEQ base flow (no clot)")
    ax1.set_xlabel("x [um]"); ax1.set_ylabel("y [um]"); ax1.legend(fontsize=8)

    ax2.quiver(pos[:, 0], pos[:, 1], ufn, vfn, color="tab:red", alpha=0.6)
    ax2.scatter(cpos[:, 0], cpos[:, 1], color="black", s=30, zorder=5, label="clot obstacle")
    ax2.set_title("Corrected flow (with diversion)")
    ax2.set_xlabel("x [um]"); ax2.legend(fontsize=8)

    # Overlay: base vs corrected on the SAME axes with a SHARED arrow scale so the
    # diversion is directly comparable (independent autoscaling would be misleading).
    ext = float(pos[:, 0].max() - pos[:, 0].min()) if pos.shape[0] else 0.0
    target = 0.08 * ext if ext > 0 else 1.0
    sp = np.sqrt(np.concatenate([u0n, ufn]) ** 2 + np.concatenate([v0n, vfn]) ** 2)
    maxsp = float(sp.max()) if sp.size else 0.0
    qkw = (
        dict(angles="xy", scale_units="xy", scale=maxsp / target)
        if (target > 0 and maxsp > 0)
        else {}
    )
    ax3.quiver(pos[:, 0], pos[:, 1], u0n, v0n, color="tab:blue", alpha=0.45, label="base", **qkw)
    ax3.quiver(pos[:, 0], pos[:, 1], ufn, vfn, color="tab:red", alpha=0.7, label="corrected", **qkw)
    ax3.scatter(cpos[:, 0], cpos[:, 1], color="black", s=30, zorder=5, label="clot")
    ax3.set_title("Overlay: base (blue) vs corrected (red)")
    ax3.set_xlabel("x [um]"); ax3.legend(fontsize=8)

    for ax in (ax1, ax2, ax3):
        ax.set_aspect("equal")
    fig.tight_layout()

    if out_png is None:
        out_png = reports_dir() / "figures" / "kinematics" / "local_corrector_diversion.png"
    out_png = Path(out_png); out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130)
    if gui:
        plt.show()
    plt.close(fig)
    return out_png


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Live quiver verification of the local kinematic corrector.")
    p.add_argument("--graph", type=str, default=str(DEFAULT_GRAPH))
    p.add_argument("--corrector", type=str, default=str(DEFAULT_CORRECTOR))
    p.add_argument("--kine-ckpt", type=str, default=None)
    p.add_argument("--num-hops", type=int, default=4)
    p.add_argument("--clot-mu", type=float, default=1.5, help="Clot peak viscosity [Pa*s].")
    p.add_argument("--clot-seed-hops", type=int, default=1)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--gui", action="store_true")
    p.add_argument("--device", type=str, default=None)
    return p


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    run(
        graph_path=args.graph,
        corrector_path=args.corrector,
        kine_ckpt=args.kine_ckpt,
        num_hops=args.num_hops,
        clot_mu_si=args.clot_mu,
        clot_seed_hops=args.clot_seed_hops,
        out_png=args.out,
        gui=args.gui,
        device=args.device,
    )
