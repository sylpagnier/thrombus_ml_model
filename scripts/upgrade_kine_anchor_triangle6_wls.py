"""Upgrade corner-only kinematics patient anchors to P2 triangle6 edges + WLS ops.

Only rewrites ``graphs_kinematics_anchors/<rheology>/patient*.pt`` when the graph is
still on corner-only connectivity (undirected E ~ half of mesh triangle6). Rebuilds
``edge_index``, ``edge_attr``, ``V``, ``W``, ``M_inv``, ``G_x``, ``G_y``, and refreshes
WSS channels in ``x`` / ``y`` from existing GT flow labels.

Example:
    python scripts/upgrade_kine_anchor_triangle6_wls.py
    python scripts/upgrade_kine_anchor_triangle6_wls.py --dry-run
    python scripts/upgrade_kine_anchor_triangle6_wls.py --stems patient001,patient007
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.config import NodeFeat, VesselConfig
from src.data_gen.lib.centerline_utils import resolve_anchor_mesh_path
from src.data_gen.lib.kinematics_graph_builder import comsol_fields_to_kinematics_y
from src.data_gen.lib.mesh_triangle6_edges import edge_index_from_mesh_path_checked
from src.data_gen.lib.mesh_wls import precompute_wls_operators
from src.data_gen.lib.node_feature_assembly import apply_gt_flow_priors_to_kine_x
from src.utils.kinematics_paths import kinematics_anchor_graph_dir


def _sparse_grad_ops(edge_index: torch.Tensor, num_nodes: int, M_inv: torch.Tensor, V: torch.Tensor, W: torch.Tensor):
    """Match PatientDataExtractor._precompute_sparse_operators (G_x, G_y only)."""
    row, col = edge_index
    M_inv_edges = M_inv[row]
    WV = (W.unsqueeze(1) * V).unsqueeze(2)
    C = torch.bmm(M_inv_edges, WV).squeeze(2)
    Cx = C[:, 0]
    Cy = C[:, 1]

    def build_sparse(edge_weights: torch.Tensor) -> torch.Tensor:
        diag_values = torch.zeros(num_nodes, dtype=torch.float32, device=C.device)
        diag_values.scatter_add_(0, row, -edge_weights)
        diag_indices = torch.arange(num_nodes, device=C.device).repeat(2, 1)
        indices = torch.cat([edge_index, diag_indices], dim=1)
        values = torch.cat([edge_weights, diag_values])
        return torch.sparse_coo_tensor(indices, values, size=(num_nodes, num_nodes)).coalesce()

    return build_sparse(Cx), build_sparse(Cy)


def _undirected_e(edge_index: torch.Tensor) -> int:
    return int(edge_index.shape[1]) // 2


def upgrade_one(
    graph_path: Path,
    raw_dir: Path,
    *,
    dry_run: bool,
    backup_dir: Path,
    force: bool,
) -> str:
    stem = graph_path.stem
    mesh_path = resolve_anchor_mesh_path(raw_dir, stem)
    if mesh_path is None:
        return f"[skip] {stem}: no mesh (.nas/.msh)"

    data = torch.load(graph_path, map_location="cpu", weights_only=False)
    n = int(data.num_nodes)
    old_e = _undirected_e(data.edge_index)

    try:
        new_ei = edge_index_from_mesh_path_checked(mesh_path, num_nodes=n, stem=stem)
    except Exception as exc:
        return f"[skip] {stem}: {exc}"

    new_e = _undirected_e(new_ei)
    if new_e == old_e and not force:
        return f"[ok] {stem}: already triangle6 (E={old_e})"
    if new_e < old_e:
        return f"[skip] {stem}: mesh E={new_e} < graph E={old_e} (unexpected)"

    if dry_run:
        return f"[dry] {stem}: edges {old_e} -> {new_e}  mesh={mesh_path.name}"

    pos_nd = data.x[:, NodeFeat.XY].to(dtype=torch.float32)
    row, col = new_ei
    edge_vec = pos_nd[row] - pos_nd[col]
    edge_len = torch.linalg.norm(edge_vec, dim=1, keepdim=True)
    edge_attr = torch.cat([edge_vec, edge_len], dim=1)

    V, W, M_inv = precompute_wls_operators(new_ei, n, pos_nd)
    G_x, G_y = _sparse_grad_ops(new_ei, n, M_inv, V, W)

    y = data.y
    if y.dim() == 3:
        y0 = y[0]
    else:
        y0 = y
    u_nd = y0[:, 0].contiguous()
    v_nd = y0[:, 1].contiguous()
    p_nd = y0[:, 2].contiguous()
    mu_nd = y0[:, 3].contiguous()
    mask_wall = data.mask_wall.bool().view(-1)
    wall_normal = data.x[:, NodeFeat.WALL_NORMAL]

    y_new = comsol_fields_to_kinematics_y(
        u_nd=u_nd,
        v_nd=v_nd,
        p_nd=p_nd,
        mu_nd=mu_nd,
        wall_normal_vec=wall_normal,
        mask_wall=mask_wall,
        edge_index=new_ei,
        M_inv=M_inv,
        V=V,
        W=W,
        ref_mu=1.0,
    )
    if y.dim() == 3:
        y = y.clone()
        y[0] = y_new
    else:
        y = y_new

    x = apply_gt_flow_priors_to_kine_x(
        data.x.clone(),
        u_nd=u_nd,
        v_nd=v_nd,
        mu_nd=mu_nd,
        mask_wall=mask_wall,
        wall_normal=wall_normal,
        edge_index=new_ei,
        M_inv=M_inv,
        V=V,
        W=W,
    )

    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    bak = backup_dir / f"{stem}_pre_tri6_wls_{stamp}.pt"
    shutil.copy2(graph_path, bak)

    data.edge_index = new_ei
    data.edge_attr = edge_attr
    data.V = V
    data.W = W
    data.M_inv = M_inv
    data.G_x = G_x
    data.G_y = G_y
    data.x = x
    data.y = y
    torch.save(data, graph_path)
    return (
        f"[OK] {stem}: edges {old_e} -> {new_e}  "
        f"V={tuple(V.shape)} G_x.nnz={int(G_x._nnz())} backup={bak.name}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="Rewrite even if edge count already matches mesh.")
    ap.add_argument("--rheology", default="carreau", choices=("carreau", "newtonian"))
    ap.add_argument("--stems", default="", help="Comma list; default = all patient*.pt needing upgrade.")
    ap.add_argument("--raw-dir", type=Path, default=None)
    args = ap.parse_args()

    kine_dir = kinematics_anchor_graph_dir(rheology=args.rheology)
    raw_dir = args.raw_dir or Path(VesselConfig(phase="biochem_anchors").mesh_input_dir)
    backup_dir = kine_dir / "_backup_pre_tri6_wls"

    if args.stems.strip():
        paths = [kine_dir / f"{s.strip()}.pt" for s in args.stems.split(",") if s.strip()]
    else:
        paths = sorted(kine_dir.glob("patient*.pt"))

    print(f"[i] kine_dir={kine_dir}")
    print(f"[i] raw_dir={raw_dir} dry_run={bool(args.dry_run)}")
    ok = skip = 0
    for p in paths:
        if not p.is_file():
            print(f"[skip] missing {p.name}", flush=True)
            skip += 1
            continue
        line = upgrade_one(
            p,
            raw_dir,
            dry_run=bool(args.dry_run),
            backup_dir=backup_dir,
            force=bool(args.force),
        )
        print(line, flush=True)
        if line.startswith("[OK]") or line.startswith("[dry]") or line.startswith("[ok]"):
            ok += 1
        else:
            skip += 1
    print(f"[i] done: touched_or_ok={ok} skipped={skip}", flush=True)
    return 0 if ok > 0 or skip == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
