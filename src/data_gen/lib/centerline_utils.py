"""Centerline metadata for Poiseuille priors (synthetic + patient anchors)."""

from __future__ import annotations

from collections import deque
import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

from src.config import VesselConfig


def centerline_from_graph_path(
    pos_nd: torch.Tensor,
    edge_index: torch.Tensor,
    mask_inlet: torch.Tensor,
    mask_outlet: torch.Tensor,
    *,
    mask_wall: Optional[torch.Tensor] = None,
    n_samples: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    """Approximate centerline by shortest path on the mesh graph (inlet -> outlet).

    Works for curved patient anchors when JSON sidecar centerline is missing; much
    better than a straight inlet-outlet chord for Poiseuille flow-direction alignment.
    """
    pos = pos_nd.detach().cpu().numpy()
    n = int(pos.shape[0])
    mi = mask_inlet.detach().cpu().numpy().astype(bool).reshape(-1)
    mo = mask_outlet.detach().cpu().numpy().astype(bool).reshape(-1)
    if not mi.any() or not mo.any():
        raise ValueError("centerline_from_graph_path requires nonempty inlet and outlet masks.")

    wall = mask_wall.detach().cpu().numpy().astype(bool).reshape(-1) if mask_wall is not None else np.zeros(n, dtype=bool)

    in_idx = np.flatnonzero(mi)
    out_idx = np.flatnonzero(mo)
    in_cent = pos[mi].mean(axis=0)
    out_cent = pos[mo].mean(axis=0)
    start = int(in_idx[np.argmin(np.linalg.norm(pos[in_idx] - in_cent, axis=1))])
    goal = int(out_idx[np.argmin(np.linalg.norm(pos[out_idx] - out_cent, axis=1))])

    def _allowed(node: int) -> bool:
        return (not wall[node]) or node in (start, goal)

    row, col = edge_index[0].cpu().numpy(), edge_index[1].cpu().numpy()
    adj: list[list[int]] = [[] for _ in range(n)]
    for r, c in zip(row, col):
        if r == c:
            continue
        if not _allowed(r) or not _allowed(c):
            continue
        adj[r].append(c)

    parent = [-1] * n
    seen = [False] * n
    q: deque[int] = deque([start])
    seen[start] = True
    found = False
    while q:
        u = q.popleft()
        if u == goal:
            found = True
            break
        for v in adj[u]:
            if not seen[v]:
                seen[v] = True
                parent[v] = u
                q.append(v)

    if not found:
        return centerline_from_inlet_outlet(pos_nd, mask_inlet, mask_outlet, n_samples=n_samples)

    path = []
    cur = goal
    while cur >= 0:
        path.append(cur)
        if cur == start:
            break
        cur = parent[cur]
    path = path[::-1]
    pts = pos[path]
    if pts.shape[0] < 2:
        return centerline_from_inlet_outlet(pos_nd, mask_inlet, mask_outlet, n_samples=n_samples)

    seg_len = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = float(cum[-1]) if cum[-1] > 0 else 1.0
    ts = np.linspace(0.0, total, max(4, int(n_samples)), dtype=np.float64)
    resampled = np.zeros((ts.shape[0], 2), dtype=np.float64)
    j = 0
    for i, t in enumerate(ts):
        while j + 1 < len(cum) and cum[j + 1] < t:
            j += 1
        if j + 1 >= len(cum):
            resampled[i] = pts[-1]
            continue
        alpha = (t - cum[j]) / max(cum[j + 1] - cum[j], 1e-12)
        resampled[i] = (1.0 - alpha) * pts[j] + alpha * pts[j + 1]
    tangents = np.gradient(resampled, axis=0)
    tangents = tangents / (np.linalg.norm(tangents, axis=1, keepdims=True) + 1e-12)
    return resampled.astype(np.float64), tangents.astype(np.float64)


def centerline_from_inlet_outlet(
    pos_nd: torch.Tensor,
    mask_inlet: torch.Tensor,
    mask_outlet: torch.Tensor,
    *,
    n_samples: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fallback spine: straight segment from inlet centroid to outlet centroid (ND)."""
    pos = pos_nd.detach().cpu().numpy()
    mi = mask_inlet.detach().cpu().numpy().astype(bool).reshape(-1)
    mo = mask_outlet.detach().cpu().numpy().astype(bool).reshape(-1)
    if not mi.any() or not mo.any():
        raise ValueError("centerline_from_inlet_outlet requires nonempty inlet and outlet masks.")
    p_in = pos[mi].mean(axis=0)
    p_out = pos[mo].mean(axis=0)
    ts = np.linspace(0.0, 1.0, max(4, int(n_samples)), dtype=np.float64)
    pts = p_in[None, :] + ts[:, None] * (p_out[None, :] - p_in[None, :])
    tangents = np.gradient(pts, axis=0)
    tangents = tangents / (np.linalg.norm(tangents, axis=1, keepdims=True) + 1e-12)
    return pts.astype(np.float64), tangents.astype(np.float64)


def load_sidecar_centerline_nd(
    stem: str,
    *,
    raw_dir: Optional[Path] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if not stem:
        return None, None
    root = Path(raw_dir) if raw_dir is not None else Path(VesselConfig(phase="biochem_anchors").mesh_input_dir)
    if not root.is_absolute():
        from src.utils.paths import get_project_root

        root = get_project_root() / root
    json_path = root / f"{stem}.json"
    if not json_path.is_file():
        return None, None
    import json

    with open(json_path, encoding="utf-8") as f:
        meta = json.load(f)
    cl = meta.get("centerline_pts")
    ct = meta.get("centerline_tangents")
    if cl is None or ct is None:
        return None, None
    pts = np.asarray(cl, dtype=np.float64)
    tan = np.asarray(ct, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2 or tan.shape != pts.shape or pts.shape[0] < 2:
        return None, None
    return pts, tan


def resolve_anchor_mesh_path(raw_dir: Path, stem: str) -> Optional[Path]:
    """Prefer ``<stem>.msh``, else ``<stem>.nas`` (COMSOL anchors often ship NAS only)."""
    raw_dir = Path(raw_dir)
    msh = raw_dir / f"{stem}.msh"
    if msh.is_file():
        return msh
    nas = raw_dir / f"{stem}.nas"
    if nas.is_file():
        return nas
    return None


def write_anchor_sidecar_from_masks(
    json_path: Path,
    *,
    mesh_nodes_si: np.ndarray,
    mask_inlet: torch.Tensor,
    mask_outlet: torch.Tensor,
    mask_wall: torch.Tensor,
    edge_index: Optional[torch.Tensor],
    d_bar_si: float,
    stem: str,
    unit: str = "cm",
    level: int = 2,
    existing: Optional[dict] = None,
) -> bool:
    """Persist centerline / d_bar using COMSOL boundary masks (no Gmsh line tags required)."""
    from src.utils.units import CGS_to_SI

    json_path = Path(json_path)
    base = dict(existing or {})
    if not base and json_path.is_file():
        base = json.loads(json_path.read_text(encoding="utf-8"))

    scale = float(CGS_to_SI.LENGTH) if unit == "cm" else 1.0
    d_bar_json = float(d_bar_si) / scale
    nodes_nd = torch.tensor(np.asarray(mesh_nodes_si, dtype=np.float64) / float(d_bar_si), dtype=torch.float32)
    cl_pts, cl_tan, cl_src = resolve_centerline_nd(
        nodes_nd,
        mask_inlet,
        mask_outlet,
        edge_index=edge_index,
        mask_wall=mask_wall,
        stem=stem,
        raw_sidecar_dir=json_path.parent,
    )
    payload = {
        **base,
        "unit": unit,
        "d_bar": d_bar_json,
        "level": int(level),
        "centerline_pts": cl_pts.tolist(),
        "centerline_tangents": cl_tan.tolist(),
        "centerline_source": cl_src,
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return True


def enrich_anchor_sidecar_json(
    mesh_path: Path,
    *,
    stem: str,
    level: int = 2,
    unit: str = "cm",
    overwrite: bool = False,
) -> bool:
    """Write ``d_bar``, ``centerline_*``, and ``level`` into ``<stem>.json`` (``.msh`` or ``.nas``)."""
    import meshio

    from src.config import VesselConfig
    from src.data_gen.lib.mesh_wls import gmsh_line_boundary_masks
    from src.utils.units import CGS_to_SI

    json_path = mesh_path.parent / f"{stem}.json"
    if json_path.is_file() and not overwrite:
        existing = json.loads(json_path.read_text(encoding="utf-8"))
        if (
            existing.get("centerline_pts")
            and existing.get("centerline_tangents")
            and existing.get("d_bar") is not None
        ):
            return False
    elif json_path.is_file():
        existing = json.loads(json_path.read_text(encoding="utf-8"))
    else:
        existing = {}

    mesh = meshio.read(str(mesh_path))
    scale = float(CGS_to_SI.LENGTH) if unit == "cm" else 1.0
    nodes_si = mesh.points[:, :2] * scale
    tags = dict(VesselConfig(phase="kinematics").TAGS)
    try:
        mask_inlet, mask_outlet, mask_wall = gmsh_line_boundary_masks(mesh, len(nodes_si), tags)
    except ValueError as exc:
        raise ValueError(
            f"{stem}: Gmsh boundary tags missing on {mesh_path.name} ({exc}). "
            "Use write_anchor_sidecar_from_masks during COMSOL extract, or re-export .msh with tagged walls."
        ) from exc
    if "triangle" in mesh.cells_dict:
        tris = mesh.cells_dict["triangle"]
    elif "triangle6" in mesh.cells_dict:
        tris = mesh.cells_dict["triangle6"][:, :3]
    else:
        tris = None
    edge_index = None
    if tris is not None and len(tris) > 0:
        edges = np.unique(
            np.sort(
                np.vstack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]]),
                axis=1,
            ),
            axis=0,
        )
        edge_index = torch.tensor(
            np.hstack([edges.T, edges[:, [1, 0]].T]),
            dtype=torch.long,
        )
    inlet_coords = nodes_si[mask_inlet.numpy()]
    d_bar_si = (
        float(np.max(np.linalg.norm(inlet_coords[:, None] - inlet_coords, axis=-1)))
        if len(inlet_coords) > 1
        else 0.0198
    )
    d_bar_json = d_bar_si / scale
    nodes_nd = torch.tensor(nodes_si / d_bar_si, dtype=torch.float32)
    cl_pts, cl_tan, cl_src = resolve_centerline_nd(
        nodes_nd,
        mask_inlet,
        mask_outlet,
        edge_index=edge_index,
        mask_wall=mask_wall,
        stem=stem,
    )
    payload = {
        **existing,
        "unit": unit,
        "d_bar": float(d_bar_json),
        "level": int(level),
        "centerline_pts": cl_pts.tolist(),
        "centerline_tangents": cl_tan.tolist(),
        "centerline_source": cl_src,
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return True


def resolve_centerline_nd(
    pos_nd: torch.Tensor,
    mask_inlet: torch.Tensor,
    mask_outlet: torch.Tensor,
    *,
    edge_index: Optional[torch.Tensor] = None,
    mask_wall: Optional[torch.Tensor] = None,
    stem: str = "",
    raw_sidecar_dir: Optional[Path] = None,
    n_fallback_samples: int = 64,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """Return ``(centerline_pts_nd, centerline_tangents_nd, source_tag)``."""
    pts, tan = load_sidecar_centerline_nd(stem, raw_dir=raw_sidecar_dir)
    if pts is not None and tan is not None:
        return pts, tan, "sidecar"
    if edge_index is not None:
        pts, tan = centerline_from_graph_path(
            pos_nd,
            edge_index,
            mask_inlet,
            mask_outlet,
            mask_wall=mask_wall,
            n_samples=n_fallback_samples,
        )
        return pts, tan, "graph_path"
    pts, tan = centerline_from_inlet_outlet(
        pos_nd,
        mask_inlet,
        mask_outlet,
        n_samples=n_fallback_samples,
    )
    return pts, tan, "inlet_outlet"
