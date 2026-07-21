"""Customer geometry inbox + load helpers for the Predict app.

Supports:
  - existing HemoRGP ``.pt`` graphs
  - tagged Gmsh ``.msh`` / ``.nas`` with a same-stem sidecar ``.json``
  - parametric vessel build (caller supplies mesh+meta via ``graph_from_mesh_meta``)

Does not invent boundary tags for untagged STL.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import meshio
import numpy as np
import torch
from torch_geometric.data import Data

from src.config import PhysicsConfig, VesselConfig
from src.data_gen.lib.mesh_to_graph import MeshToGraph
from src.data_gen.lib.mesh_to_graph_biochem import default_biochem_bio_inlet_bc
from src.utils.channel_schema import (
    BIO_Y_SCHEMA,
    Y_SCHEMAS,
    attach_channel_metadata,
    infer_missing_schema,
)
from src.utils.paths import get_project_root

INBOX_DIRNAME = "customer_geometries"
SUPPORTED_SUFFIXES = (".pt", ".msh", ".nas")
DEFAULT_N_STEPS = 60
DEFAULT_RE = 450.0


class CustomerGeometryError(ValueError):
    """User-facing geometry / tag / sidecar failure."""


def inbox_dir(root: Path | None = None) -> Path:
    return (root or get_project_root()) / INBOX_DIRNAME


def ensure_inbox(root: Path | None = None) -> Path:
    d = inbox_dir(root)
    d.mkdir(parents=True, exist_ok=True)
    return d


def list_inbox(root: Path | None = None) -> list[Path]:
    d = ensure_inbox(root)
    files = [
        p
        for p in sorted(d.iterdir(), key=lambda q: q.name.lower())
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    ]
    return files


def copy_into_inbox(src: Path | str, root: Path | None = None) -> Path:
    """Copy a customer file into the inbox (same name). Returns destination path."""
    src_p = Path(src).resolve()
    if not src_p.is_file():
        raise CustomerGeometryError(f"File not found: {src_p}")
    if src_p.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise CustomerGeometryError(
            f"Unsupported type '{src_p.suffix}'. Use .pt, .msh, or .nas."
        )
    dest_dir = ensure_inbox(root)
    dest = dest_dir / src_p.name
    if src_p.resolve() != dest.resolve():
        shutil.copy2(src_p, dest)
        sidecar = src_p.with_suffix(".json")
        if src_p.suffix.lower() in (".msh", ".nas") and sidecar.is_file():
            shutil.copy2(sidecar, dest_dir / sidecar.name)
    return dest


def _physics(re_target: float) -> PhysicsConfig:
    # Mesh build uses kinematics feature layout; Re only affects u_ref / priors.
    return PhysicsConfig(phase="kinematics", re_target=float(re_target))


def _validate_masks(data: Data, *, stem: str) -> None:
    for name in ("mask_inlet", "mask_outlet", "mask_wall"):
        if not hasattr(data, name) or getattr(data, name) is None:
            raise CustomerGeometryError(
                f"{stem}: missing {name}. Meshes need Gmsh tags "
                f"Inlet={VesselConfig().TAGS['Inlet']}, "
                f"Outlet_1={VesselConfig().TAGS['Outlet_1']}, "
                f"Walls={VesselConfig().TAGS['Walls']}. "
                f"See {INBOX_DIRNAME}/README.txt."
            )
    n_in = int(data.mask_inlet.reshape(-1).bool().sum().item())
    n_out = int(data.mask_outlet.reshape(-1).bool().sum().item())
    n_wall = int(data.mask_wall.reshape(-1).bool().sum().item())
    if n_in < 1 or n_out < 1 or n_wall < 1:
        raise CustomerGeometryError(
            f"{stem}: bad boundary masks (inlet={n_in}, outlet={n_out}, wall={n_wall}). "
            f"Check physical line tags in {INBOX_DIRNAME}/README.txt."
        )


def apply_re_target(data: Data, re_target: float) -> Data:
    """Rescale ``u_ref`` / inlet velocity BCs for a new Reynolds number."""
    out = data.clone() if hasattr(data, "clone") else data
    phys = _physics(re_target)
    d_bar = float(out.d_bar.reshape(-1)[0].item()) if hasattr(out, "d_bar") else 0.0
    if d_bar <= 0:
        raise CustomerGeometryError("Graph is missing a valid d_bar length scale.")
    u_new = float(phys.get_u_ref(d_bar))
    u_old = float(out.u_ref.reshape(-1)[0].item()) if hasattr(out, "u_ref") else u_new
    out.u_ref = torch.tensor([u_new], dtype=torch.float32)
    if hasattr(out, "re_actual"):
        out.re_actual = torch.tensor([float(re_target)], dtype=torch.float32)
    scale = (u_new / u_old) if abs(u_old) > 1e-12 else 1.0
    if hasattr(out, "u_inlet_bc") and out.u_inlet_bc is not None and abs(scale - 1.0) > 1e-8:
        out.u_inlet_bc = out.u_inlet_bc * scale
    # ND prior columns in x (u_prior / v_prior) stay ND; absolute Re is carried by u_ref.
    return out


def _bio_y_width() -> int:
    return int(Y_SCHEMAS[BIO_Y_SCHEMA].width)


def _seed_bio_frame_from_data(data: Data, n_nodes: int) -> torch.Tensor:
    """One (N, 16) frame: copy kinematics u/v/p/mu when present; rest resting / zero."""
    c = _bio_y_width()
    frame = torch.zeros(n_nodes, c, dtype=torch.float32)
    y = getattr(data, "y", None)
    if y is not None and torch.is_tensor(y):
        if y.dim() == 3:
            src = y[0]
        elif y.dim() == 2:
            src = y
        else:
            src = None
        if src is not None:
            n_copy = min(int(src.shape[-1]), 4, c)
            frame[:, :n_copy] = src[:, :n_copy].detach().cpu().float()
    if abs(float(frame[:, 3].mean().item())) < 1e-12:
        # mu_eff_nd ~ 1 when unknown
        frame[:, 3] = 1.0
    return frame


def synthesize_deploy_timeline(
    data: Data,
    *,
    t_final_s: float,
    n_steps: int | None = None,
) -> Data:
    """Attach a macro time axis and biochem ``y`` scaffold for kinematics-driven deploy."""
    out = data.clone() if hasattr(data, "clone") else data
    n_nodes = int(out.x.shape[0])
    steps = int(n_steps) if n_steps is not None else DEFAULT_N_STEPS
    steps = max(2, steps)
    t_end = max(float(t_final_s), 1.0)

    frame0 = _seed_bio_frame_from_data(out, n_nodes)
    y_series = frame0.unsqueeze(0).expand(steps, -1, -1).contiguous().clone()
    # Keep existing species IC from a biochem graph when present
    y_old = getattr(data, "y", None)
    if (
        y_old is not None
        and torch.is_tensor(y_old)
        and y_old.dim() == 3
        and int(y_old.shape[-1]) == _bio_y_width()
    ):
        src = y_old.detach().cpu().float()
        t_src = int(src.shape[0])
        for i in range(steps):
            j = min(int(round(i * (t_src - 1) / max(steps - 1, 1))), t_src - 1)
            y_series[i] = src[j]

    out.y = y_series
    out.t = torch.linspace(0.0, t_end, steps=steps, dtype=torch.float32)
    if not hasattr(out, "bio_inlet_bc") or out.bio_inlet_bc is None:
        out.bio_inlet_bc = default_biochem_bio_inlet_bc(n_nodes)
    infer_missing_schema(out, phase_hint="biochem")
    if getattr(out, "y_schema", None) != BIO_Y_SCHEMA:
        attach_channel_metadata(
            out,
            x_schema=getattr(out, "x_schema", None) or "kine_x_v1_18ch",
            y_schema=BIO_Y_SCHEMA,
            mask_wall=getattr(out, "mask_wall", None),
        )
    return out


def graph_from_mesh_meta(
    mesh: meshio.Mesh,
    meta: dict[str, Any],
    *,
    re_target: float = DEFAULT_RE,
    stem: str = "customer",
) -> Data:
    """Build a kinematics graph then upgrade to a deploy-ready biochem timeline scaffold."""
    phys = _physics(re_target)
    # MeshToGraph uses phys_cfg from constructor; override re via PhysicsConfig
    builder = MeshToGraph(phase="kinematics", rheology="carreau")
    builder.phys_cfg = phys
    data = builder.process_mesh(mesh, meta, stem=stem)
    if data is None:
        raise CustomerGeometryError(
            f"{stem}: mesh has no triangles or wall nodes."
        )
    _validate_masks(data, stem=stem)
    data = apply_re_target(data, re_target)
    # Default scaffold; callers usually re-synthesize with the UI horizon.
    return synthesize_deploy_timeline(data, t_final_s=8000.0, n_steps=DEFAULT_N_STEPS)


def _load_mesh_and_meta(path: Path) -> tuple[meshio.Mesh, dict[str, Any]]:
    mesh = meshio.read(path)
    json_path = path.with_suffix(".json")
    if not json_path.is_file():
        raise CustomerGeometryError(
            f"{path.name}: missing sidecar {json_path.name} with centerline_pts / "
            f"centerline_tangents / d_bar. Put both files in {INBOX_DIRNAME}/ "
            f"(see README.txt). Or use Parametric mode / a .pt graph."
        )
    meta = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(meta, dict):
        raise CustomerGeometryError(f"{json_path.name}: expected a JSON object.")
    return mesh, meta


def load_customer_geometry(
    path: Path | str,
    *,
    re_target: float = DEFAULT_RE,
    t_final_s: float | None = None,
    n_steps: int | None = None,
) -> Data:
    """Load ``.pt`` / ``.msh`` / ``.nas`` into a deploy-ready ``Data`` graph."""
    p = Path(path)
    if not p.is_file():
        raise CustomerGeometryError(f"File not found: {p}")
    suffix = p.suffix.lower()
    stem = p.stem

    if suffix == ".pt":
        data = torch.load(p, map_location="cpu", weights_only=False)
        if not isinstance(data, Data):
            raise CustomerGeometryError(f"{p.name}: expected a PyG Data graph.")
        _validate_masks(data, stem=stem)
        data = apply_re_target(data, re_target)
        t_end = float(t_final_s) if t_final_s is not None else float(
            getattr(data, "t", torch.tensor([8000.0])).reshape(-1)[-1].item()
        )
        steps = n_steps
        if steps is None and hasattr(data, "y") and torch.is_tensor(data.y) and data.y.dim() == 3:
            steps = int(data.y.shape[0])
        return synthesize_deploy_timeline(data, t_final_s=t_end, n_steps=steps)

    if suffix in (".msh", ".nas"):
        mesh, meta = _load_mesh_and_meta(p)
        data = graph_from_mesh_meta(mesh, meta, re_target=re_target, stem=stem)
        if t_final_s is not None or n_steps is not None:
            data = synthesize_deploy_timeline(
                data,
                t_final_s=float(t_final_s if t_final_s is not None else 8000.0),
                n_steps=n_steps,
            )
        return data

    raise CustomerGeometryError(
        f"Unsupported type '{suffix}'. Use .pt, .msh, or .nas (see {INBOX_DIRNAME}/README.txt)."
    )


def build_parametric_customer_graph(
    *,
    re_target: float = DEFAULT_RE,
    t_final_s: float = 8000.0,
    n_steps: int = DEFAULT_N_STEPS,
    width: float | None = None,
    angle_span: float | None = None,
    amplitude: float | None = None,
    level: int = 0,
    params_override: dict[str, Any] | None = None,
) -> Data:
    """Build a synthetic vessel mesh and return a deploy-ready graph.

    ``params_override`` may be an edited-walls params dict from
    ``geometry_to_params_override`` (skips width/angle sampling).
    """
    import tempfile

    from src.data_gen.lib.vessel_generator import (
        VesselGenerator,
        build_vessel_mesh,
        make_vessel_params,
    )

    cfg = VesselConfig(phase="kinematics")
    gen = VesselGenerator(phase="kinematics")
    cfg_dict = dict(gen._cfg_dict())
    cfg_dict["unit"] = "m"

    if params_override is not None:
        params = dict(params_override)
        params.setdefault("idx", 0)
    else:
        overrides: dict[str, Any] = {}
        if width is not None:
            overrides["width"] = float(width)
        if angle_span is not None:
            overrides["angle_span"] = float(angle_span)
            overrides["curve_type"] = "arc" if abs(float(angle_span)) > 1e-6 else "straight"
        if amplitude is not None:
            overrides["amplitude"] = float(amplitude)
            if float(amplitude) > 0:
                overrides["curve_type"] = "sine"
        params = make_vessel_params(idx=0, level=int(level), cfg=cfg, **overrides)

    with tempfile.TemporaryDirectory(prefix="customer_param_") as tmp:
        work = Path(tmp)
        idx, ok, err = build_vessel_mesh(params, cfg_dict, work)
        if not ok:
            raise CustomerGeometryError(err or "parametric mesh build failed")
        msh_path = work / f"vessel_{idx}.msh"
        json_path = work / f"vessel_{idx}.json"
        mesh = meshio.read(msh_path)
        meta = json.loads(json_path.read_text(encoding="utf-8"))
        data = graph_from_mesh_meta(mesh, meta, re_target=re_target, stem=f"vessel_{idx}")
        return synthesize_deploy_timeline(data, t_final_s=t_final_s, n_steps=n_steps)


def preview_points_from_graph(data: Data) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return SI positions and inlet/outlet/wall masks for a lightweight preview."""
    d_bar = float(data.d_bar.reshape(-1)[0].item()) if hasattr(data, "d_bar") else 1.0
    pos = data.x[:, :2].detach().cpu().numpy().astype(np.float64) * d_bar
    inlet = data.mask_inlet.reshape(-1).bool().cpu().numpy()
    outlet = data.mask_outlet.reshape(-1).bool().cpu().numpy()
    wall = data.mask_wall.reshape(-1).bool().cpu().numpy()
    return pos, inlet, outlet, wall
