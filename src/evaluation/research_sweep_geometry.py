"""Deterministic parametric vessels for research geometry-sensitivity sweeps.

Builds clean straight-channel families (optional stenosis / aneurysm / bend /
roughness) and caches deploy-ready ``.pt`` graphs under
``outputs/research_sweeps/_meshes/``.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import meshio
import numpy as np
import torch
from torch_geometric.data import Data

from src.config import VesselConfig
from src.data_gen.lib.customer_geometry_import import (
    DEFAULT_N_STEPS,
    DEFAULT_RE,
    CustomerGeometryError,
    graph_from_mesh_meta,
    synthesize_deploy_timeline,
)
from src.data_gen.lib.vessel_generator import (
    VesselGenerator,
    build_vessel_mesh,
    make_vessel_params,
    stenosis_wall_offset_for_occlusion,
)
from src.utils.paths import get_project_root

# Shared control vessel (straight, mid-width, Re=450, no pathology).
CONTROL_SEED = 42
CONTROL_WIDTH_M = 0.012
CONTROL_RE = float(DEFAULT_RE)
CONTROL_BASE_LENGTH_M = 0.1
CONTROL_PATH_LOC = 2  # symmetric both-wall pathology
CONTROL_PATHOLOGY_STD_FRAC = 0.06
DEFAULT_T_FINAL_S = 30000.0  # 8 UI hours
DEFAULT_HORIZON_N_STEPS = 120  # matches customer clamp for 8 h


def default_mesh_cache_dir(root: Path | None = None) -> Path:
    return (root or get_project_root()) / "outputs" / "research_sweeps" / "_meshes"


def _zero_noise(n: int) -> list[float]:
    return [0.0] * int(n)


def _wall_roughness(
    n: int,
    amp: float,
    *,
    phase: float = 0.0,
) -> list[float]:
    """Deterministic tapered wall roughness (no RNG). ``amp`` in meters."""
    n = int(n)
    if abs(float(amp)) < 1e-15:
        return _zero_noise(n)
    t = np.linspace(0.0, 1.0, n)
    noise = np.sin(2.0 * np.pi * 2.0 * t + float(phase)) + 0.5 * np.sin(
        2.0 * np.pi * 3.5 * t + 0.7 * float(phase)
    )
    peak = float(np.max(np.abs(noise)))
    if peak < 1e-15:
        return _zero_noise(n)
    noise = (noise / peak) * float(amp)
    noise *= np.sin(np.pi * t) ** 0.5  # taper ends
    return noise.tolist()


def _gaussian_offsets(
    n: int,
    mag: float,
    *,
    path_loc_frac: float = 0.5,
    std_frac: float = CONTROL_PATHOLOGY_STD_FRAC,
) -> list[float]:
    """Deterministic Gaussian wall offset (no RNG skew / noise)."""
    n = int(n)
    t_idx = np.arange(n, dtype=np.float64)
    min_idx, max_idx = max(3, int(n * 0.2)), min(n - 4, int(n * 0.8))
    peak = int(min_idx + float(np.clip(path_loc_frac, 0.0, 1.0)) * (max_idx - min_idx))
    std_dev = max(1.0, float(std_frac) * n)
    gauss = np.exp(-0.5 * ((t_idx - peak) / std_dev) ** 2)
    return (float(mag) * gauss).tolist()


def build_research_vessel_params(
    *,
    width: float = CONTROL_WIDTH_M,
    curve_type: str = "straight",
    angle_span: float = 0.0,
    amplitude: float = 0.0,
    bend_sign: float = 1.0,
    stenosis_occlusion: float | None = None,
    aneurysm_factor: float | None = None,
    path_loc_frac: float = 0.5,
    path_loc: int = CONTROL_PATH_LOC,
    pathology_std_frac: float = CONTROL_PATHOLOGY_STD_FRAC,
    wall_roughness_amp: float = 0.0,
    seed: int = CONTROL_SEED,
    level: int = 0,
    idx: int = 0,
) -> dict[str, Any]:
    """Deterministic vessel params for research arms.

    ``path_loc``: 0=top wall only, 1=bottom only, 2=both (symmetric).
    ``wall_roughness_amp``: absolute wall noise amplitude [m] (0 = smooth).
    ``pathology_std_frac``: Gaussian axial width as fraction of control-point count.
    """
    cfg = VesselConfig(phase="kinematics")
    rng = np.random.default_rng(int(seed))
    n = int(cfg.num_ctrl_pts)

    if stenosis_occlusion is not None and aneurysm_factor is not None:
        raise ValueError("Specify only one of stenosis_occlusion or aneurysm_factor")

    pl = int(path_loc)
    if pl not in (0, 1, 2):
        raise ValueError(f"path_loc must be 0, 1, or 2; got {path_loc!r}")

    curve = str(curve_type).lower().strip()
    if curve == "straight":
        angle_span = 0.0
        amplitude = 0.0
    elif curve == "arc":
        amplitude = 0.0
    elif curve in ("s_curve", "sine"):
        curve = "s_curve"
        angle_span = 0.0
    elif curve == "hook":
        amplitude = 0.0
    else:
        raise ValueError(f"Unsupported curve_type={curve_type!r}")

    rough = float(max(0.0, wall_roughness_amp))
    noise_top = _wall_roughness(n, rough, phase=0.0)
    noise_bot = _wall_roughness(n, rough, phase=1.3)

    params = make_vessel_params(
        idx=int(idx),
        level=int(level),
        cfg=cfg,
        rng=rng,
        width=float(width),
        curve_type=curve,
        angle_span=float(angle_span),
        amplitude=float(amplitude),
        bend_sign=float(bend_sign),
        tortuosity=[],
        jitter=[],
        noise_top=noise_top,
        noise_bot=noise_bot,
        offsets=_zero_noise(n),
        v_type="straight",
        path_loc=pl,
    )
    params["wall_roughness_amp"] = rough
    params["pathology_std_frac"] = float(pathology_std_frac)

    std_frac = float(pathology_std_frac)
    if stenosis_occlusion is not None:
        occ = float(np.clip(stenosis_occlusion, 0.0, 0.95))
        if occ <= 1e-12:
            params["v_type"] = "straight"
            params["offsets"] = _zero_noise(n)
        else:
            mag = stenosis_wall_offset_for_occlusion(float(width), cfg, occlusion_frac=occ)
            params["v_type"] = "stenosis"
            params["path_loc"] = pl
            params["offsets"] = _gaussian_offsets(
                n,
                mag,
                path_loc_frac=float(path_loc_frac),
                std_frac=std_frac,
            )
            params["stenosis_occlusion"] = occ
            params["path_loc_frac"] = float(path_loc_frac)
    elif aneurysm_factor is not None:
        af = float(max(0.0, aneurysm_factor))
        if af <= 1e-12:
            params["v_type"] = "straight"
            params["offsets"] = _zero_noise(n)
        else:
            mag = af * float(width)
            params["v_type"] = "aneurysm"
            params["path_loc"] = pl
            params["offsets"] = _gaussian_offsets(
                n,
                mag,
                path_loc_frac=float(path_loc_frac),
                std_frac=std_frac,
            )
            params["aneurysm_factor"] = af
            params["path_loc_frac"] = float(path_loc_frac)

    return params


def geometry_spec_hash(spec: dict[str, Any]) -> str:
    """Stable short hash for mesh cache keys (geometry + Re + horizon)."""
    payload = json.dumps(spec, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def arm_geometry_cache_spec(arm: dict[str, Any], control: dict[str, Any]) -> dict[str, Any]:
    """Merge control + arm geometry knobs into a cacheable spec."""
    g = dict(control.get("geometry", {}))
    g.update(arm.get("geometry", {}) or {})
    return {
        "width": float(g.get("width", CONTROL_WIDTH_M)),
        "curve_type": str(g.get("curve_type", "straight")),
        "angle_span": float(g.get("angle_span", 0.0)),
        "amplitude": float(g.get("amplitude", 0.0)),
        "bend_sign": float(g.get("bend_sign", 1.0)),
        "stenosis_occlusion": g.get("stenosis_occlusion"),
        "aneurysm_factor": g.get("aneurysm_factor"),
        "path_loc_frac": float(g.get("path_loc_frac", 0.5)),
        "path_loc": int(g.get("path_loc", CONTROL_PATH_LOC)),
        "pathology_std_frac": float(
            g.get("pathology_std_frac", CONTROL_PATHOLOGY_STD_FRAC)
        ),
        "wall_roughness_amp": float(g.get("wall_roughness_amp", 0.0)),
        "base_length": float(g.get("base_length", CONTROL_BASE_LENGTH_M)),
        "seed": int(g.get("seed", CONTROL_SEED)),
        "level": int(g.get("level", 0)),
        "re_target": float(
            arm.get("re_target", control.get("re_target", CONTROL_RE))
        ),
        "t_final_s": float(
            arm.get("t_final_s", control.get("t_final_s", DEFAULT_T_FINAL_S))
        ),
        "n_steps": int(
            arm.get("n_steps", control.get("n_steps", DEFAULT_HORIZON_N_STEPS))
        ),
    }


def build_research_graph_from_spec(
    spec: dict[str, Any],
    *,
    work_dir: Path | None = None,
) -> Data:
    """Mesh + upgrade one research vessel to a deploy-ready timeline graph."""
    gen = VesselGenerator(phase="kinematics")
    cfg_dict = dict(gen._cfg_dict())
    cfg_dict["unit"] = "m"
    cfg_dict["base_length"] = float(spec.get("base_length", CONTROL_BASE_LENGTH_M))

    sten = spec.get("stenosis_occlusion")
    aneur = spec.get("aneurysm_factor")
    params = build_research_vessel_params(
        width=float(spec["width"]),
        curve_type=str(spec["curve_type"]),
        angle_span=float(spec.get("angle_span", 0.0)),
        amplitude=float(spec.get("amplitude", 0.0)),
        bend_sign=float(spec.get("bend_sign", 1.0)),
        stenosis_occlusion=None if sten is None else float(sten),
        aneurysm_factor=None if aneur is None else float(aneur),
        path_loc_frac=float(spec.get("path_loc_frac", 0.5)),
        path_loc=int(spec.get("path_loc", CONTROL_PATH_LOC)),
        pathology_std_frac=float(
            spec.get("pathology_std_frac", CONTROL_PATHOLOGY_STD_FRAC)
        ),
        wall_roughness_amp=float(spec.get("wall_roughness_amp", 0.0)),
        seed=int(spec.get("seed", CONTROL_SEED)),
        level=int(spec.get("level", 0)),
    )

    own_tmp = work_dir is None
    if work_dir is None:
        tmp = tempfile.TemporaryDirectory(prefix="research_sweep_")
        work = Path(tmp.name)
    else:
        tmp = None
        work = Path(work_dir)
        work.mkdir(parents=True, exist_ok=True)

    try:
        idx, ok, err = build_vessel_mesh(params, cfg_dict, work)
        if not ok:
            raise CustomerGeometryError(err or "research mesh build failed")
        msh_path = work / f"vessel_{idx}.msh"
        json_path = work / f"vessel_{idx}.json"
        mesh = meshio.read(msh_path)
        meta = json.loads(json_path.read_text(encoding="utf-8"))
        data = graph_from_mesh_meta(
            mesh,
            meta,
            re_target=float(spec.get("re_target", CONTROL_RE)),
            stem=f"research_{geometry_spec_hash(spec)}",
        )
        return synthesize_deploy_timeline(
            data,
            t_final_s=float(spec.get("t_final_s", DEFAULT_T_FINAL_S)),
            n_steps=int(spec.get("n_steps", DEFAULT_HORIZON_N_STEPS)),
        )
    finally:
        if own_tmp and tmp is not None:
            tmp.cleanup()


def load_or_build_research_graph(
    arm: dict[str, Any],
    control: dict[str, Any],
    *,
    cache_dir: Path | None = None,
    force_rebuild: bool = False,
) -> tuple[Data, dict[str, Any], Path]:
    """Return (graph, cache_spec, cache_pt_path), building via Gmsh on miss."""
    cache_dir = Path(cache_dir) if cache_dir is not None else default_mesh_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    spec = arm_geometry_cache_spec(arm, control)
    key = geometry_spec_hash(spec)
    pt_path = cache_dir / f"{key}.pt"
    meta_path = cache_dir / f"{key}.json"

    if pt_path.is_file() and not force_rebuild:
        data = torch.load(pt_path, map_location="cpu", weights_only=False)
        return data, spec, pt_path

    data = build_research_graph_from_spec(spec)
    torch.save(data, pt_path)
    meta_path.write_text(
        json.dumps({"hash": key, "spec": spec}, indent=2) + "\n",
        encoding="utf-8",
    )
    return data, spec, pt_path


__all__ = [
    "CONTROL_SEED",
    "CONTROL_WIDTH_M",
    "CONTROL_RE",
    "CONTROL_BASE_LENGTH_M",
    "CONTROL_PATHOLOGY_STD_FRAC",
    "DEFAULT_T_FINAL_S",
    "DEFAULT_HORIZON_N_STEPS",
    "DEFAULT_N_STEPS",
    "default_mesh_cache_dir",
    "build_research_vessel_params",
    "geometry_spec_hash",
    "arm_geometry_cache_spec",
    "build_research_graph_from_spec",
    "load_or_build_research_graph",
]
