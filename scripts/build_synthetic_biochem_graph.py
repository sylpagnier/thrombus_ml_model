"""Build a single synthetic biochem graph for deploy viz (no COMSOL)."""

from __future__ import annotations

import shutil
from pathlib import Path

import torch

from src.config import BiochemConfig
from src.data_gen.lib.mesh_to_graph import MeshToGraphComplete
from src.data_gen.lib.mesh_to_graph_biochem import MeshToGraphPhase3
from src.data_gen.lib.vessel_generator import VesselGeneratorPhase3
from src.utils.paths import get_project_root


def default_synthetic_cache_dir(seed: int) -> Path:
    return get_project_root() / "data" / "processed" / "graphs_biochem_synthetic" / f"seed_{int(seed)}"


def build_synthetic_biochem_graph(
    *,
    seed: int = 42,
    level: int = 1,
    regenerate: bool = False,
    cache_dir: Path | str | None = None,
    n_time_steps: int | None = None,
) -> tuple[torch.Tensor | object, Path]:
    """Generate (or reuse) one synthetic vessel -> biochem ``.pt`` graph."""
    root = get_project_root()
    cache = Path(cache_dir) if cache_dir else default_synthetic_cache_dir(seed)
    raw_dir = cache / "raw_meshes"
    graph_kine_dir = cache / "graphs_kine_base"
    graph_dir = cache / "graphs_biochem"
    for d in (raw_dir, graph_kine_dir, graph_dir):
        d.mkdir(parents=True, exist_ok=True)

    pts = list(graph_dir.glob("*.pt"))
    if regenerate and pts:
        for d in (raw_dir, graph_kine_dir, graph_dir):
            if d.is_dir():
                shutil.rmtree(d, ignore_errors=True)
                d.mkdir(parents=True, exist_ok=True)
        pts = []

    if not pts:
        vg = VesselGeneratorPhase3(output_dir=raw_dir)
        vg.run_pipeline(n=1, level=int(level), num_workers=1, seed=int(seed))
        MeshToGraphComplete(
            phase="kinematics",
            raw_dir=raw_dir,
            label_dir=raw_dir,
            proc_dir=graph_kine_dir,
        ).run(max_files=1)
        MeshToGraphPhase3(
            raw_dir=raw_dir,
            label_dir=raw_dir,
            proc_dir=graph_dir,
        ).run(max_files=1)
        pts = list(graph_dir.glob("*.pt"))

    if not pts:
        raise FileNotFoundError(f"no synthetic biochem graph under {graph_dir}")

    graph_path = sorted(pts)[0]
    data = torch.load(graph_path, map_location="cpu", weights_only=False)

    if n_time_steps is not None and hasattr(data, "y") and data.y.ndim == 3:
        n_tgt = max(int(n_time_steps), 2)
        n_cur = int(data.y.shape[0])
        if n_cur != n_tgt:
            bio = BiochemConfig(phase="biochem")
            t_end = float(bio.t_final)
            if hasattr(data, "t") and torch.is_tensor(data.t) and data.t.numel() > 1:
                t_end = float(data.t[-1].item())
            times = torch.linspace(0.0, t_end, n_tgt, dtype=torch.float32)
            y_new = torch.zeros((n_tgt, int(data.y.shape[1]), int(data.y.shape[2])), dtype=data.y.dtype)
            y_new[:] = data.y[0]
            data.y = y_new
            data.t = times

    return data, graph_path
