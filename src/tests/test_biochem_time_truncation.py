"""Ensure biochem extraction and fallback timelines use full horizon by default."""

from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data

from src.config import BiochemConfig
from src.data_gen.lib.extract_biochem_comsol_data import PatientDataExtractor


def _minimal_wide_txt(path, times_s: list[float], n_rows: int = 2) -> None:
    """Tiny COMSOL-style wide export: one header line + numeric rows (``vars_per_step`` = 18 per time)."""
    parts = ["% x"]
    for t in times_s:
        parts.append(f"x y u v p mu_effective rp ap apr aps PT th at fg fi M Mas Mat @ t={t}")
    header = " ".join(parts) + "\n"
    n_times = len(times_s)
    n_cols = 2 + n_times * 18
    rng = np.random.default_rng(0)
    body = []
    for _ in range(n_rows):
        row = rng.standard_normal(n_cols).tolist()
        body.append(" ".join(f"{v:.6f}" for v in row) + "\n")
    path.write_text(header + "".join(body), encoding="utf-8")


def test_load_comsol_trajectory_keeps_full_horizon_by_default(tmp_path):
    ext = PatientDataExtractor(phase="biochem_anchors", raw_dir=tmp_path, label_dir=tmp_path, proc_dir=tmp_path)
    fp = tmp_path / "stub.txt"
    _minimal_wide_txt(fp, times_s=[0.0, 100.0, 8000.0, 13000.0, 50000.0])
    blocks = ext.load_comsol_trajectory(fp)
    assert set(blocks.keys()) == {0.0, 100.0, 8000.0, 13000.0, 50000.0}
    assert max(blocks.keys()) == 50000.0


def test_biochem_config_resolve_times_uses_t_final_for_default_linspace():
    bio = BiochemConfig(phase="biochem")
    t_steps = 5
    data = Data(
        y=torch.zeros((t_steps, 3, 16), dtype=torch.float32),
        t=None,
        is_anchor=torch.tensor([False], dtype=torch.bool),
    )
    t = bio.resolve_biochem_times(data, device=torch.device("cpu"))
    assert t.numel() == t_steps
    assert float(t[0].item()) == 0.0
    assert float(t[-1].item()) == float(bio.t_final)


def test_biochem_config_resolve_mismatched_t_uses_provided_t_last():
    bio = BiochemConfig(phase="biochem")
    t_steps = 4
    data = Data(
        y=torch.zeros((t_steps, 2, 16), dtype=torch.float32),
        t=torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32),
        is_anchor=torch.tensor([False], dtype=torch.bool),
    )
    t = bio.resolve_biochem_times(data, device=torch.device("cpu"))
    assert t.numel() == t_steps
    assert float(t[-1].item()) == 2.0


def test_synthetic_transient_horizon_matches_t_final():
    """Same rule as ``MeshToGraphPhase3`` non-anchor branch."""
    bio_cfg = BiochemConfig(phase="biochem")
    t_horizon_s = float(bio_cfg.t_final)
    num_times = bio_cfg.num_time_steps
    eval_times = torch.linspace(0.0, t_horizon_s, num_times, dtype=torch.float32)
    assert float(eval_times[0].item()) == 0.0
    assert float(eval_times[-1].item()) == t_horizon_s == 30000.0
