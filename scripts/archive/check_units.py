"""Compare non-dimensional velocity scales: kinematics vs biochem anchor graphs."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.config import PhysicsConfig, PredChannels
from src.utils.paths import data_root


def _max_nd_uv(graph) -> float:
    y = graph.y
    if y.dim() == 3:
        y = y[0]
    return float(y[:, PredChannels.UV].abs().max().item())


def _scales(graph) -> tuple[float, float, float]:
    d_bar = float(graph.d_bar.reshape(-1)[0].item())
    u_ref = float(graph.u_ref.reshape(-1)[0].item())
    u_ref_calc = float(PhysicsConfig(phase="biochem").get_u_ref(d_bar))
    return d_bar, u_ref, u_ref_calc


def run_sanity_check() -> int:
    dr = data_root()
    kine_path = dr / "processed/graphs_kinematics/newtonian/vessel_0.pt"
    anchor_dir = dr / "processed/graphs_biochem_anchors"
    bio_paths = sorted(anchor_dir.glob("*.pt")) if anchor_dir.exists() else []
    bio_path = bio_paths[0] if bio_paths else dr / "processed/graphs_biochem/patient001.pt"

    missing = [p for p in (kine_path, bio_path) if not p.exists()]
    if missing:
        print("[ERR] Paths not found:")
        for p in missing:
            print(f"  {p}")
        return 1

    kine_graph = torch.load(kine_path, map_location="cpu", weights_only=False)
    bio_graph = torch.load(bio_path, map_location="cpu", weights_only=False)

    k_max = _max_nd_uv(kine_graph)
    b_max = _max_nd_uv(bio_graph)
    k_db, k_ur, k_ur_calc = _scales(kine_graph)
    b_db, b_ur, b_ur_calc = _scales(bio_graph)

    print(f"Phase 1 (Kinematics)  {kine_path.name}")
    print(f"  max |u_nd|,|v_nd|: {k_max:.4f}")
    print(f"  d_bar={k_db:.6f} m  u_ref={k_ur:.6f} m/s  (calc {k_ur_calc:.6f})")
    print(f"Phase 3 (Biochem)     {bio_path.name}")
    print(f"  max |u_nd|,|v_nd|: {b_max:.4f}")
    print(f"  d_bar={b_db:.6f} m  u_ref={b_ur:.6f} m/s  (calc {b_ur_calc:.6f})")
    print(f"  ratio bio/kine max ND vel: {b_max / max(k_max, 1e-12):.2f}x")

    if b_max > k_max * 50:
        print("\n[ERR] BUG LIKELY: biochem ND velocity >> kinematics (~100x cm/m d_bar mix).")
        return 2
    if b_max > k_max * 5:
        print("\n[WARN] biochem ND notably larger than kinematics; investigate sidecars / extract.")
        return 0
    print("\n[OK] ND velocity scales comparable (no ~100x inflation).")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_sanity_check())
