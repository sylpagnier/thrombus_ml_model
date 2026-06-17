"""Smoke test: verify FourierTauGate, SpatialConditionedGate, and compute_adhesion_gate dispatch.

Run:
    python scripts/check_adhesion_gate_smoke.py

Exits 0 on pass, 1 on failure.
"""
from __future__ import annotations

import os
import sys
import traceback

import torch

PASS = "[OK]"
FAIL = "[FAIL]"


def check(name: str, cond: bool, detail: str = "") -> bool:
    if cond:
        print(f"{PASS}  {name}")
    else:
        print(f"{FAIL}  {name}" + (f": {detail}" if detail else ""))
    return cond


def test_fourier_tau_gate() -> bool:
    from src.architecture.gnode_biochem import FourierTauGate
    gate = FourierTauGate(t_ref=30000.0, num_freqs=8)
    feats = torch.zeros(50, 12)
    out = gate(feats, current_time_s=15000.0)
    ok = (
        out.shape == (50, 1)
        and float(out.detach().min()) >= 0.0
        and float(out.detach().max()) <= 1.0
    )
    return check("FourierTauGate shape + range", ok, f"shape={out.shape} min={float(out.detach().min()):.3f} max={float(out.detach().max()):.3f}")


def test_spatial_gate() -> bool:
    from src.architecture.gnode_biochem import SpatialConditionedGate
    gate = SpatialConditionedGate(in_node_features=76)
    feats = torch.randn(50, 76)
    out = gate(feats, current_time_s=15000.0)
    ok = (
        out.shape == (50, 1)
        and float(out.min()) >= 0.0
        and float(out.max()) <= 1.0
    )
    return check("SpatialConditionedGate shape + range", ok, f"shape={out.shape}")


def test_global_sigmoid_dispatch() -> bool:
    from src.core_physics.biochem_physics_kernels import compute_adhesion_gate
    import types

    cfg = types.SimpleNamespace(
        surface_time_gate_s=12.0,
        surface_time_gate_slope=10.0,
        t_final=30000.0,
    )
    data = types.SimpleNamespace(t_global=None)
    os.environ["BIOCHEM_ADHESION_GATE"] = "global_sigmoid"
    gate = compute_adhesion_gate(
        data, cfg, device=torch.device("cpu"), dtype=torch.float32, current_time_s=15000.0
    )
    ok = gate.shape == () or gate.numel() == 1
    return check("compute_adhesion_gate global_sigmoid scalar", ok, f"shape={gate.shape}")


def test_fourier_tau_dispatch() -> bool:
    from src.architecture.gnode_biochem import FourierTauGate
    from src.core_physics.biochem_physics_kernels import compute_adhesion_gate
    import types

    cfg = types.SimpleNamespace(
        surface_time_gate_s=12.0,
        surface_time_gate_slope=10.0,
        t_final=30000.0,
    )
    data = types.SimpleNamespace(t_global=None)
    gate_module = FourierTauGate(t_ref=30000.0, num_freqs=8)
    feats = torch.zeros(40, 12)
    os.environ["BIOCHEM_ADHESION_GATE"] = "fourier_tau"
    gate = compute_adhesion_gate(
        data, cfg,
        device=torch.device("cpu"), dtype=torch.float32,
        current_time_s=15000.0,
        node_features=feats,
        gate_module=gate_module,
    )
    ok = gate.shape == (40, 1)
    return check("compute_adhesion_gate fourier_tau per-node", ok, f"shape={gate.shape}")


def test_spatial_mlp_dispatch() -> bool:
    from src.architecture.gnode_biochem import SpatialConditionedGate
    from src.core_physics.biochem_physics_kernels import compute_adhesion_gate
    import types

    cfg = types.SimpleNamespace(
        surface_time_gate_s=12.0,
        surface_time_gate_slope=10.0,
        t_final=30000.0,
    )
    data = types.SimpleNamespace(t_global=None)
    gate_module = SpatialConditionedGate(in_node_features=12)
    feats = torch.zeros(40, 12)
    os.environ["BIOCHEM_ADHESION_GATE"] = "spatial_mlp"
    gate = compute_adhesion_gate(
        data, cfg,
        device=torch.device("cpu"), dtype=torch.float32,
        current_time_s=15000.0,
        node_features=feats,
        gate_module=gate_module,
    )
    ok = gate.shape == (40, 1)
    return check("compute_adhesion_gate spatial_mlp per-node", ok, f"shape={gate.shape}")


def main() -> int:
    results: list[bool] = []
    tests = [
        test_global_sigmoid_dispatch,
        test_fourier_tau_gate,
        test_spatial_gate,
        test_fourier_tau_dispatch,
        test_spatial_mlp_dispatch,
    ]
    for t in tests:
        try:
            results.append(t())
        except Exception:
            print(f"{FAIL}  {t.__name__}")
            traceback.print_exc()
            results.append(False)

    passed = sum(results)
    total = len(results)
    print(f"\n{'='*40}")
    print(f"Gate smoke: {passed}/{total} passed")
    if passed == total:
        print(f"{PASS}  All gate smoke checks passed.")
        return 0
    else:
        print(f"{FAIL}  {total - passed} check(s) failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
