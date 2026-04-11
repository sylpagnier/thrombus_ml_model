"""
Physics kernel tests:

* **Synthetic** WLS / Carreau smoke tests (no COMSOL).
* **COMSOL anchors**: uses ``compute_phase1_physics_terms`` — the **same** code path as
  ``train_tier1`` / ``train_tier2`` — so training physics cannot drift from what we certify.

COMSOL fields are not exact solutions of the graph WLS strong form, so we **do not** demand
near-zero NS residuals in absolute terms. We **do** require:

1. **Label identity** — with ``pred = data.y``, supervised anchor losses vanish (pipeline bug alarm).
2. **Shuffle baseline** — the same discrete losses must be **strictly worse** when nodal state rows
   are permuted (geometry–field mismatch alarm if COMSOL/WLS is worse than random reassignment).
3. **Tier 2 rheology** — Carreau supervisor on COMSOL ``μ`` must beat **μ-only** random permutation.

Tight absolute caps on ``l_bc`` (no-slip encoded in labels) catch gross export/mask errors.
``l_io`` is **not** asserted small (outlet pressure gauge vs training penalty).

Environment:

* ``PHASE1_PHYSICS_TEST_MAX_GRAPHS`` — max anchor graphs to scan (default: all found).
* ``PHASE1_PHYSICS_MIN_ANCHORS`` — minimum anchors required or skip (default 1).
* ``PHASE1_PHYSICS_RELAX_SHUFFLE=1`` — only run identity + BC checks (for debugging).
"""
from __future__ import annotations

import os
import unittest
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import numpy as np
import pytest
import torch
from torch_geometric.data import Batch, Data

from src.config import PhysicsConfig, VesselConfig
from src.phase1.physics.physics_kernels import PhysicsKernels
from src.phase1.utils.anchor_mask import anchor_node_mask, graph_has_anchor
from src.phase1.utils.phase1_physics_terms import compute_phase1_physics_terms


# ---------------------------------------------------------------------------
# Synthetic graph (no COMSOL)
# ---------------------------------------------------------------------------


def create_physical_test_graph():
    x = torch.linspace(0, 0.01, 20)
    y = torch.linspace(-0.001, 0.001, 10)
    grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")

    nodes = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
    num_nodes = nodes.size(0)

    dist = torch.cdist(nodes, nodes)
    edge_index = (dist < 0.0012).nonzero(as_tuple=False).t()
    edge_index = edge_index[:, edge_index[0] != edge_index[1]]

    data = Data(x=torch.zeros(num_nodes, 11), edge_index=edge_index)
    data.num_nodes = num_nodes

    data.mask_wall = (nodes[:, 1].abs() > 0.00095).float()
    data.mask_inlet = (nodes[:, 0] < 0.0001).float()
    data.mask_outlet = (nodes[:, 0] > 0.0099).float()

    data.x = torch.zeros(num_nodes, 11)
    data.x[:, 4] = 0.0
    data.x[:, 5] = torch.where(nodes[:, 1] > 0, -1.0, 1.0)

    phys_cfg = PhysicsConfig(tier="tier2")
    data.u_ref = torch.tensor([phys_cfg.get_u_ref(0.0015)])
    data.d_bar = torch.tensor([0.0015])

    row, col = edge_index
    dr = nodes[col] - nodes[row]
    dx, dy = dr[:, 0], dr[:, 1]
    data.V = torch.stack([dx, dy, 0.5 * dx**2, dx * dy, 0.5 * dy**2], dim=1)
    data.W = torch.ones(edge_index.size(1))
    data.M_inv = torch.eye(5).unsqueeze(0).repeat(num_nodes, 1, 1)

    return data, nodes, phys_cfg


def test_wls_derivative_dimensions():
    data, nodes, phys_cfg = create_physical_test_graph()
    kernels = PhysicsKernels(phys_cfg)

    u = (nodes[:, 0] + 2 * nodes[:, 1]).unsqueeze(1)
    props = kernels._get_geometric_props(data)
    derivs = kernels._compute_derivatives(u, props)

    assert derivs.shape == (nodes.size(0), 5, 1)
    assert not torch.isnan(derivs).any()


def test_carreau_rheology_real_bounds():
    data, _, phys_cfg = create_physical_test_graph()
    kernels = PhysicsKernels(phys_cfg)

    du_zero = torch.zeros((data.num_nodes, 4))
    mu_zero = kernels._compute_carreau_viscosity(du_zero, data)

    du_high = torch.tensor([[0.0, 1000.0, 0.0, 0.0]]).repeat(data.num_nodes, 1)
    mu_high = kernels._compute_carreau_viscosity(du_high, data)

    assert torch.all(mu_zero <= kernels.mu_0_nd + 1e-5)
    assert torch.all(mu_high >= kernels.mu_inf_nd - 1e-5)
    assert torch.all(mu_high < mu_zero)


def test_mass_conservation_logic():
    _, _, phys_cfg = create_physical_test_graph()
    kernels = PhysicsKernels(phys_cfg)

    du_ij_ok = torch.tensor([[0.5, 0.0, 0.0, -0.5]])
    loss_ok = kernels.continuity_loss(du_ij_ok)

    du_ij_bad = torch.tensor([[1.0, 0.0, 0.0, 1.0]])
    loss_bad = kernels.continuity_loss(du_ij_bad)

    assert loss_ok < 1e-7
    assert loss_bad > 1.0


def test_momentum_residual_execution():
    data, nodes, phys_cfg = create_physical_test_graph()
    kernels = PhysicsKernels(phys_cfg)

    y_norm = nodes[:, 1] / 0.001
    u = (1.0 - y_norm**2).unsqueeze(1)
    v = torch.zeros_like(u)
    p = torch.zeros_like(u)
    mu = torch.ones_like(u) * kernels.mu_inf_nd

    pred = torch.cat([u, v, p, mu], dim=1)
    loss_mom = kernels.navier_stokes_residual(pred, data)

    assert not torch.isnan(loss_mom)
    assert loss_mom >= 0


def test_wall_shear_stress_consistency():
    data, nodes, phys_cfg = create_physical_test_graph()
    kernels = PhysicsKernels(phys_cfg)

    u = nodes[:, 1].unsqueeze(1)
    v = torch.zeros_like(u)
    p = torch.zeros_like(u)
    mu = torch.ones_like(u)
    wss_pred = torch.ones_like(nodes[:, 0])

    pred = torch.cat([u, v, p, mu, wss_pred.unsqueeze(1)], dim=1)
    loss_wss = kernels.wall_shear_stress_loss(pred, data)

    assert not torch.isnan(loss_wss)


# ---------------------------------------------------------------------------
# Anchor mask — graph-level vs per-node ``is_anchor`` (PyG ``Batch``)
# ---------------------------------------------------------------------------


def test_graph_level_is_anchor_expands():
    n = 5
    x = torch.zeros(n, 3)
    d = Data(x=x, is_anchor=torch.tensor([True], dtype=torch.bool))
    m = anchor_node_mask(d)
    assert m is not None
    assert m.shape == (n,)
    assert m.all()
    assert graph_has_anchor(d)


def test_batched_one_flag_per_graph():
    g1 = Data(x=torch.zeros(3, 2), is_anchor=torch.tensor([True], dtype=torch.bool))
    g2 = Data(x=torch.zeros(4, 2), is_anchor=torch.tensor([False], dtype=torch.bool))
    b = Batch.from_data_list([g1, g2])
    m = anchor_node_mask(b)
    assert m.shape == (7,)
    assert m[:3].all()
    assert not m[3:].any()


def test_per_node_mask():
    n = 4
    ia = torch.tensor([1, 0, 1, 0], dtype=torch.bool)
    d = Data(x=torch.zeros(n, 2), is_anchor=ia)
    m = anchor_node_mask(d)
    assert (m == ia).all()


# ---------------------------------------------------------------------------
# COMSOL anchors — shared training physics path
# ---------------------------------------------------------------------------

EPS_DATA = 1e-5
EPS_BC = 5e-4
BOUNDARY_DATA_WEIGHT = 2.0


def _max_graphs_cap() -> Optional[int]:
    raw = os.environ.get("PHASE1_PHYSICS_TEST_MAX_GRAPHS", "").strip().lower()
    if raw in ("", "all"):
        return None
    n = int(raw)
    return None if n <= 0 else n


def _relax_shuffle() -> bool:
    return os.environ.get("PHASE1_PHYSICS_RELAX_SHUFFLE", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _iter_anchor_graph_files(tier: str) -> Iterator[Path]:
    cfg = VesselConfig(tier=tier)
    d = cfg.graph_output_dir
    if not d.is_dir():
        return
    for p in sorted(d.glob("vessel_*.pt")):
        yield p


def _load_anchor_graph(path: Path) -> Optional[Data]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    if not graph_has_anchor(data):
        return None
    y = getattr(data, "y", None)
    if y is None or y.dim() != 2 or y.shape[1] < 3:
        return None
    required = ("edge_index", "V", "W", "M_inv", "mask_wall", "mask_inlet", "mask_outlet", "u_ref", "d_bar")
    for attr in required:
        if not hasattr(data, attr):
            return None
    return data


def _state_from_labels(data: Data) -> torch.Tensor:
    y = data.y
    n = y.shape[0]
    out = torch.zeros(n, 5, dtype=y.dtype, device=y.device)
    c = min(5, y.shape[1])
    out[:, :c] = y[:, :c]
    return out


def _permute_rows(pred: torch.Tensor, seed: int) -> torch.Tensor:
    g = torch.Generator(device=pred.device)
    g.manual_seed(int(seed) % (2**31))
    n = pred.shape[0]
    perm = torch.randperm(n, generator=g, device=pred.device)
    return pred[perm].clone()


def _permute_mu_column(pred: torch.Tensor, seed: int) -> torch.Tensor:
    g = torch.Generator(device=pred.device)
    g.manual_seed(int(seed) % (2**31))
    n = pred.shape[0]
    perm = torch.randperm(n, generator=g, device=pred.device)
    out = pred.clone()
    out[:, 3] = pred[perm, 3]
    return out


def _terms_dict_to_float(d: Dict[str, torch.Tensor]) -> Dict[str, float]:
    return {k: float(v.detach().item()) for k, v in d.items()}


def _evaluate_terms(
    pred: torch.Tensor,
    data: Data,
    kernels: PhysicsKernels,
    *,
    tier: str,
    tier2_distillation: bool = False,
    carreau_n: Optional[float] = None,
) -> Dict[str, float]:
    t = compute_phase1_physics_terms(
        pred,
        data,
        kernels,
        tier=tier,
        boundary_data_weight=BOUNDARY_DATA_WEIGHT,
        tier2_distillation=tier2_distillation,
        carreau_n=carreau_n,
    )
    return _terms_dict_to_float(t)


def _collect_anchor_paths(tier: str) -> List[Path]:
    cap = _max_graphs_cap()
    out: List[Path] = []
    for p in _iter_anchor_graph_files(tier):
        if _load_anchor_graph(p) is None:
            continue
        out.append(p)
        if cap is not None and len(out) >= cap:
            break
    return out


class TestComsolAnchorPhysicsStrict(unittest.TestCase):
    """COMSOL labels vs training physics terms + shuffle sanity (no duplicated kernel logic)."""

    def test_tier1_comsol_training_physics_consistency(self):
        min_n = max(1, int(os.environ.get("PHASE1_PHYSICS_MIN_ANCHORS", "1")))
        paths = _collect_anchor_paths("tier1")
        if len(paths) < min_n:
            self.skipTest(
                f"Need at least {min_n} Tier 1 COMSOL anchor graphs; found {len(paths)} "
                f"under {VesselConfig(tier='tier1').graph_output_dir}."
            )

        phys_cfg = PhysicsConfig(tier="tier1")
        kernels = PhysicsKernels(phys_cfg)
        relax = _relax_shuffle()
        failures: List[str] = []

        for path in paths:
            data = _load_anchor_graph(path)
            assert data is not None
            stem = path.stem
            pred = _state_from_labels(data)
            seed = hash((stem, "row")) % (2**31)

            t_ok = _evaluate_terms(pred, data, kernels, tier="tier1")

            if t_ok["l_data_kine"] >= EPS_DATA:
                failures.append(
                    f"{stem}: l_data_kine={t_ok['l_data_kine']:.3e} (expect <{EPS_DATA} when pred=y)"
                )
            if t_ok["l_bc"] >= EPS_BC:
                failures.append(
                    f"{stem}: l_bc={t_ok['l_bc']:.3e} (expect <{EPS_BC} for no-slip labels)"
                )

            if relax:
                continue

            pred_bad = _permute_rows(pred, seed)
            t_bad = _evaluate_terms(pred_bad, data, kernels, tier="tier1")

            for key in ("l_mom", "l_cont", "l_wss"):
                if t_ok[key] >= t_bad[key]:
                    failures.append(
                        f"{stem}: {key} comsol={t_ok[key]:.6g} >= shuffled={t_bad[key]:.6g} "
                        f"(COMSOL field should score better than row-shuffled nonsense)"
                    )

        self.assertEqual(
            failures,
            [],
            "Tier 1 COMSOL / physics consistency failures:\n" + "\n".join(failures),
        )

    def test_tier2_comsol_training_physics_consistency_coupled(self):
        min_n = max(1, int(os.environ.get("PHASE1_PHYSICS_MIN_ANCHORS", "1")))
        paths = _collect_anchor_paths("tier2")
        if len(paths) < min_n:
            self.skipTest(
                f"Need at least {min_n} Tier 2 COMSOL anchor graphs; found {len(paths)} "
                f"under {VesselConfig(tier='tier2').graph_output_dir}."
            )

        phys_cfg = PhysicsConfig(tier="tier2")
        if phys_cfg.viscosity_model != "carreau":
            self.skipTest("Tier 2 config is not Carreau.")
        kernels = PhysicsKernels(phys_cfg)
        carreau_n = phys_cfg.n
        relax = _relax_shuffle()
        failures: List[str] = []

        for path in paths:
            data = _load_anchor_graph(path)
            assert data is not None
            stem = path.stem
            pred = _state_from_labels(data)
            seed_row = hash((stem, "row")) % (2**31)
            seed_mu = hash((stem, "mu")) % (2**31)

            t_ok = _evaluate_terms(
                pred, data, kernels, tier="tier2", tier2_distillation=False, carreau_n=carreau_n
            )

            if t_ok["l_data_kine"] >= EPS_DATA:
                failures.append(f"{stem}: l_data_kine={t_ok['l_data_kine']:.3e}")
            if t_ok["l_data_mu"] >= EPS_DATA:
                failures.append(f"{stem}: l_data_mu={t_ok['l_data_mu']:.3e}")
            if t_ok["l_bc"] >= EPS_BC:
                failures.append(f"{stem}: l_bc={t_ok['l_bc']:.3e}")

            if relax:
                continue

            pred_bad = _permute_rows(pred, seed_row)
            t_bad = _evaluate_terms(
                pred_bad, data, kernels, tier="tier2", tier2_distillation=False, carreau_n=carreau_n
            )
            for key in ("l_mom", "l_cont", "l_wss"):
                if t_ok[key] >= t_bad[key]:
                    failures.append(
                        f"{stem}: {key} comsol={t_ok[key]:.6g} >= row_shuf={t_bad[key]:.6g}"
                    )

            pred_mu_bad = _permute_mu_column(pred, seed_mu)
            t_mu = _evaluate_terms(
                pred_mu_bad, data, kernels, tier="tier2", tier2_distillation=False, carreau_n=carreau_n
            )
            if t_ok["l_rheo"] >= t_mu["l_rheo"]:
                failures.append(
                    f"{stem}: l_rheo comsol={t_ok['l_rheo']:.6g} >= mu_shuf={t_mu['l_rheo']:.6g}"
                )

        self.assertEqual(
            failures,
            [],
            "Tier 2 coupled COMSOL / physics consistency failures:\n" + "\n".join(failures),
        )

    def test_tier2_distillation_physics_terms_finite_and_identity(self):
        """Distillation branch: same helper with ``tier2_distillation=True`` (mom/cont unused)."""
        paths = _collect_anchor_paths("tier2")
        if not paths:
            self.skipTest("No Tier 2 anchor graphs.")

        phys_cfg = PhysicsConfig(tier="tier2")
        if phys_cfg.viscosity_model != "carreau":
            self.skipTest("Tier 2 config is not Carreau.")
        kernels = PhysicsKernels(phys_cfg)
        carreau_n = phys_cfg.n
        failures: List[str] = []

        for path in paths:
            data = _load_anchor_graph(path)
            assert data is not None
            stem = path.stem
            pred = _state_from_labels(data)
            t = _evaluate_terms(
                pred, data, kernels, tier="tier2", tier2_distillation=True, carreau_n=carreau_n
            )
            if t["l_data_kine"] >= EPS_DATA:
                failures.append(f"{stem}: distill l_data_kine={t['l_data_kine']:.3e}")
            if not np.isfinite(t["l_rheo"]) or t["l_rheo"] < 0:
                failures.append(f"{stem}: distill l_rheo bad: {t['l_rheo']}")
            if abs(t["l_mom"]) > 1e-12 or abs(t["l_cont"]) > 1e-12:
                failures.append(
                    f"{stem}: distill expects l_mom=l_cont=0, got mom={t['l_mom']} cont={t['l_cont']}"
                )

        self.assertEqual(failures, [], "\n".join(failures))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
