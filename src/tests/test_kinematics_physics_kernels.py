"""
Physics kernel tests:

* **Synthetic** WLS / Carreau smoke tests (no COMSOL).
* **COMSOL anchors**: uses ``compute_kinematics_physics_terms`` — the **same** code path as
  ``train_kinematics_predictor`` — so training physics cannot drift from what we certify.

COMSOL fields are not exact solutions of the graph WLS strong form, so we **do not** demand
near-zero NS residuals in absolute terms. We **do** require:

1. **Label identity** — with ``pred = data.y``, supervised anchor losses vanish (pipeline bug alarm).
2. **Shuffle baseline** — the same discrete losses must be **strictly worse** when nodal state rows
   are permuted (geometry–field mismatch alarm if COMSOL/WLS is worse than random reassignment).
3. **Kinematics rheology** — Carreau supervisor on COMSOL ``μ`` must beat **μ-only** random permutation.

Tight absolute caps on ``l_bc`` (no-slip encoded in labels) catch gross export/mask errors.
``l_io`` is **not** asserted small (outlet pressure gauge vs training penalty).

Environment:

* ``KINEMATICS_PHYSICS_TEST_MAX_GRAPHS`` — max anchor graphs to scan (default: all found).
* ``KINEMATICS_PHYSICS_MIN_ANCHORS`` — minimum anchors required or skip (default 1).
* ``KINEMATICS_PHYSICS_RELAX_SHUFFLE=1`` — only run identity + BC checks (for debugging).
* ``KINEMATICS_PHYSICS_CHECK_BC=0`` — skip ``l_bc`` assertions on COMSOL anchors (use when the corpus has
  systematic wall-label vs. mask mismatch; shuffle/identity checks still run unless relaxed).
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

from src.config import PhysicsConfig, PredChannels, VesselConfig
from src.core_physics.physics_kernels import PhysicsKernels
from src.utils.anchor_mask import anchor_node_mask, graph_has_anchor
from src.utils.kinematics_physics_terms import compute_kinematics_physics_terms


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

    phys_cfg = PhysicsConfig(phase="kinematics")
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


def test_dynamic_curriculum_parameters_change_physics_response():
    data, nodes, phys_cfg = create_physical_test_graph()
    kernels = PhysicsKernels(phys_cfg)

    y_norm = nodes[:, 1] / 0.001
    u = (1.0 - y_norm**2).unsqueeze(1)
    v = torch.zeros_like(u)
    p = torch.zeros_like(u)

    # Fixed velocity gradients; only rheology parameters vary.
    props = kernels._get_geometric_props(data)
    c_u = kernels._compute_derivatives(u, props)
    c_v = kernels._compute_derivatives(v, props)
    du_ij = torch.stack([c_u[:, 0, 0], c_u[:, 1, 0], c_v[:, 0, 0], c_v[:, 1, 0]], dim=1)

    residuals = []
    mu_means = []
    for n_val, mu0_si in ((1.0, 0.0035), (0.8, 0.02), (0.6, 0.035), (0.3568, 0.056)):
        kernels.mu_0_nd = float(mu0_si / kernels.cfg.mu_viscosity_nd_scale)
        mu = kernels._compute_carreau_viscosity(du_ij, data, carreau_n=n_val).unsqueeze(1)
        pred = torch.cat([u, v, p, mu], dim=1)
        residuals.append(float(kernels.navier_stokes_residual(pred, data).item()))
        mu_means.append(float(mu.mean().item()))

    # NS residual can be close on synthetic smooth fields, but should not be exactly invariant.
    assert max(residuals) - min(residuals) > 0.0
    assert max(mu_means) - min(mu_means) > 1e-9


def test_wall_shear_stress_data_loss_smoke():
    """Anchor-only supervised WSS vs ``data.y``; no ``is_anchor`` => zero loss."""
    data, nodes, phys_cfg = create_physical_test_graph()
    kernels = PhysicsKernels(phys_cfg)

    u = nodes[:, 1].unsqueeze(1)
    v = torch.zeros_like(u)
    p = torch.zeros_like(u)
    mu = torch.ones_like(u)
    wss_pred = torch.ones_like(nodes[:, 0]).unsqueeze(1)

    pred = torch.cat([u, v, p, mu, wss_pred], dim=1)
    assert not torch.isnan(kernels.wall_shear_stress_loss(pred, data))

    n = int(data.num_nodes)
    data.is_anchor = torch.tensor([True], dtype=torch.bool)
    data.y = torch.zeros(n, 5)
    data.y[:, 4] = 1.0
    loss_match = kernels.wall_shear_stress_loss(pred, data)
    assert not torch.isnan(loss_match)
    assert float(loss_match.item()) < 1e-5


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


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return float(default)
    return float(raw)


def _max_graphs_cap() -> Optional[int]:
    raw = os.environ.get("KINEMATICS_PHYSICS_TEST_MAX_GRAPHS", "").strip().lower()
    if raw in ("", "all"):
        return None
    n = int(raw)
    return None if n <= 0 else n


def _relax_shuffle() -> bool:
    return os.environ.get("KINEMATICS_PHYSICS_RELAX_SHUFFLE", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _check_bc() -> bool:
    """When false, COMSOL anchor tests skip ``l_bc`` caps (identity + shuffle checks may still run)."""
    return os.environ.get("KINEMATICS_PHYSICS_CHECK_BC", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _iter_anchor_graph_files(phase: str) -> Iterator[Path]:
    cfg = VesselConfig(phase=phase)
    d = cfg.graph_output_dir
    # Removed the legacy phase 2 subfolder logic!
    if not d.is_dir():
        return
    for p in sorted(d.rglob("vessel_*.pt")):
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
    phase: str = "kinematics",
    distillation: bool = False,
    carreau_n: Optional[float] = None,
) -> Dict[str, float]:
    t = compute_kinematics_physics_terms(
        pred,
        data,
        kernels,
        phase=phase,
        boundary_data_weight=BOUNDARY_DATA_WEIGHT,
        distillation=distillation,
        carreau_n=carreau_n,
    )
    return _terms_dict_to_float(t)


def _collect_anchor_paths(phase: str) -> List[Path]:
    cap = _max_graphs_cap()
    out: List[Path] = []
    for p in _iter_anchor_graph_files(phase):
        if _load_anchor_graph(p) is None:
            continue
        out.append(p)
        if cap is not None and len(out) >= cap:
            break
    return out


def _safe_ratio(numer: float, denom: float) -> float:
    return float(numer) / max(float(denom), 1e-12)


class TestComsolAnchorPhysicsStrict(unittest.TestCase):
    """COMSOL labels vs training physics terms + shuffle sanity (no duplicated kernel logic)."""

    def test_kinematics_comsol_training_physics_consistency(self):
        min_n = max(1, int(os.environ.get("KINEMATICS_PHYSICS_MIN_ANCHORS", "1")))
        paths = _collect_anchor_paths("kinematics")
        if len(paths) < min_n:
            self.skipTest(
                f"Need at least {min_n} Kinematics COMSOL anchor graphs; found {len(paths)} "
                f"under {VesselConfig(phase='kinematics').graph_output_dir}."
            )

        phys_cfg = PhysicsConfig(phase="kinematics")
        kernels = PhysicsKernels(phys_cfg)
        relax = _relax_shuffle()
        failures: List[str] = []
        mom_ratio_max = _env_float("KINEMATICS_T1_MOM_RATIO_MAX", 0.2)
        # Smooth L1 makes WSS ratio comparisons linear-scale (vs prior quadratic MSE scale).
        wss_ratio_max = _env_float("KINEMATICS_T1_WSS_RATIO_MAX", 0.60)
        # Absolute closeness gates for Kinematics COMSOL anchors (not only relative-vs-shuffle).
        mom_abs_max = _env_float("KINEMATICS_T1_MOM_ABS_MAX", 1.0e-3)
        wss_abs_max = _env_float("KINEMATICS_T1_WSS_ABS_MAX", 1.0e-4)
        train_cont_scale = _env_float("KINEMATICS_T1_TRAIN_CONT_SCALE", 100.0)
        train_conflict_budget_max = _env_float("KINEMATICS_T1_TRAIN_CONFLICT_BUDGET_MAX", 0.3)
        abs_tail_pct = _env_float("KINEMATICS_T1_ABS_TAIL_PERCENTILE", 99.0)
        abs_tail_mult = _env_float("KINEMATICS_T1_ABS_TAIL_MULT", 2.0)
        mom_ok_values: List[float] = []
        wss_ok_values: List[float] = []
        cont_ok_values: List[float] = []
        io_ok_values: List[float] = []
        conflict_budget_values: List[float] = []
        mom_ok_by_stem: Dict[str, float] = {}
        wss_ok_by_stem: Dict[str, float] = {}

        for path in paths:
            data = _load_anchor_graph(path)
            assert data is not None
            stem = path.stem
            pred = _state_from_labels(data)
            seed = hash((stem, "row")) % (2**31)

            t_ok = _evaluate_terms(pred, data, kernels, phase="kinematics")
            mom_ok_values.append(t_ok["l_mom"])
            wss_ok_values.append(t_ok["l_wss"])
            cont_ok_values.append(t_ok["l_cont"])
            io_ok_values.append(t_ok["l_io"])
            mom_ok_by_stem[stem] = t_ok["l_mom"]
            wss_ok_by_stem[stem] = t_ok["l_wss"]

            if t_ok["l_data_kine"] >= EPS_DATA:
                failures.append(
                    f"{stem}: l_data_kine={t_ok['l_data_kine']:.3e} (expect <{EPS_DATA} when pred=y)"
                )
            if _check_bc() and t_ok["l_bc"] >= EPS_BC:
                failures.append(
                    f"{stem}: l_bc={t_ok['l_bc']:.3e} (expect <{EPS_BC} for no-slip labels)"
                )
            if t_ok["l_mom"] > mom_abs_max:
                failures.append(
                    f"{stem}: l_mom={t_ok['l_mom']:.6g} > abs cap {mom_abs_max:.6g}"
                )
            if t_ok["l_wss"] > wss_abs_max:
                failures.append(
                    f"{stem}: l_wss={t_ok['l_wss']:.6g} > abs cap {wss_abs_max:.6g}"
                )
            train_conflict_budget = (
                t_ok["l_mom"]
                + (train_cont_scale * t_ok["l_cont"])
                + (10.0 * t_ok["l_wss"])
            )
            conflict_budget_values.append(train_conflict_budget)
            if train_conflict_budget > train_conflict_budget_max:
                failures.append(
                    f"{stem}: train_conflict_budget={train_conflict_budget:.6g} > "
                    f"{train_conflict_budget_max:.6g} "
                    f"(mom + {train_cont_scale:g}*cont + 10*wss)"
                )

            if relax:
                continue

            pred_bad = _permute_rows(pred, seed)
            t_bad = _evaluate_terms(pred_bad, data, kernels, phase="kinematics")

            mom_ratio = _safe_ratio(t_ok["l_mom"], t_bad["l_mom"])
            if mom_ratio > mom_ratio_max:
                failures.append(
                    f"{stem}: l_mom ratio={mom_ratio:.3f} > {mom_ratio_max:.3f} "
                    f"(gt={t_ok['l_mom']:.6g}, shuffled={t_bad['l_mom']:.6g})"
                )
            wss_ratio = _safe_ratio(t_ok["l_wss"], t_bad["l_wss"])
            if wss_ratio > wss_ratio_max:
                failures.append(
                    f"{stem}: l_wss ratio={wss_ratio:.3f} > {wss_ratio_max:.3f} "
                    f"(gt={t_ok['l_wss']:.6g}, shuffled={t_bad['l_wss']:.6g})"
                )

        if mom_ok_values:
            mom_cap = float(np.percentile(np.asarray(mom_ok_values, dtype=np.float64), abs_tail_pct)) * abs_tail_mult
            wss_cap = float(np.percentile(np.asarray(wss_ok_values, dtype=np.float64), abs_tail_pct)) * abs_tail_mult
            for stem, v in mom_ok_by_stem.items():
                if v > mom_cap:
                    failures.append(f"{stem}: l_mom={v:.6g} > tail cap {mom_cap:.6g}")
            for stem, v in wss_ok_by_stem.items():
                if v > wss_cap:
                    failures.append(f"{stem}: l_wss={v:.6g} > tail cap {wss_cap:.6g}")
        # Always print closeness diagnostics so CI and local runs show absolute agreement levels.
        if mom_ok_values:
            mom_arr = np.asarray(mom_ok_values, dtype=np.float64)
            wss_arr = np.asarray(wss_ok_values, dtype=np.float64)
            cont_arr = np.asarray(cont_ok_values, dtype=np.float64)
            io_arr = np.asarray(io_ok_values, dtype=np.float64)
            cb_arr = np.asarray(conflict_budget_values, dtype=np.float64)
            print(
                "\n[Phase1 COMSOL Closeness] "
                f"anchors={len(mom_ok_values)} "
                f"mom(mean/p95/max)={mom_arr.mean():.3e}/{np.percentile(mom_arr, 95):.3e}/{mom_arr.max():.3e} "
                f"cont(mean/p95/max)={cont_arr.mean():.3e}/{np.percentile(cont_arr, 95):.3e}/{cont_arr.max():.3e} "
                f"wss(mean/p95/max)={wss_arr.mean():.3e}/{np.percentile(wss_arr, 95):.3e}/{wss_arr.max():.3e} "
                f"io(mean/p95/max)={io_arr.mean():.3e}/{np.percentile(io_arr, 95):.3e}/{io_arr.max():.3e} "
                f"budget(mean/p95/max)={cb_arr.mean():.3e}/{np.percentile(cb_arr, 95):.3e}/{cb_arr.max():.3e}"
            )

        self.assertEqual(
            failures,
            [],
            "Kinematics COMSOL / physics consistency failures:\n" + "\n".join(failures),
        )

    def test_kinematics_comsol_training_physics_consistency_coupled(self):
        min_n = max(1, int(os.environ.get("KINEMATICS_PHYSICS_MIN_ANCHORS", "1")))
        paths = _collect_anchor_paths("kinematics")
        if len(paths) < min_n:
            self.skipTest(
                f"Need at least {min_n} Kinematics COMSOL anchor graphs; found {len(paths)} "
                f"under {VesselConfig(phase='kinematics').graph_output_dir}."
            )

        kernels_carreau = PhysicsKernels(PhysicsConfig(phase="kinematics", rheology="carreau"))
        kernels_newtonian = PhysicsKernels(PhysicsConfig(phase="kinematics", rheology="newtonian"))
        carreau_n = kernels_carreau.cfg.n
        relax = _relax_shuffle()
        failures: List[str] = []
        mom_ratio_max = _env_float("KINEMATICS_T2_MOM_RATIO_MAX", 0.2)
        # Smooth L1 makes WSS ratio comparisons linear-scale (vs prior quadratic MSE scale).
        wss_ratio_max = _env_float("KINEMATICS_T2_WSS_RATIO_MAX", 0.60)
        rheo_ratio_max = _env_float("KINEMATICS_T2_RHEO_RATIO_MAX", 0.5)
        # Absolute closeness gates (not just better-than-shuffle), tuned to stay consistent with
        # Kinematics coupled training scales so physics terms do not conflict in optimization.
        mom_abs_max = _env_float("KINEMATICS_T2_MOM_ABS_MAX", 1.0e-3)
        cont_abs_max = _env_float("KINEMATICS_T2_CONT_ABS_MAX", 2.5e-3)
        rheo_abs_max = _env_float("KINEMATICS_T2_RHEO_ABS_MAX", 0.55)
        train_cont_scale = _env_float("KINEMATICS_T2_TRAIN_CONT_SCALE", 100.0)
        train_rheo_scale = _env_float("KINEMATICS_T2_TRAIN_RHEO_SCALE", 1.0)
        train_conflict_budget_max = _env_float("KINEMATICS_T2_TRAIN_CONFLICT_BUDGET_MAX", 0.9)
        newtonian_mu_spread_max = _env_float("KINEMATICS_T2_NEWTONIAN_MU_SPREAD_MAX", 1.0e-3)
        abs_tail_pct = _env_float("KINEMATICS_T2_ABS_TAIL_PERCENTILE", 99.0)
        abs_tail_mult = _env_float("KINEMATICS_T2_ABS_TAIL_MULT", 2.0)
        mom_ok_values: List[float] = []
        cont_ok_values: List[float] = []
        wss_ok_values: List[float] = []
        rheo_ok_values: List[float] = []
        io_ok_values: List[float] = []
        conflict_budget_values: List[float] = []
        mom_ok_by_stem: Dict[str, float] = {}
        wss_ok_by_stem: Dict[str, float] = {}
        rheo_ok_by_stem: Dict[str, float] = {}

        for path in paths:
            data = _load_anchor_graph(path)
            assert data is not None
            stem = path.stem
            rheology = path.parent.name.strip().lower()
            pred = _state_from_labels(data)
            seed_row = hash((stem, "row")) % (2**31)
            seed_mu = hash((stem, "mu")) % (2**31)
            stem_key = f"{rheology}/{stem}"

            if rheology == "newtonian":
                kernels = kernels_newtonian
                sample_carreau_n = None
                sample_train_rheo_scale = 0.0
                mu_spread = float((pred[:, PredChannels.MU_EFF_ND].max() - pred[:, PredChannels.MU_EFF_ND].min()).item())
                if mu_spread > newtonian_mu_spread_max:
                    failures.append(
                        f"{stem_key}: mu spread={mu_spread:.6g} > {newtonian_mu_spread_max:.6g} for newtonian anchor"
                    )
            elif rheology == "carreau":
                kernels = kernels_carreau
                sample_carreau_n = carreau_n
                sample_train_rheo_scale = train_rheo_scale
            else:
                continue

            t_ok = _evaluate_terms(
                pred,
                data,
                kernels,
                phase="kinematics",
                distillation=False,
                carreau_n=sample_carreau_n,
            )
            mom_ok_values.append(t_ok["l_mom"])
            cont_ok_values.append(t_ok["l_cont"])
            wss_ok_values.append(t_ok["l_wss"])
            io_ok_values.append(t_ok["l_io"])
            mom_ok_by_stem[stem_key] = t_ok["l_mom"]
            wss_ok_by_stem[stem_key] = t_ok["l_wss"]
            if rheology == "carreau":
                rheo_ok_values.append(t_ok["l_rheo"])
                rheo_ok_by_stem[stem_key] = t_ok["l_rheo"]

            if t_ok["l_data_kine"] >= EPS_DATA:
                failures.append(f"{stem_key}: l_data_kine={t_ok['l_data_kine']:.3e}")
            if t_ok["l_data_mu"] >= EPS_DATA:
                failures.append(f"{stem_key}: l_data_mu={t_ok['l_data_mu']:.3e}")
            if _check_bc() and t_ok["l_bc"] >= EPS_BC:
                failures.append(f"{stem_key}: l_bc={t_ok['l_bc']:.3e}")
            if t_ok["l_mom"] > mom_abs_max:
                failures.append(
                    f"{stem_key}: l_mom={t_ok['l_mom']:.6g} > abs cap {mom_abs_max:.6g}"
                )
            if t_ok["l_cont"] > cont_abs_max:
                failures.append(
                    f"{stem_key}: l_cont={t_ok['l_cont']:.6g} > abs cap {cont_abs_max:.6g}"
                )
            if rheology == "carreau" and t_ok["l_rheo"] > rheo_abs_max:
                failures.append(
                    f"{stem_key}: l_rheo={t_ok['l_rheo']:.6g} > abs cap {rheo_abs_max:.6g}"
                )
            train_conflict_budget = (
                t_ok["l_mom"]
                + (train_cont_scale * t_ok["l_cont"])
                + (sample_train_rheo_scale * t_ok["l_rheo"])
                + (10.0 * t_ok["l_wss"])
            )
            conflict_budget_values.append(train_conflict_budget)
            if train_conflict_budget > train_conflict_budget_max:
                failures.append(
                    f"{stem_key}: train_conflict_budget={train_conflict_budget:.6g} > "
                    f"{train_conflict_budget_max:.6g} "
                    f"(mom + {train_cont_scale:g}*cont + {sample_train_rheo_scale:g}*rheo + 10*wss)"
                )

            if relax:
                continue

            pred_bad = _permute_rows(pred, seed_row)
            t_bad = _evaluate_terms(
                pred_bad, data, kernels, phase="kinematics", distillation=False, carreau_n=sample_carreau_n
            )
            mom_ratio = _safe_ratio(t_ok["l_mom"], t_bad["l_mom"])
            if mom_ratio > mom_ratio_max:
                failures.append(
                    f"{stem_key}: l_mom ratio={mom_ratio:.3f} > {mom_ratio_max:.3f} "
                    f"(gt={t_ok['l_mom']:.6g}, row_shuf={t_bad['l_mom']:.6g})"
                )
            wss_ratio = _safe_ratio(t_ok["l_wss"], t_bad["l_wss"])
            if wss_ratio > wss_ratio_max:
                failures.append(
                    f"{stem_key}: l_wss ratio={wss_ratio:.3f} > {wss_ratio_max:.3f} "
                    f"(gt={t_ok['l_wss']:.6g}, row_shuf={t_bad['l_wss']:.6g})"
                )

            if rheology == "carreau":
                pred_mu_bad = _permute_mu_column(pred, seed_mu)
                t_mu = _evaluate_terms(
                    pred_mu_bad, data, kernels, phase="kinematics", distillation=False, carreau_n=sample_carreau_n
                )
                rheo_ratio = _safe_ratio(t_ok["l_rheo"], t_mu["l_rheo"])
                if rheo_ratio > rheo_ratio_max:
                    failures.append(
                        f"{stem_key}: l_rheo ratio={rheo_ratio:.3f} > {rheo_ratio_max:.3f} "
                        f"(gt={t_ok['l_rheo']:.6g}, mu_shuf={t_mu['l_rheo']:.6g})"
                    )

        if mom_ok_values:
            mom_cap = float(np.percentile(np.asarray(mom_ok_values, dtype=np.float64), abs_tail_pct)) * abs_tail_mult
            wss_cap = float(np.percentile(np.asarray(wss_ok_values, dtype=np.float64), abs_tail_pct)) * abs_tail_mult
            rheo_cap = (
                float(np.percentile(np.asarray(rheo_ok_values, dtype=np.float64), abs_tail_pct)) * abs_tail_mult
                if rheo_ok_values
                else None
            )
            for stem, v in mom_ok_by_stem.items():
                if v > mom_cap:
                    failures.append(f"{stem}: l_mom={v:.6g} > tail cap {mom_cap:.6g}")
            for stem, v in wss_ok_by_stem.items():
                if v > wss_cap:
                    failures.append(f"{stem}: l_wss={v:.6g} > tail cap {wss_cap:.6g}")
            if rheo_cap is not None:
                for stem, v in rheo_ok_by_stem.items():
                    if v > rheo_cap:
                        failures.append(f"{stem}: l_rheo={v:.6g} > tail cap {rheo_cap:.6g}")
        # Always print closeness diagnostics so CI and local runs show absolute agreement levels.
        if mom_ok_values:
            mom_arr = np.asarray(mom_ok_values, dtype=np.float64)
            cont_arr = np.asarray(cont_ok_values, dtype=np.float64)
            wss_arr = np.asarray(wss_ok_values, dtype=np.float64)
            rheo_arr = np.asarray(rheo_ok_values, dtype=np.float64) if rheo_ok_values else np.asarray([0.0], dtype=np.float64)
            io_arr = np.asarray(io_ok_values, dtype=np.float64)
            cb_arr = np.asarray(conflict_budget_values, dtype=np.float64)
            print(
                "\n[Phase2 COMSOL Closeness] "
                f"anchors={len(mom_ok_values)} "
                f"mom(mean/p95/max)={mom_arr.mean():.3e}/{np.percentile(mom_arr, 95):.3e}/{mom_arr.max():.3e} "
                f"cont(mean/p95/max)={cont_arr.mean():.3e}/{np.percentile(cont_arr, 95):.3e}/{cont_arr.max():.3e} "
                f"wss(mean/p95/max)={wss_arr.mean():.3e}/{np.percentile(wss_arr, 95):.3e}/{wss_arr.max():.3e} "
                f"rheo(mean/p95/max)={rheo_arr.mean():.3e}/{np.percentile(rheo_arr, 95):.3e}/{rheo_arr.max():.3e} "
                f"io(mean/p95/max)={io_arr.mean():.3e}/{np.percentile(io_arr, 95):.3e}/{io_arr.max():.3e} "
                f"budget(mean/p95/max)={cb_arr.mean():.3e}/{np.percentile(cb_arr, 95):.3e}/{cb_arr.max():.3e}"
            )

        self.assertEqual(
            failures,
            [],
            "Kinematics coupled COMSOL / physics consistency failures:\n" + "\n".join(failures),
        )

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
