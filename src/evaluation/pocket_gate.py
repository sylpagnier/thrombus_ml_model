"""Percentile-based pocket gate: deploy post-process, no retraining.

docs/WALL_MODEL_PLAN.md s2 diagnosed the wall model's failure as pocket SELECTION, not
growth: predicted connected components are pure, but the model commits to ~6x too many of
them. s2.3 found each component's own min hop-2 wall-node speed separates true pockets
from false ones (AUC 0.02 i.e. |sep| 0.96 on patient020; mechanism holds cross-vessel in
s2.5). This module is s4 Step 1: drop predicted components whose min hop-2 speed sits above
a per-vessel percentile of that vessel's own wall-node hop-2 speed distribution.

Deliberately NOT a global speed constant -- s2.4 found the fitted constant (0.12) sharp and
holdout-fitted, and s2.5 measured a 28.8x burden spread across vessels. The threshold must
be re-derived per vessel from its own flow distribution.
"""

from __future__ import annotations

import os

import numpy as np
import torch


def resolve_pocket_gate_percentile() -> float | None:
    """Percentile in [0, 100] from ``CLOT_POCKET_GATE_PCT``, or ``None`` to leave grading alone.

    No override => identical behaviour to before this module existed (gate off).
    """
    raw = (os.environ.get("CLOT_POCKET_GATE_PCT") or "").strip()
    if not raw:
        return None
    pct = float(raw)
    if not (0.0 <= pct <= 100.0):
        raise ValueError(f"CLOT_POCKET_GATE_PCT={raw!r} outside [0, 100]")
    return pct


def _neighbor_mean_nonzero(vals: torch.Tensor, row: torch.Tensor, col: torch.Tensor, n: int) -> torch.Tensor:
    """Mean of each node's nonzero neighbour values (one hop of ``_nz_hop``, probe_pocket_ranking.py)."""
    v = vals.reshape(-1)[col].to(torch.float32)
    nz = (v > 1e-9).to(torch.float32)
    s = torch.zeros(n, device=vals.device, dtype=torch.float32)
    c = torch.zeros(n, device=vals.device, dtype=torch.float32)
    s.index_add_(0, row, v * nz)
    c.index_add_(0, row, nz)
    return s / c.clamp(min=1.0)


def hop2_speed_field(data, device: torch.device, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Two-hop nonzero-neighbour-mean speed, matching the field diagnosed in s2.3/s2.5."""
    n = int(data.num_nodes)
    edge_index = data.edge_index.to(device=device)
    row, col = edge_index[0], edge_index[1]
    speed = torch.sqrt(u.reshape(-1) ** 2 + v.reshape(-1) ** 2).to(device=device, dtype=torch.float32)
    h1 = _neighbor_mean_nonzero(speed, row, col, n)
    h2 = _neighbor_mean_nonzero(h1, row, col, n)
    return h2


# ---------------------------------------------------------------- regime routing (s10.4)
# 34% of clot-rich vessels have the flow->clot relation INVERTED (low flow predicts LESS
# clot) -- s2.5's inverted set (024/036/001/011) plus 037, which is why no gate threshold
# ever worked there (s2.7/s2.9). s10.4 found a single DEPLOYABLE t=0 statistic that tells
# the two regimes apart: the 25th-percentile flow speed in the hop<=3 near-wall band.
#
#   separation AUC 0.975 | single threshold >= 0.060 -> inverted, 93.8% (30/32)
#   leave-one-vessel-out 90.6% (29/32) | permutation p=0.000 over all 59 aggregates
#
# Physically this follows from the COMSOL law: the dominant gate is `spf.sr < lss`, an
# ABSOLUTE threshold. band_speed_q25 measures whether the vessel has a genuine stagnation
# zone at all. Below ~0.06 it does, the low-shear gate fires, and low flow ranks clot
# correctly. Above it, even the slowest quartile is moving, the low-shear gate rarely
# fires, and deposition falls to the SEPARATION gate (d(sr,x) < sgt, the 21% minority
# mechanism) -- which keys on the shear GRADIENT and has a different, often opposite,
# spatial signature. Two mechanisms; one number says which is in charge.
# IMPORTANT -- two calibrations, and the deployed gate needs the second one.
# s10.4 fitted 0.060 on **GT** t=0 flow. `apply_pocket_gate` resolves t=0 flow through
# `resolve_species_rollout_uv`, which in deploy mode returns the **kinematics-predicted**
# field -- and scripts/probe_regime_route.py measured that the RGP-DEQ predictor inflates
# band_speed_q25 by a median 1.37x, worst (up to 9x) on exactly the slowest/most stagnant
# vessels where it has a resolution floor (patient043 0.0031 -> 0.0283). Rank order survives
# (Spearman +0.967), so the router still works -- but at the GT threshold it drops to 81.2%.
# Refit on predicted flow: 0.0822 -> 93.8% (matching GT), leave-one-vessel-out 84.4%.
DEFAULT_REGIME_BAND_SPEED_THRESH = 0.0822   # calibrated on PREDICTED (deploy) flow
REGIME_BAND_SPEED_THRESH_GT_FLOW = 0.060    # s10.4's GT-flow value; use only with flow_source=gt
DEFAULT_REGIME_BAND_HOPS = 3


def resolve_regime_route() -> float | None:
    """Threshold from ``CLOT_POCKET_GATE_REGIME_ROUTE``, or ``None`` to disable routing.

    Unset => routing OFF => byte-identical to the gate's behaviour before s10.4.
    ``"1"``/``"on"``/``"true"`` selects ``DEFAULT_REGIME_BAND_SPEED_THRESH``; any float
    overrides the threshold directly.
    """
    raw = (os.environ.get("CLOT_POCKET_GATE_REGIME_ROUTE") or "").strip().lower()
    if not raw or raw in ("0", "off", "false", "no"):
        return None
    if raw in ("1", "on", "true", "yes"):
        return float(DEFAULT_REGIME_BAND_SPEED_THRESH)
    return float(raw)


def band_speed_q25(
    data,
    device: torch.device,
    u: torch.Tensor,
    v: torch.Tensor,
    wall_mask: torch.Tensor | None,
    *,
    hops: int = DEFAULT_REGIME_BAND_HOPS,
    q: float = 25.0,
) -> float:
    """25th-pct speed in the hop<=``hops`` near-wall band -- the s10.4 regime statistic.

    Matches ``scripts/eda_clot_physics.py``'s ``band_speed_q25`` (BFS band from the wall,
    plain |u| speed, no nonzero-masking) so the calibrated threshold transfers. NOTE the
    EDA computed it on GT t=0 flow; callers here pass the deploy-faithful (kinematics)
    field, so ``scripts/probe_regime_route.py`` re-checks the threshold under both.
    """
    n = int(data.num_nodes)
    speed = torch.sqrt(u.reshape(-1) ** 2 + v.reshape(-1) ** 2).to(device=device, dtype=torch.float32)
    if wall_mask is None:
        band = torch.ones(n, dtype=torch.bool, device=speed.device)
    else:
        edge_index = data.edge_index.to(device=speed.device)
        row, col = edge_index[0], edge_index[1]
        band = wall_mask.reshape(-1).bool().to(speed.device)
        for _ in range(max(int(hops), 0)):
            grown = band.clone()
            grown[row[band[col]]] = True
            band = grown
    vals = speed[band]
    if vals.numel() == 0:
        return float("nan")
    return float(np.percentile(vals.detach().cpu().numpy(), q))


def apply_pocket_gate(
    phi_pred: torch.Tensor,
    data,
    device: torch.device,
    *,
    percentile: float,
    wall_mask: torch.Tensor | None,
    phi_thresh: float = 0.5,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Drop predicted connected components whose min hop-2 speed is at/above the gate.

    Returns ``(gated_phi_pred, stats)``. ``stats`` keys are prefixed ``deploy_pocket_gate_``
    so they merge straight into the metrics dict grading already builds.

    When ``CLOT_POCKET_GATE_REGIME_ROUTE`` is set (s10.4), the vessel is first classified
    stagnation- vs inverted-regime from its own t=0 flow, and the gate is SKIPPED entirely
    on inverted-regime vessels -- where s2.7 measured it doing net harm that no threshold
    fixes. Unset => routing off => unchanged behaviour.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    from src.core_physics.species_deploy_rollout import resolve_species_rollout_uv

    n = int(phi_pred.numel())
    stats: dict[str, float] = {
        "deploy_pocket_gate_pct": float(percentile),
        "deploy_pocket_gate_thresh": 0.0,
        "deploy_pocket_gate_ncomp_total": 0.0,
        "deploy_pocket_gate_ncomp_kept": 0.0,
        "deploy_pocket_gate_dropped_nodes": 0.0,
    }

    pred_pos = phi_pred.reshape(-1) > phi_thresh
    if int(pred_pos.sum().item()) == 0:
        return phi_pred, stats

    # t=0 flow, deploy-faithful (kinematics-predicted unless flow_source=gt) -- no clot
    # exists yet at t=0, so this is the same "no-GT-leak" velocity source the rollout uses.
    u0, v0 = resolve_species_rollout_uv(data, 0, device, for_training=False)

    # --- s10.4 regime routing: skip the gate on inverted-regime vessels ---
    route_thresh = resolve_regime_route()
    if route_thresh is not None:
        bq = band_speed_q25(data, device, u0, v0, wall_mask)
        stats["deploy_regime_band_speed_q25"] = float(bq)
        stats["deploy_regime_route_thresh"] = float(route_thresh)
        inverted = bool(np.isfinite(bq) and bq >= route_thresh)
        stats["deploy_regime_inverted"] = float(inverted)
        if inverted:
            # Gate is net-harmful here (s2.7 patient037: F1 0.285 -> 0.172). Leave the
            # prediction untouched rather than applying a rule built for the other regime.
            stats["deploy_pocket_gate_skipped_inverted"] = 1.0
            return phi_pred, stats
        stats["deploy_pocket_gate_skipped_inverted"] = 0.0

    h2 = hop2_speed_field(data, device, u0, v0)

    basis_mask = wall_mask.reshape(-1).bool() if wall_mask is not None else torch.ones(n, dtype=torch.bool, device=h2.device)
    basis_vals = h2[basis_mask].detach().cpu().numpy()
    if basis_vals.size == 0:
        return phi_pred, stats
    thresh = float(np.percentile(basis_vals, percentile))
    stats["deploy_pocket_gate_thresh"] = thresh

    idx = torch.nonzero(pred_pos, as_tuple=False).reshape(-1).cpu().numpy()
    ei = data.edge_index.cpu().numpy()
    keep_edge = np.isin(ei[0], idx) & np.isin(ei[1], idx)
    if keep_edge.any():
        remap = {int(val): i for i, val in enumerate(idx)}
        rr = np.fromiter((remap[int(x)] for x in ei[0][keep_edge]), dtype=int, count=int(keep_edge.sum()))
        cc = np.fromiter((remap[int(x)] for x in ei[1][keep_edge]), dtype=int, count=int(keep_edge.sum()))
        adj = coo_matrix((np.ones(rr.size), (rr, cc)), shape=(idx.size, idx.size))
        ncomp, lab = connected_components(adj, directed=False)
    else:
        ncomp, lab = idx.size, np.arange(idx.size)

    h2_cpu = h2.detach().cpu().numpy()
    drop = np.zeros(idx.size, dtype=bool)
    ncomp_kept = 0
    for k in range(ncomp):
        comp_pos = lab == k
        comp_nodes = idx[comp_pos]
        if float(h2_cpu[comp_nodes].min()) >= thresh:
            drop[comp_pos] = True
        else:
            ncomp_kept += 1

    gated = phi_pred.clone()
    drop_nodes = idx[drop]
    if drop_nodes.size:
        drop_idx = torch.as_tensor(drop_nodes, device=phi_pred.device, dtype=torch.long)
        gated.view(-1)[drop_idx] = 0.0

    stats["deploy_pocket_gate_ncomp_total"] = float(ncomp)
    stats["deploy_pocket_gate_ncomp_kept"] = float(ncomp_kept)
    stats["deploy_pocket_gate_dropped_nodes"] = float(drop_nodes.size)
    return gated, stats
