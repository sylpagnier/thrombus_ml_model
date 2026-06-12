"""Step 11 extrapolation plausibility metrics (no GT beyond COMSOL window)."""

from __future__ import annotations

from typing import Any

import torch

from src.config import BiochemConfig
from src.core_physics.clot_continuous_time import extrapolated_t_out_max, macro_tau_at_index
from src.core_physics.clot_growth_masks import resolve_ceiling_mask


def compute_extrap_plausibility(
    data,
    phi_by_t: dict[int, torch.Tensor],
    *,
    sim_end_scale: float,
    bio_cfg: BiochemConfig | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Metrics for t > t_comsol_final (axis C)."""
    bio = bio_cfg or BiochemConfig(phase="biochem")
    dev = device or torch.device("cpu")
    n = int(data.y.shape[0])
    t_comsol_final = n - 1
    t_extrap_max = extrapolated_t_out_max(data, sim_end_scale=sim_end_scale)
    ceiling = resolve_ceiling_mask(data, dev, bio).reshape(-1).float()

    def _commit_frac(t: int) -> float:
        phi = phi_by_t.get(int(t))
        if phi is None:
            return float("nan")
        on_ceiling = phi.reshape(-1).to(dev) * ceiling
        return float((on_ceiling > 0.5).float().mean().item())

    frac_in = _commit_frac(t_comsol_final)
    frac_extrap = _commit_frac(t_extrap_max)

  # Monotonicity in tau over rolled indices
    taus: list[float] = []
    fracs: list[float] = []
    for t in sorted(phi_by_t.keys()):
        taus.append(macro_tau_at_index(data, int(t), bio_cfg=bio))
        fracs.append(_commit_frac(int(t)))
    mono_violations = 0
    for i in range(1, len(fracs)):
        if fracs[i] + 1e-4 < fracs[i - 1]:
            mono_violations += 1
    phi_monotone = mono_violations == 0

  # New seeds after COMSOL end: commit frac jump
    early_new_seeds = max(0.0, frac_extrap - frac_in) if frac_extrap == frac_extrap else 0.0

  # Growth rate proxy (pred frac / delta tau)
    if len(taus) >= 2 and taus[-1] > taus[-2]:
        growth_rate = (fracs[-1] - fracs[-2]) / max(taus[-1] - taus[-2], 1e-6)
    else:
        growth_rate = 0.0

    pred_frac_ceiling = frac_extrap
    pass_h15 = early_new_seeds <= 0.15
    pass_no_inlet_explosion = pred_frac_ceiling < 0.85

    return {
        "sim_end_scale": float(sim_end_scale),
        "t_comsol_final": t_comsol_final,
        "t_extrap_max": t_extrap_max,
        "pred_frac_in_window": frac_in,
        "pred_frac_ceiling_extrap": frac_extrap,
        "pred_frac_delta": early_new_seeds,
        "phi_monotone": float(1.0 if phi_monotone else 0.0),
        "mono_violations": mono_violations,
        "early_new_seeds": early_new_seeds,
        "growth_rate_si_proxy": growth_rate,
        "pass_h15_pred_frac": pass_h15,
        "pass_no_ceiling_paint": pass_no_inlet_explosion,
    }
