"""Per-vessel scope: legal priors in, per-vessel label scale out.

Two things must be established exactly once, when a vessel pack enters the pipeline, and then
honoured everywhere downstream. They are bundled here because applying one without the other is
a silent correctness bug, and this project has repeatedly been bitten by half-applied config
(WALL_MODEL_PLAN.md 12.3 v4/v5, 20.0, 20.1).

1. **Prior source** (section 17 Z2 contract). The stored ``u_prior``/``v_prior``/``mu_prior``
   columns are the converged clot-free CFD field, not an analytical prior (16.1). Under the
   declared geometry+IC/BC deployment contract they are illegal, so a training run must rewrite
   them. Crucially this must happen **before** the RGP-DEQ solve, because the DEQ *consumes*
   those columns (``ginodeq.py`` UV_PRIOR / MU_PRIOR) -- applying them afterwards would leave
   ``z_kin`` conditioned on the leaked field and the change would be a no-op.

2. **Label scale** (20.3 / 21.2). ``max Mat`` spans 45x across vessels against a fixed
   ``1e-4`` commit threshold, so "committed" is a stricter physical state on some vessels than
   others. ``mat_label_thresh_mode="rel_max"`` makes the *label* threshold a fraction of each
   vessel's own peak. The value is a property of the pack, so it is computed here and carried
   with it.
"""
from __future__ import annotations

from typing import Any

import torch

__all__ = [
    "prior_source_cache_tag",
    "resolve_vessel_mat_max",
    "prepare_vessel_data",
]


def prior_source_cache_tag(source: str | None = None) -> str:
    """Cache-key fragment for the active prior source.

    Pack caches are keyed on the feature stack but historically NOT on the prior source, so a
    run configured for ``analytic`` would silently reuse a pack built with the leaked ``stored``
    priors -- and, because the DEQ latent is baked into the cached pack, the leak would survive
    untouched. Including this tag makes that impossible.

    ``stored`` returns an empty tag so every pre-existing cache stays valid.
    """
    from src.data_gen.lib.legal_priors import resolve_prior_source

    src = (source or resolve_prior_source()).strip().lower()
    return "" if src == "stored" else f"_prior-{src}"


def resolve_vessel_mat_max(data: Any) -> float | None:
    """Peak GT Mat over the whole timeline, or ``None`` when the pack carries no GT.

    Returns ``None`` rather than 0.0 for a clot-free pack so callers fall back to the absolute
    threshold instead of collapsing it to zero (which would label every node committed).
    """
    y = getattr(data, "y", None)
    if y is None or not torch.is_tensor(y) or y.dim() != 3:
        return None
    from src.training.biochem_species_scope import MAT_CHANNEL
    from src.utils import species_channels as sc

    # MAT_CHANNEL is an index WITHIN the species block, not a column of `y`. `y` is
    # [u, v, p, mu_eff, <species block 4:16>], so the Mat column is SPECIES_BLOCK.start +
    # MAT_CHANNEL = 15. Indexing `y` with MAT_CHANNEL directly reads FG_log1p_nd instead, whose
    # peak (~0.69) is ~160x Mat's -- which would silently make every relative label threshold
    # far too high and mark nothing committed.
    idx = int(sc.SPECIES_BLOCK.start) + int(MAT_CHANNEL)
    if idx >= int(y.shape[-1]):
        return None
    peak = float(y[:, :, idx].max())
    return peak if peak > 0.0 else None


def prepare_vessel_data(
    data: Any,
    *,
    prior_source: str | None = None,
    phys_cfg: Any | None = None,
) -> tuple[Any, float | None]:
    """Apply the configured prior source and report the vessel's label scale.

    Returns ``(data, mat_max)``. ``data`` is the caller's object when the source is ``stored``
    (no copy), otherwise a shallow clone with the four prior columns rewritten.

    Call this immediately after loading a pack and **before** any kinematics solve.
    """
    from src.data_gen.lib.legal_priors import apply_prior_source, resolve_prior_source

    src = (prior_source or resolve_prior_source()).strip().lower()
    prepared = apply_prior_source(data, src, phys_cfg=phys_cfg)
    return prepared, resolve_vessel_mat_max(prepared)
