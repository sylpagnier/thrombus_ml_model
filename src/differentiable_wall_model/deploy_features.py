"""Rebuild the ``data.x`` prior channels from whichever flow the model is actually using.

WHY.  ``kine_x_v1_18ch`` carries four "prior" channels -- ``u_prior``, ``v_prior``,
``mu_prior_nd``, ``wss_prior_nd``.  Measured on the packs (2026-08-13):

    x.u_prior      corr 1.000 with GT u at t=0
    x.mu_prior_nd  corr 1.000 with GT mu_eff at t=0   (and only 0.05 with t_final)
    x.wss_prior_nd identically CONSTANT -- a dead channel

So they are the **GT t=0 fields**, not (as PHASE3_HANDOFF 0a asserted) the clot-affected
converged solution.  That makes them legal under the Phase-3 bandaid, which grants GT flow
at t=0.  But ``data.x`` is static: it keeps handing the model GT t=0 flow even when the
model is run with ``flow_source="pred"``, which silently re-introduces the bandaid into the
arm that is supposed to be free of it.

This module recomputes those channels from the selected flow field so the deployable arm
is actually deployable.  ``mu_prior_nd`` is regenerated with the Carreau-Yasuda law the
packs were built with -- validated against the shipped channel at pearson 0.996 and median
ratio 0.998 on patient007/013/043.
"""
from __future__ import annotations

import numpy as np
import torch

from src.core_physics.physics_wall_model import t0_flow_fields
from src.utils.channel_schema import KINE_X_SCHEMA, X_SCHEMAS

PRIOR_CHANNELS = ("u_prior", "v_prior", "mu_prior_nd", "wss_prior_nd")


def carreau_mu_nd(shear_si, phys_cfg):
    """Carreau-Yasuda viscosity from SI shear rate, in the label's non-dimensional scale."""
    mu_si = phys_cfg.mu_inf + (phys_cfg.mu_0 - phys_cfg.mu_inf) * (
        1.0 + (phys_cfg.lam * shear_si) ** phys_cfg.a
    ) ** ((phys_cfg.n - 1.0) / phys_cfg.a)
    return mu_si / phys_cfg.mu_viscosity_nd_reference


def rebuild_prior_channels(data, bio_cfg, phys_cfg, *, flow_source="pred", hops=None):
    """Return a copy of ``data.x`` whose prior channels come from ``flow_source``.

    ``flow_source='gt'`` returns the packs' own channels untouched (they already ARE the
    GT t=0 fields).  ``'pred'`` substitutes ``u0_pred``/``v0_pred`` and the viscosity
    implied by the predicted shear.
    """
    x = data.x.detach().clone()
    if flow_source == "gt":
        return x
    ch = list(X_SCHEMAS[KINE_X_SCHEMA].channels)
    if getattr(data, "u0_pred", None) is None:
        raise ValueError("pack has no u0_pred; cannot build deployable priors")
    hops = hops if hops is not None else 4          # the pred arm's stencil (13.5 / 5)
    f = t0_flow_fields(data, bio_cfg, hops=hops, flow_source="pred")
    up = data.u0_pred.reshape(-1).detach().to(x.dtype)
    vp = data.v0_pred.reshape(-1).detach().to(x.dtype)
    mu = torch.as_tensor(carreau_mu_nd(f.sr, phys_cfg), dtype=x.dtype)
    x[:, ch.index("u_prior")] = up
    x[:, ch.index("v_prior")] = vp
    x[:, ch.index("mu_prior_nd")] = mu
    # wss_prior_nd is identically constant in every pack -- it carries no information, so
    # leaving it alone cannot leak anything.
    return x


def prior_channel_audit(data, bio_cfg, phys_cfg) -> dict:
    """What each prior channel actually correlates with -- the check that produced the note above."""
    from src.config import STATE_CHANNEL_MU_EFF_ND

    ch = list(X_SCHEMAS[KINE_X_SCHEMA].channels)
    x = data.x.detach().numpy()
    out = {}

    def corr(a, b):
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    for name, ref0, refF in (
        ("u_prior", data.y[0, :, 0].numpy(), data.y[-1, :, 0].numpy()),
        ("v_prior", data.y[0, :, 1].numpy(), data.y[-1, :, 1].numpy()),
        ("mu_prior_nd", data.y[0, :, STATE_CHANNEL_MU_EFF_ND].numpy(),
         data.y[-1, :, STATE_CHANNEL_MU_EFF_ND].numpy()),
    ):
        v = x[:, ch.index(name)]
        out[name] = {"corr_t0": corr(v, ref0), "corr_tfinal": corr(v, refF),
                     "std": float(np.std(v))}
    out["wss_prior_nd"] = {"std": float(np.std(x[:, ch.index("wss_prior_nd")]))}
    return out
