"""Flow-mediated recurrent refinement -- the one form of autoregression the physics allows.

PHASE6_RESULTS 21.3 is explicit: **contact-mediated** autoregression is dead (tried twice,
failed twice) because clot appears where the FIELD is bad, not where clot already is.  What
is alive is **flow-mediated** creep: as the clot occludes the lumen the shear field
redistributes and gates that were shut at t=0 open later (PHASE7 10.4 prices the mask half
at +0.051; PHASE7 12.4 shows it is also the dominant lever on Mat ordering).

The feedback channels are chosen to be exactly the two couplings the physics names, and
nothing else:

  ``p_self``    the current occlusion estimate at this node;
  ``p_owner``   the current occlusion at this node's nearest WALL node.  Off-wall ``Mat`` is
                ~0.16x its owner's (PHASE7 3.2), so this is the attenuation law written as
                a channel -- the single most informative thing a shell node can be told;
  ``p_1hop``    / ``p_2hop`` neighbourhood occlusion, the blockage proxy that stands in for
                the cross-section the clot has closed.

An explicit cross-section ball operator was tried first and is **not** used: reaching
across the lumen makes the operator ~30M nonzeros on a 15k-node mesh and OOMs a 4 GB card,
while two rounds of mesh diffusion carry the same information.

Weights are shared across rounds (deep-equilibrium style) and ``rounds = 1`` collapses
exactly to the feed-forward model.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch

N_FEEDBACK = 4
N_FEEDBACK_ADV = 6


def neighbour_operator(ei: np.ndarray, n: int) -> sp.csr_matrix:
    """Row-normalised adjacency: ``A @ p`` is the mean occlusion of the 1-hop ring."""
    A = sp.coo_matrix((np.ones(ei.shape[1], np.float32), (ei[0], ei[1])), shape=(n, n)).tocsr()
    A = ((A + A.T) > 0).astype(np.float32)
    A.setdiag(0.0)
    A.eliminate_zeros()
    deg = np.asarray(A.sum(axis=1)).reshape(-1)
    deg[deg == 0] = 1.0
    return sp.diags(1.0 / deg).astype(np.float32) @ A


def feedback_channels(p: torch.Tensor, At: torch.Tensor, owner: torch.Tensor) -> torch.Tensor:
    """[N, 4]: self, owner-wall, 1-hop mean, 2-hop mean."""
    q = p.reshape(-1, 1)
    h1 = torch.sparse.mm(At, q)
    h2 = torch.sparse.mm(At, h1)
    return torch.cat([q, p[owner].reshape(-1, 1), h1, h2], dim=1)


def advective_operators(pos: np.ndarray, ei: np.ndarray, u: np.ndarray, v: np.ndarray,
                        ) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """Row-normalised UPSTREAM and DOWNSTREAM occlusion operators.

    Built from the same upwind flux matrix `src/clot_ml/transport.py` solves with, so the
    recurrence and the transport features agree about which way the flow goes.
    ``W_up @ p`` is the flux-weighted mean occlusion of the nodes feeding this one;
    ``W_dn @ p`` is the same for the nodes it feeds.
    """
    from src.clot_ml.transport import upwind_operator

    F, _ = upwind_operator(np.asarray(pos, float), ei,
                           np.asarray(u, float), np.asarray(v, float))

    def rownorm(M):
        d = np.asarray(M.sum(axis=1)).reshape(-1)
        d[d <= 0] = 1.0
        return (sp.diags(1.0 / d) @ M).astype(np.float32)

    return rownorm(F.T.tocsr()), rownorm(F.tocsr())


def feedback_channels_advective(p: torch.Tensor, At: torch.Tensor, Wup: torch.Tensor,
                                Wdn: torch.Tensor, owner: torch.Tensor) -> torch.Tensor:
    """[N, 6]: self, owner-wall, isotropic 1-hop, 1-hop UPSTREAM, 2-hop upstream, downstream.

    WHY THIS EXISTS.  `docs/PHASE9_ML.md` 2c calls the recurrence "flow-mediated" and
    reports it as the single largest architectural win -- but the channels it actually
    feeds back are `A p` and `A^2 p` with `A` the **isotropic** row-normalised adjacency.
    That is mesh diffusion, and `PHASE6_RESULTS` 3.4 measured isotropic smoothing of the
    source to make the fit *worse*: the non-locality here is advective.  The message
    passing was made anisotropic on exactly that evidence and the recurrence was not.

    The physics is unambiguous about the direction.  `Mat` obeys
    `dMat/dt + u.grad(Mat) = 0` with zero diffusion (PHASE7 1.1), so a node's occlusion is
    influenced by what is UPSTREAM of it, not by its undirected neighbourhood.  These
    channels apply the transport operator's own upwind weights instead.

    The isotropic 1-hop term is kept, because blockage also acts across the lumen (a clot
    on the far wall narrows this node's cross-section) and that coupling is genuinely not
    directional.
    """
    q = p.reshape(-1, 1)
    h1 = torch.sparse.mm(At, q)
    up1 = torch.sparse.mm(Wup, q)
    up2 = torch.sparse.mm(Wup, up1)
    dn1 = torch.sparse.mm(Wdn, q)
    return torch.cat([q, p[owner].reshape(-1, 1), h1, up1, up2, dn1], dim=1)
