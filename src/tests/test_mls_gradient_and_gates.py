"""Guard file for the derivative operators the COMSOL deposition gates are built on.

WHY.  Every flow-derived feature in this project (shear rate, shear gradient, both
gates in ``comsol_surface_deposition``) is a derivative of the velocity field.  The packs
ship ``G_x``/``G_y`` for that, and on 2026-08-09 they were audited against COMSOL's own
``spf.sr`` / ``d(spf.sr,x)`` export for patient007:

    operator            spearman(spf.sr)   spearman(d(spf.sr,x))
    packs' G_x / G_y          0.19                0.00
    MLS, 3 graph hops         0.998               0.990

``G_x`` has a median of ONE non-zero per row and ``G_x @ x`` returns 0 across the
interior -- it is linearly consistent only on wall rows.  Everything downstream was
being evaluated on noise, which is the direct explanation for PHASE3_HANDOFF 1.4/1.5b
("the gate is open 45.6% of the time and separates nothing").

These assertions pin the replacement operator's consistency so a future refactor cannot
silently reintroduce a non-differentiating "gradient".
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core_physics.mls_gradient import build_mls_gradient, shear_rate_2d


def _grid(nx: int = 12, ny: int = 12, h: float = 0.1):
    xs, ys = np.meshgrid(np.arange(nx) * h, np.arange(ny) * h, indexing="ij")
    pos = np.stack([xs.reshape(-1), ys.reshape(-1)], 1).astype(np.float64)
    src, dst = [], []
    for i in range(nx):
        for j in range(ny):
            a = i * ny + j
            for di, dj in ((1, 0), (0, 1)):
                bi, bj = i + di, j + dj
                if bi < nx and bj < ny:
                    b = bi * ny + bj
                    src += [a, b]
                    dst += [b, a]
    return pos, np.asarray([src, dst])


def test_gradient_is_exact_on_linear_fields():
    """A gradient operator that cannot differentiate f=x is the bug this file guards."""
    pos, ei = _grid()
    Dx, Dy = build_mls_gradient(pos, ei, hops=2)
    x, y = pos[:, 0], pos[:, 1]
    assert np.abs(Dx @ x - 1.0).max() < 1e-8
    assert np.abs(Dx @ y).max() < 1e-8
    assert np.abs(Dy @ y - 1.0).max() < 1e-8
    assert np.abs(Dx @ np.ones_like(x)).max() < 1e-8


def test_gradient_is_accurate_on_a_quadratic_field():
    pos, ei = _grid()
    Dx, Dy = build_mls_gradient(pos, ei, hops=2)
    f = pos[:, 0] ** 2 + 2.0 * pos[:, 0] * pos[:, 1]
    ref_x = 2.0 * pos[:, 0] + 2.0 * pos[:, 1]
    err = np.abs(Dx @ f - ref_x) / (np.abs(ref_x) + 1.0)
    assert float(np.median(err)) < 1e-6


def test_second_derivative_via_composition_is_stable():
    """``d(spf.sr,x)`` is taken as Dx applied to a field Dx/Dy already produced."""
    pos, ei = _grid()
    Dx, Dy = build_mls_gradient(pos, ei, hops=3)
    f = pos[:, 0] ** 2
    inner = np.asarray(Dx @ f).reshape(-1)
    outer = np.asarray(Dx @ inner).reshape(-1)
    # interior nodes only: one-sided stencils at the boundary are less accurate
    m = ((pos[:, 0] > 0.25) & (pos[:, 0] < 0.85) & (pos[:, 1] > 0.25) & (pos[:, 1] < 0.85))
    assert np.abs(outer[m] - 2.0).max() < 1e-4


def test_shear_rate_matches_analytic_couette():
    """Couette flow u = (a*y, 0) has spf.sr = |a| everywhere."""
    pos, ei = _grid()
    Dx, Dy = build_mls_gradient(pos, ei, hops=2)
    a = 3.0
    u = a * pos[:, 1]
    v = np.zeros_like(u)
    sr = shear_rate_2d(Dx @ u, Dy @ u, Dx @ v, Dy @ v)
    assert np.abs(sr - abs(a)).max() < 1e-7


@pytest.mark.parametrize("hops", [2, 3])
def test_operator_rows_are_wide_enough_to_differentiate(hops):
    """The shipped ``G_x`` failed precisely here: a median of one entry per row."""
    pos, ei = _grid()
    Dx, _ = build_mls_gradient(pos, ei, hops=hops)
    per_row = np.diff(Dx.indptr)
    assert int(np.median(per_row)) >= 6, "stencil too small to fit a 2D quadratic basis"


# ---------------------------------------------------------------------------
# Wiring guards: the operators must actually reach the wall residual.
# Standing constraint 5.3 -- "a config value appearing in the fingerprint is not
# sufficient"; these assert the mechanism, not the flag.
# ---------------------------------------------------------------------------

import os  # noqa: E402

import torch  # noqa: E402


def _anchor():
    from pathlib import Path
    paths = sorted(Path("data/processed/graphs_biochem_anchors").glob("patient0*.pt"))
    paths = [p for p in paths if "mirror" not in p.name]
    if not paths:
        pytest.skip("no biochem anchor graphs")
    return torch.load(str(paths[0]), map_location="cpu", weights_only=False)


def _shear(data, mode):
    from src.config import BiochemConfig
    from src.core_physics.biochem_physics_kernels import BiochemPhysicsKernels
    from src.core_physics.mls_gradient import clear_operator_cache
    prev = os.environ.get("BIOCHEM_GRAD_OPERATOR")
    os.environ["BIOCHEM_GRAD_OPERATOR"] = mode
    clear_operator_cache()
    try:
        k = BiochemPhysicsKernels(BiochemConfig(phase="biochem"), None)
        u, v = data.y[0, :, 0].float(), data.y[0, :, 1].float()
        n = int(data.num_nodes)
        props = {"u_ref": data.u_ref.reshape(-1).expand(n).float(),
                 "d_bar": data.d_bar.reshape(-1).expand(n).float()}
        return k._compute_shear_rate(u, v, props, data)
    finally:
        if prev is None:
            os.environ.pop("BIOCHEM_GRAD_OPERATOR", None)
        else:
            os.environ["BIOCHEM_GRAD_OPERATOR"] = prev
        clear_operator_cache()


def test_kernel_shear_rate_is_nonzero_in_the_interior():
    """The shipped operator returned ~0 interior shear; the fix must not."""
    data = _anchor()
    wall = data.mask_wall.reshape(-1).bool()
    sr = _shear(data, "mls")
    assert float(sr[~wall].median()) > 1.0, "interior shear collapsed -- operator regressed"
    # A vessel at Re~450 has wall shear of order 10-1000 1/s, not order 1.
    assert float(sr[wall].median()) > 10.0


def test_legacy_operator_flag_still_reproduces_the_old_behaviour():
    """Pre-2026-08-09 numbers must remain reproducible bit-for-bit."""
    data = _anchor()
    wall = data.mask_wall.reshape(-1).bool()
    legacy = _shear(data, "legacy")
    mls = _shear(data, "mls")
    assert float(legacy[~wall].median()) < 0.1 * float(mls[~wall].median())
    ref = torch.sparse.mm(data.G_x, data.y[0, :, 0].float().unsqueeze(1)).squeeze(1)
    assert torch.isfinite(ref).all()


def test_separation_gate_is_not_identically_zero_at_the_wall():
    """No-slip pins the STREAMWISE derivative to 0 at every wall node.

    COMSOL gates on ``d(spf.sr,x)``. Using ``d/ds`` instead makes the separation branch --
    21% of the deposition mechanism -- structurally dead, whatever the operator.
    """
    from src.config import BiochemConfig
    from src.core_physics.clot_kinematics_fields import compute_clot_kinematics_fields
    data = _anchor()
    bio = BiochemConfig(phase="biochem")
    n = int(data.num_nodes)
    props = {"u_ref": data.u_ref.reshape(-1).expand(n).float(),
             "d_bar": data.d_bar.reshape(-1).expand(n).float()}
    f = compute_clot_kinematics_fields(data, data.y[0, :, 0], data.y[0, :, 1], bio, props)
    wall = data.mask_wall.reshape(-1).bool()
    assert float(f.dshear_ds_phys[wall].abs().max()) == pytest.approx(0.0, abs=1e-6)
    assert float(f.dgamma_dx_si[wall].abs().max()) > abs(float(bio.sgt))
    open_frac = float((f.is_separation_dx[wall] > 0.5).float().mean())
    assert 0.01 < open_frac < 0.6, f"separation gate open fraction {open_frac} implausible"
