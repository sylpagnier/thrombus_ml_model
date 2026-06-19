"""patch_factory_comsol.py
-------------------------
Synthetic "Patch Factory" for the local Subgraph GNN, driven directly through the
COMSOL Python bridge (``mph``) -- *no Gmsh*.

Why bypass Gmsh
~~~~~~~~~~~~~~~
The local patch baseline is a pure linear shear (``u = shear_rate * y``, ``v = 0``).
An unstructured triangular mesh resolves a horizontal parallel flow across diagonal
faces and bleeds tiny spurious ``v`` into the field. Because the Subgraph GNN trains
on the residual ``dU = U_perturbed - U_baseline``, that mesh noise would dominate the
label. The fix is a *structured / mapped quad grid* whose element edges align with the
shear, giving ~zero numerical diffusion in the unperturbed region.

Rather than emit 1000 structured ``.msh`` files, we build ONE master ``.mph`` template
(``local_kine_template``) that already contains:
  * a flat 2000um x 350um box,
  * a mapped (structured) quad mesh,
  * a parametric continuous-viscosity clot (a Heaviside / smoothed mask scaling mu up to
    ``clot_mu`` over the clot footprint -- a high-viscosity POROUS zone, never a hole),
  * inlet linear-shear velocity ``u = shear_rate * y``, slip top wall, no-slip bottom wall.

This module then loops over the COMSOL *parameters* (no remeshing per sample), solves,
samples the field on a Python-defined structured grid, subtracts the analytical baseline
to form the residual, and writes ``patch_{i}.npz`` + ``patch_{i}.json``.

Template contract (build the .mph to match, or override via ``ComsolParamNames`` /
``PatchFactoryConfig.eval_exprs``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
COMSOL global parameters (names configurable; lengths pushed in [um] for GUI readability,
but units are explicit so any unit works):
  * ``shear_rate`` [1/s]  -- inlet/freestream shear, baseline ``u = shear_rate * y``
  * ``channel_h``  [um]   -- channel (domain) height H; the box MUST be built from this so
                            the COMSOL domain matches the Python extraction grid
  * ``clot_w``     [um]   -- clot streamwise width (footprint length)
  * ``clot_h``     [um]   -- clot wall-normal height
  * ``clot_mu``    [Pa*s] -- peak viscosity inside the clot mask
  * ``clot_x``     [um]   -- clot center x (defaults to L/2)
Field evaluation (component scope): ``u``, ``v``, ``p`` required; viscosity via the Laminar
Flow built-in ``spf.mu`` (optional/last).
Domain: x in [0, L], y in [0, H] (origin at bottom-left, bottom wall at y=0).

Clot model: ``Clot_Mask = flc2hs((clot_w/2 - |x-clot_x|)/1[um], 5) * flc2hs((clot_h-y)/1[um], 5)``
-- a smoothed rectangle (~5um edges) on the bottom wall, applied as a continuous viscosity
mask (high-viscosity porous zone, never a hole). Single morphology: metadata records
``clot_shape = "smoothed_rect"``.

Boundary conditions:
  * Inlet (x=0):   freestream shear ``u = shear_rate * y``, ``v = 0``.
  * Bottom (y=0):  no-slip wall (clot attaches here).
  * Top (y=H):     PRESCRIBED freestream velocity ``u = shear_rate * y``, ``v = 0`` (a moving
                   "lid" => exact Couette shear). This sustains the exact linear-shear baseline
                   so the analytical residual ``dU = U - shear_rate*y`` is clean. (A slip top
                   cannot sustain linear shear -- it imposes zero top stress -- so it would bias
                   the baseline.) Valid because ``channel_h`` is scaled to clot width, keeping
                   the lid in the decayed far field.
  * Outlet (x=L):  outflow / zero normal stress (pressure reference p=0).
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.utils.paths import comsol_models_dir, data_root

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sampling (single source of truth for the clot parameter sweep)
# ---------------------------------------------------------------------------

# Clot footprint sweep, expressed in *node counts* at the structured grid spacing so the
# morphology scales with resolution. Heavily biased toward wide, flat "smear" clots.
_CLOT_WIDTH_MIN_NODES = 20.0   # ~100um at 5um spacing
_CLOT_WIDTH_MAX_NODES = 125.0  # ~625um at 5um spacing
_CLOT_HEIGHT_MIN_NODES = 2.0   # ~10um
_CLOT_HEIGHT_MAX_NODES = 6.0   # ~30um (keep blockage ~3-10% of channel height)

_SHEAR_RATE_RANGE = (50.0, 5000.0)   # [1/s]
_CLOT_MU_RANGE = (0.1, 10.0)         # [Pa*s] soft gel -> near-solid
# The master template paints a single morphology: a smoothed rectangle (~5um edges) via
# the product of two flc2hs Heaviside masks. We record this so metadata matches reality.
_CLOT_SHAPE = "smoothed_rect"

# Channel-height policy. A slip/symmetry top is only a valid "infinite freestream" if it
# sits in the decayed far field. The disturbance from a shallow wide clot decays vertically
# over ~its streamwise width, so we scale H with clot width (not clot height).
_HEIGHT_FLOOR = 300e-6              # never below 300um (matches the small-clot regime)
_HEIGHT_CLEARANCE_FACTOR = 2.0     # H >= factor * clot_width keeps the lid in the far field
_HEIGHT_CEILING = 1500e-6          # cap so the widest smears stay affordable to mesh


@dataclass
class PatchSample:
    """One sampled patch (all SI units; lengths in meters)."""
    idx: int
    length: float
    height: float
    grid_spacing: float
    shear_rate: float
    clot_x_center: float
    clot_width: float
    clot_height: float
    clot_mu_peak: float
    clot_shape: str

    def to_meta(self) -> Dict[str, Any]:
        d = asdict(self)
        d["unit"] = "m"
        return d


def sample_patch_parameters(
    idx: int,
    rng: np.random.Generator,
    *,
    length: float = 2000e-6,
    grid_spacing: float = 5e-6,
    height_floor: float = _HEIGHT_FLOOR,
    height_clearance_factor: float = _HEIGHT_CLEARANCE_FACTOR,
    height_ceiling: float = _HEIGHT_CEILING,
) -> PatchSample:
    """Draw one patch parameter set.

    Domain is sized to avoid the nozzle/Venturi artifact (long channel, tall freestream),
    and the clot footprint is biased toward long, flat morphologies so the GNN learns the
    front stagnation zone vs. the parallel shear along a long clot top.

    Channel height is *adaptive*: ``H = clip(factor * clot_width, floor, ceiling)`` so the
    slip/symmetry top always sits in the decayed far field even for the widest smears (the
    vertical disturbance scale ~ clot width). ``H`` is driven into COMSOL as ``channel_h``
    and used for the extraction grid, so the two always agree.
    """
    s = grid_spacing

    wide_frac = float(rng.beta(2.0, 1.0))  # skew toward 1.0 => wide smears
    clot_width = (
        _CLOT_WIDTH_MIN_NODES + wide_frac * (_CLOT_WIDTH_MAX_NODES - _CLOT_WIDTH_MIN_NODES)
    ) * s
    clot_height = float(rng.uniform(_CLOT_HEIGHT_MIN_NODES, _CLOT_HEIGHT_MAX_NODES)) * s

    shear_rate = float(rng.uniform(*_SHEAR_RATE_RANGE))
    clot_mu_peak = float(rng.uniform(*_CLOT_MU_RANGE))

    # Far-field clearance: keep the prescribed-freestream top >= factor * clot_width above
    # the wall so the disturbance has decayed there.
    height = float(np.clip(height_clearance_factor * clot_width, height_floor, height_ceiling))

    # Center the clot so the widest (625um) smear still keeps upstream development and
    # downstream wake length inside the domain.
    clot_x_center = length / 2.0

    return PatchSample(
        idx=int(idx),
        length=float(length),
        height=float(height),
        grid_spacing=float(s),
        shear_rate=shear_rate,
        clot_x_center=float(clot_x_center),
        clot_width=float(clot_width),
        clot_height=float(clot_height),
        clot_mu_peak=float(clot_mu_peak),
        clot_shape=_CLOT_SHAPE,
    )


def baseline_shear_field(x: np.ndarray, y: np.ndarray, shear_rate: float) -> Tuple[np.ndarray, np.ndarray]:
    """Analytical unperturbed baseline: ``u = shear_rate * y``, ``v = 0``.

    Used to form the GNN training residual ``dU = U_comsol - U_baseline`` without a second
    solve (the clean baseline is exact for a flat channel under linear-shear inlet).
    """
    u_base = float(shear_rate) * np.asarray(y, dtype=np.float64)
    v_base = np.zeros_like(u_base)
    return u_base, v_base


def build_structured_grid(
    length: float, height: float, spacing: float
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Regular (nx, ny) sampling grid over [0,L] x [0,H]; returns flat x, y and (nx, ny)."""
    nx = max(2, int(round(length / spacing)) + 1)
    ny = max(2, int(round(height / spacing)) + 1)
    xs = np.linspace(0.0, length, nx)
    ys = np.linspace(0.0, height, ny)
    gx, gy = np.meshgrid(xs, ys, indexing="xy")  # shape (ny, nx)
    return gx.ravel(), gy.ravel(), nx, ny


# ---------------------------------------------------------------------------
# COMSOL contract / config
# ---------------------------------------------------------------------------

@dataclass
class ComsolParamNames:
    """Names of the COMSOL global parameters the master template exposes."""
    shear_rate: str = "shear_rate"
    channel_height: str = "channel_h"
    clot_width: str = "clot_w"
    clot_height: str = "clot_h"
    clot_mu: str = "clot_mu"
    clot_x: str = "clot_x"


@dataclass
class PatchFactoryConfig:
    template_path: Path = field(default_factory=lambda: comsol_models_dir() / "local_kine_template.mph")
    output_dir: Path = field(default_factory=lambda: data_root() / "processed" / "cfd_results_patch_factory")
    length: float = 2000e-6
    grid_spacing: float = 5e-6
    # Adaptive channel-height policy (slip-top far-field clearance vs clot width).
    height_floor: float = _HEIGHT_FLOOR
    height_clearance_factor: float = _HEIGHT_CLEARANCE_FACTOR
    height_ceiling: float = _HEIGHT_CEILING
    param_names: ComsolParamNames = field(default_factory=ComsolParamNames)
    # Field expressions evaluated at the grid (component scope). The viscosity is captured
    # via the Laminar Flow built-in ``spf.mu`` (the template defines only ``Clot_Mask`` and
    # applies it inside Fluid Properties -- there is no ``mu_final`` variable). mu is the
    # optional last expr; if it fails, u/v/p are still saved.
    eval_exprs: Sequence[str] = ("u", "v", "p", "spf.mu")
    dataset_tag: str = "dset1"
    # channel_h drives the rectangle height, so the geometry + mapped mesh must be rebuilt
    # before each solve. Mapped meshing on a rectangle is cheap.
    rebuild_geometry_each_solve: bool = True


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class PatchFactoryComsolGenerator:
    """Drive ``local_kine_template`` over sampled clot parameters; write residual datasets."""

    def __init__(self, config: Optional[PatchFactoryConfig] = None) -> None:
        self.cfg = config or PatchFactoryConfig()
        self.cfg.output_dir = Path(self.cfg.output_dir)
        self.cfg.template_path = Path(self.cfg.template_path)
        self.client = None
        self.model = None

    # -- COMSOL session lifecycle ------------------------------------------------
    def __enter__(self) -> "PatchFactoryComsolGenerator":
        import mph  # imported lazily so --dry-run works without a COMSOL install

        if not self.cfg.template_path.exists():
            raise FileNotFoundError(
                f"COMSOL template not found: {self.cfg.template_path}. Build the "
                "'local_kine_template' master .mph first (see module docstring contract)."
            )
        logger.info("Connecting to COMSOL... loading %s", self.cfg.template_path.name)
        self.client = mph.start()
        self.model = self.client.load(str(self.cfg.template_path))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if self.client is not None:
            try:
                self.client.clear()
            except Exception as exc:  # pragma: no cover - cleanup best effort
                logger.warning("mph client.clear() failed on exit: %s", exc)
        return False

    def _reconnect(self, reason: str) -> None:
        import mph

        logger.warning("COMSOL session recovery: %s", reason)
        try:
            if self.client is not None:
                self.client.clear()
        except Exception:
            pass
        self.client = mph.start(cores=1)
        self.model = self.client.load(str(self.cfg.template_path))

    @staticmethod
    def _is_solver_failure(exc: BaseException) -> bool:
        text = repr(exc)
        needles = (
            "FlException",
            "Failed to find a solution",
            "not converged",
            "Maximum number of Newton",
        )
        return any(n in text for n in needles)

    # -- Parameter push + field eval --------------------------------------------
    def _apply_parameters(self, s: PatchSample) -> None:
        p = self.cfg.param_names
        m = self.model
        um = 1e6  # push lengths in [um] for GUI readability (units are explicit either way)
        m.parameter(p.shear_rate, f"{s.shear_rate} [1/s]")
        m.parameter(p.clot_mu, f"{s.clot_mu_peak} [Pa*s]")
        # channel_h drives the domain box; it MUST match the extraction grid height.
        m.parameter(p.channel_height, f"{s.height * um} [um]")
        m.parameter(p.clot_width, f"{s.clot_width * um} [um]")
        m.parameter(p.clot_height, f"{s.clot_height * um} [um]")
        # clot_x is optional in the template (defaults to L/2); set best-effort.
        try:
            m.parameter(p.clot_x, f"{s.clot_x_center * um} [um]")
        except Exception:
            pass

    def _evaluate_grid(
        self, x: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Evaluate fields at (x, y) via the COMSOL Java ``Interp`` numerical feature."""
        coords_T = np.vstack([x, y])  # (2, N)
        model_j = self.model.java
        results = model_j.result()
        interp_name = "py_patch_interp_tmp"
        exprs = list(self.cfg.eval_exprs)

        def _run(expr_list: List[str]) -> List[np.ndarray]:
            interp_tag = results.numerical().create(interp_name, "Interp").tag()
            interp = results.numerical(interp_tag)
            try:
                interp.set("data", self.cfg.dataset_tag)
                interp.set("expr", expr_list)
                interp.setInterpolationCoordinates(coords_T.tolist())
                data = interp.getData()
                return [np.asarray(d, dtype=np.float64).ravel() for d in data]
            finally:
                try:
                    results.numerical().remove(interp_name)
                except Exception:
                    pass

        try:
            data = _run(exprs)
        except Exception as exc:
            # Retry without the (optional) viscosity expression if it is the culprit.
            if len(exprs) > 3:
                logger.warning("Interp with mu expr failed (%s); retrying u/v/p only.", exc)
                data = _run(exprs[:3])
            else:
                raise

        u, v, p = data[0], data[1], data[2]
        mu = data[3] if len(data) > 3 else None
        return u, v, p, mu

    # -- Per-sample + batch ------------------------------------------------------
    def _sidecar(self, s: PatchSample) -> Path:
        return self.cfg.output_dir / f"patch_{s.idx}.json"

    def _npz_path(self, s: PatchSample) -> Path:
        return self.cfg.output_dir / f"patch_{s.idx}.npz"

    def _write_sample(
        self,
        s: PatchSample,
        x: np.ndarray,
        y: np.ndarray,
        nx: int,
        ny: int,
        u: np.ndarray,
        v: np.ndarray,
        p: np.ndarray,
        mu: Optional[np.ndarray],
    ) -> None:
        u_base, v_base = baseline_shear_field(x, y, s.shear_rate)
        du = u - u_base
        dv = v - v_base
        payload: Dict[str, Any] = dict(
            x=x, y=y, u=u, v=v, p=p,
            u_base=u_base, du=du, dv=dv,
            grid_nx=nx, grid_ny=ny,
            length=s.length, height=s.height, grid_spacing=s.grid_spacing,
            shear_rate=s.shear_rate, clot_x_center=s.clot_x_center,
            clot_width=s.clot_width, clot_height=s.clot_height,
            clot_mu_peak=s.clot_mu_peak,
            clot_shape=s.clot_shape, config_id=s.idx,
        )
        if mu is not None:
            payload["mu"] = mu
        np.savez(self._npz_path(s), **payload)
        with open(self._sidecar(s), "w", encoding="utf-8") as f:
            json.dump(s.to_meta(), f, indent=2)

    def _rebuild_geometry_and_mesh(self) -> None:
        """Rebuild geometry + mapped mesh after ``channel_h`` changes the box height.

        COMSOL usually rebuilds upstream nodes on solve, but doing it explicitly is robust
        across mph versions. Best-effort: if these calls are unavailable, ``solve()`` still
        triggers the dependency rebuild.
        """
        for fn_name in ("build", "mesh"):
            fn = getattr(self.model, fn_name, None)
            if fn is None:
                continue
            try:
                fn()
            except Exception as exc:
                logger.warning("model.%s() failed (continuing to solve): %s", fn_name, exc)

    def _solve_one(self, s: PatchSample) -> bool:
        x, y, nx, ny = build_structured_grid(s.length, s.height, s.grid_spacing)
        self._apply_parameters(s)
        if self.cfg.rebuild_geometry_each_solve:
            self._rebuild_geometry_and_mesh()
        try:
            self.model.solve()
        except Exception as exc:
            logger.warning("[%s] solve failed: %s: %s", s.idx, type(exc).__name__, exc)
            raise
        u, v, p, mu = self._evaluate_grid(x, y)

        for name, arr in (("u", u), ("v", v), ("p", p)):
            if np.isnan(arr).any():
                logger.warning("[%s] NaNs in %s; discarding sample.", s.idx, name)
                return False
        if np.max(np.abs(u)) < 1e-9:
            logger.warning("[%s] trivial solution (u~0); discarding.", s.idx)
            return False

        self._write_sample(s, x, y, nx, ny, u, v, p, mu)
        return True

    def dry_run_one(self, s: PatchSample) -> None:
        """Write the analytical baseline grid + sidecar without COMSOL (I/O + sampling check)."""
        x, y, nx, ny = build_structured_grid(s.length, s.height, s.grid_spacing)
        u_base, v_base = baseline_shear_field(x, y, s.shear_rate)
        np.savez(
            self._npz_path(s),
            x=x, y=y, u=u_base, v=v_base, p=np.zeros_like(x),
            u_base=u_base, du=np.zeros_like(x), dv=np.zeros_like(x),
            grid_nx=nx, grid_ny=ny,
            length=s.length, height=s.height, grid_spacing=s.grid_spacing,
            shear_rate=s.shear_rate, clot_x_center=s.clot_x_center,
            clot_width=s.clot_width, clot_height=s.clot_height,
            clot_mu_peak=s.clot_mu_peak,
            clot_shape=s.clot_shape, config_id=s.idx, dry_run=True,
        )
        with open(self._sidecar(s), "w", encoding="utf-8") as f:
            json.dump(s.to_meta(), f, indent=2)

    def run_batch(
        self,
        n: int = 1000,
        *,
        seed: Optional[int] = None,
        start_idx: int = 0,
        overwrite: bool = False,
        dry_run: bool = False,
        max_reconnects: int = 20,
    ) -> Dict[str, Any]:
        """Sample ``n`` patches and (solve or dry-run) each, writing npz + json sidecars."""
        self.cfg.output_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(seed)
        samples = [
            sample_patch_parameters(
                start_idx + i, rng,
                length=self.cfg.length,
                grid_spacing=self.cfg.grid_spacing,
                height_floor=self.cfg.height_floor,
                height_clearance_factor=self.cfg.height_clearance_factor,
                height_ceiling=self.cfg.height_ceiling,
            )
            for i in range(n)
        ]

        written = 0
        skipped = 0
        failed = 0
        reconnects = 0
        consecutive_fails = 0

        try:
            from tqdm import tqdm
            iterator = tqdm(samples, desc="Patches", unit="patch")
        except Exception:
            iterator = samples

        for s in iterator:
            if not overwrite and self._npz_path(s).exists():
                skipped += 1
                continue
            if dry_run:
                self.dry_run_one(s)
                written += 1
                continue
            try:
                ok = self._solve_one(s)
                consecutive_fails = 0
            except Exception as exc:
                consecutive_fails += 1
                if (
                    reconnects < max_reconnects
                    and consecutive_fails >= 3
                    and (isinstance(exc, OSError) or self._is_solver_failure(exc))
                ):
                    self._reconnect(f"{consecutive_fails} consecutive failures: {exc}")
                    reconnects += 1
                    consecutive_fails = 0
                ok = False
            if ok:
                written += 1
            else:
                failed += 1

        summary = {
            "requested": n,
            "written": written,
            "skipped_existing": skipped,
            "failed": failed,
            "reconnects": reconnects,
            "dry_run": dry_run,
            "output_dir": str(self.cfg.output_dir),
        }
        logger.info("Patch factory done: %s", summary)
        return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Patch Factory via COMSOL mph bridge (structured grid, residual dU)."
    )
    p.add_argument("-n", "--num-patches", type=int, default=1000, help="Patches to generate.")
    p.add_argument("--seed", type=int, default=None, help="RNG seed (default: random).")
    p.add_argument("--start-idx", type=int, default=0, help="First patch index.")
    p.add_argument("--overwrite", action="store_true", help="Re-solve patches that already exist.")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip COMSOL: write analytical baseline grids + sidecars (validate sampling/IO).",
    )
    p.add_argument("--template", type=str, default=None, help="Override master .mph path.")
    p.add_argument("--output-dir", type=str, default=None, help="Override output directory.")
    p.add_argument("--grid-spacing-um", type=float, default=5.0, help="Structured grid spacing [um].")
    return p


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    cfg = PatchFactoryConfig(grid_spacing=args.grid_spacing_um * 1e-6)
    if args.template:
        cfg.template_path = Path(args.template)
    if args.output_dir:
        cfg.output_dir = Path(args.output_dir)

    gen = PatchFactoryComsolGenerator(cfg)
    if args.dry_run:
        # No COMSOL session needed for a dry run.
        gen.run_batch(
            n=args.num_patches, seed=args.seed, start_idx=args.start_idx,
            overwrite=args.overwrite, dry_run=True,
        )
    else:
        with gen:
            gen.run_batch(
                n=args.num_patches, seed=args.seed, start_idx=args.start_idx,
                overwrite=args.overwrite, dry_run=False,
            )
