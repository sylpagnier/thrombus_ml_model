"""Customer-facing deploy pipeline: kine + corrector + wall/offwall species + clot-phi.

Returns a scrubbable trajectory for the matplotlib Predict app.
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import torch

from src.biochem_gnn.mat_growth_simple import apply_mat_growth_leg_env, mat_growth_leg_spec
from src.config import BiochemConfig, PhysicsConfig
from src.core_physics.species_gnn_clot_rollout import (
    load_species_gnn_rollout_bundle,
    prepare_species_gnn_rollout_static,
    rollout_species_gnn_species_series,
)
from src.core_physics.species_viscosity_calibration import resolve_deploy_gelation_beta
from src.core_physics.t0_device import require_cuda_device
from src.core_physics.t0_mu_physics import rollout_t0_clot_phi
from src.core_physics.t0_rung_config import RUNG2_GAMMA_MODE, t0_rung2_env
from src.inference.corrector_coupling import CorrectorCoupledFlow
from src.inference.species_gnn_deploy_env import load_deploy_manifest, species_gnn_deploy_env
from src.utils.paths import get_project_root

# Canonical = WC_v7_clot_phi_mse (promoted 2026-07-19). locked/ is the source of truth;
# mat_canonical_deploy/ and species/best.pth are synced aliases.
DEFAULT_WALL_CKPT = Path("outputs/biochem/biochem_gnn/locked/species_gnn_best.pth")
# Unified WC_v7 already covers off-wall; two-model offwall is opt-in via CUSTOMER_OFFWALL_CKPT.
DEFAULT_OFFWALL_CKPT: Path | None = None
DEFAULT_MAT_LEG = "WC_v7_clot_phi_mse"


@dataclass
class CustomerTrajectory:
    """Cached per-step fields for the time slider."""

    t_sec: np.ndarray
    pos: np.ndarray
    vel_mag: dict[int, np.ndarray]
    mu_eff_si: dict[int, np.ndarray]
    phi: dict[int, np.ndarray]
    elapsed_s: float = 0.0
    n_steps: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    def frame(self, index: int) -> dict[str, np.ndarray | float]:
        i = int(max(0, min(index, self.n_steps - 1)))
        return {
            "index": i,
            "t_sec": float(self.t_sec[i]),
            "vel_mag": self.vel_mag[i],
            "mu_eff_si": self.mu_eff_si[i],
            "phi": self.phi[i],
        }


def _abs(path: Path | str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = get_project_root() / p
    return p


def _resolve_wall_ckpt(explicit: Path | str | None = None) -> Path:
    raw = str(explicit or os.environ.get("CUSTOMER_WALL_CKPT") or "").strip()
    p = _abs(raw) if raw else _abs(DEFAULT_WALL_CKPT)
    if not p.is_file():
        # Fall back to locked species GNN
        alt = _abs("outputs/biochem/biochem_gnn/locked/species_gnn_best.pth")
        if alt.is_file():
            return alt
    return p


def _resolve_offwall_ckpt(explicit: Path | str | None = None) -> Path | None:
    raw = str(explicit or os.environ.get("CUSTOMER_OFFWALL_CKPT") or "").strip()
    if not raw:
        if DEFAULT_OFFWALL_CKPT is None:
            return None
        raw = str(DEFAULT_OFFWALL_CKPT)
    p = _abs(raw)
    return p if p.is_file() else None


@contextmanager
def _customer_deploy_env(
    *,
    wall_ckpt: Path,
    offwall_ckpt: Path | None,
    mat_leg: str = DEFAULT_MAT_LEG,
) -> Iterator[dict[str, str]]:
    """Apply deploy + mat-growth env, then restore."""
    manifest = load_deploy_manifest()
    overrides: dict[str, str] = {
        "T0_R4_FLOW_SOURCE": "kinematics",
        "SPECIES_GNN_CLOUT_CKPT": str(wall_ckpt).replace("\\", "/"),
        "SPECIES_CONTINUOUS_CKPT": str(wall_ckpt).replace("\\", "/"),
        "T0_R4_SPECIES_GNN_CKPT": str(wall_ckpt).replace("\\", "/"),
        "BIOCHEM_CORRECTOR_COUPLING": "1",
    }
    # Mat-growth recipe (wall + off-wall clot physics)
    try:
        spec = mat_growth_leg_spec(mat_leg)
        overrides.update({k: str(v) for k, v in spec.env_overrides.items()})
    except Exception:
        pass

    if offwall_ckpt is not None:
        overrides["SPECIES_TWO_MODEL_MODE"] = "1"
        overrides["SPECIES_OFFWALL_MODEL_CKPT"] = str(offwall_ckpt).replace("\\", "/")
        overrides.setdefault("SPECIES_TWO_MODEL_ROUTE", os.environ.get("SPECIES_TWO_MODEL_ROUTE", "frontier"))
        overrides.setdefault(
            "SPECIES_TWO_MODEL_FRONTIER_HOPS",
            os.environ.get("SPECIES_TWO_MODEL_FRONTIER_HOPS", "2"),
        )
    else:
        overrides.setdefault("SPECIES_TWO_MODEL_MODE", "0")

    keys = set(overrides) | {
        "SPECIES_GNN_CLOUT_CKPT",
        "SPECIES_CONTINUOUS_CKPT",
        "T0_R4_SPECIES_GNN_CKPT",
        "SPECIES_OFFWALL_MODEL_CKPT",
        "SPECIES_TWO_MODEL_MODE",
        "SPECIES_TWO_MODEL_ROUTE",
        "SPECIES_TWO_MODEL_FRONTIER_HOPS",
        "BIOCHEM_CORRECTOR_COUPLING",
        "T0_R4_FLOW_SOURCE",
    }
    saved = {k: os.environ.get(k) for k in keys}
    try:
        with species_gnn_deploy_env(manifest, overrides=overrides, prefer_loao=False):
            apply_mat_growth_leg_env(mat_leg, force=True)
            for k, v in overrides.items():
                os.environ[k] = str(v)
            yield overrides
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class CustomerDeployPipeline:
    """Load once, run many geometries."""

    def __init__(
        self,
        *,
        device: torch.device | None = None,
        wall_ckpt: Path | str | None = None,
        offwall_ckpt: Path | str | None = None,
        mat_leg: str = DEFAULT_MAT_LEG,
        require_cuda: bool = True,
    ) -> None:
        self.device = device or (require_cuda_device() if require_cuda else torch.device("cpu"))
        self.wall_ckpt = _resolve_wall_ckpt(wall_ckpt)
        self.offwall_ckpt = _resolve_offwall_ckpt(offwall_ckpt)
        self.mat_leg = mat_leg
        self._bundle = None
        self._flow_provider: CorrectorCoupledFlow | None = None

    def _ensure_loaded(self) -> None:
        if self._bundle is not None:
            return
        if not self.wall_ckpt.is_file():
            raise FileNotFoundError(
                f"Wall/species checkpoint missing: {self.wall_ckpt}. "
                "Promote WC_v7_clot_phi_mse to locked/species_gnn_best.pth first."
            )
        with _customer_deploy_env(
            wall_ckpt=self.wall_ckpt,
            offwall_ckpt=self.offwall_ckpt,
            mat_leg=self.mat_leg,
        ):
            self._bundle = load_species_gnn_rollout_bundle(self.wall_ckpt, device=self.device)
        if self._bundle is None:
            raise FileNotFoundError(f"Could not load species GNN bundle: {self.wall_ckpt}")
        self._flow_provider = CorrectorCoupledFlow(device=self.device, phys_cfg=PhysicsConfig(phase="biochem"))

    def run(
        self,
        data,
        *,
        t_final_s: float | None = None,
        progress: Callable[[str], None] | None = None,
        include_velocity: bool = True,
    ) -> CustomerTrajectory:
        """Species + clot-phi trajectory; optionally couple local corrector for velocity."""
        # Avoid tqdm/signal handlers that break when called outside the main UI path.
        os.environ.setdefault("BIOCHEM_TQDM", "0")
        os.environ.setdefault("BIOCHEM_QUIET", "1")

        self._ensure_loaded()
        assert self._bundle is not None and self._flow_provider is not None
        log = progress or (lambda _msg: None)

        data = data.clone() if hasattr(data, "clone") else data
        if t_final_s is not None and hasattr(data, "t") and data.t is not None:
            n = int(data.y.shape[0])
            data.t = torch.linspace(0.0, float(t_final_s), steps=n, dtype=torch.float32)

        phys = PhysicsConfig(phase="biochem")
        t_end = float(t_final_s) if t_final_s is not None else float(data.t[-1].item())
        bio = BiochemConfig(phase="biochem", t_final=t_end)
        data = data.to(self.device)

        log("[i] Loading deploy environment and species GNN...")
        t0 = time.perf_counter()
        with _customer_deploy_env(
            wall_ckpt=self.wall_ckpt,
            offwall_ckpt=self.offwall_ckpt,
            mat_leg=self.mat_leg,
        ):
            log("[i] Preparing band features (kinematics)...")
            static = prepare_species_gnn_rollout_static(data, device=self.device)
            log("[i] Rolling out wall/off-wall species GNN...")
            pred_species = rollout_species_gnn_species_series(
                data,
                self._bundle,
                static,
                phys_cfg=phys,
                bio_cfg=bio,
                device=self.device,
            )
            gel_beta = resolve_deploy_gelation_beta(self.device)
            nuc_hops = int(os.environ.get("CLOT_V2_NUCLEATION_HOPS", "2"))
            log("[i] Rolling out clot-phi / gelation...")
            with t0_rung2_env():
                traj = rollout_t0_clot_phi(
                    data,
                    phys,
                    bio,
                    self.device,
                    gamma_mode=RUNG2_GAMMA_MODE,
                    flow_source="kinematics",
                    pred_species_series=pred_species,
                    nucleation=True,
                    nucleation_hops=nuc_hops,
                    gelation_beta=gel_beta,
                )

        pos = data.x[:, :2].detach().cpu().numpy()
        vel_all: dict[int, np.ndarray] = {}
        mu_all: dict[int, np.ndarray] = {}
        phi_all: dict[int, np.ndarray] = {}
        t_keys = sorted(traj.keys())
        if include_velocity:
            log("[i] Coupling local kinematic corrector...")
            for ti in t_keys:
                mu_eff_si = traj[ti]["mu"].to(self.device)
                u, v = self._flow_provider.couple(data, mu_eff_si, publish=False)
                vel_all[ti] = torch.sqrt(u**2 + v**2).detach().cpu().numpy()
                mu_all[ti] = traj[ti]["mu"].detach().cpu().numpy()
                phi_all[ti] = traj[ti]["phi"].detach().cpu().numpy()
        else:
            log("[i] Skipping velocity corrector (clot-only mode)...")
            for ti in t_keys:
                mu_all[ti] = traj[ti]["mu"].detach().cpu().numpy()
                phi_all[ti] = traj[ti]["phi"].detach().cpu().numpy()
                # Placeholder so scrubber APIs stay uniform
                vel_all[ti] = np.zeros_like(phi_all[ti], dtype=np.float32)

        elapsed = time.perf_counter() - t0
        t_sec = data.t.detach().cpu().numpy().astype(np.float64)
        if t_sec.shape[0] != len(t_keys):
            t_sec = np.array(
                [float(data.t[min(i, len(data.t) - 1)].item()) for i in t_keys],
                dtype=np.float64,
            )

        log(f"[OK] Rollout done in {elapsed:.1f}s ({len(t_keys)} steps)")
        return CustomerTrajectory(
            t_sec=t_sec,
            pos=pos,
            vel_mag=vel_all,
            mu_eff_si=mu_all,
            phi=phi_all,
            elapsed_s=elapsed,
            n_steps=len(t_keys),
            meta={
                "wall_ckpt": str(self.wall_ckpt),
                "offwall_ckpt": str(self.offwall_ckpt) if self.offwall_ckpt else None,
                "mat_leg": self.mat_leg,
                "t_final_s": t_end,
                "two_model": self.offwall_ckpt is not None,
                "include_velocity": bool(include_velocity),
            },
        )
