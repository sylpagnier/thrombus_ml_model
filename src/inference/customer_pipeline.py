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

from src.biochem_gnn.compound_deploy import (
    COMPOUND_LEG,
    DEFAULT_COMPOUND_DEPLOY_ENV,
    WALL_LEG,
    load_compound_manifest,
    resolve_growth_ckpt,
    resolve_wall_ckpt as resolve_compound_wall_ckpt,
)
from src.biochem_gnn.mat_growth_simple import apply_mat_growth_leg_env, mat_growth_leg_spec
from src.config import BiochemConfig, PhysicsConfig
from src.core_physics.species_gnn_clot_rollout import (
    load_species_gnn_rollout_bundle,
    prepare_species_gnn_rollout_static,
    rollout_species_gnn_species_series,
)
from src.core_physics.species_viscosity_calibration import resolve_clot_readout_beta
from src.core_physics.t0_device import require_cuda_device
from src.core_physics.t0_mu_physics import rollout_t0_clot_phi
from src.core_physics.t0_rung_config import RUNG2_GAMMA_MODE, t0_rung2_env
from src.inference.corrector_coupling import CorrectorCoupledFlow
from src.inference.species_gnn_deploy_env import load_deploy_manifest, species_gnn_deploy_env
from src.utils.paths import get_project_root

# Wall backbone = WC_v7_clot_phi_mse; customer default stacks compound growth specialist
# (WC_v8_compound_front_h1) via data/reference/mat_compound_deploy.json.
DEFAULT_WALL_CKPT = Path("outputs/biochem/biochem_gnn/locked/species_gnn_best.pth")
DEFAULT_GROWTH_CKPT = Path("outputs/biochem/biochem_gnn/locked/compound_growth_best.pth")
DEFAULT_MAT_LEG = WALL_LEG
DEFAULT_COMPOUND_LEG = COMPOUND_LEG


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
    mask_wall: np.ndarray | None = None
    mask_inlet: np.ndarray | None = None
    mask_outlet: np.ndarray | None = None
    hop_from_wall: np.ndarray | None = None

    def frame(self, index: int) -> dict[str, np.ndarray | float]:
        i = int(max(0, min(index, self.n_steps - 1)))
        return {
            "index": i,
            "t_sec": float(self.t_sec[i]),
            "vel_mag": self.vel_mag[i],
            "mu_eff_si": self.mu_eff_si[i],
            "phi": self.phi[i],
        }

    def interior_mask(self) -> np.ndarray:
        """Nodes that are neither inlet nor outlet (wall + lumen)."""
        n = int(self.pos.shape[0])
        interior = np.ones(n, dtype=bool)
        if self.mask_inlet is not None:
            interior &= ~np.asarray(self.mask_inlet, dtype=bool).reshape(-1)
        if self.mask_outlet is not None:
            interior &= ~np.asarray(self.mask_outlet, dtype=bool).reshape(-1)
        return interior

    def has_velocity_at(self, index: int) -> bool:
        idxs = (self.meta or {}).get("velocity_indices")
        if idxs is None:
            return bool((self.meta or {}).get("include_velocity", False))
        return int(index) in {int(i) for i in idxs}


def _abs(path: Path | str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = get_project_root() / p
    return p


def _compound_mode_requested() -> bool:
    """Customer default uses compound deploy unless CUSTOMER_COMPOUND=0."""
    flag = os.environ.get("CUSTOMER_COMPOUND", "1").strip().lower()
    return flag not in ("0", "false", "no", "off", "wall_only", "wall-only")


def _compound_deploy_overrides() -> dict[str, str]:
    man = load_compound_manifest()
    deploy = dict(DEFAULT_COMPOUND_DEPLOY_ENV)
    deploy.update(man.get("deploy") or {})
    return {
        str(k): str(v)
        for k, v in deploy.items()
        if k != "mat_leg" and (str(k).startswith("SPECIES_") or str(k).startswith("BIOCHEM_"))
    }


def _resolve_wall_ckpt(explicit: Path | str | None = None) -> Path:
    raw = str(explicit or os.environ.get("CUSTOMER_WALL_CKPT") or "").strip()
    if raw:
        p = _abs(raw)
    elif _compound_mode_requested():
        p = resolve_compound_wall_ckpt()
    else:
        p = _abs(DEFAULT_WALL_CKPT)
    if not p.is_file():
        alt = _abs(DEFAULT_WALL_CKPT)
        if alt.is_file():
            return alt
    return p


def _resolve_offwall_ckpt(explicit: Path | str | None = None) -> Path | None:
    raw = str(explicit or os.environ.get("CUSTOMER_OFFWALL_CKPT") or "").strip()
    if raw.lower() in ("none", "0", "off", "false", "wall_only", "wall-only"):
        return None
    if raw:
        p = _abs(raw)
        return p if p.is_file() else None
    if not _compound_mode_requested():
        return None
    growth = resolve_growth_ckpt()
    if growth.is_file():
        return growth
    fallback = _abs(DEFAULT_GROWTH_CKPT)
    return fallback if fallback.is_file() else None


@contextmanager
def _customer_deploy_env(
    *,
    wall_ckpt: Path,
    offwall_ckpt: Path | None,
    mat_leg: str = DEFAULT_MAT_LEG,
    extra_env: dict[str, str] | None = None,
) -> Iterator[dict[str, str]]:
    """Apply deploy + mat-growth env, then restore.

    ``extra_env`` is applied last so research-sweep stack knobs can override
    mat-leg defaults (e.g. corrector / dynamic occlusion ablations).
    """
    manifest = load_deploy_manifest()
    overrides: dict[str, str] = {
        "T0_R4_FLOW_SOURCE": "kinematics",
        "SPECIES_GNN_CLOUT_CKPT": str(wall_ckpt).replace("\\", "/"),
        "SPECIES_CONTINUOUS_CKPT": str(wall_ckpt).replace("\\", "/"),
        "T0_R4_SPECIES_GNN_CKPT": str(wall_ckpt).replace("\\", "/"),
        "BIOCHEM_CORRECTOR_COUPLING": "1",
    }
    # Mat-growth recipe (wall + off-wall clot physics) -- typed first, residual env last.
    try:
        from dataclasses import replace as _replace

        from src.architecture.pushforward_config import PushforwardConfig
        from src.architecture.runtime_config import BiochemRuntimeConfig
        from src.biochem_gnn.config import _bind_typed_configs

        spec = mat_growth_leg_spec(mat_leg)
        overrides.update({k: str(v) for k, v in spec.env_overrides.items()})
        pf = _replace(PushforwardConfig(), **dict(spec.config_kwargs)) if spec.config_kwargs else PushforwardConfig()
        rt = BiochemRuntimeConfig.from_kwargs(spec.runtime_kwargs or {})
        if offwall_ckpt is not None:
            rt = rt.with_overrides(
                two_model_mode=True,
                offwall_model_ckpt=str(offwall_ckpt).replace("\\", "/"),
            )
        else:
            rt = rt.with_overrides(two_model_mode=False)
        rt = rt.with_overrides(corrector_coupling=True, t0_flow_source="kinematics")
        _bind_typed_configs(pf, rt)
    except Exception:
        try:
            spec = mat_growth_leg_spec(mat_leg)
            overrides.update({k: str(v) for k, v in spec.env_overrides.items()})
        except Exception:
            pass

    if offwall_ckpt is not None:
        overrides["SPECIES_TWO_MODEL_MODE"] = "1"
        overrides["SPECIES_OFFWALL_MODEL_CKPT"] = str(offwall_ckpt).replace("\\", "/")
        overrides.update(_compound_deploy_overrides())
    else:
        overrides.setdefault("SPECIES_TWO_MODEL_MODE", "0")

    if extra_env:
        overrides.update({str(k): str(v) for k, v in extra_env.items()})

    keys = set(overrides) | {
        "SPECIES_GNN_CLOUT_CKPT",
        "SPECIES_CONTINUOUS_CKPT",
        "T0_R4_SPECIES_GNN_CKPT",
        "SPECIES_OFFWALL_MODEL_CKPT",
        "SPECIES_TWO_MODEL_MODE",
        "SPECIES_TWO_MODEL_ROUTE",
        "SPECIES_TWO_MODEL_FRONTIER_HOPS",
        "SPECIES_CONTINUOUS_VEL_DECAY",
        "SPECIES_CONTINUOUS_VEL_DECAY_WALL_ONLY",
        "BIOCHEM_CORRECTOR_COUPLING",
        "SPECIES_DYNAMIC_OCCLUSION",
        "BIOCHEM_ROLLOUT_DYNAMIC_OCCLUSION",
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
                "Promote WC_v7 wall or compound deploy first "
                "(scripts/promote_compound_deploy.py)."
            )
        if self.offwall_ckpt is not None and not self.offwall_ckpt.is_file():
            raise FileNotFoundError(
                f"Compound growth checkpoint missing: {self.offwall_ckpt}. "
                "Run scripts/promote_compound_deploy.py or set CUSTOMER_COMPOUND=0."
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
        extra_env: dict[str, str] | None = None,
    ) -> CustomerTrajectory:
        """Species + clot-phi trajectory; optionally couple local corrector for velocity.

        ``extra_env`` overrides mat-leg / default deploy knobs for this run only
        (research stack ablations).
        """
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
            extra_env=extra_env,
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
            # Explicit override only: the promoted on-disk beta keeps its historical role
            # (Mat boost inside the species rollout) and must not additionally re-grade the
            # clot phi readout. See resolve_clot_readout_beta.
            gel_beta = resolve_clot_readout_beta()
            nuc_hops = 2
            try:
                from src.architecture.runtime_config import get_active_runtime

                rt = get_active_runtime()
                if rt is not None:
                    nuc_hops = int(rt.rollout.nucleation_hops)
                else:
                    nuc_hops = int(os.environ.get("CLOT_V2_NUCLEATION_HOPS", "2"))
            except Exception:
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
        # Customer UI only needs first/last velocity (Clot+Velocity bookends + light Scientific).
        velocity_indices: list[int] = []
        if include_velocity and t_keys:
            velocity_indices = [int(t_keys[0])]
            if len(t_keys) > 1:
                velocity_indices.append(int(t_keys[-1]))
            velocity_indices = sorted(set(velocity_indices))

        if include_velocity:
            log(
                f"[i] Coupling local kinematic corrector at "
                f"{len(velocity_indices)} bookend step(s)..."
            )
            # Remesh / parametric edits change node count; never reuse prior mesh base flow.
            self._flow_provider.invalidate_base_cache()
            for ti in t_keys:
                mu_all[ti] = traj[ti]["mu"].detach().cpu().numpy()
                phi_all[ti] = traj[ti]["phi"].detach().cpu().numpy()
                if int(ti) in velocity_indices:
                    mu_eff_si = traj[ti]["mu"].to(self.device)
                    u, v = self._flow_provider.couple(data, mu_eff_si, publish=False)
                    vel_all[ti] = torch.sqrt(u**2 + v**2).detach().cpu().numpy()
                else:
                    vel_all[ti] = np.zeros_like(phi_all[ti], dtype=np.float32)
        else:
            log("[i] Skipping velocity corrector (clot-only mode)...")
            for ti in t_keys:
                mu_all[ti] = traj[ti]["mu"].detach().cpu().numpy()
                phi_all[ti] = traj[ti]["phi"].detach().cpu().numpy()
                vel_all[ti] = np.zeros_like(phi_all[ti], dtype=np.float32)

        def _mask_np(name: str) -> np.ndarray | None:
            m = getattr(data, name, None)
            if m is None:
                return None
            return m.reshape(-1).bool().detach().cpu().numpy()

        hop_from_wall: np.ndarray | None = None
        try:
            from src.core_physics.species_pushforward_continuous import compute_hop_distances

            wall_t = getattr(data, "mask_wall", None)
            ei = getattr(data, "edge_index", None)
            if wall_t is not None and ei is not None:
                hop_from_wall = (
                    compute_hop_distances(ei, wall_t.reshape(-1).bool(), int(pos.shape[0]))
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.int32)
                )
        except Exception as exc:
            log(f"[WARN] wall-hop distances unavailable: {exc}")

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
                "compound_leg": DEFAULT_COMPOUND_LEG if self.offwall_ckpt is not None else None,
                "t_final_s": t_end,
                "two_model": self.offwall_ckpt is not None,
                "include_velocity": bool(include_velocity),
                "velocity_indices": velocity_indices,
                "velocity_mode": "bookends" if include_velocity else "none",
                "extra_env": dict(extra_env) if extra_env else {},
            },
            mask_wall=_mask_np("mask_wall"),
            mask_inlet=_mask_np("mask_inlet"),
            mask_outlet=_mask_np("mask_outlet"),
            hop_from_wall=hop_from_wall,
        )
