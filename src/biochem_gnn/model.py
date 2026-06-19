"""Hybrid biochem deploy pipeline (canonical stack: ``biochem_deploy``).

Composable SciML stack (not one nn.Module):
  PMGP-DEQ kine -> GraphSAGE species pushforward -> gelation_beta -> clot_trigger_physics.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import torch

from src.biochem_gnn.config import (
    DEFAULT_KINE_CKPT,
    beta_ckpt_path,
    load_manifest,
    rel_path,
    species_ckpt_for_anchor,
)
from src.config import BiochemConfig, PhysicsConfig
from src.core_physics.clot_phi_rollout import KinematicsUvProvider
from src.core_physics.clot_phi_simple import sdf_nd_from_data
from src.core_physics.coupled_shear_gnn import (
    LocalKinematicCorrector,
    assemble_local_corrector_features,
    load_local_corrector,
)
from src.core_physics.clot_coupled_rollout import (
    reset_coupled_uv_cache,
    set_coupled_uv_cache,
)
from src.core_physics.clot_nucleation_mask import (
    project_phi_with_nucleation,
    resolve_nucleation_eligibility,
)
from src.core_physics.species_gnn_clot_rollout import (
    SpeciesGnnRolloutBundle,
    SpeciesGnnRolloutStatic,
    load_species_gnn_rollout_bundle,
    prepare_species_gnn_rollout_static,
    rollout_species_gnn_phi_trajectory,
    rollout_species_gnn_species_series,
)
from src.core_physics.species_deploy_rollout import (
    alloc_species_y_series,
    deploy_fimat_log_init,
    pin_species_block,
)
from src.core_physics.species_pushforward_continuous import (
    continuous_vel_decay_enabled,
    graph_last_time_index,
    model_vel_decay_alphas,
    predict_continuous_step_delta,
    pushforward_log_state_step,
)
from src.core_physics.species_viscosity_calibration import (
    apply_mat_beta_to_species_series,
    load_viscosity_calibration,
    resolve_deploy_gelation_beta,
)
from src.core_physics.t0_mu_physics import predict_clot_phi_at_time
from src.core_physics.t0_rung_config import RUNG2_GAMMA_MODE, t0_rung2_env
from src.training.biochem_species_scope import FI_CHANNEL, MAT_CHANNEL


class FlowMode(str, Enum):
    """How [u, v] are resolved during rollout."""

    GT = "gt"
    FROZEN_KINE = "frozen_kine"
    COUPLED = "coupled"


N_MODELED_SPECIES = 2
MODELED_SPECIES_NAMES = ("FI", "Mat")


def band_speed_from_uv(
    u_nd: torch.Tensor,
    v_nd: torch.Tensor,
    node_idx: torch.Tensor,
) -> torch.Tensor:
    from src.core_physics.species_deploy_rollout import band_speed_from_uv as _bs

    return _bs(u_nd, v_nd, node_idx)


def embed_fimat_into_species12(
    species12: torch.Tensor,
    log_state: torch.Tensor,
    node_idx: torch.Tensor,
) -> torch.Tensor:
    """Write band FI/Mat log-ND into the 12-ch species block."""
    out = species12.clone()
    st = log_state.reshape(-1, 2)
    out[node_idx, FI_CHANNEL] = st[:, 0]
    out[node_idx, MAT_CHANNEL] = st[:, 1]
    return out.clamp(min=0.0)


@dataclass
class BiochemDeployConfig:
    """Runtime bundle paths (resolved from manifest or explicit)."""

    species_ckpt: str | Path = ""
    viscosity_beta: str | Path = ""
    kinematics_ckpt: str | Path = ""
    gelation_beta: float | None = None
    flow_mode: FlowMode = FlowMode.FROZEN_KINE
    gamma_mode: str = RUNG2_GAMMA_MODE
    nucleation: bool = True
    nucleation_hops: int = 1
    pin_other_species: str = "rest"
    # Optional local kinematic corrector (velocity diversion around micro-clots).
    local_corrector_ckpt: str | Path | None = None
    local_corrector_hops: int = 4
    # SI delta-mu above fluid baseline that flags a node as clot for the corrector.
    local_corrector_mu_thresh_si: float = 1e-4


@dataclass
class BiochemDeployRollout:
    """Outputs from a full-timeline deploy rollout."""

    species_series: torch.Tensor
    phi_by_time: dict[int, torch.Tensor]
    mu_by_time: dict[int, torch.Tensor]
    flow_mode: FlowMode
    meta: dict[str, Any] = field(default_factory=dict)


class BiochemDeployStack:
    """Deploy biochem pipeline (not a single nn.Module).

    Components (SciML-accurate ids):
      - pmgp_deq_kine: frozen PMGP-DEQ flow checkpoint (``GINO_DEQ`` class)
      - species_graphsage: wall-band GraphSAGE pushforward (learned)
      - gelation_beta: global Mat scale (learned scalar)
      - clot_trigger_physics: Carreau + gelation mu + nucleation phi (equations)
      - flow_coupling: optional mu -> PMGP-DEQ refresh via ``KinematicsUvProvider``
    """

    def __init__(
        self,
        bundle: SpeciesGnnRolloutBundle,
        *,
        device: torch.device,
        cfg: BiochemDeployConfig | None = None,
        manifest: dict[str, Any] | None = None,
    ) -> None:
        self.bundle = bundle
        self.device = device
        self.manifest = manifest or {}
        self.cfg = cfg or BiochemDeployConfig()
        self.phys = PhysicsConfig(phase="biochem")
        self.bio = BiochemConfig(phase="biochem")
        self._kine: KinematicsUvProvider | None = None
        self._local_corrector: LocalKinematicCorrector | None = None

    @classmethod
    def from_manifest(
        cls,
        manifest: dict[str, Any] | None = None,
        *,
        anchor: str | None = None,
        device: torch.device | None = None,
        flow_mode: FlowMode = FlowMode.FROZEN_KINE,
    ) -> BiochemDeployStack:
        dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        m = manifest or load_manifest()
        ckpt = species_ckpt_for_anchor(anchor or "", m, prefer_loao=True)
        bundle = load_species_gnn_rollout_bundle(ckpt, device=dev)
        if bundle is None:
            raise FileNotFoundError(f"species GNN ckpt not found: {ckpt}")
        beta_p = m.get("viscosity_beta") or rel_path(beta_ckpt_path())
        kine_p = m.get("kinematics_ckpt") or rel_path(DEFAULT_KINE_CKPT)
        corrector_p = m.get("local_corrector_ckpt")
        cfg = BiochemDeployConfig(
            species_ckpt=str(ckpt),
            viscosity_beta=str(beta_p),
            kinematics_ckpt=str(kine_p),
            flow_mode=flow_mode,
            local_corrector_ckpt=str(corrector_p) if corrector_p else None,
        )
        return cls(bundle, device=dev, cfg=cfg, manifest=m)

    @classmethod
    def from_checkpoints(
        cls,
        species_ckpt: str | Path,
        *,
        viscosity_beta: str | Path | None = None,
        kinematics_ckpt: str | Path | None = None,
        device: torch.device | None = None,
        flow_mode: FlowMode = FlowMode.FROZEN_KINE,
    ) -> BiochemDeployStack:
        dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = Path(species_ckpt)
        bundle = load_species_gnn_rollout_bundle(ckpt, device=dev)
        if bundle is None:
            raise FileNotFoundError(f"species GNN ckpt not found: {ckpt}")
        cfg = BiochemDeployConfig(
            species_ckpt=str(ckpt),
            viscosity_beta=str(viscosity_beta or beta_ckpt_path()),
            kinematics_ckpt=str(kinematics_ckpt or DEFAULT_KINE_CKPT),
            flow_mode=flow_mode,
        )
        return cls(bundle, device=dev, cfg=cfg)

    def _kine_provider(self) -> KinematicsUvProvider:
        if self._kine is None:
            prev = os.environ.get("CLOT_PHI_KINE_CKPT")
            os.environ["CLOT_PHI_KINE_CKPT"] = str(self.cfg.kinematics_ckpt)
            self._kine = KinematicsUvProvider(self.device)
            if prev is None:
                os.environ.pop("CLOT_PHI_KINE_CKPT", None)
            else:
                os.environ["CLOT_PHI_KINE_CKPT"] = prev
        return self._kine

    def _gelation_beta(self) -> float | None:
        if self.cfg.gelation_beta is not None:
            return float(self.cfg.gelation_beta)
        return resolve_deploy_gelation_beta(self.device)

    def set_local_corrector(self, model: LocalKinematicCorrector) -> None:
        """Attach a trained local kinematic corrector for coupled rollout."""
        self._local_corrector = model.to(self.device).eval()

    def _local_corrector_model(self) -> LocalKinematicCorrector | None:
        if self._local_corrector is None and self.cfg.local_corrector_ckpt:
            p = Path(self.cfg.local_corrector_ckpt)
            if p.is_file():
                self._local_corrector = load_local_corrector(p, device=self.device)
        return self._local_corrector

    @torch.no_grad()
    def _apply_local_corrector(
        self,
        corrector: LocalKinematicCorrector,
        data,
        mu_pred_si: torch.Tensor,
        u0: torch.Tensor,
        v0: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Patch the frozen base flow with a local diversion around clot nodes.

        Operates on the **full** mesh graph (``data.edge_index``): flow diverts
        through the lumen, not just within the wall band. Returns full-graph
        ``(u_next, v_next)`` so downstream band lookups stay valid.
        """
        from torch_geometric.utils import k_hop_subgraph

        u_next = u0.clone()
        v_next = v0.clone()

        n = int(data.num_nodes)
        fluid_mu_si = float(getattr(self.phys, "mu_inf", 0.0035))
        # SI delta-mu flags clot nodes; the corrector feature is the ND delta-mu so
        # the synthetic-patch trainer and this deploy path share one viscosity scale.
        delta_mu_si = mu_pred_si.reshape(-1) - fluid_mu_si
        clot_nodes = torch.where(delta_mu_si > float(self.cfg.local_corrector_mu_thresh_si))[0]
        if clot_nodes.numel() == 0:
            return u_next, v_next

        subset, sub_edge_index, _, _ = k_hop_subgraph(
            clot_nodes,
            num_hops=int(self.cfg.local_corrector_hops),
            edge_index=data.edge_index,
            relabel_nodes=True,
            num_nodes=n,
        )

        pos_nd = data.x[:, 0:2].to(device=self.device, dtype=torch.float32)
        sdf_nd = sdf_nd_from_data(data, self.device, n).reshape(-1)
        # mu_pred_si and mu_inf share mu_viscosity_nd_scale, so delta_mu_nd is exact.
        delta_mu_nd = self.phys.viscosity_si_to_nd(delta_mu_si)
        x_sub = assemble_local_corrector_features(
            pos_nd, sdf_nd, u0, v0, delta_mu_nd, clot_nodes, subset
        )

        delta_uv = corrector(x_sub, sub_edge_index.to(self.device))
        u_next[subset] = u_next[subset] + delta_uv[:, 0]
        v_next[subset] = v_next[subset] + delta_uv[:, 1]
        return u_next, v_next

    def _flow_source_str(self) -> str:
        if self.cfg.flow_mode == FlowMode.GT:
            return "gt"
        return "kinematics"

    @torch.no_grad()
    def rollout(
        self,
        data,
        static: SpeciesGnnRolloutStatic | None = None,
    ) -> BiochemDeployRollout:
        """Full deploy rollout. Uses coupled interleaved loop when ``flow_mode=COUPLED``."""
        if self.cfg.flow_mode == FlowMode.COUPLED:
            return self._rollout_coupled(data, static)
        return self._rollout_decoupled(data, static)

    @torch.no_grad()
    def _rollout_decoupled(
        self,
        data,
        static: SpeciesGnnRolloutStatic | None = None,
    ) -> BiochemDeployRollout:
        stat = static or prepare_species_gnn_rollout_static(data, device=self.device)
        pin = self.cfg.pin_other_species
        species = rollout_species_gnn_species_series(
            data,
            self.bundle,
            stat,
            phys_cfg=self.phys,
            bio_cfg=self.bio,
            device=self.device,
            pin_other_species=pin,
        )
        flow = self._flow_source_str()
        with t0_rung2_env():
            phi_traj = rollout_species_gnn_phi_trajectory(
                data,
                self.bundle,
                stat,
                phys_cfg=self.phys,
                bio_cfg=self.bio,
                device=self.device,
                flow_source=flow,
            )
        mu_by_t: dict[int, torch.Tensor] = {}
        with t0_rung2_env():
            for t, phi in phi_traj.items():
                _, step = predict_clot_phi_at_time(
                    data,
                    int(t),
                    self.phys,
                    self.bio,
                    self.device,
                    gamma_mode=self.cfg.gamma_mode,
                    flow_source=flow,
                    pred_species_series=species,
                    gelation_beta=self._gelation_beta(),
                )
                mu_by_t[int(t)] = step.mu_pred_si
        return BiochemDeployRollout(
            species_series=species,
            phi_by_time=phi_traj,
            mu_by_time=mu_by_t,
            flow_mode=self.cfg.flow_mode,
            meta={"decoupled": True, "pin_other_species": pin},
        )

    @torch.no_grad()
    def _rollout_coupled(
        self,
        data,
        static: SpeciesGnnRolloutStatic | None = None,
    ) -> BiochemDeployRollout:
        """Interleaved: species step -> gelation mu -> refresh GINO-DEQ flow -> repeat."""
        if self.bundle.kind != "continuous" or self.bundle.continuous is None:
            raise NotImplementedError(
                "coupled flow requires continuous (biochem_deploy) species ckpt"
            )

        stat = static or prepare_species_gnn_rollout_static(data, device=self.device)
        model = self.bundle.continuous.model
        n_steps = int(data.y.shape[0])
        gel_beta = self._gelation_beta()
        vel_alphas = model_vel_decay_alphas(model) if continuous_vel_decay_enabled() else None

        prev_vel = os.environ.get("CLOT_TEMPORAL_VEL_SOURCE")
        os.environ["CLOT_TEMPORAL_VEL_SOURCE"] = "coupled"
        reset_coupled_uv_cache()

        from src.core_physics.clot_temporal_growth_rules import _resolve_uv_for_temporal_risk

        os.environ["CLOT_TEMPORAL_VEL_SOURCE"] = "kinematics"
        u0, v0 = _resolve_uv_for_temporal_risk(data, 0, self.device)
        set_coupled_uv_cache(data, u0, v0)
        os.environ["CLOT_TEMPORAL_VEL_SOURCE"] = "coupled"

        species_out = alloc_species_y_series(data, self.device)
        log_state = deploy_fimat_log_init(data, self.device, stat.node_idx)
        phi_by_t: dict[int, torch.Tensor] = {}
        mu_by_t: dict[int, torch.Tensor] = {}
        phi_prev: torch.Tensor | None = None
        commits_prev: torch.Tensor | None = None
        kine = self._kine_provider()
        corrector = self._local_corrector_model()

        try:
            with t0_rung2_env():
                for t in range(n_steps):
                    sp = pin_species_block(
                        data, t, self.device,
                        pin_other=self.cfg.pin_other_species,  # type: ignore[arg-type]
                    )
                    sp = embed_fimat_into_species12(sp, log_state, stat.node_idx)
                    species_out[t, :, 4:16] = sp

                    phi_raw, step = predict_clot_phi_at_time(
                        data,
                        t,
                        self.phys,
                        self.bio,
                        self.device,
                        gamma_mode=self.cfg.gamma_mode,
                        flow_source="kinematics",
                        pred_species_series=species_out,
                        gelation_beta=gel_beta,
                    )
                    phi = phi_raw
                    if self.cfg.nucleation:
                        elig = resolve_nucleation_eligibility(
                            data,
                            t,
                            self.device,
                            self.phys,
                            self.bio,
                            commits_prev=commits_prev,
                            growth_seed="pred",
                            nucleation_hops=self.cfg.nucleation_hops,
                        )
                        phi = project_phi_with_nucleation(phi_raw, phi_prev, elig)
                        commits_prev = (phi.reshape(-1) >= 0.5).bool()
                    phi_prev = phi.detach()
                    phi_by_t[t] = phi
                    mu_by_t[t] = step.mu_pred_si

                    if corrector is not None:
                        # Frozen base flow + local diversion patch around clot nodes
                        # (cheap, anisotropic) instead of a full global flow re-solve.
                        u_next, v_next = self._apply_local_corrector(
                            corrector, data, step.mu_pred_si, u0, v0
                        )
                    else:
                        u_next, v_next = kine.uv_nd_from_mu_si(data, step.mu_pred_si)
                    set_coupled_uv_cache(data, u_next, v_next)

                    if t >= n_steps - 1:
                        break
                    pred_delta = predict_continuous_step_delta(
                        model, stat.base_feats, stat.edge_index, log_state, training=False
                    )
                    spd = (
                        band_speed_from_uv(u_next, v_next, stat.node_idx)
                        if vel_alphas is not None
                        else None
                    )
                    log_state = pushforward_log_state_step(
                        log_state,
                        pred_delta,
                        straight_through=False,
                        wall_speed=spd,
                        vel_decay_alphas=vel_alphas,
                    )
        finally:
            if prev_vel is None:
                os.environ.pop("CLOT_TEMPORAL_VEL_SOURCE", None)
            else:
                os.environ["CLOT_TEMPORAL_VEL_SOURCE"] = prev_vel
            reset_coupled_uv_cache()

        if gel_beta is not None:
            beta_path = str(self.cfg.viscosity_beta)
            t_boost = graph_last_time_index(n_steps)
            if Path(beta_path).is_file():
                _, calib = load_viscosity_calibration(beta_path, device=self.device)
                t_boost = int(calib.time_index)
            species_out = apply_mat_beta_to_species_series(
                species_out,
                gel_beta,
                self.bio,
                time_index=min(t_boost, n_steps - 1),
            )

        return BiochemDeployRollout(
            species_series=species_out,
            phi_by_time=phi_by_t,
            mu_by_time=mu_by_t,
            flow_mode=FlowMode.COUPLED,
            meta={
                "interleaved": True,
                "gelation_beta": gel_beta,
                "local_corrector": corrector is not None,
            },
        )


# Backward-compatible aliases (pre-2026-06 rename)
BiochemGNN = BiochemDeployStack
BiochemGNNConfig = BiochemDeployConfig
BiochemGNNRollout = BiochemDeployRollout
