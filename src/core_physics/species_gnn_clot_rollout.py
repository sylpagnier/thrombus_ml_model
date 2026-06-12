"""Species GNN -> full-timeline species -> T0 clot phi rollout.

Builds a ``(T, N, 16)`` y-shaped species series from Phase 2 (binary pushforward)
or Phase 2.5 (continuous log-delta), pins non-modeled channels to GT, then runs
``rollout_t0_clot_phi`` under Rung2 gamma env.

See ``scripts/viz_species_gnn_clot_ladder.py``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch

from src.config import BiochemConfig, PhysicsConfig
from src.core_physics.clot_phi_simple import sdf_nd_from_data
from src.core_physics.species_pushforward_continuous import (
    SpeciesContinuousBundle,
    SpeciesDualHeadContinuousGNN,
    band_speed_at_time,
    continuous_max_sat_log,
    continuous_vel_decay_enabled,
    predict_continuous_step_delta,
    fi_mat_log_targets,
    load_continuous_bundle,
    log_series_on_band,
    model_vel_decay_alphas,
    normalize_log_state,
    pushforward_log_state_step,
)
from src.core_physics.species_pushforward_gnn import (
    STATE_DIM,
    SpeciesPushforwardBundle,
    build_band_base_features,
    load_pushforward_bundle,
    pushforward_state_step,
)
from src.core_physics.species_snapshot_gnn import (
    build_snapshot_features,
    fi_mat_active_labels,
    induced_subgraph,
    snapshot_wall_hops,
    wall_band_mask,
)
from src.core_physics.t0_rung4_ladder import resting_species_log_nd
from src.core_physics.t0_rung_config import RUNG2_GAMMA_MODE, t0_rung2_env
from src.training.biochem_species_scope import FI_CHANNEL, MAT_CHANNEL
from src.utils.kinematics_inference import (
    load_kinematics_predictor,
    predict_kinematics_latent,
    resolve_kinematics_checkpoint,
)
from src.utils.paths import get_project_root

RolloutKind = Literal["continuous", "binary"]


def species_gnn_rollout_ckpt() -> Path:
    raw = (
        os.environ.get("T0_R4_SPECIES_GNN_CKPT")
        or os.environ.get("SPECIES_GNN_CLOUT_CKPT")
        or os.environ.get("SPECIES_CONTINUOUS_CKPT")
        or os.environ.get("SPECIES_PUSHFORWARD_CKPT")
        or "outputs/biochem/species_snapshot_s34/best.pth"
    ).strip()
    p = Path(raw)
    if not p.is_absolute():
        p = get_project_root() / p
    return p


@dataclass(frozen=True)
class SpeciesGnnRolloutStatic:
    base_feats: torch.Tensor
    edge_index: torch.Tensor
    node_idx: torch.Tensor
    band: torch.Tensor
    device: torch.device


@dataclass(frozen=True)
class SpeciesGnnRolloutBundle:
    kind: RolloutKind
    label: str
    continuous: SpeciesContinuousBundle | None = None
    binary: SpeciesPushforwardBundle | None = None

    @property
    def device(self) -> torch.device:
        if self.continuous is not None:
            return self.continuous.device
        if self.binary is not None:
            return self.binary.device
        raise RuntimeError("empty SpeciesGnnRolloutBundle")


def _bundle_label_from_path(path: Path, phase: str) -> str:
    if "s35" in phase:
        return "s35"
    if "s34" in phase:
        return "s34"
    if "s33" in phase:
        return "s33"
    if "s32" in phase:
        return "s32"
    if "s31" in phase:
        return "s31"
    if "s30" in phase:
        return "s30"
    if "s26" in phase:
        return "s26"
    if "s25" in phase or "continuous" in phase:
        return "s25"
    if "s2" in phase or "pushforward" in phase:
        return "s2"
    stem = path.parent.name
    if stem.startswith("species_snapshot_"):
        return stem.replace("species_snapshot_", "")
    return stem or "gnn"


def load_species_gnn_rollout_bundle(
    ckpt_path: Path | str | None = None,
    *,
    device: torch.device | None = None,
    quiet: bool = False,
) -> SpeciesGnnRolloutBundle | None:
    path = Path(ckpt_path) if ckpt_path is not None else species_gnn_rollout_ckpt()
    if not path.is_file():
        if not quiet:
            print(f"[WARN] species GNN rollout checkpoint missing: {path}")
        return None
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(path, map_location=dev, weights_only=False)
    meta = dict(payload.get("meta") or {})
    phase = str(meta.get("phase") or payload.get("phase") or "").lower()
    if bool(meta.get("kin_per_vessel_norm")):
        os.environ["SPECIES_KIN_PER_VESSEL_NORM"] = "1"
    if bool(meta.get("dual_head")):
        os.environ["SPECIES_CONTINUOUS_DUAL_HEAD"] = "1"
    if bool(meta.get("vel_decay")):
        os.environ["SPECIES_CONTINUOUS_VEL_DECAY"] = "1"
    if bool(meta.get("saturation_gate")):
        os.environ["SPECIES_CONTINUOUS_SATURATION_GATE"] = "1"
    if bool(meta.get("temporal_gate")):
        os.environ["SPECIES_CONTINUOUS_TEMPORAL_GATE"] = "1"
    label = _bundle_label_from_path(path, phase)
    if "continuous" in phase or "dual_head" in phase or "long_horizon" in phase or "saturation" in phase or "temporal" in phase or phase in (
        "s25_continuous",
        "s26_continuous",
        "s31_dual_head",
        "s32_long_horizon",
        "s33_saturation_gate",
        "s34_temporal_gate",
    ):
        cont = load_continuous_bundle(path, device=dev, quiet=True)
        if cont is None:
            return None
        return SpeciesGnnRolloutBundle(kind="continuous", label=label, continuous=cont)
    binary = load_pushforward_bundle(path, device=dev, quiet=True)
    if binary is None:
        return None
    return SpeciesGnnRolloutBundle(kind="binary", label=label, binary=binary)


@torch.no_grad()
def prepare_species_gnn_rollout_static(
    data,
    *,
    device: torch.device,
    wall_hops: int | None = None,
) -> SpeciesGnnRolloutStatic:
    hops = int(wall_hops if wall_hops is not None else snapshot_wall_hops())
    kine = load_kinematics_predictor(resolve_kinematics_checkpoint(), device)
    stat = build_band_base_features(data, kine, device, wall_hops=hops)
    band = wall_band_mask(data, device, wall_hops=hops).reshape(-1).bool()
    return SpeciesGnnRolloutStatic(
        base_feats=stat["base_feats"],
        edge_index=stat["edge_index"],
        node_idx=stat["node_idx"],
        band=band,
        device=device,
    )


def _write_fimat_log_to_species(
    species: torch.Tensor,
    log_state: torch.Tensor,
    node_idx: torch.Tensor,
) -> torch.Tensor:
    out = species.clone()
    idx = node_idx.reshape(-1)
    st = log_state.reshape(-1, STATE_DIM)
    out[idx, FI_CHANNEL] = st[:, 0]
    out[idx, MAT_CHANNEL] = st[:, 1]
    return out.clamp(min=0.0)


def _binary_state_to_log(state: torch.Tensor) -> torch.Tensor:
    fi_sat, mat_sat = continuous_max_sat_log()
    st = state.reshape(-1, STATE_DIM)
    out = torch.zeros_like(st)
    out[:, 0] = torch.where(st[:, 0] > 0.5, torch.tensor(fi_sat, device=st.device, dtype=st.dtype), out[:, 0])
    out[:, 1] = torch.where(st[:, 1] > 0.5, torch.tensor(mat_sat, device=st.device, dtype=st.dtype), out[:, 1])
    return out


@torch.no_grad()
def rollout_species_gnn_species_series(
    data,
    bundle: SpeciesGnnRolloutBundle,
    static: SpeciesGnnRolloutStatic | None = None,
    *,
    phys_cfg: PhysicsConfig | None = None,
    bio_cfg: BiochemConfig | None = None,
    device: torch.device | None = None,
    pin_other_species: str = "gt",
) -> torch.Tensor:
    """Full-timeline species series ``(T, N, 16)`` with FI/Mat from GNN rollout."""
    phys = phys_cfg or PhysicsConfig(phase="biochem")
    bio = bio_cfg or BiochemConfig(phase="biochem")
    dev = device or bundle.device
    stat = static or prepare_species_gnn_rollout_static(data, device=dev)
    n_steps = int(data.y.shape[0])
    out = data.y.clone().to(device=dev)
    rest = resting_species_log_nd(data, dev)

    if bundle.kind == "continuous":
        assert bundle.continuous is not None
        model = bundle.continuous.model
        log_state = fi_mat_log_targets(data, 0, dev)[stat.node_idx]
        vel_alphas = model_vel_decay_alphas(model) if continuous_vel_decay_enabled() else None
        for t in range(n_steps):
            if pin_other_species == "gt":
                sp = data.y[t, :, 4:16].to(device=dev, dtype=torch.float32).clone()
            else:
                sp = rest.clone()
            sp = _write_fimat_log_to_species(sp, log_state, stat.node_idx)
            out[t, :, 4:16] = sp
            if t >= n_steps - 1:
                break
            pred_delta = predict_continuous_step_delta(
                model, stat.base_feats, stat.edge_index, log_state, training=False
            )
            spd = (
                band_speed_at_time(data, t + 1, dev, stat.node_idx)
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
        if os.environ.get("SPECIES_VISCOSITY_CALIB", "").strip().lower() in ("1", "true", "yes", "on"):
            from src.core_physics.species_viscosity_calibration import (
                apply_mat_beta_to_species_series,
                load_viscosity_calibration,
                viscosity_calibration_dir,
            )

            cal_path = os.environ.get("SPECIES_VISCOSITY_CALIB_PATH") or str(
                viscosity_calibration_dir() / "beta.pth"
            )
            if Path(cal_path).is_file():
                cal, calib_bundle = load_viscosity_calibration(cal_path, device=dev)
                t_boost = int(calib_bundle.time_index)
                out = apply_mat_beta_to_species_series(
                    out, cal.beta, bio, time_index=min(t_boost, int(out.shape[0]) - 1)
                )
        return out

    assert bundle.binary is not None
    model = bundle.binary.model
    state = fi_mat_active_labels(fi_mat_log_targets(data, 0, dev)[stat.node_idx])
    for t in range(n_steps):
        if pin_other_species == "gt":
            sp = data.y[t, :, 4:16].to(device=dev, dtype=torch.float32).clone()
        else:
            sp = rest.clone()
        log_state = _binary_state_to_log(state)
        sp = _write_fimat_log_to_species(sp, log_state, stat.node_idx)
        out[t, :, 4:16] = sp
        if t >= n_steps - 1:
            break
        feats = torch.cat([stat.base_feats, state], dim=-1)
        logits = model(feats, stat.edge_index)
        state = pushforward_state_step(state, logits, straight_through=False)
    return out


def _resolve_flow_source(flow_source: str | None) -> str:
    raw = (flow_source or os.environ.get("T0_R4_FLOW_SOURCE") or "gt").strip().lower()
    if raw in ("pred", "kine", "kinematics", "deq", "gino"):
        return "kinematics"
    return "gt"


@torch.no_grad()
def rollout_species_gnn_phi_trajectory(
    data,
    bundle: SpeciesGnnRolloutBundle,
    static: SpeciesGnnRolloutStatic | None = None,
    *,
    phys_cfg: PhysicsConfig | None = None,
    bio_cfg: BiochemConfig | None = None,
    device: torch.device | None = None,
    flow_source: str | None = None,
) -> dict[int, torch.Tensor]:
    from src.core_physics.t0_mu_physics import rollout_t0_clot_phi

    phys = phys_cfg or PhysicsConfig(phase="biochem")
    bio = bio_cfg or BiochemConfig(phase="biochem")
    dev = device or bundle.device
    pred = rollout_species_gnn_species_series(
        data, bundle, static, phys_cfg=phys, bio_cfg=bio, device=dev,
    )
    gel_beta = None
    if os.environ.get("SPECIES_VISCOSITY_CALIB", "").strip().lower() in ("1", "true", "yes", "on"):
        from src.core_physics.species_viscosity_calibration import (
            load_viscosity_calibration,
            viscosity_calibration_dir,
        )

        cal_path = os.environ.get("SPECIES_VISCOSITY_CALIB_PATH") or str(
            viscosity_calibration_dir() / "beta.pth"
        )
        if Path(cal_path).is_file():
            cal, _ = load_viscosity_calibration(cal_path, device=dev)
            gel_beta = cal.beta
    flow = _resolve_flow_source(flow_source)
    with t0_rung2_env():
        traj = rollout_t0_clot_phi(
            data, phys, bio, dev,
            gamma_mode=RUNG2_GAMMA_MODE, flow_source=flow,
            pred_species_series=pred, nucleation=True, nucleation_hops=1,
            gelation_beta=gel_beta,
        )
    return {int(t): v["phi"] for t, v in traj.items()}
