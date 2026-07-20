"""Species GNN -> full-timeline species -> physics clot trigger rollout.

Builds a ``(T, N, C)`` y-shaped species series from the species GNN pushforward
(continuous log-delta). Non-FI/Mat channels use resting plasma IC; only FI/Mat
are predicted. Clot phi uses ``clot_trigger_physics`` (gelation + nucleation), not ML.

See ``scripts/viz_species_gnn_clot_ladder.py`` and ``docs/MODEL_NOMENCLATURE.md``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch

from src.config import BiochemConfig, PhysicsConfig
from src.utils import species_channels as sc
from src.core_physics.clot_phi_simple import sdf_nd_from_data
from src.core_physics.species_deploy_rollout import (
    alloc_species_y_series,
    band_speed_for_rollout,
    deploy_fimat_log_init,
    pin_species_block,
    reset_species_rollout_flow_cache,
    species_rollout_pin_other,
)
from src.core_physics.species_pushforward_continuous import (
    SpeciesContinuousBundle,
    SpeciesDualHeadContinuousGNN,
    bind_band_geometry,
    continuous_max_sat_log,
    continuous_vel_decay_enabled,
    predict_continuous_step_delta,
    load_continuous_bundle,
    log_series_on_band,
    model_vel_decay_alphas,
    normalize_log_state,
    pushforward_log_state_step,
)
from src.core_physics.species_pushforward_gnn import (
    SpeciesPushforwardBundle,
    build_band_base_features,
    load_pushforward_bundle,
    pushforward_state_step,
)
from src.training.biochem_species_scope import (
    pushforward_state_dim,
    scatter_log_state_to_species_block,
)
from src.core_physics.species_snapshot_gnn import (
    build_snapshot_features,
    fi_mat_active_labels,
    induced_subgraph,
    snapshot_wall_hops,
    wall_band_mask,
)
from src.core_physics.t0_rung_config import RUNG2_GAMMA_MODE, t0_rung2_env
from src.training.biochem_species_scope import FI_CHANNEL, MAT_CHANNEL
from src.utils.kinematics_inference import (
    load_kinematics_predictor,
    predict_kinematics_latent,
    resolve_kinematics_checkpoint,
)
from src.utils.paths import get_project_root

RolloutKind = Literal["continuous", "binary"]

# Session cache for closed-loop kine/corrector handles (eval/viz multi-vessel).
_CLOSED_LOOP_MODELS_CACHE: dict[tuple, object] = {}


def clear_species_gnn_closed_loop_cache() -> None:
    _CLOSED_LOOP_MODELS_CACHE.clear()


def species_gnn_rollout_ckpt() -> Path:
    raw = (
        os.environ.get("T0_R4_SPECIES_GNN_CKPT")
        or os.environ.get("SPECIES_GNN_CLOUT_CKPT")
        or os.environ.get("SPECIES_CONTINUOUS_CKPT")
        or os.environ.get("SPECIES_PUSHFORWARD_CKPT")
        or "outputs/biochem/biochem_gnn/species/best.pth"
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
    pos_band: torch.Tensor | None = None
    flow_series: torch.Tensor | None = None  # [n_t, n_band, flow_dim] for dynamic flow (Trap C)
    flow_cols: tuple[int, int] | None = None  # (start, width) of the flow block in base_feats


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
    path_s = str(path).replace("\\", "/")
    if (
        "biochem_deploy" in phase
        or "biochem_gnn" in phase
        or "clot_deploy_gnn" in phase
        or "biochem_deploy" in path_s
        or "biochem_gnn" in path_s
        or "clot_deploy_gnn" in path_s
    ):
        return "biochem_deploy"
    if "continuous" in phase or "biochem_gnn" in phase or "clot_deploy_gnn" in phase:
        return "biochem_gnn"
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
    scope = meta.get("pushforward_species_scope") or meta.get("species_scope")
    channels = meta.get("pushforward_species_channels") or meta.get("species_channels")
    if channels:
        if isinstance(channels, (list, tuple)):
            os.environ["BIOCHEM_PUSHFORWARD_SPECIES_CHANNELS"] = ",".join(str(int(c)) for c in channels)
        else:
            os.environ["BIOCHEM_PUSHFORWARD_SPECIES_CHANNELS"] = str(channels)
    elif scope:
        os.environ["BIOCHEM_PUSHFORWARD_SPECIES_SCOPE"] = str(scope)
    if bool(meta.get("dual_head")):
        os.environ["SPECIES_CONTINUOUS_DUAL_HEAD"] = "1"
    if bool(meta.get("vel_decay")):
        os.environ["SPECIES_CONTINUOUS_VEL_DECAY"] = "1"
    if bool(meta.get("flow_feats")):
        # Reproduce the trained clot-aware flow feature set; deploy source stays 'auto'
        # (kine base + corrector-coupled override) -- do NOT inherit the training 'gt' source.
        os.environ["SPECIES_FLOW_FEATS"] = "1"
        os.environ.pop("SPECIES_FLOW_FEATS_SOURCE", None)
        # Time-varying flow (Trap C): if the teacher trained on per-step flow, reproduce it at deploy
        # (the per-step coupled velocity in data.y supplies the dynamic series).
        if bool(meta.get("flow_dynamic")):
            os.environ["SPECIES_FLOW_FEATS_DYNAMIC"] = "1"
    if bool(meta.get("geom_feats")):
        # Leg C/D: reproduce the static non-flow geometry discriminator block at deploy.
        os.environ["SPECIES_GEOM_FEATS"] = "1"
    if bool(meta.get("saturation_gate")):
        os.environ["SPECIES_CONTINUOUS_SATURATION_GATE"] = "1"
    # Retired: do not re-enable temporal lambda gate from legacy checkpoint metadata.
    os.environ["SPECIES_CONTINUOUS_TEMPORAL_GATE"] = "0"

    # Restore leg spec overrides if present in metadata or inferred from path
    overrides = meta.get("env_overrides")
    if overrides:
        for k, v in overrides.items():
            os.environ[k] = str(v)
    else:
        path_s = str(path).replace("\\", "/")
        if "mat_growth_ladder/" in path_s:
            parts = path_s.split("mat_growth_ladder/")
            if len(parts) > 1:
                leg = parts[1].split("/")[0]
                if leg:
                    try:
                        from src.biochem_gnn.mat_growth_simple import apply_mat_growth_leg_env
                        apply_mat_growth_leg_env(leg, force=True)
                    except Exception as e:
                        if not quiet:
                            print(f"[WARN] Failed to apply leg env for {leg} from path: {e}")

    label = _bundle_label_from_path(path, phase)
    if (
        "continuous" in phase
        or "biochem_gnn" in phase
        or "clot_deploy_gnn" in phase
        or bool(meta.get("dual_head"))
        or bool(meta.get("saturation_gate"))
        or bool(meta.get("vel_decay"))
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
    z_kin_override: torch.Tensor | None = None,
    kine_model=None,
) -> SpeciesGnnRolloutStatic:
    """Static band features for the species rollout.

    ``z_kin_override`` injects a clot-aware DEQ latent (full re-solve) so the GraphSAGE teacher's
    primary flow input tracks the rerouted flow once the clot is large enough to change it.

    When ``u0_pred``/``v0_pred`` are missing, runs one joint GINO-DEQ solve and stores them on
    ``data`` so closed-loop coupling does not re-solve the baseline later.
    """
    hops = int(wall_hops if wall_hops is not None else snapshot_wall_hops())
    kine = kine_model
    if kine is None:
        kine = load_kinematics_predictor(resolve_kinematics_checkpoint(), device)
    z_use = z_kin_override
    if z_use is None and (
        getattr(data, "u0_pred", None) is None or getattr(data, "v0_pred", None) is None
    ):
        from src.utils.kinematics_inference import predict_kinematics_and_latent

        pred_uv, z_use = predict_kinematics_and_latent(kine, data.to(device))
        data.u0_pred = pred_uv[:, 0].detach().to(device="cpu").clone()
        data.v0_pred = pred_uv[:, 1].detach().to(device="cpu").clone()
    elif z_use is None:
        z_use = predict_kinematics_latent(kine, data.to(device))
    stat = build_band_base_features(
        data, kine, device, wall_hops=hops, z_kin_override=z_use
    )
    return species_gnn_static_from_band_dict(stat, data, device=device, wall_hops=hops)


def species_gnn_static_from_band_dict(
    stat: dict,
    data,
    *,
    device: torch.device,
    wall_hops: int | None = None,
) -> SpeciesGnnRolloutStatic:
    """Wrap a ``build_band_base_features`` dict without reloading kinematics / re-solving DEQ."""
    hops = int(wall_hops if wall_hops is not None else snapshot_wall_hops())
    band = wall_band_mask(data, device, wall_hops=hops).reshape(-1).bool()
    return SpeciesGnnRolloutStatic(
        base_feats=stat["base_feats"],
        edge_index=stat["edge_index"],
        node_idx=stat["node_idx"],
        band=band,
        device=device,
        pos_band=stat.get("pos_band"),
        flow_series=stat.get("flow_series"),
        flow_cols=stat.get("flow_cols"),
    )


def _write_fimat_log_to_species(
    species: torch.Tensor,
    log_state: torch.Tensor,
    node_idx: torch.Tensor,
) -> torch.Tensor:
    return scatter_log_state_to_species_block(species, log_state, node_idx)


def _binary_state_to_log(state: torch.Tensor) -> torch.Tensor:
    fi_sat, mat_sat = continuous_max_sat_log()
    sd = pushforward_state_dim()
    st = state.reshape(-1, sd)
    out = torch.zeros_like(st)
    if sd > 0:
        out[:, 0] = torch.where(st[:, 0] > 0.5, torch.tensor(fi_sat, device=st.device, dtype=st.dtype), out[:, 0])
    if sd > 1:
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
    pin_other_species: str | None = None,
) -> torch.Tensor:
    """Full-timeline species series ``(T, N, C)`` with FI/Mat from GNN rollout.

    Non-FI/Mat channels use resting plasma IC (deploy default), not GT.
    """
    phys = phys_cfg or PhysicsConfig(phase="biochem")
    bio = bio_cfg or BiochemConfig(phase="biochem")
    dev = device or bundle.device
    stat = static or prepare_species_gnn_rollout_static(data, device=dev)
    pin_mode = pin_other_species if pin_other_species is not None else species_rollout_pin_other()
    reset_species_rollout_flow_cache()
    n_steps = int(data.y.shape[0])
    out = alloc_species_y_series(data, dev)

    if bundle.kind == "continuous":
        assert bundle.continuous is not None
        model = bundle.continuous.model
        wmask = data.mask_wall[stat.node_idx] if hasattr(data, "mask_wall") and data.mask_wall is not None else None
        bind_band_geometry(model, {"pos_band": stat.pos_band, "edge_index": stat.edge_index, "wall_mask_band": wmask})
        log_state = deploy_fimat_log_init(data, dev, stat.node_idx)
        vel_alphas = model_vel_decay_alphas(model) if continuous_vel_decay_enabled() else None

        coupler = None
        mu_bulk_si = None
        if os.environ.get("SPECIES_CLOSED_LOOP_COUPLING") == "1":
            try:
                from src.inference.corrector_coupling import (
                    ClotAwareFlow,
                    resolve_kinematics_checkpoint,
                    resolve_corrector_checkpoint,
                )
                from src.core_physics.clot_growth_masks import resolve_bulk_carreau_mu_si
                from src.core_physics.coupled_shear_gnn import load_local_corrector

                kine_ckpt = resolve_kinematics_checkpoint()
                corr_ckpt = resolve_corrector_checkpoint()
                resolve_on = os.environ.get("BIOCHEM_KINE_RESOLVE_ON_CLOT") == "1"
                if resolve_on:
                    cache_key = ("resolve", str(kine_ckpt), str(corr_ckpt), str(dev))
                    if cache_key in _CLOSED_LOOP_MODELS_CACHE:
                        kine, corr_model = _CLOSED_LOOP_MODELS_CACHE[cache_key]  # type: ignore[misc]
                    else:
                        kine = load_kinematics_predictor(kine_ckpt, dev)
                        corr_model = load_local_corrector(corr_ckpt, dev)
                        _CLOSED_LOOP_MODELS_CACHE[cache_key] = (kine, corr_model)
                else:
                    kine = None
                    corr_cache_key = ("corr", str(corr_ckpt), str(dev))
                    if corr_cache_key in _CLOSED_LOOP_MODELS_CACHE:
                        corr_model = _CLOSED_LOOP_MODELS_CACHE[corr_cache_key]  # type: ignore[assignment]
                    else:
                        corr_model = load_local_corrector(corr_ckpt, dev)
                        _CLOSED_LOOP_MODELS_CACHE[corr_cache_key] = corr_model
                coupler = ClotAwareFlow(dev, phys_cfg=phys)
                coupler._kine = kine
                coupler._corrector = corr_model
                u0, v0 = coupler.base_flow(data)
                mu_bulk_si = resolve_bulk_carreau_mu_si(data, 0, phys, dev, u_nd=u0, v_nd=v0).reshape(-1)
            except Exception as e:
                print(f"[WARN] Failed to initialize closed-loop flow coupler: {e}")

        for t in range(n_steps):
            sp = pin_species_block(data, t, dev, pin_other=pin_mode)  # type: ignore[arg-type]
            sp = _write_fimat_log_to_species(sp, log_state, stat.node_idx)
            out[t, :, sc.SPECIES_BLOCK] = sp
            if t >= n_steps - 1:
                break
            vel_val = data.y[t, stat.node_idx, 0:2] if hasattr(data, "y") and data.y is not None and t < data.y.shape[0] else None
            pred_delta = predict_continuous_step_delta(
                model,
                stat.base_feats,
                stat.edge_index,
                log_state,
                training=False,
                pos_band=stat.pos_band,
                # only thread time when dynamic flow is active (preserves prior temporal-gate behavior)
                time_index=(t if stat.flow_series is not None else None),
                flow_series=stat.flow_series,
                flow_cols=stat.flow_cols,
                wall_mask_band=wmask,
                species_block=sp,
                velocity=vel_val,
            )
            spd = (
                band_speed_for_rollout(data, t + 1, dev, stat.node_idx)
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

            # If closed loop coupling is enabled, update velocity for t+1
            if coupler is not None and t + 1 < n_steps:
                try:
                    from src.core_physics.species_gelation_readout import differentiable_clot_phi_from_species12, differentiable_mu_eff_from_species12
                    from src.core_physics.clot_phi_simple import comsol_carreau_mu_si_from_uv
                    from src.inference.corrector_coupling import write_coupled_flow_into_y
                    
                    sp_next = pin_species_block(data, t + 1, dev, pin_other=pin_mode)
                    sp_next = _write_fimat_log_to_species(sp_next, log_state, stat.node_idx)
                    species_log12 = sp_next
                    
                    phi_clot = differentiable_clot_phi_from_species12(species_log12, bio)
                    if (
                        os.environ.get("T0_R4_FLOW_SOURCE") == "kinematics"
                        or os.environ.get("CLOT_PHI_VEL_SOURCE") == "kinematics"
                        or not hasattr(data, "y") or data.y is None or data.y.numel() == 0 or bool((data.y == 0).all().item())
                    ):
                        u_t1, v_t1 = u0, v0
                    else:
                        u_t1 = data.y[t + 1, :, 0]
                        v_t1 = data.y[t + 1, :, 1]
                    gel_factor = torch.ones_like(u_t1)
                    mu_carreau_si = comsol_carreau_mu_si_from_uv(
                        data,
                        u_t1,
                        v_t1,
                        gel_factor,
                        phys,
                        device=dev,
                    )
                    mu_eff_si = differentiable_mu_eff_from_species12(species_log12, mu_carreau_si, phi_clot, bio).reshape(-1)
                    
                    state = coupler.update(data, mu_eff_si, mu_bulk_si=mu_bulk_si, publish=False)
                    write_coupled_flow_into_y(data, state.u, state.v, time_index=t + 1)
                    
                    if stat.flow_series is not None and stat.flow_cols is not None:
                        from src.core_physics.species_pushforward_gnn import _flow_feats_from_uv
                        flow_feats_next = _flow_feats_from_uv(data, state.u, state.v, dev, stat.node_idx)
                        stat.flow_series[t + 1] = flow_feats_next
                except Exception as e:
                    print(f"[WARN] Failed to apply closed-loop flow coupling at step {t+1}: {e}")
        from src.core_physics.species_viscosity_calibration import (
            apply_mat_beta_to_species_series,
            load_viscosity_calibration,
            resolve_deploy_gelation_beta,
            viscosity_calibration_dir,
        )

        gel_beta = resolve_deploy_gelation_beta(dev)
        if gel_beta is not None:
            cal_path = os.environ.get("SPECIES_VISCOSITY_CALIB_PATH") or str(
                viscosity_calibration_dir() / "beta.pth"
            )
            t_boost = max(int(out.shape[0]) - 1, 0)
            if Path(cal_path).is_file():
                _, calib_bundle = load_viscosity_calibration(cal_path, device=dev)
                t_boost = int(calib_bundle.time_index)
            out = apply_mat_beta_to_species_series(
                out, gel_beta, bio, time_index=min(t_boost, int(out.shape[0]) - 1)
            )
        return out

    assert bundle.binary is not None
    model = bundle.binary.model
    state = fi_mat_active_labels(deploy_fimat_log_init(data, dev, stat.node_idx))
    for t in range(n_steps):
        sp = pin_species_block(data, t, dev, pin_other=pin_mode)  # type: ignore[arg-type]
        log_state = _binary_state_to_log(state)
        sp = _write_fimat_log_to_species(sp, log_state, stat.node_idx)
        out[t, :, sc.SPECIES_BLOCK] = sp
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
    from src.core_physics.species_viscosity_calibration import resolve_deploy_gelation_beta

    gel_beta = resolve_deploy_gelation_beta(dev)
    flow = _resolve_flow_source(flow_source)
    import os
    nuc_hops = int(os.environ.get("CLOT_V2_NUCLEATION_HOPS", "1"))
    with t0_rung2_env():
        traj = rollout_t0_clot_phi(
            data, phys, bio, dev,
            gamma_mode=RUNG2_GAMMA_MODE, flow_source=flow,
            pred_species_series=pred, nucleation=True, nucleation_hops=nuc_hops,
            gelation_beta=gel_beta,
        )
    return {int(t): v["phi"] for t, v in traj.items()}
