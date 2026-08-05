"""Single protocol for the deploy clot metric, shared by training and offline eval.

Training and `scripts/eval_mat_growth_simple.py` both call `eval_deploy_clot_f1`, yet reported
0.877 vs 0.2996 for the same checkpoint on `patient020` (docs/GENERALIZATION_PLAN.md s2b-quater).
The rollout was never the difference -- the *protocol around it* was:

1. **Flow cache.** The offline path calls `reset_species_rollout_flow_cache()` **before** each
   anchor; training called it **after** the clot eval. So training scored using a coupled-flow
   field cached from the preceding teacher-forced training rollouts, i.e. a more-clotted (more
   occluded) state than a cold deploy would ever see -- which inflates growth.
2. **In-place `data.y`.** Closed-loop coupling writes diverted UV into `data.y`. Training reuses
   one val pack across every epoch (`.to(device)` returns self when already resident), so that
   mutation accumulates. `band_speed_at_time` reads those channels, so vel-decay drifts.
3. **Env.** Both applied `apply_deploy_env`, but only the offline path restored it around the
   call in all branches.
4. **Active PushforwardConfig.** Building a fresh deploy PushforwardConfig (via
   `build_deploy_configs` / `apply_deploy_env`) used to wipe CLI / recipe sparse-commitment
   overrides (`frontier_hops`, `nucleation_topk`, `gate_temp`, `mat_commit_thresh`). Mat F1
   (which runs first against the bound config) could move while clot metrics stayed flat.

Routing both callers through `canonical_deploy_clot_metrics` makes the protocol identical by
construction, so the two numbers cannot silently diverge again. Architecture knobs already bound
by the caller are preserved.
"""

from __future__ import annotations

import os
from dataclasses import replace
from typing import Any

import torch

# Env keys that the deploy protocol mutates and must hand back untouched.
_PROTOCOL_ENV_KEYS = (
    "SPECIES_ROLLOUT_VEL_SOURCE",
    "SPECIES_ROLLOUT_PIN_OTHER",
    "SPECIES_ROLLOUT_IC_SOURCE",
    "SPECIES_ROLLOUT_DEPLOY_FAITHFUL",
    "T0_R4_FLOW_SOURCE",
)

# Train-noise zeros for any residual legacy readers (typed pf already zeros these when preserved).
_NOISE_ENV_KEYS = (
    "SPECIES_CONTINUOUS_TEACHER_NOISE",
    "SPECIES_CONTINUOUS_TEACHER_FP_FRAC",
    "SPECIES_CONTINUOUS_TEACHER_BLUR",
    "SPECIES_PUSHFORWARD_INPUT_NOISE",
    "SPECIES_SCHEDULED_SAMPLING",
)


def _zero_train_noise_on_pf(pf: Any) -> Any:
    """Deploy: no teacher / input noise on the architecture config."""
    return replace(
        pf,
        teacher_noise=0.0,
        teacher_fp_frac=0.0,
        teacher_blur=0.0,
        input_noise=0.0,
        scheduled_sampling=False,
    )


def bind_canonical_deploy_protocol(
    *,
    flow: str,
) -> tuple[Any, Any]:
    """Bind deploy runtime + IO while preserving the caller's active PushforwardConfig.

    Returns the ``(pf, rt)`` pair that was bound. Callers that already set sparse-commitment
    overrides via ``_apply_ckpt_recipe`` / CLI must keep those fields on ``pf``.
    """
    from src.architecture.pushforward_config import get_active_config
    from src.architecture.runtime_config import get_active_runtime
    from src.biochem_gnn.config import _IO_ENV_KEYS, _bind_typed_configs, build_deploy_configs

    prev_pf = get_active_config()
    prev_rt = get_active_runtime()
    built_pf, built_rt, io_env = build_deploy_configs(overrides={"T0_R4_FLOW_SOURCE": flow})

    # Prefer the already-bound architecture config (ckpt recipe + deploy CLI overrides).
    pf = _zero_train_noise_on_pf(prev_pf) if prev_pf is not None else _zero_train_noise_on_pf(built_pf)

    rt = built_rt.with_overrides(t0_flow_source=flow, train_deploy_eval_flow=flow)
    # Preserve an explicit gelation gain bound by the caller. build_deploy_configs() rebuilds
    # the runtime from the manifest, which would otherwise drop a beta sweep value on the
    # floor and silently grade every arm at beta=1 (docs/WALL_MODEL_PLAN.md s1).
    if prev_rt is not None and str(prev_rt.gelation.beta_override or "").strip():
        rt = rt.with_overrides(
            beta_override=str(prev_rt.gelation.beta_override).strip(),
            beta_min=float(prev_rt.gelation.beta_min),
            beta_max=float(prev_rt.gelation.beta_max),
        )
    # Preserve two-model / off-wall routing bound by the offline eval launcher.
    if prev_rt is not None and prev_rt.offwall.two_model_mode:
        rt = rt.with_overrides(
            two_model_mode=True,
            offwall_model_ckpt=prev_rt.offwall.offwall_model_ckpt,
            two_model_route=prev_rt.offwall.two_model_route,
            two_model_frontier_hops=prev_rt.offwall.two_model_frontier_hops,
            frontier_hops_map=prev_rt.offwall.frontier_hops_map,
            frontier_hops_anchor=prev_rt.offwall.frontier_hops_anchor,
            spatial_gate_heads=prev_rt.offwall.spatial_gate_heads,
        )

    _bind_typed_configs(pf, rt)

    # IO / process env only -- do not rebuild typed architecture via apply_deploy_env.
    for key, val in io_env.items():
        if key in _IO_ENV_KEYS or key in _PROTOCOL_ENV_KEYS:
            os.environ[key] = str(val)
    os.environ["T0_R4_FLOW_SOURCE"] = str(flow)
    for key in _NOISE_ENV_KEYS:
        if key == "SPECIES_SCHEDULED_SAMPLING":
            os.environ[key] = "0"
        else:
            os.environ[key] = "0.0"
    return pf, rt


@torch.no_grad()
def canonical_deploy_clot_metrics(
    model: torch.nn.Module,
    data: Any,
    static: dict,
    phys_cfg: Any,
    bio_cfg: Any,
    device: torch.device,
    *,
    time_index: int | None = None,
    flow_source: str | None = None,
) -> dict[str, float]:
    """Deploy clot metrics under the canonical protocol.

    Resets the coupled-flow cache first, isolates ``data`` from in-place UV writes, applies the
    deploy runtime protocol, and restores env + typed configs afterwards. Preserves any active
    ``PushforwardConfig`` (including sparse-commitment CLI overrides).
    """
    from src.architecture.pushforward_config import get_active_config
    from src.architecture.runtime_config import get_active_runtime
    from src.biochem_gnn.config import _bind_typed_configs
    from src.core_physics.species_deploy_rollout import reset_species_rollout_flow_cache
    from src.core_physics.species_pushforward_continuous import (
        eval_deploy_clot_f1,
        train_deploy_eval_flow_source,
    )

    flow = (flow_source or train_deploy_eval_flow_source()).strip().lower()

    # (1) Never inherit a flow field cached from training rollouts.
    reset_species_rollout_flow_cache()

    # (2) Coupling writes diverted UV into data.y; keep the caller's pack pristine.
    eval_data = data.clone() if hasattr(data, "clone") else data

    env_snap = {k: os.environ.get(k) for k in (*_PROTOCOL_ENV_KEYS, *_NOISE_ENV_KEYS)}
    prev_pf = get_active_config()
    prev_rt = get_active_runtime()
    try:
        bind_canonical_deploy_protocol(flow=flow)
        metrics = eval_deploy_clot_f1(
            model,
            eval_data,
            static,
            phys_cfg,
            bio_cfg,
            device,
            time_index=time_index,
            flow_source=flow,
        )
    finally:
        # (3) Restore env + prior typed configs on every path, including exceptions.
        for key, val in env_snap.items():
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val
        if prev_pf is not None or prev_rt is not None:
            _bind_typed_configs(
                prev_pf if prev_pf is not None else get_active_config(),
                prev_rt if prev_rt is not None else get_active_runtime(),
            )
        reset_species_rollout_flow_cache()

    out = {str(k): float(v) for k, v in metrics.items()}
    out["canonical_protocol"] = 1.0
    return out
