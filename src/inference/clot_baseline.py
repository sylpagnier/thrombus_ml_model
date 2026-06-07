"""Chained inference: GNODE teacher rollout + clot-phi spatial readout."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import torch

from src.config import BiochemConfig, PhysicsConfig
from src.core_physics.clot_phi_rollout import ClotPhiRolloutState, clot_phi_rollout_enabled
from src.core_physics.clot_phi_simple import (
    apply_clot_support_projection,
    build_clot_phi_model,
    build_clot_phi_step,
    clot_phi_hard_support_projection_enabled,
    clot_phi_hybrid_enabled,
    log_blend_mu_eff_si,
    mu_eff_from_delta_log_si,
    resolve_clot_support_band_for_step,
)
from src.evaluation.clot_phi_checkpoint_env import (
    apply_clot_phi_config_from_checkpoint,
    apply_clot_phi_eval_defaults,
)
from src.inference.clot_phi_inject_attach import attach_clot_phi_injector_to_teacher
from src.inference.deploy_mu_map_env import apply_deploy_mu_map_env, clear_oracle_mu_map_env
from src.inference.clot_baseline_recipe import (
    ClotBaselineRecipe,
    baseline_manifest_path,
    default_lane_a_recipe,
    load_manifest,
)
from src.utils.channel_schema import BIO_Y_SCHEMA, assert_graph_schema, build_y_valid_mask, infer_missing_schema
from src.utils.nondim import to_t_nd
from src.utils.paths import get_project_root


def _load_clot_phi_model(ckpt_path: Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    raw = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = raw.get("config") or {}
    apply_clot_phi_config_from_checkpoint(cfg)
    apply_clot_phi_eval_defaults()
    os.environ.setdefault("CLOT_PHI_DGAMMA_FEATURE_TIME", "current")
    in_dim = int(cfg.get("in_dim", 6))
    hidden = int(cfg.get("hidden", 32))
    model = build_clot_phi_model(in_dim=in_dim, hidden=hidden).to(device)
    model.load_state_dict(raw["model_state_dict"])
    model.eval()
    return model, cfg


def _project_deploy_mu_if_enabled(
    *,
    data,
    step,
    mu_pred: torch.Tensor,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    device: torch.device,
    forecast_one_step: bool,
) -> torch.Tensor:
    if not clot_phi_hard_support_projection_enabled():
        return mu_pred.reshape(-1)
    band = resolve_clot_support_band_for_step(
        data,
        device,
        step,
        phys_cfg,
        bio_cfg,
        forecast_one_step=forecast_one_step,
    )
    return apply_clot_support_projection(step.mu_c_si, mu_pred, band)


@torch.no_grad()
def predict_clot_phi_at_time(
    model: torch.nn.Module,
    data,
    time_index: int,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    device: torch.device,
    *,
    rollout_state: ClotPhiRolloutState | None = None,
) -> dict[str, torch.Tensor]:
    """Run clot-phi at one time index (warm-start rollout carry when enabled)."""
    if rollout_state is not None and clot_phi_rollout_enabled() and time_index > 0:
        state = ClotPhiRolloutState()
        for t_run in range(0, time_index):
            step_run = build_clot_phi_step(data, t_run, phys_cfg, bio_cfg, device, rollout_state=state)
            phi_r = model(step_run.features)
            if clot_phi_hybrid_enabled() and hasattr(model, "forward_delta_log_mu"):
                mu_r = mu_eff_from_delta_log_si(
                    step_run.mu_c_si, model.forward_delta_log_mu(step_run.features)
                )
            else:
                mu_r = log_blend_mu_eff_si(step_run.mu_c_si, phi_r)
            mu_r = _project_deploy_mu_if_enabled(
                data=data,
                step=step_run,
                mu_pred=mu_r,
                phys_cfg=phys_cfg,
                bio_cfg=bio_cfg,
                device=device,
                forecast_one_step=False,
            )
            state.update_from_pred(phi_r, mu_r, detach=True)
        rollout_state = state
    else:
        rollout_state = rollout_state or (ClotPhiRolloutState() if clot_phi_rollout_enabled() else None)

    step = build_clot_phi_step(
        data, time_index, phys_cfg, bio_cfg, device, rollout_state=rollout_state
    )
    phi_pred = model(step.features)
    if clot_phi_hybrid_enabled() and hasattr(model, "forward_delta_log_mu"):
        mu_pred = mu_eff_from_delta_log_si(step.mu_c_si, model.forward_delta_log_mu(step.features))
    else:
        mu_pred = log_blend_mu_eff_si(step.mu_c_si, phi_pred)
    mu_pred = _project_deploy_mu_if_enabled(
        data=data,
        step=step,
        mu_pred=mu_pred,
        phys_cfg=phys_cfg,
        bio_cfg=bio_cfg,
        device=device,
        forecast_one_step=False,
    )
    return {
        "phi": phi_pred.reshape(-1),
        "mu_eff_si": mu_pred.reshape(-1),
        "region": step.region.reshape(-1),
        "support_band": resolve_clot_support_band_for_step(
            data,
            device,
            step,
            phys_cfg,
            bio_cfg,
            forecast_one_step=False,
        ).reshape(-1)
        if clot_phi_hard_support_projection_enabled()
        else step.region.reshape(-1),
        "phi_gt": step.phi_gt.reshape(-1),
        "mu_gt_cap": step.mu_gt_cap.reshape(-1),
    }


class ClotBaselinePredictor:
    """Deploy model: GNODE temporal rollout then clot-phi map readout."""

    def __init__(
        self,
        recipe: ClotBaselineRecipe,
        device: torch.device | None = None,
    ) -> None:
        self.recipe = recipe
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.root = get_project_root()
        paths = recipe.resolved_paths(self.root)
        teacher_path = paths.get("teacher_ckpt")
        clot_path = paths.get("clot_phi_ckpt")
        if not teacher_path or not teacher_path.is_file():
            raise FileNotFoundError(f"Teacher checkpoint missing: {recipe.teacher_ckpt}")
        if not clot_path or not clot_path.is_file():
            raise FileNotFoundError(f"Clot-phi checkpoint missing: {recipe.clot_phi_ckpt}")

        self.teacher, _, self.mu_ratio_max = load_biochem_teacher_checkpoint(
            teacher_path,
            self.device,
            mu_ratio_max=recipe.mu_ratio_max,
            pred_kine=recipe.pred_kine,
        )
        deploy_env = getattr(recipe, "deploy_mu_map_env", None) or {}
        recipe_name = getattr(recipe, "name", "") or ""
        use_wired = recipe_name in ("lane_a_wired", "B_wired") or (
            str(deploy_env.get("BIOCHEM_MLP_MU_MAP_MASK", "")).strip().lower() == "mlp_band"
        )
        if deploy_env or recipe_name in ("lane_b_deploy", "lane_a_wired", "B_wired"):
            clear_oracle_mu_map_env()
            from src.inference.deploy_mu_map_env import (
                apply_deploy_mu_map_env,
                apply_wired_deploy_mu_map_env,
            )

            if use_wired:
                apply_wired_deploy_mu_map_env(deploy_env if deploy_env else None)
            else:
                apply_deploy_mu_map_env(deploy_env if deploy_env else None)
            os.environ["BIOCHEM_MLP_CLOT_CKPT"] = str(clot_path)
            attach_clot_phi_injector_to_teacher(self.teacher, self.device, clot_path)
            self.deploy_mu_map = True
        else:
            self.deploy_mu_map = False
        self.clot_model, self.clot_cfg = _load_clot_phi_model(clot_path, self.device)
        self.phys_cfg = PhysicsConfig(phase="biochem")
        self.bio_cfg = BiochemConfig(phase="biochem")

    @classmethod
    def from_manifest(cls, path: str | Path | None = None, device: torch.device | None = None) -> ClotBaselinePredictor:
        recipe, _ = load_manifest(path)
        return cls(recipe, device=device)

    @torch.no_grad()
    def rollout_teacher(self, data) -> torch.Tensor:
        """GNODE forward; returns pred_series same shape as ``data.y``."""
        data = infer_missing_schema(data, phase_hint="biochem")
        assert_graph_schema(data, expected_y_schema=(BIO_Y_SCHEMA,))
        t_si = self.bio_cfg.resolve_biochem_times(data, self.device)
        t_ref = float(getattr(self.bio_cfg, "t_final", 30000.0))
        eval_t = to_t_nd(t_si, t_ref)
        return self.teacher(
            data,
            eval_t,
            y_true_trajectory=data.y,
            teacher_forcing_ratio=1.0,
            start_idx=0,
            initial_species=None,
            detach_macro_state=True,
        )

    def apply_teacher_rollout(self, data, pred_series: torch.Tensor):
        """Write teacher species + pred [u,v,p] into ``data.y`` (in-place on CPU copy)."""
        data_out = data.to("cpu")
        pred_cpu = pred_series.to("cpu")
        data_out.y[:, :, 4:16] = pred_cpu[:, :, 4:16]
        if self.recipe.pred_kine:
            data_out.y[:, :, 0:3] = pred_cpu[:, :, 0:3]
        if hasattr(data_out, "y_valid_mask") and torch.is_tensor(data_out.y_valid_mask):
            if tuple(data_out.y_valid_mask.shape) != tuple(data_out.y.shape):
                data_out.y_valid_mask = build_y_valid_mask(
                    data_out.y, data_out.y_schema, getattr(data_out, "mask_wall", None)
                )
        return data_out

    @torch.no_grad()
    def predict(
        self,
        data,
        time_index: int = -1,
        *,
        run_teacher: bool = True,
    ) -> dict[str, Any]:
        """
        Full pipeline on one anchor graph.

        If ``run_teacher``, rolls out GNODE and patches ``y`` before clot-phi.
        If graph already has dumped pred species/vel, pass ``run_teacher=False``.
        """
        data = infer_missing_schema(data, phase_hint="biochem").to(self.device)
        if run_teacher:
            pred_series = self.rollout_teacher(data)
            data = self.apply_teacher_rollout(data, pred_series).to(self.device)

        t_last = int(data.y.shape[0]) - 1
        ti = time_index if time_index >= 0 else t_last + 1 + time_index if time_index < 0 else time_index
        ti = max(0, min(ti, t_last))

        out = predict_clot_phi_at_time(
            self.clot_model, data, ti, self.phys_cfg, self.bio_cfg, self.device
        )
        out["time_index"] = ti
        return out

    @torch.no_grad()
    def predict_trajectory(
        self,
        data,
        *,
        run_teacher: bool = True,
        time_stride: int = 1,
    ) -> list[dict[str, Any]]:
        """Clot-phi at each time step (serial carry when rollout mode on)."""
        data = infer_missing_schema(data, phase_hint="biochem").to(self.device)
        if run_teacher:
            pred_series = self.rollout_teacher(data)
            data = self.apply_teacher_rollout(data, pred_series).to(self.device)

        t_last = int(data.y.shape[0]) - 1
        stride = max(1, int(time_stride))
        times = list(range(0, t_last + 1, stride))
        if times[-1] != t_last:
            times.append(t_last)

        rollout_state = ClotPhiRolloutState() if clot_phi_rollout_enabled() else None
        results: list[dict[str, Any]] = []
        for ti in times:
            step_out = predict_clot_phi_at_time(
                self.clot_model,
                data,
                ti,
                self.phys_cfg,
                self.bio_cfg,
                self.device,
                rollout_state=rollout_state,
            )
            step_out["time_index"] = ti
            if rollout_state is not None and clot_phi_rollout_enabled():
                rollout_state.update_from_pred(
                    step_out["phi"], step_out["mu_eff_si"], detach=True
                )
            results.append(step_out)
        return results


def _cli() -> None:
    ap = argparse.ArgumentParser(description="Clot baseline predict (GNODE + clot-phi)")
    ap.add_argument("--manifest", default="", help="manifest.json (default: outputs/biochem/clot_baseline/)")
    ap.add_argument("--anchor", default="patient007")
    ap.add_argument("--anchor-dir", default="", help="Override graph dir (default: recipe dump dir)")
    ap.add_argument("--time-index", type=int, default=-1)
    ap.add_argument("--no-teacher-rollout", action="store_true", help="Graph already has dumped pred fields")
    args = ap.parse_args()

    manifest = Path(args.manifest) if args.manifest else baseline_manifest_path()
    if manifest.is_file():
        predictor = ClotBaselinePredictor.from_manifest(manifest)
    else:
        predictor = ClotBaselinePredictor(default_lane_a_recipe())

    root = get_project_root()
    anchor_dir = Path(args.anchor_dir) if args.anchor_dir else Path(predictor.recipe.dump_anchor_dir)
    if not anchor_dir.is_absolute():
        anchor_dir = root / anchor_dir
    graph_path = anchor_dir / f"{args.anchor}.pt"
    data = torch.load(graph_path, weights_only=False)
    out = predictor.predict(data, time_index=args.time_index, run_teacher=not args.no_teacher_rollout)
    region = out["region"].bool()
    phi = out["phi"][region]
    print(
        f"[OK]  t={out['time_index']} region_n={int(region.sum())} "
        f"mean_phi={float(phi.mean()):.3f} frac_phi>=0.5={float((phi >= 0.5).float().mean()):.3f}",
        flush=True,
    )


if __name__ == "__main__":
    _cli()
