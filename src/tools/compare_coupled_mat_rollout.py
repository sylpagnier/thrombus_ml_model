"""Phase 2 system-level test: does corrector coupling fix 'Mat' nucleation localization?

Runs the deploy species rollout twice on a biochem anchor graph and scores the predicted
'Mat' (mature fibrin) species against the COMSOL ground truth:

  * Uncoupled baseline -- the standard recipe with the frozen GINO-DEQ flow.
  * Coupled surrogate  -- the same rollout, but the flow is dynamically bent around the
    growing clot by the trained local kinematic corrector (Steps A-F in
    ``src/inference/corrector_coupling.py``) before it is fed to the biochem model.

Because the issue is *localization*, the headline metric is the spatial overlap (Dice / F1)
of the active 'Mat' nodes. If the corrector works, diverting the flow shifts the stagnation
zone so the model deposits 'Mat' in the correct downstream location -> higher F1 vs the
uncoupled baseline.

CLI:
    python -m src.tools.compare_coupled_mat_rollout \
        --graph data/processed/graphs_biochem_anchors/patient007.pt \
        --species-ckpt outputs/biochem/biochem_gnn/species/best.pth \
        --corrector outputs/kinematics/local_corrector/local_kinematic_corrector_best.pth
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from typing import Any

import torch

from src.config import STATE_CHANNEL_MU_EFF_ND, BiochemConfig, PhysicsConfig
from src.core_physics.species_gnn_clot_rollout import (
    load_species_gnn_rollout_bundle,
    prepare_species_gnn_rollout_static,
    rollout_species_gnn_species_series,
    species_gnn_rollout_ckpt,
)
from src.core_physics.species_snapshot_gnn import snapshot_active_log_nd, wall_band_mask
from src.core_physics.t0_mu_physics import predict_mu_si_at_time
from src.core_physics.t0_rung_config import RUNG2_GAMMA_MODE
from src.inference.corrector_coupling import (
    ClotAwareFlow,
    clot_nodes_from_delta_mu,
    reset_coupled_flow_registry,
    resolve_corrector_checkpoint,
    set_coupled_flow,
    write_coupled_flow_into_y,
)
from src.utils import species_channels as sc
from src.utils.paths import data_root, get_project_root

DEFAULT_GRAPH = data_root() / "processed" / "graphs_biochem_anchors" / "patient007.pt"
MAT_Y_COL = sc.y_index("Mat")  # absolute column in y[..., :] for the Mat species (== 15)


def _overlap_metrics(pred_active: torch.Tensor, gt_active: torch.Tensor) -> dict[str, float]:
    """Binary spatial-overlap metrics (Dice == F1 for sets) on the eval mask."""
    p = pred_active.reshape(-1).bool()
    g = gt_active.reshape(-1).bool()
    tp = float((p & g).sum().item())
    fp = float((p & ~g).sum().item())
    fn = float((~p & g).sum().item())
    denom = 2.0 * tp + fp + fn
    f1 = (2.0 * tp / denom) if denom > 0 else 0.0
    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    return {
        "mat_dice_f1": f1,
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "n_pred": tp + fp,
        "n_gt": tp + fn,
    }


def _mat_active_from_series(series: torch.Tensor, t: int, thr: float) -> torch.Tensor:
    return (series[int(t), :, MAT_Y_COL] > thr).to(dtype=torch.float32)


def _mat_active_from_gt(data, t: int, thr: float, device: torch.device) -> torch.Tensor:
    y = data.y[int(t)].to(device=device, dtype=torch.float32)
    return (y[:, MAT_Y_COL] > thr).to(dtype=torch.float32)


@torch.no_grad()
def _rollout_mat(data, bundle, static, phys, bio, device) -> torch.Tensor:
    return rollout_species_gnn_species_series(
        data, bundle, static, phys_cfg=phys, bio_cfg=bio, device=device
    )


@torch.no_grad()
def _couple_flow_into_graph(
    data,
    species_series: torch.Tensor,
    eval_times: list[int],
    *,
    coupler: ClotAwareFlow,
    phys: PhysicsConfig,
    bio: BiochemConfig,
    device: torch.device,
    oracle_mu: bool = False,
) -> dict[str, Any]:
    """Per-macro-step clot-aware flow refresh fed back into the biochem inputs.

    For each macro step it (1) estimates the clot ``mu`` field from the *uncoupled* species
    series, (2) lets :class:`ClotAwareFlow` pick the cheapest sufficient refresh -- local
    corrector diversion for small clots, full GINO-DEQ re-solve once the clot is large enough to
    reroute the global flow -- and (3) writes the resulting velocity into ``data.y`` so shear /
    nucleation / vel-decay see it. The representative (last, largest) refresh also yields the
    clot-aware latent ``z_kin`` that becomes the GraphSAGE teacher's primary input.
    """
    from src.core_physics.clot_growth_masks import resolve_bulk_carreau_mu_si

    n_clot_max = 0
    max_div = 0.0
    mode_last = "frozen"
    z_kin_clot: torch.Tensor | None = None
    last_uv: tuple[torch.Tensor, torch.Tensor] | None = None
    n_steps = int(data.y.shape[0])
    u0, v0 = coupler.base_flow(data)
    # Clot-free (Carreau bulk) viscosity reference from the base flow: the clot mask is the
    # elevation of mu_eff over this, so the corrector only fires on actual gelation -- not on
    # the non-Newtonian bulk (which sits well above mu_inf everywhere).
    mu_bulk_si = resolve_bulk_carreau_mu_si(data, 0, phys, device, u_nd=u0, v_nd=v0).reshape(-1)
    for t in range(n_steps):
        if oracle_mu:
            # Drive the diversion from the TRUE COMSOL clot mu (removes the predicted-mu feedback
            # confound): tests the upper bound of "divert around the real clot -> better Mat?".
            mu_eff_si = phys.viscosity_nd_to_si(
                data.y[t].to(device=device, dtype=torch.float32)[:, STATE_CHANNEL_MU_EFF_ND]
            ).reshape(-1)
        else:
            step = predict_mu_si_at_time(
                data,
                t,
                phys,
                bio,
                device=device,
                gamma_mode=RUNG2_GAMMA_MODE,
                flow_source="gt",
                pred_species_series=species_series,
            )
            mu_eff_si = step.mu_pred_si.reshape(-1)
        state = coupler.update(data, mu_eff_si, mu_bulk_si=mu_bulk_si, publish=False)
        n_clot_max = max(n_clot_max, state.n_clot)
        max_div = max(
            max_div,
            float((state.u - u0).abs().max().item()),
            float((state.v - v0).abs().max().item()),
        )
        write_coupled_flow_into_y(data, state.u, state.v, time_index=t)
        last_uv = (state.u, state.v)
        mode_last = state.mode
        if state.z_kin is not None:
            z_kin_clot = state.z_kin  # keep the latest clot-aware latent (largest clot)
    if last_uv is not None:
        set_coupled_flow(data, last_uv[0], last_uv[1])
    return {
        "max_clot_nodes": n_clot_max,
        "max_abs_diversion_nd": max_div,
        "final_mode": mode_last,
        "kine_resolved": z_kin_clot is not None,
        "z_kin_clot": z_kin_clot,
    }


def run(
    graph_path: Path | str = DEFAULT_GRAPH,
    *,
    species_ckpt: Path | str | None = None,
    corrector_ckpt: Path | str | None = None,
    eval_times: list[int] | None = None,
    device: torch.device | str | None = None,
    out_json: Path | str | None = None,
    oracle_mu: bool = False,
    gt_flow: bool = False,
    gt_flow_dynamic: bool = False,
) -> dict[str, Any]:
    dev = torch.device(device) if device is not None else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    thr = snapshot_active_log_nd()

    ckpt = Path(species_ckpt) if species_ckpt else species_gnn_rollout_ckpt()
    bundle = load_species_gnn_rollout_bundle(ckpt, device=dev)
    if bundle is None:
        raise FileNotFoundError(f"species GNN checkpoint missing: {ckpt}")

    base_data = torch.load(graph_path, map_location=dev, weights_only=False)
    n_steps = int(base_data.y.shape[0])
    if eval_times is None:
        eval_times = sorted({0, n_steps // 2, n_steps - 1})
    eval_times = sorted({max(0, min(int(t), n_steps - 1)) for t in eval_times})
    band = wall_band_mask(base_data, dev, wall_hops=1).reshape(-1).bool()

    # Pin the representative flow-feature time to match training (SPECIES_FLOW_FEATS_TIME=-1 = last/
    # formed clot) across EVERY pass, so baseline/gt-flow/coupled all read the same snapshot time.
    os.environ["SPECIES_FLOW_FEATS_TIME"] = "-1"

    # --- Uncoupled baseline -------------------------------------------------------------
    os.environ["BIOCHEM_CORRECTOR_COUPLING"] = "0"
    reset_coupled_flow_registry()
    data_base = copy.deepcopy(base_data).to(dev)
    static_base = prepare_species_gnn_rollout_static(data_base, device=dev)
    series_base = _rollout_mat(data_base, bundle, static_base, phys, bio, dev)

    baseline_rows: list[dict[str, Any]] = []
    for t in eval_times:
        pred = _mat_active_from_series(series_base, t, thr)[band]
        gt = _mat_active_from_gt(data_base, t, thr, dev)[band]
        baseline_rows.append({"time": int(t), **_overlap_metrics(pred, gt)})

    # --- Coupled surrogate --------------------------------------------------------------
    if gt_flow:
        # Clean oracle diagnostic: feed the TRUE COMSOL velocity (already in data.y[:, :, 0:2])
        # to the model's flow consumers, NO corrector and NO z_kin re-solve (latent frozen). This
        # isolates the single question the corrector cannot: does the GraphSAGE even *benefit*
        # from accurate flow at inference, or has it locked onto the frozen-kine field it trained
        # on? If GT flow also regresses, the corrector is the wrong lever -- the model must be
        # retrained to consume the coupled flow.
        os.environ["BIOCHEM_CORRECTOR_COUPLING"] = "0"
        os.environ["SPECIES_ROLLOUT_VEL_SOURCE"] = "gt"
        # Route the TRUE COMSOL velocity into the flow-aware band features (speed/shear/divergence).
        # Without this the features fall back to the frozen kine flow (coupling is off) and the gate
        # reduces to a vel-decay swap -- a no-op. The baseline pass already ran with the default
        # (auto->kine) source, so this gives the real A/B: kine-flow vs GT-flow band features.
        os.environ["SPECIES_FLOW_FEATS_SOURCE"] = "gt"
        # Trap C gate: time-varying GT flow features (per-step). Run on a dynamic-trained teacher;
        # the delta vs static --gt-flow is the temporal-sharpening headroom.
        os.environ["SPECIES_FLOW_FEATS_DYNAMIC"] = "1" if gt_flow_dynamic else "0"
        data_cpl = copy.deepcopy(base_data).to(dev)  # data_cpl.y keeps the untouched GT velocity
        reset_coupled_flow_registry()
        couple_info = {
            "max_clot_nodes": 0,
            "max_abs_diversion_nd": 0.0,
            "final_mode": "gt_flow_dynamic" if gt_flow_dynamic else "gt_flow",
            "kine_resolved": False,
            "z_kin_clot": None,
        }
    else:
        # Enable coupling + kine re-solve BEFORE the refresh so the burden gate can escalate to a
        # full DEQ re-solve (which regenerates z_kin); flow lives in data_cpl.y[:, :, 0:2] after.
        os.environ["BIOCHEM_CORRECTOR_COUPLING"] = "1"
        os.environ["SPECIES_ROLLOUT_VEL_SOURCE"] = "gt"
        data_cpl = copy.deepcopy(base_data).to(dev)
        coupler = ClotAwareFlow(dev, corrector_ckpt=corrector_ckpt)
        reset_coupled_flow_registry()
        couple_info = _couple_flow_into_graph(
            data_cpl, series_base, eval_times, coupler=coupler, phys=phys, bio=bio, device=dev,
            oracle_mu=oracle_mu,
        )
        # Free the coupler's kine model NOW: prepare_species_gnn_rollout_static loads its own kine
        # model for the flow features, and two GINO-DEQ models will not fit on a small (4 GiB) card.
        del coupler
    z_kin_clot = couple_info.pop("z_kin_clot", None)
    # Reclaim memory from the (possible) clot-aware DEQ re-solve before the flow-feature solve --
    # on small GPUs the two back-to-back solves fragment the heap and OOM (corrector_resolve rung).
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    # When the clot was big enough to re-solve the DEQ, feed the clot-aware latent in as the
    # GraphSAGE teacher's primary flow input (otherwise keep the frozen latent + corrector uv).
    static_cpl = prepare_species_gnn_rollout_static(
        data_cpl, device=dev, z_kin_override=z_kin_clot
    )
    series_cpl = _rollout_mat(data_cpl, bundle, static_cpl, phys, bio, dev)

    coupled_rows: list[dict[str, Any]] = []
    for t in eval_times:
        pred = _mat_active_from_series(series_cpl, t, thr)[band]
        gt = _mat_active_from_gt(data_cpl, t, thr, dev)[band]
        coupled_rows.append({"time": int(t), **_overlap_metrics(pred, gt)})

    reset_coupled_flow_registry()
    os.environ["BIOCHEM_CORRECTOR_COUPLING"] = "0"
    # Don't leak per-run sources/times into later calls (matters when run() is looped in-process).
    os.environ.pop("SPECIES_FLOW_FEATS_SOURCE", None)
    os.environ.pop("SPECIES_ROLLOUT_VEL_SOURCE", None)
    os.environ.pop("SPECIES_FLOW_FEATS_TIME", None)
    os.environ.pop("SPECIES_FLOW_FEATS_DYNAMIC", None)

    t_last = eval_times[-1]
    base_last = baseline_rows[-1]["mat_dice_f1"]
    cpl_last = coupled_rows[-1]["mat_dice_f1"]
    print("\n========== Mat localization: uncoupled vs corrector-coupled ==========")
    print(f"  graph={Path(graph_path).name} band_nodes={int(band.sum())} "
          f"clot_nodes<= {couple_info['max_clot_nodes']} max|div|_nd={couple_info['max_abs_diversion_nd']:.3e}")
    print(f"  flow refresh: final_mode={couple_info['final_mode']} "
          f"kine_resolved={couple_info['kine_resolved']} "
          f"(z_kin {'updated (clot-aware)' if couple_info['kine_resolved'] else 'frozen'})")
    print(f"  {'t':>5} | {'baseline F1':>12} | {'coupled F1':>11} | {'delta':>7}")
    for b, c in zip(baseline_rows, coupled_rows):
        d = c["mat_dice_f1"] - b["mat_dice_f1"]
        print(f"  {b['time']:>5} | {b['mat_dice_f1']:>12.3f} | {c['mat_dice_f1']:>11.3f} | {d:>+7.3f}")
    verdict = "IMPROVED" if cpl_last > base_last + 1e-6 else (
        "no change" if abs(cpl_last - base_last) <= 1e-6 else "REGRESSED")
    print(f"  t_last Mat Dice/F1: baseline {base_last:.3f} -> coupled {cpl_last:.3f}  [{verdict}]")
    print("======================================================================\n")

    result = {
        "graph": str(graph_path),
        "species_ckpt": str(ckpt),
        "corrector_ckpt": str(resolve_corrector_checkpoint(corrector_ckpt)),
        "eval_times": eval_times,
        "band_nodes": int(band.sum().item()),
        "coupling": couple_info,
        "baseline": baseline_rows,
        "coupled": coupled_rows,
        "t_last_baseline_f1": base_last,
        "t_last_coupled_f1": cpl_last,
        "verdict": verdict,
    }
    if out_json is None:
        out_json = get_project_root() / "outputs" / "biochem" / "corrector_coupling" / (
            f"{Path(graph_path).stem}_mat_overlap.json"
        )
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[save] {out_json}")
    return result


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compare Mat localization: uncoupled vs corrector-coupled flow.")
    p.add_argument("--graph", type=str, default=str(DEFAULT_GRAPH))
    p.add_argument("--species-ckpt", type=str, default=None)
    p.add_argument("--corrector", type=str, default=None)
    p.add_argument("--times", type=str, default="", help="comma macro indices (default: 0, mid, last)")
    p.add_argument("--oracle-mu", action="store_true",
                   help="Drive the diversion from the TRUE COMSOL clot mu (isolate from predicted-mu error).")
    p.add_argument("--gt-flow", action="store_true",
                   help="Diagnostic: feed the TRUE COMSOL velocity (no corrector, frozen z_kin) to "
                        "isolate whether the GraphSAGE benefits from accurate flow at all.")
    p.add_argument("--gt-flow-dynamic", action="store_true",
                   help="Trap C gate: like --gt-flow but TIME-VARYING flow features (per-step GT "
                        "velocity). Run on a dynamic-trained teacher; compare vs static --gt-flow to "
                        "read the temporal-sharpening headroom.")
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    return p


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    times = [int(x.strip()) for x in args.times.split(",") if x.strip()] if args.times.strip() else None
    run(
        graph_path=args.graph,
        species_ckpt=args.species_ckpt,
        corrector_ckpt=args.corrector,
        eval_times=times,
        device=args.device,
        out_json=args.out,
        oracle_mu=args.oracle_mu,
        gt_flow=args.gt_flow or args.gt_flow_dynamic,
        gt_flow_dynamic=args.gt_flow_dynamic,
    )
