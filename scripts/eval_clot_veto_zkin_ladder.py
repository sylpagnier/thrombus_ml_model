"""Final flow-lever ladder, scored on the REAL deploy clot F1 (not Mat-Dice).

Four configs per anchor, all deployable except the GT-shear ceiling:

  none        - standard deploy rollout (frozen kine), clot phi from the trigger.
  kine_veto   - same phi, then VETO candidate clot nodes whose kine-shear is high
                (low-shear keeps clot). Percentile threshold swept -> oracle-calibrated
                upper bound of the deployable veto.
  gt_veto     - same, but the shear used for the veto comes from the COMSOL GT velocity
                (data.y[t][:, 0:2]) -> the CEILING of the shear-veto idea (perfect shear,
                same operator as kine so the only difference is flow fidelity).
  tiled_zkin  - no veto, but z_kin is refreshed during the rollout by GEOMETRY OCCLUSION:
                predicted-clot nodes are marked as wall (SDF=0, UV_PRIOR=0) and the DEQ
                latent is re-solved, so the species teacher's primary flow input becomes
                clot-aware. NB: z_kin canNOT be set from a corrector (u,v) field -- it is
                the DEQ equilibrium and UV_PRIOR is only a warm start the solve washes out;
                geometry occlusion is the only in-distribution way to make z_kin clot-aware
                (mu-injection is OOD, see docs/SPECIES_LEARNING_STRATEGY.md 6.3).

Usage::

    python scripts/eval_clot_veto_zkin_ladder.py \
        --species-ckpt outputs/biochem/biochem_gnn/flow_aware_leashed_dynamic/sage/species/best.pth \
        --times 53,200
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.config import BiochemConfig, NodeFeat, PhysicsConfig  # noqa: E402
from src.core_physics.species_deploy_rollout import (  # noqa: E402
    alloc_species_y_series,
    band_speed_for_rollout,
    deploy_fimat_log_init,
    pin_species_block,
)
from src.core_physics.species_gnn_clot_rollout import (  # noqa: E402
    load_species_gnn_rollout_bundle,
    prepare_species_gnn_rollout_static,
    rollout_species_gnn_phi_trajectory,
)
from src.core_physics.species_pushforward_continuous import (  # noqa: E402
    BIOCHEM_ANCHORS_6,
    bind_band_geometry,
    continuous_vel_decay_enabled,
    discover_biochem_anchors,
    model_vel_decay_alphas,
    predict_continuous_step_delta,
    pushforward_log_state_step,
)
from src.core_physics.clot_phi_simple import sdf_nd_from_data  # noqa: E402
from src.core_physics.species_snapshot_gnn import (  # noqa: E402
    build_snapshot_features,
    kin_per_vessel_norm_enabled,
    kinematic_latent_band_stats,
    snapshot_active_log_nd,
    snapshot_wall_hops,
)
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time, rollout_t0_clot_phi  # noqa: E402
from src.core_physics.t0_rung_config import RUNG2_GAMMA_MODE, t0_rung2_env  # noqa: E402
from src.core_physics.species_viscosity_calibration import (  # noqa: E402
    apply_mat_beta_to_species_series,
    load_viscosity_calibration,
    resolve_deploy_gelation_beta,
    viscosity_calibration_dir,
)
from src.evaluation.clot_relaxed_metrics import (  # noqa: E402
    compute_clot_relaxed_metrics,
    legacy_clot_f1_metrics,
)
from src.training.biochem_species_scope import scatter_log_state_to_species_block  # noqa: E402
from src.utils.kinematics_inference import (  # noqa: E402
    load_kinematics_predictor,
    predict_kinematics,
    predict_kinematics_latent,
    resolve_kinematics_checkpoint,
)
from src.utils.paths import get_project_root  # noqa: E402

VETO_PERCENTILES = (50.0, 60.0, 70.0, 80.0, 90.0, 100.0)


def _apply_deploy_env() -> None:
    """Deploy-faithful rollout env (matches go_species_flow_aware deploy recipe)."""
    os.environ.setdefault("SPECIES_ROLLOUT_DEPLOY_FAITHFUL", "1")
    os.environ["SPECIES_ROLLOUT_IC_SOURCE"] = "resting"
    os.environ["SPECIES_ROLLOUT_PIN_OTHER"] = "rest"
    os.environ["SPECIES_ROLLOUT_VEL_SOURCE"] = "kinematics"
    os.environ.setdefault("SPECIES_VISCOSITY_CALIB", "1")
    # flow features deploy on the kine base flow (auto), never the training 'gt' source.
    os.environ.pop("SPECIES_FLOW_FEATS_SOURCE", None)
    os.environ.pop("SPECIES_FLOW_FEATS_ABLATE", None)


def _shear_proxy(u: torch.Tensor, v: torch.Tensor, edge_index: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """Per-node mean neighbour speed-gradient (same operator as the flow band features)."""
    u = u.reshape(-1)
    v = v.reshape(-1)
    speed = torch.sqrt(u * u + v * v)
    row, col = edge_index
    diff = pos[row] - pos[col]
    dist = diff.norm(dim=1).clamp(min=1e-6)
    grad = (speed[row] - speed[col]).abs() / dist
    n = int(pos.shape[0])
    acc = torch.zeros(n, device=pos.device, dtype=speed.dtype)
    deg = torch.zeros(n, device=pos.device, dtype=speed.dtype)
    acc.index_add_(0, row, grad)
    deg.index_add_(0, row, torch.ones_like(grad))
    return acc / deg.clamp(min=1.0)


def _veto_phi(phi: torch.Tensor, shear: torch.Tensor, percentile: float) -> torch.Tensor:
    """Drop candidate clot nodes (phi>=0.5) whose shear exceeds the q-th percentile."""
    pp = phi.reshape(-1).clone()
    cand = pp >= 0.5
    if percentile >= 100.0 or not bool(cand.any()):
        return pp
    s = shear.reshape(-1)
    thr = torch.quantile(s[cand], float(percentile) / 100.0)
    veto = cand & (s > thr)
    pp[veto] = 0.0
    return pp


def _score(phi_pred: torch.Tensor, data, t: int, phys: PhysicsConfig, dev: torch.device) -> dict[str, float]:
    phi_gt = gt_clot_phi_at_time(data, int(t), phys, dev).reshape(-1)
    pp = phi_pred.reshape(-1)
    ei = data.edge_index.to(dev)
    mask = torch.ones(int(data.num_nodes), device=dev, dtype=torch.bool)
    legacy = legacy_clot_f1_metrics(pp, phi_gt, mask)
    relaxed = compute_clot_relaxed_metrics(pp, phi_gt, ei)
    return {
        "f1": float(legacy["clot_f1"]),
        "prec": float(legacy["clot_prec"]),
        "rec": float(legacy["clot_rec"]),
        "guiding": float(relaxed["clot_guiding"]),
    }


def _best_veto(phi: torch.Tensor, shear: torch.Tensor, data, t: int, phys, dev) -> dict:
    best: dict = {"f1": -1.0}
    for q in VETO_PERCENTILES:
        sc = _score(_veto_phi(phi, shear, q), data, t, phys, dev)
        if sc["f1"] > best["f1"]:
            best = {**sc, "percentile": q}
    return best


def _kine_device(kine) -> torch.device:
    try:
        return next(kine.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _predict_uv_safe(kine, data, dev: torch.device) -> torch.Tensor:
    """Steady kine (u,v,p); robust to the shared model having fallen back to CPU."""
    kdev = _kine_device(kine)
    return predict_kinematics(kine, data.clone().to(kdev)).to(dev)


def _solve_latent_safe(kine, data_occ, dev: torch.device) -> torch.Tensor:
    """Clot-aware DEQ latent; mirrors the flow-feature OOM->CPU fallback so 4 GB GPUs survive."""
    kdev = _kine_device(kine)
    try:
        return predict_kinematics_latent(kine, data_occ.to(kdev)).to(dev)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        kine.to("cpu")
        return predict_kinematics_latent(kine, data_occ.clone().to("cpu")).to(dev)


def _occlude_graph(data, clot_global: torch.Tensor):
    """Mark predicted-clot nodes as wall: SDF=0, UV_PRIOR=0 -> clot-aware DEQ re-solve."""
    d = data.clone()
    x = d.x.clone()
    x[clot_global, NodeFeat.SDF] = 0.0
    x[clot_global, NodeFeat.UV_PRIOR] = 0.0
    d.x = x
    return d


def _refresh_base_feats(
    base_feats_init: torch.Tensor,
    z_occ: torch.Tensor,
    sdf_full: torch.Tensor,
    node_idx: torch.Tensor,
    dev: torch.device,
) -> torch.Tensor:
    """Swap ONLY the clot-aware z_kin block; reuse the (clot-independent) base-flow block.

    base_feats = [build_snapshot_features(z_kin, sdf) | flow_block]; the flow block depends on the
    frozen base flow, not the clot, so re-solving it every refresh is pure waste on a small GPU.
    """
    mean = std = None
    if kin_per_vessel_norm_enabled():
        mean, std = kinematic_latent_band_stats(z_occ, node_idx)
    snap = build_snapshot_features(z_occ, sdf_full, kin_mean=mean, kin_std=std)[node_idx]
    ld = int(z_occ.shape[1])  # snapshot width = latent_dim + 1 (sdf)
    flow_block = base_feats_init[:, ld + 1:]
    return torch.cat([snap.to(dtype=base_feats_init.dtype), flow_block], dim=1).to(dev)


def _apply_gelation_beta(out: torch.Tensor, bio: BiochemConfig, dev: torch.device) -> torch.Tensor:
    gel_beta = resolve_deploy_gelation_beta(dev)
    if gel_beta is None:
        return out
    cal_path = os.environ.get("SPECIES_VISCOSITY_CALIB_PATH") or str(viscosity_calibration_dir() / "beta.pth")
    t_boost = max(int(out.shape[0]) - 1, 0)
    if Path(cal_path).is_file():
        _, cb = load_viscosity_calibration(cal_path, device=dev)
        t_boost = int(cb.time_index)
    return apply_mat_beta_to_species_series(out, gel_beta, bio, time_index=min(t_boost, int(out.shape[0]) - 1))


@torch.no_grad()
def _species_series_occlusion_zkin(
    data, bundle, kine, phys, bio, dev, *, min_clot: int, growth: float, max_refresh: int
) -> torch.Tensor:
    """Continuous species rollout with hysteretic geometry-occlusion z_kin refresh."""
    if bundle.kind != "continuous" or bundle.continuous is None:
        raise NotImplementedError("occlusion z_kin refresh requires a continuous (biochem_deploy) ckpt")
    model = bundle.continuous.model
    hops = snapshot_wall_hops()
    stat = prepare_species_gnn_rollout_static(data, device=dev, wall_hops=hops)
    base_feats = stat.base_feats
    base_feats_init = stat.base_feats
    edge_index, node_idx, pos_band = stat.edge_index, stat.node_idx, stat.pos_band
    sdf_full = sdf_nd_from_data(data, dev, int(data.num_nodes))
    bind_band_geometry(model, {"pos_band": pos_band, "edge_index": edge_index})
    log_state = deploy_fimat_log_init(data, dev, node_idx)
    vel_alphas = model_vel_decay_alphas(model) if continuous_vel_decay_enabled() else None
    out = alloc_species_y_series(data, dev)
    n_steps = int(data.y.shape[0])
    thr = snapshot_active_log_nd()
    last_resolve_n = 0
    n_refresh = 0
    for t in range(n_steps):
        sp = pin_species_block(data, t, dev, pin_other="rest")  # type: ignore[arg-type]
        sp = scatter_log_state_to_species_block(sp, log_state, node_idx)
        out[t, :, 4:16] = sp

        mat = log_state.reshape(-1, 2)[:, 1]
        clot_local = torch.where(mat > thr)[0]
        n_clot = int(clot_local.numel())
        if (
            n_refresh < int(max_refresh)
            and n_clot >= int(min_clot)
            and (last_resolve_n == 0 or n_clot >= last_resolve_n * float(growth))
        ):
            try:
                data_occ = _occlude_graph(data, node_idx[clot_local])
                z_occ = _solve_latent_safe(kine, data_occ, dev)
                base_feats = _refresh_base_feats(base_feats_init, z_occ, sdf_full, node_idx, dev)
                last_resolve_n = n_clot
                n_refresh += 1
            except torch.cuda.OutOfMemoryError:
                if dev.type == "cuda":
                    torch.cuda.empty_cache()

        if t >= n_steps - 1:
            break
        pred_delta = predict_continuous_step_delta(
            model, base_feats, edge_index, log_state, training=False, pos_band=pos_band
        )
        spd = band_speed_for_rollout(data, t + 1, dev, node_idx) if vel_alphas is not None else None
        log_state = pushforward_log_state_step(
            log_state, pred_delta, straight_through=False, wall_speed=spd, vel_decay_alphas=vel_alphas
        )
    out = _apply_gelation_beta(out, bio, dev)
    out._n_zkin_refresh = n_refresh  # type: ignore[attr-defined]
    return out


@torch.no_grad()
def _phi_from_species(data, species, phys, bio, dev) -> dict[int, torch.Tensor]:
    gel_beta = resolve_deploy_gelation_beta(dev)
    with t0_rung2_env():
        traj = rollout_t0_clot_phi(
            data, phys, bio, dev,
            gamma_mode=RUNG2_GAMMA_MODE, flow_source="kinematics",
            pred_species_series=species, nucleation=True, nucleation_hops=1,
            gelation_beta=gel_beta,
        )
    return {int(t): v["phi"] for t, v in traj.items()}


@torch.no_grad()
def eval_anchor(anchor: str, *, bundle, kine, dev, times: list[int], min_clot: int, growth: float, max_refresh: int) -> dict:
    root = get_project_root()
    data = torch.load(
        root / "data/processed/graphs_biochem_anchors" / f"{anchor}.pt",
        map_location=dev, weights_only=False,
    )
    phys = PhysicsConfig(phase="biochem")
    bio = BiochemConfig(phase="biochem")
    n_steps = int(data.y.shape[0])
    eval_times = sorted({max(0, min(int(t), n_steps - 1)) for t in times})
    t_main = eval_times[-1]
    ei = data.edge_index.to(dev)
    pos = data.x[:, :2].to(device=dev, dtype=torch.float32)

    static = prepare_species_gnn_rollout_static(data, device=dev)
    phi_traj = rollout_species_gnn_phi_trajectory(
        data, bundle, static, phys_cfg=phys, bio_cfg=bio, device=dev, flow_source="kinematics"
    )

    kine_uv = _predict_uv_safe(kine, data, dev)
    kine_sh = _shear_proxy(kine_uv[:, 0], kine_uv[:, 1], ei, pos)

    rows: dict[str, dict] = {"none": {}, "kine_veto": {}, "gt_veto": {}, "tiled_zkin": {}}
    for t in eval_times:
        phi = phi_traj[int(t)]
        rows["none"][str(t)] = _score(phi, data, int(t), phys, dev)
        rows["kine_veto"][str(t)] = _best_veto(phi, kine_sh, data, int(t), phys, dev)
        gt_uv = data.y[int(t)].to(device=dev, dtype=torch.float32)
        gt_sh = _shear_proxy(gt_uv[:, 0], gt_uv[:, 1], ei, pos)
        rows["gt_veto"][str(t)] = _best_veto(phi, gt_sh, data, int(t), phys, dev)

    # config 4: occlusion z_kin refresh (guarded -- OOM -> nan, run still completes)
    n_refresh = -1
    try:
        species_occ = _species_series_occlusion_zkin(
            data, bundle, kine, phys, bio, dev, min_clot=min_clot, growth=growth, max_refresh=max_refresh
        )
        n_refresh = int(getattr(species_occ, "_n_zkin_refresh", -1))
        phi_occ = _phi_from_species(data, species_occ, phys, bio, dev)
        for t in eval_times:
            rows["tiled_zkin"][str(t)] = _score(phi_occ[int(t)], data, int(t), phys, dev)
    except (torch.cuda.OutOfMemoryError, NotImplementedError, RuntimeError) as exc:
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        for t in eval_times:
            rows["tiled_zkin"][str(t)] = {"f1": float("nan"), "prec": float("nan"),
                                          "rec": float("nan"), "guiding": float("nan"), "error": str(exc)[:120]}

    return {
        "anchor": anchor,
        "t_main": int(t_main),
        "n_steps": n_steps,
        "zkin_refresh": n_refresh,
        "modes": rows,
    }


def _fmt(d: dict, key: str) -> str:
    v = d.get(key)
    return f"{v:.3f}" if isinstance(v, (int, float)) and v == v else "  nan"


def main() -> int:
    ap = argparse.ArgumentParser(description="deploy clot-F1 flow-lever ladder (veto + occlusion z_kin)")
    ap.add_argument("--species-ckpt", required=True)
    ap.add_argument("--anchors", default="")
    ap.add_argument("--val-anchor", default="patient007")
    ap.add_argument("--times", default="53,200")
    ap.add_argument("--min-clot", type=int, default=30, help="min clot nodes before an occlusion re-solve")
    ap.add_argument("--growth", type=float, default=2.0, help="hysteresis: re-solve only after this growth factor")
    ap.add_argument("--max-refresh", type=int, default=3, help="cap occlusion z_kin re-solves per anchor (CPU-bound on small GPU)")
    ap.add_argument("--out", default="outputs/biochem/corrector_coupling/veto_zkin_ladder/ladder.json")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("[WARN] CUDA not available; running on CPU (slow).", flush=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _apply_deploy_env()

    root = get_project_root()
    bundle = load_species_gnn_rollout_bundle(Path(args.species_ckpt), device=dev)
    if bundle is None:
        print(f"[ERR] species ckpt not found: {args.species_ckpt}", flush=True)
        return 1
    kine = load_kinematics_predictor(resolve_kinematics_checkpoint(), dev)

    anchors = (
        [a.strip() for a in args.anchors.split(",") if a.strip()]
        or discover_biochem_anchors(root)
        or list(BIOCHEM_ANCHORS_6)
    )
    times = [int(x.strip()) for x in args.times.split(",") if x.strip()]

    print(f"[i] ckpt={args.species_ckpt}", flush=True)
    print(f"[i] anchors={anchors} times={times} min_clot={args.min_clot} growth={args.growth} max_refresh={args.max_refresh}", flush=True)
    t0 = time.perf_counter()
    results = []
    for anc in anchors:
        print(f"  eval {anc} ...", flush=True)
        r = eval_anchor(anc, bundle=bundle, kine=kine, dev=dev, times=times,
                        min_clot=args.min_clot, growth=args.growth, max_refresh=args.max_refresh)
        results.append(r)
        tm = str(r["t_main"])
        m = r["modes"]
        print(
            f"    t{tm}  none f1={_fmt(m['none'][tm],'f1')} (p={_fmt(m['none'][tm],'prec')} r={_fmt(m['none'][tm],'rec')})"
            f" | kine_veto f1={_fmt(m['kine_veto'][tm],'f1')}@{m['kine_veto'][tm].get('percentile')}"
            f" | gt_veto f1={_fmt(m['gt_veto'][tm],'f1')}@{m['gt_veto'][tm].get('percentile')}"
            f" | tiled_zkin f1={_fmt(m['tiled_zkin'][tm],'f1')} (refresh={r['zkin_refresh']})",
            flush=True,
        )
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    # summary table (main time per anchor) + holdout means
    print("\n==================== DEPLOY CLOT-F1 FLOW-LEVER LADDER ====================", flush=True)
    hdr = f"  {'anchor':<12}{'none':>9}{'kine_veto':>11}{'gt_veto':>9}{'tiled_zkin':>12}"
    print(hdr, flush=True)
    print("  " + "-" * (len(hdr) - 2), flush=True)
    mode_keys = ["none", "kine_veto", "gt_veto", "tiled_zkin"]
    sums = {k: [] for k in mode_keys}
    hold = {k: [] for k in mode_keys}
    for r in results:
        tm = str(r["t_main"])
        line = f"  {r['anchor']:<12}"
        for k in mode_keys:
            f1 = r["modes"][k][tm].get("f1", float("nan"))
            line += f"{(f1 if f1 == f1 else float('nan')):>{11 if k=='kine_veto' else (12 if k=='tiled_zkin' else 9)}.3f}"
            if f1 == f1:
                sums[k].append(f1)
                if r["anchor"] != args.val_anchor:
                    hold[k].append(f1)
        print(line, flush=True)
    print("  " + "-" * (len(hdr) - 2), flush=True)

    def _mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    print("\n  p007 (val) / holdout-mean per mode:", flush=True)
    for k in mode_keys:
        p007 = next((r["modes"][k][str(r["t_main"])].get("f1") for r in results if r["anchor"] == args.val_anchor), None)
        p007 = p007 if (p007 is not None and p007 == p007) else float("nan")
        print(f"    {k:<12} p007={p007:>6.3f}  holdout={_mean(hold[k]):>6.3f}", flush=True)

    print("\n  READ:", flush=True)
    print("    gt_veto >> none but kine_veto ~ none  -> veto works, deployable shear too weak (flow fidelity).", flush=True)
    print("    gt_veto ~ none                        -> the shear veto itself is empty (precision lever not here).", flush=True)
    print("    tiled_zkin > none                     -> occlusion-coupled z_kin is the live lever.", flush=True)
    print("=========================================================================", flush=True)

    out = Path(args.out)
    if not out.is_absolute():
        out = root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "species_ckpt": args.species_ckpt,
        "anchors": anchors,
        "val_anchor": args.val_anchor,
        "times": times,
        "min_clot": args.min_clot,
        "growth": args.growth,
        "max_refresh": args.max_refresh,
        "results": results,
        "elapsed_s": time.perf_counter() - t0,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n[OK] wrote {out} ({payload['elapsed_s']:.1f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
