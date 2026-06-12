"""Train Phase 2 species pushforward GNN (growth residual + unrolled closed loop).

Usage::

    python -m src.training.train_species_snapshot_pushforward
    python -m src.training.train_species_snapshot_pushforward --anchor patient007 --epochs 120 --unroll 5
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import torch
import torch.optim as optim

from src.config import PhysicsConfig, VesselConfig
from src.core_physics.clot_phi_simple import sdf_nd_from_data
from src.core_physics.species_pushforward_gnn import (
    DEFAULT_PUSHFORWARD_CKPT,
    SpeciesPushforwardGNN,
    active_series_on_band,
    eval_pushforward_window,
    filter_pushforward_windows,
    init_pushforward_from_snapshot,
    iter_pushforward_windows,
    pushforward_channel_weights,
    pushforward_feature_dim,
    pushforward_focal_alpha_channels,
    pushforward_step_stride,
    pushforward_train_t0_max,
    pushforward_unroll_steps,
    pushforward_val_score_weights,
    save_pushforward_checkpoint,
    snapshot_hidden_dim,
    snapshot_wall_hops,
    unroll_pushforward_loss,
    wall_band_mask,
)
from src.core_physics.species_snapshot_gnn import DEFAULT_SNAPSHOT_CKPT
from src.core_physics.species_snapshot_gnn import (
    build_snapshot_features,
    induced_subgraph,
)
from src.utils.kinematics_inference import (
    load_kinematics_predictor,
    predict_kinematics_latent,
    resolve_kinematics_checkpoint,
)
from src.utils.paths import get_project_root


def _split_band_nodes(n_sub: int, val_frac: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_sub, generator=g)
    n_val = max(1, int(round(n_sub * val_frac)))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    if train_idx.numel() == 0:
        train_idx = val_idx
    train_m = torch.zeros(n_sub, dtype=torch.bool)
    val_m = torch.zeros(n_sub, dtype=torch.bool)
    train_m[train_idx] = True
    val_m[val_idx] = True
    return train_m, val_m


@torch.no_grad()
def _prepare_static(
    data,
    *,
    device: torch.device,
    kine_model,
    wall_hops: int,
) -> dict:
    data = data.to(device)
    n = int(data.num_nodes)
    band = wall_band_mask(data, device, wall_hops=wall_hops)
    node_idx, edge_sub, _ = induced_subgraph(band, data.edge_index)
    z_kin = predict_kinematics_latent(kine_model, data)
    sdf = sdf_nd_from_data(data, device, n)
    base_feats = build_snapshot_features(z_kin, sdf)[node_idx]
    return {
        "base_feats": base_feats,
        "edge_index": edge_sub,
        "node_idx": node_idx,
        "n_band": int(node_idx.numel()),
        "n_full": n,
        "n_times": int(data.y.shape[0]),
    }


def _val_windows(static: dict, *, unroll: int, stride: int) -> list[list[int]]:
    """Anchor eval windows: active growth + late plateau."""
    anchors = [10, 28]
    wins: list[list[int]] = []
    n_times = int(static["n_times"])
    for t0 in anchors:
        win = [t0 + i * stride for i in range(unroll + 1)]
        if win[-1] < n_times:
            wins.append(win)
    return wins


def _eval_window(
    model,
    static: dict,
    data,
    window: list[int],
    *,
    device: torch.device,
    val_mask: torch.Tensor,
) -> dict[str, float]:
    series = active_series_on_band(data, window, device, static["node_idx"])
    m = eval_pushforward_window(
        model,
        base_feats=static["base_feats"],
        edge_index=static["edge_index"],
        active_series=series,
        mask=val_mask,
        state0=series[0],
    )
    return {
        "growth_f1": float(m["mean_growth_f1"]),
        "growth_mat_f1": float(m["mean_growth_mat_f1"]),
        "state_f1": float(m["final_state_f1"]),
        "state_mat_f1": float(m["final_state_mat_f1"]),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Train species pushforward GNN (phase 2)")
    ap.add_argument("--anchor", default="patient007")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--unroll", type=int, default=None)
    ap.add_argument("--stride", type=int, default=None)
    ap.add_argument("--wall-hops", type=int, default=None)
    ap.add_argument("--hidden", type=int, default=None)
    ap.add_argument("--kine-ckpt", default="")
    ap.add_argument("--init-s1", default="", help="Warm-start from phase-1 snapshot ckpt")
    ap.add_argument("--out", default=DEFAULT_PUSHFORWARD_CKPT)
    ap.add_argument("--early-stop", type=int, default=20)
    ap.add_argument("--max-windows", type=int, default=0, help="0 = all valid windows")
    args = ap.parse_args()

    if args.unroll is not None:
        os.environ["SPECIES_PUSHFORWARD_UNROLL"] = str(args.unroll)
    if args.stride is not None:
        os.environ["SPECIES_PUSHFORWARD_STEP_STRIDE"] = str(args.stride)
    if args.wall_hops is not None:
        os.environ["SPECIES_SNAPSHOT_WALL_HOPS"] = str(args.wall_hops)

    unroll = pushforward_unroll_steps()
    stride = pushforward_step_stride()
    wall_hops = snapshot_wall_hops()
    hidden = snapshot_hidden_dim() if args.hidden is None else max(int(args.hidden), 16)
    focal_alpha = pushforward_focal_alpha_channels()
    ch_w = pushforward_channel_weights()
    score_gw, score_sw = pushforward_val_score_weights()
    t0_max = pushforward_train_t0_max()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = get_project_root()
    graph_path = root / VesselConfig(phase="biochem_anchors").graph_output_dir / f"{args.anchor.strip()}.pt"
    data = torch.load(graph_path, map_location="cpu", weights_only=False)

    kine_ckpt = args.kine_ckpt.strip() or str(resolve_kinematics_checkpoint())
    kine_model = load_kinematics_predictor(
        kine_ckpt, device, phys_cfg=PhysicsConfig(phase="kinematics")
    )

    static = _prepare_static(data, device=device, kine_model=kine_model, wall_hops=wall_hops)
    train_m, val_m = _split_band_nodes(static["n_band"], args.val_frac, args.seed)

    windows = iter_pushforward_windows(static["n_times"], unroll=unroll, stride=stride)
    windows = filter_pushforward_windows(
        windows,
        data,
        static["node_idx"],
        device,
        t0_max=t0_max,
        min_growth_nodes=1,
    )
    if args.max_windows > 0:
        windows = windows[: int(args.max_windows)]
    val_windows = _val_windows(static, unroll=unroll, stride=stride)

    latent_dim = int(static["base_feats"].shape[1] - 1)
    in_dim = pushforward_feature_dim(latent_dim)
    model = SpeciesPushforwardGNN(in_dim, hidden=hidden).to(device)
    init_path = args.init_s1.strip() or str(root / DEFAULT_SNAPSHOT_CKPT)
    if Path(init_path).is_file():
        init_pushforward_from_snapshot(model, init_path)

    print(
        f"[i] anchor={args.anchor} unroll={unroll} stride={stride} windows={len(windows)} "
        f"band={static['n_band']} focal_alpha=({focal_alpha[0]:.2f},{focal_alpha[1]:.2f}) "
        f"ch_w=({ch_w[0]:.1f},{ch_w[1]:.1f}) t0_max={t0_max}",
        flush=True,
    )

    opt = optim.Adam(model.parameters(), lr=float(args.lr), weight_decay=1e-5)
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = out_path.parent / "train_log.jsonl"

    meta_base = {
        "anchor": args.anchor,
        "phase": "s2_pushforward",
        "unroll": unroll,
        "stride": stride,
        "wall_hops": wall_hops,
        "latent_dim": latent_dim,
        "hidden": hidden,
        "kine_ckpt": kine_ckpt,
        "n_band": static["n_band"],
        "n_windows": len(windows),
        "focal_alpha_fi": focal_alpha[0],
        "focal_alpha_mat": focal_alpha[1],
        "channel_weight_fi": ch_w[0],
        "channel_weight_mat": ch_w[1],
        "train_t0_max": t0_max,
        "score_growth_w": score_gw,
        "score_state_w": score_sw,
    }

    best_score = -1.0
    stale = 0
    t0 = time.perf_counter()

    for ep in range(1, int(args.epochs) + 1):
        model.train()
        random.shuffle(windows)
        ep_losses: list[float] = []

        for win in windows:
            series = active_series_on_band(data, win, device, static["node_idx"])
            loss, _, _ = unroll_pushforward_loss(
                model,
                base_feats=static["base_feats"],
                edge_index=static["edge_index"],
                active_series=series,
                train_mask=train_m,
                state0=series[0],
                focal_alpha=focal_alpha,
                training=True,
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            ep_losses.append(float(loss.item()))

        model.eval()
        val_growth_f1: list[float] = []
        val_state_f1: list[float] = []
        with torch.no_grad():
            for win in val_windows:
                m = _eval_window(model, static, data.to(device), win, device=device, val_mask=val_m)
                val_growth_f1.append(m["growth_f1"])
                val_state_f1.append(m["state_f1"])

        row = {
            "epoch": ep,
            "loss": sum(ep_losses) / max(len(ep_losses), 1),
            "val_growth_f1": sum(val_growth_f1) / max(len(val_growth_f1), 1),
            "val_state_f1": sum(val_state_f1) / max(len(val_state_f1), 1),
        }
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

        print(
            f"[ep {ep:03d}] loss={row['loss']:.4f} "
            f"val_growth_f1={row['val_growth_f1']:.3f} val_state_f1={row['val_state_f1']:.3f}",
            flush=True,
        )

        score = score_gw * row["val_growth_f1"] + score_sw * row["val_state_f1"]
        if score > best_score:
            best_score = score
            stale = 0
            meta = {**meta_base, "best_score": best_score, "best_epoch": ep, **row}
            save_pushforward_checkpoint(out_path, model, meta)
        else:
            stale += 1
            if stale >= int(args.early_stop):
                print(f"[i] early stop @ ep {ep} (best_score={best_score:.3f})", flush=True)
                break

    print(f"[OK] best_score={best_score:.3f} elapsed={time.perf_counter() - t0:.1f}s ckpt={out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
