"""Train Phase 1 species snapshot GNN (static FI/Mat at one macro time).

Usage::

    python -m src.training.train_species_snapshot_gnn
    python -m src.training.train_species_snapshot_gnn --anchor patient007 --time-s 5000 --epochs 80
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.optim as optim

from src.config import PhysicsConfig, VesselConfig
from src.core_physics.species_snapshot_gnn import (
    DEFAULT_SNAPSHOT_CKPT,
    SpeciesSnapshotGNN,
    build_snapshot_features,
    fi_mat_active_labels,
    fi_mat_log_targets,
    induced_subgraph,
    resolve_time_index,
    save_snapshot_checkpoint,
    snapshot_active_log_nd,
    snapshot_channel_weights,
    snapshot_focal_alpha_channels,
    snapshot_focal_gamma_channels,
    logits_to_probs,
    snapshot_feature_dim,
    snapshot_hidden_dim,
    snapshot_loss,
    snapshot_loss_mode,
    snapshot_wall_hops,
    trigger_metrics,
    wall_band_mask,
)
from src.core_physics.clot_phi_simple import sdf_nd_from_data
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
def _prepare_batch(
    data,
    *,
    device: torch.device,
    kine_model,
    time_index: int,
    wall_hops: int,
) -> dict:
    data = data.to(device)
    n = int(data.num_nodes)
    band = wall_band_mask(data, device, wall_hops=wall_hops)
    node_idx, edge_sub, _ = induced_subgraph(band, data.edge_index)
    z_kin = predict_kinematics_latent(kine_model, data)
    sdf = sdf_nd_from_data(data, device, n)
    feats_full = build_snapshot_features(z_kin, sdf)
    feats = feats_full[node_idx]
    tgt_log = fi_mat_log_targets(data, time_index, device)[node_idx]
    tgt_act = fi_mat_active_labels(tgt_log)
    return {
        "feats": feats,
        "edge_index": edge_sub,
        "tgt_log": tgt_log,
        "tgt_active": tgt_act,
        "n_band": int(node_idx.numel()),
        "n_full": n,
    }


def _pred_probs(logits: torch.Tensor, loss_mode: str) -> torch.Tensor:
    return logits_to_probs(logits, loss_mode=loss_mode)


def main() -> int:
    ap = argparse.ArgumentParser(description="Train species snapshot GNN (phase 1)")
    ap.add_argument("--anchor", default="patient007")
    ap.add_argument("--time-s", type=float, default=None, help="Physical time in seconds (default 5000)")
    ap.add_argument("--time-index", type=int, default=None, help="Override macro step index")
    ap.add_argument("--wall-hops", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--loss", choices=("focal", "bce", "mse"), default=None)
    ap.add_argument("--pos-weight", type=float, default=5.0)
    ap.add_argument("--hidden", type=int, default=None)
    ap.add_argument("--kine-ckpt", default="")
    ap.add_argument("--out", default=DEFAULT_SNAPSHOT_CKPT)
    ap.add_argument("--early-stop", type=int, default=15)
    args = ap.parse_args()

    if args.loss:
        os.environ["SPECIES_SNAPSHOT_LOSS"] = args.loss
    if args.hidden is not None:
        os.environ["SPECIES_SNAPSHOT_HIDDEN"] = str(args.hidden)
    if args.wall_hops is not None:
        os.environ["SPECIES_SNAPSHOT_WALL_HOPS"] = str(args.wall_hops)
    if args.time_s is not None:
        os.environ["SPECIES_SNAPSHOT_TIME_S"] = str(args.time_s)

    loss_mode = snapshot_loss_mode()
    wall_hops = snapshot_wall_hops()
    hidden = snapshot_hidden_dim() if args.hidden is None else max(int(args.hidden), 16)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = get_project_root()
    graph_dir = root / VesselConfig(phase="biochem_anchors").graph_output_dir
    graph_path = graph_dir / f"{args.anchor.strip()}.pt"
    if not graph_path.is_file():
        raise FileNotFoundError(graph_path)

    data = torch.load(graph_path, map_location="cpu", weights_only=False)
    t_idx = int(args.time_index) if args.time_index is not None else resolve_time_index(data, time_s=args.time_s)
    time_s = float(args.time_s) if args.time_s is not None else float(
        data.t[t_idx].item() if hasattr(data, "t") and data.t is not None else t_idx
    )

    kine_ckpt = args.kine_ckpt.strip() or str(resolve_kinematics_checkpoint())
    kine_model = load_kinematics_predictor(
        kine_ckpt, device, phys_cfg=PhysicsConfig(phase="kinematics")
    )

    print(
        f"[i] anchor={args.anchor} time_index={t_idx} time_s={time_s:.1f} "
        f"wall_hops={wall_hops} loss={loss_mode} device={device}",
        flush=True,
    )

    batch = _prepare_batch(
        data, device=device, kine_model=kine_model, time_index=t_idx, wall_hops=wall_hops
    )
    print(f"[i] wall_band_nodes={batch['n_band']} / {batch['n_full']}", flush=True)

    band_pos = int((batch["tgt_active"].max(dim=-1).values > 0.5).sum().item())
    band_neg = int(batch["n_band"]) - band_pos
    focal_alpha = snapshot_focal_alpha_channels(n_neg=band_neg, n_pos=max(band_pos, 1))
    focal_gamma = snapshot_focal_gamma_channels()
    channel_weight = snapshot_channel_weights()
    imb_ratio = band_neg / max(band_pos, 1)
    print(
        f"[i] band_trigger_pos={band_pos} neg={band_neg} neg:pos={imb_ratio:.1f} "
        f"focal_alpha_fi={focal_alpha[0]:.3f} focal_alpha_mat={focal_alpha[1]:.3f} "
        f"focal_gamma=({focal_gamma[0]:.1f},{focal_gamma[1]:.1f}) "
        f"ch_w=({channel_weight[0]:.2f},{channel_weight[1]:.2f})",
        flush=True,
    )

    train_m, val_m = _split_band_nodes(batch["n_band"], args.val_frac, args.seed)
    latent_dim = int(batch["feats"].shape[1] - 1)
    in_dim = snapshot_feature_dim(latent_dim)
    model = SpeciesSnapshotGNN(in_dim, hidden=hidden).to(device)
    opt = optim.Adam(model.parameters(), lr=float(args.lr), weight_decay=1e-5)

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = out_path.parent / "train_log.jsonl"

    meta_base = {
        "anchor": args.anchor,
        "time_index": t_idx,
        "time_s": time_s,
        "wall_hops": wall_hops,
        "loss_mode": loss_mode,
        "active_log_nd": snapshot_active_log_nd(),
        "latent_dim": latent_dim,
        "hidden": hidden,
        "kine_ckpt": kine_ckpt,
        "n_band": batch["n_band"],
        "band_trigger_pos": band_pos,
        "band_trigger_neg": band_neg,
        "focal_alpha_fi": focal_alpha[0],
        "focal_alpha_mat": focal_alpha[1],
        "focal_gamma_fi": focal_gamma[0],
        "focal_gamma_mat": focal_gamma[1],
        "channel_weight_fi": channel_weight[0],
        "channel_weight_mat": channel_weight[1],
    }

    best_f1 = -1.0
    stale = 0
    t0 = time.perf_counter()

    for ep in range(1, int(args.epochs) + 1):
        model.train()
        logits = model(batch["feats"], batch["edge_index"])
        loss = snapshot_loss(
            logits,
            batch["tgt_log"],
            batch["tgt_active"],
            train_m,
            loss_mode=loss_mode,
            pos_weight=float(args.pos_weight),
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
            channel_weight=channel_weight,
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            logits_v = model(batch["feats"], batch["edge_index"])
            pred_v = _pred_probs(logits_v, loss_mode)
            val_metrics = trigger_metrics(pred_v, batch["tgt_active"], val_m)
            train_metrics = trigger_metrics(pred_v, batch["tgt_active"], train_m)

        row = {
            "epoch": ep,
            "loss": float(loss.item()),
            "val_trigger_f1": val_metrics["trigger_f1"],
            "val_trigger_rec": val_metrics["trigger_rec"],
            "val_trigger_prec": val_metrics["trigger_prec"],
            "train_trigger_f1": train_metrics["trigger_f1"],
        }
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

        print(
            f"[ep {ep:03d}] loss={row['loss']:.4f} "
            f"val_f1={row['val_trigger_f1']:.3f} rec={row['val_trigger_rec']:.3f} "
            f"prec={row['val_trigger_prec']:.3f}",
            flush=True,
        )

        if row["val_trigger_f1"] > best_f1:
            best_f1 = row["val_trigger_f1"]
            stale = 0
            meta = {**meta_base, "best_val_trigger_f1": best_f1, "best_epoch": ep}
            save_snapshot_checkpoint(out_path, model, meta)
        else:
            stale += 1
            if stale >= int(args.early_stop):
                print(f"[i] early stop @ ep {ep} (best_f1={best_f1:.3f})", flush=True)
                break

    elapsed = time.perf_counter() - t0
    print(f"[OK] best_val_trigger_f1={best_f1:.3f} elapsed={elapsed:.1f}s ckpt={out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
