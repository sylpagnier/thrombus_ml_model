"""Train wall-local clot phase (phi) with capped GT mu and GT kinematics.

Usage (from repo root)::

    python -m src.training.train_clot_phi_simple

Checkpoint: ``outputs/biochem/clot_phi_best.pth``
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from src.config import BiochemConfig, PhysicsConfig, VesselConfig
from src.core_physics.clot_phi_simple import (
    ClotPhiSpeciesHead,
    build_clot_phi_model,
    build_clot_phi_step,
    cap_mu_eff_si,
    clot_phi_dropout,
    clot_phi_feature_dim,
    clot_phi_mlp_depth,
    clot_phi_hybrid_enabled,
    clot_phi_joint_bio_enabled,
    clot_phi_mask_mode,
    clot_phi_minimal_features_enabled,
    clot_phi_model_kind,
    clot_phi_mu_cap_si,
    clot_phi_oracle_mu_enabled,
    clot_phi_physics_oracle_enabled,
    clot_phi_prior_feature_count,
    clot_phi_species_features_enabled,
    clot_phi_thresh_si,
    clot_phi_use_prior_features,
    log_blend_mu_eff_si,
    mu_eff_from_delta_log_si,
    physics_mu_eff_si,
    physics_phi_from_mu,
    rule_phi_from_mu_cap,
)
from src.utils.channel_schema import BIO_Y_SCHEMA, assert_graph_schema, infer_missing_schema
from src.utils.paths import get_project_root


class AnchorFileDataset(Dataset):
    def __init__(self, file_list: List[str]):
        self.file_list = list(file_list)

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int):
        data = torch.load(self.file_list[idx], weights_only=False)
        data = infer_missing_schema(data, phase_hint="biochem")
        assert_graph_schema(data, expected_y_schema=(BIO_Y_SCHEMA,))
        return data


def _list_anchor_paths(root: Path) -> List[str]:
    paths = sorted(str(p) for p in root.glob("*.pt") if p.is_file())
    if not paths:
        raise FileNotFoundError(f"No anchor graphs in {root}")
    return paths


def _split_train_val(paths: List[str], val_stem: str) -> Tuple[List[str], List[str]]:
    val_stem = val_stem.strip().lower()
    val_paths = [p for p in paths if Path(p).stem.lower() == val_stem]
    train_paths = [p for p in paths if Path(p).stem.lower() != val_stem]
    if not val_paths:
        split = max(1, len(paths) // 5)
        return paths[split:], paths[:split]
    if not train_paths:
        train_paths = paths[:]
    return train_paths, val_paths


def _env_bool(name: str, default: bool) -> bool:
    raw = (os.environ.get(name) or "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def _loss_indices(
    pred: torch.Tensor,
    tgt: torch.Tensor,
    mask: torch.Tensor,
    *,
    balanced: bool,
) -> torch.Tensor:
    """Node indices for BCE; optional 1:1 pos/neg subsample inside the supervision mask."""
    idx = mask.nonzero(as_tuple=False).view(-1)
    if not balanced or idx.numel() == 0:
        return idx
    pos = idx[(tgt[idx] > 0.5)]
    neg = idx[(tgt[idx] <= 0.5)]
    if pos.numel() == 0 or neg.numel() == 0:
        return idx
    k = int(min(pos.numel(), neg.numel()))
    pos_pick = pos[torch.randperm(pos.numel(), device=pos.device)[:k]]
    neg_pick = neg[torch.randperm(neg.numel(), device=neg.device)[:k]]
    return torch.cat([pos_pick, neg_pick])


def _dice_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    pb = (pred.reshape(-1) > 0.5).float()
    tb = (target.reshape(-1) > 0.5).float()
    inter = float((pb * tb).sum().item())
    return (2.0 * inter + eps) / (float(pb.sum()) + float(tb.sum()) + eps)


def _clot_metrics(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    """Precision/recall/F1 and positive rate inside the supervision mask."""
    if not bool(mask.any().item()):
        return {
            "clot_prec": 0.0,
            "clot_rec": 0.0,
            "clot_f1": 0.0,
            "pred_pos_frac": 0.0,
            "gt_pos_frac": 0.0,
        }
    pb = (pred[mask] > 0.5).float()
    tb = (target[mask] > 0.5).float()
    tp = float((pb * tb).sum().item())
    fp = float((pb * (1.0 - tb)).sum().item())
    fn = float(((1.0 - pb) * tb).sum().item())
    prec = tp / max(tp + fp, 1e-6)
    rec = tp / max(tp + fn, 1e-6)
    f1 = (2.0 * prec * rec) / max(prec + rec, 1e-6)
    return {
        "clot_prec": prec,
        "clot_rec": rec,
        "clot_f1": f1,
        "pred_pos_frac": float(pb.mean().item()),
        "gt_pos_frac": float(tb.mean().item()),
    }


def _checkpoint_score(va: dict[str, float]) -> float:
    """Prefer non-collapsed val F1; penalize predict-none / predict-all / memorization."""
    f1 = float(va.get("clot_f1", 0.0))
    pp = float(va.get("pred_pos_frac", 0.0))
    gt = float(va.get("gt_pos_frac", 0.0))
    rec = float(va.get("clot_rec", 0.0))
    prec = float(va.get("clot_prec", 0.0))
    mae = float(va.get("mu_log_mae", 99.0))
    if pp < 0.05 or pp > 0.92:
        return -1.0
    if rec > 0.92:
        return -1.0
    if gt > 1e-6 and (pp < 0.35 * gt or pp > 2.5 * gt):
        return f1 * 0.35
    if rec > 0.72 and prec < 0.45:
        return f1 * 0.4
    mae_bonus = max(0.0, 1.2 - mae) * 0.15
    return f1 + mae_bonus


def _bio_lambda() -> float:
    return max(float(os.environ.get("CLOT_PHI_BIO_LAMBDA", "1.0") or "0"), 0.0)


def _physics_blend_alpha() -> float:
    return max(0.0, min(float(os.environ.get("CLOT_PHI_PHYSICS_BLEND_ALPHA", "0.5") or "0.5"), 1.0))


def _species_hidden() -> int:
    return max(int(os.environ.get("CLOT_PHI_SPECIES_HIDDEN", "32")), 8)


def _species_data_mse(
    pred_log: torch.Tensor,
    tgt_log: torch.Tensor,
    idx: torch.Tensor,
    bio_cfg: BiochemConfig,
) -> torch.Tensor:
    """SI-scaled species fit (same spirit as biochem ``L_Data_Bio`` on anchors)."""
    scales = bio_cfg.get_species_scales(device=pred_log.device)[:12].view(1, -1)
    p_log = pred_log[idx].clamp(-10.0, 8.0)
    t_log = tgt_log[idx].clamp(-10.0, 8.0)
    # Log1p-ND MSE (stable); SI MSE can explode when the head is poorly initialized.
    return F.mse_loss(p_log, t_log)


def _run_epoch(
    model: torch.nn.Module | None,
    paths: List[str],
    *,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    device: torch.device,
    train: bool,
    time_stride: int,
    pos_weight: float,
    balanced: bool,
    rule_baseline: bool = False,
    physics_oracle: bool = False,
    species_head: torch.nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    if train:
        if model is None or rule_baseline or physics_oracle:
            raise ValueError("rule/physics oracle cannot train")
        model.train()
        if species_head is not None:
            species_head.train()
    elif model is not None:
        model.eval()
        if species_head is not None:
            species_head.eval()

    bce_sum = 0.0
    mu_mse_sum = 0.0
    bio_mse_sum = 0.0
    dice_sum = 0.0
    log_mae_sum = 0.0
    hybrid = clot_phi_hybrid_enabled()
    mu_log_lambda = max(float(os.environ.get("CLOT_PHI_MU_LOG_LAMBDA", "1.0") or "0"), 0.0)
    bio_lambda = _bio_lambda()
    joint_bio = species_head is not None
    use_soft = _env_bool("CLOT_PHI_SOFT_LABELS", False)
    metric_sums = {
        "clot_prec": 0.0,
        "clot_rec": 0.0,
        "clot_f1": 0.0,
        "pred_pos_frac": 0.0,
        "gt_pos_frac": 0.0,
    }
    n_steps = 0
    n_graphs = 0
    dice_lambda = max(float(os.environ.get("CLOT_PHI_DICE_LAMBDA", "0") or "0"), 0.0)

    with torch.set_grad_enabled(train):
        for path in paths:
            data = torch.load(path, weights_only=False).to(device)
            data = infer_missing_schema(data, phase_hint="biochem")
            assert_graph_schema(data, expected_y_schema=(BIO_Y_SCHEMA,))
            if not hasattr(data, "y") or data.y.dim() != 3:
                continue
            n_graphs += 1
            t_steps = data.y.shape[0]
            for ti in range(0, t_steps, max(1, time_stride)):
                step = build_clot_phi_step(data, ti, phys_cfg, bio_cfg, device)
                m = step.loss_mask
                if not bool(m.any().item()):
                    continue
                tgt = step.phi_gt
                if rule_baseline:
                    phi_pred = rule_phi_from_mu_cap(step.mu_gt_cap, step.region, phys_cfg)
                    mu_pred = log_blend_mu_eff_si(step.mu_c_si, phi_pred)
                    idx = m.nonzero(as_tuple=False).view(-1)
                    pm = phi_pred[idx].clamp(1e-6, 1.0 - 1e-6)
                    tm = tgt[idx]
                    bce = F.binary_cross_entropy(pm, tm)
                    mu_mse = torch.tensor(0.0, device=device)
                    bio_mse = torch.tensor(0.0, device=device)
                elif physics_oracle:
                    y_sl = data.y[ti].to(device=device)
                    mu_pred = cap_mu_eff_si(
                        physics_mu_eff_si(
                            step.mu_c_si,
                            step.species_log_gt,
                            bio_cfg,
                            device=device,
                            data=data,
                            u_nd=y_sl[:, 0],
                            v_nd=y_sl[:, 1],
                        )
                    )
                    phi_pred = physics_phi_from_mu(
                        mu_pred, step.mu_c_si, step.region, phys_cfg, soft=use_soft
                    )
                    idx = m.nonzero(as_tuple=False).view(-1)
                    pm = phi_pred[idx].clamp(1e-6, 1.0 - 1e-6)
                    tm = tgt[idx]
                    bce = F.binary_cross_entropy(pm, tm)
                    mu_mse = F.mse_loss(
                        torch.log(mu_pred[idx].clamp(min=1e-8)),
                        torch.log(step.mu_gt_cap[idx].clamp(min=1e-8)),
                    )
                    bio_mse = torch.tensor(0.0, device=device)
                else:
                    assert model is not None
                    if train:
                        if optimizer is None:
                            raise ValueError("optimizer is required when train=True")
                        optimizer.zero_grad(set_to_none=True)
                    feats = step.features
                    bio_mse = torch.tensor(0.0, device=device)
                    sp_for_physics = step.species_log_gt
                    if joint_bio:
                        assert species_head is not None
                        sp_pred = species_head(feats).clamp(-10.0, 8.0)
                        idx_all = m.nonzero(as_tuple=False).view(-1)
                        bio_mse = _species_data_mse(
                            sp_pred, step.species_log_gt, idx_all, bio_cfg
                        )
                        if _env_bool("CLOT_PHI_JOINT_USE_PRED_SPECIES", True):
                            sp_for_physics = sp_pred
                    y_sl = data.y[ti].to(device=device)
                    u_sl, v_sl = y_sl[:, 0], y_sl[:, 1]
                    logits = model.forward_logits(feats)
                    idx = _loss_indices(logits, tgt, m, balanced=balanced and train)
                    tm = tgt[idx]
                    pw = torch.tensor([pos_weight], device=device, dtype=logits.dtype)
                    blend = _env_bool("CLOT_PHI_PHYSICS_BLEND", False)
                    alpha = _physics_blend_alpha()
                    phi_ml = torch.sigmoid(logits)
                    mu_mse = torch.tensor(0.0, device=device)
                    if hybrid:
                        dlog = model.forward_delta_log_mu(feats)
                        mu_ml = mu_eff_from_delta_log_si(step.mu_c_si, dlog)
                    else:
                        mu_ml = log_blend_mu_eff_si(step.mu_c_si, phi_ml)
                    if blend:
                        mu_phys = cap_mu_eff_si(
                            physics_mu_eff_si(
                                step.mu_c_si,
                                sp_for_physics,
                                bio_cfg,
                                device=device,
                                data=data,
                                u_nd=u_sl,
                                v_nd=v_sl,
                            )
                        )
                        phi_phys = physics_phi_from_mu(
                            mu_phys, step.mu_c_si, step.region, phys_cfg, soft=use_soft
                        )
                        phi_mix = (alpha * phi_ml + (1.0 - alpha) * phi_phys).clamp(1e-6, 1.0 - 1e-6)
                        pm = phi_mix[idx]
                        bce_n = F.binary_cross_entropy(pm, tm, reduction="none")
                        w = torch.where(tm > 0.5, pw, torch.ones_like(tm))
                        bce = (bce_n * w).mean()
                        if mu_log_lambda > 0.0:
                            log_mu_p = (
                                alpha * torch.log(mu_ml.clamp(min=1e-8))
                                + (1.0 - alpha) * torch.log(mu_phys.clamp(min=1e-8))
                            )
                            log_mu_t = torch.log(step.mu_gt_cap.clamp(min=1e-8))
                            mu_mse = F.mse_loss(log_mu_p[idx], log_mu_t[idx])
                        phi_pred = phi_mix
                        mu_pred = alpha * mu_ml + (1.0 - alpha) * mu_phys
                    else:
                        lm = logits[idx]
                        bce = F.binary_cross_entropy_with_logits(lm, tm, pos_weight=pw)
                        if hybrid and mu_log_lambda > 0.0:
                            log_mu_p = torch.log(step.mu_c_si.clamp(min=1e-8)) + dlog
                            log_mu_t = torch.log(step.mu_gt_cap.clamp(min=1e-8))
                            mu_mse = F.mse_loss(log_mu_p[idx], log_mu_t[idx])
                        phi_pred = phi_ml
                        mu_pred = mu_ml
                    if train and dice_lambda > 0.0:
                        pm = phi_pred[idx].clamp(1e-6, 1.0 - 1e-6)
                        inter = (pm * tm).sum()
                        dice_loss = 1.0 - (2.0 * inter + 1e-6) / (pm.sum() + tm.sum() + 1e-6)
                        loss = bce + dice_lambda * dice_loss + mu_log_lambda * mu_mse + bio_lambda * bio_mse
                    else:
                        loss = bce + mu_log_lambda * mu_mse + bio_lambda * bio_mse
                    if train:
                        loss.backward()
                        optimizer.step()
                with torch.no_grad():
                    log_mae = (torch.log(mu_pred[m].clamp(min=1e-8)) - torch.log(step.mu_gt_cap[m].clamp(min=1e-8))).abs().mean()
                    dice_sum += _dice_score(phi_pred[m], tgt[m])
                    log_mae_sum += float(log_mae.item())
                    bce_sum += float(bce.item())
                    mu_mse_sum += float(mu_mse.item())
                    bio_mse_sum += float(bio_mse.item())
                    for k, v in _clot_metrics(phi_pred, tgt, m).items():
                        metric_sums[k] += v
                    n_steps += 1

    denom = max(n_steps, 1)
    out = {
        "bce": bce_sum / denom,
        "mu_log_mse": mu_mse_sum / denom,
        "bio_mse": bio_mse_sum / denom,
        "dice": dice_sum / denom,
        "mu_log_mae": log_mae_sum / denom,
        "n_steps": float(n_steps),
        "n_graphs": float(n_graphs),
    }
    out.update({k: v / denom for k, v in metric_sums.items()})
    return out


def main() -> None:
    root = get_project_root()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    phys_cfg = PhysicsConfig(phase="biochem")
    bio_cfg = BiochemConfig(phase="biochem")

    raw_dir = (os.environ.get("CLOT_PHI_ANCHOR_DIR") or "").strip()
    if raw_dir:
        anchor_dir = Path(raw_dir).expanduser()
        if not anchor_dir.is_absolute():
            anchor_dir = root / anchor_dir
    else:
        anchor_dir = root / VesselConfig(phase="biochem_anchors").graph_output_dir
    anchor_dir = anchor_dir.resolve()
    paths = _list_anchor_paths(anchor_dir)
    val_stem = (os.environ.get("CLOT_PHI_VAL_ANCHOR") or "patient007").strip()
    train_paths, val_paths = _split_train_val(paths, val_stem)

    epochs = max(int(os.environ.get("CLOT_PHI_EPOCHS", "40")), 1)
    lr = float(os.environ.get("CLOT_PHI_LR", "3e-3"))
    time_stride = max(int(os.environ.get("CLOT_PHI_TIME_STRIDE", "2")), 1)
    hidden = max(int(os.environ.get("CLOT_PHI_HIDDEN", "64")), 4)
    rule_baseline = _env_bool("CLOT_PHI_RULE_BASELINE", False)
    physics_oracle = clot_phi_physics_oracle_enabled()
    joint_bio = clot_phi_joint_bio_enabled()
    in_dim = clot_phi_feature_dim()

    if rule_baseline:
        print("[i]  clot_phi_simple: RULE BASELINE (no training)", flush=True)
        va = _run_epoch(
            None,
            val_paths,
            phys_cfg=phys_cfg,
            bio_cfg=bio_cfg,
            device=device,
            train=False,
            time_stride=1,
            pos_weight=1.0,
            balanced=False,
            rule_baseline=True,
        )
        print(
            f"[OK]  rule val dice={va['dice']:.3f} bce={va['bce']:.4f} logMAE={va['mu_log_mae']:.4f}",
            flush=True,
        )
        return

    if physics_oracle:
        print("[i]  clot_phi_simple: PHYSICS ORACLE (Carreau x gelation, no training)", flush=True)
        for label, paths_eval in (("val", val_paths), ("train", train_paths[:2])):
            va = _run_epoch(
                None,
                paths_eval,
                phys_cfg=phys_cfg,
                bio_cfg=bio_cfg,
                device=device,
                train=False,
                time_stride=1,
                pos_weight=1.0,
                balanced=False,
                physics_oracle=True,
            )
            print(
                f"[OK]  physics {label} dice={va['dice']:.3f} f1={va['clot_f1']:.3f} "
                f"prec={va['clot_prec']:.3f} rec={va['clot_rec']:.3f} "
                f"logMAE={va['mu_log_mae']:.4f} bio_mse={va.get('bio_mse', 0):.4f} "
                f"pred+={va['pred_pos_frac']:.3f} score={_checkpoint_score(va):.3f}",
                flush=True,
            )
        return

    model = build_clot_phi_model(in_dim=in_dim, hidden=hidden).to(device)
    species_head = None
    if joint_bio:
        species_head = ClotPhiSpeciesHead(in_dim=in_dim, hidden=_species_hidden()).to(device)
    wd = float(os.environ.get("CLOT_PHI_WEIGHT_DECAY", "0.0") or "0.0")
    params = list(model.parameters())
    if species_head is not None:
        params += list(species_head.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd)

    # pos_weight from a quick pass on train set
    pos = 0.0
    tot = 0.0
    for path in train_paths[: min(4, len(train_paths))]:
        data = torch.load(path, weights_only=False)
        for ti in range(0, data.y.shape[0], time_stride):
            step = build_clot_phi_step(data, ti, phys_cfg, bio_cfg, torch.device("cpu"))
            m = step.loss_mask
            if m.any():
                pos += float(step.phi_gt[m].sum().item())
                tot += float(m.sum().item())
    pos_frac = pos / max(tot, 1.0)
    pos_weight = min(max((tot - pos) / max(pos, 1.0), 1.0), float(os.environ.get("CLOT_PHI_POS_WEIGHT_CAP", "15")))
    balanced = _env_bool("CLOT_PHI_BALANCED", False)
    mu_log_lambda = max(float(os.environ.get("CLOT_PHI_MU_LOG_LAMBDA", "1.0") or "0"), 0.0)
    use_soft = _env_bool("CLOT_PHI_SOFT_LABELS", False)
    if balanced:
        # If we subsample 1:1 pos/neg, do NOT also upweight positives.
        pos_weight = 1.0

    # Final-layer bias ~ logit(prior) so the net does not start at sigmoid(0)=0.5 everywhere.
    prior = max(min(pos_frac, 0.45), 0.02)
    with torch.no_grad():
        if hasattr(model, "phi_fc") and isinstance(model.phi_fc, torch.nn.Linear):
            model.phi_fc.bias.fill_(float(torch.log(torch.tensor(prior / (1.0 - prior)))))
        elif hasattr(model, "net"):
            last = model.net[-1]
            if isinstance(last, torch.nn.Linear) and last.bias is not None:
                last.bias.fill_(float(torch.log(torch.tensor(prior / (1.0 - prior)))))

    print(
        f"[i]  clot_phi_simple: mask={clot_phi_mask_mode()} cap={clot_phi_mu_cap_si():.3f} Pa*s "
        f"thr={clot_phi_thresh_si(phys_cfg):.3f} soft={int(use_soft)} balanced={int(balanced)} "
        f"hybrid={int(clot_phi_hybrid_enabled())} model={clot_phi_model_kind()} "
        f"hidden={hidden} depth={clot_phi_mlp_depth()} dropout={clot_phi_dropout():.2f} "
        f"minimal={int(clot_phi_minimal_features_enabled())} species_feat={int(clot_phi_species_features_enabled())} "
        f"joint_bio={int(joint_bio)} in_dim={in_dim} "
        f"train={len(train_paths)} val={len(val_paths)} "
        f"prior={prior:.3f} pos_weight={pos_weight:.2f} mu_log_w={mu_log_lambda:.2f} "
        f"bio_w={_bio_lambda():.2f}",
        flush=True,
    )

    sweep_root = (os.environ.get("CLOT_PHI_SWEEP_DIR") or "").strip()
    sweep_leg = (os.environ.get("CLOT_PHI_SWEEP_LEG") or "").strip()
    if sweep_root and sweep_leg:
        out_dir = (root / sweep_root / sweep_leg).resolve()
    else:
        out_dir = (root / "outputs" / "biochem").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "clot_phi_best.pth"
    best_score = -1.0
    best_dice = -1.0
    log_path = out_dir / "clot_phi_train_log.jsonl"
    if ckpt_path.is_file():
        log_path.unlink(missing_ok=True)

    for epoch in range(epochs):
        tr = _run_epoch(
            model,
            train_paths,
            phys_cfg=phys_cfg,
            bio_cfg=bio_cfg,
            device=device,
            train=True,
            time_stride=time_stride,
            pos_weight=pos_weight,
            balanced=balanced,
            species_head=species_head,
            optimizer=opt,
        )
        with torch.no_grad():
            va = _run_epoch(
                model,
                val_paths,
                phys_cfg=phys_cfg,
                bio_cfg=bio_cfg,
                device=device,
                train=False,
                time_stride=1,
                pos_weight=pos_weight,
                balanced=False,
                species_head=species_head,
            )
        score = _checkpoint_score(va)
        print(
            f"Ep {epoch:02d} | train bce={tr['bce']:.4f} mu_mse={tr.get('mu_log_mse', 0):.4f} "
            f"bio_mse={tr.get('bio_mse', 0):.4f} dice={tr['dice']:.3f} f1={tr['clot_f1']:.3f} "
            f"logMAE={tr['mu_log_mae']:.3f} pred+={tr['pred_pos_frac']:.3f} | "
            f"val bce={va['bce']:.4f} mu_mse={va.get('mu_log_mse', 0):.4f} bio_mse={va.get('bio_mse', 0):.4f} "
            f"dice={va['dice']:.3f} "
            f"f1={va['clot_f1']:.3f} prec={va['clot_prec']:.3f} rec={va['clot_rec']:.3f} "
            f"logMAE={va['mu_log_mae']:.3f} pred+={va['pred_pos_frac']:.3f} gt+={va['gt_pos_frac']:.3f} score={score:.3f}",
            flush=True,
        )
        row = {"epoch": epoch, "train": tr, "val": va, "val_score": score, "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        if score > best_score:
            best_score = score
            best_dice = float(va["dice"])
            payload: dict[str, Any] = {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_dice": best_dice,
                "val_f1": float(va["clot_f1"]),
                "val_score": best_score,
                "config": {
                    "mu_cap_si": clot_phi_mu_cap_si(),
                    "mu_thresh_si": clot_phi_thresh_si(phys_cfg),
                    "hidden": hidden,
                    "in_dim": in_dim,
                    "oracle_mu": clot_phi_oracle_mu_enabled(),
                    "species_features": clot_phi_species_features_enabled(),
                    "joint_bio": joint_bio,
                    "use_prior_features": clot_phi_use_prior_features(),
                    "prior_n": clot_phi_prior_feature_count(),
                    "hybrid": clot_phi_hybrid_enabled(),
                    "minimal_features": clot_phi_minimal_features_enabled(),
                    "model_kind": clot_phi_model_kind(),
                    "dropout": clot_phi_dropout(),
                    "mlp_depth": clot_phi_mlp_depth(),
                    "lr": lr,
                    "weight_decay": wd,
                    "mu_log_lambda": mu_log_lambda,
                    "bio_lambda": _bio_lambda(),
                },
            }
            if species_head is not None:
                payload["species_head_state_dict"] = species_head.state_dict()
            torch.save(payload, ckpt_path)
            print(
                f"   [OK]  saved {ckpt_path} (val f1={va['clot_f1']:.3f} dice={best_dice:.3f} score={best_score:.3f})",
                flush=True,
            )

    print(f"[OK]  Done. Best val score={best_score:.3f} dice={best_dice:.3f} -> {ckpt_path}", flush=True)


if __name__ == "__main__":
    main()
