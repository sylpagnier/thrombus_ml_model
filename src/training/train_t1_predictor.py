"""Tier 1 predictor training (GINO-DEQ).

**Validation composite (``tier1_best_physics.pth``):** after the physics warm-up, each validation
step computes ``val_composite_loss = rel_l2_anchor + 100 * continuity``, where ``continuity`` is the
mean ``|∇·u|`` on the fluid interior (see ``quantify_performance``). **Lower is better**; the
checkpoint with the smallest ``val_composite_loss`` so far is written to ``tier1_best_physics.pth``.
This scalar is unrelated to the training loss and is logged as ``best_val_composite_loss`` in
experiment JSONs and the training diary.

**Model recipe** is fixed in this module (V3 WLS: SIREN decoder, hard BCs, width priors, dynamic PDE
loss weighting, uniform anchor kinematics, WLS NS residuals). Only the run label and training
schedule are configurable via env (epochs, checkpoints, resume, debugging).

**Operational env:** ``TIER1_EXPERIMENT_NAME`` (or ``--experiment-name``), ``TIER1_EPOCHS``,
``TIER1_ADAM_EPOCHS``, ``TIER1_WARM_UP_EPOCHS``, ``TIER1_USE_LBFGS``, checkpoint/resume flags, and
debug toggles — see the training loop below.

Example::

    set TIER1_EXPERIMENT_NAME=smoke
    set TIER1_EPOCHS=25
    py -m src.training.train_t1_predictor
"""
import argparse
import os
import sys
if sys.platform != "win32":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import atexit
import csv
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from src.utils.paths import reports_training_dir, stage_a_dir, resolve_checkpoint
from src.architecture.ginodeq import GINO_DEQ
from src.core_physics.physics_kernels import PhysicsKernels
from src.config import VesselConfig, PhysicsConfig
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR, ReduceLROnPlateau
from src.utils.metrics import quantify_performance, validate_and_plot, DynamicLossWeighter
from src.utils.anchor_mask import graph_has_anchor, anchor_node_mask
from src.utils.kinematics_physics_terms import compute_kinematics_physics_terms
from src.utils.training_diary import TrainingDiary, env_snapshot, write_t1_experiment_artifact
import random

# Scale for continuity (mean |∇·u|) in the validation composite; see module docstring.
TIER1_VAL_COMPOSITE_CONTINUITY_SCALE = 100.0

# --- Locked production Tier 1 hyperparameters (no alternate branches) ---
TIER1_LATENT_DIM = 256
TIER1_DEQ_MAX_ITERS = 25
TIER1_NUM_FOURIER_FREQS = 16
TIER1_ACTIVATION_FN = "silu"
TIER1_FOURIER_BASE = 1.5
TIER1_ANDERSON_BETA = 0.8
TIER1_LAMBDA_CONT = 1.0
TIER1_P_GRAD_SUPERVISION = 1.0
TIER1_NUM_GLOBAL_TOKENS = 16


@dataclass
class Tier1TrainConfig:
    """Run label for reports. Architecture and loss stack are fixed in this module."""

    experiment_name: str = "default"

    @staticmethod
    def from_env() -> "Tier1TrainConfig":
        return Tier1TrainConfig(
            experiment_name=os.environ.get("TIER1_EXPERIMENT_NAME", "default").strip() or "default",
        )

    def to_serializable(self) -> Dict[str, Any]:
        base = asdict(self)
        base.update(
            {
                "latent_dim": TIER1_LATENT_DIM,
                "deq_max_iters": TIER1_DEQ_MAX_ITERS,
                "num_fourier_freqs": TIER1_NUM_FOURIER_FREQS,
                "activation_fn": TIER1_ACTIVATION_FN,
                "fourier_base": TIER1_FOURIER_BASE,
                "loss_weight_mode": "dynamic",
                "anderson_beta": TIER1_ANDERSON_BETA,
                "lambda_cont": TIER1_LAMBDA_CONT,
                "p_grad_supervision": TIER1_P_GRAD_SUPERVISION,
                "kine_weight_mode": "uniform",
                "use_hard_bcs": True,
                "use_siren_decoder": True,
                "use_width_priors": True,
                "num_global_tokens": TIER1_NUM_GLOBAL_TOKENS,
            }
        )
        return base


@dataclass
class DatasetSplit:
    train: list
    val: list
    train_anchors: int
    train_physics: int


def split_anchor_physics(dataset: Sequence, seed: int = 42, train_ratio: float = 0.9) -> DatasetSplit:
    anchors = [d for d in dataset if d.is_anchor.any().item()]
    physics = [d for d in dataset if not d.is_anchor.any().item()]
    rng = random.Random(seed)
    rng.shuffle(anchors)
    rng.shuffle(physics)
    split_idx_a = int(train_ratio * len(anchors))
    split_idx_p = int(train_ratio * len(physics))
    train_data = anchors[:split_idx_a] + physics[:split_idx_p]
    val_data = anchors[split_idx_a:] + physics[split_idx_p:]
    n_train_anchors = len([d for d in train_data if d.is_anchor.any().item()])
    return DatasetSplit(
        train=train_data,
        val=val_data,
        train_anchors=n_train_anchors,
        train_physics=max(0, len(train_data) - n_train_anchors),
    )


def load_dataset():
    cfg = VesselConfig(tier="tier1")
    if not cfg.graph_output_dir.exists():
        print(f"⚠️ Tier 1 graph dir not found: {cfg.graph_output_dir}")
        return []
    paths = sorted(cfg.graph_output_dir.glob("vessel_*.pt"))
    max_load_raw = os.environ.get("TIER1_MAX_LOAD_VESSELS", "").strip()
    if max_load_raw:
        try:
            max_load = max(1, int(max_load_raw))
        except ValueError:
            max_load = 0
        if max_load > 0 and len(paths) > max_load:
            shuffle_before_cap = os.environ.get("TIER1_MAX_LOAD_SHUFFLE", "1").strip().lower() in (
                "1",
                "true",
                "yes",
                "on",
            )
            if shuffle_before_cap:
                rng = random.Random(42)
                rng.shuffle(paths)
            paths = paths[:max_load]
            print(
                f"✂️ Pre-load cap active (TIER1_MAX_LOAD_VESSELS={max_load}): "
                f"loading {len(paths)} graph files before split."
            )
    dataset = []
    print(f"📂 Loading Tier 1 graphs from {cfg.graph_output_dir}...")
    for f in tqdm(paths):
        dataset.append(torch.load(f, weights_only=False))
    return dataset


def compute_step_loss(
    model,
    data,
    kernels,
    loss_weighter: DynamicLossWeighter,
    current_solver,
    lambda_phys,
    device,
    data_scale=500.0,
    bc_scale=10.0,
    io_scale=5.0,
    wss_scale=10.0,
    boundary_data_weight=2.0,
    tier1_kine_p_weight: float = 1.0,
    re_ref: Optional[float] = None,
    re_scale: Optional[float] = None,
    train_loss_weighter: bool = True,
):
    """Forward pass and Tier 1 loss stack (dynamic PDE weighting only).

    ``l_wss`` is supervised WSS (pred vs label) on **anchor ∩ wall** nodes; see
    :meth:`~src.core_physics.physics_kernels.PhysicsKernels.wall_shear_stress_loss`.
    """
    out = model(
        data,
        solver=current_solver,
        anderson_beta=TIER1_ANDERSON_BETA,
        anderson_warmup_iters=5,
    )
    if isinstance(out, tuple):
        pred, jac_loss = out
    else:
        pred = out
        jac_loss = torch.tensor(0.0, device=device)

    node_is_anchor = anchor_node_mask(data)
    terms = compute_kinematics_physics_terms(
        pred,
        data,
        kernels,
        tier="tier1",
        boundary_data_weight=boundary_data_weight,
        tier1_kine_p_weight=float(tier1_kine_p_weight),
        anchor_kine_importance=None,
        re_ref=re_ref,
        re_scale=re_scale,
    )
    l_wss = terms["l_wss"]
    l_data_kine = terms["l_data_kine"]
    l_mom = terms["l_mom"]
    l_cont = terms["l_cont"]
    l_bc = terms["l_bc"]
    l_io = terms["l_io"]

    lambda_cont = TIER1_LAMBDA_CONT

    # Optional gauge-invariant pressure supervision via pressure gradients on anchor nodes.
    p_grad_loss = torch.tensor(0.0, device=device)
    p_grad_weight = TIER1_P_GRAD_SUPERVISION
    if p_grad_weight > 0.0:
        props = kernels._get_geometric_props(data)
        c_p_pred = kernels._compute_derivatives(pred[:, 2:3], props)
        c_p_true = kernels._compute_derivatives(data.y[:, 2:3], props)
        p_pred_grad = c_p_pred[:, 0:2, 0]
        p_true_grad = c_p_true[:, 0:2, 0]
        if node_is_anchor is not None and int(node_is_anchor.sum().item()) > 0:
            p_grad_loss = torch.nn.functional.mse_loss(p_pred_grad[node_is_anchor], p_true_grad[node_is_anchor])
        else:
            p_grad_loss = torch.tensor(0.0, device=device)

    data_terms = [
        float(data_scale) * l_data_kine,
        float(bc_scale) * l_bc,
        float(io_scale) * l_io,
        float(wss_scale) * l_wss,
        float(p_grad_weight) * p_grad_loss,
    ]

    pde_losses = [
        lambda_phys * l_mom,
        lambda_phys * lambda_cont * l_cont,
    ]
    safe_log_vars = loss_weighter.clamped_log_vars()
    if not train_loss_weighter:
        safe_log_vars = safe_log_vars.detach()
    weighted_terms = []
    for idx, lv in enumerate(safe_log_vars):
        precision = torch.exp(-lv)
        weighted_terms.append(precision * pde_losses[idx] + lv)
    weighted_pdes = sum(weighted_terms)
    weighted_data = sum(data_terms)

    loss = weighted_pdes + weighted_data + (0.1 * jac_loss)

    metrics = {
        "L_data": l_data_kine.item(),
        "L_mom": l_mom.item(),
        "L_cont": l_cont.item(),
        "L_bc": l_bc.item(),
        "L_io": l_io.item(),
        "L_jac": jac_loss.item(),
        "L_wss": l_wss.item(),
        "L_pgrad": p_grad_loss.item(),
        "A_nodes": int(node_is_anchor.sum().item()) if node_is_anchor is not None else 0,
    }
    return loss, metrics


def train_t1_predictor(
    epochs: Optional[int] = None,
    lr: float = 1e-4,
    warm_up_epochs: Optional[int] = None,
    adam_epochs: Optional[int] = None,
    explorer: Optional[Tier1TrainConfig] = None,
):
    explorer = explorer or Tier1TrainConfig.from_env()
    if epochs is None:
        epochs = max(1, int(os.environ.get("TIER1_EPOCHS", "150")))
    if adam_epochs is None:
        raw_adam = os.environ.get("TIER1_ADAM_EPOCHS", "").strip()
        if raw_adam:
            adam_epochs = int(raw_adam)
        else:
            adam_epochs = min(120, int(epochs))
    adam_epochs = max(1, min(int(adam_epochs), int(epochs)))
    if warm_up_epochs is None:
        raw_w = os.environ.get("TIER1_WARM_UP_EPOCHS", "").strip()
        if raw_w:
            warm_up_epochs = int(raw_w)
        else:
            warm_up_epochs = 30
    warm_up_epochs = max(0, min(int(warm_up_epochs), adam_epochs - 1))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device being used:", device)
    print(
        f"⏱️ epochs={epochs} | warm_up={warm_up_epochs} | adam_phase={adam_epochs} "
        f"(set TIER1_EPOCHS / TIER1_WARM_UP_EPOCHS / TIER1_ADAM_EPOCHS to override)"
    )
    print(f"🔬 Run: experiment_name={explorer.experiment_name!r} (fixed arch: latent={TIER1_LATENT_DIM}, SIREN+hard BCs)")

    phys_cfg = PhysicsConfig(tier="tier1")
    model = GINO_DEQ(
        in_channels=15,
        out_channels=5,
        latent_dim=TIER1_LATENT_DIM,
        max_iters=TIER1_DEQ_MAX_ITERS,
        num_fourier_freqs=TIER1_NUM_FOURIER_FREQS,
        phys_cfg=phys_cfg,
        activation_fn=TIER1_ACTIVATION_FN,
        fourier_base=TIER1_FOURIER_BASE,
        use_hard_bcs=True,
        num_global_tokens=TIER1_NUM_GLOBAL_TOKENS,
        use_siren_decoder=True,
        use_width_priors=True,
    ).to(device)

    kernels = PhysicsKernels(phys_cfg=phys_cfg)

    loss_weighter = DynamicLossWeighter(num_losses=2).to(device)

    disable_figures = (os.environ.get("TIER1_DISABLE_FIGURES", "0").strip().lower() in ("1", "true", "yes", "on"))
    skip_validation = os.environ.get("TIER1_SKIP_VALIDATION", "0").strip().lower() in ("1", "true", "yes", "on")
    train_batch_trace = os.environ.get("TIER1_TRAIN_BATCH_TRACE", "0").strip().lower() in ("1", "true", "yes", "on")
    _slow_raw = os.environ.get("TIER1_SLOW_BATCH_LOG_SEC", "20").strip()
    if _slow_raw.lower() in ("0", "false", "no", "off"):
        slow_batch_log_sec = 0.0
    else:
        try:
            slow_batch_log_sec = float(_slow_raw)
        except ValueError:
            slow_batch_log_sec = 20.0
    if not disable_figures:
        fig_dir = reports_training_dir("tier1", "figures")

    opt_params = list(model.parameters()) + list(loss_weighter.parameters())
    optimizer = optim.AdamW(opt_params, lr=lr, weight_decay=1e-5)

    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=warm_up_epochs)
    decay_epochs = adam_epochs - warm_up_epochs
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=decay_epochs, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warm_up_epochs])
    plateau_scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, threshold=5e-4, min_lr=1e-6
    )

    dataset_cfg = VesselConfig(tier="tier1")
    dataset = load_dataset()
    if not dataset:
        return {
            "status": "no_data",
            "experiment_name": explorer.experiment_name,
            "dataset_tier": "tier1",
            "graph_dir": str(dataset_cfg.graph_output_dir),
        }

    split = split_anchor_physics(dataset, seed=42, train_ratio=0.9)
    train_data = split.train
    val_data = split.val

    max_train_vessels = int(os.environ.get("TIER1_MAX_TRAIN_VESSELS", "0"))
    if max_train_vessels > 0 and len(train_data) > max_train_vessels:
        rng = random.Random(42)
        rng.shuffle(train_data)
        train_data = train_data[:max_train_vessels]
        print(f"✂️ Truncated train_data to {max_train_vessels} vessels (TIER1_MAX_TRAIN_VESSELS).")

    n_anchor_train = len([d for d in train_data if d.is_anchor.any().item()])
    n_phys_train = max(0, len(train_data) - n_anchor_train)

    micro_batch_size = int(os.environ.get("TIER1_MICRO_BATCH_SIZE", "1"))
    accumulation_steps = int(os.environ.get("TIER1_ACCUMULATION_STEPS", "8"))

    target_anchor_fraction = float(os.environ.get("TIER1_TARGET_ANCHOR_FRACTION", "0.5"))
    target_anchor_fraction = min(max(target_anchor_fraction, 0.0), 1.0)
    hard_alpha = max(float(os.environ.get("TIER1_HARD_MINING_ALPHA", "0.8")), 0.0)
    hard_refresh = max(int(os.environ.get("TIER1_HARD_MINING_REFRESH_EPOCHS", "4")), 1)
    boundary_data_weight = max(float(os.environ.get("TIER1_BOUNDARY_DATA_WEIGHT", "2.0")), 1.0)
    tier1_kine_p_weight = max(float(os.environ.get("TIER1_KINE_P_WEIGHT", "5.0")), 0.0)
    use_lbfgs_requested = (os.environ.get("TIER1_USE_LBFGS", "1").strip().lower() in ("1", "true", "yes", "on"))
    stop_after_adam = (os.environ.get("TIER1_STOP_AFTER_ADAM", "0").strip().lower() in ("1", "true", "yes", "on"))
    lbfgs_lr = float(os.environ.get("TIER1_LBFGS_LR", "0.01"))
    lbfgs_max_iter = max(1, int(os.environ.get("TIER1_LBFGS_MAX_ITER", "20")))
    lbfgs_history_size = max(1, int(os.environ.get("TIER1_LBFGS_HISTORY_SIZE", "30")))
    lbfgs_max_batches = max(0, int(os.environ.get("TIER1_LBFGS_MAX_BATCHES", "0")))
    use_lbfgs = bool(use_lbfgs_requested)
    if use_lbfgs:
        _lbfgs_tail = max(0, int(epochs) - int(adam_epochs))
        print("✅ TIER1_USE_LBFGS=1: AdamW warm-up/phase then L-BFGS refinement is enabled.")
        if _lbfgs_tail > 0:
            print(f"   L-BFGS tail: {_lbfgs_tail} epoch(s) (epochs {adam_epochs}–{epochs - 1}).")
        else:
            print("   L-BFGS tail: 0 epochs — raise TIER1_EPOCHS above TIER1_ADAM_EPOCHS for a refinement phase.")
        print("ℹ️ Safety note: L-BFGS path snapshots the full loader and can be memory-heavy on large graph datasets.")
        if stop_after_adam:
            print("⏹️ TIER1_STOP_AFTER_ADAM=1: training will stop at Adam/L-BFGS boundary and save a handoff checkpoint.")
    dynamic_freeze_during_warmup = (os.environ.get("TIER1_DYNAMIC_FREEZE_DURING_WARMUP", "1").strip().lower() in ("1", "true", "yes", "on"))
    hard_anchor_multiplier = {}

    def _graph_sampling_key(data, list_idx: int) -> int:
        cid = getattr(data, "config_id", None)
        if cid is None:
            return list_idx
        if torch.is_tensor(cid):
            return int(cid.view(-1)[0].item()) if cid.numel() else list_idx
        return int(cid)

    def _make_train_loader():
        if n_anchor_train > 0 and n_phys_train > 0:
            w_anchor = target_anchor_fraction / max(n_anchor_train, 1)
            w_phys = (1.0 - target_anchor_fraction) / max(n_phys_train, 1)
            sample_weights = []
            for gi, d in enumerate(train_data):
                if graph_has_anchor(d):
                    gkey = _graph_sampling_key(d, gi)
                    hard_mult = float(hard_anchor_multiplier.get(gkey, 1.0))
                    sample_weights.append(w_anchor * hard_mult)
                else:
                    sample_weights.append(w_phys)
            sampler = torch.utils.data.WeightedRandomSampler(
                weights=torch.as_tensor(sample_weights, dtype=torch.double),
                num_samples=max(len(train_data), accumulation_steps * 4),
                replacement=True,
            )
            return DataLoader(train_data, batch_size=micro_batch_size, sampler=sampler)
        return DataLoader(train_data, batch_size=micro_batch_size, shuffle=True)

    loader = _make_train_loader()
    print(
        f"🎯 Weighted sampling enabled (target anchor fraction ~{target_anchor_fraction:.2f}; "
        f"anchors={n_anchor_train}, physics={n_phys_train})."
    )
    val_loader = DataLoader(val_data, batch_size=micro_batch_size, shuffle=False)

    best_val_composite_loss = float("inf")
    best_loss = float('inf')
    ckpt_dir_override = os.environ.get("TIER1_CKPT_DIR", "").strip()
    if ckpt_dir_override:
        model_dir = Path(ckpt_dir_override)
        model_dir.mkdir(parents=True, exist_ok=True)
    else:
        model_dir = stage_a_dir()
    lbfgs_initialized = False
    static_lbfgs_batches_cpu = None
    start_epoch = 0
    latest_ckpt_save = model_dir / "tier1_latest_checkpoint.pth"
    latest_ckpt_path = model_dir / "tier1_latest_checkpoint.pth"
    best_physics_path = resolve_checkpoint("a", "tier1_best_physics.pth")
    ckpt_every = max(1, int(os.environ.get("TIER1_CKPT_EVERY", "5")))
    resume_enabled = (os.environ.get("TIER1_RESUME", "0").strip().lower() in ("1", "true", "yes", "on"))
    init_from_best_enabled = (os.environ.get("TIER1_INIT_FROM_BEST", "0").strip().lower() in ("1", "true", "yes", "on"))
    init_done = False

    if init_from_best_enabled and best_physics_path.exists():
        print(f"🧷 Initializing Tier 1 weights from best physics checkpoint: {best_physics_path}")
        best_state = torch.load(best_physics_path, map_location=device, weights_only=True)
        model.load_state_dict(best_state, strict=False)
        init_done = True

    if resume_enabled and latest_ckpt_path.exists():
        print(f"🔄 Resuming Tier 1 from checkpoint: {latest_ckpt_path}")
        ckpt = torch.load(latest_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        try:
            if ckpt.get("loss_weighter_state_dict") is not None:
                loss_weighter.load_state_dict(ckpt["loss_weighter_state_dict"])
        except RuntimeError:
            print("ℹ️ Reinitializing Tier 1 PDE loss weighter for current setup.")

        start_epoch = int(ckpt.get("epoch", -1)) + 1
        best_val_composite_loss = float(ckpt.get("best_val_composite_loss", best_val_composite_loss))
        best_loss = float(ckpt.get("best_loss", best_loss))

        ckpt_optimizer_type = ckpt.get("optimizer_type", "AdamW")
        if start_epoch < adam_epochs and ckpt_optimizer_type == "AdamW":
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if ckpt.get("scheduler_state_dict") is not None:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        elif ckpt_optimizer_type == "LBFGS":
            print("ℹ️ Ignoring stored LBFGS optimizer state in checkpoint; continuing with AdamW.")

        print(f"✅ Tier 1 resume complete at epoch {start_epoch}.")
    elif resume_enabled:
        print(f"ℹ️ TIER1_RESUME is enabled but no checkpoint found at {latest_ckpt_path}. Starting fresh.")
    elif init_done:
        print("✅ Started from tier1_best_physics.pth (fresh optimizer/schedulers).")

    best_rel_l2 = float("inf")
    plateau_patience = int(os.environ.get("TIER1_EARLY_STOP_PATIENCE", "15"))
    plateau_min_delta = float(os.environ.get("TIER1_EARLY_STOP_MIN_DELTA", "0.005"))
    val_no_improve = 0
    early_stopped = False
    cfg_paths = VesselConfig(tier="tier1")
    diary = TrainingDiary("tier1")
    diary.log_run_start(
        device=device,
        re_target=float(phys_cfg.re_target),
        graph_dir=str(cfg_paths.graph_output_dir),
        n_graphs_total=len(dataset),
        n_train=len(train_data),
        n_val=len(val_data),
        n_anchor_train=n_anchor_train,
        n_phys_train=n_phys_train,
        n_val_anchors=sum(1 for d in val_data if graph_has_anchor(d)),
        micro_batch_size=micro_batch_size,
        accumulation_steps=accumulation_steps,
        target_anchor_fraction=target_anchor_fraction,
        hard_alpha=hard_alpha,
        hard_refresh_epochs=hard_refresh,
        boundary_data_weight=boundary_data_weight,
        epochs=epochs,
        warm_up_epochs=warm_up_epochs,
        adam_epochs=adam_epochs,
        lr=lr,
        stage_split_epochs=int(os.environ.get("TIER1_DATA_STAGE_EPOCHS", "10")),
        plateau_patience=plateau_patience,
        plateau_min_delta=plateau_min_delta,
        ckpt_every=ckpt_every,
        resume_enabled=resume_enabled,
        init_from_best_enabled=init_from_best_enabled,
        init_from_best_applied=init_done,
        use_lbfgs=use_lbfgs,
        start_epoch=start_epoch,
        resumed_checkpoint=bool(resume_enabled and latest_ckpt_path.exists()),
        best_val_composite_loss_checkpoint=float(best_val_composite_loss),
        best_loss_checkpoint=float(best_loss),
        env_tier1_phase1=env_snapshot("TIER1_", "PHASE1_"),
        tier1_train_config=explorer.to_serializable(),
    )

    run_end_emitted = False
    last_epoch_completed: Optional[int] = None

    def _emit_tier1_run_end(interrupted: bool = False) -> None:
        nonlocal run_end_emitted
        if run_end_emitted or not diary.enabled:
            return
        run_end_emitted = True
        if interrupted:
            print("\n⚠️ Training interrupted; appending training diary run_end (JSONL report).")
        diary.log_run_end(
            best_val_composite_loss=float(best_val_composite_loss),
            best_loss=float(best_loss),
            best_rel_l2=float(best_rel_l2),
            early_stopped=bool(early_stopped),
            interrupted=bool(interrupted),
            last_epoch_completed=last_epoch_completed,
            diary_path=str(diary.path) if diary.path else None,
        )

    atexit.register(lambda: _emit_tier1_run_end(True))

    hard_csv_path = reports_training_dir("tier1") / "anchor_hardness.csv"
    hard_csv_path.parent.mkdir(parents=True, exist_ok=True)

    def _refresh_hard_mining(epoch_idx: int):
        if n_anchor_train == 0:
            return
        model.eval()
        rows = []
        with torch.no_grad():
            for gi, d in enumerate(train_data):
                if not graph_has_anchor(d):
                    continue
                dd = d.clone().to(device)
                out = model(dd, solver="anderson", anderson_beta=TIER1_ANDERSON_BETA, anderson_warmup_iters=5)
                pred = out[0] if isinstance(out, tuple) else out
                mask = anchor_node_mask(dd)
                if mask is None or int(mask.sum().item()) == 0:
                    continue
                rel = torch.norm(pred[mask, :2] - dd.y[mask, :2]) / torch.clamp(torch.norm(dd.y[mask, :2]), min=1e-8)
                gkey = _graph_sampling_key(dd, gi)
                rows.append((gkey, float(rel.item())))
        if not rows:
            model.train()
            return
        errs = torch.tensor([r[1] for r in rows], dtype=torch.float32)
        q = float(torch.quantile(errs, torch.tensor(0.7)))
        for gkey, err in rows:
            hard_anchor_multiplier[gkey] = (1.0 + hard_alpha) if err >= q else 1.0
        write_header = (not hard_csv_path.exists()) or (hard_csv_path.stat().st_size == 0)
        with open(hard_csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["epoch", "config_id", "anchor_rel_l2", "hard_mult"])
            for gkey, err in rows:
                w.writerow([epoch_idx, gkey, err, hard_anchor_multiplier.get(gkey, 1.0)])
        model.train()

    for epoch in range(start_epoch, epochs):
        last_epoch_completed = epoch
        if use_lbfgs and stop_after_adam and epoch >= adam_epochs:
            checkpoint = {
                "epoch": epoch - 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss_weighter_state_dict": loss_weighter.state_dict(),
                "best_val_composite_loss": best_val_composite_loss,
                "best_loss": best_loss,
                "optimizer_type": "AdamW",
                "handoff_for_lbfgs": True,
            }
            torch.save(checkpoint, latest_ckpt_save)
            print(
                f"🧳 Saved Tier 1 Adam handoff checkpoint at epoch {epoch - 1} -> {latest_ckpt_save.name}. "
                "Resume with TIER1_STOP_AFTER_ADAM=0 (or unset) to continue L-BFGS."
            )
            break
        if epoch == start_epoch or (epoch % hard_refresh == 0):
            _refresh_hard_mining(epoch)
            loader = _make_train_loader()
        model.train()
        physics_active = epoch >= warm_up_epochs
        lambda_phys = min(1.0, max(0.0, (epoch - warm_up_epochs) / 20.0))
        re_scale_epoch = 1.0
        stage_split = int(os.environ.get("TIER1_DATA_STAGE_EPOCHS", "10"))
        if epoch < stage_split:
            data_scale = float(os.environ.get("TIER1_DATA_SCALE_STAGE1", "700.0"))
            bc_scale = float(os.environ.get("TIER1_BC_SCALE_STAGE1", "8.0"))
            io_scale = float(os.environ.get("TIER1_IO_SCALE_STAGE1", "4.0"))
            wss_scale = float(os.environ.get("TIER1_WSS_SCALE_STAGE1", "8.0"))
        else:
            data_scale = float(os.environ.get("TIER1_DATA_SCALE_STAGE2", "500.0"))
            bc_scale = float(os.environ.get("TIER1_BC_SCALE_STAGE2", "10.0"))
            io_scale = float(os.environ.get("TIER1_IO_SCALE_STAGE2", "5.0"))
            wss_scale = float(os.environ.get("TIER1_WSS_SCALE_STAGE2", "10.0"))
        total_loss_epoch = 0.0

        current_solver = "picard" if epoch < 5 else "anderson"

        if use_lbfgs and epoch >= adam_epochs and not lbfgs_initialized:
            print(f"\n⚡ Switching to L-BFGS Optimizer for the final {epochs - adam_epochs} epochs...")
            torch.cuda.empty_cache()
            loss_weighter.requires_grad_(False)
            lbfgs_params = [p for p in model.parameters() if p.requires_grad]

            optimizer = optim.LBFGS(
                lbfgs_params,
                lr=lbfgs_lr,
                max_iter=lbfgs_max_iter,
                history_size=lbfgs_history_size,
                line_search_fn="strong_wolfe",
                tolerance_grad=1e-6,
                tolerance_change=1e-8
            )
            print(
                f"   L-BFGS settings: lr={lbfgs_lr:.3g}, max_iter={lbfgs_max_iter}, "
                f"history_size={lbfgs_history_size}, max_batches={'all' if lbfgs_max_batches <= 0 else lbfgs_max_batches}"
            )
            static_lbfgs_batches_cpu = list(loader)
            if lbfgs_max_batches > 0 and len(static_lbfgs_batches_cpu) > lbfgs_max_batches:
                static_lbfgs_batches_cpu = static_lbfgs_batches_cpu[:lbfgs_max_batches]
                print(
                    f"   ⚡ TIER1_LBFGS_MAX_BATCHES cap active: using {len(static_lbfgs_batches_cpu)} "
                    "mini-batch(es) in static closure subset."
                )
            else:
                print(
                    f"   📌 Cached static L-BFGS closure subset with {len(static_lbfgs_batches_cpu)} mini-batch(es)."
                )
            lbfgs_initialized = True

        if not lbfgs_initialized:
            pbar = tqdm(loader, desc=f"Tier 1 Epoch {epoch:02d} [Re={phys_cfg.re_target}] (AdamW)")

            optimizer.zero_grad()
            accum_counter = 0

            for batch_idx, data in enumerate(pbar):
                t_batch = time.perf_counter()
                data = data.to(device)
                n_nodes_batch = int(getattr(data, "num_nodes", 0) or 0)
                if train_batch_trace:
                    tqdm.write(
                        f"[Tier1] batch {batch_idx + 1}/{len(loader)}  num_nodes={n_nodes_batch}",
                        flush=True,
                    )

                loss, metrics = compute_step_loss(
                    model,
                    data,
                    kernels,
                    loss_weighter,
                    current_solver,
                    lambda_phys,
                    device,
                    data_scale=data_scale,
                    bc_scale=bc_scale,
                    io_scale=io_scale,
                    wss_scale=wss_scale,
                    boundary_data_weight=boundary_data_weight,
                    tier1_kine_p_weight=tier1_kine_p_weight,
                    re_scale=re_scale_epoch,
                    train_loss_weighter=not (dynamic_freeze_during_warmup and epoch < warm_up_epochs),
                )
                accum_counter += 1
                loss = loss / float(accumulation_steps)

                if torch.isnan(loss):
                    print(f"\n⚠️ NaN detected! Skipping micro-batch.")
                    accum_counter = max(0, accum_counter - 1)
                    continue

                loss.backward()

                is_step = ((batch_idx + 1) % accumulation_steps == 0) or (batch_idx + 1 == len(loader))
                if is_step:
                    if accum_counter > 0 and accum_counter < accumulation_steps:
                        scale = float(accumulation_steps) / float(accum_counter)
                        for p in opt_params:
                            if p.grad is not None:
                                p.grad.mul_(scale)
                    torch.nn.utils.clip_grad_norm_(
                        opt_params,
                        max_norm=1.0,
                    )
                    optimizer.step()
                    optimizer.zero_grad()
                    accum_counter = 0

                total_loss_epoch += (loss.item() * accumulation_steps)

                pbar.set_postfix({
                    "L_tot": f"{(loss.item() * accumulation_steps):.3f}",
                    "L_data": f"{metrics['L_data']:.3f}",
                    "L_mom": f"{metrics['L_mom']:.3f}",
                    "L_cont": f"{metrics['L_cont']:.3f}",
                    "L_bc": f"{metrics['L_bc']:.3f}",
                    "L_io": f"{metrics['L_io']:.3f}",
                    "L_wss": f"{metrics['L_wss']:.3f}",
                    "L_pgrad": f"{metrics['L_pgrad']:.3f}",
                    "L_jac": f"{metrics['L_jac']:.3f}",
                    "LR": f"{optimizer.param_groups[0]['lr']:.2e}",
                    "nodes": n_nodes_batch,
                })
                batch_elapsed = time.perf_counter() - t_batch
                if slow_batch_log_sec > 0.0 and batch_elapsed >= slow_batch_log_sec:
                    tqdm.write(
                        f"[Tier1] slow micro-batch {batch_idx + 1}/{len(loader)} took {batch_elapsed:.1f}s "
                        f"(num_nodes={n_nodes_batch}); forward+DEQ time varies a lot by mesh size.",
                        flush=True,
                    )

            scheduler.step()

        else:
            print(f"⏳ Tier 1 Epoch {epoch:02d} [Re={phys_cfg.re_target}] (L-BFGS Line Search...)")
            epoch_batches_cpu = static_lbfgs_batches_cpu if static_lbfgs_batches_cpu is not None else list(loader)
            n_batches = max(len(epoch_batches_cpu), 1)

            def closure():
                optimizer.zero_grad()
                accumulated_loss = torch.tensor(0.0, device=device)

                for closure_data_cpu in epoch_batches_cpu:
                    closure_data = closure_data_cpu.to(device)
                    loss, _ = compute_step_loss(
                        model,
                        closure_data,
                        kernels,
                        loss_weighter,
                        current_solver,
                        lambda_phys,
                        device,
                        data_scale=data_scale,
                        bc_scale=bc_scale,
                        io_scale=io_scale,
                        wss_scale=wss_scale,
                        boundary_data_weight=boundary_data_weight,
                        tier1_kine_p_weight=tier1_kine_p_weight,
                        re_scale=re_scale_epoch,
                        train_loss_weighter=not (dynamic_freeze_during_warmup and epoch < warm_up_epochs),
                    )
                    loss = loss / n_batches
                    loss.backward()
                    accumulated_loss += loss.detach()
                    del closure_data

                # Keep external watchdogs (e.g., Colab log readers) alive during long LBFGS steps.
                print(
                    f"   [L-BFGS Iteration] Current Closure Loss: {accumulated_loss.item():.4f}",
                    flush=True,
                )
                return accumulated_loss

            loss_tensor = optimizer.step(closure)
            total_loss_epoch = loss_tensor.item() * n_batches

            print(f"✅ L-BFGS Step Complete. Accumulated Full-Batch Loss: {loss_tensor.item():.4f}")

        avg_loss = total_loss_epoch / max(1, len(loader))
        if avg_loss < best_loss and physics_active:
            best_loss = avg_loss
            save_path = model_dir / "tier1_best_loss.pth"
            torch.save(model.state_dict(), save_path)
            print(f"⭐ Saved Best Loss Model to {save_path}")

        if (not disable_figures) and epoch % 5 == 0:
            validate_and_plot(model, val_data[0], epoch, device, tier="tier1")

        should_save_ckpt = ((epoch + 1) % ckpt_every == 0) or (epoch == epochs - 1)
        if should_save_ckpt:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": (scheduler.state_dict() if not lbfgs_initialized else None),
                "loss_weighter_state_dict": loss_weighter.state_dict(),
                "best_val_composite_loss": best_val_composite_loss,
                "best_loss": best_loss,
                "optimizer_type": ("LBFGS" if lbfgs_initialized else "AdamW"),
            }
            torch.save(checkpoint, latest_ckpt_save)
            print(f"💾 Saved Tier 1 checkpoint -> {latest_ckpt_save.name} (every {ckpt_every} epoch(s))")

        diary.log_epoch_end(
            epoch,
            avg_epoch_loss=float(avg_loss),
            lr=float(optimizer.param_groups[0]["lr"]),
            lambda_phys=float(lambda_phys),
            data_scale=float(data_scale),
            bc_scale=float(bc_scale),
            io_scale=float(io_scale),
            wss_scale=float(wss_scale),
            solver=current_solver,
            lbfgs=bool(lbfgs_initialized),
            physics_active=bool(physics_active),
            best_loss_so_far=float(best_loss),
            best_val_composite_loss_so_far=float(best_val_composite_loss),
            best_rel_l2_so_far=float(best_rel_l2),
        )

        if (not skip_validation) and (epoch % 2 == 0):
            if device == "cuda":
                torch.cuda.empty_cache()
            print(
                f"\n⏳ Validation: {len(val_data)} graph(s), {len(val_loader)} minibatch(es) "
                f"(set TIER1_SKIP_VALIDATION=1 to skip; TIER1_VAL_PROGRESS=0 disables the progress bar).",
                flush=True,
            )
            scores = quantify_performance(model, val_loader, kernels, device, tier="tier1")

            print(
                f"\n📊 [Validation] Rel L2 (anchor): {scores.get('rel_l2', float('nan')):.4f} "
                f"(σ {scores.get('rel_l2_std', float('nan')):.4f}, p90 {scores.get('rel_l2_p90', float('nan')):.4f})"
            )
            print(
                f"   Components u / v / p: {scores.get('rel_l2_u', float('nan')):.4f} / "
                f"{scores.get('rel_l2_v', float('nan')):.4f} / {scores.get('rel_l2_p', float('nan')):.4f}"
            )
            print(
                f"   |∇·u| mean (fluid interior): {scores.get('continuity', float('nan')):.3e} "
                f"(p90 {scores.get('continuity_p90', float('nan')):.3e}) | "
                f"Wall |u| mean: {scores.get('wall_slip', float('nan')):.4f} "
                f"(p90 {scores.get('wall_slip_p90', float('nan')):.4f})"
            )
            print(
                f"   γ̇ MSE (anchor): {scores.get('shear_mse', float('nan')):.4e} "
                f"(p90 {scores.get('shear_mse_p90', float('nan')):.4e}) | "
                f"Batches w/ anchors: {int(scores.get('val_anchor_batches', 0))}/"
                f"{int(scores.get('val_total_batches', 0))}"
            )
            print(
                f"   Rel L2 near-wall (anchor low-|SDF|): {scores.get('rel_l2_near_wall', float('nan')):.4f} "
                f"(p90 {scores.get('rel_l2_near_wall_p90', float('nan')):.4f}) | "
                f"Rel L2 high-|∇SDF| (anchor): {scores.get('rel_l2_high_sdf_grad', float('nan')):.4f} "
                f"(p90 {scores.get('rel_l2_high_sdf_grad_p90', float('nan')):.4f})"
            )

            with torch.no_grad():
                safe_vars = loss_weighter.clamped_log_vars()
                weights = torch.exp(-safe_vars)
                w_mom = float(weights[0].item()) if weights.numel() > 0 else 1.0
                w_cont = float(weights[1].item()) if weights.numel() > 1 else float(TIER1_LAMBDA_CONT)
                print(f"⚖️ Learned PDE Weights -> Cont: {w_cont:.2f} | Mom: {w_mom:.2f}")
                n_val_anchor = sum(1 for d in val_data if graph_has_anchor(d))
                print(f"📌 Val split: anchors={n_val_anchor} | physics={max(0, len(val_data) - n_val_anchor)}")

            val_composite_loss = float(scores.get("rel_l2", 0.0)) + TIER1_VAL_COMPOSITE_CONTINUITY_SCALE * float(
                scores.get("continuity", 0.0)
            )
            diary.log_validation(
                epoch,
                scores,
                weight_cont=w_cont,
                weight_mom=w_mom,
                best_rel_l2=best_rel_l2,
                val_no_improve=val_no_improve,
                val_composite_loss=val_composite_loss,
                n_val_anchors=n_val_anchor,
                n_val_physics=max(0, len(val_data) - n_val_anchor),
            )
            if not lbfgs_initialized:
                plateau_scheduler.step(float(scores.get('rel_l2', 0)))

            rel_l2 = float(scores.get("rel_l2", float("inf")))
            if (best_rel_l2 - rel_l2) > plateau_min_delta:
                best_rel_l2 = rel_l2
                val_no_improve = 0
            else:
                val_no_improve += 1
                if val_no_improve >= plateau_patience:
                    print(
                        f"🛑 Early stopping: validation Rel L2 did not improve by > {plateau_min_delta:.4f} "
                        f"for {plateau_patience} validation checks."
                    )
                    early_stopped = True
                    diary.log_event(
                        "early_stop",
                        epoch=epoch,
                        reason="val_rel_l2_plateau",
                        plateau_min_delta=plateau_min_delta,
                        plateau_patience=plateau_patience,
                        last_rel_l2=rel_l2,
                    )
                    break

            if val_composite_loss < best_val_composite_loss and physics_active:
                best_val_composite_loss = val_composite_loss
                save_path = model_dir / "tier1_best_physics.pth"
                torch.save(model.state_dict(), save_path)
                print(f"⭐ Saved Best Physics Model to {save_path}")

    _emit_tier1_run_end(interrupted=False)
    print(
        f"Tier 1 Training Complete. Best val composite loss: {best_val_composite_loss:.4f} "
        f"(rel_l2 + {TIER1_VAL_COMPOSITE_CONTINUITY_SCALE:g}×continuity; lower is better) | "
        f"Best Loss: {best_loss:.4f}"
    )

    if os.environ.get("TIER1_SKIP_EXPERIMENT_ARTIFACT", "0").strip().lower() not in ("1", "true", "yes", "on"):
        write_t1_experiment_artifact(
            explorer,
            best_rel_l2=best_rel_l2,
            best_val_composite_loss=best_val_composite_loss,
            best_loss=best_loss,
            early_stopped=early_stopped,
            n_graphs=len(dataset),
            n_train=len(train_data),
            n_val=len(val_data),
            graph_dir=str(cfg_paths.graph_output_dir),
        )

    return {
        "status": "ok",
        "experiment_name": explorer.experiment_name,
        "best_rel_l2": float(best_rel_l2),
        "best_val_composite_loss": float(best_val_composite_loss),
        "best_loss": float(best_loss),
        "early_stopped": bool(early_stopped),
        "n_graphs": int(len(dataset)),
        "n_train": len(train_data),
        "n_val": len(val_data),
        "graph_dir": str(cfg_paths.graph_output_dir),
    }


def _parse_args():
    p = argparse.ArgumentParser(description="Tier 1 GINO-DEQ predictor training (run label via TIER1_EXPERIMENT_NAME).")
    p.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Sets TIER1_EXPERIMENT_NAME for reports/experiments JSON (same as env).",
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--resume",
        action="store_true",
        help="Resume from tier1_latest_checkpoint.pth (sets TIER1_RESUME=1).",
    )
    mode.add_argument(
        "--new",
        action="store_true",
        help="Start a new run (sets TIER1_RESUME=0).",
    )
    return p.parse_args()


def _prompt_resume_or_new_t1() -> bool:
    """Ask user whether to resume checkpointed training."""
    while True:
        raw = input("Training mode [1=resume / 2=start new] [1]: ").strip()
        if raw in ("", "1"):
            return True
        if raw == "2":
            return False
        print("  Enter 1 or 2.")


if __name__ == "__main__":
    args = _parse_args()
    if args.experiment_name:
        os.environ["TIER1_EXPERIMENT_NAME"] = args.experiment_name
    if args.resume:
        resume_enabled = True
    elif args.new:
        resume_enabled = False
    else:
        resume_enabled = _prompt_resume_or_new_t1()
    os.environ["TIER1_RESUME"] = "1" if resume_enabled else "0"
    print(
        "🔄 Resuming Tier 1 from latest checkpoint."
        if resume_enabled
        else "🆕 Starting a new Tier 1 run."
    )
    train_t1_predictor()
