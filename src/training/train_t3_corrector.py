import atexit
import os
import sys
import math
import time
import copy

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
import gc
import json
from typing import Any, Dict, List, Optional, Tuple, Union

if sys.platform != "win32":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
else:
    # Fallback for Windows if you face OOM issues, otherwise leave empty
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.loader import DataLoader


def resolve_training_device() -> torch.device:
    """Pick compute device. Honors ``TIER3_DEVICE=auto|cuda|cpu`` (default ``auto``).

    Plain ``pip install torch`` is often CPU-only; use ``scripts/install_torch_cuda.ps1``
    or install from https://pytorch.org so ``torch.cuda.is_available()`` is True.
    """
    want = (os.environ.get("TIER3_DEVICE") or "auto").strip().lower()
    cuda_ok = torch.cuda.is_available()

    if want in ("cuda", "gpu"):
        if not cuda_ok:
            print(
                "TIER3_DEVICE=cuda but this PyTorch build has no CUDA "
                "(torch.cuda.is_available() is False).\n"
                "Install a GPU wheel, e.g. run: .\\scripts\\install_torch_cuda.ps1\n"
                "or: py -3 -m pip install torch --upgrade "
                "--index-url https://download.pytorch.org/whl/cu124\n"
                "Verify: py -3 -c \"import torch; print(torch.cuda.is_available())\""
            )
            sys.exit(1)
        dev = torch.device("cuda:0")
        torch.cuda.set_device(0)
        return dev

    if want == "cpu":
        return torch.device("cpu")

    if want != "auto":
        print(f"Unknown TIER3_DEVICE={want!r}; use auto, cuda, or cpu.", file=sys.stderr)
        sys.exit(1)

    if cuda_ok:
        dev = torch.device("cuda:0")
        torch.cuda.set_device(0)
        return dev

    if torch.version.cuda is None:
        print(
            "PyTorch is CPU-only (no CUDA runtime in this install). "
            "GPU present but unused. To enable CUDA, run .\\scripts\\install_torch_cuda.ps1 "
            "or see https://pytorch.org — then rerun training."
        )
    return torch.device("cpu")


def configure_cuda_for_training(device: torch.device) -> None:
    if device.type != "cuda":
        return
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.empty_cache()
    props = torch.cuda.get_device_properties(device)
    try:
        free_b, total_b = torch.cuda.mem_get_info()
        mem_str = f"{free_b / (1024 ** 3):.2f} / {total_b / (1024 ** 3):.2f} GiB free / total"
    except Exception:
        mem_str = f"{props.total_memory / (1024 ** 3):.1f} GiB total"
    print(f"CUDA device: {props.name} | {mem_str}")
from tqdm import tqdm
import random
from torch_geometric.data import Dataset
from src.utils.paths import data_root, get_project_root, reports_dir, stage_b_dir, resolve_checkpoint
from src.architecture.gnode_tier3 import GNODE_Tier3, tier3_truth_node_mask
from src.architecture.lora_injection import inject_lora_to_spectral_linears
from src.core_physics.biochem_physics_kernels import BiochemPhysicsKernels
from src.core_physics.physics_kernels import PhysicsKernels
from src.config import (
    VesselConfig,
    PhysicsConfig,
    BiochemConfig,
    CurriculumConfig,
    STATE_CHANNEL_MU_EFF_ND,
)
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import WeightedRandomSampler
from src.utils.batching import get_batch_tensor
from src.utils.metrics import DynamicLossWeighter
from src.utils.training_diary import TrainingDiary, env_snapshot
from src.training.physics_curriculum import ease01 as _ease01


def _tier3_metrics_jsonl_path():
    return reports_dir() / "tier3_metrics.jsonl"


def _tier3_append_jsonl(record: Dict[str, Any]) -> None:
    path = _tier3_metrics_jsonl_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except OSError:
        pass


def _graph_has_anchor_nodes(data) -> bool:
    """True if this graph carries any COMSOL-matched (patient/anchor) nodes."""
    ia = getattr(data, "is_anchor", None)
    if ia is None:
        return False
    if torch.is_tensor(ia):
        return bool(ia.any().item())
    return bool(ia)


# --- Optional diagnostics (compare with f958b74 ~2026-04-03: that revision mutated
# ``phys_cfg.re_target`` inside ``compute_tier3_loss``; current code keeps config fixed
# and passes ``re_ref`` from ``data.re_actual`` only into ``navier_stokes_residual``.)
def _tier3_debug_enabled() -> bool:
    return (os.environ.get("TIER3_DEBUG") or "").strip().lower() in ("1", "true", "yes", "on")


def _tier3_debug_batches_cap() -> int:
    try:
        return max(0, int(os.environ.get("TIER3_DEBUG_BATCHES", "3")))
    except ValueError:
        return 3


def _tier3_should_log_batch(epoch: int, batch_idx: int) -> bool:
    if not _tier3_debug_enabled():
        return False
    return batch_idx < _tier3_debug_batches_cap()


def _tier3_debug_log_path():
    return reports_dir() / "tier3_debug.log"


def _tier3_dbg_line(msg: str) -> None:
    """Stdout + append to ``<reports_dir>/tier3_debug.log`` (tqdm often obscures raw prints)."""
    print(msg, flush=True)
    if not _tier3_debug_enabled():
        return
    try:
        path = _tier3_debug_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
    except OSError:
        pass


def _tensor_stat(x: torch.Tensor) -> str:
    d = x.detach()
    if not d.numel():
        return "empty"
    finite = torch.isfinite(d)
    n_fin = int(finite.sum().item())
    if n_fin == 0:
        return "all non-finite"
    d2 = d[finite]
    return f"min={d2.min().item():.4g} max={d2.max().item():.4g} finite={n_fin}/{d.numel()}"


def _scalar_fin(name: str, t: Union[torch.Tensor, float]) -> Tuple[bool, float]:
    v = float(t.detach().item() if torch.is_tensor(t) else t)
    ok = math.isfinite(v)
    return ok, v


def _debug_kendall_terms(
    loss_weighter: DynamicLossWeighter,
    losses: List,
    task_active: List[bool],
) -> None:
    """Print per-task precision, raw loss, and contribution precision*L + log_var (matches weighter)."""
    names = [
        "ADR_F", "ADR_S", "W_Bio", "W_Phy", "Bio_IO", "NS_mom", "Data_Kine", "Data_Bio",
    ]
    min_lv = loss_weighter.per_task_min_log_var
    max_lv = loss_weighter.per_task_max_log_var
    loss_weighter.clamped_log_vars().detach()
    _tier3_dbg_line("   [Kendall breakdown] task | active | L_raw | prec=exp(-lv) | lv | contrib=prec*L+lv")
    with torch.no_grad():
        for i, loss in enumerate(losses):
            ta_i = task_active[i]
            act = bool(ta_i.item()) if torch.is_tensor(ta_i) else bool(ta_i)
            if not act:
                _tier3_dbg_line(f"      {names[i]:9} | off")
                continue
            li = loss.detach().item() if torch.is_tensor(loss) else float(loss)
            lv = float(
                torch.clamp(
                    loss_weighter.log_vars[i].detach(), min=min_lv[i], max=max_lv[i]
                ).item()
            )
            prec = math.exp(-lv)
            contrib = prec * li + lv
            flag = "" if math.isfinite(contrib) and math.isfinite(li) else " **NON-FINITE**"
            _tier3_dbg_line(
                f"      {names[i]:9} | on  | L={li:.6e} | prec={prec:.4g} | lv={lv:.4f} | contrib={contrib:.6e}{flag}"
            )


def _debug_tier3_batch(
    *,
    epoch: int,
    batch_idx: int,
    data,
    pred_series: torch.Tensor,
    all_losses: list,
    task_active: list,
    loss_weighter: DynamicLossWeighter,
    loss_total: torch.Tensor,
    l_latent_reg: torch.Tensor,
    metrics: dict,
    re_ref: Optional[float],
    r_lo: float,
    r_hi: float,
    evaluation_times: torch.Tensor,
    start_idx: int,
    end_idx: int,
    truth_count: int,
) -> None:
    src = getattr(data, "_tier3_path", None) or getattr(data, "_tier3_source", None)
    _tier3_dbg_line(
        f"\n[TIER3_DEBUG] epoch={epoch} batch={batch_idx} "
        f"src={src!r} "
        f"N={int(data.num_nodes)} T_y={int(data.y.shape[0])} T_eval={int(evaluation_times.shape[0])} "
        f"window=[{start_idx}:{end_idx}] truth_nodes={truth_count}"
    )
    _tier3_dbg_line(
        f"   scales: u_ref={data.u_ref!r} d_bar={data.d_bar!r} "
        f"re_actual={getattr(data, 're_actual', None)!r} "
        f"re_ref_for_NS={re_ref!r} get_re(u,d)_range=[{r_lo:.4g},{r_hi:.4g}]"
    )
    if hasattr(data, "t") and data.t is not None:
        _tier3_dbg_line(
            f"   data.t: shape={tuple(data.t.shape)} min={data.t.min().item():.6g} max={data.t.max().item():.6g}"
        )
    te = evaluation_times.detach().cpu()
    dt = te[1:] - te[:-1] if te.numel() > 1 else te
    _tier3_dbg_line(
        f"   eval_times: min={te.min().item():.6g} max={te.max().item():.6g} "
        f"dt_min={(dt.min().item() if dt.numel() else float('nan')):.6g}"
    )
    ps = pred_series.detach()
    _tier3_dbg_line(f"   pred_series last step: {_tensor_stat(ps[-1])}")
    bad = []
    loss_names = [
        "L_ADR_F", "L_ADR_S", "L_W_Bio", "L_W_Phy", "L_B_IO", "L_mom", "L_Data_Kine", "L_Data_Bio",
    ]
    for j, ln in enumerate(loss_names):
        ok, v = _scalar_fin(ln, all_losses[j])
        if not ok:
            bad.append((ln, v))
    if bad:
        _tier3_dbg_line(f"   ** non-finite raw losses: {bad}")
    _tier3_dbg_line(
        f"   metrics TF_eff={metrics.get('TF_eff')} L_Latent_Reg={metrics.get('L_Latent_Reg')}"
    )
    lt = loss_total.detach().item()
    lr = (1e-3 * l_latent_reg).detach().item() if torch.is_tensor(l_latent_reg) else 1e-3 * float(l_latent_reg)
    _tier3_dbg_line(
        f"   loss_weighter()+1e-3*latent: total={lt:.6e} latent_term={lr:.6e} finite={math.isfinite(lt)}"
    )
    _debug_kendall_terms(loss_weighter, all_losses, task_active)


class PatientDataset(Dataset):
    def __init__(self, root, file_list):
        super().__init__(root, transform=None, pre_transform=None)
        self.file_list = file_list

    def len(self):
        return len(self.file_list)

    def get(self, idx):
        path = self.file_list[idx]
        data = torch.load(path, weights_only=False)
        # Provenance for TIER3_DEBUG=1 (PyG allows extra attributes on Data).
        data._tier3_path = str(path)
        # Also keep a public attribute name; some code paths / collate behavior are
        # more reliable with non-private keys.
        data.tier3_path = str(path)
        return data


def _tier3_data_source_key(data) -> Optional[str]:
    """Resolve a stable source-path key for pseudo-label lookup/debug."""
    src = getattr(data, "tier3_path", None)
    if src is None:
        src = getattr(data, "_tier3_path", None)
    if src is None:
        return None
    return str(src)


def remap_stage_a_encoder_to_corrector(
    tier2_weight: torch.Tensor,
    target_weight_template: torch.Tensor,
) -> torch.Tensor:
    """
    Remap Tier-2 encoder input channels to Tier-3 layout.

    Tier-2 encoded input width is 63; Tier-3 is 64 because one channel was
    inserted in the "rest" block before uv/mu/wss priors. Preserve the prior
    channels by shifting the Tier-2 tail by +1.
    """
    if tier2_weight.shape == target_weight_template.shape:
        return tier2_weight

    new_weight = torch.zeros_like(target_weight_template)
    old_in = int(tier2_weight.shape[1])
    new_in = int(target_weight_template.shape[1])

    if old_in == 63 and new_in == 64:
        # Keep shared prefix, reserve one inserted channel at index 59,
        # then shift uv/mu/wss-related tail by +1 to preserve semantics.
        new_weight[:, :59] = tier2_weight[:, :59]
        new_weight[:, 60:64] = tier2_weight[:, 59:63]
        return new_weight

    # Safe fallback for unexpected shape pairs.
    min_dim = min(old_in, new_in)
    new_weight[:, :min_dim] = tier2_weight[:, :min_dim]
    return new_weight


def load_dataset():
    cfg_patients = VesselConfig(tier="tier3_patients")
    cfg_synthetic = VesselConfig(tier="tier3")

    patient_dir = cfg_patients.graph_output_dir
    synthetic_dir = cfg_synthetic.graph_output_dir

    patient_files = sorted(list(patient_dir.glob("*.pt"))) if patient_dir.exists() else []
    synthetic_files = sorted(list(synthetic_dir.glob("*.pt"))) if synthetic_dir.exists() else []

    if not patient_files and not synthetic_files:
        print(
            f"No Tier 3 graphs found in {patient_dir} or {synthetic_dir}. "
            f"Please generate/extract Tier 3 data first."
        )
        return []

    file_list = patient_files + synthetic_files
    print(
        f"📂 Found {len(patient_files)} Tier 3 anchor/patient graphs + "
        f"{len(synthetic_files)} Tier 3 synthetic graphs for lazy loading..."
    )

    # PyG ``Dataset`` requires a ``root``; loads use absolute paths in ``file_list``.
    return PatientDataset(root=str(data_root()), file_list=file_list)


def initialize_biochem_priors(model):
    print("🧬 Injecting physical priors into biochemistry decoder biases...")
    target_layer = model.biochem_decoder.linear if hasattr(model.biochem_decoder, 'linear') else model.biochem_decoder

    # FIX: Do not use strict zeros. It kills the backward gradient (grad_in = grad_out @ W).
    # Use a very small random initialization so the Neural ODE can learn.
    torch.nn.init.normal_(target_layer.weight, std=1e-4)

    bias_vals = torch.zeros(12, dtype=torch.float32)

    # Resting bulk species: C_nd = 1 ⇒ decoder output is log1p(1) = ln(2) (see _decode_species_log1p).
    resting_indices = [ 0, 4, 6, 7 ]  # RP, PT, AT, FG

    for idx in resting_indices:
        bias_vals[ idx ] = math.log(2.0)

    # Apply the biases
    with torch.no_grad():
        target_layer.bias.copy_(bias_vals)

    print("🛑 Initializing ODE function to near-zero derivative...")

    def _init_linear_like_near_zero(module, eps=1e-5):
        linear = module.linear if hasattr(module, 'linear') else module
        if not isinstance(linear, torch.nn.Linear):
            return
        weight = getattr(linear, 'weight_orig', None)
        if weight is None:
            weight = linear.weight
        torch.nn.init.uniform_(weight, a=-eps, b=eps)
        if linear.bias is not None:
            torch.nn.init.zeros_(linear.bias)

    # Target terminal projection layers in the ODE network so dz/dt starts near zero.
    terminal_layers = []
    for name, module in model.ode_func.named_modules():
        if not isinstance(module, (torch.nn.Linear, type(model.biochem_decoder))):
            continue

        has_linear_child = any(
            child_name and isinstance(child, (torch.nn.Linear, type(model.biochem_decoder)))
            for child_name, child in module.named_modules()
        )
        if not has_linear_child:
            terminal_layers.append((name, module))

    # Fallback safety: if architecture inspection misses terminals, damp all ODE linear projections.
    if not terminal_layers:
        terminal_layers = [
            (name, module)
            for name, module in model.ode_func.named_modules()
            if isinstance(module, (torch.nn.Linear, type(model.biochem_decoder)))
        ]

    with torch.no_grad():
        for _, layer in terminal_layers:
            _init_linear_like_near_zero(layer)
        if hasattr(model.ode_func, 'derivative_scale'):
            model.ode_func.derivative_scale.fill_(1e-3)


def make_tier3_dynamic_loss_weighter(curriculum: CurriculumConfig, device) -> DynamicLossWeighter:
    """Per-task Kendall bounds: cap physics weights, floor supervised data weights."""
    # Hard cap the physics precision so PDE terms cannot be effectively muted.
    phys_ceiling = 10.0
    data_floor = max(float(curriculum.tier3_data_precision_floor), 1e-12)
    adr_s_floor = max(float(curriculum.tier3_adr_s_precision_floor), 1e-12)
    w_phys_floor = max(float(curriculum.tier3_w_phys_precision_floor), 1e-12)
    phys_min_lv = -math.log(phys_ceiling)
    data_max_lv = -math.log(data_floor)
    adr_s_max_lv = -math.log(adr_s_floor)
    w_phys_max_lv = -math.log(w_phys_floor)
    # 0–5: ADR_F, ADR_S, W_Bio, W_Phy, Bio_IO, NS_mom — 6–7: supervised Data_Kine, Data_Bio
    min_lv = [phys_min_lv] * 6 + [-8.0, -8.0]
    max_lv = [
        float("inf"),      # ADR_F
        adr_s_max_lv,      # ADR_S
        float("inf"),      # W_Bio
        w_phys_max_lv,     # W_Phy
        float("inf"),      # Bio_IO
        float("inf"),      # NS_mom
        data_max_lv,       # Data_Kine
        data_max_lv,       # Data_Bio
    ]
    print(
        f"⚖️ Tier 3 loss weighter: physics prec ≤ {phys_ceiling:g} (log_var ≥ {phys_min_lv:.3f}), "
        f"ADR_S prec ≥ {adr_s_floor:g} (log_var ≤ {adr_s_max_lv:.3f}), "
        f"W_Phys prec ≥ {w_phys_floor:g} (log_var ≤ {w_phys_max_lv:.3f}), "
        f"data prec ≥ {data_floor:g} (log_var ≤ {data_max_lv:.3f}), "
        f"freeze_in_warmup={curriculum.tier3_weighter_freeze_during_warmup}"
    )
    return DynamicLossWeighter(num_losses=8, min_log_var=min_lv, max_log_var=max_lv).to(device)


def inject_tier3_kinematic_lora(model: GNODE_Tier3, rank: int = 4, alpha: float = 1.0) -> None:
    """Attach LoRA to SpectralLinear layers in the kinematic stack (call before ``setup_tier3_optimization``)."""
    n_enc = inject_lora_to_spectral_linears(model.kin_encoder, rank=rank, alpha=alpha)
    n_proc = inject_lora_to_spectral_linears(model.kin_processor, rank=rank, alpha=alpha)
    n_dec = inject_lora_to_spectral_linears(model.kinematics_decoder, rank=rank, alpha=alpha)
    print(
        f"💉 LoRA injected (SpectralLinear count): kin_encoder={n_enc}, "
        f"kin_processor={n_proc}, kinematics_decoder={n_dec} "
        f"(rank={rank}, alpha={alpha}); plain nn.Linear modules contribute 0."
    )


def setup_tier3_optimization(model, loss_weighter, base_lr=1e-3):
    print("❄️  Verifying Kinematic Backbone is Frozen.")
    print("🔥 Activating LoRA layers, Biochemistry Encoders/Decoders, and Loss Weighter.")

    # Set the frozen kinematic backbone to eval mode!
    model.kin_encoder.eval()
    model.kin_processor.eval()
    model.kinematics_decoder.eval()

    # Freeze everything by default to be absolutely safe
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze specifically intended modules
    for name, param in model.named_parameters():
        if 'lora' in name.lower():
            param.requires_grad = True

    for param in model.bio_encoder.parameters():
        param.requires_grad = True

    for param in model.ode_func.parameters():
        param.requires_grad = True

    for name, param in model.biochem_decoder.named_parameters():
        if 'lora' not in name.lower():
            param.requires_grad = True

    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))

    return optim.AdamW([
        {'params': trainable_params, 'lr': base_lr},
        {'params': loss_weighter.parameters(), 'lr': 5e-2, 'weight_decay': 0.0}
    ], weight_decay=1e-5)


def pretrain_autoencoder(model, loader, optimizer, device, kernels, epochs=5, ode_reaction_epochs=8):
    print("\n🚀 --- Phase 3a: Autoencoder Pre-Training (Freezing ODE) ---")
    prior_requires_grad = {name: p.requires_grad for name, p in model.named_parameters()}

    for param in model.ode_func.parameters():
        param.requires_grad = False

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0

        for data in loader:
            data = data.to(device)
            mask = tier3_truth_node_mask(data, int(data.x.shape[0]), device)
            if not mask.any():
                continue

            optimizer.zero_grad()

            pred_species = model.autoencode(data)
            targ_species = data.y[0, :, 4:16]

            loss = F.mse_loss(pred_species[mask], targ_species[mask])
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        if num_batches == 0:
            print(
                f"AE Epoch {epoch:02d}: skipped (no graphs with COMSOL-labeled nodes — check is_anchor / re-extract)."
            )
            continue
        avg_loss = total_loss / num_batches
        print(f"AE Epoch {epoch:02d}: Recon Loss = {avg_loss:.4e}")

    print("\n🧪 --- Phase 3a.5: ODE Reaction-Rate Imitation Pre-Training ---")
    for _, param in model.named_parameters():
        param.requires_grad = False
    for param in model.ode_func.parameters():
        param.requires_grad = True

    base_lr = float(optimizer.param_groups[0].get("lr", 1e-3))
    ode_rxn_lr = base_lr * 3.0
    ode_rxn_optimizer = optim.AdamW(model.ode_func.parameters(), lr=ode_rxn_lr, weight_decay=0.0)
    on_frac = float(os.environ.get("TIER3_ODE_MANIFOLD_FRAC", "0.6"))
    print(
        f"🧪 ODE-RXN optimizer: lr={ode_rxn_lr:.3e} (base_lr x3.0) | "
        f"on-manifold COMSOL species frac ≈ {on_frac:.2f} (TIER3_ODE_MANIFOLD_FRAC)"
    )

    # Species ordering must match kinetics.compute_species_reactions inputs.
    rxn_keys = ['RP', 'AP', 'APR', 'APS', 'PT', 'T', 'AT', 'FG', 'FI']
    scales = kernels.cfg.get_species_scales(device=device)[:9].view(1, 9)
    dt_ode_probe = 1e-2
    max_reaction_batches = 32
    prev_rxn_avg = None
    plateau_streak = 0

    model.train()
    for epoch in range(ode_reaction_epochs):
        total_loss = 0.0
        num_batches = 0

        for batch_idx, data in enumerate(loader):
            if batch_idx >= max_reaction_batches:
                break
            data = data.to(device)
            n_nodes = int(data.x.shape[0])
            if n_nodes == 0:
                continue

            # Mix off-manifold random states with on-manifold COMSOL trajectory samples so
            # d(log species)/dt targets align with states the encoder actually sees in training.
            ti = 0
            use_manifold = (
                hasattr(data, "y")
                and data.y is not None
                and data.y.shape[-1] >= 13
                and data.y.shape[0] >= 1
                and torch.rand(1, device=device).item() < on_frac
            )
            if use_manifold:
                ti = int(torch.randint(0, int(data.y.shape[0]), (1,), device=device).item())
                species_log = data.y[ti, :, 4:13].to(device=device, dtype=torch.float32)
            else:
                species_lin_si = torch.rand(n_nodes, 9, device=device) * (2.0 * scales)
                species_log = torch.log1p(species_lin_si / scales)
            wall_species = torch.zeros(n_nodes, 3, device=device)
            random_species = torch.cat([species_log, wall_species], dim=1)

            if hasattr(data, "y") and data.y is not None and data.y.shape[-1] >= 3:
                ti_uv = int(ti) if use_manifold else 0
                ti_uv = min(ti_uv, int(data.y.shape[0]) - 1)
                u_v_p = data.y[ti_uv, :, :3]
            else:
                u_v = data.x[:, 11:13]
                p0 = torch.zeros(n_nodes, 1, device=device)
                u_v_p = torch.cat([u_v, p0], dim=1)
            u_det = u_v_p[:, 0]
            v_det = u_v_p[:, 1]

            bio_in = torch.cat([random_species, u_v_p, data.x[:, :15]], dim=-1)
            with torch.no_grad():
                z0 = model.bio_encoder(bio_in)
            z0 = z0.detach()

            batch_idx_nodes = get_batch_tensor(data, n_nodes, device)
            edge_index = data.edge_index
            edge_attr = data.edge_attr

            ode_rxn_optimizer.zero_grad()

            dz_dt = model.ode_func(0.0, z0, edge_index, edge_attr, batch_idx_nodes)
            species_now = model._decode_species_log1p(model.biochem_decoder(z0))[:, :9]
            species_next = model._decode_species_log1p(model.biochem_decoder(z0 + dt_ode_probe * dz_dt))[:, :9]
            pred_dlog_dt = (species_next - species_now) / dt_ode_probe

            species_now_si = torch.clamp(torch.expm1(species_now), min=0.0) * scales
            species_dict = {k: species_now_si[:, i] for i, k in enumerate(rxn_keys)}
            props = kernels.core._get_geometric_props(data)
            if isinstance(data.u_ref, torch.Tensor) and data.u_ref.numel() == n_nodes:
                props['u_ref'] = data.u_ref
                props['d_bar'] = data.d_bar
            else:
                props['u_ref'] = data.u_ref[batch_idx_nodes]
                props['d_bar'] = data.d_bar[batch_idx_nodes]
            shear_rate = kernels._compute_shear_rate(u_det, v_det, props, data)
            reaction_terms = kernels.kinetics.compute_species_reactions(species_dict, shear_rate)
            target_dlog_dt = torch.stack(
                [
                    reaction_terms[k]
                    / (scales[:, i] * torch.clamp(torch.exp(species_now[:, i]), min=1e-8))
                    for i, k in enumerate(rxn_keys)
                ],
                dim=1,
            )
            target_dlog_dt = torch.clamp(target_dlog_dt, min=-20.0, max=20.0)

            loss = F.mse_loss(pred_dlog_dt, target_dlog_dt)
            loss.backward()
            ode_rxn_optimizer.step()

            total_loss += float(loss.item())
            num_batches += 1

        if num_batches == 0:
            print(f"ODE-RXN Epoch {epoch:02d}: skipped (no usable batches).")
            continue
        avg_loss = total_loss / num_batches
        print(f"ODE-RXN Epoch {epoch:02d}: Reaction Mimic Loss = {avg_loss:.4e}")
        if prev_rxn_avg is not None:
            rel_change = abs(avg_loss - prev_rxn_avg) / max(abs(prev_rxn_avg), 1e-12)
            if rel_change < 1e-3:
                plateau_streak += 1
            else:
                plateau_streak = 0
            if plateau_streak >= 1:
                print(
                    f"⚠️ ODE-RXN plateau signal: rel_change={rel_change:.2e} "
                    f"(streak={plateau_streak + 1} epochs)"
                )
        prev_rxn_avg = avg_loss

    for name, param in model.named_parameters():
        param.requires_grad = prior_requires_grad.get(name, True)


def compute_tier3_loss(
    model,
    data,
    kernels,
    loss_weighter,
    device,
    bio_cfg,
    epoch=0,
    total_epochs=25,
    curriculum: Optional[CurriculumConfig] = None,
    debug_batch: Optional[Tuple[int, int]] = None,
    pseudo_target_trajectory: Optional[torch.Tensor] = None,
    pseudo_loss_weight: float = 0.0,
):
    curriculum = curriculum or CurriculumConfig()

    num_nodes_d = int(data.x.shape[0])
    truth_mask = tier3_truth_node_mask(data, num_nodes_d, device)

    re_ref = None
    if hasattr(data, 're_actual') and data.re_actual is not None:
        ra = data.re_actual
        re_ref = float(ra.mean().item()) if torch.is_tensor(ra) else float(ra)

    # NS momentum uses Re = get_re(u_ref, d_bar), not PhysicsConfig.re_target directly. Tier 3 can
    # override via ``re_actual`` (passed as re_ref). Fail fast with tensor dumps if Re would be <= 0.
    phys_cfg_ns = kernels.core.cfg
    u_s = data.u_ref.squeeze() if torch.is_tensor(data.u_ref) else data.u_ref
    d_s = data.d_bar.squeeze() if torch.is_tensor(data.d_bar) else data.d_bar
    Re_from_graph = phys_cfg_ns.get_re(u_s, d_s)
    if torch.is_tensor(Re_from_graph):
        r_lo = float(Re_from_graph.detach().min().item())
        r_hi = float(Re_from_graph.detach().max().item())
    else:
        r_lo = r_hi = float(Re_from_graph)
    Re_effective = float(re_ref) if re_ref is not None else r_lo
    if not math.isfinite(Re_effective) or Re_effective <= 0:
        raise ValueError(
            "Invalid Reynolds number for Navier–Stokes residual (1/Re blows up). "
            f"PhysicsConfig.re_target={phys_cfg_ns.re_target} only defines scaling when graphs are built; "
            f"runtime Re is get_re(u_ref, d_bar) unless overridden by data.re_actual → re_ref. "
            f"Effective Re={Re_effective}, re_ref={re_ref!r}, get_re(u_ref,d_bar) in [{r_lo}, {r_hi}], "
            f"u_ref={data.u_ref!r}, d_bar={data.d_bar!r}, re_actual={getattr(data, 're_actual', None)!r}"
        )

    full_times = bio_cfg.resolve_tier3_times(data, device)

    actual_num_steps = int(data.y.shape[0])
    start_idx = 0
    end_idx = actual_num_steps
    y_true_trajectory = data.y
    teacher_forcing_ratio = 0.0

    wu = curriculum.tier3_warmup_epochs
    if model.training:
        if epoch < wu:
            teacher_forcing_ratio = 1.0
        else:
            decay_progress = (epoch - wu) / float(curriculum.tier3_teacher_force_decay_epochs)
            decay_progress = _ease01(decay_progress, curriculum.tier3_curriculum_easing)
            teacher_forcing_ratio = max(0.0, 1.0 - decay_progress)

        # Teacher forcing uses COMSOL labels only where ``tier3_truth_node_mask`` is True
        # (synthetic graphs: all False; patient graphs: spatially matched nodes only).
        if not truth_mask.any():
            teacher_forcing_ratio = 0.0

        # TBPTT: shorten the time window to limit pred_trajectory length and autograd memory.
        # Anchors may start at random start_idx (ground truth exists at every time). Synthetic graphs
        # must use start_idx=0 so species ICs stay the resting prior at physical t=0; only the window
        # length is capped (avoids simulating all ~60 steps in one graph when truth_mask is all False).
        if actual_num_steps > 2:
            window_cap = max(2, actual_num_steps - 1)
            # Keep windows small for stability/speed; random start_idx covers different trajectory regions.
            # Override via TIER3_TBPTT_MAX_WINDOW=5|8|... when needed.
            tbptt_cap = max(2, int(os.environ.get("TIER3_TBPTT_MAX_WINDOW", "8")))
            proposed_window = 5 + (epoch // 4)
            window_size = min(proposed_window, tbptt_cap, window_cap)
            if truth_mask.any():
                max_start = actual_num_steps - window_size
                if max_start > 0:
                    start_idx = int(torch.randint(0, max_start, (1,), device=device).item())
                else:
                    start_idx = 0
            else:
                early_frac = float(os.environ.get("TIER3_SYNTH_TBPTT_EARLY_FRAC", "0.4"))
                early_frac = min(max(early_frac, 0.0), 1.0)
                early_epochs = max(1, int(total_epochs * early_frac))
                if epoch < early_epochs:
                    half = max(1, early_epochs // 2)
                    synth_window = 1 if epoch < half else 2
                    window_size = min(window_cap, max(1, synth_window))
                start_idx = 0
            end_idx = start_idx + window_size
            y_true_trajectory = data.y[start_idx:end_idx]
            evaluation_times = full_times[start_idx:end_idx]
        else:
            evaluation_times = full_times
    else:
        evaluation_times = full_times

    initial_species_for_window = None
    if model.training and start_idx > 0:
        with torch.no_grad():
            warmup_times = full_times[:start_idx + 1]
            warmup_series = model(
                data,
                warmup_times,
                y_true_trajectory=data.y[:start_idx + 1],
                teacher_forcing_ratio=1.0,
                start_idx=0,
            )
            warmup_species = warmup_series[-1, :, 4:16]

        if truth_mask.any():
            gt_species_at_start = data.y[start_idx, :, 4:16].to(device)
            initial_species_for_window = torch.where(
                truth_mask.unsqueeze(-1),
                gt_species_at_start,
                warmup_species
            )
        else:
            initial_species_for_window = warmup_species

    # 2. Forward Pass (Trajectory Generation)
    pred_series = model(
        data,
        evaluation_times,
        y_true_trajectory=y_true_trajectory,
        teacher_forcing_ratio=teacher_forcing_ratio,
        start_idx=start_idx,
        initial_species=initial_species_for_window,
    )

    props = kernels.core._get_geometric_props(data)
    batch_idx_nodes = get_batch_tensor(data, data.num_nodes, device)
    if isinstance(data.u_ref, torch.Tensor) and data.u_ref.numel() == data.num_nodes:
        props['u_ref'] = data.u_ref
        props['d_bar'] = data.d_bar
    else:
        props['u_ref'] = data.u_ref[batch_idx_nodes]
        props['d_bar'] = data.d_bar[batch_idx_nodes]

    # 3. Supervised data loss (full supervised time window on anchor nodes)
    pred_final = pred_series[-1]
    l_data_kine = torch.tensor(0.0, device=device)
    l_data_bio = torch.tensor(0.0, device=device)
    has_anchor_supervision = bool(truth_mask.any().item())

    # FIX: Since we removed the 3x dense multiplier, the prediction frequency
    # perfectly matches the data frequency. No need to slice [::3] anymore!
    pred_series_data_freq = pred_series
    target_series = y_true_trajectory.to(device)

    # Supervised loss only on COMSOL-trusted nodes (entire trajectory window).
    # l_data_kine: Huber on [u,v,p,mu_nd] vs batch variance scale (anchors only).
    # l_data_bio: Huber on log1p species channels vs per-channel floors (bulk + wall).
    if has_anchor_supervision:
        node_is_anchor = truth_mask
        pred_kine = pred_series_data_freq[:, node_is_anchor, :4]
        targ_kine = target_series[:, node_is_anchor, :4]
        kine_var = torch.clamp(torch.var(targ_kine, dim=(0, 1), keepdim=True), min=1e-2)
        l_data_kine = torch.mean(F.huber_loss(pred_kine, targ_kine, reduction='none') / kine_var)

        pred_bio = pred_series_data_freq[:, node_is_anchor, 4:16]
        targ_bio = target_series[:, node_is_anchor, 4:16]
        raw_bio_var = torch.var(targ_bio, dim=(0, 1), keepdim=True, unbiased=False)

        scales = bio_cfg.get_species_scales(device=device)
        apr_floor = torch.log1p((bio_cfg.APRcrit * bio_cfg.bulk_scale) / scales[2])
        aps_floor = torch.log1p((bio_cfg.APScrit * bio_cfg.bulk_scale) / scales[3])
        t_floor = torch.log1p((bio_cfg.Tcrit * bio_cfg.bulk_scale) / scales[5])
        baseline_floor = torch.tensor(0.01, dtype=targ_bio.dtype, device=device)

        bio_floors = torch.stack([
            baseline_floor, baseline_floor, apr_floor, aps_floor,
            baseline_floor, t_floor, baseline_floor, baseline_floor,
            baseline_floor, baseline_floor, baseline_floor, baseline_floor
        ]).view(1, 1, 12)

        safe_bio_var = torch.maximum(raw_bio_var, bio_floors)
        l_data_bio = torch.mean(F.huber_loss(pred_bio, targ_bio, reduction='none', delta=1.0) / safe_bio_var)

    l_pseudo = torch.tensor(0.0, device=device)
    has_pseudo_supervision = False
    if pseudo_target_trajectory is not None and (not has_anchor_supervision):
        pseudo_target = pseudo_target_trajectory.to(device)
        pseudo_target = pseudo_target[start_idx:end_idx]
        if pseudo_target.shape == pred_series.shape:
            has_pseudo_supervision = True
            l_pseudo = F.mse_loss(pred_series[:, :, 4:16], pseudo_target[:, :, 4:16])

    # 4. Physics PDE Loss (Evaluated over dense time sequence)
    num_steps = len(evaluation_times) - 1
    # Keep fallback zero losses attached to the current forward graph.
    # Some sparse/synthetic batches can yield no valid residual nodes; if every
    # active loss is a detached constant, backward() will fail.
    z = pred_final.sum() * 0.0
    if num_steps <= 0:
        l_adr_fast = l_adr_slow = l_wall_bio = l_wall_phys = l_bio_io = z
    else:
        dt_intervals = (evaluation_times[1:] - evaluation_times[:-1]).view(-1, 1, 1)
        dt_intervals = torch.clamp(dt_intervals, min=1e-9)
        d_pred_dt = (pred_series[1:] - pred_series[:-1]) / dt_intervals
        l_adr_fast = l_adr_slow = l_wall_bio = l_wall_phys = l_bio_io = z

        for t_idx in range(num_steps):
            # Evaluate physics at step t+1 using finite difference gradient
            pred_t = pred_series[t_idx + 1]
            d_dt_t = d_pred_dt[t_idx]

            vel_t = pred_t[:, 0:2]
            # Clamp species to physically safe ranges before computing residual gradients.
            biochem_t = torch.clamp(pred_t[:, 4:13], min=-10.0, max=8.0)
            wall_t = torch.clamp(pred_t[:, 13:16], min=-10.0, max=8.0)

            dC_dt_t = d_dt_t[:, 4:13]
            dM_dt_t = d_dt_t[:, 13:16]

            l_af, l_as = kernels.biochem_adr_residual(biochem_t, vel_t, props, data, d_pred_dt=dC_dt_t)
            l_wb, l_wp = kernels.biochem_wall_residual(biochem_t, wall_t, vel_t, props, data, dM_dt_t)
            l_bi, l_bo = kernels.biochem_inlet_outlet_residual(biochem_t, props, data)

            l_adr_fast = l_adr_fast + l_af
            l_adr_slow = l_adr_slow + l_as
            l_wall_bio = l_wall_bio + l_wb
            l_wall_phys = l_wall_phys + l_wp
            l_bio_io = l_bio_io + (l_bi + l_bo)

        inv = 1.0 / float(num_steps)
        l_adr_fast = l_adr_fast * inv
        l_adr_slow = l_adr_slow * inv
        l_wall_bio = l_wall_bio * inv
        l_wall_phys = l_wall_phys * inv
        l_bio_io = l_bio_io * inv

    # Fluid Mechanics (pseudo-steady snapshot at final time in the window)
    l_mom = kernels.core.navier_stokes_residual(
        pred_final[:, 0:4], data, props=props, re_ref=re_ref
    )
    l_visc_reg = kernels.compute_dual_viscosity_penalty(
        pred_final[:, 13:14],  # M_wall proxy (first wall-species channel)
        pred_final[:, 12:13],  # FI_field proxy (FI bulk channel)
        props,
        data,
    )

    # Scale volume-heavy residuals by ~1/sqrt(N) so different mesh sizes are comparable in Kendall weighting.
    if curriculum.tier3_physics_geom_normalization:
        geom_inv = 1.0 / math.sqrt(max(1.0, float(num_nodes_d)))
        l_adr_fast = l_adr_fast * geom_inv
        l_adr_slow = l_adr_slow * geom_inv
        l_wall_bio = l_wall_bio * geom_inv
        l_wall_phys = l_wall_phys * geom_inv
        l_bio_io = l_bio_io * geom_inv
        l_mom = l_mom * geom_inv

    # --- Auxiliary segmentation (soft clot Dice) + COMSOL temporal derivative match (physics-informed) ---
    l_seg = torch.tensor(0.0, device=device)
    l_phys_temp = torch.tensor(0.0, device=device)
    w_seg = float(os.environ.get("TIER3_SOFT_DICE_WEIGHT", "0.05"))
    w_pt = float(os.environ.get("TIER3_COMSOL_TEMPORAL_WEIGHT", "0.02"))
    phys_cfg = kernels.core.cfg
    mu_ch = STATE_CHANNEL_MU_EFF_ND
    if model.training and has_anchor_supervision and w_seg > 0.0:
        mu_p = phys_cfg.viscosity_nd_to_si(pred_final[:, mu_ch])
        mu_g = phys_cfg.viscosity_nd_to_si(y_true_trajectory[-1, :, mu_ch])
        thr = 20.0 * phys_cfg.mu_viscosity_nd_scale
        tau_si = max(float(os.environ.get("TIER3_SOFT_DICE_TEMP_SI", "5e-4")), 1e-12)
        p = torch.sigmoid((mu_p - thr) / tau_si)
        g = (mu_g > thr).float()
        m = truth_mask
        pv, gc = p[m], g[m]
        inter = (pv * gc).sum()
        union = pv.sum() + gc.sum() + 1e-8
        dice_soft = (2.0 * inter) / union
        l_seg = 1.0 - dice_soft
    if (
        model.training
        and has_anchor_supervision
        and w_pt > 0.0
        and pred_series_data_freq.shape[0] >= 2
        and evaluation_times.numel() >= 2
    ):
        node_is_anchor = truth_mask
        dtv = (evaluation_times[1:] - evaluation_times[:-1]).view(-1, 1, 1).clamp(min=1e-9)
        pd = (pred_series_data_freq[1:, node_is_anchor] - pred_series_data_freq[:-1, node_is_anchor]) / dtv
        gd = (target_series[1:, node_is_anchor] - target_series[:-1, node_is_anchor]) / dtv
        l_phys_temp = F.huber_loss(pd, gd, reduction="mean", delta=1.0)

    latent_scale = float(os.environ.get("TIER3_LATENT_REG_SCALE", "1e-3"))

    # Eight Kendall tasks: skip supervised heads on non-anchor batches.
    all_losses = [
        l_adr_fast, l_adr_slow, l_wall_bio, l_wall_phys, l_bio_io, l_mom,
        l_data_kine, l_data_bio,
    ]
    task_active = [True] * 6 + [has_anchor_supervision, has_anchor_supervision]
    l_latent_reg = torch.tensor(0.0, device=device)
    ode_eval_count = int(getattr(model.ode_func, "derivative_eval_count", 0))
    if model.training and ode_eval_count > 0:
        # Memory-safe detached metric from ODE evaluations in this forward pass.
        avg_deriv_energy = model.ode_func.derivative_energy_sum / max(model.ode_func.derivative_eval_count, 1)
        l_latent_reg = torch.tensor(avg_deriv_energy, dtype=torch.float32, device=device)
        model.ode_func.derivative_energy_sum = 0.0
        model.ode_func.derivative_eval_count = 0

    loss = (
        loss_weighter(all_losses, task_active=task_active)
        + (float(pseudo_loss_weight) * l_pseudo)
        + (latent_scale * l_latent_reg)
        + (1e-3 * l_visc_reg)
        + (w_seg * l_seg)
        + (w_pt * l_phys_temp)
    )

    # Guard for sparse pseudo-only windows where every active term can become
    # graph-disconnected (e.g., all-zero residual/mimic terms in low-anchor mode).
    # Tethering with a zero-valued parameter term keeps autograd valid without
    # altering the scalar objective value.
    grad_tether_active = False
    if model.training and (not loss.requires_grad):
        for p in model.parameters():
            if p.requires_grad:
                loss = loss + (p.reshape(-1)[0] * 0.0)
                grad_tether_active = True
                break

    metrics = {
        "L_mom": l_mom.item(),
        "L_ADR_F": l_adr_fast.item(),
        "L_ADR_S": l_adr_slow.item(),
        "L_W_Bio": l_wall_bio.item(),
        "L_W_Phy": l_wall_phys.item(),
        "L_B_IO": l_bio_io.item(),
        # Supervised COMSOL labels on anchor nodes only (Huber / variance-normalized).
        "L_Data_Kine": l_data_kine.item(),
        "L_Data_Bio": l_data_bio.item(),
        "L_Pseudo": l_pseudo.item(),
        "L_Latent_Reg": l_latent_reg.item(),
        "L_Visc_Reg": l_visc_reg.item(),
        "TF_eff": float(teacher_forcing_ratio),
        "ODE_Evals": ode_eval_count,
        "Has_Anchor_Supervision": float(has_anchor_supervision),
        "Has_Pseudo_Supervision": float(has_pseudo_supervision),
        "Grad_Tether_Active": float(grad_tether_active),
        "PDE_Steps": float(num_steps),
        "L_Seg": l_seg.item(),
        "L_PhysTemp": l_phys_temp.item(),
    }
    if debug_batch is not None:
        de, dbi = debug_batch
        _debug_tier3_batch(
            epoch=de,
            batch_idx=dbi,
            data=data,
            pred_series=pred_series,
            all_losses=all_losses,
            task_active=task_active,
            loss_weighter=loss_weighter,
            loss_total=loss,
            l_latent_reg=l_latent_reg,
            metrics=metrics,
            re_ref=re_ref,
            r_lo=r_lo,
            r_hi=r_hi,
            evaluation_times=evaluation_times,
            start_idx=start_idx,
            end_idx=end_idx,
            truth_count=int(truth_mask.sum().item()),
        )
    return loss, metrics


def calculate_validation_metrics(pred, data, kernels, device):
    props = kernels.core._get_geometric_props(data)

    num_nodes = int(data.num_nodes)
    truth_mask = tier3_truth_node_mask(data, num_nodes, pred.device)

    if pred.shape[0] != num_nodes:
        raise ValueError(
            "calculate_validation_metrics: pred rows must equal data.num_nodes "
            f"({pred.shape[0]} != {num_nodes})."
        )
    if data.y.dim() != 3:
        raise ValueError(
            "calculate_validation_metrics expects data.y shaped [T, N, C] (tier-3 trajectories); "
            f"got {tuple(data.y.shape)}."
        )
    if data.y.shape[1] != num_nodes:
        raise ValueError(
            "calculate_validation_metrics: data.y spatial dim must match num_nodes "
            f"({data.y.shape[1]} != {num_nodes})."
        )

    y_last = data.y[-1]

    mu_ch = STATE_CHANNEL_MU_EFF_ND
    mu_eff_nd = pred[ :, mu_ch ]
    mu_scale = kernels.core.cfg.mu_viscosity_nd_scale
    clot_threshold = 20.0 * mu_scale

    mu_pred_dimensional = kernels.core.cfg.viscosity_nd_to_si(mu_eff_nd)
    pred_clot = (mu_pred_dimensional > clot_threshold).float()

    dice = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    gt_clot = torch.zeros_like(pred_clot)

    if truth_mask.any() and data.y.shape[-1] > mu_ch + 1:
        mu_gt_dimensional = kernels.core.cfg.viscosity_nd_to_si(y_last[:, mu_ch])
        gt_clot = (mu_gt_dimensional > clot_threshold).float()
        pc = pred_clot[truth_mask]
        gc = gt_clot[truth_mask]
        intersection = (pc * gc).sum()
        dice = (2.0 * intersection) / (pc.sum() + gc.sum() + 1e-8)

    # --- Hemodynamic Metric: WSS Pearson (patent lumen, COMSOL-trusted wall nodes only) ---
    mask_wall = data.mask_wall.view(-1).bool()
    zero_pearson = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    wss_diag: Dict[str, Any] = {
        "wss_pearson_reason": "unset",
        "patent_wall_count": 0,
        "std_wss_pred": None,
        "std_wss_targ": None,
    }
    pearson_corr = zero_pearson

    if not truth_mask.any():
        wss_diag["wss_pearson_reason"] = "no_comsol_truth_nodes"
    elif not mask_wall.any():
        wss_diag["wss_pearson_reason"] = "no_wall_mask"
    elif data.y.shape[-1] <= 1:
        wss_diag["wss_pearson_reason"] = "insufficient_label_channels"
    else:
        patent_wall_mask = mask_wall & truth_mask
        if data.y.shape[-1] > mu_ch + 1:
            patent_wall_mask = patent_wall_mask & (gt_clot == 0)

        if not patent_wall_mask.any():
            wss_diag["wss_pearson_reason"] = "empty_patent_wall_mask"
        else:
            wss_diag["patent_wall_count"] = int(patent_wall_mask.sum().item())
            # Compute Predicted WSS ([N,1] fields for WLS — same contract as physics_kernels)
            c_u = kernels.core._compute_derivatives(pred[ :, 0:1 ], props)
            c_v = kernels.core._compute_derivatives(pred[ :, 1:2 ], props)
            dudx_p, dudy_p = c_u[ :, 0, 0 ], c_u[ :, 1, 0 ]
            dvdx_p, dvdy_p = c_v[ :, 0, 0 ], c_v[ :, 1, 0 ]

            mu_wall_p = pred[ patent_wall_mask, mu_ch ]
            tau_xx_p = 2.0 * mu_wall_p * dudx_p[ patent_wall_mask ]
            tau_yy_p = 2.0 * mu_wall_p * dvdy_p[ patent_wall_mask ]
            tau_xy_p = mu_wall_p * (dudy_p[ patent_wall_mask ] + dvdx_p[ patent_wall_mask ])

            nx = data.x[ patent_wall_mask, 3 ]
            ny = data.x[ patent_wall_mask, 4 ]

            tx_p, ty_p = tau_xx_p * nx + tau_xy_p * ny, tau_xy_p * nx + tau_yy_p * ny
            tn_p = tx_p * nx + ty_p * ny
            wss_pred = torch.sqrt((tx_p - tn_p * nx) ** 2 + (ty_p - tn_p * ny) ** 2 + 1e-8)

            # Ground-truth WSS from final timestep velocities (same [N, C] layout as pred)
            c_u_t = kernels.core._compute_derivatives(y_last[ :, 0:1 ], props)
            c_v_t = kernels.core._compute_derivatives(y_last[ :, 1:2 ], props)
            dudx_t, dudy_t = c_u_t[ :, 0, 0 ], c_u_t[ :, 1, 0 ]
            dvdx_t, dvdy_t = c_v_t[ :, 0, 0 ], c_v_t[ :, 1, 0 ]

            mu_wall_t = (
                y_last[ patent_wall_mask, mu_ch ]
                if data.y.shape[ -1 ] > mu_ch + 1
                else torch.ones_like(mu_wall_p)
            )
            tau_xx_t = 2.0 * mu_wall_t * dudx_t[ patent_wall_mask ]
            tau_yy_t = 2.0 * mu_wall_t * dvdy_t[ patent_wall_mask ]
            tau_xy_t = mu_wall_t * (dudy_t[ patent_wall_mask ] + dvdx_t[ patent_wall_mask ])

            tx_t, ty_t = tau_xx_t * nx + tau_xy_t * ny, tau_xy_t * nx + tau_yy_t * ny
            tn_t = tx_t * nx + ty_t * ny
            wss_targ = torch.sqrt((tx_t - tn_t * nx) ** 2 + (ty_t - tn_t * ny) ** 2 + 1e-8)

            min_std = 1e-12
            if wss_pred.numel() < 2:
                wss_diag["wss_pearson_reason"] = "too_few_patent_wall_points_for_correlation"
                pearson_corr = zero_pearson
            else:
                std_p = wss_pred.std(unbiased=False)
                std_t = wss_targ.std(unbiased=False)
                wss_diag["std_wss_pred"] = float(std_p.item())
                wss_diag["std_wss_targ"] = float(std_t.item())
                if std_p < min_std and std_t < min_std:
                    wss_diag["wss_pearson_reason"] = "both_wss_vectors_near_constant"
                    pearson_corr = zero_pearson
                elif std_p < min_std:
                    wss_diag["wss_pearson_reason"] = "pred_wss_near_constant_on_patent_wall"
                    pearson_corr = zero_pearson
                elif std_t < min_std:
                    wss_diag["wss_pearson_reason"] = "comsol_wss_near_constant_on_patent_wall"
                    pearson_corr = zero_pearson
                else:
                    stacked = torch.stack([wss_pred, wss_targ])
                    pearson_corr = torch.corrcoef(stacked)[ 0, 1 ]
                    if torch.isnan(pearson_corr):
                        wss_diag["wss_pearson_reason"] = "corrcoef_nan"
                        pearson_corr = zero_pearson
                    else:
                        wss_diag["wss_pearson_reason"] = "ok"

    # Species channels are stored as log1p(species_nd). Convert FI to SI for reporting.
    fi_log1p = pred[:, 12]
    fi_scale = kernels.cfg.get_species_scales(device=pred.device)[8]
    max_fibrin_pred = torch.clamp(torch.expm1(fi_log1p), min=0.0).max().mul(fi_scale).item()

    return dice.item(), pearson_corr.item(), max_fibrin_pred, wss_diag


def _tier3_save_val_debug_plot(
    out_dir,
    epoch: int,
    pred_last,
    v_data,
    kernels,
    device: torch.device,
) -> None:
    """Sparse validation figure: COMSOL vs predicted viscosity on truth nodes (1:1 scatter)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    truth_mask = tier3_truth_node_mask(v_data, int(v_data.num_nodes), device)
    if not truth_mask.any():
        return
    phys_cfg = kernels.core.cfg
    mu_ch = STATE_CHANNEL_MU_EFF_ND
    p_mu = phys_cfg.viscosity_nd_to_si(pred_last[:, mu_ch][truth_mask]).detach().float().cpu().numpy()
    y_last = v_data.y[-1].to(device)
    g_mu = phys_cfg.viscosity_nd_to_si(y_last[:, mu_ch][truth_mask]).detach().float().cpu().numpy()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.scatter(g_mu, p_mu, s=2, alpha=0.35, c="C0")
    lo = float(min(float(g_mu.min()), float(p_mu.min())))
    hi = float(max(float(g_mu.max()), float(p_mu.max())))
    ax.plot([lo, hi], [lo, hi], "r--", lw=1, alpha=0.8)
    ax.set_xlabel("COMSOL μ (SI)")
    ax.set_ylabel("Pred μ (SI)")
    ax.set_title(f"Val μ (truth nodes) epoch {epoch}")
    fig.tight_layout()
    fig.savefig(out_dir / f"tier3_val_mu_epoch_{epoch:04d}.png", dpi=120)
    plt.close(fig)


def _compute_anchor_dice(model, loader, kernels, bio_cfg, device) -> float:
    if len(loader) == 0:
        return 0.0
    model.eval()
    val_dice_total = 0.0
    with torch.no_grad():
        for v_data in loader:
            v_data = v_data.to(device)
            val_eval_times = bio_cfg.resolve_tier3_times(v_data, device)
            v_pred = model(v_data, val_eval_times)
            if isinstance(v_pred, tuple):
                v_pred = v_pred[0]
            d, _, _, _ = calculate_validation_metrics(v_pred[-1], v_data, kernels, device)
            val_dice_total += d
    return val_dice_total / max(len(loader), 1)


def train_teacher_on_anchors(
    student_model,
    train_anchor_dataset,
    val_anchor_dataset,
    kernels,
    bio_cfg,
    curriculum,
    device,
    base_lr,
    low_anchor_mode: bool = False,
):
    """Train a teacher only on anchor graphs for pseudo-label distillation."""
    if len(train_anchor_dataset) == 0:
        print("⚠️ Teacher stage skipped: no anchor graphs in training split.")
        return None, 0.0

    teacher = copy.deepcopy(student_model).to(device)
    teacher_weighter = make_tier3_dynamic_loss_weighter(curriculum, device)
    teacher_optimizer = setup_tier3_optimization(teacher, teacher_weighter, base_lr=base_lr)
    teacher_loader = DataLoader(train_anchor_dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=False)
    teacher_val_loader = DataLoader(val_anchor_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
    train_anchor_keys = {str(p) for p in getattr(train_anchor_dataset, "file_list", [])}
    val_anchor_keys = {str(p) for p in getattr(val_anchor_dataset, "file_list", [])}
    overlap = train_anchor_keys & val_anchor_keys
    overlap_ratio = (
        len(overlap) / max(1, len(val_anchor_keys))
        if len(val_anchor_keys) > 0 else 0.0
    )
    if len(train_anchor_keys) == 1 and train_anchor_keys == val_anchor_keys:
        print(
            "⚠️ One-anchor regime: teacher train/val use the same anchor file. "
            "Treat teacher val Dice as a pipeline-health metric, not generalization."
        )
    elif overlap_ratio > 0.0:
        print(
            f"⚠️ Teacher anchor split overlap: {len(overlap)}/{max(1, len(val_anchor_keys))} "
            f"({overlap_ratio:.1%}) shared between train and val."
        )

    max_epochs = max(1, int(os.environ.get("TIER3_TEACHER_MAX_EPOCHS", "12")))
    target_dice = float(os.environ.get("TIER3_TEACHER_TARGET_DICE", "0.55"))
    accumulation_steps = 4
    best_state = None
    best_dice = -1.0

    print(
        f"\n👩‍🏫 --- Teacher Stage (anchors only): max_epochs={max_epochs}, "
        f"target_val_dice={target_dice:.3f} ---"
    )
    early_stop_allowed = not low_anchor_mode and overlap_ratio == 0.0
    if not early_stop_allowed:
        print("   ℹ️ Low-anchor mode: disabling teacher Dice early-stop to avoid misleading stop signals.")
    for epoch in range(max_epochs):
        teacher.train()
        teacher_optimizer.zero_grad()
        for batch_idx, data in enumerate(teacher_loader):
            data = data.to(device)
            data.x.requires_grad_(True)
            loss, _ = compute_tier3_loss(
                teacher,
                data,
                kernels,
                teacher_weighter,
                device,
                bio_cfg,
                epoch=epoch,
                total_epochs=max_epochs,
                curriculum=curriculum,
                pseudo_target_trajectory=None,
                pseudo_loss_weight=0.0,
            )
            (loss / accumulation_steps).backward()
            if ((batch_idx + 1) % accumulation_steps == 0) or (batch_idx + 1 == len(teacher_loader)):
                torch.nn.utils.clip_grad_norm_(teacher.parameters(), max_norm=1.0)
                teacher_optimizer.step()
                teacher_optimizer.zero_grad()

        val_dice = _compute_anchor_dice(teacher, teacher_val_loader, kernels, bio_cfg, device)
        print(f"   Teacher epoch {epoch:02d} | anchor val dice = {val_dice:.4f}")
        if val_dice > best_dice:
            best_dice = val_dice
            best_state = copy.deepcopy(teacher.state_dict())
        if early_stop_allowed and val_dice >= target_dice:
            print(f"   ✅ Teacher reached target Dice ({val_dice:.4f} >= {target_dice:.4f}); stopping early.")
            break

    if best_state is not None:
        teacher.load_state_dict(best_state, strict=False)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"✅ Teacher frozen. Best anchor validation Dice: {best_dice:.4f}")
    return teacher, float(best_dice)


def build_synthetic_pseudo_labels(teacher, synthetic_dataset, bio_cfg, device):
    """Run frozen teacher on synthetic graphs and cache pseudo trajectories."""
    pseudo = {}
    if teacher is None or len(synthetic_dataset) == 0:
        return pseudo

    teacher.eval()
    synth_loader = DataLoader(synthetic_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
    print(f"🧾 Building pseudo-label bank for {len(synthetic_dataset)} synthetic graphs...")
    with torch.no_grad():
        for data in synth_loader:
            src = _tier3_data_source_key(data)
            if src is None:
                continue
            data = data.to(device)
            eval_times = bio_cfg.resolve_tier3_times(data, device)
            pred = teacher(data, eval_times)
            if isinstance(pred, tuple):
                pred = pred[0]
            pseudo[src] = pred.detach().cpu()
    print(f"✅ Pseudo-label bank ready: {len(pseudo)} synthetic trajectories.")
    return pseudo


def train_t3_corrector(epochs=25, lr=1e-3):
    device = resolve_training_device()
    print(f"Device: {device}")
    if device.type == "cpu":
        print(
            "CPU: biochem ODE uses 32 RK4 substeps per segment by default "
            "(faster; set TIER3_ADJOINT_RK4_SUBSTEPS=128 to match GPU fidelity)."
        )
    else:
        configure_cuda_for_training(device)

    phys_cfg = PhysicsConfig(tier="tier3")
    bio_cfg = BiochemConfig(tier="tier3")
    curriculum = CurriculumConfig()
    core_kernels = PhysicsKernels(phys_cfg=phys_cfg)
    kernels = BiochemPhysicsKernels(biochem_cfg=bio_cfg, core_physics_kernels=core_kernels)

    # PASS PHYS_CFG TO MODEL
    model = GNODE_Tier3(
        phys_cfg=phys_cfg,
        in_channels=12,
        spatial_channels=15,
        latent_dim=64,
        max_inner_iters=10,
        mu_ratio_max=bio_cfg.mu_ratio_max,
        mat_crit=bio_cfg.viscosity_mat_crit,
        fi_crit=bio_cfg.viscosity_fi_crit,
        temp_mat=bio_cfg.viscosity_gnode_temp_mat,
        temp_fi=bio_cfg.viscosity_gnode_temp_fi,
    ).to(device)

    # 1. Resume Tier 3 if checkpoint exists; otherwise load Tier 2 backbone
    root = get_project_root()
    model_dir = stage_b_dir()
    tier3_resume_path = resolve_checkpoint("b", "tier3_best_bio.pth")
    tier2_path = resolve_checkpoint("a", "tier2_best_physics.pth")
    resume_enabled = (os.environ.get("TIER3_RESUME", "0").strip().lower() in ("1", "true", "yes", "on"))

    if resume_enabled and tier3_resume_path.exists():
        resume_state = torch.load(tier3_resume_path, map_location=device, weights_only=True)
        model.load_state_dict(resume_state, strict=False)
        print(f"🔁 Resumed Tier 3 from checkpoint: {tier3_resume_path.name}")
    elif resume_enabled and tier2_path.exists():
        state_dict = torch.load(tier2_path, map_location=device, weights_only=True)

        mapped_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('encoder.'):
                mapped_state_dict[key.replace('encoder.', 'kin_encoder.')] = value
            elif key.startswith('core.'):
                mapped_state_dict[key.replace('core.', 'kin_processor.')] = value
            elif key.startswith('kinematics_decoder.'):
                mapped_state_dict[key] = value
            # Extract the frozen mu_encoder
            elif key.startswith('mu_encoder.'):
                mapped_state_dict[key] = value

        # --- Dynamic channel expansion surgery (Tier 2 -> Tier 3) ---
        if 'kin_encoder.0.weight' in mapped_state_dict:
            tier2_weight = mapped_state_dict['kin_encoder.0.weight']
            model_weight = model.kin_encoder[0].weight
            if tier2_weight.shape[1] != model_weight.shape[1]:
                print(f"🔧 Adapting Tier 2 encoder weights ({tier2_weight.shape[1]} -> {model_weight.shape[1]})...")
                new_weight = remap_stage_a_encoder_to_corrector(tier2_weight, model_weight)
                mapped_state_dict['kin_encoder.0.weight'] = new_weight
        # ------------------------------------------------------------

        model.load_state_dict(mapped_state_dict, strict=False)
        print("✅ Successfully loaded Tier 2 kinematic weights into Tier 3 backbone.")
    elif resume_enabled:
        print("⚠️ Warning: neither Tier 3 resume checkpoint nor Tier 2 weights were found.")

    if not (resume_enabled and tier3_resume_path.exists()):
        initialize_biochem_priors(model)
    elif resume_enabled:
        print("⏭️ Skipping biochem prior initialization because Tier 3 checkpoint was loaded.")
    loss_weighter = make_tier3_dynamic_loss_weighter(curriculum, device)

    print("💉 Injecting LoRA into kinematic modules (SpectralLinear layers)...")
    inject_tier3_kinematic_lora(model)

    if _tier3_debug_enabled():
        cap = _tier3_debug_batches_cap()
        try:
            lp = _tier3_debug_log_path()
            lp.parent.mkdir(parents=True, exist_ok=True)
            lp.write_text("", encoding="utf-8")
        except OSError:
            pass
        _tier3_dbg_line(
            f"[TIER3_DEBUG] Logging first {cap} batches each epoch → {_tier3_debug_log_path()} "
            "(set TIER3_DEBUG_BATCHES). vs f958b74 (2026-04-03): that tree wrote phys_cfg.re_target "
            "from each batch inside compute_tier3_loss; HEAD uses fixed PhysicsConfig + per-batch re_ref for NS only."
        )

    dataset = load_dataset()
    if len(dataset) == 0:
        return

    # Keep loading lazy: split by file path metadata instead of materializing all graphs.
    all_files = list(dataset.file_list)
    anchors, physics = [], []
    print("🔎 Indexing Tier 3 files by anchor flag (lazy split)...")
    for graph_path in all_files:
        graph = torch.load(graph_path, map_location="cpu", weights_only=False)
        ia = getattr(graph, "is_anchor", None)
        if ia is None:
            is_anchor = False
        elif torch.is_tensor(ia):
            is_anchor = bool(ia.any().item())
        else:
            is_anchor = bool(ia)
        if is_anchor:
            anchors.append(graph_path)
        else:
            physics.append(graph_path)
        del graph
    gc.collect()

    random.seed(42)
    random.shuffle(anchors)
    random.shuffle(physics)
    n_anchors_total = len(anchors)
    low_anchor_threshold = max(1, int(os.environ.get("TIER3_LOW_ANCHOR_THRESHOLD", "5")))
    force_low_anchor_mode = (os.environ.get("TIER3_LOW_ANCHOR_MODE", "").strip().lower() in ("1", "true", "yes", "on"))
    low_anchor_mode = force_low_anchor_mode or (0 < n_anchors_total < low_anchor_threshold)
    if low_anchor_mode:
        print(
            f"🧪 Low-anchor mode enabled: anchors={n_anchors_total} (<{low_anchor_threshold}). "
            "Training emphasizes pipeline health/debug over generalization."
        )

    min_trust = int(curriculum.tier3_min_anchors_for_trusted_metrics)
    metrics_trustworthy = n_anchors_total >= min_trust
    if not metrics_trustworthy:
        print(
            f"⚠️ Validation Dice / WSS are **not** reliable generalization metrics with "
            f"{n_anchors_total} anchor graph(s) (< {min_trust}). Interpret as pipeline health only."
        )

    # Robust split: keep at least one anchor in training whenever anchors exist.
    train_anchors, val_anchors = [], []
    train_physics, val_physics = [], []
    if len(dataset) == 1:
        print("⚠️ Only one graph found. Using it for both Training and Validation.")
        only = [all_files[0]]
        if len(anchors) == 1:
            train_anchors = only[:]
            val_anchors = only[:]
        else:
            train_physics = only[:]
            val_physics = only[:]
        train_data = only
        val_data = only
    else:
        if len(anchors) <= 1:
            # Low-anchor regime: keep anchor in both splits so validation metrics remain meaningful.
            train_anchors = anchors[:]
            val_anchors = anchors[:]
        else:
            split_idx_a = int(0.9 * len(anchors))
            split_idx_a = max(1, min(split_idx_a, len(anchors) - 1))
            train_anchors = anchors[:split_idx_a]
            val_anchors = anchors[split_idx_a:]

        if len(physics) <= 1:
            train_physics = physics[:]
            val_physics = []
        else:
            split_idx_p = int(0.9 * len(physics))
            split_idx_p = max(1, min(split_idx_p, len(physics) - 1))
            train_physics = physics[:split_idx_p]
            val_physics = physics[split_idx_p:]

        train_data = train_anchors + train_physics
        val_data = val_anchors + val_physics

        # Safety fallback if split produced empty validation
        if len(val_data) == 0:
            val_data = train_data

    _ds_root = str(data_root())
    train_dataset = PatientDataset(root=_ds_root, file_list=train_data)
    val_dataset = PatientDataset(root=_ds_root, file_list=val_data)
    train_anchor_dataset = PatientDataset(root=_ds_root, file_list=train_anchors)
    val_anchor_dataset = PatientDataset(
        root=_ds_root,
        file_list=val_anchors if len(val_anchors) > 0 else train_anchors
    )
    train_synth_dataset = PatientDataset(root=_ds_root, file_list=train_physics)

    # IMPORTANT:
    # Tier 3 graphs store trajectories as y: [T, N, 16]. With vanilla PyG batching,
    # x concatenates over nodes while y concatenates over time, which misaligns tensors.
    # Use batch_size=1 and gradient accumulation for stable/equivalent optimization.
    accumulation_steps = 4

    # Use simple loaders with batch_size=1 to preserve [T, N, 16] integrity.
    train_anchor_count = len(train_anchors) if len(dataset) > 1 else 1
    if train_anchor_count == 0:
        print("⚠️ No anchors in training split; running physics-only updates.")

    # Anchor oversampling for sparse-anchor phases (especially 1-anchor runs).
    # Target a minimum anchor sampling fraction without changing graph contents.
    anchor_set = set(train_anchors)
    train_physics_count = max(0, len(train_data) - len(anchor_set))
    if len(anchor_set) > 0 and train_physics_count > 0:
        if low_anchor_mode:
            target_anchor_fraction = 0.70 if len(anchor_set) == 1 else 0.55
        else:
            target_anchor_fraction = 0.5 if len(anchor_set) == 1 else 0.35
        w_anchor = target_anchor_fraction / max(len(anchor_set), 1)
        w_phys = (1.0 - target_anchor_fraction) / max(train_physics_count, 1)
        sample_weights = [w_anchor if p in anchor_set else w_phys for p in train_data]
        train_sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=max(len(train_data), accumulation_steps * 2),
            replacement=True,
        )
        print(
            f"🎯 Weighted sampling enabled (anchor frac target ~{target_anchor_fraction:.2f}; "
            f"anchors={len(anchor_set)}, physics={train_physics_count})."
        )
        loader = DataLoader(train_dataset, batch_size=1, shuffle=False, sampler=train_sampler, num_workers=0, pin_memory=False)
    else:
        loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)

    optimizer = setup_tier3_optimization(model, loss_weighter, base_lr=lr)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    start_epoch = 0
    best_composite = -1.0e9
    dice_ema: Optional[float] = None
    latest_ckpt_save = model_dir / "tier3_latest_checkpoint.pth"
    latest_ckpt_path = resolve_checkpoint("b", "tier3_latest_checkpoint.pth")
    ckpt_every = max(1, int(os.environ.get("TIER3_CKPT_EVERY", "1")))

    if resume_enabled and latest_ckpt_path.exists():
        print(f"🔄 Resuming Tier 3 from checkpoint: {latest_ckpt_path}")
        ckpt = torch.load(latest_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler_state = ckpt.get("scheduler_state_dict")
        if scheduler_state is not None:
            scheduler.load_state_dict(scheduler_state)
        loss_weighter.load_state_dict(ckpt["loss_weighter_state_dict"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        best_composite = float(ckpt.get("best_composite", best_composite))
        dice_ema = ckpt.get("dice_ema", dice_ema)
        if dice_ema is not None:
            dice_ema = float(dice_ema)
        teacher_best_dice = float(ckpt.get("teacher_best_dice", 0.0))
        pseudo_w = float(ckpt.get("pseudo_w", 0.0))
        print("🧾 Rebuilding pseudo-label bank from resumed Tier 3 weights...")
        temp_teacher = copy.deepcopy(model).to(device)
        temp_teacher.eval()
        for p in temp_teacher.parameters():
            p.requires_grad = False
        pseudo_bank = build_synthetic_pseudo_labels(
            teacher=temp_teacher,
            synthetic_dataset=train_synth_dataset,
            bio_cfg=bio_cfg,
            device=device,
        )
        del temp_teacher
        pseudo_cov = len(pseudo_bank) / max(1, len(train_physics))
        if len(train_physics) > 0:
            print(
                f"🧾 Pseudo-label coverage after resume: {len(pseudo_bank)}/{len(train_physics)} "
                f"({pseudo_cov:.1%}) synthetic graphs."
            )
        print(f"✅ Tier 3 resume complete at epoch {start_epoch}.")
    elif resume_enabled:
        print(f"ℹ️ TIER3_RESUME is enabled but no checkpoint found at {latest_ckpt_path}. Continuing fresh.")
    else:
        pretrain_autoencoder(model, loader, optimizer, device, kernels, epochs=5)
        teacher, teacher_best_dice = train_teacher_on_anchors(
            student_model=model,
            train_anchor_dataset=train_anchor_dataset,
            val_anchor_dataset=val_anchor_dataset,
            kernels=kernels,
            bio_cfg=bio_cfg,
            curriculum=curriculum,
            device=device,
            base_lr=lr,
            low_anchor_mode=low_anchor_mode,
        )
        pseudo_bank = build_synthetic_pseudo_labels(
            teacher=teacher,
            synthetic_dataset=train_synth_dataset,
            bio_cfg=bio_cfg,
            device=device,
        )
        pseudo_cov = len(pseudo_bank) / max(1, len(train_physics))
        if len(train_physics) > 0:
            print(
                f"🧾 Pseudo-label coverage: {len(pseudo_bank)}/{len(train_physics)} "
                f"({pseudo_cov:.1%}) synthetic graphs."
            )
        pseudo_w_base = float(os.environ.get("TIER3_SYNTH_PSEUDO_WEIGHT", "0.5"))
        min_td = float(os.environ.get("TIER3_PSEUDO_MIN_TEACHER_DICE", "0.08"))
        if teacher_best_dice < min_td:
            pseudo_w = 0.0
            print(
                f"🧷 Synthetic pseudo-label weight set to 0 (teacher val Dice {teacher_best_dice:.4f} < "
                f"TIER3_PSEUDO_MIN_TEACHER_DICE={min_td})."
            )
        else:
            ramp = min(1.0, teacher_best_dice / max(float(os.environ.get("TIER3_PSEUDO_TEACHER_REF_DICE", "0.45")), 1e-6))
            pseudo_w = pseudo_w_base * ramp
            print(
                f"🧷 Synthetic pseudo-label loss weight: {pseudo_w:.3f} "
                f"(base={pseudo_w_base:.3f}, teacher_dice={teacher_best_dice:.4f}, ramp={ramp:.3f})"
            )

    dice_ema_beta = float(os.environ.get("TIER3_VAL_DICE_EMA", "0.25"))
    ckpt_pearson_w = float(os.environ.get("TIER3_CKPT_PEARSON_WEIGHT", "0.02"))

    cfg_paths = VesselConfig(tier="tier3")
    diary = TrainingDiary("tier3")
    diary.log_run_start(
        device=str(device),
        re_target=float(phys_cfg.re_target),
        graph_dir=str(cfg_paths.graph_output_dir),
        n_graphs_total=len(dataset),
        n_files_indexed=len(all_files),
        n_anchors_total=int(n_anchors_total),
        n_train_graphs=len(train_data),
        n_val_graphs=len(val_data),
        n_train_anchors=len(train_anchors),
        n_val_anchors=len(val_anchors),
        n_train_physics=len(train_physics),
        n_val_physics=len(val_physics),
        low_anchor_mode=bool(low_anchor_mode),
        metrics_trustworthy=bool(metrics_trustworthy),
        accumulation_steps=int(accumulation_steps),
        epochs=int(epochs),
        lr=float(lr),
        start_epoch=int(start_epoch),
        best_composite_checkpoint=float(best_composite),
        dice_ema_checkpoint=float(dice_ema) if dice_ema is not None else None,
        teacher_best_dice=float(teacher_best_dice),
        pseudo_w=float(pseudo_w),
        pseudo_bank_size=len(pseudo_bank),
        tier3_warmup_epochs=int(curriculum.tier3_warmup_epochs),
        dice_ema_beta=float(dice_ema_beta),
        ckpt_pearson_weight=float(ckpt_pearson_w),
        resume_enabled=bool(resume_enabled),
        resumed_latest_checkpoint=bool(resume_enabled and latest_ckpt_path.exists()),
        ckpt_every=int(ckpt_every),
        env_tier3_phase1=env_snapshot("TIER3_", "PHASE1_"),
    )

    run_end_emitted = False
    last_epoch_completed: Optional[int] = None

    def _emit_tier3_run_end(interrupted: bool = False) -> None:
        nonlocal run_end_emitted
        if run_end_emitted or not diary.enabled:
            return
        run_end_emitted = True
        if interrupted:
            print("\n⚠️ Training interrupted; appending training diary run_end (JSONL report).")
        diary.log_run_end(
            best_composite=float(best_composite),
            teacher_best_dice=float(teacher_best_dice),
            pseudo_w=float(pseudo_w),
            dice_ema=float(dice_ema) if dice_ema is not None else None,
            diary_path=str(diary.path) if diary.path else None,
            tier3_best_bio=str(model_dir / "tier3_best_bio.pth"),
            tier3_latest_checkpoint=str(latest_ckpt_save),
            interrupted=bool(interrupted),
            last_epoch_completed=last_epoch_completed,
        )

    atexit.register(lambda: _emit_tier3_run_end(True))

    print("\n🚀 --- Starting Phase 3: Segregated Bio-Fluid Coupling ---")

    watchdog_sec = float(os.environ.get("TIER3_BATCH_WATCHDOG_SEC", "300"))

    for epoch in range(start_epoch, epochs):
        last_epoch_completed = epoch
        wu = curriculum.tier3_warmup_epochs

        ease = curriculum.tier3_curriculum_easing
        if epoch < wu:
            # --- STAGE A: THE PREDICTOR ---
            current_mu_ratio = 1.0  # Force strictly neutral rheology
            span = max(float(wu - 1), 1.0)
            t_w = _ease01(epoch / span, ease)
            current_T_scale = curriculum.tier3_t_scale_warmup_initial - t_w * (
                curriculum.tier3_t_scale_warmup_initial - curriculum.tier3_t_scale_warmup_final
            )

            # Freeze LoRA to prevent overfitting to the static flow field
            if epoch == 0:
                print("🔒 Stage A (Predictor): Freezing LoRA layers for pure transport learning.")
            for _name, param in model.named_parameters():
                if "lora" in _name.lower():
                    param.requires_grad = False
        else:
            # --- STAGE B: THE CORRECTOR ---
            coupled_denom = max(1, epochs - wu - 1)
            progress = _ease01((epoch - wu) / float(coupled_denom), ease)
            current_mu_ratio = bio_cfg.mu_ratio_init + progress * (
                bio_cfg.mu_ratio_max - bio_cfg.mu_ratio_init
            )
            current_T_scale = curriculum.tier3_t_scale_coupled_initial - progress * (
                curriculum.tier3_t_scale_coupled_initial - curriculum.tier3_t_scale_coupled_final
            )

            # Unfreeze LoRA to allow kinematic co-adaptation
            if epoch == wu:
                print("🔥 Stage B (Corrector): Unfreezing LoRA layers. Activating rheological feedback.")
            for _name, param in model.named_parameters():
                if "lora" in _name.lower():
                    param.requires_grad = True

        # Push updates to the network and kernels
        model.mu_ratio_max = current_mu_ratio

        # Unify the curriculum temperature
        model.T_scale = current_T_scale
        kernels.kinetics.T_scale = current_T_scale

        # FIX: Capitalized 'T' here as well
        print(f"\n⏳ Epoch {epoch:02d} | mu_ratio: {current_mu_ratio:.1f}x | T_scale: {current_T_scale:.2f}")

        if curriculum.tier3_weighter_freeze_during_warmup:
            phys_start = wu + int(curriculum.tier3_weighter_physics_grace_epochs)
            if epoch < wu:
                loss_weighter.log_vars.requires_grad_(False)
            elif epoch < phys_start:
                loss_weighter.log_vars.requires_grad_(False)
                loss_weighter.log_vars[6:].requires_grad_(True)
                if epoch == wu:
                    print(
                        "⚖️  Tier 3 warmup done: unfreezing **data** Kendall log_vars "
                        f"(indices 6–7); physics log_vars frozen until epoch {phys_start}."
                    )
            else:
                loss_weighter.log_vars.requires_grad_(True)
                if epoch == phys_start:
                    print("⚖️  Unfreezing **physics** Kendall log_vars after grace period.")
        else:
            loss_weighter.log_vars.requires_grad_(True)

        model.train()
        total_loss_epoch = 0.0
        optimizer.zero_grad()

        # Epoch-level TF schedule (matches compute_tier3_loss; per-batch TF_eff may be 0 without truth nodes).
        if epoch < wu:
            teacher_forcing_ratio = 1.0
        else:
            decay_progress = (epoch - wu) / float(curriculum.tier3_teacher_force_decay_epochs)
            decay_progress = _ease01(decay_progress, curriculum.tier3_curriculum_easing)
            teacher_forcing_ratio = max(0.0, 1.0 - decay_progress)

        # EMA-smoothed progress metrics for less noisy tqdm feedback.
        ema_metrics = None
        ema_alpha = 0.05
        anchor_supervised_batches = 0
        pseudo_supervised_batches = 0
        no_grad_skipped_batches = 0
        total_batches = 0
        ode_zero_batches = 0

        pbar = tqdm(loader, desc=f"Tier 3 Ep {epoch:02d}")
        for batch_idx, data in enumerate(pbar):
            total_batches += 1
            batch_t0 = time.perf_counter()
            data = data.to(device)
            data.x.requires_grad_(True)
            data_src = _tier3_data_source_key(data)
            pseudo_target = pseudo_bank.get(data_src) if (data_src is not None and data_src in pseudo_bank) else None

            dbg = (epoch, batch_idx) if _tier3_should_log_batch(epoch, batch_idx) else None
            loss, metrics = compute_tier3_loss(
                model,
                data,
                kernels,
                loss_weighter,
                device,
                bio_cfg,
                epoch=epoch,
                total_epochs=epochs,
                curriculum=curriculum,
                debug_batch=dbg,
                pseudo_target_trajectory=pseudo_target,
                pseudo_loss_weight=pseudo_w,
            )
            if metrics.get("Has_Anchor_Supervision", 0.0) > 0.5:
                anchor_supervised_batches += 1
            if metrics.get("Has_Pseudo_Supervision", 0.0) > 0.5:
                pseudo_supervised_batches += 1
            if int(metrics.get("ODE_Evals", 0)) == 0 and float(metrics.get("PDE_Steps", 0.0)) > 0.5:
                ode_zero_batches += 1
            loss = loss / accumulation_steps

            if torch.isnan(loss):
                print(f"\n⚠️ NaN detected in loss at epoch {epoch}! Skipping micro-batch.")
                continue

            if not loss.requires_grad:
                src = getattr(data, "_tier3_path", "<unknown>")
                no_grad_skipped_batches += 1
                _tier3_dbg_line(
                    "⚠️ Tier3 loss has no grad_fn before backward(); skipping micro-batch. "
                    f"epoch={epoch} batch={batch_idx} src={src} "
                    f"TF_eff={metrics.get('TF_eff')} "
                    f"Has_Anchor_Supervision={metrics.get('Has_Anchor_Supervision')} "
                    f"Has_Pseudo_Supervision={metrics.get('Has_Pseudo_Supervision')} "
                    f"L_ADR_F={metrics.get('L_ADR_F'):.3e} "
                    f"L_W_Phy={metrics.get('L_W_Phy'):.3e} "
                    f"L_Data_Kine={metrics.get('L_Data_Kine'):.3e} "
                    f"L_Data_Bio={metrics.get('L_Data_Bio'):.3e}"
                )
                continue
            loss.backward()
            if _tier3_should_log_batch(epoch, batch_idx):
                sq = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        g = p.grad.detach().data
                        sq += float((g * g).sum().item())
                _tier3_dbg_line(
                    f"[TIER3_DEBUG] epoch={epoch} batch={batch_idx} grad_L2={math.sqrt(sq):.4e} (micro-batch)"
                )

            if ((batch_idx + 1) % accumulation_steps == 0) or (batch_idx + 1 == len(loader)):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            batch_dt = time.perf_counter() - batch_t0
            if batch_dt > watchdog_sec:
                _tier3_dbg_line(
                    f"⏱️ [Watchdog] slow batch epoch={epoch} batch={batch_idx} "
                    f"dt={batch_dt:.2f}s ODE_evals={int(metrics.get('ODE_Evals', 0))}"
                )

            current_l_tot = loss.item() * accumulation_steps
            total_loss_epoch += current_l_tot

            if ema_metrics is None:
                ema_metrics = {
                    "L_tot": current_l_tot,
                    "L_Data_Kine": metrics["L_Data_Kine"],
                    "L_Data_Bio": metrics["L_Data_Bio"],
                    "L_ADR_F": metrics['L_ADR_F'],
                    "L_W_Bio": metrics['L_W_Bio'],
                    "L_W_Phy": metrics['L_W_Phy']
                }
            else:
                ema_metrics["L_tot"] = (1 - ema_alpha) * ema_metrics["L_tot"] + ema_alpha * current_l_tot
                ema_metrics["L_Data_Kine"] = (1 - ema_alpha) * ema_metrics["L_Data_Kine"] + ema_alpha * metrics["L_Data_Kine"]
                ema_metrics["L_Data_Bio"] = (1 - ema_alpha) * ema_metrics["L_Data_Bio"] + ema_alpha * metrics["L_Data_Bio"]
                ema_metrics["L_ADR_F"] = (1 - ema_alpha) * ema_metrics["L_ADR_F"] + ema_alpha * metrics['L_ADR_F']
                ema_metrics["L_W_Bio"] = (1 - ema_alpha) * ema_metrics["L_W_Bio"] + ema_alpha * metrics['L_W_Bio']
                ema_metrics["L_W_Phy"] = (1 - ema_alpha) * ema_metrics["L_W_Phy"] + ema_alpha * metrics['L_W_Phy']

            pbar.set_postfix({
                "L_tot": f"{ema_metrics['L_tot']:.2e}",
                "L_Kine": f"{ema_metrics['L_Data_Kine']:.2e}",
                "L_Bio": f"{ema_metrics['L_Data_Bio']:.2e}",
                "L_ADR_F": f"{ema_metrics['L_ADR_F']:.2e}",
                "L_W_Bio": f"{ema_metrics['L_W_Bio']:.2e}",
                "L_W_Phy": f"{ema_metrics['L_W_Phy']:.2e}",
                "TF_eff": f"{metrics['TF_eff']:.2f}",
                "ODE": f"{int(metrics.get('ODE_Evals', 0))}",
                "t_batch": f"{batch_dt:.2f}s",
                "A_sup": f"{anchor_supervised_batches}/{total_batches}",
                "P_sup": f"{pseudo_supervised_batches}/{total_batches}",
            })

        scheduler.step()
        if total_batches > 0:
            frac = anchor_supervised_batches / float(total_batches)
            print(
                f"📌 Anchor-supervised batches: {anchor_supervised_batches}/{total_batches} "
                f"({frac:.1%})"
            )
            pfrac = pseudo_supervised_batches / float(total_batches)
            print(
                f"🧾 Pseudo-supervised batches: {pseudo_supervised_batches}/{total_batches} "
                f"({pfrac:.1%})"
            )
            if low_anchor_mode:
                print(
                    f"🩺 Low-anchor health: anchor_frac={frac:.1%}, pseudo_frac={pfrac:.1%}, "
                    f"pseudo_graph_cov={len(pseudo_bank)}/{max(1, len(train_physics))} ({pseudo_cov:.1%})"
                )
            if no_grad_skipped_batches > 0:
                print(
                    f"🛟 No-grad micro-batches skipped: {no_grad_skipped_batches}/{total_batches} "
                    f"({(no_grad_skipped_batches / float(total_batches)):.1%})"
                )
            if total_batches > 0 and ode_zero_batches / float(total_batches) > 0.5:
                print(
                    f"⚠️ Sanity: ODE_Evals==0 on {ode_zero_batches}/{total_batches} batches "
                    "despite PDE_Steps>0 — check TBPTT windows / synthetic batches."
                )

        train_loss_mean = float(total_loss_epoch) / max(float(total_batches), 1.0)

        # Validation & Metrics
        val_log: Optional[Dict[str, Any]] = None
        if epoch % 2 == 0:
            model.eval()
            val_dice_total, val_pearson_total, val_fibrin_total = 0.0, 0.0, 0.0
            val_anchor_dice_sum, val_synth_dice_sum = 0.0, 0.0
            n_val_anchor, n_val_synth = 0, 0
            wss_reason_hist: Dict[str, int] = {}
            viz_every = int(os.environ.get("TIER3_VAL_VIZ_EVERY", "0"))
            viz_dir = reports_dir() / "tier3_val_viz"

            with torch.no_grad():
                safe_vars = loss_weighter.clamped_log_vars()
                weights = torch.exp(-safe_vars)

                print(
                    f"⚖️ Learned Weights -> ADR_F: {weights[0]:.2f} | ADR_S: {weights[1]:.2f} | "
                    f"W_Bio: {weights[2]:.2f} | W_Phys: {weights[3]:.2f} | Bio_IO: {weights[4]:.2f} | "
                    f"NS_mom: {weights[5]:.2f} | Data_Kine: {weights[6]:.2f} | Data_Bio: {weights[7]:.2f}"
                )
                learned_w = {
                    "w_ADR_F": float(weights[0].item()),
                    "w_ADR_S": float(weights[1].item()),
                    "w_W_Bio": float(weights[2].item()),
                    "w_W_Phys": float(weights[3].item()),
                    "w_Bio_IO": float(weights[4].item()),
                    "w_NS_mom": float(weights[5].item()),
                    "w_Data_Kine": float(weights[6].item()),
                    "w_Data_Bio": float(weights[7].item()),
                }

                for v_data in val_loader:
                    v_data = v_data.to(device)
                    val_eval_times = bio_cfg.resolve_tier3_times(v_data, device)
                    v_pred = model(v_data, val_eval_times)
                    if isinstance(v_pred, tuple):
                        v_pred = v_pred[0]

                    d, p, f, wss_diag = calculate_validation_metrics(v_pred[-1], v_data, kernels, device)
                    val_dice_total += d
                    val_pearson_total += p
                    val_fibrin_total += f
                    rk = str(wss_diag.get("wss_pearson_reason", "unset"))
                    wss_reason_hist[rk] = wss_reason_hist.get(rk, 0) + 1
                    is_anc = _graph_has_anchor_nodes(v_data)
                    if is_anc:
                        val_anchor_dice_sum += d
                        n_val_anchor += 1
                    else:
                        val_synth_dice_sum += d
                        n_val_synth += 1
                    if viz_every > 0 and (epoch % viz_every == 0) and is_anc:
                        _tier3_save_val_debug_plot(viz_dir, epoch, v_pred[-1], v_data, kernels, device)

            model.train()
            n_val = max(len(val_loader), 1)
            avg_dice = val_dice_total / n_val
            avg_pearson = val_pearson_total / n_val
            avg_fibrin = val_fibrin_total / n_val

            if dice_ema is None:
                dice_ema_local = avg_dice
            else:
                dice_ema_local = (1.0 - dice_ema_beta) * dice_ema + dice_ema_beta * avg_dice
            composite_score = float(dice_ema_local) + ckpt_pearson_w * float(avg_pearson)
            dice_ema = dice_ema_local

            trust_tag = "" if metrics_trustworthy else " [HEALTH-ONLY: few anchors]"
            wss_top = sorted(wss_reason_hist.items(), key=lambda kv: -kv[1])[0][0] if wss_reason_hist else "n/a"
            print(
                f"📊 [Validation]{trust_tag} Clot Dice: {avg_dice:.4f} | "
                f"Patent WSS Pearson: {avg_pearson:.4f} | Max Fibrin (SI): {avg_fibrin:.2e}"
            )
            print(
                f"   WSS Pearson: top reason '{wss_top}' "
                f"(hist={dict(wss_reason_hist)})."
            )
            if n_val_anchor > 0 or n_val_synth > 0:
                ad = val_anchor_dice_sum / max(n_val_anchor, 1)
                sd = val_synth_dice_sum / max(n_val_synth, 1)
                print(
                    f"   Per-split Dice: anchor={ad:.4f} (n={n_val_anchor}) | "
                    f"synthetic={sd:.4f} (n={n_val_synth})"
                )

            if composite_score > best_composite:
                best_composite = composite_score
                model_dir.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), model_dir / "tier3_best_bio.pth")
                print(
                    f"⭐ Saved best checkpoint (composite={composite_score:.4f} = dice_ema + "
                    f"{ckpt_pearson_w:g}*pearson; dice_ema={dice_ema_local:.4f})"
                )

            val_log = {
                "avg_dice": avg_dice,
                "avg_pearson": avg_pearson,
                "avg_fibrin": avg_fibrin,
                "dice_ema": dice_ema_local,
                "composite_score": composite_score,
                "wss_reason_hist": dict(wss_reason_hist),
                "val_anchor_dice": val_anchor_dice_sum / max(n_val_anchor, 1) if n_val_anchor else None,
                "val_synth_dice": val_synth_dice_sum / max(n_val_synth, 1) if n_val_synth else None,
                "metrics_trustworthy": metrics_trustworthy,
            }
            diary.log_validation(epoch, val_log, **learned_w)

        diary.log_epoch_end(
            epoch,
            train_loss_mean=float(train_loss_mean),
            lr=float(optimizer.param_groups[0]["lr"]),
            mu_ratio=float(current_mu_ratio),
            T_scale=float(current_T_scale),
            teacher_forcing_ratio=float(teacher_forcing_ratio),
            best_composite_so_far=float(best_composite),
            dice_ema=float(dice_ema) if dice_ema is not None else None,
            total_batches=int(total_batches),
            anchor_supervised_batches=int(anchor_supervised_batches),
            pseudo_supervised_batches=int(pseudo_supervised_batches),
            low_anchor_mode=bool(low_anchor_mode),
            pseudo_w=float(pseudo_w),
            teacher_best_dice=float(teacher_best_dice),
        )
        log_row: Dict[str, Any] = {
            "epoch": int(epoch),
            "train_loss_mean_microbatch": float(train_loss_mean),
            "mu_ratio": float(current_mu_ratio),
            "T_scale": float(current_T_scale),
            "teacher_forcing_epoch": float(teacher_forcing_ratio),
            "metrics_trustworthy": metrics_trustworthy,
        }
        if val_log is not None:
            log_row.update(val_log)
        _tier3_append_jsonl(log_row)

        should_save_ckpt = ((epoch + 1) % ckpt_every == 0) or (epoch == epochs - 1)
        if should_save_ckpt:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss_weighter_state_dict": loss_weighter.state_dict(),
                "best_composite": best_composite,
                "dice_ema": dice_ema,
                "teacher_best_dice": teacher_best_dice,
                "pseudo_w": pseudo_w,
            }
            torch.save(checkpoint, latest_ckpt_save)
            print(f"💾 Saved Tier 3 checkpoint -> {latest_ckpt_save.name} (every {ckpt_every} epoch(s))")

    _emit_tier3_run_end(interrupted=False)


if __name__ == "__main__":
    try:
        train_t3_corrector()
    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user (KeyboardInterrupt).")
        raise
    except torch.cuda.OutOfMemoryError as e:
        print(f"\n💥 CUDA out of memory during Tier 3 training: {e}")
        raise
    except Exception as e:
        print(f"\n💥 Unhandled exception during Tier 3 training: {type(e).__name__}: {e}")
        raise