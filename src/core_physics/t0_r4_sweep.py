"""Unified T0 Rung4 architecture recipes: s4 variants, s5 FI/Mat heads, S-star stages.

S-star (deploy-faithful mini-models on s0 prior)::

    gate   -- where/when: GNN gate residual or risk reweight in E(t)
    species -- magnitude: FI/Mat delta toward gelation crit on gated nodes
    dyn    -- temporal: GRU smooths FI/Mat on active patches across macro steps

Each stage writes only inside E(t) from predicted commits. s0 remains frozen prior.
Checkpoint embeds ``recipe`` id + hyperparams for sweep replay.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

from src.config import BiochemConfig, PhysicsConfig
from src.core_physics.clot_continuous_time import macro_tau_at_index
from src.core_physics.clot_nucleation_mask import resolve_nucleation_eligibility
from src.core_physics.t0_mu_physics import predict_clot_phi_at_time
from src.core_physics.t0_r4_s2_species import (
    FEATURE_NAMES,
    T0R4S2LocMLP,
    _apply_loc_gate_residual,
    _apply_loc_risk_adjustment,
    _risk_n_at_time,
    _s0_gate_from_species,
    apply_s2_species_delta,
    build_s2_features,
    build_s3_features,
    feature_dim,
    s3_feature_dim,
    species_from_gate,
)
from src.core_physics.t0_r4_s3_temporal import S3_EXTRA_FEATURE_NAMES
from src.core_physics.t0_r4_s4_band_ml import T0R4S4BandGNN
from src.core_physics.t0_rung4_ladder import (
    FI_SLICE_IDX,
    MAT_SLICE_IDX,
    _build_s0_deploy_species,
    _s0_onset_factor,
    resting_species_log_nd,
    rung4_use_dgamma_wall_seed,
)
from src.core_physics.t0_rung_config import RUNG2_GAMMA_MODE, t0_rung2_env
from src.utils.paths import get_project_root

DEFAULT_SWEEP_CKPT = "outputs/biochem/t0_r4_sweep/best.pth"


@dataclass(frozen=True)
class SweepRecipe:
    """Training / rollout recipe for overnight sweep legs."""

    id: str
    family: str  # s4 | s5 | s_star | ref
    hypothesis: str
    gate: str = "none"  # none | gnn | risk_mlp
    species: str = "none"  # none | gate_ramp | gnn_delta | mlp_delta
    dyn: str = "none"  # none | gru_smooth
    hidden: int = 32
    loc_scale: float = 0.75
    delta_scale: float = 0.35
    dyn_alpha: float = 0.25
    w_fn: float = 3.0
    w_fp: float = 2.0
    w_commit: float = 0.0
    w_species: float = 0.0
    fn_target: float = 0.85
    fp_target: float = -0.85
    time_stride: int = 2
    early_stop: int = 10
    epochs: int = 32
    minutes: int = 25


RECIPES: dict[str, SweepRecipe] = {
    "ref_s0": SweepRecipe(
        id="ref_s0", family="ref", hypothesis="Deploy s0 rules baseline (eval only)",
        minutes=3, epochs=0,
    ),
    "s_star_g0_rules": SweepRecipe(
        id="s_star_g0_rules", family="s_star",
        hypothesis="S* G0: s0 deploy rules (eval only; alias ref_s0)",
        gate="none", species="gate_ramp", minutes=3, epochs=0,
    ),
    "smoke_s4": SweepRecipe(
        id="smoke_s4", family="s4", hypothesis="Plumbing: 4ep gate GNN",
        gate="gnn", species="gate_ramp", epochs=4, early_stop=3, minutes=5,
    ),
    # --- s4 family ---
    "s4_gate_gnn": SweepRecipe(
        id="s4_gate_gnn", family="s4",
        hypothesis="2L band GNN gate residual (current s4)",
        gate="gnn", species="gate_ramp", loc_scale=0.75,
    ),
    "s4_delta_gnn": SweepRecipe(
        id="s4_delta_gnn", family="s4",
        hypothesis="GNN FI/Mat delta on s0 species (bypass rank barrier)",
        gate="none", species="gnn_delta", delta_scale=0.40, w_species=1.0,
    ),
    "s4_gate_fpstrong": SweepRecipe(
        id="s4_gate_fpstrong", family="s4",
        hypothesis="Gate GNN + stronger FP suppression",
        gate="gnn", species="gate_ramp", w_fp=4.0, fp_target=-0.95, loc_scale=0.65,
    ),
    "s4_gate_commit": SweepRecipe(
        id="s4_gate_commit", family="s4",
        hypothesis="Gate GNN + commit BCE on FN/FP in E(t)",
        gate="gnn", species="gate_ramp", w_commit=1.5, loc_scale=0.75,
    ),
    "s4_risk_gnn": SweepRecipe(
        id="s4_risk_gnn", family="s4",
        hypothesis="GNN risk reweight before s0 top-8% (s2_loc + neighbors)",
        gate="risk_gnn", species="gate_ramp", loc_scale=1.25,
    ),
    # --- s5 family (narrow 2-ch writes) ---
    "s5_mlp_fimat": SweepRecipe(
        id="s5_mlp_fimat", family="s5",
        hypothesis="MLP writes FI/Mat delta on s0 gate hotspots",
        gate="none", species="mlp_delta", delta_scale=0.35, w_species=1.5, hidden=48,
    ),
    "s5_gnn_fimat": SweepRecipe(
        id="s5_gnn_fimat", family="s5",
        hypothesis="Band GNN writes FI/Mat delta in E(t)",
        gate="none", species="gnn_delta", delta_scale=0.40, w_species=1.5,
    ),
    "s5_gru_fimat": SweepRecipe(
        id="s5_gru_fimat", family="s5",
        hypothesis="GRU smooths FI/Mat delta on active s0 patches",
        gate="none", species="mlp_delta", dyn="gru_smooth",
        delta_scale=0.30, dyn_alpha=0.35, w_species=1.0, w_commit=0.5,
    ),
    # Canonical ladder step (standalone train via go_t0_rung4_s5.ps1)
    "s5_gnode_fimat": SweepRecipe(
        id="s5_gnode_fimat", family="s5",
        hypothesis="Narrow 2-ch: tanh FI/Mat delta on s0; FN delta-target + commit BCE",
        gate="none", species="gnn_delta", delta_scale=0.45,
        w_species=2.0, w_commit=0.75, w_fn=0.0, w_fp=0.25,
        hidden=32, epochs=40, early_stop=12, minutes=35,
    ),
    # --- S-star components ---
    "s_star_gate": SweepRecipe(
        id="s_star_gate", family="s_star",
        hypothesis="S* gate only: GNN where/when hotspot rank in E(t)",
        gate="gnn", species="gate_ramp", w_commit=1.0, loc_scale=0.85,
    ),
    "s_star_species": SweepRecipe(
        id="s_star_species", family="s_star",
        hypothesis="S* species only: GNN magnitude on frozen s0 gate",
        gate="none", species="gnn_delta", delta_scale=0.45, w_species=2.0, w_commit=0.75,
    ),
    "s_star_dyn": SweepRecipe(
        id="s_star_dyn", family="s_star",
        hypothesis="S* dynamics only: GRU temporal smooth on s0 species",
        gate="none", species="none", dyn="gru_smooth", dyn_alpha=0.30, w_species=1.0,
    ),
    "s_star_gate_species": SweepRecipe(
        id="s_star_gate_species", family="s_star",
        hypothesis="S* gate + species magnitude (no dyn)",
        gate="gnn", species="gnn_delta", loc_scale=0.75, delta_scale=0.35,
        w_commit=1.0, w_species=1.5,
    ),
    "s_star_full": SweepRecipe(
        id="s_star_full", family="s_star",
        hypothesis="S* full stack: gate + species + temporal smooth",
        gate="gnn", species="gnn_delta", dyn="gru_smooth",
        loc_scale=0.75, delta_scale=0.35, dyn_alpha=0.25,
        w_commit=1.25, w_species=1.5, w_fn=3.0, w_fp=2.5,
    ),
    "s_star_small_ml": SweepRecipe(
        id="s_star_small_ml", family="s_star",
        hypothesis="Small ML on rules: tiny gate+species, high commit weight",
        gate="gnn", species="mlp_delta", hidden=24, loc_scale=0.55, delta_scale=0.25,
        w_commit=2.0, w_species=1.0, w_fp=3.0, epochs=36, minutes=30,
    ),
}

DEFAULT_SWEEP_ORDER = [
    "smoke_s4",
    "ref_s0",
    "s4_gate_gnn",
    "s4_delta_gnn",
    "s4_gate_fpstrong",
    "s4_gate_commit",
    "s4_risk_gnn",
    "s5_mlp_fimat",
    "s5_gnn_fimat",
    "s5_gru_fimat",
    "s_star_gate",
    "s_star_species",
    "s_star_dyn",
    "s_star_gate_species",
    "s_star_full",
    "s_star_small_ml",
]


def recipe_from_id(recipe_id: str) -> SweepRecipe:
    rid = recipe_id.strip().lower()
    if rid not in RECIPES:
        known = ", ".join(sorted(RECIPES))
        raise KeyError(f"Unknown sweep recipe {recipe_id!r}; known: {known}")
    return RECIPES[rid]


def sweep_ckpt_path() -> Path:
    raw = (os.environ.get("T0_R4_SWEEP_CKPT") or DEFAULT_SWEEP_CKPT).strip()
    p = Path(raw)
    if not p.is_absolute():
        p = get_project_root() / p
    return p


class _BandConv(MessagePassing):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__(aggr="add")
        self.lin_nei = nn.Linear(in_dim, out_dim)
        self.lin_self = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.propagate(edge_index, x=x)

    def message(self, x_j: torch.Tensor, x_i: torch.Tensor) -> torch.Tensor:
        return F.silu(self.lin_nei(x_j) + self.lin_self(x_i))


class T0R4SweepGNN(nn.Module):
    """2L mesh GNN with configurable output width (1=logit, 2=FI/Mat delta)."""

    def __init__(self, in_dim: int, out_dim: int, *, hidden: int = 32):
        super().__init__()
        h = max(int(hidden), 16)
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.hidden = h
        self.conv1 = _BandConv(self.in_dim, h)
        self.conv2 = _BandConv(h, h)
        self.head = nn.Linear(h, self.out_dim)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.35)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.conv1(x, edge_index))
        h = F.silu(self.conv2(h, edge_index))
        return self.head(h)


class T0R4SweepMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, *, hidden: int = 48):
        super().__init__()
        h = max(int(hidden), 16)
        self.net = nn.Sequential(
            nn.Linear(in_dim, h),
            nn.SiLU(),
            nn.Linear(h, h),
            nn.SiLU(),
            nn.Linear(h, out_dim),
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.35)
                nn.init.zeros_(m.bias)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor | None = None) -> torch.Tensor:
        del edge_index
        return self.net(x)


class T0R4DynSmooth(nn.Module):
    """Per-node GRU that outputs FI/Mat correction on active patches."""

    def __init__(self, in_dim: int, *, hidden: int = 24):
        super().__init__()
        h = max(int(hidden), 8)
        self.gru = nn.GRUCell(in_dim, h)
        self.head = nn.Linear(h, 2)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def init_state(self, n: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(n, self.gru.hidden_size, device=device, dtype=torch.float32)

    def forward_step(
        self, feats: torch.Tensor, h: torch.Tensor, active: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h_new = self.gru(feats, h)
        delta = self.head(h_new)
        a = active.reshape(-1).bool()
        h_out = torch.where(a.unsqueeze(-1), h_new, h)
        d_out = torch.where(a.unsqueeze(-1), delta, torch.zeros_like(delta))
        return d_out, h_out


@dataclass
class T0R4SweepBundle:
    recipe: SweepRecipe
    gate_model: nn.Module | None
    species_model: nn.Module | None
    dyn_model: T0R4DynSmooth | None
    in_dim: int
    feat_dim_dyn: int
    device: torch.device
    dyn_state: torch.Tensor | None = field(default=None, repr=False)


def _feat_dim(recipe: SweepRecipe, *, with_memory: bool) -> int:
    return s3_feature_dim() if with_memory else feature_dim()


def _build_models(recipe: SweepRecipe, device: torch.device) -> T0R4SweepBundle:
    mem = recipe.dyn != "none" or recipe.species in ("gnn_delta", "mlp_delta") and recipe.gate != "none"
    in_dim = _feat_dim(recipe, with_memory=True)
    in_dim_base = _feat_dim(recipe, with_memory=False)

    gate_model: nn.Module | None = None
    if recipe.gate == "gnn":
        gate_model = T0R4SweepGNN(in_dim, 1, hidden=recipe.hidden).to(device)
    elif recipe.gate == "risk_gnn":
        gate_model = T0R4SweepGNN(in_dim_base, 1, hidden=recipe.hidden).to(device)
    elif recipe.gate == "risk_mlp":
        gate_model = T0R4S2LocMLP(in_dim=in_dim_base, hidden=recipe.hidden).to(device)

    species_model: nn.Module | None = None
    if recipe.species == "gnn_delta":
        species_model = T0R4SweepGNN(in_dim, 2, hidden=recipe.hidden).to(device)
    elif recipe.species == "mlp_delta":
        species_model = T0R4SweepMLP(in_dim, 2, hidden=recipe.hidden).to(device)

    dyn_model: T0R4DynSmooth | None = None
    feat_dyn = in_dim + 2
    if recipe.dyn == "gru_smooth":
        dyn_model = T0R4DynSmooth(feat_dyn, hidden=max(recipe.hidden // 2, 16)).to(device)

    return T0R4SweepBundle(
        recipe=recipe,
        gate_model=gate_model,
        species_model=species_model,
        dyn_model=dyn_model,
        in_dim=in_dim,
        feat_dim_dyn=feat_dyn,
        device=device,
    )


def _s0_context(
    data,
    t: int,
    device: torch.device,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    *,
    commits_prev: torch.Tensor | None,
    phi_prev: torch.Tensor | None,
):
    elig = resolve_nucleation_eligibility(
        data, t, device, phys_cfg, bio_cfg,
        commits_prev=commits_prev, growth_seed="pred", nucleation_hops=1,
        use_dgamma_wall_seed=rung4_use_dgamma_wall_seed(),
    ).reshape(-1).bool()
    s0_sp = _build_s0_deploy_species(
        data, t, device, bio_cfg, elig=elig, commits_prev=commits_prev,
    )
    gate = _s0_gate_from_species(s0_sp, data, device, bio_cfg, elig)
    onset = float(_s0_onset_factor(float(macro_tau_at_index(data, t, bio_cfg=bio_cfg))))
    return elig, s0_sp, gate, onset


def _feats_at_t(
    data,
    t: int,
    device: torch.device,
    bio_cfg: BiochemConfig,
    *,
    elig: torch.Tensor,
    s0_sp: torch.Tensor,
    s0_gate: torch.Tensor,
    commits_prev: torch.Tensor | None,
    phi_prev: torch.Tensor | None,
    with_memory: bool,
) -> torch.Tensor:
    if with_memory:
        return build_s3_features(
            data, t, device, bio_cfg, elig=elig, s0_species=s0_sp, s0_gate=s0_gate,
            commits_prev=commits_prev, phi_prev=phi_prev,
        )
    return build_s2_features(
        data, t, device, bio_cfg, elig=elig, s0_species=s0_sp, s0_gate=s0_gate,
    )


def apply_sweep_species_at_time(
    data,
    time_index: int,
    device: torch.device,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    bundle: T0R4SweepBundle,
    *,
    commits_prev: torch.Tensor | None,
    phi_prev: torch.Tensor | None = None,
    dyn_h: torch.Tensor | None = None,
    train: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, torch.Tensor]]:
    """Build species for one macro step; returns (species, new_dyn_h, aux_tensors)."""
    recipe = bundle.recipe
    t = int(time_index)
    elig, s0_sp, s0_gate, onset = _s0_context(
        data, t, device, phys_cfg, bio_cfg,
        commits_prev=commits_prev, phi_prev=phi_prev,
    )
    edge_index = data.edge_index.to(device=device)
    with_mem = recipe.dyn != "none" or recipe.gate == "gnn" or recipe.species == "gnn_delta"
    feats = _feats_at_t(
        data, t, device, bio_cfg, elig=elig, s0_sp=s0_sp, s0_gate=s0_gate,
        commits_prev=commits_prev, phi_prev=phi_prev, with_memory=with_mem,
    )

    gate = s0_gate.reshape(-1).clone()
    sp = s0_sp.clone()
    aux: dict[str, torch.Tensor] = {"logit": torch.zeros(int(data.num_nodes), device=device)}

    if recipe.gate == "gnn" and bundle.gate_model is not None:
        logit = bundle.gate_model(feats, edge_index).reshape(-1)
        gate = _apply_loc_gate_residual(
            s0_gate, logit, elig, onset=onset, loc_scale=recipe.loc_scale,
        )
        sp = species_from_gate(data, device, bio_cfg, gate)
        aux["logit"] = logit
    elif recipe.gate in ("risk_gnn", "risk_mlp") and bundle.gate_model is not None:
        risk_n = _risk_n_at_time(data, t, device, bio_cfg, elig=elig)
        logit = bundle.gate_model(feats, edge_index).reshape(-1)
        risk_adj = _apply_loc_risk_adjustment(
            risk_n, logit, elig, onset=onset, loc_scale=recipe.loc_scale,
        )
        sp = _build_s0_deploy_species(
            data, t, device, bio_cfg, elig=elig, commits_prev=commits_prev,
            risk_n_override=risk_adj,
        )
        gate = _s0_gate_from_species(sp, data, device, bio_cfg, elig)
        aux["logit"] = logit
    elif recipe.species == "gate_ramp":
        sp = species_from_gate(data, device, bio_cfg, gate)

    if recipe.species in ("gnn_delta", "mlp_delta") and bundle.species_model is not None:
        raw = bundle.species_model(feats, edge_index) * float(onset)
        # Bounded residual: unbounded linear delta collapses gelation at tiny weights.
        delta = torch.tanh(raw)
        sp = apply_s2_species_delta(sp, delta, elig, delta_scale=recipe.delta_scale)
        aux["delta"] = delta
        aux["delta_raw"] = raw

    new_h = dyn_h
    if recipe.dyn == "gru_smooth" and bundle.dyn_model is not None:
        if dyn_h is None:
            dyn_h = bundle.dyn_model.init_state(int(data.num_nodes), device)
        active = (gate.reshape(-1) > 0.08) & elig.reshape(-1).bool()
        dyn_in = torch.cat([feats, sp[:, FI_SLICE_IDX : MAT_SLICE_IDX + 1]], dim=1)
        d_smooth, new_h = bundle.dyn_model.forward_step(dyn_in, dyn_h, active)
        sp = apply_s2_species_delta(
            sp, d_smooth * float(recipe.dyn_alpha), elig, delta_scale=1.0,
        )
        aux["dyn_delta"] = d_smooth

    return sp, new_h, aux


@torch.no_grad()
def rollout_sweep_species_series(
    data,
    phys_cfg: PhysicsConfig,
    bio_cfg: BiochemConfig,
    device: torch.device,
    bundle: T0R4SweepBundle,
    *,
    nucleation_hops: int = 1,
) -> torch.Tensor:
    out = data.y.clone().to(device=device)
    commits_prev: torch.Tensor | None = None
    phi_prev: torch.Tensor | None = None
    dyn_h: torch.Tensor | None = None
    with t0_rung2_env():
        for t in range(int(data.y.shape[0])):
            sp, dyn_h, _ = apply_sweep_species_at_time(
                data, t, device, phys_cfg, bio_cfg, bundle,
                commits_prev=commits_prev, phi_prev=phi_prev, dyn_h=dyn_h,
            )
            out[t, :, 4:16] = sp
            phi_raw, _ = predict_clot_phi_at_time(
                data, t, phys_cfg, bio_cfg, device,
                gamma_mode=RUNG2_GAMMA_MODE, flow_source="gt", pred_species_series=out,
            )
            phi_prev = phi_raw.reshape(-1).clamp(0.0, 1.0)
            commits_prev = (phi_prev >= 0.5).bool()
    return out


def save_sweep_checkpoint(
    path: str | Path,
    bundle: T0R4SweepBundle,
    *,
    meta: dict[str, Any] | None = None,
) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "recipe": asdict(bundle.recipe),
        "recipe_id": bundle.recipe.id,
        "in_dim": bundle.in_dim,
        "feat_dim_dyn": bundle.feat_dim_dyn,
        "meta": meta or {},
    }
    if bundle.gate_model is not None:
        payload["gate_state"] = bundle.gate_model.state_dict()
    if bundle.species_model is not None:
        payload["species_state"] = bundle.species_model.state_dict()
    if bundle.dyn_model is not None:
        payload["dyn_state"] = bundle.dyn_model.state_dict()
    torch.save(payload, out)
    side = {k: v for k, v in payload.items() if not k.endswith("_state")}
    out.with_suffix(".json").write_text(json.dumps(side, indent=2), encoding="utf-8")


def load_sweep_bundle(
    ckpt_path: str | Path | None = None,
    *,
    device: torch.device | None = None,
    quiet: bool = False,
) -> T0R4SweepBundle | None:
    path = Path(ckpt_path) if ckpt_path is not None else sweep_ckpt_path()
    if not path.is_file():
        if not quiet:
            print(f"[WARN] sweep checkpoint missing: {path}")
        return None
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(path, map_location=dev, weights_only=False)
    rid = str(payload.get("recipe_id") or payload.get("recipe", {}).get("id", ""))
    recipe = recipe_from_id(rid) if rid in RECIPES else SweepRecipe(**payload["recipe"])
    bundle = _build_models(recipe, dev)
    if bundle.gate_model is not None and "gate_state" in payload:
        bundle.gate_model.load_state_dict(payload["gate_state"])
    if bundle.species_model is not None and "species_state" in payload:
        bundle.species_model.load_state_dict(payload["species_state"])
    if bundle.dyn_model is not None and "dyn_state" in payload:
        bundle.dyn_model.load_state_dict(payload["dyn_state"])
    for m in (bundle.gate_model, bundle.species_model, bundle.dyn_model):
        if m is not None:
            m.eval()
    return bundle


def feature_names_for_recipe(recipe: SweepRecipe) -> list[str]:
    names = list(FEATURE_NAMES)
    if recipe.dyn != "none" or recipe.gate == "gnn" or recipe.species == "gnn_delta":
        names = names + list(S3_EXTRA_FEATURE_NAMES)
    return names
