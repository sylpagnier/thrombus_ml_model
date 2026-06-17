"""Canonical SciML model identifiers for HemoGINO.

Single source of truth for stack/component IDs, SciML categories, and legacy aliases.
Human-readable rationale: docs/MODEL_NOMENCLATURE.md
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SciMLModel:
    """One trainable component, physics closure, or composable pipeline."""

    id: str
    display_name: str
    sciml_category: str
    architecture: str
    code_class: str | None = None
    acronym: str = ""
    legacy_ids: tuple[str, ...] = ()
    distinguishing_features: tuple[str, ...] = ()

    def matches(self, name: str) -> bool:
        key = (name or "").strip().lower()
        if not key:
            return False
        if key == self.id.lower():
            return True
        if self.acronym:
            acr = self.acronym.lower().replace("-", "").replace("_", "")
            norm_key = key.replace("-", "").replace("_", "")
            if norm_key == acr:
                return True
        return key in {x.lower() for x in self.legacy_ids}


# --- Stage A: kinematics flow surrogate (RGP-DEQ) ---

PMGP_DEQ_KINE = SciMLModel(
    id="pmgp_deq_kine",
    acronym="RGP-DEQ",
    display_name="mu-coupled rheology-guided graph-perceiver DEQ (Stage A flow)",
    sciml_category="rheology-coupled graph DEQ with physics-modulated attention",
    architecture=(
        "Fourier node encoding -> MLP encoder -> Anderson/Picard DEQ fixed point "
        "z* = f(z*, mu(z*)); each DEQ step = physics-modulated multi-head GAT "
        "(adv/rheo/curvature log-modulators on edge attention) + Perceiver-style "
        "global token cross-attention; SIREN or linear decode [u,v,p]; sigmoid mu head"
    ),
    code_class="GINO_DEQ",
    distinguishing_features=(
        "physics-modulated GAT: edge attention logits biased by advection, "
        "wall-rheology, and curvature priors (SDF-decayed)",
        "Perceiver global mixing: fixed global tokens cross-attend the mesh, "
        "then broadcast back (strictly within-graph batched)",
        "mu feedback inside DEQ loop: mu(z) re-encodes into latent before each "
        "equilibrium step (rheology-coupled fixed point, not post-hoc mu head only)",
    ),
    legacy_ids=(
        "rgp_deq_kine",
        "rgp-deq-kine",
        "pmgp_deq_kine",
        "pmgp-deq-kine",
        "pmgp-deq",
        "pmgp_deq",
        "gino_deq_kine",
        "gino-deq-kine",
        "gino_deq",
        "pi_gnn_deq",
        "pi-gnn-deq",
        "kinematics",
        "stage_a_kine",
    ),
)

# Backward-compatible alias for imports predating RGP naming
GINO_DEQ_KINE = PMGP_DEQ_KINE

# --- Deploy biochem: learned species operator ---

SPECIES_GRAPHSAGE = SciMLModel(
    id="species_graphsage",
    display_name="Wall-band GraphSAGE species pushforward",
    sciml_category="discrete-time graph autoregressive operator (learned dynamics)",
    architecture=(
        "3-layer GraphSAGE on wall-band subgraph; inputs = frozen kinematic latent z_kin "
        "+ normalized SDF; continuous deploy variant uses dual-head spatial gate x magnitude "
        "delta for FI/Mat log-ND; autoregressive pushforward rollout"
    ),
    code_class="SpeciesDualHeadContinuousGNN",
    legacy_ids=("species_gnn", "species_snapshot_gnn", "wall_band_species_gnn"),
)

GELATION_BETA = SciMLModel(
    id="gelation_beta",
    display_name="Global Mat gelation scale calibrator",
    sciml_category="scalar calibration (1 learned parameter)",
    architecture="Single global multiplier on Mat channel before physics gelation readout",
    code_class=None,
    legacy_ids=("viscosity_beta",),
)

CLOT_TRIGGER_PHYSICS = SciMLModel(
    id="clot_trigger_physics",
    display_name="Mechanistic clot trigger readout",
    sciml_category="physics closure / mechanistic readout (not learned)",
    architecture=(
        "Carreau-Yasuda mu from shear rate + COMSOL-faithful gelation multiplier "
        "(1 + mu1(Mat) + mu2(FI)); clot phi from mu threshold with nucleation mask "
        "projection on wall-adjacent band"
    ),
    code_class=None,
    legacy_ids=("clot_phi", "clot_phi_physics", "physics_nucleation"),
)

FLOW_COUPLING = SciMLModel(
    id="flow_coupling",
    display_name="Closed-loop mu -> flow refresh",
    sciml_category="hybrid coupling stage (planned, not trained in baseline)",
    architecture=(
        "Predicted mu_eff_si fed back into RGP-DEQ as MU_PRIOR for refreshed [u,v]"
    ),
    code_class=None,
    legacy_ids=("flow_coupling_adr",),
)

# --- Full deploy stack ---

BIOCHEM_DEPLOY_STACK = SciMLModel(
    id="biochem_deploy",
    display_name="Hybrid biochem deploy pipeline",
    sciml_category="composable hybrid SciML (multi-module, not one nn.Module)",
    architecture=(
        "pmgp_deq_kine (frozen) -> species_graphsage (trained) -> gelation_beta (trained) "
        "-> clot_trigger_physics (equations) -> [future] flow_coupling"
    ),
    code_class="BiochemDeployStack",
    legacy_ids=("biochem_gnn", "clot_deploy_gnn", "species_gnn_deploy", "species_gnn_deploy_baseline"),
)

# --- Research biochem (GNODE path) ---

GNODE_BIOCHEM = SciMLModel(
    id="gnode_biochem",
    display_name="Graph Neural ODE biochem corrector",
    sciml_category="graph neural ODE (continuous-time latent dynamics)",
    architecture=(
        "Full-mesh GNODE_Phase3: torchdiffeq odeint on latent state; derivative block reuses "
        "PMGP-style physics-modulated GAT (legacy GINOBlock); frozen or co-trained kine "
        "backbone; learned mu/species heads and PDE/ADR losses"
    ),
    code_class="GNODE_Phase3",
    legacy_ids=("biochem_corrector", "gnode_phase3", "train_biochem", "t3"),
)

# Registry for lookup helpers
_ALL_MODELS: tuple[SciMLModel, ...] = (
    PMGP_DEQ_KINE,
    SPECIES_GRAPHSAGE,
    GELATION_BETA,
    CLOT_TRIGGER_PHYSICS,
    FLOW_COUPLING,
    BIOCHEM_DEPLOY_STACK,
    GNODE_BIOCHEM,
)


def resolve_model_id(name: str, *, default: str | None = None) -> str:
    """Map legacy or canonical name to canonical ``SciMLModel.id``."""
    key = (name or "").strip()
    if not key:
        if default is None:
            raise ValueError("empty model name")
        return default
    for model in _ALL_MODELS:
        if model.matches(key):
            return model.id
    if default is not None:
        return default
    return key


def is_legacy_kine_id(name: str) -> bool:
    """True when ``name`` is a pre-PMGP alias (``gino_deq_kine``, etc.)."""
    key = (name or "").strip().lower()
    if not key or key == PMGP_DEQ_KINE.id.lower():
        return False
    return PMGP_DEQ_KINE.matches(key) and key != PMGP_DEQ_KINE.id.lower()


def is_legacy_stack_id(name: str) -> bool:
    """True when ``name`` is a known alias, not the canonical stack id."""
    key = (name or "").strip().lower()
    return bool(key) and key != BIOCHEM_DEPLOY_STACK.id.lower() and BIOCHEM_DEPLOY_STACK.matches(key)


def stack_display_line() -> str:
    """One-line stack summary for logs and manifests."""
    return (
        f"{BIOCHEM_DEPLOY_STACK.id}: "
        f"{PMGP_DEQ_KINE.id} ({PMGP_DEQ_KINE.acronym}) + {SPECIES_GRAPHSAGE.id} + "
        f"{GELATION_BETA.id} + {CLOT_TRIGGER_PHYSICS.id}"
    )


def pmgp_deq_feature_lines() -> tuple[str, ...]:
    """Bullet lines for training banners / paper methods (ASCII-safe)."""
    return PMGP_DEQ_KINE.distinguishing_features
