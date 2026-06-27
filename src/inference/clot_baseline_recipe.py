"""Canonical clot baseline recipe (Lane A) and runtime manifest I/O."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.utils.paths import biochem_dir, get_project_root


@dataclass
class ClotBaselineRecipe:
    """Frozen deploy recipe for geometry -> clot maps (GraphSAGE species + clot-phi).

    Species temporal evolution comes from the GraphSAGE ``biochem_deploy`` stack
    (``species_ckpt``); the learned clot-phi MLP (``clot_phi_ckpt``) reads the
    rolled-out species block and produces the deploy mu map.

    ``teacher_ckpt`` is the legacy (pre-2026-06 GNODE) field, retained only so old
    manifests still load; it is ignored by the GraphSAGE predictor.
    """

    name: str = "lane_a"
    description: str = "GraphSAGE species pushforward + hybrid clot-phi MLP mu map"
    mu_ratio_max: float = 20.0
    pred_kine: bool = False
    june_anchor_dir: str = "outputs/biochem/gnode_8h_ladder/anchors_stride_72"
    dump_anchor_dir: str = "outputs/biochem/gnode10_sweep/anchors_gnode12_predkine_uvp"
    species_ckpt: str = ""
    teacher_ckpt: str = ""  # legacy GNODE field (ignored; kept for manifest back-compat)
    clot_phi_ckpt: str = ""
    mu_unlock_epochs: int = 6
    clot_epochs: int = 35
    clot_leg: str = "gnode12_lane_a_clotphi"
    init_teacher_ckpt: str = ""
    clot_phi_env: dict[str, str] = field(default_factory=dict)
    deploy_mu_map_env: dict[str, str] = field(default_factory=dict)

    def resolved_paths(self, root: Path | None = None) -> dict[str, Path]:
        root = root or get_project_root()
        out: dict[str, Path] = {}
        for key in ("june_anchor_dir", "dump_anchor_dir", "species_ckpt", "teacher_ckpt", "clot_phi_ckpt", "init_teacher_ckpt"):
            rel = getattr(self, key, "") or ""
            if rel:
                p = Path(rel)
                out[key] = p if p.is_absolute() else root / p
        return out

    def apply_clot_phi_env(self) -> None:
        """Set clot-phi training/eval env from recipe (Lane A defaults)."""
        import os

        defaults = default_lane_a_clot_phi_env()
        defaults.update(self.clot_phi_env or {})
        defaults["CLOT_PHI_ANCHOR_DIR"] = self.dump_anchor_dir
        for k, v in defaults.items():
            os.environ[k] = str(v)


def default_lane_a_clot_phi_env() -> dict[str, str]:
    return {
        "CLOT_PHI_MODEL": "mlp",
        "CLOT_PHI_HIDDEN": "32",
        "CLOT_PHI_MLP_DEPTH": "2",
        "CLOT_PHI_DROPOUT": "0.15",
        "CLOT_PHI_HYBRID": "1",
        "CLOT_PHI_MINIMAL_FEATURES": "1",
        "CLOT_PHI_JOINT_BIO": "1",
        "CLOT_PHI_JOINT_USE_PRED_SPECIES": "1",
        "CLOT_PHI_PHYSICS_BLEND": "1",
        "CLOT_PHI_PHYSICS_BLEND_ALPHA": "0.75",
        "CLOT_PHI_DGAMMA_FEATURE_TIME": "current",
        "CLOT_PHI_VEL_SOURCE": "gt",
        "CLOT_PHI_DGAMMA_SLICE": "1",
        "CLOT_PHI_MASK_MODE": "neighbor",
    }


def default_lane_a_recipe() -> ClotBaselineRecipe:
    root = get_project_root()
    promoted_species = root / "outputs/biochem/clot_baseline/species_best.pth"
    promoted_clot = root / "outputs/biochem/clot_baseline/clot_phi_best.pth"
    # GraphSAGE species pushforward checkpoint (biochem_deploy stack).
    from src.biochem_gnn.config import global_ckpt_path

    if promoted_species.is_file():
        species = str(promoted_species.relative_to(root))
    else:
        canonical = global_ckpt_path()
        try:
            species = str(canonical.relative_to(root))
        except ValueError:
            species = str(canonical)
    clot = (
        str(promoted_clot.relative_to(root))
        if promoted_clot.is_file()
        else "outputs/biochem/passive_species_focus_compare/gnode12_lane_a_clotphi/clot_phi_best.pth"
    )
    return ClotBaselineRecipe(
        species_ckpt=species,
        clot_phi_ckpt=clot,
        clot_phi_env=default_lane_a_clot_phi_env(),
    )


def baseline_dir() -> Path:
    d = biochem_dir() / "clot_baseline"
    d.mkdir(parents=True, exist_ok=True)
    return d


def baseline_manifest_path() -> Path:
    return baseline_dir() / "manifest.json"


def recipe_reference_path() -> Path:
    return get_project_root() / "data/reference/clot_baseline_lane_a.json"


def load_recipe_json(path: str | Path | None = None) -> ClotBaselineRecipe:
    p = Path(path) if path else recipe_reference_path()
    if not p.is_file():
        return default_lane_a_recipe()
    raw = json.loads(p.read_text(encoding="utf-8"))
    known = {f.name for f in ClotBaselineRecipe.__dataclass_fields__.values()}
    kwargs = {k: v for k, v in raw.items() if k in known}
    return ClotBaselineRecipe(**kwargs)


def save_manifest(
    recipe: ClotBaselineRecipe,
    *,
    eval_summary: dict[str, Any] | None = None,
    path: Path | None = None,
) -> Path:
    out = path or baseline_manifest_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "recipe": asdict(recipe),
        "eval": eval_summary or {},
        "version": 1,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[save] manifest -> {out}", flush=True)
    return out


def load_manifest(path: str | Path | None = None) -> tuple[ClotBaselineRecipe, dict[str, Any]]:
    p = Path(path) if path else baseline_manifest_path()
    if not p.is_file():
        return default_lane_a_recipe(), {}
    raw = json.loads(p.read_text(encoding="utf-8"))
    recipe_dict = raw.get("recipe") or raw
    known = {f.name for f in ClotBaselineRecipe.__dataclass_fields__.values()}
    kwargs = {k: v for k, v in recipe_dict.items() if k in known}
    return ClotBaselineRecipe(**kwargs), dict(raw.get("eval") or {})
