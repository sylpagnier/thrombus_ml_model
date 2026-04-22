from pathlib import Path


def get_project_root() -> Path:
    """Returns the project root folder."""
    current_path = Path(__file__).resolve()
    for parent in [current_path] + list(current_path.parents):
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent
    return current_path.parent


def outputs_root() -> Path:
    """Unified generated-artifact root (checkpoints, reports, logs)."""
    return get_project_root() / "outputs"


def data_root() -> Path:
    """Canonical dataset tree: ``data/raw``, ``data/processed``, ``data/benchmark`` (scratch), etc."""
    return get_project_root() / "data"


def comsol_models_dir() -> Path:
    """Reference COMSOL projects and templates (versioned assets, not training checkpoints)."""
    return get_project_root() / "comsol_models"


def reports_dir() -> Path:
    """Generated reports: CSVs, figures, validation PNGs, training diaries, debug logs → ``outputs/reports``."""
    p = outputs_root() / "reports"
    p.mkdir(parents=True, exist_ok=True)
    return p


def reports_subdir(*parts: str) -> Path:
    """Create and return a scoped reports subdirectory under ``outputs/reports``."""
    p = reports_dir().joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


def reports_training_dir(*parts: str) -> Path:
    """Training-origin artifacts under ``outputs/reports/training``."""
    return reports_subdir("training", *parts)


def reports_evaluation_dir(*parts: str) -> Path:
    """Evaluation / benchmark artifacts under ``outputs/reports/evaluation``."""
    return reports_subdir("evaluation", *parts)


def reports_inspection_dir(*parts: str) -> Path:
    """Inspection-tool artifacts under ``outputs/reports/inspection``."""
    return reports_subdir("inspection", *parts)


def stage_a_dir() -> Path:
    """Predictor warm-up & Newtonian / Tier-1–2 checkpoints."""
    p = outputs_root() / "stage_a"
    p.mkdir(parents=True, exist_ok=True)
    return p


def stage_b_dir() -> Path:
    """Corrector (coupled biochem + non-Newtonian) checkpoints."""
    p = outputs_root() / "stage_b"
    p.mkdir(parents=True, exist_ok=True)
    return p


def resolve_checkpoint(stage: str, filename: str) -> Path:
    """Canonical checkpoint path: ``outputs/stage_a/<filename>`` or ``outputs/stage_b/<filename>``."""
    base = stage_a_dir() if stage == "a" else stage_b_dir()
    return base / filename
