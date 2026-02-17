from pathlib import Path

def get_project_root() -> Path:
    """Returns the project root folder."""
    current_path = Path(__file__).resolve()
    for parent in [current_path] + list(current_path.parents):
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent
    return current_path.parent