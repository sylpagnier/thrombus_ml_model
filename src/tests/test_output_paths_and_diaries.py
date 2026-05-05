from __future__ import annotations

import src.utils.paths as paths_mod
import src.utils.training_diary as diary_mod


def test_resolve_checkpoint_prefers_new_names_but_reads_legacy(tmp_path, monkeypatch):
    monkeypatch.setattr(paths_mod, "get_project_root", lambda: tmp_path)

    legacy = tmp_path / "outputs" / "stage_a" / "kinematics_best.pth"
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_text("legacy", encoding="utf-8")

    resolved = paths_mod.resolve_checkpoint("a", "kinematics_best.pth")

    assert resolved == legacy
    assert paths_mod.kinematics_dir() == tmp_path / "outputs" / "kinematics"
    assert paths_mod.biochem_dir() == tmp_path / "outputs" / "biochem"
    assert paths_mod.stage_a_dir() == paths_mod.kinematics_dir()
    assert paths_mod.stage_b_dir() == paths_mod.biochem_dir()


def test_training_diary_keeps_only_five_most_recent_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(paths_mod, "get_project_root", lambda: tmp_path)
    monkeypatch.delenv("KINEMATICS_TRAINING_DIARY_PATH", raising=False)
    monkeypatch.delenv("KINEMATICS_TRAINING_DIARY", raising=False)

    base = tmp_path / "outputs" / "reports" / "training" / "kinematics"
    for idx in range(1, 7):
        run_dir = base / f"2026042{idx}T184600Z"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "training_diary_main.jsonl").write_text("{}", encoding="utf-8")

    diary = diary_mod.TrainingDiary("kinematics")

    kept = sorted(p.name for p in base.iterdir() if p.is_dir())

    assert diary.run_dir is not None
    assert diary.run_dir.exists()
    assert len(kept) == 5
    assert "20260421T184600Z" not in kept
