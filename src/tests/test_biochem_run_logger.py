from __future__ import annotations

import json

import src.utils.paths as paths_mod
import src.utils.training_diary as diary_mod


def test_biochem_run_logger_writes_compact_run_and_index(tmp_path, monkeypatch):
    monkeypatch.setattr(paths_mod, "get_project_root", lambda: tmp_path)
    monkeypatch.delenv("BIOCHEM_TRAINING_LOG_INDEX", raising=False)

    log = diary_mod.BiochemRunLogger()
    assert log.run_dir is not None
    assert log.path is not None

    log.log_meta(run_note="smoke", n_train_anchors=5)
    log.log_val(
        "teacher",
        0,
        {
            "mu_log_mae": 1.2,
            "mu_log_mae_wall": 2.0,
            "mu_log_mae_high_mu": 0.9,
            "mu_pearson": 0.3,
        },
        dbg_gate_mean_wall=0.0,
    )
    log.log_val("teacher", 2, {"mu_log_mae": 0.8, "mu_log_mae_wall": 1.7, "mu_pearson": 0.4})
    log.log_end(teacher_best_mu_score=-0.8, teacher_best_epoch=2, stop_after_teacher=True)

    lines = log.path.read_text(encoding="utf-8").strip().splitlines()
    events = [json.loads(line)["event"] for line in lines]
    assert events == ["meta", "val", "val", "end"]

    index_path = paths_mod.reports_training_dir("biochem") / diary_mod.BiochemRunLogger.INDEX_JSONL
    assert index_path.exists()
    index_row = json.loads(index_path.read_text(encoding="utf-8").strip().splitlines()[-1])
    assert index_row["run_note"] == "smoke"
    assert index_row["val_mu_log_mae"] == 0.8
    assert index_row["best_epoch"] == 2


def test_biochem_run_logger_prunes_old_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(paths_mod, "get_project_root", lambda: tmp_path)
    monkeypatch.setenv("BIOCHEM_TRAINING_LOG_KEEP", "3")
    monkeypatch.setenv("BIOCHEM_TRAINING_LOG_INDEX", "0")

    base = tmp_path / "outputs" / "reports" / "training" / "biochem"
    for idx in range(1, 6):
        run_dir = base / f"2026042{idx}T184600Z"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / diary_mod.BiochemRunLogger.RUN_JSONL).write_text("{}", encoding="utf-8")

    log = diary_mod.BiochemRunLogger()
    kept = sorted(p.name for p in base.iterdir() if p.is_dir())
    assert log.run_dir is not None
    assert len(kept) == 3
    assert "20260421T184600Z" not in kept
