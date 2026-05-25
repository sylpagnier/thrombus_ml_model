from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import torch

import src.training.train_biochem_corrector as train_mod


def test_persist_teacher_checkpoints_keeps_global_high_mu(tmp_path, monkeypatch):
    model_dir = tmp_path / "biochem"
    model_dir.mkdir(parents=True)
    teacher = MagicMock()
    teacher.state_dict.return_value = {"w": torch.zeros(1)}

    global_path = model_dir / train_mod.BIOCHEM_TEACHER_BEST_HIGH_MU_CKPT_NAME
    torch.save(
        {
            "model_state_dict": {"w": torch.zeros(1)},
            "val_mu_log_mae_high_mu": 0.45,
            "val_mu_log_mae": 0.30,
            "run_note": "old_best",
            "checkpoint_role": "teacher_best_high_mu",
        },
        global_path,
    )

    monkeypatch.setattr(train_mod, "_biochem_env_truthy", lambda _k, default=True: default)
    train_mod._persist_biochem_teacher_checkpoints(
        model_dir,
        teacher,
        teacher_best_mu_score=-0.52,
        best_epoch=10,
        run_note="worse_run",
        val_mu_log_mae_high_mu=0.62,
        best_high_epoch=10,
        best_high_state={"w": torch.zeros(1)},
    )

    last_path = model_dir / train_mod.BIOCHEM_TEACHER_LAST_CKPT_NAME
    assert last_path.is_file()

    kept = train_mod._read_teacher_checkpoint_meta(global_path)
    assert float(kept["val_mu_log_mae_high_mu"]) == 0.45
    assert kept["run_note"] == "old_best"


def test_persist_teacher_checkpoints_updates_global_high_mu(tmp_path, monkeypatch):
    model_dir = tmp_path / "biochem"
    model_dir.mkdir(parents=True)
    teacher = MagicMock()
    teacher.state_dict.return_value = {"w": torch.ones(1)}

    global_path = model_dir / train_mod.BIOCHEM_TEACHER_BEST_HIGH_MU_CKPT_NAME
    torch.save(
        {
            "model_state_dict": {"w": torch.zeros(1)},
            "val_mu_log_mae_high_mu": 0.55,
            "run_note": "old",
            "checkpoint_role": "teacher_best_high_mu",
        },
        global_path,
    )

    monkeypatch.setattr(train_mod, "_biochem_env_truthy", lambda _k, default=True: default)
    train_mod._persist_biochem_teacher_checkpoints(
        model_dir,
        teacher,
        teacher_best_mu_score=-0.29,
        best_epoch=32,
        run_note="new_best",
        val_mu_log_mae_high_mu=0.41,
        best_high_epoch=32,
        best_high_state={"w": torch.ones(1)},
    )

    meta = train_mod._read_teacher_checkpoint_meta(global_path)
    assert float(meta["val_mu_log_mae_high_mu"]) == 0.41
    assert meta["run_note"] == "new_best"
