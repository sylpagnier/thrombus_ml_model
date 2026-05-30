"""Probe-mode species gate (short legs, saturated FI ok)."""

from __future__ import annotations

from scripts.check_passive_x_species_gate import eval_species_gate


def _run_dir_with_val_rows(rows: list[dict]):
    from pathlib import Path
    import json
    import tempfile

    td = Path(tempfile.mkdtemp())
    (td / "run.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return td


def test_probe_gate_passes_saturated_fi() -> None:
    rows = [
        {"event": "val", "stage": "teacher", "epoch": 0, "val_species_fi_log_mae": 0.03, "val_viz_t0_speed_mean": 0.9},
        {"event": "val", "stage": "teacher", "epoch": 1, "val_species_fi_log_mae": 0.029, "val_viz_t0_speed_mean": 0.91},
    ]
    rd = _run_dir_with_val_rows(rows)
    out = eval_species_gate(rd, species_fi_max=0.05, train_fi_max=0.04, min_speed=0.5, probe=True)
    assert out["ok"] is True
    assert out["mode"] == "probe"


def test_probe_gate_fails_regression() -> None:
    rows = [
        {"event": "val", "stage": "teacher", "epoch": 0, "val_species_fi_log_mae": 0.03, "val_viz_t0_speed_mean": 0.9},
        {"event": "val", "stage": "teacher", "epoch": 1, "val_species_fi_log_mae": 0.20, "val_viz_t0_speed_mean": 0.9},
    ]
    rd = _run_dir_with_val_rows(rows)
    out = eval_species_gate(rd, species_fi_max=0.05, train_fi_max=0.04, min_speed=0.5, probe=True)
    assert out["ok"] is False
