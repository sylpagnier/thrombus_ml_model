"""Clot timeline FP/FN aggregate metrics."""

from __future__ import annotations

import torch

from src.evaluation.clot_timeline_metrics import (
    clot_frame_metrics,
    summarize_clot_timeline,
)


def test_clot_frame_metrics_counts():
    phi_gt = torch.tensor([0.0, 1.0, 1.0, 0.0])
    phi_pred = torch.tensor([1.0, 1.0, 0.0, 0.0])
    m = clot_frame_metrics(phi_pred, phi_gt, n_band=4)
    assert m["clot_fp"] == 1.0
    assert m["clot_fn"] == 1.0
    assert m["clot_tp"] == 1.0
    assert m["clot_err"] == 2.0


def test_summarize_clot_timeline_median_fp():
    frames = [
        {"clot_fp": 200.0, "clot_fn": 0.0, "clot_err": 200.0, "clot_fp_per_gt": 0.5, "clot_fn_per_gt": 0.0, "clot_fp_rate_band": 0.01},
        {"clot_fp": 75.0, "clot_fn": 4.0, "clot_err": 79.0, "clot_fp_per_gt": 0.1, "clot_fn_per_gt": 0.02, "clot_fp_rate_band": 0.004},
        {"clot_fp": 60.0, "clot_fn": 122.0, "clot_err": 182.0, "clot_fp_per_gt": 0.05, "clot_fn_per_gt": 0.3, "clot_fp_rate_band": 0.003},
    ]
    s = summarize_clot_timeline(frames)
    assert s["clot_fp_median"] == 75.0
    assert s["clot_fp_max"] == 200.0
    assert s["clot_fp_early_mean"] == (200.0 + 75.0 + 60.0) / 3.0
    assert s["clot_err_p90"] >= 182.0
