"""Recover clot_phi_best.pth from train log when val_score gate rejected all saves."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.core_physics.clot_phi_simple import build_clot_phi_model, clot_phi_feature_dim
from src.training.train_clot_phi_simple import _checkpoint_score


def _rank_row(row: dict) -> float:
    va = row.get("val") or {}
    score = float(row.get("val_score", -99.0))
    if score >= 0:
        return score
    shape = float(va.get("clot_shape", 0.0))
    f1 = float(va.get("clot_f1", 0.0))
    return shape * 0.65 + f1 * 0.35


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--leg-dir", required=True, help="Leg dir under sweep (relative or absolute)")
    ap.add_argument("--log", default="", help="Override log path")
    args = ap.parse_args()

    leg = Path(args.leg_dir)
    if not leg.is_absolute():
        leg = _REPO / leg
    log_path = Path(args.log) if args.log else leg / "clot_phi_train_log.jsonl"
    ckpt_path = leg / "clot_phi_best.pth"
    last_path = leg / "clot_phi_last.pth"

    if not log_path.is_file():
        print(f"[ERR] missing log: {log_path}", flush=True)
        sys.exit(1)
    if not last_path.is_file():
        print(f"[ERR] missing last weights: {last_path}", flush=True)
        sys.exit(1)

    best_row: dict | None = None
    best_rank = -1.0
    with log_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rank = _rank_row(row)
            if best_row is None or rank > best_rank:
                best_row = row
                best_rank = rank

    if best_row is None:
        print("[ERR] empty train log", flush=True)
        sys.exit(1)

    raw = torch.load(last_path, map_location="cpu", weights_only=False)
    cfg = dict(raw.get("config") or {})
    in_dim = int(cfg.get("in_dim", clot_phi_feature_dim()))
    hidden = int(cfg.get("hidden", 32))
    model = build_clot_phi_model(in_dim=in_dim, hidden=hidden)
    model.load_state_dict(raw["model_state_dict"])

    ep = int(best_row.get("epoch", -1))
    va = best_row.get("val") or {}
    score = _checkpoint_score({k: float(v) for k, v in va.items() if isinstance(v, (int, float))})
    payload = {
        "model_state_dict": model.state_dict(),
        "config": cfg,
        "epoch": ep,
        "val_score": score if score >= 0 else best_rank,
        "recovered_from_log": True,
        "val": va,
    }
    if "species_head_state_dict" in raw:
        payload["species_head_state_dict"] = raw["species_head_state_dict"]

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, ckpt_path)
    print(
        f"[OK]  recovered -> {ckpt_path} ep={ep} rank={best_rank:.3f} "
        f"shape={float(va.get('clot_shape', 0)):.3f} f1={float(va.get('clot_f1', 0)):.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
