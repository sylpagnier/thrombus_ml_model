"""Train Star 1 hybrid clot trigger (GT flow + GT species)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.training.clot_trigger_stack import apply_star1_train_env  # noqa: E402
from src.training.train_clot_phi_simple import main as train_main  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="T1 clot trigger train")
    ap.add_argument("--fast", action="store_true", help="16 epochs instead of 48")
    ap.add_argument(
        "--oracle-band",
        action="store_true",
        help="Legacy: GT-mu seeds + dgamma slice for loss (debug only)",
    )
    args = ap.parse_args()
    apply_star1_train_env(fast=bool(args.fast))
    if bool(args.oracle_band):
        from src.training.clot_trigger_stack import apply_oracle_neighbor_mask_env

        apply_oracle_neighbor_mask_env()
    train_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
