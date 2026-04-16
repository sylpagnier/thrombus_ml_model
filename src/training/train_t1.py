"""Thin Tier 1 entrypoint wrapper around predictor training."""

from __future__ import annotations

import argparse
import os

from src.training.train_t1_predictor import train_t1_predictor


def _parse_args():
    p = argparse.ArgumentParser(description="Tier 1 GINO-DEQ predictor training.")
    p.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Sets TIER1_EXPERIMENT_NAME for reports/experiments JSON.",
    )
    return p.parse_args()


def main():
    args = _parse_args()
    if args.experiment_name:
        os.environ["TIER1_EXPERIMENT_NAME"] = args.experiment_name
    train_t1_predictor()


if __name__ == "__main__":
    main()
