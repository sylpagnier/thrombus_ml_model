"""Thin Tier 2 entrypoint wrapper around predictor training."""

from __future__ import annotations

from src.training.train_t2_predictor import train_t2_predictor


def main():
    train_t2_predictor()


if __name__ == "__main__":
    main()
