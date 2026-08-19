"""Differentiable Physics Wall Model.

A fully differentiable PyTorch implementation of the Phase-3/Phase-5 wall-clot physics
(COMSOL deposition law, flow-derived gates, wake redistribution, and front growth).
Supports global scalar parameter tuning (Level 1.1) and local neural prediction (Level 1.2).
"""
from __future__ import annotations

from src.differentiable_wall_model.differentiable_ode import DifferentiableWallModel
from src.differentiable_wall_model.parameters import (
    GlobalPhysicsParameters,
    ParameterMap,
    ParameterProvider,
)

__all__ = [
    "DifferentiableWallModel",
    "GlobalPhysicsParameters",
    "ParameterMap",
    "ParameterProvider",
]
