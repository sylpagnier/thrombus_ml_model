"""Baseline Newtonian and Carreau non-Newtonian momentum kernels (graph WLS PINN)."""

from src.core_physics.physics_kernels import PhysicsKernels


class FluidKinematicsKernels(PhysicsKernels):
    """Shared fluid kinematics: NS residual, continuity, BCs, Carreau rheology supervision."""

    pass


__all__ = ["FluidKinematicsKernels"]
