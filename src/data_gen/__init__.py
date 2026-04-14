"""Mesh generation, COMSOL hooks, and graph conversion (Tier 1/2/3)."""

from src.data_gen.lib.anchor_generator import AnchorGenerator
from src.data_gen.lib.extract_tier3_comsol_data import PatientDataExtractor
from src.data_gen.lib.mesh_to_graph import MeshToGraphComplete, build_mesh_converter
from src.data_gen.lib.mesh_to_graph_tier3 import MeshToGraphTier3
from src.data_gen.lib.vessel_generator import (
    VesselGenerator,
    VesselGeneratorTier3,
    summarize_vessel_mesh_inventory,
)

__all__ = [
    "AnchorGenerator",
    "MeshToGraphComplete",
    "MeshToGraphTier3",
    "PatientDataExtractor",
    "VesselGenerator",
    "VesselGeneratorTier3",
    "build_mesh_converter",
    "summarize_vessel_mesh_inventory",
]
