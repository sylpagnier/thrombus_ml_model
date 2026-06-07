"""Mesh generation, COMSOL hooks, and graph conversion (Kinematics/2/3)."""

from src.data_gen.lib.anchor_generator import AnchorGenerator
from src.data_gen.lib.extract_biochem_comsol_data import PatientDataExtractor
from src.data_gen.lib.mesh_to_graph import MeshToGraphComplete, build_mesh_converter
from src.data_gen.lib.mesh_to_graph_biochem import MeshToGraphPhase3
from src.data_gen.lib.vessel_generator import (
    VesselGenerator,
    VesselGeneratorPhase3,
    build_vessel_mesh,
    make_vessel_params,
    recompute_pathology_offsets,
    summarize_vessel_mesh_inventory,
)
from src.data_gen.lib.vessel_geometry import (
    VesselGeometry,
    compute_geometry_from_params,
    compute_geometry_from_walls,
    validate_geometry,
)

__all__ = [
    "AnchorGenerator",
    "MeshToGraphComplete",
    "MeshToGraphPhase3",
    "PatientDataExtractor",
    "VesselGenerator",
    "VesselGeneratorPhase3",
    "build_mesh_converter",
    "build_vessel_mesh",
    "make_vessel_params",
    "recompute_pathology_offsets",
    "summarize_vessel_mesh_inventory",
    "VesselGeometry",
    "compute_geometry_from_params",
    "compute_geometry_from_walls",
    "validate_geometry",
]
