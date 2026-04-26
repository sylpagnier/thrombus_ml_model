"""Generate Phase-1 mesh-resolution sweep datasets.

Creates three dataset phases by varying ``GMSH_SIZE_FACTOR``:
- coarse: 1.5
- medium: 0.75
- fine: 0.4
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
from typing import Dict

from src.data_gen.lib.vessel_generator import VesselGenerator
from src.data_gen.lib.anchor_generator import AnchorGenerator
from src.data_gen.lib.mesh_to_graph import MeshToGraphComplete

def generate_resolution_sweep(n_vessels: int = 100):
    resolutions = {
        "coarse": 1.5,
        "medium": 0.75,
        "fine": 0.4
    }
    
    for name, factor in resolutions.items():
        print(f"\n========== Generating Resolution: {name} (factor={factor}) ==========")
        os.environ["GMSH_SIZE_FACTOR"] = str(factor)
        phase_name = f"kinematics_res_{name}"
        
        try:
            # 1. Generate Vessels
            vg = VesselGenerator(phase=phase_name)
            vg.run_pipeline(n=n_vessels, level=1)
            
            # 2. Run COMSOL CFD for Anchors
            ag = AnchorGenerator(phase=phase_name)
            with ag:
                ag.run_batch(max_new=n_vessels)
                
            # 3. Convert to PyG Graphs
            MeshToGraphComplete(phase=phase_name).run()
            
        finally:
            # Cleanup env to prevent bleeding between iterations
            if "GMSH_SIZE_FACTOR" in os.environ:
                del os.environ["GMSH_SIZE_FACTOR"]

if __name__ == "__main__":
    mp.freeze_support()
    generate_resolution_sweep()
