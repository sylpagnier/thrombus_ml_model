import gmsh
from pathlib import Path
from src.phase1.data_gen.vessel_generator import VesselGenerator
from src.phase1.data_gen.anchor_generator import AnchorGenerator
from src.phase1.data_gen.mesh_to_graph import MeshToGraphComplete

def run_pipeline(num_samples=125, level=1):
    """
    Executes the full Phase 1 data preparation flow.
    1. Vessel Generation (Gmsh)
    2. CFD Anchor Generation (COMSOL)
    3. Graph Processing (PyTorch Geometric)
    """
    print("🚀 Starting Phase 1 Data Pipeline\n" + "="*30)

    # --- STEP 1: GEOMETRY GENERATION ---
    print(f"\n[Step 1/3] Generating {num_samples} geometries (Level {level})...")
    v_gen = VesselGenerator(output_dir="data/raw/synthetic_v1")
    # run_pipeline in vessel_generator.py handles the loop and visualization
    v_gen.run_pipeline(n=num_samples, level=level)
    gmsh.finalize() # Ensure Gmsh is cleaned up before moving to COMSOL
    print("✅ Geometry generation complete.")

    # --- STEP 2: CFD ANCHOR GENERATION ---
    print("\n[Step 2/3] Solving CFD via COMSOL (Anchors)...")
    try:
        a_gen = AnchorGenerator(
            template_path='comsol_models/phase1_template.mph',
            mesh_dir='data/raw/synthetic_v1',
            output_dir='data/raw/cfd_anchors'
        )
        # Solving the generated batch
        a_gen.run_batch(start_idx=0, end_idx=num_samples)
        print("✅ CFD solving complete.")
    except Exception as e:
        print(f"❌ COMSOL Error: {e}")
        print("Continuing to Step 3 with available data...")

    # --- STEP 3: MESH-TO-GRAPH CONVERSION ---
    print("\n[Step 3/3] Converting meshes and labels to Graphs...")
    processor = MeshToGraphComplete(
        raw_dir="data/raw/synthetic_v1",
        label_dir="data/raw/cfd_anchors",
        proc_dir="data/processed/tier1_graphs"
    )
    processor.run()
    print("✅ Graph processing complete.")

    print("\n" + "="*30 + "\n🎉 Pipeline Finished! Data is ready in 'data/processed/tier1_graphs'.")

if __name__ == "__main__":
    # You can adjust num_samples to match your computational budget
    run_pipeline(num_samples=125, level=1)