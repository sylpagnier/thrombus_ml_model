import matplotlib
matplotlib.use('Agg')
import sys
import shutil
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import time

# --- Path Setup ---
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent.parent
sys.path.append(str(project_root))

# Import your existing modules
from src.phase1.data_gen.vessel_generator import VesselGenerator
from src.phase1.data_gen.anchor_generator import AnchorGenerator
from src.phase1.data_gen.mesh_to_graph import MeshToGraphComplete
from src.validation.phase1.validate_tier1 import Tier1Validator


def run_pipeline_for_level(level_idx, level_name, num_samples=20):
    """
    Runs the full Generative -> COMSOL -> Graph -> Validation pipeline.
    """
    print(f"\n{'=' * 60}")
    print(f"🚀 STARTING BENCHMARK: {level_name} (Level {level_idx})")
    print(f"{'=' * 60}")

    # Define Paths
    base_dir = project_root / "data" / "benchmark" / f"level_{level_idx}"
    raw_mesh_dir = base_dir / "raw_meshes"
    label_dir = base_dir / "comsol_solutions"
    graph_dir = base_dir / "processed_graphs"

    # Clean previous runs (optional, but safer for benchmarking)
    for d in [raw_mesh_dir, label_dir, graph_dir]:
        if d.exists(): shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Geometry Generation ---
    print(f"\n[1/4] 📐 Generating {num_samples} Geometries...")
    v_gen = VesselGenerator(output_dir=str(raw_mesh_dir.relative_to(project_root)))

    # Generate samples (suppress visualization to avoid extra GUI calls)
    for i in tqdm(range(num_samples), desc="Meshing"):
        v_gen.generate(i, level=level_idx, show_viz=False)

    # --- Step 2: COMSOL Simulation ---
    print(f"\n[2/4] 🌪️ Solving Navier-Stokes in COMSOL...")
    try:
        a_gen = AnchorGenerator(
            template_path='comsol_models/phase1_template.mph',
            mesh_dir=str(raw_mesh_dir.relative_to(project_root)),
            output_dir=str(label_dir.relative_to(project_root))
        )
        # Use a try/except block inside the batch loop in AnchorGenerator,
        # but here we just trigger the batch run.
        a_gen.run_batch(start_idx=0, end_idx=num_samples)
    except Exception as e:
        print(f"❌ COMSOL Error: {e}")
        print("   (Proceeding with available data...)")

    # --- Step 3: Graph Conversion ---
    print(f"\n[3/4] 🕸️ Converting to Graphs (Injecting Labels)...")
    m_gen = MeshToGraphComplete(
        raw_dir=str(raw_mesh_dir.relative_to(project_root)),
        label_dir=str(label_dir.relative_to(project_root)),
        proc_dir=str(graph_dir.relative_to(project_root))
    )
    m_gen.run()

    # --- Step 4: Validation Inference ---
    print(f"\n[4/4] 🧠 Running Model Inference & Metrics...")
    model_path = project_root / "models/tier1_best_physics.pth"

    if not model_path.exists():
        print(f"❌ Model not found at {model_path}. Skipping validation.")
        return None

    validator = Tier1Validator(model_path=model_path)

    # Validate
    # Note: We pass the relative path as string to match your Validator's expectations
    metrics = validator.validate_dataset(str(graph_dir.relative_to(project_root)), level_name=level_name)

    return metrics


if __name__ == "__main__":
    # Define Curriculum Levels
    benchmarks = [
        (1, "Level 1 (Straight)"),
        # (2, "Level 2 (Curved_Pathology)"),  # <-- Commented out for now
        # (3, "Level 3 (Bifurcations)")      # <-- Commented out for now
    ]

    all_results = {}

    for lvl_idx, name in benchmarks:
        metrics = run_pipeline_for_level(lvl_idx, name, num_samples=20)
        if metrics is not None:
            all_results[name] = metrics

        # Brief pause to allow file system I/O to settle
        time.sleep(1)

    # --- Final Report ---
    print("\n\n")
    print("*" * 50)
    print("🏆 FINAL MULTI-FIDELITY BENCHMARK REPORT")
    print("*" * 50)

    if all_results:
        df = pd.DataFrame(all_results).T
        print(df)

        save_path = project_root / "reports" / "tier1_full_benchmark.csv"
        # Ensure reports dir exists
        save_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(save_path)
        print(f"\n📄 Detailed report saved to: {save_path}")
    else:
        print("❌ No results generated.")