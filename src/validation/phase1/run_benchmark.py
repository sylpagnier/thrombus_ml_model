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

from src.phase1.data_gen.vessel_generator import VesselGenerator
from src.phase1.data_gen.anchor_generator import AnchorGenerator
from src.phase1.data_gen.mesh_to_graph import MeshToGraphComplete
from src.validation.phase1.validate_tier1 import Tier1Validator


def run_pipeline_for_level(level_idx, level_name, num_samples=20):
    print(f"\n{'=' * 60}")
    print(f"🚀 STARTING BENCHMARK: {level_name} (Level {level_idx})")
    print(f"{'=' * 60}")

    # Define Absolute Paths
    base_dir = project_root / "data" / "benchmark" / f"level_{level_idx}"
    raw_mesh_dir = base_dir / "raw_meshes"
    label_dir = base_dir / "comsol_solutions"
    graph_dir = base_dir / "processed_graphs"

    # Clean previous runs
    for d in [raw_mesh_dir, label_dir, graph_dir]:
        if d.exists(): shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

        # --- Step 1: Geometry Generation ---
        print(f"\n[1/4] 📐 Generating {num_samples} Geometries...")
        v_gen = VesselGenerator(output_dir=str(raw_mesh_dir))

        # 1. Loop completes FIRST
        for i in tqdm(range(num_samples), desc="Meshing"):
            v_gen.generate(i, level=level_idx, show_viz=False)

        # 2. THEN run COMSOL once for the whole batch (Unindented)
        # --- Step 2: COMSOL Simulation ---
        print(f"\n[2/4] 🌪️ Solving Navier-Stokes in COMSOL...")
        try:
            template_absolute = project_root / "comsol_models" / "phase1_template.mph"

            with AnchorGenerator(
                    template_path=str(template_absolute),
                    mesh_dir=str(raw_mesh_dir),
                    output_dir=str(label_dir)
            ) as a_gen:
                a_gen.run_batch(start_idx=0, end_idx=num_samples)

        except Exception as e:
            print(f"❌ COMSOL Error: {e}")
            print("   (Proceeding with available data...)")

    # --- Step 3: Graph Conversion ---
    print(f"\n[3/4] 🕸️ Converting to Graphs (Injecting Labels)...")
    # FIX: Pass ABSOLUTE paths
    m_gen = MeshToGraphComplete(
        raw_dir=str(raw_mesh_dir),
        label_dir=str(label_dir),
        proc_dir=str(graph_dir)
    )
    m_gen.run()

    # --- Step 4: Validation Inference ---
    print(f"\n[4/4] 🧠 Running Model Inference & Metrics...")
    model_path = project_root / "models/tier1_best_physics.pth"

    if not model_path.exists():
        print(f"❌ Model not found at {model_path}. Skipping validation.")
        return None

    validator = Tier1Validator(model_path=model_path)
    # FIX: Validate dataset expects a string path relative to project root OR absolute
    metrics = validator.validate_dataset(str(graph_dir), level_name=level_name)

    return metrics


if __name__ == "__main__":
    benchmarks = [
        (1, "Level 1 (Straight)"),
        # (2, "Level 2 (Curved_Pathology)"),
        # (3, "Level 3 (Bifurcations)")
    ]

    all_results = {}

    for lvl_idx, name in benchmarks:
        metrics = run_pipeline_for_level(lvl_idx, name, num_samples=20)
        if metrics is not None:
            all_results[name] = metrics
        time.sleep(1)

    print("\n\n" + "*" * 50)
    print("🏆 FINAL MULTI-FIDELITY BENCHMARK REPORT")
    print("*" * 50)

    if all_results:
        df = pd.DataFrame(all_results).T
        print(df)
        save_path = project_root / "reports" / "tier1_full_benchmark.csv"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path)
        print(f"\n📄 Detailed report saved to: {save_path}")
    else:
        print("❌ No results generated.")