import matplotlib
matplotlib.use('Agg')
import shutil
import pandas as pd
import time
from datetime import datetime
from tqdm import tqdm
from src.utils.paths import get_project_root

project_root = get_project_root()
from src.phase1.data_gen.vessel_generator import VesselGenerator
from src.phase1.data_gen.anchor_generator import AnchorGenerator
from src.phase1.data_gen.mesh_to_graph import MeshToGraphComplete
from src.phase1.validation.validate_phase1_model import ModelValidator


def run_pipeline_for_level(tier, level_idx, level_name, num_samples=10):
    print(f"\n{'=' * 60}")
    print(f"🚀 STARTING BENCHMARK: [{tier.upper()}] {level_name} (Level {level_idx})")
    print(f"{'=' * 60}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = project_root / "data" / "benchmark" / f"{tier}_level_{level_idx}_{timestamp}"

    raw_mesh_dir = base_dir / "raw_meshes"
    label_dir = base_dir / "comsol_solutions"
    graph_dir = base_dir / "processed_graphs"

    for d in [raw_mesh_dir, label_dir, graph_dir]:
        d.mkdir(parents=True, exist_ok=True)

    try:
        print(f"\n[1/4] 📐 Generating {num_samples} Geometries in {base_dir.name}...")
        v_gen = VesselGenerator(tier=tier, output_dir=str(raw_mesh_dir))
        for i in tqdm(range(num_samples), desc="Meshing"):
            v_gen.generate(i, level=level_idx, show_viz=False)

        print(f"\n[2/4] 🌪️ Solving Navier-Stokes in COMSOL...")
        template_absolute = project_root / "comsol_models" / "phase1_template.mph"

        with AnchorGenerator(
                tier=tier,
                template_path=str(template_absolute),
                mesh_dir=str(raw_mesh_dir),
                output_dir=str(label_dir)
        ) as a_gen:
            a_gen.run_batch(start_idx=0, end_idx=num_samples)

        print(f"\n[3/4] 🕸️ Converting to Graphs...")
        m_gen = MeshToGraphComplete(
            tier=tier,
            raw_dir=str(raw_mesh_dir),
            label_dir=str(label_dir),
            proc_dir=str(graph_dir)
        )
        m_gen.run()

        print(f"\n[4/4] 🧠 Running Model Inference & Metrics...")
        model_path = project_root / f"models/{tier}_best_physics.pth"

        if not model_path.exists():
            print(f"❌ Model not found at {model_path}. Skipping validation.")
            return None

        validator = ModelValidator(model_path=model_path, tier=tier)
        metrics = validator.validate_dataset(str(graph_dir), level_name=level_name)

        return metrics

    except Exception as e:
        print(f"❌ Critical Benchmark Error: {e}")
        return None


if __name__ == "__main__":
    target_tiers = ["tier1", "tier2"]

    benchmarks = [
        (0, "Level 0 (Straight pathologies)"),
    ]

    for current_tier in target_tiers:
        all_results = {}
        for lvl_idx, name in benchmarks:
            metrics = run_pipeline_for_level(current_tier, lvl_idx, name, num_samples=10)
            if metrics is not None:
                all_results[name] = metrics
            time.sleep(1)

        print("\n\n" + "*" * 50)
        print(f"🏆 FINAL MULTI-FIDELITY BENCHMARK REPORT: {current_tier.upper()}")
        print("*" * 50)

        if all_results:
            df = pd.DataFrame(all_results).T
            print(df)
            save_path = project_root / "reports" / f"{current_tier}_full_benchmark.csv"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(save_path)
            print(f"\n📄 Detailed report saved to: {save_path}")
        else:
            print(f"❌ No results generated for {current_tier}.")

    # Full aggressive cleanup of all benchmark data once everything finishes
    benchmark_data_dir = project_root / "data" / "benchmark"
    if benchmark_data_dir.exists():
        print(f"\n🧹 Final Cleanup: Sweeping up all generated temporary benchmark data at {benchmark_data_dir}...")
        shutil.rmtree(benchmark_data_dir, ignore_errors=True)
        print("✅ Cleanup complete.")