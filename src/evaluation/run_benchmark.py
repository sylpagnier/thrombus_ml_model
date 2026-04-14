import matplotlib
matplotlib.use('Agg')
import shutil
import pandas as pd
import time
import argparse
from datetime import datetime
from src.utils.paths import comsol_models_dir, data_root, get_project_root, reports_dir, resolve_checkpoint

project_root = get_project_root()
from src.data_gen import AnchorGenerator, MeshToGraphComplete, VesselGenerator
from src.evaluation.validate_model import ModelValidator


def run_pipeline_for_level(tier, level_idx, level_name, num_samples=10):
    print(f"\n{'=' * 60}")
    print(f"🚀 STARTING BENCHMARK: [{tier.upper()}] {level_name} (Level {level_idx})")
    print(f"{'=' * 60}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = data_root() / "benchmark" / f"{tier}_level_{level_idx}_{timestamp}"

    raw_mesh_dir = base_dir / "raw_meshes"
    label_dir = base_dir / "comsol_solutions"
    graph_dir = base_dir / "processed_graphs"

    for d in [raw_mesh_dir, label_dir, graph_dir]:
        d.mkdir(parents=True, exist_ok=True)

    try:
        print(f"\n[1/4] 📐 Generating {num_samples} Geometries in {base_dir.name}...")
        v_gen = VesselGenerator(tier=tier, output_dir=str(raw_mesh_dir))
        v_gen.run_pipeline(
            n=num_samples,
            level=level_idx,
            start_idx=0,
            seed=None,
        )

        print(f"\n[2/4] 🌪️ Solving Navier-Stokes in COMSOL...")
        template_absolute = comsol_models_dir() / "phase1_template.mph"

        with AnchorGenerator(
                tier=tier,
                template_path=str(template_absolute),
                mesh_dir=str(raw_mesh_dir),
                output_dir=str(label_dir)
        ) as a_gen:
            a_gen.run_batch(max_new=num_samples)

        print(f"\n[3/4] 🕸️ Converting to Graphs...")
        m_gen = MeshToGraphComplete(
            tier=tier,
            raw_dir=str(raw_mesh_dir),
            label_dir=str(label_dir),
            proc_dir=str(graph_dir)
        )
        m_gen.run()

        print(f"\n[4/4] 🧠 Running Model Inference & Metrics...")
        ckpt_name = f"{tier}_best_physics.pth"
        model_path = resolve_checkpoint("a", ckpt_name)

        if not model_path.exists():
            print(f"❌ Model not found at {model_path} (expected under outputs/stage_a/). Skipping validation.")
            return None

        validator = ModelValidator(model_path=model_path, tier=tier)
        metrics = validator.validate_dataset(str(graph_dir), level_name=level_name)

        return metrics

    except Exception as e:
        print(f"❌ Critical Benchmark Error: {e}")
        return None


def _prompt_text(label, default):
    raw = input(f"{label} [{default}]: ").strip()
    return raw if raw else str(default)


def _prompt_int(label, default):
    while True:
        raw = input(f"{label} [{default}]: ").strip()
        if raw == "":
            return int(default)
        try:
            return int(raw)
        except ValueError:
            print("Invalid input. Enter an integer value.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multi-fidelity benchmark pipeline")
    parser.add_argument("--tiers", type=str, default=None, help='Comma-separated tiers (for example: "tier1,tier2")')
    parser.add_argument("--num-samples", type=int, default=None, help="Number of vessels per benchmark level")
    parser.add_argument("--levels", type=str, default=None, help='Comma-separated benchmark levels (for example: "0,1")')
    args = parser.parse_args()

    if args.tiers is None:
        tiers_raw = _prompt_text("Tiers (comma-separated)", "tier1,tier2")
    else:
        tiers_raw = args.tiers
    target_tiers = [t.strip() for t in tiers_raw.split(",") if t.strip()]

    if args.num_samples is None:
        num_samples = _prompt_int("Number of vessels per level", 10)
    else:
        num_samples = args.num_samples

    if args.levels is None:
        levels_raw = _prompt_text("Levels (comma-separated)", "0")
    else:
        levels_raw = args.levels

    level_ids = []
    for s in levels_raw.split(","):
        s = s.strip()
        if s:
            level_ids.append(int(s))

    level_names = {
        0: "Level 0 (Straight pathologies)",
        1: "Level 1 (Curved pathologies)",
    }
    benchmarks = [(lvl, level_names.get(lvl, f"Level {lvl}")) for lvl in level_ids]

    for current_tier in target_tiers:
        all_results = {}
        for lvl_idx, name in benchmarks:
            metrics = run_pipeline_for_level(current_tier, lvl_idx, name, num_samples=num_samples)
            if metrics is not None:
                all_results[name] = metrics
            time.sleep(1)

        print("\n\n" + "*" * 50)
        print(f"🏆 FINAL MULTI-FIDELITY BENCHMARK REPORT: {current_tier.upper()}")
        print("*" * 50)

        if all_results:
            df = pd.DataFrame(all_results).T
            print(df)
            save_path = reports_dir() / f"{current_tier}_full_benchmark.csv"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(save_path)
            print(f"\n📄 Detailed report saved to: {save_path}")
        else:
            print(f"❌ No results generated for {current_tier}.")

    # Full aggressive cleanup of all benchmark data once everything finishes
    benchmark_data_dir = data_root() / "benchmark"
    if benchmark_data_dir.exists():
        print(f"\n🧹 Final Cleanup: Sweeping up all generated temporary benchmark data at {benchmark_data_dir}...")
        shutil.rmtree(benchmark_data_dir, ignore_errors=True)
        print("✅ Cleanup complete.")