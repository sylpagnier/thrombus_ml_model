import shutil
import pandas as pd
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.widgets import Button
import torch
import random
from datetime import datetime
from src.utils.paths import (
    comsol_models_dir,
    data_root,
    get_project_root,
    reports_subdir,
    resolve_checkpoint,
)

project_root = get_project_root()
from src.data_gen import AnchorGenerator, MeshToGraphComplete, VesselGenerator
from src.evaluation.lib.validate_model import ModelValidator
from src.config import PredChannels

KINEMATICS_FINAL_ANNEALED_N = 0.358


def _plot_field(fig, ax, pos, val, title, cmap, vmin=None, vmax=None):
    triang = mtri.Triangulation(pos[:, 0], pos[:, 1])
    tri_pts = pos[triang.triangles]
    d1 = np.sum((tri_pts[:, 0, :] - tri_pts[:, 1, :]) ** 2, axis=1)
    d2 = np.sum((tri_pts[:, 1, :] - tri_pts[:, 2, :]) ** 2, axis=1)
    d3 = np.sum((tri_pts[:, 2, :] - tri_pts[:, 0, :]) ** 2, axis=1)
    max_edge_sq = np.max(np.vstack([d1, d2, d3]), axis=0)
    mask = max_edge_sq > (np.median(max_edge_sq) * 10.0)
    triang.set_mask(mask)

    tc = ax.tripcolor(triang, val, cmap=cmap, shading="gouraud", vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=11)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.colorbar(tc, ax=ax, fraction=0.046, pad=0.04)


def _show_benchmark_visualization(validator, graph_dir, phase, level_idx, level_name):
    graph_path = graph_dir if hasattr(graph_dir, "glob") else project_root / graph_dir
    files = sorted(graph_path.glob("*.pt"))
    if not files:
        print(f"⚠️ No graph files found for visualization in {graph_path}")
        return
    current_idx = random.randrange(len(files))
    use_itpc = False
    while True:
        next_idx = None
        toggle_mode = False
        data = torch.load(files[current_idx], weights_only=False).to(validator.device)
        with torch.no_grad():
            pred_base = validator.model(data, solver="anderson")
        if phase == "kinematics" and use_itpc:
            _, pred = validator._predict_with_physics_correction(data, correction_steps=20)
        else:
            pred = pred_base
        pred_np = pred.detach().cpu().numpy()
        pos = data.x[:, :2].detach().cpu().numpy()

        has_labels = hasattr(data, "y") and data.y is not None and data.y.abs().sum() > 1e-6
        gt_np = data.y.detach().cpu().numpy() if has_labels else None

        fields = [
            ("Velocity Magnitude", "jet", lambda arr: np.linalg.norm(arr[:, PredChannels.UV], axis=1)),
            ("Pressure", "coolwarm", lambda arr: arr[:, PredChannels.P]),
            ("Viscosity", "viridis", lambda arr: arr[:, PredChannels.MU_EFF_ND]),
        ]

        ncols = 3 if has_labels else 1
        fig, axes = plt.subplots(len(fields), ncols, figsize=(7 * ncols, 4 * len(fields)))
        if len(fields) == 1:
            axes = np.array([axes])
        if ncols == 1:
            axes = axes.reshape(len(fields), 1)

        for row_idx, (name, cmap, extractor) in enumerate(fields):
            pred_val = extractor(pred_np)
            gt_val = extractor(gt_np) if gt_np is not None else None
            vmin = min(pred_val.min(), gt_val.min()) if gt_val is not None else pred_val.min()
            vmax = max(pred_val.max(), gt_val.max()) if gt_val is not None else pred_val.max()
            if gt_val is not None:
                _plot_field(fig, axes[row_idx, 0], pos, gt_val, f"GT {name}", cmap, vmin=vmin, vmax=vmax)
                pred_label = "Pred ITPC" if (phase == "kinematics" and use_itpc) else "Pred Base"
                _plot_field(fig, axes[row_idx, 1], pos, pred_val, f"{pred_label} {name}", cmap, vmin=vmin, vmax=vmax)
                abs_err = np.abs(pred_val - gt_val)
                _plot_field(
                    fig,
                    axes[row_idx, 2],
                    pos,
                    abs_err,
                    f"|Pred-GT| {name}",
                    "magma",
                    vmin=0.0,
                    vmax=float(abs_err.max()) if abs_err.size else 1.0,
                )
            else:
                pred_label = "Pred ITPC" if (phase == "kinematics" and use_itpc) else "Pred Base"
                _plot_field(fig, axes[row_idx, 0], pos, pred_val, f"{pred_label} {name}", cmap, vmin=vmin, vmax=vmax)

        fig.suptitle(
            f"{phase.upper()} Benchmark Visualization ({'ITPC' if (phase == 'kinematics' and use_itpc) else 'Base'}) - {level_name} ({files[current_idx].name})",
            fontsize=14,
        )
        fig.tight_layout(rect=(0, 0.06, 1, 0.97))

        if len(files) > 1 or phase == "kinematics":
            btn_x = 0.57
            if phase == "kinematics":
                toggle_ax = fig.add_axes([btn_x, 0.01, 0.18, 0.045])
                toggle_label = "Show Base" if use_itpc else "Show ITPC"
                toggle_btn = Button(toggle_ax, toggle_label, color="lightgray", hovercolor="gainsboro")

            def _pick_next_index():
                candidates = [i for i in range(len(files)) if i != current_idx]
                return random.choice(candidates) if candidates else None

            def _toggle_view():
                nonlocal toggle_mode, use_itpc
                if phase != "kinematics":
                    return
                use_itpc = not use_itpc
                toggle_mode = True
                print(f"🔁 Visualization mode: {'ITPC' if use_itpc else 'Base'}")
                plt.close(fig)

            def _request_regen():
                nonlocal next_idx
                next_idx = _pick_next_index()
                if next_idx is not None:
                    print(f"🔁 Switching to vessel sample: {files[next_idx].name}")
                    plt.close(fig)

            if phase == "kinematics":
                toggle_btn.on_clicked(lambda _event: _toggle_view())

            if len(files) > 1:
                regen_ax = fig.add_axes([0.77, 0.01, 0.2, 0.045])
                regen_btn = Button(regen_ax, "Regenerate Vessel", color="lightgray", hovercolor="gainsboro")
                regen_btn.on_clicked(lambda _event: _request_regen())

            def _on_key(event):
                if event.key == "r" and len(files) > 1:
                    _request_regen()
                elif event.key == "t" and phase == "kinematics":
                    _toggle_view()

            fig.canvas.mpl_connect("key_press_event", _on_key)

        plt.show()
        if toggle_mode:
            continue
        if next_idx is None:
            break
        current_idx = next_idx


def run_pipeline_for_level(phase, level_idx, level_name, num_samples=10, visualize=False, carreau_n=None):
    print(f"\n{'=' * 60}")
    print(f"🚀 STARTING BENCHMARK: [{phase.upper()}] {level_name} (Level {level_idx})")
    print(f"{'=' * 60}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = data_root() / "benchmark" / f"{phase}_level_{level_idx}_{timestamp}"

    raw_mesh_dir = base_dir / "raw_meshes"
    label_dir = base_dir / "comsol_solutions"
    graph_dir = base_dir / "processed_graphs"

    for d in [raw_mesh_dir, label_dir, graph_dir]:
        d.mkdir(parents=True, exist_ok=True)

    try:
        print(f"\n[1/4] 📐 Generating {num_samples} Geometries in {base_dir.name}...")
        v_gen = VesselGenerator(phase=phase, output_dir=str(raw_mesh_dir))
        v_gen.run_pipeline(
            n=num_samples,
            level=level_idx,
            start_idx=0,
            seed=None,
        )

        print(f"\n[2/4] 🌪️ Solving Navier-Stokes in COMSOL...")
        template_absolute = comsol_models_dir() / "kinematics_template.mph"
        comsol_available = True

        try:
            with AnchorGenerator(
                    phase=phase,
                    template_path=str(template_absolute),
                    mesh_dir=str(raw_mesh_dir),
                    output_dir=str(label_dir)
            ) as a_gen:
                if phase == "kinematics":
                    # Kinematics benchmark should default to the final continuation target.
                    target_n = KINEMATICS_FINAL_ANNEALED_N if carreau_n is None else float(carreau_n)
                    a_gen.phys_cfg.n = target_n
                    a_gen.model.parameter("n_index", str(target_n))
                    print(f"   ↳ Using Kinematics Carreau n_index = {target_n:.3f}")
                elif carreau_n is not None and phase != "kinematics":
                    target_n = float(carreau_n)
                    a_gen.phys_cfg.n = target_n
                    a_gen.model.parameter("n_index", str(target_n))
                    print(f"   ↳ Using Carreau n_index override = {target_n:.3f}")
                a_gen.run_batch(max_new=num_samples)
        except Exception as comsol_exc:
            comsol_available = False
            print(f"⚠️ COMSOL step skipped: {comsol_exc}")
            print("   ↳ Falling back to AI-only benchmark mode (no COMSOL labels).")

        print(f"\n[3/4] 🕸️ Converting to Graphs...")
        m_gen = MeshToGraphComplete(
            phase=phase,
            raw_dir=str(raw_mesh_dir),
            label_dir=str(label_dir),
            proc_dir=str(graph_dir)
        )
        m_gen.run()

        print(f"\n[4/4] 🧠 Running Model Inference & Metrics...")
        ckpt_candidates = [
            "kinematics_best.pth",
            "kinematics_ckpt_latest.pth",
            "kinematics_ckpt_100.pth",  # legacy fallback
        ]
        model_path = None
        for ckpt_name in ckpt_candidates:
            candidate = resolve_checkpoint("a", ckpt_name)
            if candidate.exists():
                model_path = candidate
                break

        if model_path is None:
            expected_dir = resolve_checkpoint("a", "kinematics_best.pth").parent
            print(
                "❌ Model not found. Tried: "
                + ", ".join(str(expected_dir / name) for name in ckpt_candidates)
                + ". Skipping validation."
            )
            return None

        print(f"   ↳ Using model checkpoint: {model_path.name}")

        validator = ModelValidator(model_path=model_path, phase=phase)
        metrics = validator.validate_dataset(
            str(graph_dir),
            level_name=level_name,
            save_comparison_images=False,
        )
        if not comsol_available and metrics is not None:
            print("ℹ️ Running in AI-only mode: supervised GT metrics (for example rel_l2 / wss_corr) are expected to be NaN.")
        if visualize:
            _show_benchmark_visualization(
                validator=validator,
                graph_dir=graph_dir,
                phase=phase,
                level_idx=level_idx,
                level_name=level_name,
            )

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
    parser.add_argument("--phases", type=str, default=None, help='Comma-separated phases (for example: "kinematics,kinematics")')
    parser.add_argument("--num-samples", type=int, default=None, help="Number of vessels per benchmark level")
    parser.add_argument("--levels", type=str, default=None, help='Comma-separated benchmark levels (for example: "0,1")')
    parser.add_argument(
        "--carreau-n",
        type=float,
        default=None,
        help=f'Override Carreau n_index for non-kinematics runs (default for kinematics: {KINEMATICS_FINAL_ANNEALED_N:.3f})',
    )
    parser.add_argument(
        "--visualize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save per-level benchmark field visualizations (default: enabled)",
    )
    args = parser.parse_args()

    if args.phases is None:
        phases_raw = _prompt_text("Phases (comma-separated)", "kinematics")
    else:
        phases_raw = args.phases
    target_phases = [t.strip() for t in phases_raw.split(",") if t.strip()]

    if args.num_samples is None:
        num_samples = _prompt_int("Number of vessels per level", 10)
    else:
        num_samples = args.num_samples

    if args.levels is None:
        levels_raw = _prompt_text("Levels (comma-separated: 0=Straight pathologies, 1=Curved pathologies)", "0,1")
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

    for current_phase in target_phases:
        all_results = {}
        for lvl_idx, name in benchmarks:
            metrics = run_pipeline_for_level(
                current_phase,
                lvl_idx,
                name,
                num_samples=num_samples,
                visualize=bool(args.visualize),
                carreau_n=args.carreau_n,
            )
            if metrics is not None:
                all_results[name] = metrics
            time.sleep(1)

        print("\n\n" + "*" * 50)
        print(f"🏆 FINAL MULTI-FIDELITY BENCHMARK REPORT: {current_phase.upper()}")
        print("*" * 50)

        if all_results:
            df = pd.DataFrame(all_results).T
            print(df)
            save_path = reports_subdir("benchmark") / f"{current_phase}_full_benchmark.csv"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(save_path)
            print(f"\n📄 Detailed report saved to: {save_path}")
        else:
            print(f"❌ No results generated for {current_phase}.")

    # Full aggressive cleanup of all benchmark data once everything finishes
    benchmark_data_dir = data_root() / "benchmark"
    if benchmark_data_dir.exists():
        print(f"\n🧹 Final Cleanup: Sweeping up all generated temporary benchmark data at {benchmark_data_dir}...")
        shutil.rmtree(benchmark_data_dir, ignore_errors=True)
        print("✅ Cleanup complete.")