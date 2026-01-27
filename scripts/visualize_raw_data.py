"""
Visualize all processed patient contours on a single overlay.

This script loads all processed .npy files from data/processed/ and creates
overlay visualizations to assess the need for Procrustes Alignment.

Usage:
    python scripts/visualize_raw_data.py [--output-dir OUTPUT_DIR] [--no-display]
    
    --output-dir: Directory to save plot images (default: data/figures/)
    --no-display: Don't display plots interactively (useful for headless environments)
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_patient_data(data_dir: Path) -> dict[str, np.ndarray]:
    """Load all processed patient coordinate data."""
    npy_files = sorted(data_dir.glob("*.npy"))
    
    if not npy_files:
        raise FileNotFoundError(f"No .npy files found in {data_dir}")
    
    patient_data = {}
    for npy_file in npy_files:
        coords = np.load(npy_file)
        patient_name = npy_file.stem
        patient_data[patient_name] = coords
        print(f"  {patient_name}: {coords.shape[0]} nodes, shape {coords.shape}")
    
    return patient_data


def create_overlay_plot(patient_data: dict[str, np.ndarray], save_path: Path | None = None):
    """Create overlay plot of all patient contours."""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Generate distinct colors for each patient
    colors = plt.cm.tab20(np.linspace(0, 1, len(patient_data)))
    
    for idx, (patient_name, coords) in enumerate(patient_data.items()):
        x, y = coords[:, 0], coords[:, 1]
        
        # Plot as scatter points
        ax.scatter(x, y, s=1, alpha=0.6, color=colors[idx], label=patient_name)
    
    ax.set_xlabel('X coordinate (mm)', fontsize=12)
    ax.set_ylabel('Y coordinate (mm)', fontsize=12)
    ax.set_title(f'Overlay of {len(patient_data)} Patient Contours (Raw Data)', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved overlay plot to: {save_path}")
    
    return fig


def create_individual_plots(patient_data: dict[str, np.ndarray], save_path: Path | None = None):
    """Create individual subplots for each patient."""
    n_patients = len(patient_data)
    n_cols = 4
    n_rows = (n_patients + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten() if n_patients > 1 else [axes]
    
    for idx, (patient_name, coords) in enumerate(patient_data.items()):
        ax = axes[idx]
        x, y = coords[:, 0], coords[:, 1]
        
        ax.scatter(x, y, s=0.5, alpha=0.7)
        ax.set_title(patient_name, fontsize=10)
        ax.set_xlabel('X (mm)', fontsize=8)
        ax.set_ylabel('Y (mm)', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
    
    # Hide unused subplots
    for idx in range(len(patient_data), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved individual plots to: {save_path}")
    
    return fig


def print_statistics(patient_data: dict[str, np.ndarray]):
    """Print coordinate statistics for each patient."""
    print("\n" + "=" * 60)
    print("Coordinate Statistics:")
    print("=" * 60)
    for patient_name, coords in patient_data.items():
        x, y = coords[:, 0], coords[:, 1]
        print(f"\n{patient_name}:")
        print(f"  X range: [{x.min():.2f}, {x.max():.2f}], mean: {x.mean():.2f}")
        print(f"  Y range: [{y.min():.2f}, {y.max():.2f}], mean: {y.mean():.2f}")
        print(f"  Centroid: ({x.mean():.2f}, {y.mean():.2f})")
    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize all processed patient contours on a single overlay"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save plot images (default: data/figures/)"
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Don't display plots interactively"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory containing processed .npy files (default: data/processed/)"
    )
    
    args = parser.parse_args()
    
    # Get project root
    project_root = Path(__file__).parent.parent
    data_dir = args.data_dir or (project_root / "data" / "processed")
    output_dir = args.output_dir or (project_root / "data" / "figures")
    
    print(f"Loading patient data from: {data_dir}")
    patient_data = load_patient_data(data_dir)
    print(f"\nTotal patients loaded: {len(patient_data)}")
    
    # Print statistics
    print_statistics(patient_data)
    
    # Create overlay plot
    overlay_path = output_dir / "patient_contours_overlay.png"
    create_overlay_plot(patient_data, save_path=overlay_path)
    
    # Create individual plots
    individual_path = output_dir / "patient_contours_individual.png"
    create_individual_plots(patient_data, save_path=individual_path)
    
    # Display plots if requested
    if not args.no_display:
        plt.show()
    else:
        plt.close('all')
    
    print("\n" + "=" * 60)
    print("Visualization complete!")
    print("=" * 60)
    print("\nAssessment for Procrustes Alignment:")
    print("  - Check for scale differences")
    print("  - Check for translation differences (centroid locations)")
    print("  - Check for rotation differences")
    print("  - Identify true anatomical shape variations")
    print("\nIf significant differences are observed, apply Procrustes Alignment.")


if __name__ == "__main__":
    main()
