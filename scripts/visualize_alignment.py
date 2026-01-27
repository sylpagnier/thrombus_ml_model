"""
Visualize aligned patient contours with landmark verification.

This script loads aligned .npy files from data/aligned/ and creates:
1. An overlay plot showing all patients with first 20 points in Red and rest in Blue
2. A before/after comparison showing processed vs aligned data

The first 20 points are highlighted to verify that the 'Inlet Centroid' landmarking
has successfully aligned the starting indices and fixed the previous spiderwebbing.

Usage:
    python scripts/visualize_alignment.py [--output-dir OUTPUT_DIR] [--no-display]
    
    --output-dir: Directory to save plot images (default: data/figures/)
    --no-display: Don't display plots interactively (useful for headless environments)
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_patient_data(data_dir: Path) -> dict[str, np.ndarray]:
    """Load all patient coordinate data from .npy files."""
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


def create_aligned_overlay_plot(patient_data: dict[str, np.ndarray], save_path: Path | None = None):
    """
    Create overlay plot of all aligned patient contours.
    
    First 20 points of each contour are plotted in Red, rest in Blue.
    Uses line plots with proper contour closure and smart subsampling for clarity.
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Generate distinct colors for each patient (for the full contour)
    colors = plt.cm.tab20(np.linspace(0, 1, len(patient_data)))
    
    # Plot all patients
    for idx, (patient_name, coords) in enumerate(patient_data.items()):
        x, y = coords[:, 0], coords[:, 1]
        
        # Close the contour by adding first point at the end
        x_closed = np.append(x, x[0])
        y_closed = np.append(y, y[0])
        
        # Plot first 20 points in Red (as line, closed) - emphasized
        if len(coords) >= 20:
            # First 20 points + close back to start
            x_first20 = np.append(x[:20], x[0])
            y_first20 = np.append(y[:20], y[0])
            ax.plot(x_first20, y_first20, 'r-', linewidth=3, alpha=1.0, zorder=4, 
                   label='First 20 points' if idx == 0 else '')
            
            # Plot rest of contour in Blue (from point 20 onwards, closed)
            # For aligned data (1000 points), subsample to ~200 points for clarity
            if len(coords) > 200:
                step = max(1, len(coords) // 200)
                rest_indices = list(range(20, len(coords), step))
                # Ensure we close the contour
                if rest_indices[-1] != len(coords) - 1:
                    rest_indices.append(len(coords) - 1)
                rest_indices.append(0)  # Close the loop
                x_rest = x[rest_indices]
                y_rest = y[rest_indices]
            else:
                x_rest = x_closed[20:]
                y_rest = y_closed[20:]
            
            # Use patient-specific color for the full contour
            ax.plot(x_rest, y_rest, '-', linewidth=1.5, alpha=0.6, 
                   color=colors[idx], zorder=2, label=patient_name)
        else:
            # If fewer than 20 points, plot all in red
            ax.plot(x_closed, y_closed, 'r-', linewidth=3, alpha=1.0, label=patient_name)
    
    ax.set_xlabel('X coordinate (normalized)', fontsize=12)
    ax.set_ylabel('Y coordinate (normalized)', fontsize=12)
    ax.set_title('Aligned Patient Contours Overlay\n(Red: First 20 points, Colored: Rest of contour)', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved aligned overlay plot to: {save_path}")
    
    return fig


def create_before_after_comparison(
    processed_data: dict[str, np.ndarray],
    aligned_data: dict[str, np.ndarray],
    save_path: Path | None = None
):
    """
    Create a before/after comparison figure showing processed vs aligned data.
    
    Left panel: Processed data (before alignment) - uses scatter for dense data
    Right panel: Aligned data (after Procrustes alignment) - uses line plot
    Both show first 20 points in Red, rest in Blue.
    """
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    # Left panel: Processed data (before alignment)
    # For dense point clouds, use scatter plot with subsampling
    ax_before = axes[0]
    
    for idx, (patient_name, coords) in enumerate(processed_data.items()):
        x, y = coords[:, 0], coords[:, 1]
        
        # For very dense data, subsample for visualization
        if len(coords) > 1000:
            # Subsample: take every Nth point, but always include first 20
            step = max(1, len(coords) // 2000)  # Show ~2000 points max
            indices = list(range(20)) + list(range(20, len(coords), step))
            indices = sorted(set(indices))  # Remove duplicates, keep sorted
            x_viz = x[indices]
            y_viz = y[indices]
        else:
            x_viz = x
            y_viz = y
        
        # Plot first 20 points in Red (as line)
        if len(coords) >= 20:
            ax_before.plot(x[:21], y[:21], 'r-', linewidth=2.5, alpha=0.9, zorder=3)
            # Plot rest as scatter (for dense data) or line (for sparse)
            if len(coords) > 1000:
                # Scatter plot for dense data
                ax_before.scatter(x_viz[20:], y_viz[20:], s=0.5, c='blue', alpha=0.4, 
                                 zorder=1, label=patient_name if idx == 0 else '')
            else:
                # Line plot for sparse data
                x_closed = np.append(x[19:], x[0])
                y_closed = np.append(y[19:], y[0])
                ax_before.plot(x_closed, y_closed, 'b-', linewidth=1, alpha=0.5, 
                              zorder=2, label=patient_name)
        else:
            ax_before.plot(x, y, 'r-', linewidth=2.5, alpha=0.9, label=patient_name)
    
    ax_before.set_xlabel('X coordinate (mm)', fontsize=12)
    ax_before.set_ylabel('Y coordinate (mm)', fontsize=12)
    ax_before.set_title('Before Alignment (Processed Data)\n(Red: First 20 points, Blue: Rest)', 
                       fontsize=14, fontweight='bold')
    ax_before.grid(True, alpha=0.3)
    ax_before.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax_before.set_aspect('equal', adjustable='box')
    
    # Right panel: Aligned data (after alignment)
    # For resampled data (1000 points), use line plot with closed contour and subsampling
    ax_after = axes[1]
    colors_after = plt.cm.tab20(np.linspace(0, 1, len(aligned_data)))
    
    for idx, (patient_name, coords) in enumerate(aligned_data.items()):
        x, y = coords[:, 0], coords[:, 1]
        
        # Close the contour
        x_closed = np.append(x, x[0])
        y_closed = np.append(y, y[0])
        
        if len(coords) >= 20:
            # Plot first 20 points in Red (as line, closed) - emphasized
            x_first20 = np.append(x[:20], x[0])
            y_first20 = np.append(y[:20], y[0])
            ax_after.plot(x_first20, y_first20, 'r-', linewidth=3, alpha=1.0, zorder=4)
            
            # Plot rest in patient-specific color (as line, closed)
            # Subsample to ~200 points for clarity
            if len(coords) > 200:
                step = max(1, len(coords) // 200)
                rest_indices = list(range(20, len(coords), step))
                if rest_indices[-1] != len(coords) - 1:
                    rest_indices.append(len(coords) - 1)
                rest_indices.append(0)  # Close the loop
                x_rest = x[rest_indices]
                y_rest = y[rest_indices]
            else:
                x_rest = x_closed[20:]
                y_rest = y_closed[20:]
            
            ax_after.plot(x_rest, y_rest, '-', linewidth=1.5, alpha=0.6, 
                         color=colors_after[idx], zorder=2, label=patient_name)
        else:
            ax_after.plot(x_closed, y_closed, 'r-', linewidth=3, alpha=1.0, label=patient_name)
    
    ax_after.set_xlabel('X coordinate (normalized)', fontsize=12)
    ax_after.set_ylabel('Y coordinate (normalized)', fontsize=12)
    ax_after.set_title('After Alignment (Procrustes Aligned)\n(Red: First 20 points, Colored: Rest)', 
                      fontsize=14, fontweight='bold')
    ax_after.grid(True, alpha=0.3)
    ax_after.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax_after.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved before/after comparison to: {save_path}")
    
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Visualize aligned patient contours with landmark verification"
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
        "--aligned-dir",
        type=Path,
        default=None,
        help="Directory containing aligned .npy files (default: data/aligned/)"
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=None,
        help="Directory containing processed .npy files for comparison (default: data/processed/)"
    )
    
    args = parser.parse_args()
    
    # Get project root
    project_root = Path(__file__).parent.parent
    aligned_dir = args.aligned_dir or (project_root / "data" / "aligned")
    processed_dir = args.processed_dir or (project_root / "data" / "processed")
    output_dir = args.output_dir or (project_root / "data" / "figures")
    
    print("=" * 60)
    print("Loading aligned patient data...")
    print(f"From: {aligned_dir}")
    aligned_data = load_patient_data(aligned_dir)
    print(f"\nTotal aligned patients loaded: {len(aligned_data)}")
    
    # Create aligned overlay plot
    print("\n" + "=" * 60)
    print("Creating aligned overlay plot...")
    overlay_path = output_dir / "aligned_overlay.png"
    create_aligned_overlay_plot(aligned_data, save_path=overlay_path)
    
    # Create before/after comparison if processed data exists
    if processed_dir.exists():
        print("\n" + "=" * 60)
        print("Loading processed patient data for comparison...")
        print(f"From: {processed_dir}")
        try:
            processed_data = load_patient_data(processed_dir)
            print(f"\nTotal processed patients loaded: {len(processed_data)}")
            
            # Match patient names between processed and aligned
            common_patients = set(processed_data.keys()) & set(aligned_data.keys())
            if common_patients:
                processed_data = {k: processed_data[k] for k in common_patients}
                aligned_data_filtered = {k: aligned_data[k] for k in common_patients}
                
                print("\n" + "=" * 60)
                print("Creating before/after comparison...")
                comparison_path = output_dir / "alignment_before_after.png"
                create_before_after_comparison(
                    processed_data, 
                    aligned_data_filtered, 
                    save_path=comparison_path
                )
            else:
                print("\nWarning: No common patient names found between processed and aligned data.")
        except FileNotFoundError:
            print(f"\nWarning: Could not load processed data from {processed_dir}")
            print("Skipping before/after comparison.")
    else:
        print(f"\nNote: Processed data directory {processed_dir} not found.")
        print("Skipping before/after comparison.")
    
    # Display plots if requested
    if not args.no_display:
        plt.show()
    else:
        plt.close('all')
    
    print("\n" + "=" * 60)
    print("Visualization complete!")
    print("=" * 60)
    print("\nVerification Checklist:")
    print("  [*] Check that red segments (first 20 points) align across patients")
    print("  [*] Verify no spiderwebbing at the inlet region")
    print("  [*] Confirm consistent starting indices (Inlet Centroid landmarking)")
    print("  [*] Assess overall shape alignment after Procrustes transformation")


if __name__ == "__main__":
    main()
