import numpy as np
from pathlib import Path
from src.geometry.registration import (
    extract_ordered_boundary, 
    reorder_contour, 
    resample_vessel, 
    ProcrustesAligner
)

def main():
    input_dir = Path("data/processed")
    output_dir = Path("data/aligned")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    raw_files = sorted(input_dir.glob("*.npy"))
    patient_shapes = {}
    
    print("--- Phase 1: Anatomical Registration ---")
    for f in raw_files:
        coords = np.load(f)
        # 1. Extract boundary (No ConvexHull!)
        boundary = extract_ordered_boundary(coords)
        # 2. Landmark Indexing (Inlet Centroid)
        ordered = reorder_contour(boundary)
        # 3. Resample to 1,000 nodes
        resampled = resample_vessel(ordered, n_nodes=1000)
        patient_shapes[f.stem] = resampled
        print(f"Registered {f.stem}")

    print("\n--- Phase 2: Procrustes Alignment ---")
    aligner = ProcrustesAligner()
    final_aligned = aligner.fit_transform(patient_shapes)
    
    for name, coords in final_aligned.items():
        np.save(output_dir / f"{name}.npy", coords)
    
    print(f"\nSuccess. Aligned {len(final_aligned)} patients to data/aligned/")

if __name__ == "__main__":
    main()