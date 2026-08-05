import argparse
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
from src.config import NodeFeat

def main():
    parser = argparse.ArgumentParser(description="Precache y-mirrored versions of anchor graph .pt files.")
    parser.add_argument("--anchors", type=str, required=True, help="Comma-separated list of anchors to mirror.")
    parser.add_argument("--graph-dir", type=str, default="data/processed/graphs_biochem_anchors", help="Directory containing the graph .pt files.")
    
    args = parser.parse_args()
    
    anchor_list = [a.strip() for a in args.anchors.split(",") if a.strip()]
    graph_dir = Path(args.graph_dir)
    
    if not graph_dir.exists():
        print(f"[WARN] Directory {graph_dir} does not exist.")
        return

    for anchor in anchor_list:
        file_path = graph_dir / f"{anchor}.pt"
        if not file_path.exists():
            print(f"[WARN] File not found: {file_path}")
            continue
        
        # Load the PyG Data object
        data = torch.load(file_path, weights_only=False)
        
        # 2. Flip y-coordinate: data.x[:, 1] = -data.x[:, 1]
        data.x[:, 1] = -data.x[:, 1]
        
        # 3. Flip v-velocity in time series: data.y[:, :, 1] = -data.y[:, :, 1]
        data.y[:, :, 1] = -data.y[:, :, 1]
        
        # 4. Flip wall normal y-component. Wall normals are at NodeFeat.WALL_NORMAL = slice(4, 6)
        data.x[:, 5] = -data.x[:, 5]
        
        # 5. If data has u0_pred, flip v
        if hasattr(data, 'u0_pred') and data.u0_pred is not None and data.u0_pred.shape[1] > 1:
            data.u0_pred[:, 1] = -data.u0_pred[:, 1]
            
        # 6. Save to {graph_dir}/{anchor}_mirror_y.pt
        out_path = graph_dir / f"{anchor}_mirror_y.pt"
        torch.save(data, out_path)
        
        # 7. Print [OK]
        print(f"[OK] Mirrored {anchor} -> {out_path.name}")

if __name__ == '__main__':
    main()
