"""How many nodes do GT clots span, per anchor, over the trajectory.

GT clot = growth-only relu(mu_eff(t) - mu_eff(0)) >= thresh (gt_clot_phi_at_time).
Reports total band/graph nodes, clot node count over time, peak, fraction, and the
largest connected clot component (a proxy for clot 'length' in nodes).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.config import PhysicsConfig  # noqa: E402
from src.core_physics.species_pushforward_continuous import BIOCHEM_ANCHORS_6  # noqa: E402
from src.core_physics.t0_mu_physics import gt_clot_phi_at_time  # noqa: E402
from src.utils.paths import get_project_root  # noqa: E402


def _largest_cc(mask: torch.Tensor, edge_index: torch.Tensor) -> int:
    """Largest connected component size among clot nodes (undirected BFS)."""
    idx = torch.nonzero(mask, as_tuple=False).reshape(-1).tolist()
    if not idx:
        return 0
    idx_set = set(idx)
    src, dst = edge_index[0].tolist(), edge_index[1].tolist()
    adj: dict[int, list[int]] = {i: [] for i in idx}
    for a, b in zip(src, dst):
        if a in idx_set and b in idx_set:
            adj[a].append(b)
            adj[b].append(a)
    seen: set[int] = set()
    best = 0
    for s in idx:
        if s in seen:
            continue
        stack = [s]
        seen.add(s)
        cnt = 0
        while stack:
            u = stack.pop()
            cnt += 1
            for w in adj[u]:
                if w not in seen:
                    seen.add(w)
                    stack.append(w)
        best = max(best, cnt)
    return best


def main() -> None:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    phys = PhysicsConfig(phase="biochem")
    root = get_project_root()
    print(f"{'anchor':<11}{'nodes':>7}{'T':>5}{'peakClot':>10}{'peakFrac':>10}{'peakCC':>8}{'tPeak':>7}")
    print("-" * 58)
    for anchor in BIOCHEM_ANCHORS_6:
        p = root / "data/processed/graphs_biochem_anchors" / f"{anchor}.pt"
        if not p.is_file():
            print(f"{anchor:<11} (missing)")
            continue
        data = torch.load(p, map_location=dev, weights_only=False)
        n = int(data.num_nodes)
        T = int(data.y.shape[0])
        edge = data.edge_index.to(dev)
        peak = 0
        peak_t = 0
        for t in range(T):
            m = gt_clot_phi_at_time(data, t, phys, dev).reshape(-1) > 0.5
            c = int(m.sum().item())
            if c > peak:
                peak, peak_t = c, t
        m_peak = gt_clot_phi_at_time(data, peak_t, phys, dev).reshape(-1) > 0.5
        cc = _largest_cc(m_peak, edge)
        print(f"{anchor:<11}{n:>7}{T:>5}{peak:>10}{peak/max(n,1):>10.4f}{cc:>8}{peak_t:>7}")


if __name__ == "__main__":
    main()
