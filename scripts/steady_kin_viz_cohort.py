"""Batch steady GINO-DEQ viz: patient anchors and/or kinematics vessels by geometry level.

Prints rel_L2(uvp) per case and opens matplotlib figures (close each to advance).

Examples:
    python scripts/steady_kin_viz_cohort.py --patients
    python scripts/steady_kin_viz_cohort.py --patients --stems patient001,patient007
    python scripts/steady_kin_viz_cohort.py --level 2 --max-vessels 5
    python scripts/steady_kin_viz_cohort.py --level 2 --rheology carreau --max-vessels 5
    python scripts/steady_kin_viz_cohort.py --level 2 --max-vessels 3 --patients --no-show
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.evaluation.visualize_pipeline import (
    _kinematics_graph_dir,
    _kinematics_anchor_graph_path,
    _list_anchor_stems,
    _load_graph_pt,
    _load_kinematics_gino_deq,
    _rel_l2_uvp,
    _run_model_once,
    _show_steady_kinematics_pred_vs_gt,
    _steady_kine_target_tensor,
)


def _vessel_paths_for_level(level: int, kine_dir: Path, max_vessels: int) -> List[Tuple[str, Path]]:
    out: List[Tuple[str, Path]] = []
    for path in sorted(kine_dir.glob("vessel_*.pt")):
        if max_vessels > 0 and len(out) >= max_vessels:
            break
        data = torch.load(path, map_location="cpu", weights_only=False)
        lvl = -1
        if hasattr(data, "geometry_level"):
            lvl = int(data.geometry_level.view(-1)[0].item())
        elif path.with_suffix(".json").exists():
            import json

            meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
            lvl = int(meta.get("level", -1))
        else:
            mesh_json = _REPO / "data/raw/kinematics/meshes" / f"{path.stem}.json"
            if mesh_json.is_file():
                import json

                meta = json.loads(mesh_json.read_text(encoding="utf-8"))
                lvl = int(meta.get("level", -1))
        if lvl == int(level):
            out.append((path.stem, path))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--patients", action="store_true", help="Include biochem kine-anchor graphs.")
    p.add_argument("--stems", type=str, default="", help="Comma-separated patient stems (default: all).")
    p.add_argument("--level", type=int, default=2, help="Kinematics geometry level for --vessels (default: 2).")
    p.add_argument(
        "--rheology",
        type=str,
        default="newtonian",
        choices=("newtonian", "carreau"),
        help="Graph tree under data/processed/graphs_kinematics/<rheology>/ (default: newtonian).",
    )
    p.add_argument("--max-vessels", type=int, default=5, help="Cap synthetic vessels at --level (0 = all).")
    p.add_argument("--vessels", action="store_true", help="Include synthetic kinematics graphs at --level.")
    p.add_argument("--time-index", type=int, default=0, help="Label time index for transient y (default: 0).")
    p.add_argument("--no-show", action="store_true", help="Print metrics only; do not open figures.")
    args = p.parse_args()

    if not args.patients and not args.vessels:
        args.patients = True
        args.vessels = True

    cases: List[Tuple[str, str, Path]] = []
    if args.patients:
        stems = [s.strip() for s in args.stems.split(",") if s.strip()] or _list_anchor_stems()
        for stem in stems:
            kpath = _kinematics_anchor_graph_path(stem, "newtonian")
            if kpath.is_file():
                cases.append(("patient", stem, kpath))
            else:
                print(f"[WARN] skip {stem}: missing {kpath}")

    rheology = str(args.rheology).strip().lower()
    if args.vessels:
        kdir = _kinematics_graph_dir(rheology)
        if not kdir.is_dir():
            print(f"[WARN] kinematics graph dir missing: {kdir}")
            print(
                f"  Generate with: python -m src.data_gen.pipeline_kinematics "
                f"--non-interactive --rheology {rheology} --mixed-levels -n <N> ..."
            )
        else:
            found = _vessel_paths_for_level(int(args.level), kdir, int(args.max_vessels))
            if not found:
                print(
                    f"[WARN] no level={args.level} vessels under {kdir}. "
                    "Run backfill: python -m src.data_gen.backfill_kinematics_geometry_level"
                )
            for label, path in found:
                cases.append(("kinematics", label, path))

    if not cases:
        raise SystemExit("[ERR] no cases selected")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[i] steady kin cohort on {device} ({len(cases)} case(s))")
    model = _load_kinematics_gino_deq(device)

    rows = []
    for cohort, label, graph_path in cases:
        data = _load_graph_pt(graph_path, device, phase_hint=cohort)
        with torch.no_grad():
            pred = _run_model_once(model, data)
        pred_np = pred.detach().cpu().numpy()
        tgt = _steady_kine_target_tensor(data, time_index=int(args.time_index))
        rel = None
        if tgt is not None:
            gt_np = tgt.detach().cpu().numpy()
            gt_uv_norm = float((gt_np[:, :3] ** 2).sum())
            if gt_uv_norm < 1e-12:
                print(
                    f"  {cohort:10s} {label:12s} [skip] GT u,v,p ~= 0 in .pt "
                    "(missing Carreau npz labels; re-run mesh_to_graph or copy complete graphs)"
                )
                continue
            rel = _rel_l2_uvp(pred_np, gt_np)
            rows.append((cohort, label, rel, graph_path))
            print(f"  {cohort:10s} {label:12s} rel_L2(uvp)={rel:.4f}")
        else:
            print(f"  {cohort:10s} {label:12s} (no GT y)")

        if not args.no_show:
            pos = data.x[:, :2].detach().cpu().numpy()
            gt_np = tgt.detach().cpu().numpy() if tgt is not None else None
            _show_steady_kinematics_pred_vs_gt(
                pos, pred_np, gt_np, case_label=label, cohort=cohort, rel_l2=rel
            )

    if rows:
        print("\n[i] summary (sorted by rel_L2):")
        for cohort, label, rel, _ in sorted(rows, key=lambda r: r[2]):
            print(f"  {rel:.4f}  {cohort}/{label}")

    if not args.no_show:
        print("[i] Close each figure window to exit.")
        plt.show()


if __name__ == "__main__":
    main()
