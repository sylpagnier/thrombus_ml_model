"""Pick per-anchor ckpt: LOAO fold vs global s34 by deploy clot F1 @ t=53."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.biochem_gnn.config import staging_ckpt_pick_path  # noqa: E402
from src.inference.predict_species_gnn_deploy import predict_species_gnn_deploy  # noqa: E402
from src.inference.species_gnn_deploy_env import (  # noqa: E402
    DEFAULT_MANIFEST,
    load_deploy_manifest,
    resolve_loao_ckpt_for_anchor,
)
from src.core_physics.species_pushforward_continuous import BIOCHEM_ANCHORS_6  # noqa: E402
from src.core_physics.t0_device import require_cuda_device  # noqa: E402
from src.utils.paths import get_project_root  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Select LOAO vs global ckpt per anchor")
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--anchors", default=",".join(BIOCHEM_ANCHORS_6))
    ap.add_argument("--flow", default="gt")
    ap.add_argument("--s0-f1", type=float, default=0.408, help="Target to beat (p007 rules baseline)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    device = require_cuda_device()
    root = get_project_root()
    manifest = load_deploy_manifest(args.manifest.strip() or None)
    global_ckpt = str(manifest.get("species_gnn_ckpt"))
    anchors = [a.strip() for a in args.anchors.split(",") if a.strip()]
    loao_preferred: list[str] = []
    overrides: dict[str, str] = {}
    rows: list[dict] = []

    for anc in anchors:
        graph = root / "data/processed/graphs_biochem_anchors" / f"{anc}.pt"
        loao_ckpt = resolve_loao_ckpt_for_anchor(anc, manifest.get("loao_dir", ""))
        r_global = predict_species_gnn_deploy(
            graph, device=device, flow_source=args.flow, manifest=manifest,
            loao=False, species_ckpt=global_ckpt, times=[53],
        )
        r_loao = None
        if loao_ckpt.is_file():
            r_loao = predict_species_gnn_deploy(
                graph, device=device, flow_source=args.flow, manifest=manifest,
                loao=True, species_ckpt=str(loao_ckpt), times=[53],
            )
        f1_g = float(r_global["clot_f1_t_last"])
        f1_l = float(r_loao["clot_f1_t_last"]) if r_loao else -1.0
        use_loao = r_loao is not None and f1_l > f1_g + 1e-4
        if use_loao:
            loao_preferred.append(anc)
            chosen = str(loao_ckpt)
        else:
            chosen = global_ckpt
        overrides[anc] = chosen
        beat_s0 = f1_g >= float(args.s0_f1) or (use_loao and f1_l >= float(args.s0_f1))
        row = {
            "anchor": anc,
            "f1_global": f1_g,
            "f1_loao": f1_l,
            "pick": "loao" if use_loao else "global",
            "ckpt": chosen,
            "beats_s0_target": beat_s0,
            "health_global": r_global.get("health_pass"),
            "health_loao": r_loao.get("health_pass") if r_loao else None,
        }
        rows.append(row)
        print(
            f"[i] {anc} global={f1_g:.3f} loao={f1_l:.3f} -> {row['pick']} "
            f"beat_{args.s0_f1:.2f}={beat_s0}",
            flush=True,
        )

    manifest["loao_preferred"] = loao_preferred
    manifest["ckpt_overrides"] = overrides
    out = Path(args.out) if args.out.strip() else Path(args.manifest or DEFAULT_MANIFEST)
    if not out.is_absolute():
        out = root / out
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    summary = staging_ckpt_pick_path()
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    mean_g = sum(r["f1_global"] for r in rows) / max(len(rows), 1)
    mean_best = sum(max(r["f1_global"], r["f1_loao"]) for r in rows) / max(len(rows), 1)
    print(f"[OK] mean_f1 global={mean_g:.3f} best-of={mean_best:.3f} manifest={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
