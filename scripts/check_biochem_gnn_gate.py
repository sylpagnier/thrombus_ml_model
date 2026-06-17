"""Gate: promoted biochem GNN baseline artifacts + p007 F1 floor."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.biochem_gnn.config import (
    GATE_F1_MIN_P007,
    S0_F1_TARGET_P007,
    STACK_NAME,
    load_manifest,
    locked_root_path,
    reference_manifest_path,
)
from src.utils.paths import get_project_root


def main() -> int:
    root = get_project_root()
    ref_path = reference_manifest_path()
    manifest = load_manifest(ref_path)
    failures: list[str] = []

    for key in ("species_gnn_ckpt", "viscosity_beta"):
        p = root / str(manifest.get(key, ""))
        if not p.is_file():
            failures.append(f"missing {key}: {p}")

    loao_dir = root / str(manifest.get("loao_dir", ""))
    for anc in manifest.get("loao_preferred") or []:
        fold = loao_dir / f"holdout_{anc}" / "best.pth"
        if not fold.is_file():
            failures.append(f"missing LOAO fold: {fold}")

    eval_path = locked_root_path() / "eval_summary.json"
    eval_data: dict = {}
    if ref_path.is_file():
        ref = json.loads(ref_path.read_text(encoding="utf-8"))
        eval_data = dict(ref.get("eval") or {})
    if eval_path.is_file():
        eval_data.update(json.loads(eval_path.read_text(encoding="utf-8")))

    p007 = eval_data.get("patient007_clot_f1_t53") or eval_data.get("patient007_clot_f1_t_last")
    if p007 is None:
        failures.append("missing patient007 clot F1 (patient007_clot_f1_t_last) in eval summary")
    elif float(p007) < GATE_F1_MIN_P007:
        failures.append(f"patient007 F1 {float(p007):.3f} < {GATE_F1_MIN_P007}")
    elif float(p007) < S0_F1_TARGET_P007:
        failures.append(f"patient007 F1 {float(p007):.3f} does not beat s0 {S0_F1_TARGET_P007}")

    if failures:
        for f in failures:
            print(f"[FAIL] {f}", file=sys.stderr)
        return 1

    print(f"[OK] {STACK_NAME} gate pass (p007 F1={float(p007):.3f} vs s0 {S0_F1_TARGET_P007})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
