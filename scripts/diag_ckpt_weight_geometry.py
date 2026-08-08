"""Weight-space geometry between checkpoints sharing one warm start.

WALL_MODEL_PLAN.md s25.1 measured the finding this project now turns on:

    ||Phase1cfg - warm|| / ||warm||   = 0.3058
    ||3a cfg    - warm|| / ||warm||   = 0.3016
    ||Phase1cfg - 3a cfg|| / ||warm|| = 0.0795

Two materially different objectives moved ~30% from the shared warm start and ended only 26%
as far APART as either had moved -- about three quarters of the weight update is common to
both. That number was computed ad hoc. It is the premise of T1, T3 and the reading of five
nulls, so it gets a script.

Usage:
    python scripts/diag_ckpt_weight_geometry.py \
        --ref outputs/biochem/eda/wall_gen_clotrich_nplus/WG_clotrich_nplus/best.pth \
        --ckpt A=outputs/biochem/eda/t1/WG_t1a_perstep_only/best.pth \
        --ckpt B=outputs/biochem/eda/t1/WG_t1b_rolledf1_only/best.pth

`--scope trainable` (default) restricts to the tensors a frozen-backbone leg can actually
move. Comparing over all 40 tensors dilutes the measurement with 32 that are identical by
construction, which understates every distance by the same factor and makes legs look more
similar than they are.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

# The frozen-backbone legs train these two readout MLPs and nothing else: 45,186 of 186,887
# parameters (24%). See the `freeze_backbone: frozen=32 trainable_heads=8` line in any leg log.
TRAINABLE_PREFIXES = ("spatial_head.", "magnitude_head.")


def load_state(path: Path) -> tuple[dict[str, torch.Tensor], int | None]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck.get("model_state") if isinstance(ck, dict) else None
    if sd is None:
        sd = ck.get("model") or ck.get("state_dict") or ck
    meta = ck.get("meta", {}) if isinstance(ck, dict) else {}
    ep = meta.get("epoch", meta.get("best_epoch", meta.get("salvage_epoch")))
    tensors = {k: v.detach().float() for k, v in sd.items() if isinstance(v, torch.Tensor)}
    return tensors, (int(ep) if ep is not None else None)


def select(sd: dict[str, torch.Tensor], scope: str) -> dict[str, torch.Tensor]:
    if scope == "all":
        return sd
    return {k: v for k, v in sd.items() if k.startswith(TRAINABLE_PREFIXES)}


def flat_norm(sd: dict[str, torch.Tensor], keys: list[str]) -> float:
    return math.sqrt(sum(float(sd[k].pow(2).sum()) for k in keys))


def flat_dist(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor], keys: list[str]) -> float:
    return math.sqrt(sum(float((a[k] - b[k]).pow(2).sum()) for k in keys))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="the shared warm start")
    ap.add_argument("--ckpt", action="append", default=[], metavar="NAME=PATH",
                    help="repeatable; NAME=PATH, or bare PATH to name it after its parent dir")
    ap.add_argument("--scope", choices=("trainable", "all"), default="trainable")
    args = ap.parse_args()

    ref_sd, _ = load_state(Path(args.ref))
    ref = select(ref_sd, args.scope)
    named: list[tuple[str, dict[str, torch.Tensor]]] = []
    epochs: dict[str, int | None] = {}
    for spec in args.ckpt:
        name, _, path = spec.partition("=")
        if not path:
            path, name = name, Path(name).parent.name
        sd, ep = load_state(Path(path))
        named.append((name, select(sd, args.scope)))
        epochs[name] = ep

    if not named:
        print("[ERR] pass at least one --ckpt")
        return 1

    keys = sorted(set(ref) & set.intersection(*(set(sd) for _, sd in named)))
    if not keys:
        print("[ERR] no tensors in common")
        return 1
    n_par = sum(ref[k].numel() for k in keys)
    ref_norm = flat_norm(ref, keys)
    print(f"[i] scope={args.scope}  tensors={len(keys)}  params={n_par}  ||ref||={ref_norm:.6g}")
    if ref_norm <= 0:
        print("[ERR] reference has zero norm")
        return 1

    print("\n-- movement from the warm start (relative) --")
    for name, sd in named:
        ep = epochs.get(name)
        tag = f"  [epoch {ep}]" if ep is not None else "  [epoch unknown]"
        print(f"   ||{name} - ref|| / ||ref|| = {flat_dist(ref, sd, keys) / ref_norm:.4f}{tag}")

    # Distance grows with training time. Comparing a leg's epoch 6 against another's epoch 1
    # measures mostly "more epochs moved further", not "a different objective went elsewhere",
    # and reads as divergence whether or not the objectives differ at all. s25.1's numbers were
    # one epoch against one epoch; note that its Phase-1 figure (0.3058) is NOT what the
    # SELECTED Phase-1 checkpoint gives, because selection kept epoch 4.
    seen = {name: ep for name, ep in epochs.items() if ep is not None}
    if len(set(seen.values())) > 1:
        print("\n[WARN] epoch mismatch: " + ", ".join(f"{n}=ep{e}" for n, e in seen.items()))
        print("[WARN] pairwise separation below is CONFOUNDED by training time. Compare "
              "checkpoints from the same epoch (e.g. last.pth from equal-length runs).")

    if len(named) > 1:
        print("\n-- pairwise separation, on the same denominator --")
        for i in range(len(named)):
            for j in range(i + 1, len(named)):
                (na, a), (nb, b) = named[i], named[j]
                d = flat_dist(a, b, keys) / ref_norm
                ma = flat_dist(ref, a, keys) / ref_norm
                mb = flat_dist(ref, b, keys) / ref_norm
                mean_move = 0.5 * (ma + mb)
                ratio = d / mean_move if mean_move > 0 else float("nan")
                print(f"   ||{na} - {nb}|| / ||ref|| = {d:.4f}")
                print(f"      as a fraction of how far they each moved: {ratio:.1%}")
                # s25.1's reading: a small ratio means the objectives are landing in the same
                # place regardless of what they ask for, which is a parameterisation problem
                # (T3), not an objective problem.
                verdict = ("CONVERGENT -- the objectives are not steering"
                           if ratio < 0.5 else
                           "DIVERGENT -- the objective does steer the weights")
                print(f"      {verdict}")

    print("\n-- per-tensor relative movement --")
    hdr = "   " + "tensor".ljust(28) + "".join(f"{n[:11]:>13s}" for n, _ in named)
    print(hdr)
    for k in keys:
        kn = math.sqrt(float(ref[k].pow(2).sum()))
        cells = "".join(
            f"{(math.sqrt(float((sd[k] - ref[k]).pow(2).sum())) / kn if kn > 0 else float('nan')):>13.4f}"
            for _, sd in named
        )
        print("   " + k.ljust(28) + cells)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
