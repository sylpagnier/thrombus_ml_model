# Biochem legacy lessons

Condensed takeaways from retired ladders (GNODE, clot-ML rules, T0, graybox S0–S3). Do not treat archived launchers as the supported surface.

## Canonical baseline going forward

- Stack id: **`biochem_gnn`** (`src/biochem_gnn/`; alias `src.biochem_deploy`)
- Lock artifacts under `outputs/biochem/biochem_gnn/locked/` (local)
- Reference: `data/reference/biochem_gnn_baseline.json`
- Design: [BIOCHEM_GNN.md](BIOCHEM_GNN.md), [MAT_GROWTH.md](MAT_GROWTH.md)

## Lessons worth keeping

- **Global time normalization:** one static `t_ref` (default `t_final`), never per-graph max.
- **Deploy-faithful evaluation:** no oracle GT clot bands in gates used for deploy claims.
- **Small gates first:** short smoke on one anchor before long ladders.
- **Physics consistency over metric hacks:** avoid settings that improve one number while breaking spatial clot behavior.
- **Alias control:** legacy names may resolve for compatibility; ownership stays on `biochem_gnn`.

## Why ladders were archived

- Overlapping launchers and duplicated trainers raised maintenance cost.
- Provenance still matters; clutter does not. Active scripts: [`scripts/README.md`](../scripts/README.md).

## Resurrecting an archived path

1. Read [archive/2026-06-16-biochem-cleanup.md](archive/2026-06-16-biochem-cleanup.md).
2. Restore the minimum files from git history.
3. Prefer re-implementing against `src/biochem_gnn/` when possible.
