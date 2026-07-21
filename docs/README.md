# Documentation index

Active design and operator docs for **HemoRGP**. Lab notebooks, sweep logs, and retired ladders live under [`archive/`](archive/).

## Start here

| Doc | Purpose |
|-----|---------|
| [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) | Goals, stages, source map, CLI entry points |
| [MODEL_NOMENCLATURE.md](MODEL_NOMENCLATURE.md) | RGP-DEQ, biochem_gnn, local corrector IDs |
| [PUBLISHING.md](PUBLISHING.md) | What is git-tracked vs local-only |

## Stage A — flow (RGP-DEQ)

| Doc | Purpose |
|-----|---------|
| [KINEMATICS_BEST_ARCHITECTURE.md](KINEMATICS_BEST_ARCHITECTURE.md) | Locked architecture + training recipe |
| [LOCAL_KINEMATIC_CORRECTOR.md](LOCAL_KINEMATIC_CORRECTOR.md) | Optional k-hop clot diversion GNN |
| [COMSOL_PHYSICS_VALIDATION.md](COMSOL_PHYSICS_VALIDATION.md) | Flow / physics parity vs COMSOL |
| [COMSOL_MU_RHEOLOGY_CHECKLIST.md](COMSOL_MU_RHEOLOGY_CHECKLIST.md) | Viscosity / rheology checklist |

## Stage B — biochemistry / clot

| Doc | Purpose |
|-----|---------|
| [BIOCHEM_GNN.md](BIOCHEM_GNN.md) | Deploy stack (`biochem_gnn`) |
| [MAT_GROWTH.md](MAT_GROWTH.md) | Canonical mat-growth baseline and how to extend it |
| [BIOCHEM_LEGACY_LESSONS.md](BIOCHEM_LEGACY_LESSONS.md) | Condensed takeaways from retired ladders |

## Operators

| Doc | Purpose |
|-----|---------|
| [../scripts/README.md](../scripts/README.md) | Supported launchers |
| [../AGENTS.md](../AGENTS.md) | Short agent / contributor cheat sheet |
| [../data/reference/README.md](../data/reference/README.md) | Tracked baseline manifests |

## Archive

Historical chronicles, baseline leaderboards, decision dumps, and cleanup notes:

- [archive/README.md](archive/README.md)
