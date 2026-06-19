# Biochem Legacy Lessons (Consolidated)

This file preserves the main lessons from legacy ladders (S ladder, T0 ladder, rules ladder, GNODE ladder, passive/m3 ladders) after repository cleanup.

## Canonical baseline going forward

- Use `biochem_deploy` as the default stack.
- Lock artifacts in `outputs/biochem/biochem_gnn/locked/`.
- Use `data/reference/biochem_gnn_baseline.json` as the canonical reference.

## Lessons worth keeping

- **Global time normalization:** use one static `t_ref` (`t_final`, 30000s default), never per-graph max.
- **Deploy-faithful evaluation:** no oracle GT clot bands in training/eval gates for deploy claims.
- **Small, cheap gates first:** short smoke passes on one anchor before long ladders.
- **Physics consistency over metric hacks:** avoid settings that improve one metric while breaking spatial clot behavior.
- **Alias control:** legacy names can exist for compatibility, but baseline ownership must be canonical (`biochem_deploy`).

## Why legacy ladders were archived

- Large overlap and duplicated launchers caused maintenance overhead.
- Historical ladders remained useful for provenance, but they were cluttering active baseline work.
- Cleanup goal: preserve learnings in one place and keep only active baseline scripts/docs in top-level paths.

## If we need to resurrect an archived path

- Use the cleanup archive record in `docs/archive/2026-06-16-biochem-cleanup.md`.
- Restore from git history by file path and commit.
- Re-add only the minimum scripts needed for the immediate experiment.

