# Kinematics recovery sweep (~12 h)

Recover **Stage-A** val quality toward the April-2026 reference (~**Rel L2 0.10** @ epoch 84, 2000 graphs). Stretch goal: **Rel L2 &lt; 0.05** on the full stratified val set.

Uses the **main** graph tree (`data/processed/graphs_kinematics/newtonian`) — **not** `ab_bend_*` A/B dirs.

## Prerequisite (once)

```powershell
python -m src.data_gen.backfill_kinematics_geometry_level
```

Confirm graph count:

```powershell
(Get-ChildItem data\processed\graphs_kinematics\newtonian\vessel_*.pt).Count
```

**Do not** use `--limit-data` for this sweep (sorted-prefix bias; only sees first N files).

## One line (Comsol / P2200)

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_kinematics_recovery12h.ps1"
```

## Legs (8 × ~40–55 epochs ≈ 12 h @ ~3 min/epoch on ~500 graphs)

| Leg | Idea |
|-----|------|
| `A0_april_ratio` | LadHyX stage timing (40/60), no L0/L1-only warmstart |
| `F0_foundation` | Default geometry curriculum (6 ep L0/L1-only) |
| `F1_long_l0l1` | 12 ep L0/L1-only, hard mining @24 |
| `F2_no_curriculum` | Uniform sampling (legacy) |
| `F3_l1_warm_mining` | 10 ep L0/L1-only + hard mining @12 |
| `S0_shuffle_full` | `--shuffle-graphs` (fix lexicographic load bias) |
| `H0_data_heavy` | `--weight-data 800 --weight-wss 15` |
| `H1_low_mu` | `--weight-mu 5` |

Each leg writes an isolated checkpoint under `outputs/kinematics/sweep_recovery_12h/<leg_id>/` via `KINEMATICS_OUTPUT_DIR`.

## Morning leaderboard

```powershell
Get-Content outputs\kinematics\sweep_recovery_12h\manifest.jsonl |
  ForEach-Object { $_ | ConvertFrom-Json } |
  Sort-Object { [double]$_.best_rel_l2 } |
  Format-Table leg_id, best_rel_l2, best_l0, best_l1, best_l2, best_epoch, n_graphs
```

Promote winner:

```powershell
Copy-Item outputs\kinematics\sweep_recovery_12h\F0_foundation\kinematics_best.pth outputs\kinematics\kinematics_best.pth
```

Replace `F0_foundation` with the best `leg_id`.

## Notes

- **&lt; 5% Rel L2** on full mixed val may require **full graph count** (2000+) and **longer** than 12 h; this sweep finds the best *recipe* to scale next.
- Bend-sign A/B showed **bidirectional ≥ down_only** on 120-graph smoke; sweep does not regen meshes.
- April reference: [data/reference/kinematics_best_20260426T184600Z.json](../../data/reference/kinematics_best_20260426T184600Z.json).
