# Species temporal ML

Fresh ML path for wall-band species (see diagnostic:
`scripts/diagnose_species_temporal_patterns.py`).

## M0: wall-band GNN (1-step teacher-forced)

Channel sets:

| Set | Channels | Role |
|-----|----------|------|
| `fimat` | FI, Mat | Gelation pair only |
| `cascade4` | APR, APS, FI, Mat | Cascade + gelation |

```powershell
# FI+Mat (2 ch)
powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_wall_band_species_m0.ps1" -ChannelSet fimat -Fresh

# Cascade + gelation (4 ch)
powershell ... -ChannelSet cascade4 -Fresh

# Compare both
powershell ... -CompareBoth -Fresh

# Eval/viz only
powershell ... -ChannelSet fimat -SkipTrain -Anchor patient007
```

Artifacts:

- `outputs/biochem/wall_band_species_m0_fimat/best.pth`
- `outputs/biochem/wall_band_species_m0_cascade4/best.pth`
- eval `outputs/biochem/clot_trigger/wall_band_m0_<set>_<anchor>.json`
- viz `outputs/biochem/viz/clot_trigger/wall_band_m0_<set>_<anchor>.png` (GT | s0 | M0)

Code: `src/core_physics/wall_band_species_m0.py`, `src/training/train_wall_band_species_m0.py`.
