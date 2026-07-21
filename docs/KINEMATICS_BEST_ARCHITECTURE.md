# Kinematics best-model architecture (Stage A)

Canonical architecture and training flags for the production **RGP-DEQ** run (`RGP_DEQ` class, `rgp_deq_kine` id) that biochem should match when building the frozen kinematics backbone. Weights are not required in this repo; the committed reference JSON is the source of truth for constructor kwargs and curriculum.

## Reference files

| Path | Purpose |
|------|---------|
| [data/reference/kinematics_best_20260426T184600Z.json](../data/reference/kinematics_best_20260426T184600Z.json) | **`model_config`** + **`training_recipe`** + best-epoch val metrics (no torch, no weights) |
| `outputs/kinematics/kinematics_architecture.json` | Optional: updated when you train kinematics here and save a new best |
| `outputs/kinematics/kinematics_best.pth` | Optional: weights; if present, may also embed the same `model_config` |

Override the reference JSON with `KINEMATICS_MODEL_CONFIG_REF=/path/to/manifest.json`.

## Best run (2026-04-26, LadHyX reference)

- **Source project**: `LadHyX_ml_cfd_thrombus_predictions`
- **Run id**: `20260426T184600Z`
- **Best epoch**: 84 (Stage 3 Carreau), **before** L-BFGS steps produced NaN validation
- **Val**: Rel L2 ≈ 0.1007, \|∇·u\| mean ≈ 0.157, composite ≈ 15.80

## Production run (2026-06-01, thrombus_ml_model)

Script: `scripts/go_kinematics_production_allfix.ps1` — allfix toggles, **3000 graphs** (no cap), shuffle seed 42, 100 ep / Adam 85 / LBFGS 85–99.

| Field | Value |
|-------|-------|
| **Diary run id** | `20260601T180106Z` |
| **Checkpoint dir** | `outputs/kinematics/production_allfix/` |
| **Best epoch (saved)** | **80** (Adam, Stage 3 Carreau) |
| **Val Rel L2** | **0.1263** |
| **L0 / L1 / L2** | 0.113 / **0.120** / 0.167 |
| **div_u mean** | 0.307 |
| **composite** | 30.85 |
| **Graphs** | 3000 newtonian + 3000 carreau (L0=1200, L1=1200, L2=600) |

**L-BFGS (epochs 85–99):** same failure mode as April — ep 85 flat, ep 86+ Rel L2 rises, ep 98–99 **NaN** val. `kinematics_best.pth` correctly kept at **ep 80** (pre-L-BFGS).

### Run log (validation highlights)

| Epoch | Stage | Rel L2 | L1 | div_u | Note |
|-------|-------|--------|-----|-------|------|
| 0 | 1 Newtonian | 0.981 | 1.011 | 1.00 | cold start |
| 12 | 1 | **0.645** | 0.613 | 0.318 | best Newtonian |
| 40 | 2 ramp | 0.736 | 0.737 | 0.198 | stage-2 bump |
| 60 | 3 Carreau | 0.168 | 0.162 | 0.407 | first Carreau save |
| 72 | 3 | 0.130 | 0.124 | 0.317 | |
| 80 | 3 | **0.126** | **0.120** | 0.307 | **best saved** |
| 84 | 3 | 0.126 | 0.120 | 0.308 | last Adam (April analog) |
| 85 | 3 LBFGS | 0.127 | 0.120 | 0.309 | LBFGS start |
| 99 | 3 LBFGS | NaN | — | NaN | |

### Comparisons

| Run | Graphs | Best Rel L2 | L1 | div_u | Best ep |
|-----|--------|-------------|-----|-------|---------|
| April reference | ~2000 | 0.101 | — | 0.157 | 84 (pre-LBFGS) |
| 30-ep allfix smoke | 2000 cap | 0.132 | 0.135 | 0.335 | 29 |
| **Production allfix** | **3000** | **0.126** | **0.120** | **0.307** | **80** |
| **Finetune allfix** (ContinuityFocus) | **3000** | **0.087** | **0.083** | **0.233** | **119** |

Full graphs + long Carreau beat the 30-ep smoke (~4% Rel L2 gain; L1 **0.120** vs 0.135). Production ep 80 still **~0.026** above April on Rel L2. **Continuity finetune ep 119 beats April** on Rel L2 (0.087 vs 0.101) and div_u (0.233 vs 0.157); L2 subset still ~0.120.

### Finetune run (2026-06-03, `go_kinematics_production_allfix_finetune.ps1`)

Resume from production `kinematics_best.pth` (ep 80). `kinematics_validation.jsonl` appends multiple attempts; **use ep 119 as the promoted best**.

| Phase | Epochs | Best Rel L2 | L1 | L2 | div_u | composite | Note |
|-------|--------|-------------|-----|-----|-------|-----------|------|
| Finetune start (5e-6) | 82 | 0.127 | 0.121 | 0.169 | 0.300 | **30.10** | saved best (composite) |
| LR scheduler bug | 83–87 | 0.153–0.165 | — | ~0.19 | 0.37+ | 37–42 | cosine restored -> LR ~1e-4 |
| Recovery + long Carreau tail | 88–118 | 0.122 -> 0.090 | — | 0.16 -> 0.12 | 0.32 -> 0.24 | 32 -> 24 | steady gain at ~1e-4 LR |
| **Best saved** | **119** | **0.087** | **0.083** | **0.120** | **0.233** | **23.37** | **beats April Rel L2** |
| End | 121 | 0.089 | 0.087 | 0.120 | 0.245 | 24.54 | slight regression |

**Lessons:** (1) LBFGS tail still harmful (production ep 85–99). (2) Finetune ep 83+ LR jump (pre-`cf0308c`) caused one bad week of val; extended training afterward still helped. (3) Constant 5e-6 finetune with the scheduler fix is the clean recipe going forward; this run's ep 119 weights are the current repo best.

**Promote for biochem/viz:**

```powershell
Copy-Item outputs\kinematics\production_allfix\kinematics_best.pth outputs\kinematics\kinematics_best.pth -Force
```

(`kinematics_best.pth` under `production_allfix/` should reflect **ep 119** if finetune completed.)

**Next lever:** phase 3 clinical anchors from finetune best — `go_kinematics_clinical_anchor_finetune.ps1` + dual promotion gates.

### Clinical anchor finetune (2026-06-05, `go_kinematics_precision_long.ps1` phase 3)

Resume from synthetic-polish `production_allfix/kinematics_best.pth` at **epoch 167**. Run diary: `20260605T194538Z`. Data: **GRAPH_CAP=120** synthetic + **11** patient kine anchors; holdout **`patient007`** val-only; **50** clinical epochs (167–216).

| Phase | Epoch | Global Rel L2 | L0 / L1 / L2 | div_u | patient p007 | synth val | synth L2 val | composite | Note |
|-------|-------|---------------|--------------|-------|----------------|-----------|--------------|-----------|------|
| Cold start (clinical mix) | 167 | 0.175 | 0.183 / 0.182 / 0.161 | 0.387 | 0.164 | 0.176 | 0.161 | 38.90 | momentum reset + capped corpus shock |
| Recovery | 168 | 0.124 | 0.113 / 0.120 / 0.136 | 0.348 | 0.149 | 0.122 | 0.133 | 34.92 | |
| Patient spike (gate fail) | 171 | 0.150 | — / — / 0.179 | 0.354 | **0.275** | 0.141 | 0.155 | 35.50 | dual gates blocked (patient) |
| Patient best (manual pick) | **176** | 0.115 | 0.097 / 0.100 / 0.145 | 0.319 | **0.129** | 0.114 | 0.150 | 31.97 | gates OK; composite blocked save |
| Patient best (manual pick) | **214** | 0.103 | 0.083 / 0.079 / 0.142 | 0.298 | **0.128** | 0.101 | 0.145 | 29.85 | **best p007** in run |
| Synth-val best | **216** | **0.101** | 0.073 / 0.080 / 0.145 | 0.293 | 0.150 | **0.098** | 0.144 | 29.42 | end epoch; latest copied to best |
| Prior (no clinical) | 119 | **0.087** | 0.083 / — / 0.120 | 0.233 | ~0.191 | — | — | 23.37 | synthetic-only reference |

**Outcome:** Training **finished** at ep 216. **`[kin] WARN no Carreau best saved`** — no epoch passed **dual gates + composite &lt; resumed best** (`best_val_composite_loss` ~23.4 from ep 119; clinical val composite ~29–31 because `div_u` ~0.29–0.31). Latest weights copied to `kinematics_best.pth`, **not** a gated optimum.

**Gate checklist (ep 214 / 216 vs defaults):**

| Gate | Limit | ep 214 | ep 216 |
|------|-------|--------|--------|
| patient holdout rel_L2 | ≤ 0.25 | **0.128 Pass** | 0.150 Pass |
| synthetic val rel_L2 | ≤ 0.20 | **0.101 Pass** | **0.098 Pass** |
| synthetic L2 val rel_L2 | ≤ 0.22 | **0.145 Pass** | **0.144 Pass** |

**Readout:** Clinical FT **helped patient007** (~0.191 → **~0.128–0.15**); **hurt** full-corpus synthetic rel_L2 (~0.087 → ~0.10) and **div_u** (~0.23 → ~0.29). Plateau ep **198–216** (global rel_L2 ~0.102–0.108; patient noisy 0.13–0.17). **Stop training**; manually promote **ep 214** (best p007) or **ep 216** (best synth val) — do not assume auto `kinematics_best.pth` is optimal.

**Manual promote (Comsol):**

```powershell
# After copying ep-214 weights from kinematics_validation.jsonl / diary checkpoint if saved:
python scripts/check_kinematics_promotion_gates.py --checkpoint outputs\kinematics\clinical_anchor_finetune\kinematics_best.pth
powershell -File .\scripts\promote_kinematics_checkpoint.ps1 -Checkpoint outputs\kinematics\clinical_anchor_finetune\kinematics_best.pth
```

If only `kinematics_ckpt_latest.pth` exists (ep 216), prefer it for synth gates; for **biochem on patient007**, re-run val on archived ep-176/214 weights if available, else accept ep 216 (p007 0.150).

## Stage-A default training loop (3 phases)

**One command** runs foundation -> synthetic polish -> clinical geometry finetune -> gated promote:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_kinematics_production_allfix.ps1"
```

| Phase | What | Data |
|-------|------|------|
| **1 Foundation** | 3000 synthetic graphs, 100 ep, Adam-only | `graphs_kinematics/` |
| **2 Synthetic polish** | +40 ep ContinuityFocus finetune | same synthetic corpus |
| **3 Clinical geometry** | +25 ep patient-anchor finetune | `graphs_kinematics_anchors/carreau/patient*.pt` + synthetic cap |

Phase 3 adapts the model to **geometries this deployment will see** (patient vessels). Holdout stems (default `patient007`) stay val-only; promotion requires patient + synthetic + synthetic-L2 gates. If no `patient*.pt` exist, phase 3 is skipped with a warning and phase-2 best is copied to global `kinematics_best.pth`.

**Opt out flags** (on `go_kinematics_production_allfix.ps1`):

| Flag | Effect |
|------|--------|
| `-FoundationOnly` | Phase 1 only (legacy) |
| `-SkipSyntheticPolish` | Skip phase 2 |
| `-SkipClinicalAnchors` | Skip phase 3 |
| `-SkipPromote` | Train phase 3 but do not copy to global best |
| `-RequireClinical` | Fail if phase 3 cannot run (no patient graphs) |
| `-Holdout patient007,patient003` | Val-only patient stems for phase 3 |
| `-NoContinuityFocus` | Phase 2 without BC bump |

**Resume mid-ladder** (phase 1 already done):

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_kinematics_stage_a_ladder.ps1" -SkipFoundation -Holdout patient007
```

Orchestrator only: `go_kinematics_stage_a_ladder.ps1` (same flags + `-Fresh` for phase 1).

**Clinical phase details**

- Data: `KINEMATICS_INCLUDE_PATIENT_ANCHORS=1` merges `patient*.pt` into Carreau load; `KINEMATICS_GRAPH_CAP` keeps ~80–120 synthetic graphs for regularization.
- Val (dual holdout): patient stems in `KINEMATICS_VAL_HOLDOUT_PATIENT_STEMS` (default `patient007`) are **val-only**; synthetic holdout uses `KINEMATICS_SYNTHETIC_VAL_RATIO` / `SYNTHETIC_VAL_MIN` / `SYNTHETIC_VAL_MIN_L2` (L2 floor) so promotion catches synthetic drift.
- Sampling: `KINEMATICS_CLINICAL_ANCHOR_BOOST=10` on train steps.
- Best ckpt: `KINEMATICS_DUAL_PROMOTION_GATES=1` — save only when patient + synthetic + synthetic-L2 gates pass and composite improves.
- Writes to `outputs/kinematics/clinical_anchor_finetune/` (does not overwrite global best until promoted).

**Promotion gates** (before `Copy-Item` to `outputs/kinematics/kinematics_best.pth`):

```powershell
python scripts/check_kinematics_promotion_gates.py --checkpoint outputs/kinematics/clinical_anchor_finetune/kinematics_best.pth
powershell -File .\scripts\promote_kinematics_checkpoint.ps1 -Checkpoint outputs\kinematics\clinical_anchor_finetune\kinematics_best.pth
```

Default gates: holdout patient **rel_L2 <= 0.25**, synthetic val **rel_L2 <= 0.20**, synthetic **L2** val **rel_L2 <= 0.22** (automatic at end of default loop).

## RGP-DEQ (`RGP_DEQ`) constructor (must match for biochem load)

| Field | Value |
|-------|-------|
| `latent_dim` | 256 |
| `num_fourier_freqs` | 16 |
| `use_siren_decoder` | true |
| `use_hard_bcs` | true |
| `use_width_priors` | true |
| `max_iters` | 25 |
| `fourier_base` | 2.0 |
| `activation_fn` | silu |

Code: `snapshot_gino_deq_model_config` / `resolve_gino_deq_ctor_kwargs` in [src/architecture/kinematics_model_config.py](../src/architecture/kinematics_model_config.py).

## Geometry curriculum (L0 / L1 / L2)

Stage-A training supports **geometry-level weighted sampling** and **stratified validation** (default **on**).

| Phase | Epochs (default) | Sampling intent |
|-------|------------------|-----------------|
| `l0l1_only` | Stage 1, first **6** epochs (`--l0l1-only-epochs`) | **Train pool = L0+L1 only** (L2 held out of training; still in val) |
| `foundation` | Rest of stage 1 | 45% L0, 45% L1, 10% L2 — introduce L2 |
| `ramp` | Stage 2 (40–59) | Blend → 30/30/40 |
| `l2_heavy` | Stage 3 (60+) | 15/15/70 — thrombus-target geometry |

- **Foundation train**: `powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\go_kinematics_foundation.ps1 -Fresh`
- **L2 finetune** (after foundation ckpt): `.\scripts\go_kinematics_l2_finetune.ps1`
- **Backfill** `geometry_level` on existing `.pt` (no COMSOL): `python -m src.data_gen.backfill_kinematics_geometry_level`
- Disable: `--no-geometry-curriculum`

**Data:** Mixed cohort needs L0+L1 meshes (`pipeline_kinematics --mixed-levels`). L2-only disks cannot run `foundation`; use finetune-only or regen mixed vessels.

## Agents / biochem

`train_biochem_corrector.py` reads Stage-A shape from, in order: `model_config` inside `kinematics_best.pth` (if any), then **`data/reference/kinematics_best_20260426T184600Z.json`**, then tensor-shape inference. For this project, the reference JSON is enough—you do not need LadHyX weights in `outputs/kinematics/`.
