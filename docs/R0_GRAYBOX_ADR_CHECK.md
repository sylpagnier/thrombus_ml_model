# R0 oracle check - gray-box ADR for species -> clots

**Question.** Before building a gray-box (predict upstream species + flow, then push them
through the known COMSOL advection-diffusion-reaction operator to get FI/Mat -> clots),
verify with an oracle: fed **ground-truth** species + flow + the known COMSOL rate
constants, does the in-repo ADR operator reproduce the GT fibrin/matrix fields that drive
clots?

**Runner.** `python scripts/r0_oracle_adr_consistency.py`
-> `outputs/biochem/biochem_gnn/r0_adr_consistency/r0_adr_consistency.json`

## What was verified

- **Kinetics + units are correct.** The kernel fibrin source
  `R_FI = kfi*T*FG/(kmfi+FG)` (`BiochemKinetics.compute_fibrin_kinetics`) matches the
  COMSOL `reac1` analytic column in `src/tests/fixtures/oracle_kinetics.csv`
  (`phase2_nowound_001.mph`): `th`, `fg` in uM, `R_fi` in uM/s. No units bug.
- Channel map confirmed from `y_channel_names`: species block `y[:,:,4:16]` =
  `[RP,AP,APR,APS,PT,T,AT,FG,FI,M,Mas,Mat]`; FI=idx8, Mat=idx11, T=idx5, FG=idx7.

## Decisive finding: fibrin is NOT the clot driver - the platelet matrix `Mat` is

Per-patient, final frame (ceiling region), physical units:

| patient | FI max (uM) | gels @0.6uM? | FG depletion | GT clot nodes | clot&Mat | clot&FI | driver |
|---------|-------------|--------------|--------------|---------------|----------|---------|--------|
| 001 | 0.0007 | never | ~0% | 1764 | 295 | 0 | Mat |
| 002 | 0.0012 | never | ~0% | 4098 | 125 | 0 | Mat |
| 003 | 0.0013 | never | ~0% | 3624 | 161 | 0 | Mat |
| 004 | 0.0009 | never | ~0% | 2992 | 294 | 0 | Mat |
| 006 | 0.0038 | never | ~0% | 3235 | 104 | 0 | Mat |
| 007 | 0.0130 | never | ~0% | 3892 | 459 | 0 | Mat |

- **Fibrin never gels.** GT FI peaks at 0.0007-0.013 uM across all patients, ~50-900x
  below the COMSOL fibrin gelation threshold (`mu2` step at 0.6 uM). Fibrinogen depletes
  <0.2%. Fibrin overlaps zero GT clot nodes at the physical threshold.
- **Matrix drives clots.** `Mat` (wall platelet deposition) reaches 1e11-1e12 >> the
  `mu1` threshold (2e7) and is the only field overlapping the GT clots.
- Bulk fibrin that *is* produced is almost entirely advected downstream and exits (at an
  interior oracle point, net FI ~1% of cumulative production), so it never accumulates to
  clot-forming levels.

## Implications for the gray-box

1. **Do not route the gray-box through bulk fibrin.** The fibrin ADR is correct but
   physically negligible for clots in this dataset.
2. **The real coupling for clots is the WALL surface-deposition system** that produces
   `Mat` (COMSOL `tds2` surface reactions `srf1`): adhesion rates `k_rs/k_as/k_aa`, gated
   by low-shear `lss` and shear-gradient separation `sgt`, with platelet availability
   `Sat(M)=1-M/M_inf`. Bulk species (RP/AP + agonists APR/APS/thrombin) matter only as
   upstream drivers of wall adhesion, not via fibrin. This explains why simply adding bulk
   species channels to the GNN did not help: they do not feed the clot trigger.
3. Next rung (R1) should be the oracle for the **wall deposition law**: feed GT wall RP/AP
   + shear + shear-gradient through the surface-adhesion kernel and check it reproduces GT
   `Mat` (and thus clots).

## Unit / scaling audit (R0b) - `scripts/r0_unit_diagnostics.py`

COMSOL defines `mu = Carreau(shear) * (mu1(Mat) + mu2(FI))`; canonical GT clot =
growth-only `relu(mu_eff(t) - mu_eff(t0)) >= 0.055 Pa*s` (`gt_clot_phi_at_time`). We audited
every scale on the species->viscosity->clot path using the production decode functions.
-> `outputs/biochem/biochem_gnn/r0_adr_consistency/r0_unit_audit.json`

Config (verified): `mu_ref=3.5e-3 Pa*s`, `mu_ratio_max=80`, `Minf=7e10`, `surface_scale=1e4`,
`scale_FI=7000` (working = uM*1000), `viscosity_mat_crit=2e7`, `viscosity_fi_crit=0.6`.
Mat gelation decode = `expm1(nd)*Minf` (NOT `*surface_scale`).

| patient | GT clot | F1 Mat-only | F1 deploy(+FI) | FI gel deploy | FI gel physical | FI max (uM) | Mat@clot p50 | Mat@nonclot p95 |
|--|--|--|--|--|--|--|--|--|
| 001 | 222 | 0.993 | 0.993 | 7   | 0 | 0.0007 | 2.5e7 | 4.0e6 |
| 002 | 56  | 0.954 | 0.894 | 23  | 0 | 0.0012 | 3.4e7 | 2.1e6 |
| 003 | 39  | 0.781 | 0.485 | 39  | 0 | 0.0013 | 2.1e7 | 2.9e6 |
| 004 | 87  | 0.899 | 0.768 | 27  | 0 | 0.0009 | 2.2e7 | 1.4e7 |
| 006 | 44  | 0.966 | 0.808 | 49  | 0 | 0.0038 | 5.0e7 | 1.1e4 |
| 007 | 284 | 0.977 | 0.921 | 183 | 0 | 0.0130 | 6.9e7 | 7.2e6 |

**Mat units are CORRECT.** `viscosity_mat_crit = 2e7` sits cleanly between Mat-at-clot
(median 2.1e7-6.9e7) and Mat-at-nonclot (95th pct <= 1.4e7). Mat alone reproduces the
canonical GT clot at **F1 0.78-0.99** (mean ~0.93). No Mat scaling error.

**FI threshold IS a (confirmed, harmful) units bug.** `viscosity_fi_crit = 0.6` is compared
to `fi_si` in **working units** (uM*1000), so the effective fibrin gelation threshold is
0.6 nM = 6e-4 uM - ~1000x too lenient. Physically FI never reaches 0.6 uM (max 0.0007-0.013),
so `mu2` should trigger on **0** nodes; instead the deploy decode spuriously gels 7-183
ceiling nodes and **lowers clot F1 in every patient** (deploy F1 < Mat-only F1, e.g.
p003 0.781->0.485, p007 0.977->0.921). The earlier "clot intersects FI" overlap was entirely
this bug. **Fix:** compare fibrin in uM (divide `fi_si` by 1000) or set `viscosity_fi_crit`
to 600 (working). Either makes `mu2` inert -> clot F1 returns to the Mat-only level and the
trigger matches COMSOL (where `mu2(FI)` never fires for these cases).

**Net:** the only scaling error on the clot path is the FI gelation threshold (cosmetic-to-
harmful, not strategy-changing). Mat decode/threshold and the bulk kinetics/units are all
verified correct. Clots are Mat-driven; fibrin is physically inert.

## Fix implemented + verified (R0c)

Added a canonical `fi_si_for_gelation_from_log1p(fi_log1p, bio_cfg)` (mirrors
`mat_si_for_gelation_from_log1p`) that decodes FI to **uM** (`working * 1e3 / bulk_scale`) so
it is unit-consistent with `viscosity_fi_crit = 0.6 uM`. Routed every gelation site through it:

- `clot_phi_simple.species_log1p_nd_to_si` now overrides FI (idx 8) to uM, like Mat (idx 11).
  This fixes `_resolve_gelation_legs` and `species_gelation_readout.differentiable_clot_phi_from_species12`.
- `species_viscosity_calibration.differentiable_clot_phi_from_full_y` uses the helper.
- `biochem_physics_kernels.compute_dual_viscosity_penalty` converts FI working -> uM before
  the soft step.
- The learned GNODE-teacher viz head (`visualize_pipeline` `mu2_sigmoid`) is a separate trained
  module (own threshold in weights) and is intentionally left untouched.

**Verification** (`scripts/r0_unit_diagnostics.py`, production readout end-to-end): FI now gels
**0** nodes in every patient and the production clot F1 returns to the Mat-only level, recovering
all F1 lost to the bug:

| patient | F1 deploy (buggy) | F1 production (fixed) | FI gel (physical) |
|--|--|--|--|
| 001 | 0.993 | 0.993 | 0 |
| 002 | 0.894 | 0.954 | 0 |
| 003 | 0.485 | 0.781 | 0 |
| 004 | 0.768 | 0.899 | 0 |
| 006 | 0.808 | 0.966 | 0 |
| 007 | 0.921 | 0.977 | 0 |

Regression check: new unit tests in `src/tests/test_mat_gelation_scale.py` (FI decode -> uM,
sub-/super-threshold gelation) pass; the orphaned dead-API tests in that file (deleted COMSOL
debug-sidecar subsystem) were dropped. Kinetics/transport suites pass. The 3 pre-existing
`test_clot_phi_neighbor_mask` failures are data-threshold drift unrelated to this fix (confirmed
identical with the fix stashed).
