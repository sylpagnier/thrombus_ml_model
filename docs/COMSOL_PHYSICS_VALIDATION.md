# COMSOL phase-2 physics validation (definitive)

**Status: VALIDATED.** Every reaction law, threshold, and scale in the COMSOL
phase-2 thrombosis model has been reconstructed from ground-truth exports and
matches the repo's `BiochemConfig` **to machine precision** — once the unit
system is correctly identified.

- Source data: `data/reference/comsol_calibration/patient007_calibration_{wall,domain}.txt`
  (COMSOL 6.4, `phase2_nowound_007.mph`, exported 2026-06-18).
- Validator: `scripts/validate_comsol_calibration.py`
  → `outputs/reports/comsol_validation/patient007_validation.json`
- Model assumptions audited: `src/config.py` (`BiochemConfig`),
  `src/core_physics/biochem_physics_kernels.py` (`biochem_wall_residual`).
- Canonical law (single source of truth): `src/core_physics/comsol_surface_deposition.py`,
  pinned by `src/tests/test_comsol_wall_deposition_calibration.py` (`Da=1e-4` vs exports).
- **Strategy that follows from this validation:** [docs/SPECIES_LEARNING_STRATEGY.md](SPECIES_LEARNING_STRATEGY.md)
  (Mat-centric gray-box; fibrin dropped; learn AP + Mat autocatalysis).

The exports are COMSOL wide-format spreadsheets:

| File | Nodes | Times | Per-step expressions | Contents |
|---|---|---|---|---|
| `..._wall.txt` | 876 (wall) | 201 (t=0…30000 s, dt=150 s) | 41 | flow, surface species M/Mas/Mat, `d(.,t)`, **exported reaction rates `J0_*`**, gates, bulk species at wall |
| `..._domain.txt` | 51240 (full mesh) | 4 (0/6000/18000/30000 s) | 28 | flow, viscosity `mu1(Mat)`/`mu2(fi)`, bulk species (no surface species) |

---

## 1. THE key finding: COMSOL runs in a physiological/CGS unit system, **not SI**

This single fact is the root cause of every unit struggle in the R0/R1 checks.
The `.mph` is internally unit-consistent but uses cm / µM / poise, while the
repo's `BiochemConfig` stores everything in SI (m / mol·m⁻³ / Pa·s).

| Quantity | COMSOL export unit | `BiochemConfig` (SI) | export = SI × |
|---|---|---|---|
| length (x, y) | **cm** | m | 1e-2 |
| bulk platelets (`rp`, `ap`) | **plt/cm³** | plt/m³ (`c_RP0=2.5e14`) | 1e-6 |
| bulk solutes (`PT`,`fg`,`at`,`th`,`apr`,`aps`) | **µM** | mol/m³ (`c_pT0=1.2e-3`) | 1e3 |
| surface platelets (`M`,`Mas`,`Mat`) | **plt/cm²** | plt/m² (`Minf=7e10`) | 1e-4 |
| dynamic viscosity (`spf.mu`) | **poise** (g·cm⁻¹·s⁻¹) | Pa·s | 10 |
| shear rate (`spf.sr`) | 1/s | 1/s | 1 |
| shear gradient (`d(spf.sr,x)`) | **1/(s·cm)** | 1/(s·m) | 1e-2 |

Evidence (medians at t=0): `rp`=2.5e8 plt/cm³ (cfg `c_RP0`=2.5e14 plt/m³, ratio
1e-6); `PT`=1.2 µM (cfg `c_pT0`=1.2e-3 mol/m³); `fg`=7 µM (cfg 7e-3 mol/m³);
`at`=2.84 µM (cfg 2.84e-3). Geometry spans ±7 (cm) for a ~10 cm vessel.

**The constants in `BiochemConfig` that are tagged "SI" but were copied straight
from the COMSOL parameter list are actually CGS-calibrated** (`surface_damkohler`,
`sgt`, `L_char`, `gamma_m`, `viscosity_mat_crit`, etc. — see §3). They only
reproduce COMSOL when the species/gradients are also in CGS.

---

## 2. Surface platelet-deposition law — reconstructed EXACTLY

COMSOL surface matrix source (header column `J0_Mat`):

```
J0_Mat = Da * (  if(d(spf.sr,x) < sgt, (L/gamma_m)*|d(spf.sr,x)| * common, 0)
               + if(spf.sr < lss,                              common, 0) ) * step2t(t)

common = Sat(M)*k_rs*rp + Sat(M)*k_as*ap + (Mas/M_inf)*k_aa*ap
```

Rebuilding `J0_Mat` from the exported **inputs** (`Sat(M)`, `rp`, `ap`, `Mas`,
`|d(spf.sr,x)|`, `step2t`, and COMSOL's own exported boolean gates) in
**fully consistent CGS units** recovers `Da` per gate branch:

| subset | active pts | recovered Da | CV |
|---|---|---|---|
| all | 44 422 | **1.0e-4** | 3.6e-16 |
| low-shear gate only | 33 801 | 1.0e-4 | 2.4e-16 |
| separation gate only | 4 713 | 1.0e-4 | 1.7e-16 |
| both gates | 5 908 | 1.0e-4 | 1.6e-16 |

`pearson(reconstructed, exported J0_Mat) = 1.000000`. Recovered
`Da = 1.0e-4` is **exactly** `cfg.surface_damkohler`. The law form, every rate
constant (`k_rs`, `k_as`, `k_aa`), `L_char`, `gamma_m`, `Minf` and the `Da`
prefactor are all correct (in CGS).

`Sat(M)` is COMSOL's availability function; the repo approximates it as
`1 - (M+Mas+Mat)/Minf`. The exported `Sat(M)` column lets us use the exact form
going forward.

---

## 3. Thrombin generation & gelation thresholds — exact

**Thrombin source** (`J0_th = beta*phi_at*Mat*PT*step2t`): recovered
`beta*phi_at = 3.362e-11` with CV = 2e-16 (perfectly constant, no gate). The
ratio to `cfg.beta*cfg.phi_at = 3.362e-17` is exactly **1e6** — a pure unit
factor (Mat plt/cm² × PT µM vs plt/m² × mol/m³). Thrombin is generated in
**direct proportion to the platelet matrix `Mat`** → autocatalytic loop
(more Mat → more thrombin → more activation → more deposition).

**Platelet gelation** (`mu1(Mat)`): steps 1 → 80 with the transition midpoint
(`mu1≈40`) at **Mat = 2.00e7 plt/cm² = `cfg.viscosity_mat_crit` exactly**
(transition band 1.78e7–2.21e7). Ceiling 80 = `cfg.mu_ratio_max`. Confirmed.

**Fibrin gelation** (`mu2(fi)`): `fi` peaks at 0.016 µM, far below
`cfg.viscosity_fi_crit = 0.6 µM`, so **`mu2(fi) ≡ 0` everywhere, at all times**.
Fibrin never contributes to viscosity/clotting in this model. This independently
confirms the R0 conclusion (fibrin physically negligible) and validates the
direction of the FI-gelation unit fix (FI should essentially never trip the
gate; the previous ~1000× too-lenient threshold made it trip spuriously).

---

## 4. Clot definition & flow

- Effective viscosity `spf.mu` = Carreau(shear) **×** `mu1(Mat)` × `mu2(fi)`
  (multiplicative step factors). Domain `spf.mu` ∈ [0.043, 42] poise; median
  0.085 poise (~0.0085 Pa·s, blood-like). The clot is the **localized viscosity
  jump** where `mu1(Mat)` switches to 80.
- `mu_eff` growth over the run: median ≈ 0 (bulk unchanged), max ≈ 42 poise at
  the clot. This is exactly the canonical GT clot label
  `mu_eff(t) − mu_eff(t0)` used by `t0_mu_physics.gt_clot_phi_at_time`.
- Surface pools: `M`, `Mas` saturate near `Minf` (~7e6 plt/cm²); `Mat` grows
  unbounded to ~3.9e8 plt/cm² (≈56× Minf) — `Mat` is the accumulating matrix,
  not a monolayer.

---

## 5. What actually drives deposition (mechanism)

Gate activation over the wall × time grid:

- `(spf.sr < lss)` (low-shear **stagnation**) active **22.6 %**.
- `(d(spf.sr,x) < sgt)` (shear-gradient **separation**) active **6.1 %**.
- Among nodes where `Mat` actually grows: **low-shear gate on 79.7 %**,
  separation gate on 21.2 %.

**Deposition is dominated by the low-shear stagnation gate**, not the
shear-gradient/separation gate. Final `Mat` growth localizes to ~39 % of wall
nodes (recirculation/stagnation zones).

This explains the **R1 oracle failure**: it relied on a graph-WLS reconstruction
of `d(spf.sr,x)` to fire the separation gate (under-resolved → rarely tripped),
while the real driver is the much cheaper, more robust low-shear gate.

### RESOLVED: why `d(Mat,t) ≠ J0_Mat` — Mat grows by autocatalytic aggregation
Exported `d(Mat,t)` is **not** equal to `J0_Mat`: `d(Mat,t) ≈ 146 × J0_Mat`
(pearson 0.86), and the ratio is **not constant** — it rises from ~40 early to
~150 late. Root-caused via integration + term decomposition:

- **Integration check.** `∫d(Mat,t)dt = 3.18e10` matches the actual matrix gain
  `ΔMat = 3.20e10` (ratio 1.008) — so `d(Mat,t)` is the true derivative. But
  `∫J0_Mat dt = 2.40e8` is only **1/133** of it. So `J0_Mat` is a *minor*
  contribution, not the dominant Mat source.
- **Decomposition** (`d(Mat,t) ~ 16.9·J0_M + k_aa_eff·T3`, R²=0.88, where
  `T3 = gate·(Mas/Minf)·AP·step2t` is the autocatalytic platelet-aggregation
  term): the **autocatalytic term `(Mas/Minf)·k_aa·AP` explains ~90 % of all Mat
  growth**; fresh wall deposition (`Sat·(k_rs·RP+k_as·AP)` = `J0_M`) is only ~7 %.
- The recovered autocatalytic coefficient `k_aa_eff ≈ 6.5e-4 cm/s` is **~143×
  larger** than the value baked into the `J0_Mat` analytic column
  (`Da·k_aa = 4.5e-6`). That ~143× is exactly the `d(Mat,t)/J0_Mat` ratio.

**Physical meaning.** The thrombus grows mainly by **platelet–platelet
aggregation onto already-deposited activated platelets** (a snowball / positive
feedback that scales with `Mas`), gated by **low-shear stagnation** — not by
fresh recruitment of platelets from the bulk at the wall. Early on `Mas` is
small so deposition (`J0_M`) dominates and the ratio is ~40; once `Mas`
saturates near `Minf` the autocatalytic term takes over and the ratio settles
~150. (`M ≡ Mas` exactly in this model, and `R_M = R_Mas`, so
`d(M,t)/J0_M = d(Mas,t)/J0_Mas`.)

**Consequence.** The `J0_*` columns are the COMSOL *deposition-flux probe*, not
the full surface ODE RHS. An **isolated** oracle fed only `J0_Mat` undershoots
Mat by ~100–130× — exactly what the R1 oracle did. A faithful model must include
the autocatalytic `Mas`-driven aggregation term (with its true ~140× coefficient)
and let it run self-reinforcing under the low-shear gate.

---

## 6. Implications & risks for the repo

1. **Unit system mismatch is real and must be handled explicitly.**
   `biochem_wall_residual` applies CGS-calibrated constants
   (`surface_damkohler=1e-4`, `sgt`, `L_char`, `gamma_m`, `viscosity_mat_crit`,
   `viscosity_fi_crit`) to **SI-decoded** species and **SI** shear gradients.
   Unless every term is converted consistently to one system, the effective
   `Da`/thresholds are off by powers of ten. The validated path is: **work in
   CGS internally (cm, µM, plt/cm², plt/cm³, 1/(s·cm)) or convert COMSOL
   constants to SI** — do not mix.
2. **Fibrin should be dropped from the clot trigger.** `mu2(fi) ≡ 0`; FI carries
   no clot signal. Any FI-gelation path is dead weight / a spurious-trigger risk.
3. **Center the deposition gate on low-shear stagnation**, not the WLS
   shear-gradient (which graph operators under-resolve). The separation gate is a
   minor (~20 %) contributor.
4. **The clot is Mat-driven.** `Mat` (and its autocatalytic thrombin feedback)
   is the single mechanistic predictor of the viscosity jump; `mu1(Mat)` is a
   hard step at `Mat=2e7 plt/cm²`.

---

## 7. Re-evaluation of the species / coagulation-cascade modelling approach

With the physics now fully known, the earlier A/B species results make sense:

- **Why extra species channels didn't help the GNN.** The clot label depends
  *only* on `Mat`, which is produced by a low-shear-gated platelet-adhesion law
  fed by `rp`/`ap` and modulated by activation agonists (`apr`,`aps`,`th`) and
  thrombin substrate (`PT`). **`FG`/`FI` are causally irrelevant** to the clot
  target, so adding them as channels injects noise → consistent with "fi_mat
  baseline is hard to beat" and FG/FI combos hurting.
- **The minimal causal feature set** for the clot is: geometry-derived
  **low-shear/stagnation indicator**, **platelet availability** (`Sat(M)` ~
  `rp`/`ap` supply), and the **thrombin/activation amplifier** (`th`,`apr`,`aps`,
  `PT`). This is a handful of physically-motivated inputs, not the full 9-species
  bulk vector.
- **Gray-box direction.** A faithful mechanistic/gray-box model should: (a) run
  in CGS (or convert correctly); (b) use the low-shear gate as the primary
  deposition trigger; (c) model the coupled `Mat ↔ thrombin` autocatalysis
  rather than an isolated surface ODE (the §5 nuance); (d) ignore fibrin. The
  deposition source law itself needs **no learned parameters** — it reproduces
  COMSOL exactly. What is worth *learning* is the closure that the isolated
  oracle misses: the effective deposition→matrix scaling and the
  stagnation/availability gating from predicted (not GT) flow.

### Suggested next steps
1. Decide the canonical internal unit system (recommend CGS to match COMSOL) and
   make `biochem_wall_residual` / gelation readouts unit-consistent end to end;
   add a regression test that reconstructs `J0_Mat`/`J0_th` from a graph and
   checks `Da=1e-4`, `beta*phi_at` against these exports.
2. Re-scope the deploy clot trigger to {low-shear stagnation, platelet
   availability, thrombin/activation}, drop fibrin.
3. Re-run the species A/B only over the causal set to confirm the minimal-feature
   hypothesis.
