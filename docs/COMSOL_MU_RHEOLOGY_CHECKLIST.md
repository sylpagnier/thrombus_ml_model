# COMSOL mu rheology checklist (phase2_nowound mph)

Use this when debugging **GT channel 3** (`mu_eff` / `spf.mu`) vs Python
`carreau_mu_si_from_uv` or T0 physics triggers. Model reference:
`comsol_models/phase2_nowound_XXX.mph` (e.g. `phase2_nowound_007.mph`).

---

## Quick map (where to click)

| What you need | COMSOL path | Variable / expression |
|---------------|-------------|------------------------|
| **Python export label** | Results > Export > `sol_data` > Expressions | **`spf.mu`** (Dynamic viscosity) |
| **Solver rheology** | Laminar Flow (spf) > Fluid Properties 1 | Carreau; mu0 / mu_inf below |
| **Gelation-only plot** (not export) | Results > Viscosity > Surface 1 | `mu_b*(mu2(FI)+mu1(Mat))` |
| **Material Carreau group** | Materials > Material 1 > Carreau model | `mu0`, `mu_inf`, `lam_car`, `n_car` |
| **Material basic mu** | Materials > Material 1 > Basic | `mu = mu_b*(mu1+mu2)` (gelation baseline) |
| **Blood baseline** | Global Definitions > Parameters | `mu_b` (typically `3.5e-2` g/(cm*s) = **0.035 Poise**) |
| **Gelation steps** | Definitions > Variables (or Functions) | `mu1(Mat)`, `mu2(FI)`, `mu_ratio_max` |
| **Shear diagnostic plot** | Results > `shear rate` / `shea_rate` | compare to Python gamma_dot |

**Do not confuse** the **Viscosity** result plot (`mu_b*(mu1+mu2)`) with the **export**
(`spf.mu`). They are different fields.

---

## COMSOL rheology (canonical for T0)

From **Laminar Flow (spf) > Fluid Properties 1** (Carreau, inelastic non-Newtonian):

```text
mu0(M, FI, Mat)  = 0.56 * (mu2(FI) + mu1(Mat))     [Poise in CGS mph]
mu_inf(M, FI, Mat) = mu_b * (mu2(FI) + mu1(Mat))   [Poise]
lam = 3.313 s
n   = 0.3568
```

**Apparent viscosity** used in momentum: **`spf.mu`** = Carreau(mu0, mu_inf, lam, n, gamma_dot).

At bulk (mu1=1, mu2=0, M gelation factor = 1):

| Quantity | CGS (Poise) | SI (Pa*s) after x0.1 |
|----------|-------------|----------------------|
| mu_b | 0.035 | **0.0035** |
| mu0 | 0.56 | **0.056** |
| mu_inf | 0.035 | **0.0035** |
| spf.mu (typical shear) | ~0.084 | **~0.0084** |

`spf.mu` sits **between** mu_inf and mu0 — it is shear-thinned Carreau, not mu_b alone.

---

## Python pipeline (what we store)

1. **Export:** `Results > Export > sol_data` writes `spf.mu` into
   `data/processed/cfd_results_biochem/<stem>.txt`.
2. **Extract:** `extract_biochem_comsol_data.py` applies
   `mu_si = raw * cgs_mu_to_pa_s` (`0.1`, Poise -> Pa*s).
3. **Graph channel 3:** `mu_nd = mu_si / mu_viscosity_nd_scale` (scale 0.0035 Pa*s).

Repo fallback (only if mph Export nodes fail):

```text
BIOCHEM_COMSOL_USE_MPH_EXPORTS=0  -> Interp uses mu_b*(mu1(Mat)+mu2(FI))  [gelation only]
```

Live anchors use **`spf.mu`** via mph Export nodes (`BIOCHEM_COMSOL_USE_MPH_EXPORTS=1`, default).

---

## mu_ratio_max vs mu_eff (do not conflate)

| Name | Meaning | Typical scale |
|------|---------|---------------|
| `mu_ratio_max` (BiochemConfig) | COMSOL **step ceiling** for mu1/mu2 branches | 80 |
| GT `mu_eff` / `spf.mu` bulk | Shear-thinned dynamic viscosity | ~0.008 Pa*s |
| GT clot growth | Rise above per-node t=0 mu | ~2-3x, not 80x |
| Viscosity **plot** color bar | Gelation product `mu_b*(mu1+mu2)` | 0.035-0.1 Poise |

---

## Python code mismatches (known, 2026-06)

| Issue | Location | Wrong | Target |
|-------|----------|-------|--------|
| Gelation blood constant | `clot_phi_physics_mu_blood_si` comsol mode | `0.035` Pa*s | **`0.0035` Pa*s** (= mu_b Poise) |
| T0 physics formula | `physics_mu_eff_si` default `carreau` | `mu_c * (1+gel)` soft | **Carreau(gel-scaled mu0, mu_inf)** |
| Offline Carreau vs GT | `carreau_mu_si_from_uv` | fixed mu0/mu_inf; interior gamma~0 | gel-scaled limits + better gamma |
| Fallback export expr | `biochem_comsol_auto_export.py` default | gelation only | mph uses `spf.mu` (OK) |

**Not a primary bug:** Poise->Pa*s `x0.1` on export (confirmed on patient007).

---

## Diagnostic commands (repo root)

```powershell
# Export vs Carreau vs mu_b hypotheses
python scripts/diagnose_mu_export_shear.py --anchor patient007

# T0 factorized Carreau x gelation (GT u,v + species)
python scripts/diagnose_t0_carreau_gelation.py --anchor patient007
```

Outputs:

- `outputs/biochem/diagnostics/mu_export_shear_p007.json`
- `outputs/biochem/clot_trigger/t0_carreau_gelation_diag.json`

**Pass criteria for physics fix:**

- Bulk M=1: median `GT / mu_pred` in [0.9, 1.1] using **spf.mu-aligned** formula.
- Pearson r(GT, mu_pred) > 0.5 on wall nodes (interior gamma remains hard).
- T0 LOAO band F1 improves vs current `mu_c*(1+gel)` baseline.

---

## Optional mph verification (manual)

1. Open `phase2_nowound_007.mph` in COMSOL.
2. Confirm `sol_data` sixth expression is **`spf.mu`**.
3. Results > Viscosity: note plot uses **`mu_b*(mu2+mu1)`** — different from export.
4. Note `mu_b` numeric value in Parameters (expect 0.035 Poise).
5. (Optional) See **Optional gamma validation export** below.

---

## Optional gamma validation export

**Do you need `gamma_dot` in production graphs?** No. Deploy physics uses
`max(WLS graph, Poiseuille wall, |u|/width_nd)`; bulk `spf.mu` at M=1 already
matches COMSOL on patient007 without a COMSOL shear channel.

**When to export anyway:** One-time validation on **patient007** (or any single
anchor) to confirm wall shear and tune `CLOT_PHI_PHYSICS_GAMMA_SCALE` if needed.

1. In COMSOL Results > Export > `sol_data`, add a **separate** export node (do
   not change the 16-field biochem layout) OR export a one-off text file with
   columns `x, y, sr @ t=...` using expression **`spf.sr`** (shear-rate
   magnitude used by COMSOL Carreau; [1/s] in SI). Do **not** use `spf.gammat`
   — it is not a standard Laminar Flow built-in and the Description column
   stays empty. Pick via Laminar Flow > Variables > **Shear rate (spf.sr)**.
2. Match nodes to the graph (same mesh / spatial join as `patient007.txt`).
3. Save matched per-node SI shear as::

       data/processed/cfd_results_biochem_diag/patient007_gammat.pt

   with contents ``{"gamma_si": tensor shape [N]}`` (float32, units 1/s).
4. Re-run::

       python scripts/diagnose_t0_physics_baseline.py --anchor patient007

   The JSON will include `pearson_gamma_kinematic_vs_comsol` and
   `pearson_gamma_resolved_vs_comsol`.

T0 field viz (mu, gamma, phi side-by-side)::

    powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_t0_physics_baseline.ps1"

---

## T0 target formula (Star 1a)

Match COMSOL **spf.mu**, not the Viscosity plot expression:

```text
M_gel = mu1(Mat) + mu2(FI)          # COMSOL step functions (hard or soft)
mu0_si = 0.056 * M_gel              # 0.56 Poise * M
mu_inf_si = 0.0035 * M_gel           # mu_b Poise * M
mu_pred = Carreau_Yasuda(gamma_dot; mu0_si, mu_inf_si, lam, n)
```

**Shear rate on graphs:** WLS `G_x`/`G_y` gradients underestimate interior
`gamma_dot` vs COMSOL FEM (~1000x on patient007 bulk). Use
`gamma_dot_nd = max(graph, poiseuille, |u|/width_nd)` where `width_nd` is
hydraulic width from node features (`NodeFeat.WIDTH_ND`). Kinematic
`|u|/width` alone matches bulk `spf.mu` at M=1 (median GT/pred ~1.04 on p007).

Env knobs: `CLOT_PHI_PHYSICS_MU_BASE=comsol_carreau`,
`CLOT_PHI_PHYSICS_GAMMA_MODE=max` (deploy proxy),
`CLOT_PHI_PHYSICS_POISEUILLE_SCALE=0.85` (COMSOL ``spf.sr`` calib on p007),
`CLOT_PHI_PHYSICS_MU_BLOOD_SI=0.0035`, `CLOT_PHI_PHYSICS_MU_RATIO_MAX=4`,
`CLOT_PHI_PHYSICS_SUBTRACT_T0_MU=1`.

Oracle validation: build sidecar via ``scripts/build_comsol_sr_sidecar.py``,
then ``CLOT_PHI_PHYSICS_GAMMA_MODE=comsol_sr`` +
``CLOT_PHI_PHYSICS_COMSOL_SR_ANCHOR=patient007`` reproduces GT ``spf.mu`` exactly.

See [CLOT_TRIGGER_LADDER.md](CLOT_TRIGGER_LADDER.md) Star 1 and T0 sweep scripts.
