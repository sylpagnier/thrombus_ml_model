> **Resolved.** The R1 failure is fully explained by the definitive COMSOL
> validation in [COMSOL_PHYSICS_VALIDATION.md](COMSOL_PHYSICS_VALIDATION.md):
> (1) COMSOL runs in **CGS/µM**, not SI; (2) the **low-shear** gate (not the
> WLS shear-gradient gate) drives ~80% of deposition; (3) **Mat grows ~90% by
> autocatalytic platelet aggregation** `(Mas/Minf)·k_aa·AP` (effective coef ~143×
> the `J0_Mat` analytic column), not by the fresh-deposition `J0_Mat` flux — so
> an isolated oracle fed `J0_Mat` undershoots Mat ~130×. The deposition law form
> + constants still reconstruct COMSOL's `J0_Mat`/`J0_th` to machine precision
> (Da=1e-4 exact) in CGS.

# R1 oracle check - wall platelet-deposition law for `Mat` -> clots

**Question.** R0 proved clots are driven by the wall platelet matrix `Mat` (COMSOL surface
physics `tds2`/`srf1`), not bulk fibrin. `Mat` has D=0, so M/Mas/Mat are *purely local* surface
ODEs (no transport). Oracle: feed **ground-truth** wall species + flow through the known COMSOL
deposition law and check it reproduces the GT `Mat` field (and thus clots).

**Runner.** `python scripts/r1_wall_deposition_oracle.py [--grad streamwise|x] [--sat comsol|kernel]`
-> `outputs/biochem/biochem_gnn/r1_wall_deposition/r1_wall_deposition.json`

## The law (ported verbatim from COMSOL `J0_*`, mirrors `biochem_wall_residual`)

```
dMat/dt = Da * R_Mat            (active for t > step2t = 12 s; here t_end ~ 30000 s, always on)
R_M   = [sep](L/gamma_m)|dgamma/ds| Sat (k_rs*RP + k_as*AP) + [lss] Sat (k_rs*RP + k_as*AP)
R_Mat = R_M + [sep](L/gamma_m)|dgamma/ds|(Mas/Minf) k_aa*AP + [lss](Mas/Minf) k_aa*AP
Sat   = 1 - M/Minf      sep = dgamma/ds < sgt (-7.5e4)      lss = gamma < lss_crit (25 1/s)
```
Constants from `BiochemConfig` (`k_rs/k_as/k_aa`, `L_char`, `gamma_m`, `sgt`, `lss`, `Minf`, `Da`).
Two checks: (1) teacher-forced rate `Da*R_Mat` vs observed `dMat/dt` (scale-free corr); (2)
free-run: integrate M/Mas/Mat from t0 using only GT bulk species + GT velocity, compare to GT.

## Result: the forward oracle FAILS (the law cannot be pushed forward from graph fields)

| patient | rate corr | Mat corr | F1 free-run | F1 free-run (+effective-Da fit) | F1 GT-Mat (upper bound) |
|--|--|--|--|--|--|
| 001 | 0.00 | 0.02 | 0.000 | 0.408 | 0.995 |
| 002 | 0.16 | 0.21 | 0.000 | 0.000 | 0.954 |
| 003 | 0.14 | 0.21 | 0.000 | 0.000 | 0.781 |
| 004 | 0.16 | 0.24 | 0.000 | 0.000 | 0.899 |
| 006 | 0.08 | 0.10 | 0.000 | 0.000 | 0.987 |
| 007 | 0.12 | 0.18 | 0.000 | 0.583 | 0.993 |
| **mean** | **0.11** | **0.16** | **0.00** | **0.17** | **0.93** |

- **Pipeline is correct.** GT `Mat` -> mu1 step reproduces the canonical GT clot at **F1 0.93**
  (the "upper bound" column == R0). Decode/masking/threshold are right.
- **Magnitude is ~100x low and unit-ambiguous.** Free-run `Mat` peaks ~1e6 vs GT ~1e8-1e9. This
  is the flagged CGS->SI `Da` ambiguity (1 scalar). The effective-`Da` least-squares fit removes
  it - but F1 only reaches 0.17, so magnitude is **not** the main problem.
- **The pattern is wrong (scale-free corr ~0.11-0.16).** Even teacher-forced (GT M/Mas/Mat fed
  in) the predicted `dMat/dt` does not match observed. `--grad streamwise` vs `x` is irrelevant.

## Why it fails (probe on patient007)

1. **Low-shear gate is non-selective.** With this slow flow (u_ref ~0.09 m/s) wall shear is
   ~0.5-100 1/s, so `gamma < 25` fires on **344/583 (59%)** of wall nodes. `k_rs*RP+k_as*AP`
   is then nearly uniform (RP ~2.5e14 everywhere) -> near-uniform predicted deposition, while
   GT `Mat` is concentrated. Low-shear membership only weakly predicts growth (corr +0.25).
2. **Separation (shear-gradient) gate is dead.** COMSOL's selective term needs `dgamma/ds < -7.5e4`;
   the graph WLS reconstruction peaks at ~+-65000 (Cartesian) / ~0 (streamwise) and **never trips
   the threshold** (`|dgamma/dx|` corr with growth = -0.09). The second-derivative-sensitive gate
   is under-resolved by graph gradient operators vs COMSOL's FEM `spf.sr`.
3. **Surface adhesion is bidirectionally coupled to bulk consumption.** At the final frame, GT
   `Mat`-growth anti-correlates with the platelet pools (**AP -0.54, RP -0.45**): platelets are
   *consumed* where they deposit. Feeding already-consumed GT RP/AP into the isolated surface ODE
   breaks causality - the deposition cannot be decoupled from the bulk transport it feeds back on.

## What actually localizes `Mat` (drivers of growth, patient007 wall)

| driver | corr with Mat-growth |
|--|--|
| thrombin `T` | **+0.49** |
| `APR` | **+0.46** |
| `APS` | +0.25 |
| low-shear indicator | +0.25 |
| `AP` (pool) | -0.54 (consumed) |
| `RP` (pool) | -0.45 (consumed) |
| `|dgamma/dx|` | -0.09 |

Clot location tracks the **activation chemistry** (thrombin / agonist field) plus a weak low-shear
preference - not the platelet pools and not the (unresolved) shear gradient.

## Implications for the gray-box

1. **A hand-coded mechanistic push-forward of the deposition law does not work** in isolation:
   the rate constant is unit-ambiguous (1 scalar), the selective shear-gradient gate is
   under-resolved by graph operators, and the surface system is coupled to bulk consumption.
2. **Keep the law's FORM, learn its sensitivities.** A gray-box should retain the deposition
   structure (adhesion driven by activation agonists, low-shear/separation gating, `Mas` autocat,
   `Sat` saturation) but **learn** the effective rate + gate from data, supervised on GT `Mat` at
   the wall. The repo already has hooks: `compute_adhesion_gate` with `FourierTauGate` /
   `SpatialConditionedGate` (`BIOCHEM_ADHESION_GATE`).
3. **Feature the activation agonists, not the platelet pools.** `T` (+0.49) and `APR` (+0.46) are
   the most predictive upstream signals; this explains why thrombin helped individually in the
   earlier species-ranking work, and why bulk-platelet channels added to a GNN did not (they feed
   the trigger only after the coupled consumption that a feed-forward GNN cannot see).
4. **Model on the wall band.** Restrict learning to wall / wall-adjacent nodes (where `Mat` lives)
   with wall-local shear + activation features, rather than full-domain bulk-species transport.
