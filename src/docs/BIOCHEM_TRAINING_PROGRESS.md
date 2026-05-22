# Biochem training progress log

Living notes for **Phase 3 biochem corrector** (`src/training/train_biochem_corrector.py`): what we tried, what mattered, and how far we are from a **full-complexity** run.

**Maintained by:** humans + Cursor agents (see `.cursor/rules/biochem-training-progress.mdc` and root `AGENTS.md`). Agents should append the run log and adjust gates when you paste training results; you do not need to ask each time unless you want to skip updates for a chat.

---

## Complexity ladder (what “full run” means)

Training is staged by **loss complexity** and **pipeline length**, not a single switch.

| Level | Label | Backprop loss | Pipeline | Typical preset / env |
|-------|--------|---------------|----------|----------------------|
| **0** | Pretrain | AE recon; ODE reaction mimic | AE → ODE-RXN → … | Default fast budgets |
| **1** | Teacher (anchors) | Supervised COMSOL on anchors only | Same script, teacher loop | `BIOCHEM_STOP_AFTER_TEACHER=1` |
| **2** | **Step 2** (current target) | `L_Data_Kine + L_Data_Bio + W_MuSI·L_MuSI` (+ optional `L_PhysTemp`) | Teacher (+ optional early stop) | `BIOCHEM_LOSS_DATA_ONLY=1`, `BIOCHEM_COMPLEXITY_STEP=2` |
| **2.5** | Step 2 + temporal | Step 2 + `w_pt·L_PhysTemp` on anchor trajectories | Teacher / short corrector | `BIOCHEM_PRESET=step2p5` or `DATA_ONLY_PHYS_TEMP=1` |
| **2+** | Thrombus corona bundle (**experimental / unvalidated**) | Step 2 + gelation prior gate + 3-hop corona + phys temp | **Teacher + full corrector** (mixed graphs, pseudo labels) | `BIOCHEM_PRESET=thrombus_corona`, `STOP_AFTER_TEACHER=0` — **not recommended yet** |
| **3** | Full multitask | Kendall sum: PDE + walls + ADR + data heads (not data-only) | Full corrector, LoRA on | `BIOCHEM_COMPLEXITY_STEP=3` → forces `LOSS_DATA_ONLY=0` |
| **Prod** | Long schedules | Step 2 or 3 + long AE/ODE/teacher/corrector | Overnight wall time | `BIOCHEM_PRESET=overnight_step2` (still step-2 loss tier) |

**“All losses”** in code terms = **complexity step 3** (`BIOCHEM_LOSS_DATA_ONLY=0`): physics Kendall terms enter `backward()`, not only metrics.

**“Full run”** (aspirational) = teacher + corrector to completion with stable μ/species on val anchors — **after** step-2 teacher is healthy, then optional step 2.5 / spatial priors / step 3. The **`thrombus_corona` preset** is one *unvalidated* bundle for that path; do not treat it as the default iteration entry point.

---

## Experimental presets (`thrombus_corona`, `comprehensive_mu`)

**Status: unvalidated — keep in code, do not use for current μ iteration.**

| Preset | What it bundles | Evidence |
|--------|-----------------|----------|
| `thrombus_corona` | `GELATION_PRIOR_GATE=1`, `PRIOR_THROMBUS_CORONA_HOPS=3`, `DATA_ONLY_PHYS_TEMP`, `STOP_AFTER_TEACHER=0`, step-2 data-only | **One run** (2026-05-16): teacher μ **flat ~1.484**; corrector **1.569→1.548**; confounds: `MU_RATIO_MAX` default **1.0**, TF≈1, early TBPTT windows |
| `comprehensive_mu` | Corona + long AE/ODE/teacher/corrector + μ best-practice env | **No run** showing μ unlock vs patient007 ~1.48 plateau or A0 (~0.44) |

**Why it exists:** convenience for a future “full pipeline + wall-localized gelation” experiment. **Why not now:** bundles corrector, joint losses, and spatial priors before μ formulation is understood; overwrites env (e.g. forces `PhysTemp`).

**When to revisit:** after joint step-2 (Phase D) on a good teacher checkpoint — test **`GELATION_PRIOR_GATE=1`** and **`PRIOR_THROMBUS_CORONA_HOPS=3`** as **separate env flags**, not the full preset.

**Preferred iteration:** `scripts/run_biochem_mu_formulation_study.ps1` (teacher-only, `MU_LOG` isolate).

---

## Where we are now (2026-05)

### Gate checklist

| Gate | Target (teacher) | Status | Notes |
|------|------------------|--------|--------|
| Preflight μ (train anchors, t0→t1) | median logMAE ≲ 2.5 | **Pass** | ~1.43–1.45 |
| Val μ (held-out anchor, e.g. patient007) | improve / stabilize logMAE | **Partial** | Overnight A (TBPTT=6, `MU_LOG`, 18ep): best **0.3868** ep17; Marathon T2 **0.40** ep6; I1/I2/I4 ~0.44–0.49 ep3; SAFEVAL/V4 family around **~0.50** (best **0.5030** ep8) and 64-epoch runs confirm no new best with late degradation after ~30 ep; step-3 teacher max-complexity stayed flat **~1.51** with grad-skip; wall still **~1.7–1.8** |
| Val spatial correlation `r` | ≳ 0.5+ stable | **Partial** | Marathon T2 ep6 **r≈0.40**; bulk **r** often negative; high-μ **r** can be positive while all-truth **r** low |
| Wall μ logMAE | ≲ 1.5 | **Fail** | Marathon μ winners still **wall ~1.76–1.92**; bulk logMAE can be **~0.28–0.40** |
| `L_bio` on anchors | Decrease without μ stall | **Pass** | **I3** `DATA_BIO` isolate: train `L_bio`↓, val μ **flat ~1.47** |
| Phase A: `MU_SI` isolate, TF≈1 | Val logMAE drops | **Fail** | Flat ~1.59 (old config, no μ-path / high TF) |
| Phase B: `MU_SI` + low TF + μ-path | Val logMAE drops | **Pass** | Marathon **I2** best **0.44** ep3 (same recipe as MU_LOG) |

### Distance to full run (honest)

- **Step-2 teacher “done”**: **Interim pass on patient007** — overnight A best **0.3868** (MU_LOG, TBPTT=6, 18ep); marathon **T2** **0.40**; A0/I1/I2/I4 **~0.44–0.49**. Wall/high-μ still weak; **J2** joint (+`W_MuSI`) blocked by flux-debug crash (fixed locally). Corrector not started.
- **Corrector + optional spatial priors** (corona *components*, not preset): only after joint step-2 stable; corona preset itself **unvalidated**.
- **Step 3 (all PDE losses in backward)**: **Blocked** until (1) μ + bio stable at step 2, (2) `DETACH_MACRO_STATE=0` stable without OOM, (3) adjoint not dominating with junk gradients. Latest teacher-only step-3 attempt hit pervasive bio-grad cap skips and flat μ.
- **Overnight / production**: Run only after fast probes pass with `VAL_TIME_STRIDE=10`; confirm once with `stride=1`.

**We are roughly at: μ formulation validated on patient007** (MU_LOG / MU_SI / DATA_KINE isolates + TBPTT=6 all reach **~0.40–0.49** val logMAE) **with subset caveats** (wall ~1.8, high-μ tail often worsens when bulk improves, bulk **r** weak). Next: finish **J2**, confirm **J3** (laptop B), then step-2 joint without isolate; not at corona / step 3.

---

## Metrics that matter (what to log each run)

| Metric | Why |
|--------|-----|
| **`mu_log_mae` (all truth)** | Primary checkpoint score (`mu_score = -logMAE`) |
| **`mu_log_mae` wall / high-μ_gt / bulk** | Wall was the blocker; high-μ tail shows clot tail |
| **`mu_pearson` (r)** | Spatial pattern, not just scale |
| **`mu_mae_si`** | Physical units sanity |
| **Train `L_tot` / `L_Back`** | Under isolate, equals weighted μ objective |
| **`L_bio(avg)`** | Only when *not* isolating — if ↓ but μ flat, bio is stealing the step |
| **`L_kine`** | Poor proxy for μ; mixed u,v,p,μ_nd + variance norm |
| **Preflight median logMAE** | t0→t1 sanity (note: cap may differ from teacher epochs) |

**Do not** use train `L_kine` alone to judge μ success.

Report in diary: `outputs/reports/training/biochem/<timestamp>/` (`metrics.jsonl`, `training_diary_main.jsonl`).

---

## Chronicle (issues → cause → fix / next)

### 1. `BIOCHEM_DETACH_MACRO_STATE` / TBPTT

- **Symptom**: With `DETACH_MACRO_STATE=1`, species/μ state graph severed each macro step; bio/μ improve slowly.
- **Fix**: Keep **`BIOCHEM_DETACH_MACRO_STATE=0`** for μ work unless OOM; shorten `BIOCHEM_TBPTT_MAX_WINDOW` instead.
- **Status**: Understood; default fast preset uses `0`.

### 2. `L_bio` collapses early; μ flat

- **Symptom**: Full teacher: `L_bio` 46 → 0.1; val `logMAE` ~1.5 flat; `L_kine` ~2–3 noisy.
- **Cause**: Species loss dominates and is easier; μ uses different path (rheology closure, final-step SI Huber, variance-normalized `L_Data_Kine`).
- **Fix**: **`LOSS_ISOLATE=MU_LOG`** + μ-path capacity + low TF; keep bio out of backward until μ moves on **patient007**.
- **Status**: **Partially addressed** (see §9, §16–§18); joint step-2 still blocked for μ-first work.

### 3. Teacher `mu_ratio_max = 1.0` (PDE escape hatch)

- **Symptom**: Viscosity capped at Newtonian scale during teacher; could not match COMSOL high-μ.
- **Fix**: **`BIOCHEM_TEACHER_MU_RATIO_MAX`** env (e.g. `80.0`) set each teacher epoch in `train_teacher_on_anchors`.
- **Status**: **Implemented** in code; use in all μ experiments.

### 4. Phase A — `BIOCHEM_LOSS_ISOLATE=MU_SI`, TF ≈ 1, window 2

- **Symptom**: Train `L_MuSI` ↓ slightly; val `logMAE` **flat** ~1.59.
- **Interpretation**: With GT species on anchors, explicit gelation is fixed; frozen kin + tiny `learned_clot_penalty` cannot represent COMSOL μ.
- **Status**: **Failed** as capacity test → need low TF or more rheology DOF.

### 5. Full teacher + `MU_RATIO_MAX=80` (24 ep)

- **Symptom**: Same as (2): bio down, μ flat ~1.49–1.58; grad skip ep 14.
- **Status**: Confirmed multi-task is not the only issue.

### 6. Low TF + `MU_SI` isolate (Run 1, 2026-05)

- **Config**: `TEACHER_FORCE_MIN=0`, `TF_WARMUP=4`, `MU_SI` isolate, `STOCK_DEFAULTS=1`, `SKIP_PRETRAIN=1`.
- **Result ep0→1**: all logMAE **1.474→1.467**; wall **1.981→1.876**; **r 0.40→0.43**.
- **Status**: **One-run early movement**, but not yet consistently reproduced across later repeats.

### 7. Validation slow (stride myth)

- **Symptom**: ~**2100 s** (~35 min) per val with `BIOCHEM_VAL_TIME_STRIDE=1` *and still* ~**2065 s** with **`stride=10`** on patient007 (large graph, DEQ + micro-ODE per retained step).
- **Cause**: Stride reduces **macro time indices**, not node count; each forward remains heavy.
- **Fix**: For iteration use **`BIOCHEM_TEACHER_SKIP_VAL=1`** and watch train `L_tot` / `L_MuSI`, or **`BIOCHEM_MAX_LOAD_VESSELS=1`** / smaller anchor for dev; full val only when needed. Final report: `stride=1` once.
- **Status**: Documented — do not expect 10× speedup from stride alone on this workload.

### 9. `MU_LOG` isolate + μ-path + delta head (2026-05-18, Quadro OOM-safe)

- **Symptom**: `MU_SI` isolate + low TF flat ~1.48–1.51 on patient007; bulk μ stuck while bio easy.
- **Config**: `LOSS_ISOLATE=MU_LOG`, `W_MuLog=2`, `W_MuSI=0`, `TRAIN_MU_ENCODER=1`, `USE_MU_PATH_GROUP=1`, `USE_DELTA_MU_HEAD=1`, `TEACHER_FORCE_MIN=0`, `MU_RATIO_MAX=20`, `DETACH_MACRO_STATE=1`, `TBPTT=4`, `MAX_LOAD_VESSELS=3` (val **patient003**).
- **Result**: val logMAE **1.41 → 0.51** ep0→5; wall **1.97 → 1.42**; high-μ **0.85 → 0.95** (tail regressed ep4→5); **r ~0.11→0.14**. Train `W·L_MuLog` **3.05 → 1.61**.
- **Interpretation**: Aligning backward with log-μ + extra rheology DOF breaks the plateau on this split; **r** still poor; high-μ vs bulk trade-off at ep5; **not comparable** to patient007 runs until repeated with same val anchor.
- **Status**: **Promising** — next: same recipe, `MAX_LOAD_VESSELS` unset or 5+, val **patient007**.

### 8. Preflight vs training μ cap

- **Symptom**: Preflight median ~1.44 at cap **1.0**; val ~1.51 at cap **80**.
- **Fix (todo)**: Run preflight at same `BIOCHEM_TEACHER_MU_RATIO_MAX` as epoch 0.
- **Status**: Known mismatch.

### 9. Preset overwrites env

- **Symptom**: `thrombus_corona` sets `DATA_ONLY_PHYS_TEMP=1` even if user set `0`.
- **Fix**: Use **`BIOCHEM_STOCK_DEFAULTS=1`** and no preset for μ probes; or re-export vars after preset (preset runs at import).
- **Status**: Documented.

### 19. `thrombus_corona` / `comprehensive_mu` presets — experimental, not validated

- **Symptom**: Docs/scripts once called corona “recommended”; single corona run did not improve μ vs ~1.48 plateau; A0 (`MU_LOG` + μ-path, no corona) reached patient007 **~0.44**.
- **Cause**: Preset bundles corrector + joint step-2 + spatial priors + `PhysTemp` — too many moving parts; not isolated as helpful.
- **Fix**: Mark **experimental / unvalidated**; iterate with `run_biochem_mu_formulation_study.ps1`; test `GELATION_PRIOR_GATE` / `CORONA_HOPS` individually only after step-2 teacher works.
- **Status**: Documented (see **Experimental presets** section).

### 10. `MU_SI` isolate + **TBPTT window = 2** (12 ep, stride=10 val)

- **Symptom**: Train **`L_tot` ≈ `L_Back` frozen ~4.29×10⁻³** every epoch; val **logMAE ~1.489→1.488** (noise); best **-1.4880** ep5; **`r` ~0.357** flat; wall **~2.25** vs **~1.88** in TBPTT=4 run.
- **Cause**: Two-step windows mostly stress **t0→t1**; same regime as preflight; little gradient pressure to fix **held-out spatial / late-time** μ. Teacher forcing → **0** by late epochs → debug **`L_Data_Bio` explodes** (499→710): autoregressive species drift **without** bio loss in backward — scary in logs, **not** the optimized objective under isolate.
- **Fix**: Use **`BIOCHEM_TBPTT_MAX_WINDOW=4–8`** (OOM permitting) for μ probes; do not shrink to 2 for val generalization. Next: **code** — `L_mu_log` + multi-step `L_MuSI` (backlog below).
- **Status**: Run **not** worth continuing; cancel OK.

### 11. `BIOCHEM_DEBUG=1` Kendall table vs `LOSS_ISOLATE`

- **Symptom**: Debug prints full Kendall breakdown every batch while `BIOCHEM_LOSS_ISOLATE=MU_SI`.
- **Reality**: **Backward uses only the isolated term**; the table is from forward/metrics, not the scalar `loss.backward()`.
- **Fix**: Turn off **`BIOCHEM_DEBUG=0`** unless diagnosing; trust **`L_tot`/`L_Back`** line for isolate.
- **Status**: Clarified.

### 12. Step-2 low-TF sweep (teacher-only) did not unlock μ on RTX 500

- **Symptom**: On RTX 500 teacher-only legs (`MU_SI` isolate, joint step-2 with `W_MuLog=2`, and `+PhysTemp`) all converged to **val logMAE ~1.5128–1.5150** with wall **~2.39**.
- **Cause**: Changing isolate/joint weighting and adding `L_PhysTemp` in this regime did not materially alter the held-out anchor trajectory; teacher-only ceiling remained near ~1.51.
- **Fix**: Treat these knobs as second-order until loss-path alignment changes (`L_mu_log`, multi-step μ) and/or broader context (corrector/corona) is introduced.
- **Status**: Confirmed by 3-leg sweep (`baseline_lowTF_MU_SI`, `S2_joint_step2_lowTF`, `S25_step2_plus_phys_temp`).

### 13. Cross-machine "earlywin" is reproducible as ~1.48 band, not a new SOTA

- **Symptom**: Quadro run reproduced low-TF "earlywin" around **1.4799** (`MU_SI`) and **1.4805** (`MU_LOG`) on the held-out anchor; 1-anchor overfit legs were not evaluable (`TEACHER_SKIP_VAL=1`).
- **Interpretation**: This is a real improvement vs the ~1.51 plateau from the RTX 500 sweep, but still above the existing best noted in this log (~1.4666), so not a decisive breakthrough.
- **Fix**: Keep 5-anchor split as the acceptance test and avoid drawing conclusions from 1-anchor skip-val debug runs.
- **Status**: Confirmed as an incremental gain, not a gate flip.

### 14. μ smoke script runs validate optimization signal, not μ generalization

- **Symptom**: `run_biochem_mu_smoke_fast.ps1` (`MAX_LOAD_VESSELS=1`, `LOW_ANCHOR_MODE=1`, `TEACHER_SKIP_VAL=1`) gives smooth decreases in `L_Back` for `MU_LOG`, `MU_SI`, and `MU_LOG+delta_head`.
- **Interpretation**: Useful for proving gradients flow through μ path (`mu_encoder`/mu-head groups), but not evidence of held-out μ improvement because train/val anchor file is identical and val is skipped.
- **Fix**: Treat smoke runs as a pre-check only; require multi-anchor held-out val (`patient007`) before claiming μ progress.
- **Status**: Confirmed.

### 15. RTX500 repeat (`P_repro_lowTF_earlywin_MU_SI`) shows flat val after strong ep0

- **Symptom**: On 5-anchor teacher run (`MU_SI`, low-TF, TBPTT=4, stride=10), val starts at **1.4860** (wall **2.2418**, high-μ **0.9233**) and stays essentially flat through ep8 (**~1.4861–1.4867**).
- **Interpretation**: Better initialization regime than ~1.51 sweeps, but no meaningful epoch-wise μ learning trend yet (best at ep0). This does **not** confirm a solved μ-training recipe.
- **Fix**: Keep objective alignment (`MU_LOG` where possible), preserve held-out validation, and avoid reporting one-epoch wins as solved until repeated across runs/seeds.
- **Status**: Confirms partial/stalled, not solved.

### 16. μ is a hybrid closure, not a single learned field

- **Symptom**: Treating μ like a species channel in `L_Data_Kine` / `L_Data_Bio` multitask gives flat val logMAE.
- **Cause**: Forward μ = **Carreau(γ̇)** × **(1 + explicit Mat/FI gelation + learned_clot_penalty)** × **exp(delta_log_mu)**; `mu_encoder` couples μ into frozen kinematics DEQ. Gradients must flow through this path.
- **Fix**: Train **`mu_encoder` + `learned_clot_penalty` + `mu_delta_head`** (`USE_MU_PATH_GROUP=1`) under a μ-specific loss; do not expect frozen-kin + tiny penalty alone to match COMSOL.
- **Status**: **Lesson locked in** — use μ-path group for all μ studies.

### 17. Optimize the metric you report (`MU_LOG` vs `MU_SI`)

- **Symptom**: `MU_SI` isolate: train `L_MuSI` drifts; val **logMAE** flat ~1.48–1.59. `MU_LOG` on patient007 alone (~1.48) barely beats `MU_SI`.
- **Cause**: Huber in SI Pa·s is the wrong geometry for clot/lumen **orders-of-magnitude** μ; val always uses **|log μ_pred − log μ_gt|**.
- **Fix**: Default μ probes to **`LOSS_ISOLATE=MU_LOG`**, `W_MuLog=2`, `W_MuSI=0`. Add small `W_MuSI` only in later coupling legs.
- **Status**: **Default for μ formulation study** (see study plan below).

### 18. Big val wins need the right held-out patient *and* capacity

- **Symptom**: `mu_learned_only_oomsafe`: logMAE **1.41→0.51** on **patient003** (3-vessel cap); patient007 repro with `MU_LOG` only stays **~1.48**.
- **Cause**: Easier val split + full μ-path stack; bulk log loss improved while **high-μ tail regressed** ep5 (0.66→0.95); **`r` stayed ~0.14** (magnitude not pattern).
- **Fix**: Acceptance = **patient007**, **no `MAX_LOAD_VESSELS` cap**, log **wall / high-μ / bulk** every epoch. Treat patient003 0.51 as a signal, not SOTA.
- **Status**: **Gate for next runs** — study script Phase A.

### 20. Dual-laptop complexity marathon (2026-05-18) — isolate then combine

- **Setup**: `run_biochem_teacher_complexity_laptop_a.ps1` / `_b.ps1`; patient007 val, stride=10, val every 3 ep, `DETACH=1`, TBPTT=4 default, μ-path on, `TEACHER_FORCE_MIN=0`, warm-start pretrain.
- **μ isolates (laptop A, RTX 500 4GB)**: **I1** `MU_LOG` best **0.49** ep3; **I2** `MU_SI` best **0.44** ep3; **I4** `DATA_KINE` best **0.48** ep3; **I3** `DATA_BIO` val μ **flat ~1.47** (species-only backward does not move μ).
- **Joint (A)**: **J1** step-2 (`L_Data_*` + `W_MuLog=2`) best **0.48** ep3 — matches isolates, not clearly better. **J2** (`+W_MuSI=4`) **crashed** ep0 in `boundary_flux.inlet_effective_width_nd` (mask vs `flow_hint` shape); wrap flux debug in try/except + shape-aware inlet width.
- **Physics / temporal (laptop B, P2200 5GB)**: **I5** `PHYS_TEMP` isolate: val μ **1.45→1.36** (small). **I6** `ADR_F`: val μ **~1.48** flat. **T1** TBPTT=5: **0.47** ep3. **T2** TBPTT=6: **0.40** ep6 (**best marathon**). **J3** was still running at log cutoff.
- **Runtime**: ~**11 min/val epoch** on patient007 → **~5–6 h** per laptop, not the scripted ~3 h target.
- **Status**: Isolated μ losses **validated** on patient007; physics-only isolates **do not replace** `MU_LOG`+μ-path; longer TBPTT helps.

### 21. `MU_SI` vs `MU_LOG` under μ-path (revises §17)

- **Symptom**: Older runs: `MU_SI` flat ~1.48–1.59 without μ-path / high TF.
- **Marathon**: With **μ-path + low TF + TBPTT≥4**, **I2** best **0.44** ep3 vs **I1** **0.49** ep3.
- **Fix**: Prefer **`MU_LOG`** (matches val metric); **`MU_SI` is viable** in this stack when capacity + TF match.
- **Status**: Revises “MU_SI always fails” — config-specific, not law.

### 22. High-μ tail vs bulk tradeoff (persistent)

- **Symptom**: All-truth logMAE can drop while **high-μ_gt** worsens (I1: **0.89→1.54**); wall **~1.75–2.0** across μ-winning legs.
- **Interpretation**: Bulk scale improves; clot-tail and wall remain hard; positive high-μ **r** ≠ good spatial μ (**bulk r** often negative).
- **Status**: Open.

### 23. Overnight A vs B (step-2 teacher): `PhysTemp=1` does not beat baseline

- **Setup**: Same teacher-only step-2 recipe (`STOP_AFTER_TEACHER=1`, TBPTT=6, `DETACH=1`, `W_MuLog=2`, 18 epochs, patient007 val), comparing A (`DATA_ONLY_PHYS_TEMP=0`) vs B (`DATA_ONLY_PHYS_TEMP=1`).
- **Result**: A best **logMAE 0.3868** (ep17) vs B best **0.4081** (ep12); wall **1.718** vs **1.762**; high-μ **1.356** vs **1.415**. B is worse by ~0.02 all-truth on the main score.
- **Interpretation**: In this teacher-only regime, adding temporal SI anchor loss does not improve held-out μ error and slightly degrades the key subsets.
- **Fix**: Keep overnight default at step-2 (`DATA_ONLY_PHYS_TEMP=0`) for now; treat `step2p5`/PhysTemp as a later coupling probe after joint step-2 (corrector-on) is stable.
- **Status**: Confirmed by cross-machine overnight pair (RTX 500 Ada vs Quadro P2200).

### 24. Architecture sweep (A0-A4, B0-B4): `delta_mu_head` gate dominates width/latent tweaks

- **Setup**: Teacher-only (`STOP_AFTER_TEACHER=1`), `LOSS_ISOLATE=MU_LOG`, `W_MuLog=2`, `W_MuSI=0`, TBPTT=6, `DETACH=1`, 8 epochs, patient007 val, stride=10, low-TF schedule.
- **Result (both laptops)**: All `delta1` legs converge to a tight band (**~0.47-0.51** all-truth logMAE). Bests: A3 **0.4756** (RTX 500), B1 **0.4738** (P2200).  
- **Failure mode**: Both `delta0` legs (A4/B4) stay near **~1.45** with almost no epoch-wise movement despite identical training setup otherwise.
- **Interpretation**: In this recipe, the residual rheology correction path (`USE_DELTA_MU_HEAD`) is a first-order requirement; latent width/prior width are second-order for all-truth logMAE.
- **Caveat**: Best all-truth legs can still have weak/negative `r` or high wall error; e.g., A3 wins logMAE while all-truth `r` is negative, so architecture ranking must include subset metrics.
- **Status**: Lesson locked in; keep `delta1` as default in architecture probes and avoid investing in `delta0` variants.

### 25. Teacher max-complexity (step-3) run: unstable gradients, no μ learning (2026-05-20)

- **Setup**: `BIOCHEM_PRESET=teacher_max_complexity`, teacher-only (`STOP_AFTER_TEACHER=1`), Quadro P2200, full pretrain + teacher, `DETACH=0`, TBPTT=8, `W_MuSI=8`, `W_MuLog=2`, expected 30 ep from CLI but preset pinned teacher to 24.
- **Symptom**: Every teacher batch triggered bio-grad cap skip (`bio grad L2` far above cap 5000, often `1e6`-`1e14`), so optimizer steps were effectively starved.
- **Result**: Val μ remained flat: all-truth ~**1.5116-1.5136**, wall ~**2.428-2.430**, high-μ ~**0.915-0.926**, `r` ~**0.395**; no epoch-wise viscosity learning despite long run.
- **Interpretation**: Turning on full step-3 teacher loss too early destabilizes optimization on this stack; PDE/multitask gradients dominate and trip safety caps before μ path can improve patient007.
- **Fix**: Keep mainline training at step-2 teacher (`MU_LOG`/joint step-2), and only retry step-3 after reducing teacher LR / rebalancing caps and verifying non-skipped updates. Also fix preset-vs-CLI epoch precedence so `-TeacherEpochs` is honored.
- **Status**: Confirms step-3 remains blocked for current teacher-only viscosity target.

### 26. Viscosity-baseline preset (`teacher_visc_baseline`): strong early gain, then late drift (2026-05-20)

- **Setup**: Teacher-only step-2 baseline with warm-start, Quadro P2200, `DETACH=1`, TBPTT=6, `W_MuSI=2`, `W_MuLog=2`, plus subset log losses (`W_MuLogWall=2.5`, `W_MuLogHigh=1.5`), 18 epochs.
- **Result**: Rapid μ improvement by ep6: all-truth **1.4044 -> 0.5418** (best), wall **2.3677 -> 2.0983**, high-μ **0.7994 -> 0.9935** (worse than ep0). Late epochs drifted: all worsened to **0.8451** by ep17, while high-μ improved to **0.5961** and `r` rose to **~0.47**.
- **Interpretation**: Baseline objective can quickly improve global μ scale but is not yet stable; wall remains the dominant blocker (~2.06-2.14), and late training shifts capacity toward tail/correlation at the expense of all-truth error.
- **Fix**: Keep this as the new incremental base, but add checkpoint selection/early-stop on all-truth μ around ep4-8, then ablate added terms one at a time (wall/high weights, temporal term, then selective physics).
- **Status**: Useful base model for incremental ablations; not a replacement for current best (~0.39-0.40) yet.

### 27. Dual baseline runs (2026-05-20): wall-vs-all tradeoff + preset override confound

- **Setup**: Two runs of `run_biochem_teacher_visc_baseline.ps1` on different GPUs (Quadro P2200 vs RTX 500 Ada) with different CLI knobs (A: more aggressive wall/high + `DETACH=0`, B: milder wall/high + early stop target 0.55).
- **Result A (Quadro)**: best all-truth **0.5196** (ep14), wall **2.0581**, high-μ **0.9014**, `r` **0.405**.
- **Result B (RTX500)**: best all-truth **0.5398** (ep12), wall **1.9456**, high-μ **0.9426**, `r` **0.446**; early-stop fired at target.
- **Interpretation**: A is better on global all-truth μ; B is better on wall and correlation. Both remain far from wall target and both underperform prior best all-truth (~0.39-0.40). This confirms a persistent wall-vs-all tradeoff.
- **Critical confound**: runtime logs show `W_MuSI=8.0` and `DETACH_MACRO=1` despite CLI attempts to set lower `W_MuSI` / `DETACH=0`; preset defaults are overriding some script knobs, so these A/B runs are not clean ablations yet.
- **Fix**: make preset truly override-safe for CLI knobs (or switch to `BIOCHEM_STOCK_DEFAULTS=1` in ablation script), then rerun A/B before interpreting subtle weight effects.
- **Status**: Actionable but partially confounded evidence; next iteration should first remove override ambiguity.

### 28. SAFEVAL dual runs (2026-05-20): stable execution, improved all-truth, wall still bottleneck

- **Setup**: Both laptops rerun with explicit `BIOCHEM_STOCK_DEFAULTS=1`, `VAL_TIME_STRIDE=20`, `TEACHER_VAL_EVERY=4`, TBPTT=6, `DETACH=1`, warm-start, and early-stop thresholds (0.55 / 0.52).
- **Run 1 (Quadro, wall-focused weights 2.6/1.0)**: best all-truth **0.5249** (ep8), wall **2.0795**, high-μ **0.9621**, `r` **0.402**.
- **Run 2 (RTX500, global-stable weights 1.4/0.6)**: best all-truth **0.5055** (ep8), wall **1.9687**, high-μ **0.9978**, `r` **0.419**.
- **Interpretation**: SAFEVAL fixed the validation hang and produced cleaner A/B behavior. Lower wall/high weights improved global μ and wall simultaneously (run 2 beats run 1 on all-truth and wall), but high-μ tail remains weak and wall is still far from target.
- **Fix**: use run 2 as the base checkpoint line; next changes should target high-μ and wall without regressing all-truth (e.g., mild high-μ curriculum, then selective wall-local temporal/physics term).
- **Status**: New best for this baseline family is **0.5055**; still below prior global best (~0.39-0.40).

### 29. VISC_V3 pair (`TAIL_RECOVERY` vs `WALL_PUSH`): all-truth win vs stalled wall push (2026-05-20)

- **Setup**: Teacher-only step-2 with explicit stock env, warm-start, `VAL_STRIDE=20`, `VAL_EVERY=4`, TBPTT=6, `DETACH=1`, `W_MuSI=2`, `W_MuLog=2`, `MU_RATIO_MAX=80`, low-TF schedule.
- **Run 1 (`VISC_V3_TAIL_RECOVERY`)**: best all-truth **0.5153** (ep12, early-stop target 0.52 hit), wall **1.9728**, high-μ **0.9655**, `r` **0.443**.
- **Run 2 (`VISC_V3_WALL_PUSH`, in progress)**: best so far all-truth **0.5289** (ep8), wall **2.0814**, high-μ **0.9874**, `r` **0.402**; later vals drifted to **0.5395** by ep16.
- **Interpretation**: Increasing wall weight while reducing high-μ weight (`Wall/High = 2.2/0.6`) did not improve wall on patient007; it degraded all-truth and correlation vs `1.4/1.2`.
- **Status**: Keep Run 1 weighting as the safer baseline for this family; treat Run 2 as a negative ablation unless late epochs reverse trend.

### 30. V4 `global_plus` first attempt (2026-05-20): 4GB OOM with wide latent

- **Setup**: `run_biochem_teacher_visc_v4.ps1 -Profile global_plus` on RTX 500 4GB; latent **320**, prior=2, TBPTT=6, RK4=8, warm-start on.
- **Symptom**: OOM before first val/epoch (`torch.OutOfMemoryError` in ODE adjoint + GAT softmax path) after startup.
- **Additional signal**: warm-start reported many shape mismatches/skips due width change (`latent 256 -> 320`), increasing instability/risk for this hardware budget.
- **Fix**: Make V4 script **4GB-safe by default** (`latent=256`, TBPTT=5, RK4=6, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`) and keep `-WideArch` as opt-in for larger VRAM.
- **Status**: Script updated; rerun `global_plus` and `high_mu_only` in safe mode first.

### 31. V4 safe reruns (`global_plus` + `high_mu_only`): early optimum + late collapse, tail isolate confirms tradeoff (2026-05-20)

- **`global_plus` (RTX500, latent256/prior2, TBPTT=5/RK4=6)**: best all-truth **0.5030** (ep8), wall **1.9661**, high-μ **0.9495**, `r` **0.432**; then severe late drift (**0.7782** ep16, **1.3298** ep20, **0.9836** ep23) while train-side `final_anchor_logMAE` stayed ~0.50.
- **Signal**: this profile can hit the target band quickly but lacks stability; once teacher forcing decays and long-horizon rollout dominates, held-out all/bulk degrade sharply.
- **`high_mu_only` (P2200, latent320/prior4, isolate `MU_LOG_HIGH`)**: strong high-tail gain (**0.9434 -> 0.5822** by ep4, ~0.5838 ep8), but all-truth remains poor (~**1.00**) and wall stays high (~**2.02-2.09**).
- **Interpretation**: pure high-tail isolate is useful as a diagnostic for clot-region capacity, but not as a final teacher objective; it needs a bridge back to global/wall terms to avoid sacrificing full-field fidelity.
- **Fix**: move to long-run low-LR profiles: (1) stable global objective with TF floor + reduced μ-path LR, and (2) tail-emphasis **without isolate** on wider arch for 5GB+ cards.

### 32. Long-horizon V4 runs (`global_long_stable` + `tail_bridge_long`): best remains early, late epochs mostly trade all-truth for tail (2026-05-21)

- **`global_long_stable` (RTX500, 64ep, latent256/prior2, LR 1e-3, μ-path LR mult 0.65)**: best all-truth **0.5068** (ep30), wall **1.9507**, high-μ **0.9062**, `r` **0.441**. After ep33, all-truth degrades sharply (**0.61 → 1.07** range), while high-μ improves (**0.77 → 0.63/0.60**) and wall remains ~**1.95–2.08**.
- **`tail_bridge_long` (P2200, 64ep, latent320/prior4, LR 8e-4, μ-path LR mult 0.50, tail-heavy joint loss)**: best all-truth **0.5184** (ep9), wall **2.0761**, high-μ **0.9290**, `r` **0.434**. Late epochs consistently favor high-tail (**~0.42–0.49**) with strong high-μ `r` (~**0.74**), but all-truth stays poor (**~0.76–0.86**) and wall stays high (~**2.05**).
- **Interpretation**: both long runs reinforce the same regime: after early epochs, optimization shifts toward tail/wall-local behavior while bulk/all-truth deteriorates. Lower LR and TF floor slowed catastrophic collapse on RTX500 but did not prevent objective drift.
- **Actionable rule**: for current teacher-only step-2, select checkpoints in the **ep8–30 window** (before drift) and avoid assuming longer schedules improve held-out all-truth.
- **Status**: no gate flip; long horizon did not surpass existing bests and confirms persistent wall-vs-all-vs-tail tradeoff.

### 33. New physics branch (Carreau baseline + trigger-gated tail correction) — implementation note (2026-05-21)

- **Hypothesis**: bulk should remain near Carreau; high-μ uplift should activate only in clot-triggered regions (species + mechanics).
- **Architecture change**: optional split residual log-μ heads (`BIOCHEM_USE_SPLIT_MU_HEAD=1`) with gate:
  `log μ = log μ_carreau + (1-g)*Δ_bulk + g*Δ_tail`, where `g` is a learned trigger gate.
- **Loss change**: wall objective can be disabled (`MU_LOG_WALL_WEIGHT=0`) to focus on global + high-tail.
- **Anti-collapse priors** (opt-in): floor penalties on trigger gate and learned gelation on high-μ truth nodes (`BIOCHEM_TRIGGER_*` env knobs) to prevent tail path collapse late in training.
- **Checkpointing change** (opt-in): Pareto checkpoint rule (`BIOCHEM_TEACHER_PARETO_CHECKPOINT=1`) updates best model only when all/high tradeoff improves within configured tolerances.
- **Safety**: all features are env-gated defaults-off so prior behavior is preserved for A/B comparison.

### 34. First split-head tail-focused runs (`carreau_tail_split_4g/5g`) expose an early-training failure mode (2026-05-21)

- **Run 1 (`carreau_tail_split_4g`, RTX500)**: all-truth stayed flat/bad (**~1.519–1.524**), while high-μ improved only mildly (**0.946 -> 0.897** by ep15). Wall error rose to **~3.22** with wall loss disabled.
- **Run 2 (`carreau_tail_split_5g`, P2200)**: started in a poor basin (all **~1.48**, high **~1.27**, negative `r` on all/high), with no meaningful improvement by ep6.
- **Interpretation**: turning on split-head + strong anti-collapse floors from epoch 0 can trap optimization before trigger/tail pathway aligns; Pareto kept "least-bad" checkpoints but did not produce useful teacher quality.
- **Lesson**: stage the objective and regularizers. Early tail-probe should be lighter-constrained (smaller floors, boundary-aware signal), then add stronger global recovery in Stage B.
- **Action**: add V7 staged profiles: `carreau_tail_stageA_diag_4g` (tail bug-check) and `carreau_tail_stageAB_5g` (A->B curriculum) with separate bulk/tail/gate LR multipliers.

### 35. V7 staged split-head runs show major recovery vs V6, but still far from prior best global score (2026-05-21)

- **Run 1 (`carreau_tail_stageA_diag_4g`, RTX500)**: clear improvement from V6 failure mode. Best so far reached by ep21: all-truth **~0.8922**, high-μ **~0.5016**, `r` **~0.288** (high-μ `r` ~0.56), while wall stays unconstrained/high (~**2.48**) by design.
- **Run 1 trajectory**: initially poor (~1.54) then drops sharply (ep18/21), confirming tail pathway is trainable with lighter floors + stronger tail LR. This satisfies the “bug-check” intent but does not recover global fidelity.
- **Run 2 (`carreau_tail_stageAB_5g`, P2200, early)**: much better than prior 5G split run already by ep6: all-truth **~0.8694**, high-μ **~0.4761**, high-μ `r` **~0.71**, wall ~**2.24** (unconstrained).
- **Interpretation**: staged training fixed the early optimization trap from V6 on both machines (especially 5G), and high-tail metrics now move strongly. Remaining gap is global/bulk quality vs historical teacher best (~0.39-0.41) and wall (expected with wall loss disabled).
- **Status**: promising direction for tail-focused science; not yet a production teacher for overall viscosity fidelity.

### 36. V8 `carreau_tail_stageAB_wall_4g` repeated twice: deterministic replay, no new global/wall gain (2026-05-21)

- **Setup**: `run_biochem_teacher_visc_v4.ps1 -Profile carreau_tail_stageAB_wall_4g`, teacher-only (`STOP_AFTER_TEACHER=1`), RTX500 4GB, 48 epochs, split-head+gate, Pareto checkpointing on, Stage-B wall reintroduced at epoch 18.
- **Result (both runs)**: metric traces are effectively identical run-to-run; final saved teacher checkpoint remains epoch 21 with all-truth **0.7471**, wall **3.2250**, high-μ **0.4778**, `r` **0.376**.
- **Observed tradeoff**: all-truth reaches local minima later (e.g., **0.5694** at ep36) but with weaker high-μ (**0.6805**) and no wall recovery; Pareto rule preserves the earlier all/high compromise.
- **Interpretation**: this profile is reproducible, but still off target for clot-usable global fidelity: high-tail can be good while wall stays catastrophically high (~**3.22**) and all-truth remains worse than prior teacher best (~**0.39-0.40**).
- **Status**: no gate change; keep as reproducibility evidence and revisit Stage-B weighting/checkpoint policy before further long runs.

### 37. V7 `carreau_tail_stageAB_5g` mid/late continuation: strong recovery vs V6, but still a tail-first compromise (2026-05-21)

- **Setup**: continuation of `carreau_tail_stageAB_5g` on P2200 through ep35 (teacher-only, split-head staged A→B, wall objective disabled by profile design, Pareto checkpointing on).
- **Result**: best all-truth improved from early **0.8694** (ep6) to **0.7121** (ep30); wall reached **~2.083** (ep27); high-μ best **0.4447** (ep15); all-truth `r` peaked around **0.471** (ep21).
- **Trajectory**: optimization is non-monotonic (e.g., regressions around ep15/24), but recovers repeatedly and stays far better than V6 split-head basins (~1.48+).
- **Interpretation**: staged split-head training is now clearly viable on 5GB for tail-sensitive behavior, yet still underperforms the existing global best (~0.39-0.40 all-truth) and does not solve wall error.
- **Status**: no gate flip; keep this branch as a promising tail-focused track, not the primary production teacher objective.

### 38. Architecture update: wall-aware residual μ branch added for split-head runs (2026-05-21)

- **Change**: added optional `BIOCHEM_USE_WALL_DELTA_HEAD=1` path in `GNODE_Phase3` with near-wall gating:
  `Δlogμ_total = (1-g)Δbulk + gΔtail + gain * g_wall * Δwall`.
- **Why**: current split-head runs can improve high-μ tail while leaving wall μ severely underfit; a dedicated wall branch gives separate capacity for near-wall correction.
- **Control knobs**: `BIOCHEM_MU_WALL_GATE_TEMP`, `BIOCHEM_MU_WALL_GATE_CENTER`, `BIOCHEM_MU_WALL_DELTA_GAIN`, and optimizer `BIOCHEM_MU_WALL_LR_MULT`.
- **Runner support**: new profiles `walltail_arch_v1_4g` and `walltail_arch_v1_5g` added to `run_biochem_teacher_visc_v4.ps1`.
- **Status**: ready for A/B runs on both laptops; no gate change until validation metrics arrive.

### 39. V9 wall-aware runs (`walltail_arch_v1_4g/5g`): mixed outcome, clear cues for next architecture step (2026-05-21)

- **4G run (`walltail_arch_v1_4g`, ongoing)**: strong all-truth recovery to **0.5778** (ep30) and high-μ **~0.3837-0.4269**, but wall remains stuck at **~3.07** despite Stage-B wall loss activation. After ep30, run shows sharp instability (ep33 all-truth **1.1755**).
- **5G run (`walltail_arch_v1_5g`, ongoing early)**: very unstable trajectory (all-truth swings **1.53 -> 0.77 -> 1.72 -> 0.85** by ep12), while wall holds around **~2.44-2.47** and high-μ can be good (**0.435** at ep9).
- **Interpretation**: wall-aware branch improved global/tail capacity on 4G but did not convert to wall fidelity; abrupt stage transitions and weakly anchored wall gating likely contribute to late collapses.
- **Action**: implement smoother Stage-A->B interpolation and stronger wall-mask-informed wall-gate signal; launch long V2 runs with lower LR + higher wall-branch LR.

### 40. Architecture update V2: smooth stage blending + stronger wall-gate anchoring (2026-05-21)

- **Stage blending**: added `BIOCHEM_MU_STAGE_TRANSITION_EPOCHS` to interpolate Stage-A/B loss weights with smoothstep instead of hard switching.
- **Wall gating**: wall branch now mixes geometric wall proximity with explicit `mask_wall` (`BIOCHEM_MU_WALL_MASK_MIX`) before gate activation, reducing under-activation on near-wall truth nodes.
- **New long profiles**: `walltail_arch_v2_long_4g` and `walltail_arch_v2_long_5g` configured for ~10h teacher runs with lower base LR, smoother transition, and stronger wall branch learning rate.
- **Status**: code-complete and ready for dual-laptop launch.

### 41. Unattended sweep results: 4G completed, 5G partly completed (2026-05-22)

- **4G sweep leg A (`carreau_tail_stageAB_wall_4g`)**: best all-truth **0.5738** (ep47), wall **1.9561**, high-μ **0.5055**, `r` **0.395**. This is a major wall recovery versus prior ~3.2 wall runs.
- **4G sweep leg B (`walltail_arch_v2_long_4g`)**: best all-truth **0.5506** (ep54), high-μ **0.3513** (ep48), `r` **0.434**, but wall stayed poor (**~3.07**).
- **5G sweep leg A (`walltail_arch_v1_5g`)**: checkpoint selected at ep18 with all **0.5028**, wall **3.3940**, high-μ **0.2516**; later epochs improved all-truth to ~**0.439** but worsened high-μ, so Pareto kept earlier checkpoint.
- **5G sweep leg B (`walltail_arch_v2_long_5g`)**: pasted log is **incomplete** (startup + very early epochs only), so no final comparison yet.
- **Runtime note**: despite higher VRAM, the 5G machine is **not faster** in wall-clock terms on these profiles; per-val time is roughly ~800s (vs ~400s on 4G), so full runs can take as long or longer.
- **Decision signal**: on 4G, no single objective dominates — leg A wins wall, leg B wins all/high. On 5G, all-high Pareto tension is now the main blocker.

### 42. Pareto checkpoint policy now blocks “best-for-goal” selection in some runs (2026-05-22)

- **Symptom**: runs can reach much better all-truth later, but saved checkpoint stays earlier because high-μ regresses slightly (example: 5G `walltail_arch_v1_5g`).
- **Risk**: for clot-use, we need explicit control over wall/high/all tradeoffs; strict 2-objective Pareto can hide practically better checkpoints.
- **Action**: keep Pareto for safety, but add post-hoc selection from `metrics.jsonl` with a clot-focused score (e.g., weighted all + wall + high) before deciding deployment checkpoint.
- **Status**: open workflow fix (analysis-side), not a training-kernel blocker.

### 43. Step-2 isolate sweep smoke (`sweep_bio_suppressor` + `sweep_wall_sentinel`): config fix validated, early μ metric still flat (2026-05-22)

- **Setup**: both presets run with `BIOCHEM_STOCK_DEFAULTS=1`, `BIOCHEM_LOSS_ISOLATE=MU_LOG`, `W_MuSI=0.0`, latent=320, prior dim=4, teacher-only startup logs.
- **What worked**: desired preset wiring is active (explicit isolate banner, `μ SI w=0.0`); no non-finite grad-skip spam was observed before termination.
- **Validation metric reality**: both runs are effectively flat in early teacher validation: `sweep_bio_suppressor` all-truth `1.5155` (ep00) -> `1.5158/1.5159` (ep06/ep03); `sweep_wall_sentinel` `1.5096` (ep00) -> `1.5101/1.5102` (ep06/ep03).
- **Subset nuance**: wall/high-μ tradeoff is tiny and non-decisive in both runs; no clear early separation between suppressor-on and suppressor-off at this loss tier.
- **Resource note**: 4GB run (`sweep_bio_suppressor`) OOMed at ep07 during adjoint backward; 5GB run (`sweep_wall_sentinel`) reached ep06 shown without OOM under the same latent/prior settings.

### 44. Fast split-μ probe (`sweep_wall_sentinel`) unlocked all-truth quickly, but wall remained stuck and gates collapsed (2026-05-22)

- **Setup**: `BIOCHEM_PRESET=sweep_wall_sentinel` with updated fast architecture defaults (`TRAIN_MU_ENCODER=1`, `USE_SPLIT_MU_HEAD=1`, `USE_DELTA_MU_HEAD=1`, `TBPTT=5`, `DETACH_MACRO=1`, teacher-only 14 ep).
- **What improved**: held-out all-truth `mu_log_mae` dropped strongly from `1.0131` (ep00) to **`0.5496`** (ep08 best checkpoint), confirming the new sweep architecture is trainable and informative within <1h probe budget.
- **What stayed broken**: wall logMAE remained effectively pinned around **`2.57`** across epochs; bulk `r` stayed weak/negative and final all-truth `r` degraded vs the early best.
- **Gate diagnosis**: trigger gates rapidly collapsed toward zero (`gate_all` ~`5e-1` -> `1e-20`; `gate_wall` ~`8e-1` -> ~`1.9e-22`), which likely explains improving global magnitude while failing to recover wall-region behavior.
- **Selection nuance**: best checkpoint by all-truth is ep08 (`all=0.5496`, `high=0.8368`), while best high-μ arrives later (ep12 `high=0.4046`) with weaker all-truth, reaffirming objective tradeoff.

### 45. Fast split-μ probe (`sweep_bio_suppressor`) confirms suppressor can preserve clot-tail but still fails wall recovery (2026-05-22)

- **Setup**: RTX500 4GB, `BIOCHEM_PRESET=sweep_bio_suppressor`, latent=320, prior dim=4, teacher-only 14 ep, `DETACH_MACRO=1`, TBPTT=5.
- **Result**: best all-truth **0.5923** (ep10), high-μ **0.5563** at that checkpoint (best high-μ reached later **0.3182** ep13), but wall degraded to **~2.59** baseline and **~3.40** at late spikes.
- **Gate signal**: `gate_all` and `gate_clot` remain high/stable (~0.48), while `gate_wall` is numerically pinned near zero (~1.9e-22) throughout, indicating wall branch starvation despite non-zero wall loss weight.
- **Interpretation**: this run isolates the current failure mode: suppressor protects tail selectivity but over-suppresses wall correction, producing a strong high-μ vs wall tradeoff rather than balanced gains.

### 46. Gate-floor architecture patch removed gate collapse, but wall paradox remains (2026-05-22)

- **Setup**: both patched fast probes (`sweep_wall_sentinel` on RTX500 4GB, `sweep_bio_suppressor` on P2200 5GB) with new gate floors (`TRIGGER_GATE_MIN=0.06`, `WALL_GATE_MIN=0.08`) and suppressor wall-mixing controls.
- **What improved**: catastrophic gate collapse is gone; gates stay finite (`gate_all` around `0.06–0.72`, `gate_wall` floor-clamped at `0.06`), and both runs complete quickly without OOM.
- **Wall behavior**: still poor and nearly flat despite heavy wall loss. Sentinel run wall settles around **2.49** (better than earlier ~2.57 baseline), while suppressor run remains around **2.588**.
- **Tradeoff**: suppressor run gives better global/high balance (`all=0.5758`, `high=0.6040`) than sentinel (`all=0.6622`, `high=0.6849`) but does not improve wall.
- **Interpretation**: architecture fix solved gate-collapse pathology, but not the wall-identifiability bottleneck; wall target is still underfit even when gate starvation is prevented.

### 47. Wall-overcompensation probe validates directionality limits: wall modestly improves vs old baselines but remains sticky (2026-05-22)

- **Run 1 (`sweep_bio_suppressor`, RTX500 4GB)**: best all-truth **0.5055** (ep13), high-μ **0.6268**, wall **2.4888**. This is the best all-truth seen in the latest patched fast probes, with wall better than earlier ~2.57-3.40 failures but still far from target.
- **Run 2 (`sweep_wall_overcomp`, P2200 5GB)**: best all-truth **0.5682** (ep13), high-μ **1.1207**, wall **2.4951**. Despite aggressive wall weighting/gating, wall does not materially beat Run 1 and high-μ degrades strongly.
- **Gate behavior evidence**: overcomp run drives gates to saturation (`gate_wall≈1.0`, `gate_all≈0.98+`), proving the architectural controls work, but the metric response stays wall-limited.
- **Conclusion**: we now have causal evidence that simply forcing wall-gate activation and wall loss magnitude is insufficient; current wall error is dominated by representational/confounding limits rather than gate starvation alone.

### 48. MU_LOG_WALL isolate confirms wall is controllable but expensive to global/high fit (2026-05-22)

- **Run 1 (`sweep_bio_suppressor`, RTX500 4GB)**: best all-truth **0.6365** (ep06), wall **2.4953**, high-μ **0.7107**. Better high-tail balance than wall-overcomp, but wall remains essentially unchanged.
- **Run 2 (`sweep_wall_overcomp`, P2200 5GB, `LOSS_ISOLATE=MU_LOG_WALL`)**: wall improves materially to **2.0927** (ep10 best wall-point), proving wall loss path is active and not bugged, but all-truth/high-μ regress (`all=0.6640`, `high=1.1330` at best-all checkpoint).
- **Interpretation**: this is direct causal proof that architecture can move wall logMAE when forced; remaining challenge is multi-objective interference and representation sharing, not a dead wall-loss plumbing path.

### 49. Latest rerun pair confirms unstable wall tradeoff and motivates wall-species decoupling (2026-05-22)

- **Run 1 (`sweep_bio_suppressor`, RTX500 4GB)**: best all-truth **0.5195** (ep12), wall remains **~2.5906**, high-μ fluctuates (**0.7849** at best-all). Gate metrics collapse to floor values (`gate_all/gate_wall/gate_clot` ~0.06) for long stretches.
- **Run 2 (`sweep_wall_overcomp`, P2200 5GB, `MU_LOG_WALL` isolate)**: best all-truth **0.7012** (ep02), wall reaches only **~2.306** late, worse than prior overcomp best (~2.09). High-μ remains poor (**~1.25**) and run is wall-heavy but not wall-winning.
- **Takeaway**: wall can be moved by objective pressure, but response is brittle and non-monotonic; additional decoupling from clot-species channels is required to reduce spatial confounding on the wall branch.

### 50. New rerun pair after wall-decoupling patch: global recovered, wall still plateaus near ~2.1-2.6 band (2026-05-22)

- **Run 1 (`sweep_bio_suppressor`, RTX500 4GB)**: best all-truth **0.5195** (ep12), wall **2.5906**, high-μ **0.7849**. This recovers strong global fit but does not move wall off its sticky band.
- **Run 2 (`sweep_wall_overcomp`, P2200 5GB, `MU_LOG_WALL` isolate)**: best all-truth **0.5085** (ep06), wall **2.0868** (best shown), high-μ **0.9300** at best-all checkpoint. Confirms wall can still be pushed down, but not without high-μ/global tradeoff.
- **Gate evidence**: suppressor run often sits at floor-clamped gates (`~0.06`), while overcomp run keeps `gate_wall` near floor (`~0.03`) with intermittent increases in `gate_all`; wall movement is present but saturates quickly.
- **Interpretation**: architecture now reliably demonstrates wall-path responsiveness (not a bug), but balanced optimization remains unresolved; best practical point is still a global-vs-wall compromise rather than simultaneous improvement.

### 51. Newest pair (`sweep_bio_suppressor` MU_LOG vs `sweep_wall_overcomp` MU_LOG_WALL): global improves, wall isolate destabilizes (2026-05-22)

- **Run 1 (`sweep_bio_suppressor`, RTX500 4GB, `LOSS_ISOLATE=MU_LOG`)**: strong held-out all-truth recovery to **0.4720** (ep10), with high-μ best **0.4174** (ep12), but wall remains effectively pinned at **~2.593** throughout.
- **Run 2 (`sweep_wall_overcomp`, P2200 5GB, `LOSS_ISOLATE=MU_LOG_WALL`)**: early all-truth improves to **0.6115** (ep4) but then degrades; wall gets **worse** than baseline to **~3.06** and high-μ drifts to **~1.13**.
- **Gate evidence**: overcomp wall-isolate run rapidly floor-clamps gates (`gate_all/gate_wall/gate_clot -> 0.03`), indicating a collapsed low-capacity basin rather than productive wall specialization.
- **Interpretation**: this pair reinforces that pure wall-isolate pressure is not sufficient and can be counterproductive; `MU_LOG` remains the safer teacher objective for global fit while wall needs targeted capacity/curriculum, not isolate-only weighting.

### 52. New architecture hotfix: wall-branch feature width mismatch in detach path (2026-05-22)

- **Symptom**: immediate preflight crash on `sweep_bio_suppressor` (`RuntimeError: mat1 and mat2 shapes cannot be multiplied (16127x336 and 339x64)`), before teacher epoch 0.
- **Cause**: after extending trigger features from 16 -> 19 (`+wall_mask + adverse_shear_cue + low_shear_cue`), the wall-detached feature path still built 16-D tensors (`wall_trigger_feats` / `wall_feats_phys`), while `mu_delta_wall_head` expects `latent + 19`.
- **Fix**: align both wall feature constructors to the new 19-D trigger schema in `GNODE_Phase3.forward` (including detached cues), restoring shape parity with `mu_delta_wall_head`.
- **Status**: fixed in code; rerun the same preset pair to resume A/B testing.

### 53. First A/B after nucleation-growth rollout: better all-truth, unchanged wall, unstable high-tail tradeoff (2026-05-22)

- **A (new preset `sweep_clot_nuc_growth`, RTX500 4GB)**: best all-truth improved to **0.4854** (ep15), but wall stayed pinned at **~2.589** and high-μ at the best-all checkpoint was weak (**0.9232**). Best high-μ occurred on a different checkpoint (**0.4812** at ep14) where all-truth regressed (**0.6198**).
- **B (old baseline `sweep_bio_suppressor`, P2200 5GB)**: best all-truth **0.5220** (ep13), wall still **~2.589**, with better high-μ at best-all (**0.6200**).
- **Gate behavior contrast**: run A showed near-saturated wall gate throughout (`gate_wall≈1.0`, `gate_all≈0.41` almost flat), while run B remained floor-clamped at wall (`gate_wall≈0.06`) with broader gate_all variation; neither behavior translated to wall-logMAE movement.
- **Interpretation**: new architecture gives a meaningful global μ gain (~0.037 all-truth vs baseline) but does not solve wall, and still shows all-vs-high checkpoint tension. This is progress, but not a gate flip for wall.

### 54. New boundary presets (`sweep_hard_bc`, `sweep_decoupled_wall`) fail by memory before signal (2026-05-22)

- **Setup**: both runs used latent `320`, prior dim `4`, teacher-only objective isolate `MU_LOG`, stock env path active (`BIOCHEM_STOCK_DEFAULTS=1`) with teacher epochs `25`.
- **Observed training behavior**: both presets stayed essentially flat around the known plateau (`all ~1.48`) through available validation checkpoints; subset movement was negligible despite different wall handling strategies.
- **Failure mode**: both crashed in teacher backward (`torchdiffeq` adjoint path) with CUDA OOM (`+40-44 MiB` alloc failure) after several epochs; this occurred on both 4GB and 5GB GPUs.
- **Likely cause**: with current preset bundles, runtime still used heavy memory settings (`TBPTT_cap=12`, `DETACH_MACRO=0`) and did not expose split/gated wall diagnostics (`gate_*` reported `nan`), so the runs were dominated by adjoint memory pressure before boundary-learning hypotheses could be tested.
- **Implication**: these two presets are not yet valid scientific A/B probes in their current runtime envelope; they need an explicit 4-5GB-safe teacher profile before comparing boundary mechanisms.

### 55. Boundary A/B rerun with VRAM-safe profile completed; expected wall effects did not appear (2026-05-22)

- **Setup**: reran both presets after adding memory guardrails (`TBPTT=5`, `DETACH=1`, `ADJOINT_RK4_SUBSTEPS=8`, teacher-only). Both runs completed all 25 teacher epochs without OOM.
- **Run A (`sweep_hard_bc`) outcome**: wall stayed flat at **~2.4594** from ep00 to ep24; all-truth remained poor/flat (**1.5145 -> 1.5142**), high-μ was nearly unchanged (**0.9671 -> 0.9668**).
- **Run B (`sweep_decoupled_wall`) outcome**: no wall unfreeze (`~2.2517` throughout), while all/high improved only marginally (**all 1.4593 -> 1.4589**, high 0.9508 -> 0.9507).
- **Expectation check**: predicted signatures did not occur — no ep00 wall collapse in hard-BC and no progressive wall drop in decoupled-wall.
- **Additional signal**: both runs still show `gate_all/gate_wall/gate_clot = nan`, meaning split-gate diagnostics are not active in this preset path; `μ trainability` also reports `BIOCHEM_TRAIN_MU_ENCODER=0`.
- **Interpretation**: memory stability is fixed, but these settings still sit in the same teacher plateau regime; boundary overrides alone (without an actively trainable split/gated μ path and matching wall objective behavior) are insufficient in this stack.

### 56. Decoupled-wall rerun with split-head explicitly on still shows early plateau behavior (2026-05-22, in progress)

- **Setup**: `BIOCHEM_PRESET=sweep_decoupled_wall` on RTX500 with explicit `BIOCHEM_USE_SPLIT_MU_HEAD=1`, VRAM-safe teacher settings (`TBPTT=5`, `DETACH=1`, `ADJOINT_RK4_SUBSTEPS=8`).
- **What improved**: startup now confirms split-head optimizer groups are active (`μ split-head lrs` printed), and run remains stable (no OOM through ep10 shown).
- **What has not improved yet**: validation metrics remain nearly flat in early teacher epochs (`all 1.4805 -> 1.4803`, wall fixed at `2.4299`, high `0.9372` unchanged), with `gate_all/gate_wall/gate_clot` still logged as `nan`.
- **Interpretation**: enabling split-head alone did not immediately unlock wall dynamics in this configuration; continue run for full curve, but early trend still matches the historical plateau family.

### 57. Decoupled-wall rerun with master μ switches on: major global recovery, wall improved but still bottleneck (2026-05-22, in progress)

- **Setup**: `sweep_decoupled_wall` on RTX500 with explicit μ-path switches enabled (`USE_MU_PATH_GROUP=1`, `TRAIN_MU_ENCODER=1`, `USE_DELTA_MU_HEAD=1`) plus split-head and VRAM-safe teacher profile.
- **Immediate effect**: run escaped the old ~1.48 plateau quickly (ep00 all-truth `0.7680`, ep02 `0.5475`, ep10 best-so-far `0.3236`).
- **Wall signal**: wall error improved from `2.1440` (ep00) to `~1.43` (ep06/ep14 neighborhood), so wall is no longer fully pinned in the old ~2.4-2.6 band.
- **Tradeoff still active**: high-μ tail is unstable across checkpoints (e.g., `1.0205` ep00 -> `0.8271` ep08 -> `1.1690` ep12 -> `0.5702` ep14) while all-truth also oscillates (`0.3236` ep10 then worse at ep12/14).
- **Diagnostics note**: gates are now finite (`gate_all ~0.48`, `gate_clot ~0.47`), but `gate_wall` remains `0.000e+00` in printed μ debug.

---

## Lessons learned — μ formulation (2026-05-18)

Consolidated principles before re-introducing step-2 / corona / multitask losses.

### What μ is in this codebase

| Layer | Mechanism | Learned? |
|-------|-----------|----------|
| Baseline | Carreau from **u, v**, γ̇ (`μ_kin_baseline`) | No (physics constants) |
| Explicit gelation | `mu1_sigmoid(Mat)` + `mu2_sigmoid(FI)` | No (fixed sigmoid params; capped by `TEACHER_MU_RATIO_MAX`) |
| Learned gelation | `learned_clot_penalty(species_log1p)` | **Yes** |
| Residual | `exp(clamp(mu_delta_head(z_kin, species)))` | **Yes** (optional head) |
| Kinematic coupling | `mu_encoder(μ_nd)` → DEQ processor | **Yes** (optional) |

**μ is derived + corrected**, not predicted by a standalone μ-MLP on `x`.

### What actually moves val logMAE

1. **`BIOCHEM_LOSS_ISOLATE=MU_LOG`** — backward = `W_MuLog × L_MuLog` only; matches val metric (multi-step over TBPTT when `MU_SI_MULTI_STEP=1` and `W_MuLog>0`).
2. **μ-path optimizer group** — `TRAIN_MU_ENCODER=1`, `USE_MU_PATH_GROUP=1`, `USE_DELTA_MU_HEAD=1` for capacity.
3. **Low teacher forcing** — `TEACHER_FORCE_MIN=0`, warmup 2–4 ep so closure sees **model species**, not frozen GT chemistry.
4. **TBPTT window ≥ 4** — window=2 traps optimization near t0→t1 / preflight regime.
5. **`TEACHER_MU_RATIO_MAX` ≫ 1** — use **20–80**; cap=1.0 makes high-μ physically unreachable.

### What does *not* move μ (or misleads)

| Knob / observation | Why it fails |
|--------------------|--------------|
| `L_Data_Bio` + `L_Data_Kine` in backward | Bio collapses; steals step from rheology path |
| `MU_SI` isolate alone | Val logMAE flat despite train Huber movement |
| Step-2 joint + `W_MuLog=2` **without** isolate | Still ~1.51 on patient007 (RTX500 sweep) |
| `L_PhysTemp` add-on at ~1.51 plateau | Second-order; no unlock |
| Train `L_kine`, `L_bio` under isolate | **Diagnostic only** — not in `backward()` |
| `MU_SI` smoke / 1-anchor `SKIP_VAL` | Proves gradients, not generalization |
| patient003 + 3-vessel cap | Can show 0.5 logMAE without breaking patient007 ceiling |
| High logMAE drop + flat **`r`** | Model learns **scale**, not spatial μ pattern |

### Honest status

- **Can optimize log-μ on a favorable split** with the recipe above.
- **Cannot yet claim** step-2 teacher done on **patient007** (~1.48 band).
- **Wall** and **high-μ tail** remain weak; ep5 run traded tail for bulk.
- **Next science**: reproduce on patient007 → ablate μ-path components → widen temporal/gradient path (`DETACH=0`, longer TBPTT) → *then* add `L_Data_Kine` / species coupling one term at a time.

---

## μ formulation study plan (pre–step-2 multitask)

**Goal:** Understand and improve the **μ closure** (derived + learned path) on the **standard held-out anchor** before `L_Data_Bio`, corona, or Kendall PDE enter `backward()`.

**Runner:** [`scripts/run_biochem_mu_formulation_study.ps1`](../../scripts/run_biochem_mu_formulation_study.ps1)

**Acceptance (Phase A pass):** val `mu_log_mae` (all) **&lt; 1.2** for **2 consecutive epochs** on **patient007**, `VAL_TIME_STRIDE=10`, full anchor load. Secondary: wall logMAE trending down; high-μ not worse than ep0 by &gt; 0.1; `r` &gt; 0.25 would be a bonus.

### Phase A — Reproduce on the real val anchor (required)

| Leg | Purpose | Key deltas vs `mu_learned_only_oomsafe` |
|-----|---------|----------------------------------------|
| **A0** | Baseline transfer | Unset `MAX_LOAD_VESSELS`; same MU_LOG + μ-path; 12 ep; val every 2 ep |
| **A1** | Full TBPTT (≥8GB VRAM) | `DETACH_MACRO_STATE=0`, `TBPTT_MAX_WINDOW=6` — **OOM on 5GB P2200** ep0 backward |
| **A1s** | 5GB-safe temporal | `TBPTT=5`, `DETACH=1`, `RK4=10` — compromise before A1 |
| **A2** | High-μ headroom | `TEACHER_MU_RATIO_MAX=80` (match preflight cap) |

**Read:** If A0 snaps back to ~1.48, the 0.51 result was mostly split difficulty. If A0 drops below 1.2, mechanism is real.

### Phase B — Ablation (which part of μ matters?)

Run 8 ep each, same split as A0, `LOSS_ISOLATE=MU_LOG`:

| Leg | `USE_DELTA_MU_HEAD` | `TRAIN_MU_ENCODER` | `learned_clot` (via path group) | Question |
|-----|---------------------|--------------------|----------------------------------|----------|
| **B0** | 1 | 1 | on | Full stack (reference) |
| **B1** | 0 | 1 | on | Is delta head necessary? |
| **B2** | 1 | 0 | on | Is μ_encoder necessary? |
| **B3** | 0 | 0 | on | Explicit + learned_clot only? |
| **B4** | — | — | — | Joint `W_MuLog=2` + `W_MuSI=4` (no isolate) — script leg **B4**; only after B0 beats 1.2 on patient007 |

### Phase C — Temporal / autoregressive stress (still μ-only backward)

| Leg | Knobs | Hypothesis |
|-----|-------|------------|
| **C0** | `TBPTT=8`, `TEACHER_EPOCHS=16`, TF warmup 4 | Longer context fixes late-time μ |
| **C1** | `DETACH_MACRO_STATE=0`, `TBPTT=6` | Full TBPTT through species→μ helps wall |
| **C2** | `TEACHER_FORCE_MIN=0.2` (not 0) | Softer AR may stabilize high-μ tail vs bulk |

Log **`L_Data_Bio` in debug** under isolate — exploding bio is expected and **not** the optimized loss.

### Phase D — Coupling probes (still *not* full multitask)

Only after Phase A pass. One change per leg:

| Leg | Backward | Purpose |
|-----|----------|---------|
| **D0** | `MU_LOG` isolate + `W_MuLog=2` | Frozen reference |
| **D1** | Unset isolate; `DATA_ONLY=1`; `W_MuLog=2`, `W_MuSI=0`, **no** `L_Data_Bio` weight bump | Add `L_Data_Kine` only (species still detached from μ path if `DETACH=1`) |
| **D2** | Same + `W_MuSI=4` | Joint log + SI anchor |
| **D3** | `DATA_ONLY_PHYS_TEMP=1`, `w_pt` small | Does μ trajectory need temporal SI smoothness? |

**Stop rule:** If val logMAE rises &gt; 0.05 vs D0 for 2 epochs, revert — multitask is hurting μ.

### What we are *not* doing yet

- `BIOCHEM_PRESET=thrombus_corona` / corrector / pseudo bank
- `BIOCHEM_COMPLEXITY_STEP=3` (Kendall PDE in backward)
- Overnight production schedules

---

## Code / architecture backlog (μ)

Ordered by impact:

1. ~~**`L_mu_log`** on all TBPTT timesteps~~ — **done** (`_anchor_mu_si_and_log_losses`, `W_MuLog`, `LOSS_ISOLATE=MU_LOG`).
2. **Multi-step `L_MuSI`** (not only `pred_final`) — partial via `MU_SI_MULTI_STEP`; wall/high-μ weighting still open.
3. **Preflight** uses `BIOCHEM_TEACHER_MU_RATIO_MAX`.
4. **`BIOCHEM_GELATION_USE_MODEL_SPECIES`** — decouple μ gelation from TF-injected GT species.
5. **Rheology-only optimizer group** (`learned_clot_penalty`, `mu_encoder`; teacher currently `freeze_lora=True`).
6. **`BIOCHEM_FAST_MU_PROBE=1`** preset: **`SKIP_VAL=1`** or tiny dev graph — not “stride=10 ⇒ fast val” on patient007.

**Increasing complexity order (do not skip):** (A) **μ formulation study** (patient007, MU_LOG + ablations) → (B) widen TBPTT / `DETACH=0` under μ-only backward → (C) joint **step-2** one term at a time (`DATA_KINE` then `MU_SI`) → (D) **step 2.5** `L_PhysTemp` → (E) corrector + optional corona *flags* (not full preset until validated) → (F) **step 3** multitask only if stable.

---

## Recommended run profiles

### μ formulation study (primary — use script)

```powershell
# Phase A: reproduce on patient007 (full anchors)
.\scripts\run_biochem_mu_formulation_study.ps1 -Phase A -Leg A0

# Phase B ablation (after A0 pass or informative fail)
.\scripts\run_biochem_mu_formulation_study.ps1 -Phase B -Leg B1

# List legs
.\scripts\run_biochem_mu_formulation_study.ps1 -ListLegs
```

See **μ formulation study plan** above for leg definitions and acceptance criteria.

### Fast μ probe (gradient sanity only — not generalization)

```powershell
.\scripts\run_biochem_mu_smoke_fast.ps1 -LossIsolate MU_LOG -UseDeltaMuHead -TeacherEpochs 3
```

### Fast μ probe (held-out val, short)

```powershell
$env:BIOCHEM_STOCK_DEFAULTS = "1"
$env:BIOCHEM_SKIP_PRETRAIN = "1"
$env:BIOCHEM_REUSE_LAST_PRETRAIN = "1"
$env:BIOCHEM_STOP_AFTER_TEACHER = "1"
$env:BIOCHEM_NO_TEACHER_DEFAULTS = "1"
$env:BIOCHEM_LOSS_ISOLATE = "MU_LOG"
$env:BIOCHEM_MU_LOG_ANCHOR_WEIGHT = "2.0"
$env:BIOCHEM_MU_SI_ANCHOR_AUX_WEIGHT = "0.0"
$env:BIOCHEM_TRAIN_MU_ENCODER = "1"
$env:BIOCHEM_USE_MU_PATH_GROUP = "1"
$env:BIOCHEM_USE_DELTA_MU_HEAD = "1"
$env:BIOCHEM_TEACHER_MU_RATIO_MAX = "80.0"
$env:BIOCHEM_TEACHER_FORCE_MIN = "0.0"
$env:BIOCHEM_TEACHER_TF_WARMUP_EPOCHS = "4"
$env:BIOCHEM_VAL_TIME_STRIDE = "10"
$env:BIOCHEM_TEACHER_VAL_EVERY = "2"
$env:BIOCHEM_TEACHER_EPOCHS = "12"
$env:BIOCHEM_TBPTT_MAX_WINDOW = "4"
$env:BIOCHEM_NO_TEACHER_DEFAULTS = "1"
# Do NOT set MAX_LOAD_VESSELS for patient007 val
```

### Step-2 teacher (next milestone after μ probe)

```powershell
Remove-Item Env:BIOCHEM_LOSS_ISOLATE -ErrorAction SilentlyContinue
$env:BIOCHEM_LOSS_DATA_ONLY = "1"
$env:BIOCHEM_MU_SI_ANCHOR_AUX_WEIGHT = "8.0"
$env:BIOCHEM_TEACHER_FORCE_MIN = "0.3"
$env:BIOCHEM_NO_TEACHER_DEFAULTS = "0"   # teacher defaults OK
$env:BIOCHEM_TEACHER_MU_RATIO_MAX = "80.0"
$env:BIOCHEM_VAL_TIME_STRIDE = "10"
```

### Corona preset (experimental — not recommended)

```powershell
# Unvalidated bundle; see "Experimental presets" section above.
.\scripts\run_biochem_thrombus_corona.ps1
# Only consider after joint step-2 teacher stable on patient007 (not MU_LOG isolate only).
```

### Step 3 multitask (not yet)

```powershell
$env:BIOCHEM_COMPLEXITY_STEP = "3"
$env:BIOCHEM_STOCK_DEFAULTS = "0"   # or explicit env
# Expect BIOCHEM_LOSS_DATA_ONLY=0
```

---

## Run log (append rows)

| Date | Config summary | Val logMAE (all) | Wall | r | Notes |
|------|----------------|------------------|------|---|--------|
| 2026-05 | Full teacher, μ cap 80, 24ep, TF high | ~1.49–1.58 | ~2.4 | ~0.2–0.4 | L_bio ↓, μ flat |
| 2026-05 | Phase A MU_SI isolate, TF≈1, win=2 | ~1.59 flat | ~2.29 | ~0.21 | Capacity fail at TF=1 |
| 2026-05 | Low TF MU_SI ep0→1, stride=1 val | 1.474→1.467 | 1.98→1.88 | 0.40→0.43 | **μ moving**; val 35 min/ep |
| 2026-05 | MU_SI isolate, TF decay, TBPTT=2, stride=10 val | ~1.489→1.488 | ~2.25 | ~0.357 | **Flat**; L_tot ~4.29e-3 stuck; val still ~34 min/ep |
| 2026-05-16 | thrombus_corona, W_MuSI=8 W_MuLog=2, TBPTT=6 cur=1, TF=1 teacher+corr ep0–10 | teacher **flat** ~1.484; corr ep0→17 **1.569→1.548** | ~2.54→2.57 | 0.28→0.22 | Teacher μ cap default **1.0** (no `TEACHER_MU_RATIO_MAX`); pseudo_w=0; TBPTT start_idx=0 → early-time windows |
| 2026-05-18 | RTX500 step-2 sweep, teacher-only (`STOP_AFTER_TEACHER=1`): baseline MU_SI isolate vs S2 joint (`W_MuLog=2`) vs S2.5 `+PhysTemp` | **1.5138 / 1.5132 / 1.5128** | ~2.39 | ~0.35 | No material μ movement; S2.5 best but tiny gain; high-μ tail improved transiently only |
| 2026-05-18 | Quadro repro run, teacher-only: `P_repro_lowTF_earlywin_MU_SI` and `B_MU_LOG_earlywin` | **1.4799 / 1.4805** | ~2.43 | ~0.39 | Better than ~1.51 plateau but not better than prior ~1.4666 best; 1-anchor O1/O2 legs not comparable (`SKIP_VAL=1`) |
| 2026-05-18 | RTX500 smoke triad (`MU_LOG`, `MU_SI`, `MU_LOG+delta_head`), 1-anchor skip-val | n/a (skip val) | n/a | n/a | `L_Back` decreases (signal flow OK), but train=val same file + no held-out val ⇒ optimization sanity only |
| 2026-05-18 | RTX500 `P_repro_lowTF_earlywin_MU_SI` repeat (5 anchors, teacher-only) | **1.4860 → ~1.4861–1.4867** | ~2.242 | ~0.92 | ~0.383 | Strong ep0 baseline but flat thereafter; no reproducible epoch-wise μ improvement yet |
| 2026-05-18 | Quadro `mu_learned_only_oomsafe`: `MU_LOG` isolate, μ-path+delta head, TF→0, `DETACH=1`, 3 vessels, val **patient003** | **1.41→0.51** (ep5 best) | **1.97→1.42** | high **0.85→0.95** | **0.11→0.14** | First strong epoch-wise μ drop; tail worsened ep5; reproduce on patient007 |
| 2026-05-18 | Study **A0** (`mu_study_P_A_A0`): full anchors, patient007 val, MU_LOG+μ-path, 12ep, `DETACH=1`, TBPTT=4 | **1.28→0.44** (ep8 best) | **2.13→1.82** | high **0.89→1.43** | **0.28→0.37** (ep8) | **Phase A pass** (&lt;1.2 ep6+8); ep4 spike 1.04; ep10–11 drift; wall stuck |
| 2026-05-18 | Study **A1** (`DETACH=0`, TBPTT=6, P2200 5GB) | n/a | n/a | n/a | n/a | **CUDA OOM** ep0 backward (ODE adjoint); use **A1s** or **A2** on 5GB |
| 2026-05-18 | Marathon **I1** `MU_LOG` (RTX500, laptop A) | **0.49** ep3 (best) | 1.75 | 0.23 | high 1.54 | 8 ep; late val ~0.51; ~69 min/leg |
| 2026-05-18 | Marathon **I2** `MU_SI` | **0.44** ep3 | 1.90 | 0.34 | high 1.46 | 5 ep; train `L_MuSI` tiny but val μ moves |
| 2026-05-18 | Marathon **I3** `DATA_BIO` | **~1.47** flat | 2.08 | 0.18 | — | Confirms bio ⊥ val μ |
| 2026-05-18 | Marathon **I4** `DATA_KINE` | **0.48** ep3 | 1.91 | 0.22 | high 1.40 | μ_nd in kine loss moves val μ |
| 2026-05-18 | Marathon **J1** joint step-2 partial | **0.48** ep3 | 1.76 | 0.24 | high 1.52 | `L_Data_Bio` in backward; not beat isolate |
| 2026-05-18 | Marathon **J2** joint + `W_MuSI=4` | n/a | n/a | n/a | n/a | **Crash** ep0 `boundary_flux` mask/`flow_hint` |
| 2026-05-18 | Marathon **I5** `PHYS_TEMP` (P2200, laptop B) | **1.36** ep4 | 2.16 | 0.27 | — | Train `L_PhysTemp`↓; μ second-order |
| 2026-05-18 | Marathon **I6** `ADR_F` | **~1.48** flat | 2.16 | 0.27 | — | PDE residual alone does not fit μ |
| 2026-05-18 | Marathon **T1** `MU_LOG` TBPTT=5 | **0.47** ep3 | 1.88 | 0.36 | high 1.38 | |
| 2026-05-18 | Marathon **T2** `MU_LOG` TBPTT=6 | **0.40** ep6 | 1.81 | 0.40 | high 1.46 | **Best marathon**; 7 ep |
| 2026-05-18 | Marathon **J3** `MU_LOG`+phys_temp flag (B) | (in progress) | — | — | — | `LOSS_ISOLATE=MU_LOG` ⇒ PhysTemp not in backward |
| 2026-05-19 | Overnight A teacher-only (`overnight_step2`, `PhysTemp=0`, TBPTT=6, `DETACH=1`, `W_MuLog=2`, 18ep) | **0.3868** (ep17) | **1.7183** | **0.335** | high **1.3558** | New best on patient007; 194 min |
| 2026-05-19 | Overnight B teacher-only (`overnight_step2` + `DATA_ONLY_PHYS_TEMP=1`, TBPTT=6, `DETACH=1`, `W_MuLog=2`, 18ep) | **0.4081** (ep12) | **1.7618** | **0.414** | high **1.4153** | PhysTemp variant underperforms A on all/wall/high-μ; 235 min |
| 2026-05-19 | Laptop A architecture sweep A0-A4 (`MU_LOG` isolate, TBPTT=6, 8ep, `delta1` except A4 `delta0`) | **A3 0.4756** (best); A1 0.4911; A0 0.5027; A2 0.5090; **A4 1.4548** | best wall **1.7086** (A3) | best all-r **0.383** (A2); A3 r **-0.105** | high best **1.0677** (A0) | `delta0` collapses (~1.45); compact `lat192` legs are much faster (~69m) than `lat256` (~92-95m) with similar all-truth logMAE |
| 2026-05-19 | Laptop B architecture sweep B0-B4 (`MU_LOG` isolate, TBPTT=6, 8ep, wide and prior variants) | **B1 0.4738** (best); B0 0.4740; B3 0.4743; B2 0.4794; **B4 1.4453** | best wall **1.7476** (B2) | best all-r **0.340** (B0); B2/B4 near zero or negative | high best **1.0463** (B4, despite bad all) | Width/prior changes are minor vs `delta0/delta1` switch; wide legs cost more time (~143-145m) for tiny or no gain vs non-wide |
| 2026-05-20 | Teacher max-complexity preset (`teacher_max_complexity`, step-3 multitask, teacher-only, Quadro P2200; TBPTT=8, `DETACH=0`, `W_MuSI=8`, `W_MuLog=2`) | **1.5116** (best, ep6) | **2.4279** | **0.395** | high **0.9148** | Failed run for μ learning: pervasive bio-grad cap skips every epoch (L2 >> 5000), val μ flat; preset also overrode CLI `-TeacherEpochs 30` to 24 |
| 2026-05-20 | Viscosity baseline preset (`teacher_visc_baseline`, teacher-only step-2, warm-start, TBPTT=6, `DETACH=1`, `W_MuSI=2`, `W_MuLog=2`, `W_MuLogWall=2.5`, `W_MuLogHigh=1.5`) | **0.5418** (best, ep6) | **2.0983** | **0.401** (best epoch) | high **0.5961** (best late, ep17) | Fast early gain then degradation (ep16-17 all-truth **0.90/0.85**); wall remains weak; useful ablation baseline but below current best (~0.39-0.40) |
| 2026-05-20 | Dual-run A (Quadro): baseline script with aggressive wall/high CLI (`MuLogWall=2.8`, `MuLogHigh=1.6`, target `DETACH=0`, TBPTT=5) | **0.5196** (best, ep14) | **2.0581** | **0.405** | high **0.9014** | Better all-truth than run B; logs still show runtime `DETACH_MACRO=1` and `W_MuSI=8.0` (preset override), so this is partially confounded |
| 2026-05-20 | Dual-run B (RTX500): baseline script with milder wall/high CLI (`MuLogWall=1.8`, `MuLogHigh=0.8`, early-stop 0.55) | **0.5398** (best, ep12) | **1.9456** | **0.446** | high **0.9426** | Better wall + `r`, slightly worse all-truth; early stop prevented late drift; same preset-override confound (`W_MuSI=8.0`, `DETACH=1`) |
| 2026-05-20 | SAFEVAL run 1 (Quadro): explicit stock env, wall-focused (`MuLogWall=2.6`, `MuLogHigh=1.0`), `VAL_STRIDE=20`, `VAL_EVERY=4`, early-stop 0.55 | **0.5249** (best, ep8) | **2.0795** | **0.402** | high **0.9621** | Stable completion (no val hang), but weaker than run 2 on all-truth and wall |
| 2026-05-20 | SAFEVAL run 2 (RTX500): explicit stock env, global-stable (`MuLogWall=1.4`, `MuLogHigh=0.6`), `VAL_STRIDE=20`, `VAL_EVERY=4`, early-stop 0.52 | **0.5055** (best, ep8) | **1.9687** | **0.419** | high **0.9978** | Best in this baseline family so far; improves all-truth and wall vs run 1, but high-μ tail still lags |
| 2026-05-20 | VISC_V3 `TAIL_RECOVERY` (RTX500): explicit stock env, teacher-only, `MuLogWall=1.4`, `MuLogHigh=1.2`, TBPTT=6, `DETACH=1`, early-stop target 0.52 | **0.5153** (best, ep12) | **1.9728** | **0.443** | high **0.9655** | Hit early-stop threshold; stronger than paired wall-push on all/wall/r, still above global best |
| 2026-05-20 | VISC_V3 `WALL_PUSH` (P2200, in progress): explicit stock env, teacher-only, `MuLogWall=2.2`, `MuLogHigh=0.6`, TBPTT=6, `DETACH=1`, target 0.52 | **0.5289** (best so far, ep8) | **2.0814** | **0.402** | high **0.9874** | Did not hit target yet; val drift after ep8 (0.5360 ep12, 0.5395 ep16), wall remains worse than Run 1 |
| 2026-05-20 | V4 `global_plus` first try (RTX500 4GB): latent320/prior2, TBPTT=6, RK4=8, early-stop 0.50 | n/a (failed pre-epoch) | n/a | n/a | n/a | **OOM** in ODE adjoint/GAT path before ep0 val; prompted switch to 4GB-safe defaults in script |
| 2026-05-20 | V4 `global_plus` safe rerun (RTX500 4GB): latent256/prior2, TBPTT=5, RK4=6, `W(MuLog/MuSI/Wall/High)=2.0/2.0/1.6/1.4`, target 0.50 | **0.5030** (best, ep8) | **1.9661** | **0.432** | high **0.9495** | Early strong checkpoint then unstable late drift (ep16 0.7782, ep20 1.3298); indicates optimizer/AR stability issue rather than capacity floor |
| 2026-05-20 | V4 `high_mu_only` (P2200, in progress to ep12): latent320/prior4, isolate `MU_LOG_HIGH`, `W_high=3.0` | **0.9962** (best so far, ep4) | **2.0248** | **0.373** | high **0.5822** | Confirms high-tail can be learned in isolation, but all/wall stay poor; use as curriculum signal, not standalone objective |
| 2026-05-21 | V4 `global_long_stable` (RTX500 4GB): 64ep, latent256/prior2, TBPTT=5, RK4=6, LR 1e-3, μ-path LR mult 0.65, TFmin 0.10 | **0.5068** (best, ep30) | **1.9507** | **0.441** | high **0.9062** | More stable than earlier collapse runs but still drifts late (all ~0.95–1.08 by ep63); high-tail improves while all/bulk regresses |
| 2026-05-21 | V4 `tail_bridge_long` (P2200 5GB): 64ep, latent320/prior4, TBPTT=6, RK4=8, LR 8e-4, μ-path LR mult 0.50, `W(MuLog/MuSI/Wall/High)=1.2/0.8/1.6/2.8` | **0.5184** (best, ep9) | **2.0761** | **0.434** | high **0.9290** | Tail emphasis improved late high-μ (to ~0.42) and high-tail r (~0.74) but did not improve all-truth or wall on patient007 |
| 2026-05-21 | V6 `carreau_tail_split_4g` (RTX500 4GB): split-head+gate, wall=0, Pareto on, trigger floors 0.8/0.4 | **~1.5194** (best so far, ep3) | **~3.22** | **0.374** | high **0.8971** (ep15) | Mild tail-only gain with severe global failure; demonstrates early split-head config is over-constrained and wall-unbounded |
| 2026-05-21 | V6 `carreau_tail_split_5g` (P2200 5GB, in progress): split-head+gate, wall=0, Pareto on, trigger floors 0.8/0.4 | **~1.4784** (best so far, ep0) | **~2.51** | **-0.135** | high **~1.266** (ep3) | Poor initial basin (negative high/all correlation) with little movement; needs staged/lighter-constraint curriculum |
| 2026-05-21 | V7 `carreau_tail_stageA_diag_4g` (RTX500 4GB, in progress): split-head, wall=0, lighter floors, stronger tail/gate LR, boundary loss on | **~0.8922** (best so far, ep21) | **~2.48** | **~0.288** | high **~0.5016** | Large recovery vs V6; validates tail pathway movement and loss signal, but global still far from best teacher band |
| 2026-05-21 | V7 `carreau_tail_stageAB_5g` (P2200 5GB, early): split-head with staged A->B schedule, wall=0, boundary loss on | **~0.8694** (best so far, ep6) | **~2.24** | **~0.421** | high **~0.4761** | Early trajectory is strongly improved vs V6 and better than 4G run at same phase; continue to Stage B to test global recovery |
| 2026-05-21 | V8 `carreau_tail_stageAB_wall_4g` repeated x2 (RTX500 4GB): split-head staged run with Stage-B wall reintroduced, Pareto checkpoint on, 48ep | **0.7471** (saved best, ep21) | **3.2250** | **0.376** | high **0.4778** | Two runs reproduced nearly identical curves; later all-truth minima (e.g. 0.5694 ep36) sacrificed high-μ and still did not fix wall; no improvement vs prior global best |
| 2026-05-21 | V8 `carreau_tail_stageAB_wall_4g` replay (RUN 1, RTX500 4GB): same profile/seed path as prior V8, 48ep | **0.7471** (ep21 checkpoint) | **3.2250** | **0.376** | high **0.4778** | Confirms deterministic replay of prior V8 curve/checkpoint selection; no new gain on all/wall |
| 2026-05-21 | V7 `carreau_tail_stageAB_5g` continuation (RUN 2, P2200 5GB to ep35): staged split-head A→B, wall=0, Pareto on | **0.7121** (ep30) | **2.0834** | **0.445** (best all-r ~0.471 ep21) | high **0.4447** (ep15) | Significant improvement vs early V7 and V6 basins; still tail-first compromise and below global best (~0.39-0.40) |
| 2026-05-21 | V9 `walltail_arch_v1_4g` (RTX500 4GB, ongoing to ep36 shown): split-head + wall-delta branch, staged wall-on, Pareto on | **0.5778** (ep30 best so far) | **3.0722** | **0.433** | high **0.3837** (ep27) | Strong all/high gains but wall remains catastrophic; late instability after ep30 (ep33 all=1.1755) |
| 2026-05-21 | V9 `walltail_arch_v1_5g` (P2200 5GB, ongoing to ep15 shown): split-head + wall-delta branch, staged wall-on, Pareto on | **0.7744** (ep3 best so far) | **2.4378** | **0.285** | high **0.4350** (ep9) | Highly non-monotonic early dynamics; wall modestly better than 4G but all-truth unstable; motivates smoother stage transition + stronger wall gating |
| 2026-05-22 | Sweep 4G leg A (`carreau_tail_stageAB_wall_4g`, completed 48ep): split-head staged run, wall reintroduced in Stage B, Pareto on | **0.5738** (ep47) | **1.9561** | **0.395** | high **0.5055** | Best wall result in this sweep; stable late improvement and strong global recovery |
| 2026-05-22 | Sweep 4G leg B (`walltail_arch_v2_long_4g`, completed 66ep): wall-delta + smooth stage transition | **0.5506** (ep54) | **3.0708** | **0.434** | high **0.3513** (ep48) | Best all/high on 4G but wall remained poor; objective tradeoff persists |
| 2026-05-22 | Sweep 5G leg A (`walltail_arch_v1_5g`, completed 64ep): wall-delta staged run, Pareto on | **0.5028** (saved ep18) | **3.3940** | **0.312** | high **0.2516** | Later all-truth improved (~0.439) but high-μ worsened; Pareto kept early checkpoint |
| 2026-05-22 | Sweep 5G leg B (`walltail_arch_v2_long_5g`, in progress / pasted partial): startup + early epochs only | n/a | n/a | n/a | n/a | Await full run completion before ranking against leg A |
| 2026-05-22 | Step-2 isolate smoke (`sweep_bio_suppressor`, RTX500 4GB, latent320/prior4): stock defaults on + `LOSS_ISOLATE=MU_LOG`, `W_MuSI=0`, suppressor on | **1.5155** (ep00; ep06 **1.5158**) | **2.2552** (ep00; ep06 **2.2505**) | **0.369** (ep00; ep06 **0.358**) | high **0.9019** (ep00; ep06 **0.9078**) | Wiring fix verified; metrics flat; OOM at ep07 during adjoint backward |
| 2026-05-22 | Step-2 isolate smoke (`sweep_wall_sentinel`, P2200 5GB, latent320/prior4): stock defaults on + `LOSS_ISOLATE=MU_LOG`, `W_MuSI=0`, suppressor off | **1.5096** (ep00; ep06 **1.5101**) | **2.2391** (ep00; ep06 **2.2510**) | **0.392** (ep00; ep06 **0.358**) | high **0.8968** (ep00; ep06 **0.9010**) | Same early plateau behavior as suppressor run; no clear separation yet from suppressor toggle alone |
| 2026-05-22 | Fast split-μ probe (`sweep_wall_sentinel`, P2200 5GB, latent320/prior4, updated preset with μ encoder + split head): 14ep teacher-only, TBPTT=5, `DETACH=1` | **0.5496** (ep08 best) | **2.5698** | **0.402** | high **0.8368** (best-all ckpt; high best **0.4046** ep12) | Major all-truth recovery vs prior ~1.51 plateau, but wall remains stuck and gate values collapse toward zero by late epochs |
| 2026-05-22 | Fast split-μ probe (`sweep_bio_suppressor`, RTX500 4GB, latent320/prior4, updated preset with μ encoder + split head): 14ep teacher-only, TBPTT=5, `DETACH=1` | **0.5923** (ep10 best) | **2.5887** (late spikes **~3.3988**) | **0.398** | high **0.5563** (best-all ckpt; high best **0.3182** ep13) | Suppressor run improves all/high vs old plateau but keeps wall poor; `gate_wall` pinned near zero suggests wall-branch suppression bottleneck |
| 2026-05-22 | Patched fast sentinel (`sweep_wall_sentinel`, RTX500 4GB, latent320/prior4): gate-floor architecture update active | **0.6622** (ep13 best) | **2.4937** | **0.363** | high **0.6849** | Gate collapse fixed (`gate_all/gate_wall/gate_clot` floor at 0.06), modest wall gain vs prior sentinel, but weaker all/high than prior best split-μ run |
| 2026-05-22 | Patched fast suppressor (`sweep_bio_suppressor`, P2200 5GB, latent320/prior4): suppressor wall-mix + gate floors active | **0.5758** (ep13 best) | **2.5878** | **0.399** | high **0.6040** | Best global score among patched pair; high-μ reasonable, but wall remains flat (~2.588) despite non-collapsing gates |
| 2026-05-22 | Latest patched suppressor (`sweep_bio_suppressor`, RTX500 4GB, latent320/prior4): wall-bias architecture + floor controls | **0.5055** (ep13 best) | **2.4888** | **0.375** | high **0.6268** | Strongest all-truth among newest runs; wall improves vs earlier suppressor failures but remains far above target |
| 2026-05-22 | Wall-overcomp probe (`sweep_wall_overcomp`, P2200 5GB, latent320/prior4): aggressive wall-weight + wall-gate bias/boost | **0.5682** (ep13 best) | **2.4951** | **0.370** | high **1.1207** | Gates saturate (`gate_wall≈1.0`) confirming overcomp path activation, but wall barely improves and high-μ severely regresses |
| 2026-05-22 | Latest suppressor rerun (`sweep_bio_suppressor`, RTX500 4GB, latent320/prior4): patched wall-bias/floor controls | **0.6365** (ep06 best) | **2.4953** | **0.340** | high **0.7107** | Global/high weaker than prior best suppressor run; wall still sticky near ~2.50 despite non-collapsing gates |
| 2026-05-22 | Latest overcomp rerun (`sweep_wall_overcomp`, P2200 5GB, latent320/prior4, `LOSS_ISOLATE=MU_LOG_WALL`) | **0.6640** (best-all ckpt ep10) | **2.0927** (best wall shown) | **0.424** | high **1.1330** | Wall objective can be pushed down strongly, but high-μ/global degrade — confirms tradeoff, not plumbing bug |
| 2026-05-22 | New suppressor rerun (`sweep_bio_suppressor`, RTX500 4GB, latent320/prior4) | **0.5195** (ep12 best) | **2.5906** | **0.403** | high **0.7849** | Strong all-truth recovery but wall regresses to sticky ~2.59 band; gate metrics frequently floor-clamped |
| 2026-05-22 | New overcomp rerun (`sweep_wall_overcomp`, P2200 5GB, latent320/prior4, `MU_LOG_WALL` isolate) | **0.7012** (ep02 best-all) | **2.3064** (best shown late) | **0.397** | high **1.2564** | Overcomp still demonstrates wall path activity, but this seed/hardware pairing underperforms prior wall-best and harms high-μ/global |
| 2026-05-22 | Latest suppressor rerun (`sweep_bio_suppressor`, RTX500 4GB, latent320/prior4; wall-decoupling patch active) | **0.5195** (ep12 best) | **2.5906** | **0.403** | high **0.7849** | Global fit remains strong, but wall is still stuck; episodic instability after best checkpoint (e.g., ep10/ep12 swings) |
| 2026-05-22 | Latest overcomp rerun (`sweep_wall_overcomp`, P2200 5GB, latent320/prior4; wall-decoupling patch active, `MU_LOG_WALL` isolate) | **0.5085** (best-all ckpt ep06) | **2.0868** (best shown) | **0.413** | high **0.9300** | Best wall recovery in this pair confirms wall pathway works; however, high-μ remains weak and wall improvement still plateaus above target |
| 2026-05-22 | Crash-only run (`sweep_bio_suppressor`, P2200 5GB, latent320/prior4, post nucleation-growth patch) | n/a (preflight crash) | n/a | n/a | n/a | Runtime shape mismatch in wall residual path (`16127x336` vs `339x64`) due to 16-D wall-detach features feeding 19-D wall head; fixed same day |
| 2026-05-22 | Newest Run1 (`sweep_bio_suppressor`, RTX500 4GB): teacher-only, `LOSS_ISOLATE=MU_LOG`, latent320/prior4, TBPTT=5, `DETACH=1`, 14ep | **0.4720** (ep10 best) | **2.5933** | **0.403** | high **0.6169** (best-all ckpt; high best **0.4174** ep12) | Strong global recovery and decent tail checkpoint, but wall remains locked near ~2.59; confirms persistent wall bottleneck under MU_LOG isolate |
| 2026-05-22 | Newest Run2 (`sweep_wall_overcomp`, P2200 5GB): teacher-only, `LOSS_ISOLATE=MU_LOG_WALL`, latent320/prior4, TBPTT=5, `DETACH=1`, 14ep | **0.6115** (ep04 best) | **3.0573** (ep13 shown; ~3.06 band after ep2) | **0.409** | high **1.1266** | Wall-isolate objective overcompensates and collapses gates to floor (0.03), degrading wall and high-μ despite early all-truth gains |
| 2026-05-22 | A/B post-hotfix **A** (`sweep_clot_nuc_growth`, RTX500 4GB, latent320/prior4, 16ep, `MU_LOG` isolate, nucleation-growth enabled) | **0.4854** (ep15 best) | **2.5885** | **0.403** | high **0.9232** (best-all ckpt; high best **0.4812** ep14) | Best all-truth beats paired baseline, but wall is unchanged and high-μ remains checkpoint-sensitive (all-vs-high tradeoff persists) |
| 2026-05-22 | A/B post-hotfix **B** (`sweep_bio_suppressor`, P2200 5GB, latent320/prior4, 14ep, `MU_LOG` isolate, nucleation-growth disabled) | **0.5220** (ep13 best) | **2.5890** | **0.401** | high **0.6200** | Baseline underperforms new preset on all-truth but gives better high-μ at best-all checkpoint; wall remains effectively pinned |
| 2026-05-22 | Boundary A/B **Run 1** (`sweep_hard_bc`, RTX500 4GB): latent320/prior4, teacher-only isolate `MU_LOG`, hard wall override enabled (`FORCE_WALL_MU0=1`, wall head off) | **1.4808** (ep03-06 best shown) | **2.4594** | **0.037** | high **0.9761** | Flat μ trajectory with no meaningful epoch-wise gain; crashed at teacher ep07 backward with adjoint CUDA OOM (+40 MiB alloc fail) |
| 2026-05-22 | Boundary A/B **Run 2** (`sweep_decoupled_wall`, P2200 5GB): latent320/prior4, teacher-only isolate `MU_LOG`, uncapped delta + wall decoupling | **1.4828** (ep15 best shown) | **2.2522** | **0.352** | high **0.9514** | Also flat around ~1.48 despite longer survival; crashed at/after ep15 with adjoint CUDA OOM (+44 MiB alloc fail); run not completed to 25 epochs |
| 2026-05-22 | Boundary A/B rerun **Run 1** (`sweep_hard_bc`, RTX500 4GB, VRAM-safe profile): `TBPTT=5`, `DETACH=1`, teacher-only 25ep | **1.5142** (ep24) | **2.4594** | **0.036** | high **0.9668** | Completed without OOM; expected hard-BC wall collapse did not occur (wall flat across all vals) |
| 2026-05-22 | Boundary A/B rerun **Run 2** (`sweep_decoupled_wall`, P2200 5GB, VRAM-safe profile): `TBPTT=5`, `DETACH=1`, teacher-only 25ep | **1.4589** (ep24) | **2.2517** | **0.353** | high **0.9507** | Completed without OOM; no wall unfreeze trend and only tiny all/high drift within same plateau family |
| 2026-05-22 | Decoupled-wall split-head rerun (RTX500 4GB, **in progress** through ep10): `sweep_decoupled_wall`, `USE_SPLIT_MU_HEAD=1`, `TBPTT=5`, `DETACH=1` | **1.4803** (ep10 best so far) | **2.4299** | **0.395** | high **0.9372** | Stable/no OOM and split-head LR groups are active, but early metrics remain flat; wall has not unfrozen yet |
| 2026-05-22 | Decoupled-wall rerun with master μ switches (RTX500 4GB, **in progress** through ep14): `sweep_decoupled_wall` + `USE_MU_PATH_GROUP=1` + `TRAIN_MU_ENCODER=1` + `USE_DELTA_MU_HEAD=1` + split-head | **0.3236** (ep10 best so far) | **1.4307** (best shown) | **0.420** (peak shown ep00; ~0.258-0.358 later) | high **0.5702** (best shown ep14; unstable) | Strong escape from plateau and meaningful wall improvement; still non-monotonic with all/high tradeoff and `gate_wall` printed as zero |

---

## References

- Module header: `src/training/train_biochem_corrector.py` (presets, complexity steps).
- Project overview: [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md).
- Corona script (experimental): `scripts/run_biochem_thrombus_corona.ps1`.
- Comprehensive μ script (experimental): `scripts/run_biochem_comprehensive_mu.ps1`.
- Teacher-best checkpoint (after teacher stage): `outputs/biochem/biochem_teacher_best.pth` — load in viz via `python -m src.evaluation.visualize_pipeline` (prefers this over `biochem_best_bio.pth`).
