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

---

## References

- Module header: `src/training/train_biochem_corrector.py` (presets, complexity steps).
- Project overview: [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md).
- Corona script (experimental): `scripts/run_biochem_thrombus_corona.ps1`.
- Comprehensive μ script (experimental): `scripts/run_biochem_comprehensive_mu.ps1`.
- Teacher-best checkpoint (after teacher stage): `outputs/biochem/biochem_teacher_best.pth` — load in viz via `python -m src.evaluation.visualize_pipeline` (prefers this over `biochem_best_bio.pth`).
