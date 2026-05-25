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
| Preflight μ (train anchors, t0→t1) | median logMAE ≲ 2.5 | **Partial** | **K1/K0** ~1.45; **K2** explicit gelation **5.77** (§91) — triggers flood IC |
| Val μ (held-out anchor, e.g. patient007) | improve / stabilize logMAE | **Partial** | **K1** Δμ+`DATA_KINE` **0.464** (§90); **K2** step-3 multitask+gelation **4.22** ep9 (§91, **regress**); sentinel **0.294**; visc3h **0.408** |
| Val spatial correlation `r` | ≳ 0.5+ stable | **Partial** | Marathon T2 ep6 **r≈0.40**; bulk **r** often negative; high-μ **r** can be positive while all-truth **r** low |
| Viz rollout health (t0 \|u\|, clot channel) | t0 \|u\| ≳ 1.0; localized clot | **Fail** | **K2** (§91): **clot_frac=1.0**, **μ₂=80** domain-wide, score **~18**; **K1** **t0\|u\|≈0.61**, **clot_frac=0**; **K0** **~0.4**; still **< COMSOL** on t0 speed |
| Wall μ logMAE | ≲ 1.5 | **Partial** | Fair sweep (2026-05-23): **`sweep_wall_sentinel` ep17 all **0.3185** / wall **1.5479** with **`gate_wall=1.0`** (train); fair baseline ep24 wall **1.6753** / all **0.4951** but **`gate_wall=0`**; `sweep_free_wall_a` ep33 all **0.3422** best-all, wall still **~1.9–2.3** |
| `L_bio` on anchors | Decrease without μ stall | **Pass** | **I3** `DATA_BIO` isolate: train `L_bio`↓, val μ **flat ~1.47** |
| Phase A: `MU_SI` isolate, TF≈1 | Val logMAE drops | **Fail** | Flat ~1.59 (old config, no μ-path / high TF) |
| Phase B: `MU_SI` + low TF + μ-path | Val logMAE drops | **Pass** | Marathon **I2** best **0.44** ep3 (same recipe as MU_LOG) |

### Distance to full run (honest)

- **Step-2 teacher “done”**: **Interim pass on patient007** — **K1** (§90): **0.464** ep11 (`DATA_KINE`, Δμ, no explicit gelation); **K0** **1.471** flat. **Do not** stack step-3 multitask + raw explicit gelation on K1 ckpt without stabilization — **K2** (§91) **4.22** val / **clot_frac=1**. Sentinel **0.294**; visc3h **0.408**; health10h **S0** **0.451**. Corrector not started.
- **Corrector + optional spatial priors** (corona *components*, not preset): only after joint step-2 stable; corona preset itself **unvalidated**.
- **Step 3 (multitask backward)**: **In progress** — **K2** `COMPLEXITY_STEP=3`, `LOSS_DATA_ONLY=0`, explicit gelation, OomSafe **12ep complete** on RTX 500 4GB (no OOM); val all **5.58→4.22** (still **>> K1 0.464**); train **`L_tot` ~700–1700** (Kendall dominates); preflight **5.77**. Next: `GELATION_PRIOR_GATE=1` and/or cap μ₂ / staged re-enable gelation; keep **DATA_KINE** or **MU_LOG** until val **<1.0** before full PDE sum.
- **Overnight / production**: Run only after fast probes pass with `VAL_TIME_STRIDE=10`; confirm once with `stride=1`.

**We are roughly at: μ formulation validated on patient007** (MU_LOG / MU_SI / DATA_KINE isolates + TBPTT=6 all reach **~0.40–0.49** val logMAE, and latest spatial-decay run reached **0.3016**) **with subset caveats** (wall can reach ~**1.47** in geometry-isolate mode but is not robust; high-μ tail still trades off against wall in several checkpoints, bulk **r** weak/inconsistent). Next: finish **J2**, confirm **J3** (laptop B), then step-2 joint without isolate; not at corona / step 3.

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

Report per run: `outputs/reports/training/biochem/<run_id>/run.jsonl` (`meta` / `val` / `end` events). Cross-run index: `outputs/reports/training/biochem/runs_index.jsonl`.

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

### 57. New A/B wall controls split the tradeoff: geometry-isolate improves wall, spatial-decay improves all/high (2026-05-22)

- **Run A (`sweep_free_wall_a` + `BIOCHEM_WALL_HEAD_ISOLATE_GEOM=1`, RTX500 4GB)**: best all-truth **0.4547** (ep18), wall **1.5478** at best-all and best wall **1.4669** (ep21), high-μ degrades late (to **~1.49** by ep24).
- **Run B (`sweep_free_wall_b` + `BIOCHEM_WALL_SPATIAL_DECAY=1`, P2200 5GB)**: best all-truth **0.3016** (ep24) with strong high-μ **0.6915**, but wall remains weaker (**2.1246**).
- **Expectation check**: partially met — wall/clot conflicts are now more clearly separable by configuration, but no single setting simultaneously "nails" boundary and high-μ tail.
- **Gate signal caveat**: `gate_wall` remains numerically tiny (~`1.9e-22`) in both runs after startup, so improvements are likely coming from branch-feature effects and loss weighting, not sustained wall-gate activation.

### 58. Decoupled-wall rerun with master μ switches on: major global recovery, wall improved but still bottleneck (2026-05-22, in progress)

- **Setup**: `sweep_decoupled_wall` on RTX500 with explicit μ-path switches enabled (`USE_MU_PATH_GROUP=1`, `TRAIN_MU_ENCODER=1`, `USE_DELTA_MU_HEAD=1`) plus split-head and VRAM-safe teacher profile.
- **Immediate effect**: run escaped the old ~1.48 plateau quickly (ep00 all-truth `0.7680`, ep02 `0.5475`, ep10 best-so-far `0.3236`).
- **Wall signal**: wall error improved from `2.1440` (ep00) to `~1.43` (ep06/ep14 neighborhood), so wall is no longer fully pinned in the old ~2.4-2.6 band.
- **Tradeoff still active**: high-μ tail is unstable across checkpoints (e.g., `1.0205` ep00 -> `0.8271` ep08 -> `1.1690` ep12 -> `0.5702` ep14) while all-truth also oscillates (`0.3236` ep10 then worse at ep12/14).
- **Diagnostics note**: gates are now finite (`gate_all ~0.48`, `gate_clot ~0.47`), but `gate_wall` remains `0.000e+00` in printed μ debug.

### 59. Geom-blend + spatial-decay retest became bimodal: strong all-truth possible, wall collapses by regime (2026-05-22)

- **Setup**: both laptops used `sweep_free_wall_b` with new controls (`WALL_HEAD_GEOM_BLEND=0.35`, `WALL_GATE_MIN=0.05`, `WALL_SPATIAL_DECAY=1`, `DECAY_FACTOR=7.0`, `DECAY_FLOOR=0.05`), teacher-only 25 ep, `MU_LOG` isolate, latent320/prior4, `TBPTT=5`, `DETACH=1`.
- **Run 1 (RTX500 4GB)**: best all-truth **0.5296** (ep15), but wall stayed poor (**3.8510** at best-all; many vals around **6.19**) and gate values collapsed to near-zero (`gate_all -> 1e-12`, `gate_wall -> 0`).
- **Run 2 (P2200 5GB)**: best all-truth **0.3145** (ep24), but wall remained poor (**3.4304** at best-all) with regime flips to catastrophic wall (**~6.61**) and alternating `gate_wall` behavior (`1.0` vs `~1.9e-22`).
- **Expectation check**: not met for boundary stabilization — this setting can still produce strong global fit, but it does not robustly anchor wall; it introduces a two-basin dynamic (moderate-wall vs wall-collapse) instead of a stable compromise.
- **Most likely cause**: wall branch is still too sensitive to gate/suppressor dynamics under this blend/decay profile, so optimization jumps between incompatible wall regimes.

### 60. New exploratory pair confirms high-geom blend is too aggressive; lower blend is safer but still wall-limited (2026-05-22)

- **Run A (`sweep_free_wall_a`, RTX500 4GB; `GEOM_BLEND=0.80`, `WALL_GATE_MIN=0.12`, decay `3.0` floor `0.30`)**: best all-truth remained **0.7295** (ep00), wall stayed poor (**~3.84**), and training degraded afterward; `gate_wall` stayed at `0.0` while learned wall contribution exploded late (`learned` rising to `~3.9e-1`), indicating unstable compensation rather than boundary recovery.
- **Run B (`sweep_free_wall_b`, P2200 5GB; `GEOM_BLEND=0.15`, `WALL_GATE_MIN=0.10`, decay `4.0` floor `0.20`)**: best all-truth **0.4323** (ep09), wall still pinned around **3.70**, high-μ remained weak (**~0.96-1.07**), with `gate_wall` effectively near-zero throughout.
- **Expectation check**: partially met only for global fit (Run B); not met for boundary-layer objective — neither run approaches the prior best wall band (~1.47-2.12), and high-μ remains in tradeoff.
- **Interpretation**: pushing geometric dominance too hard (Run A) destabilizes the wall branch; softer blend (Run B) is more stable but still lacks a mechanism to reliably reduce wall error.

### 61. Budgeted safe sweeps remove OOM but re-enter a teacher plateau basin (2026-05-23)

- **Comp-A (RTX500 4GB, safe profile, 8 legs)** completed all legs without OOM after checkpointing + safe TBPTT defaults.
- **Best comp-A leg**: `compA_L3_S0_B8_R0` with val all **1.4599**, wall **2.2509**, high-μ **0.9404**, `r~0.354`.
- **Architecture readout**: deeper safe stack (`layers=3`) beats `layers=2`; SIREN and high Fourier bands did not help in this short detached teacher regime; LoRA-on legs often worsened all/wall.
- **Comp-B (P2200 5GB, safe profile, 8 legs)** also completed without OOM; the dense-backward failure mode is fixed by adjoint+safe profile.
- **Best comp-B basin** appears at low rheology caps (**100**) with all-truth around **~1.49**; raising cap to **500/1000** systematically degrades wall and global fit (wall often **~2.85-3.27**).
- **Interpretation**: safe profile is now operationally robust, but with `DETACH_MACRO_STATE=1` the teacher signal is heavily truncated and runs sit in the known ~1.46-1.59 plateau family; no gate flip vs prior best teacher (~0.3868).

### 62. Fair head-to-head (6 ep, warm start): geometry-isolate beats spatial-decay; runtime ~5× faster than budgeted (2026-05-23)

- **Setup**: `scripts/run_biochem_best_vs_second_30min.ps1` on **SPAGNIER only** (RTX500 4GB; both variants run sequentially on the faster laptop, not in parallel on two machines). Shared fair base: teacher-only 6ep, `MU_LOG` isolate, `TBPTT=5`, `DETACH=1`, warm-start from `biochem_post_pretrain.pth`, val every 2ep, balanced loss weights (anchor/wall/high = 1.0/2.0/2.0). Single architecture axis: **BestAllArch** = `WALL_SPATIAL_DECAY=1`; **BalancedArch** = `WALL_HEAD_ISOLATE_GEOM=1`.
- **Timing vs expectation**: script targeted **~30 min per variant** (from overnight sweep leg estimates); actual wall-clock **~3 min** (BestAllArch) + **~2 min** (BalancedArch), **~6.7 min** sequential including gap between runs (11:32:26 → 11:39:06). Cause: skipped pretrain, only 6 teacher epochs, `VAL_TIME_STRIDE=10`, and fast val passes (~5.6 s each).
- **Best checkpoint (ep04, patient007)**:
  - **BestAllArch** (`wall_spatial_decay`): all **0.6496**, wall **3.8468**, high-μ **1.1247**, `r≈-0.20`; `gate_wall=0` throughout.
  - **BalancedArch** (`wall_geom_isolate`): all **0.5259**, wall **1.6610**, high-μ **1.0205**, `r≈0.43`; `gate_wall=0` throughout.
- **Readout**: under identical detached safe training, **geometry-isolate wins on all three subsets** at best ckpt (including wall by ~2.2 logMAE units). Spatial-decay path re-enters the familiar **~3.85 wall plateau** seen in longer `sweep_free_wall_b` runs. Preflight medians: BestAll **1.6605** vs Balanced **1.4815**.
- **Caveat**: still short-horizon / detached; does not beat historical long-run bests (e.g. overnight all **0.3868**, geometry-isolate wall **1.4669** at 25ep). For a longer fair rematch, rerun at **12–18 ep** with `DETACH=0` if VRAM allows.

### 63. Wall-μ₀ override A/B: `FORCE_WALL_MU0=0` (fix) beats forced resting μ at wall (2026-05-23)

- **Setup**: `scripts/run_biochem_wall_mu0_ab_10min.ps1` on SPAGNIER (RTX500 4GB), sequential both legs, geometry-isolate best arch, 8ep warm-start, `DETACH=1`, `MU_LOG` isolate. **LearnWallMu** = `FORCE_WALL_MU0=0`; **ForceWallMu0** = `FORCE_WALL_MU0=1` (CFD strict override in `gnode_biochem.py`).
- **Timing**: **~6.3 min** total (**~3.1 min/leg**); under the ~10 min budget.
- **Best checkpoint (patient007)**:
  - **LearnWallMu**: all **0.4763** (ep06), wall **1.7743**, high-μ **0.6933**, `r≈0.36`.
  - **ForceWallMu0**: all **0.4883** (ep06), wall **2.4594** (flat every val epoch), high-μ **0.9204** at best-all ckpt, `r=0` on wall.
- **Readout**: the recommended fix is **optimal in this probe** — forcing `μ=μ₀` at walls removes the learning signal and pins wall error at the historical **~2.4594** plateau (same as `sweep_hard_bc`). Keep **`BIOCHEM_FORCE_WALL_MU0=0`**; do not enable the override for clot/wall μ work.

### 64. Removed `BIOCHEM_FORCE_WALL_MU0` override from `gnode_biochem.py` (2026-05-23)

- **Change**: deleted the CFD strict wall-μ₀ block in `GNODE_Phase3` forward (previously forced `μ=μ₀` on `mask_wall` nodes). No-slip is kinematic (u,v); wall clot viscosity must remain learnable.
- **Env**: `BIOCHEM_FORCE_WALL_MU0` is **ignored** (removed from code). `scripts/run_biochem_wall_mu0_ab_10min.ps1` is now a single-leg wall-learning probe only.

### 65. ND surface-physics A/B: opt-in fix wins all-truth, marginal wall, high-μ tradeoff at best ckpt (2026-05-23)

- **Setup**: `scripts/run_biochem_nd_surface_ab_10min.ps1` on SPAGNIER (RTX500), sequential 8ep warm-start, geom-isolate best arch, `MU_LOG` + `WALL_BIO_BLEND=0.15`, `DETACH=1`, val every 2ep. **Baseline** = legacy surface ODEs; **NdSurfaceFix** = `BIOCHEM_ND_SURFACE_PHYSICS=1` (Da=1e-4, full AP/Mas adhesion, step2t gate 12s).
- **Timing**: **~7.0 min** total (~198s + ~199s/leg).
- **Best checkpoint (patient007)**:
  - **Baseline** (ep04): all **0.7267**, wall **1.8845**, high-μ **0.6194**, `r≈0.33`.
  - **NdSurfaceFix** (ep06): all **0.4941**, wall **1.8015**, high-μ **0.9425**, `r≈0.36`.
- **Final epoch (ep07)**: Baseline **regressed** (all **1.4418**, high **1.1595**); Fix remained stable (all **0.6129**, wall **1.7434**, high **0.7687**).
- **Readout**: fix is **clearly better on primary score (all logMAE)** and more stable late-epoch; wall improves slightly at best ckpt (~0.08 logMAE) but high-μ is worse at best ckpt (0.94 vs 0.62) yet better at ep07. Not yet at long-run bests (overnight all **0.3868**). `gate_wall=0` both runs. Promoted to default in code after this A/B (see §66).

### 66. ND surface physics promoted to default; legacy wall ODE path removed (2026-05-23)

- **Code**: `biochem_wall_residual` always uses Da (`BiochemConfig.surface_damkohler=1e-4`), full AP/Mas adhesion in `R_M`/`R_Mat`, step2t gate (`surface_time_gate_s=12`), and gated Neumann fluxes. Legacy RP-only deposition path and `BIOCHEM_ND_SURFACE_PHYSICS` env removed.
- **Tests**: `TestWallSurfaceNdPhysics` in `test_biochem_physics.py` — GT Da sensitivity (~50× inflation when Da wrong), early step2t gate, AP-in-R_M formula guard. Full `test_biochem_physics.py` + `test_unit_consistency.py`: **44 passed**.

### 67. Wall-gate A/B: `sweep_free_wall_a` helps wall/high at cost of `gate_wall` still zero; preset re-applies 25 ep (2026-05-23)

- **Setup**: manual fair base on SPAGNIER (RTX500 4GB), geom-isolate, `MU_LOG` isolate, `DETACH=1`, warm-start, val stride 10. **A** = baseline (`wall/wall=2.0`, `high=2.0`, `WALL_BIO_BLEND=0.15`, intended 8ep). **B** = `BIOCHEM_PRESET=sweep_free_wall_a` + `WALL_HEAD_ISOLATE_GEOM=1` (`wall=3.0`, `high=2.5`, `BIO_SUPPRESS_WALL_ALPHA=0`; `--epochs 8` passed but **`train_biochem_corrector()` re-applies preset → 25 teacher ep**).
- **Best checkpoint (patient007)**:
  - **A (8ep)**: all **0.3729** (ep07), wall **1.7928** (ep04 best wall), high **0.5558** (ep02), `r≈0.46` at best-all.
  - **B (25ep)**: all **0.3472** (ep20), wall **1.5667** (ep20), high **0.6777** (ep18), `r≈0.22` at best-all.
- **Matched ~8ep (B ep08 vs A ep07)**: all **0.4009** vs **0.3729** (A wins); wall **1.7512** vs **1.9621** (B wins ~0.21); high **0.7087** vs **0.8292** (B wins).
- **Gate**: `gate_wall=0.000e+00` **both legs every epoch**; B drives `gate_clot` → **~0.94** late while `gate_all` ~0.48 (A). Wall gains are **not** from opening the wall transition gate.
- **Readout**: **Adopt loss-weight + longer-train recipe for wall** (B beats A on wall at ep08 and at best ckpt), but **reject “fix the shut wall gate”** as validated — same starvation signature. Re-run fair 8×8 with `BIOCHEM_TEACHER_EPOCHS=8` set **after** preset or without preset bundle epoch override.

### 68. `wall_ab_fix_8ep` retry: preset still forced 25 ep; best ep06 only; unstable after (2026-05-23)

- **Setup**: `sweep_free_wall_a` + `WALL_HEAD_ISOLATE_GEOM=1`, shell `BIOCHEM_TEACHER_EPOCHS=8` + `--epochs 8` — **preset re-apply still set 25 ep** (fixed in code via `BIOCHEM_CLI_TEACHER_EPOCHS` restore after presets).
- **Best ckpt (ep06)**: all **0.4369**, wall **1.6777**, high **0.8202**, `r≈0.49`; saved teacher-best at ep06 (`gate_wall=0`).
- **vs fair A/B baseline (8ep)**: baseline ep07 all **0.3729** / wall **1.9621** — **baseline wins global**; this run wins wall at best (~0.18) but **regresses at ep08** (all **0.9003**, wall **1.7312**). Never reached prior 25ep B best (all **0.3472**, wall **1.5667**).
- **Readout**: minimal preset-only launch is **less stable** than full fair-base A/B; do not treat as definitive 8ep loss. Re-run with full fair base + `BIOCHEM_CLI_TEACHER_EPOCHS=8` after code fix.

### 69. `wall_ab_fix_8ep_v2`: CLI epoch cap works (8ep); confounded run loses to baseline on all (2026-05-23)

- **Setup**: `sweep_free_wall_a` + geom-isolate, `--epochs 8` → **`BIOCHEM_TEACHER_EPOCHS=8` confirmed** (`BIOCHEM_CLI_TEACHER_EPOCHS` restore). **Not fair vs §67 A**: fresh AE+ODE-RXN pretrain (no warm-skip), **LoRA rank=4**, `val_every=3`, `TBPTT curriculum=1`, `workers=4` / `pin_memory=1` (stock defaults), no `WALL_BIO_BLEND` / fair loss weights.
- **Best ckpt (ep06, mu_score)**: all **0.7789**, wall **1.6271**, high **1.0421**, `r≈0.22`; `gate_wall=0`; `gate_clot→~0.74`.
- **Ep07 (not saved)**: all **0.8810**, wall **1.5403** (best wall in run), high **0.9544**.
- **vs fair A/B baseline (8ep, warm-start, fair base)**: baseline all **0.3729** / wall **1.79–1.96** — **baseline wins global by ~0.41 logMAE**; v2 best wall **1.54** @ ep07 beats baseline best wall **1.79** @ ep04 but with poor all/high and low `r`.
- **Readout**: epoch-cap fix validated; **do not** conclude preset loses — rerun with full `Set-FairBase` + `$env:BIOCHEM_CLI_TEACHER_EPOCHS='8'` only changing preset/weights.

### 70. Fair wall-gate A/B (8ep, warm-start, `Set-FairBase`): **B wins all-truth**; wall mixed; `gate_wall=0` (2026-05-23)

- **Setup**: identical fair base (geom-isolate, `MU_LOG`, `DETACH=1`, `LORA=0`, blend 0.15 on A only, val/2, 8ep CLI cap). **A** = wall/high weights 2/2; **B** = `sweep_free_wall_a` (3.0 / 2.5). Same `biochem_post_pretrain.pth` (post v2 LoRA pretrain — `L_bio~3e2`, not ~1.4e2 from first A/B).
- **Saved best ckpt (mu_score = all logMAE)**:
  - **A** ep02: all **0.7615**, wall **2.0946**, high **0.9370**, `r≈0.36`.
  - **B** ep06: all **0.5081**, wall **1.9513**, high **0.9889**, `r≈0.30`.
- **Not saved but notable**: A ep04 wall **1.5027** (all 0.8866); B ep07 all **0.5384** wall **1.8189** high **0.8214** `r≈0.39`; B ep02 high **0.6406** but all **1.1606** (spike).
- **Matched ep06**: A all **0.8880** wall **1.5472** vs B all **0.5081** wall **1.9513** — B wins global by **0.38**, A wins wall by **0.40**.
- **Gate**: `gate_wall=0` both legs; B does not open wall gate.
- **Readout**: under fair 8ep, **adopt `sweep_free_wall_a` loss weights for teacher** (clear all-truth win at best ckpt); add **wall-aware or Pareto ckpt** if wall target matters (A ep04 wall beat B saved wall). Absolute scores still above first-session A/B (0.37/0.35) — warm-start changed; optional rerun from clean post-pretrain without LoRA.

### 71. Wall3h fair epoch-ladder sweep (2026-05-23): sentinel opens `gate_wall`; baseline/`free_wall_a` do not

- **Setup**: `scripts/run_biochem_wall_gate_fair_sweep_3h.ps1` — **Arm A** (SPAGNIER RTX500): fair base (`MU_LOG`, geom-isolate, `LORA=0`, warm-start, val/2, `DETACH=1`, `CLI_TEACHER_EPOCHS` honored) + epoch ladder **8→34** on baseline, **Pareto** baseline@20ep, **`sweep_wall_sentinel`@18ep**. **Arm B** (SILKSPECTRE P2200): same fair base + **`sweep_free_wall_a`** ladder, **`sweep_free_wall_b`@20ep**, **`sweep_bio_suppressor`@18ep**. Batch **~62m + ~76m**; summary `outputs/reports/training/biochem/wall_gate_fair_sweep_3h_summary.txt`.
- **Best all-truth (saved ckpt, patient007)**:
  - **Baseline** `B0_ep34`: all **0.4239** (ep33); wall **1.9708** @ best-all ep; best wall **1.6753** ep24.
  - **`sweep_free_wall_a`** `FWa_ep34`: all **0.3422** (ep33); wall **2.1341** @ best-all ep.
  - **`sweep_wall_sentinel`** `WS_ep18`: all **0.3185** (ep17); wall **1.5479** @ best-all ep; best wall **1.5605** ep12.
- **`gate_wall` (train μ debug)**:
  - Baseline + `sweep_free_wall_a`: **`0.000e+00`** every val line (same as prior fair A/B).
  - **`sweep_wall_sentinel`**: ep00 **0.812** → ep02 **0.06** floor → ep10–17 train **`gate_wall=1.000e+00`** (val still reports wall logMAE ~1.5–1.7).
  - **`sweep_bio_suppressor`**: **`gate_wall≈6e-02`** (floor); all **0.7279** ep08 best.
- **Matched-epoch snapshots** (same fair base, warm-start):
  - **~20ep**: baseline all **0.5982** / wall **1.9737**; `FWa_ep20` all **0.4704** / wall **1.7016** — B wins global, comparable wall.
  - **~14ep**: baseline all **0.4577** (ep04); `FWa_ep14` all **0.4790** (ep12).
- **Pareto @20ep** (`B0_ep20_pareto`): saved **ep02** all **0.7504** / wall **1.5388** — Pareto did not retain later wall wins (ep24 wall **1.6753** unreachable).
- **Cause / readout**:
  - **Loss weights alone** (`sweep_free_wall_a`: wall=3, high=2.5) improve **all** vs fair baseline at long horizons but **do not** activate the wall transition gate.
  - **`sweep_wall_sentinel`** (high `MU_LOG_WALL_WEIGHT`, wall gate bias) is the **only** leg in this sweep that saturates **`gate_wall`**; wall logMAE still **~1.55** at best-all — gate open ≠ wall metric solved.
  - **Next**: wall-aware checkpoint selection (not mu_score-only); short sentinel continuation from ep17 ckpt; avoid conflating train `gate_wall` with val wall logMAE.

### 72. Gate-fix sweep (2026-05-23): Fix A / relu-add win **all** on 18ep; wall still ~1.9–2.3; bypass helps high-μ on P2200

- **Setup**: `scripts/run_biochem_gate_fix_sweep.ps1` — fair base (`MU_LOG`, geom-isolate, `LORA=0`, warm-start, val/2, 18ep, `DETACH=1`). **Arm A** (SPAGNIER RTX500): baseline, Fix A (curriculum 12ep), B (bypass 1.5), C (gate pos-init 3.0), D relu-add, D siren-add. **Arm B** (SILKSPECTRE P2200): sentinel ref, AB, AC, ABC combos. Batch **~43m + ~33m**.
- **Best all-truth (saved ckpt, patient007)**:
  - **Arm A**: **baseline 0.6227** ep16; **Fix A 0.5155** ep16; **Fix D relu 0.5091** ep17; Fix B **0.5335** ep14 (ep16 collapse 0.85); Fix C **0.5936** ep16; Fix D siren **0.5650** ep12 (ep17 regress 0.91).
  - **Arm B**: **sentinel 0.4282** ep16; **fix_ab 0.4354** ep16; fix_ac **0.4356** ep17; fix_abc **0.4752** ep10 (combo underperforms AB/AC alone).
- **Wall @ best-all ckpt**: still **~1.87–2.35** on every leg (best wall in batch **fix_ac 1.8957**); **no leg reached wall3h sentinel 1.55** or geom-isolate **~1.66** band on this 18ep schedule.
- **High-μ @ best-all**: **fix_ab 0.5258** ep16 (P2200) — best high-μ in sweep; Fix D relu **0.8945** ep17 (SPAGNIER); sentinel **1.2191** ep16.
- **Diagnostics**: printed **`gate_wall=0`** on all Arm A legs (trigger gate on wall nodes, not `_last_mu_wall_gate`). Sentinel B: **`gate_wall` 0.93 ep00 → floor ~0.06**; curriculum AB shows **`gate_wall` on wall nodes ~0.45–0.64** during ep0–7 then ~0. Fix C: **`gate_all` collapse** ep12–14 (0.12–0.17) after pos-init — hurts bulk trigger path.
- **Train signal**: Fix B **`L_tot` ~1.5–2× baseline** (bypass term in backward); watch **`W·L_MuWall_bypass`** separately from anchor μ losses.
- **Readout**: **Fix A (curriculum)** and **Fix D (relu_add)** are the best **all-truth** interventions in this fair 18ep test; **fix_ab** is the best **high-μ** trade but wall explodes. None replaces **`sweep_wall_sentinel` @ 0.3185** (wall3h, longer ladder) or **overnight 0.3868** without more epochs / wall-aware ckpt. **Next**: 24–34ep Fix A or D relu; Pareto; log **`DBG_wall_gate_mean_wall`**; try **fix_ab + sentinel wall weights** (not full ABC); wall-aware save on wall logMAE.

### 73. Gate-fix **deep 4h** (2026-05-23): **sentinel @34ep all=0.2938** on SPAGNIER; Arm B metrics on other host

- **Setup**: `scripts/run_biochem_gate_fix_deep_4h.ps1` — ladders 8→40 + @34 anchors. **Arm A** 17 legs / **174m** (SPAGNIER); **Arm B** 16 legs / **204m** (SILKSPECTRE). Parsed from **`outputs/reports/training/biochem/metrics.jsonl`** (segmented by teacher `epoch==0`; no `run_note` in JSONL — order matched sweep + `biochem_teacher_best.pth` run_note).
- **Arm A — Fix D relu ladder** (best saved all @ val): ep8 **0.579** | ep14 **0.510** | ep20 **0.554** | ep26 **0.504** | ep30 **0.580** | ep34 **0.524** | ep40 **0.484** @ep34. vs gate-fix-18ep **0.509** — **modest gain** at 40ep, noisy mid-ladder.
- **Arm A — Fix A curriculum ladder**: ep14 **0.664** | ep20 **0.740** | ep26 **0.552** | ep30 **0.329** @ep29 | ep34 **0.383** | ep40 **0.373** @ep38. **ep30 curriculum** strong (**0.329** all); wall **~1.85**, high **~0.75**.
- **Arm A — @34ep specials** (best all | wall | high): **WS sentinel preset `0.294` @ep32** | **1.50** | **0.74** | **new session best all** (beats wall3h **0.3185** @18ep); D relu **0.524** | 2.00 | 0.83; Pareto **0.564** | 2.00 | 0.77; D+A **0.581** | 1.97 | 0.72; TBPTT=6 **0.517** @ep18 | 2.01 | 0.56.
- **Arm B**: not in this repo’s `metrics.jsonl` (SILKSPECTRE run) — pull from that machine’s `outputs/reports/training/biochem/metrics.jsonl` or diary folders.
- **Checkpoint**: `biochem_teacher_best.pth` on SPAGNIER = last leg **D_relu_tbptt6** (all **0.517** @ep18), **not** batch best (**WS 0.294**).
- **Readout**: **Promote `sweep_wall_sentinel` @ 34ep** for follow-up (not fair D relu alone). Curriculum **30ep** worth a dedicated sentinel-weight run. Pareto @34 did not beat mu_score-only on all. Longer fair MU_LOG ladders alone are a weak path.

### 74. Supervised data leash (`SUPERVISED_DATA_LEASH=1`): bulk μ improves, wall trades off (2026-05-24)

- **Symptom**: `MU_LOG` isolate on sentinel reaches strong all-truth (**~0.29–0.31**) but wall stays **~1.48–1.55** with `gate_wall→1` only on long MU_LOG runs.
- **Hypothesis**: un-isolate `L_Data_Kine` + `L_Data_Bio` with `DATA_ONLY=1`, `DETACH=0`, `W_MuSI=2` — kinematic “leash” without PDE losses.
- **Result**: fix **is active** (no isolate banner; `L_Back` tracks kine+bio; `L_bio` collapses). Val all **0.223** and high-μ **0.47** @ ep14 beat sentinel; **wall 1.92** @ same ckpt (worse). Late epochs oscillate (ep22: all **0.289**, wall **1.47**, high **1.23**). Checkpoint policy still saves on **all-truth mu_score** only.
- **Next**: wall-aware save or Pareto; try leash from **post-pretrain** (not teacher-finetune) to reduce bulk overfit; consider raising kine weight vs dominant `W·L_MuLogWall` terms.

### 75. Bulk-fluid surgical lock (`BULK_FLUID_SURGICAL_FIX=1`, `CLIP_BULK=0.05`, bio suppressor floor 0): bulk subset wins, high-μ checkpoint regresses (2026-05-24)

- **Setup**: same data leash as §74 + **`BIOCHEM_DELTA_MU_LOG_CLIP_BULK=0.05`**, **`BIOCHEM_USE_BIO_GATE_SUPPRESSOR=1`**, **`BIOCHEM_BIO_SUPPRESSOR_GATE_FLOOR=0.0`** (re-applied after sentinel preset); init-from **`biochem_teacher_best_high_mu.pth`** (prior leash run); 26ep ~23m.
- **Result @ best-all ep16**: all **0.353**, bulk **0.253**, wall **2.065**, high **1.256**, **r≈0.29**. **Did not** update global high-μ ckpt (kept prior leash **0.470** @ ep14). vs leash-without-lock: all **0.353** vs **0.223**, high **1.256** vs **0.470**, wall **2.06** vs **1.92** — surgical lock **helps bulk logMAE** but **worsens high-μ** and does not fix wall. Training volatile (ep18/25 regress to **~1.42–1.45** all); one bio-grad skip ep5.
- **Readout**: bulk clamp + bio-gate suppressor is **not** a full fix for catastrophic wall; may be useful as a **stabilizer on bulk-only** objective, not as finetune on a strong high-μ teacher. Viz this run: **`biochem_teacher_last.pth`** (ep16); default **`biochem_teacher_best_high_mu.pth`** is still the earlier leash run.

### 76. Cold start + bulk surgical lock + data leash: clean prior, weak all-truth, wall still stuck (2026-05-24)

- **Setup**: deleted poisoned `.pth`; **`NoInitFromBest -ForcePretrain`** (kinematics backbone → AE ep13 → ODE-RXN ep11 → teacher); same leash + **`BULK_FLUID_SURGICAL_FIX=1`** (`CLIP_BULK=0.05`, bio suppressor floor 0); sentinel preset; 26ep ~23m (`20260524T140238Z`).
- **Preflight**: median logMAE **1.52** (healthy).
- **Result @ best-all ep12**: all **0.907**, bulk **0.922**, wall **2.195**, high **0.773**, **r≈0.24**. **High-μ ckpt** saved @ ep22: high **0.702** (all **1.185** that epoch). Late **collapse** ep18 all **1.524** (preflight-like). Final ep25: all **1.103**, wall **2.246**, high **0.751**.
- **Compare**: vs sentinel MU_LOG @34ep (**0.307** / wall **1.48**); vs warm data leash (**0.223** / **0.470** high); vs warm bulk-lock finetune (**0.353**). Cold + bulk lock **does not** reach prior tiers in 26ep; bulk subset **not** clearly safer than warm leash (bulk **0.92** vs leash bulk often **&lt;0.35** at best).
- **Readout**: deleting bad teacher weights was correct for diagnosis; **cold + surgical lock** is a **baseline builder**, not a shortcut to sentinel μ. Next: longer cold schedule and/or leash **from `biochem_post_pretrain.pth` only** (no teacher init); wall still needs dedicated objective or `gate_wall` training signal.

### 77. Cold + strict μ-freeze (`-StrictMuFreeze`, μ-path only @ teacher): bulk/all improve vs cold-only; wall numerically frozen (2026-05-24)

- **Setup**: cold start (`ForcePretrain`, no teacher init) + data leash + bulk lock + **`TRAIN_ODE=0` `TRAIN_BIO_ENC=0` `TRAIN_KIN_LORA=0` `TRAIN_BIO_DEC=0`** (22 μ-path tensors @ teacher); AE ep13 / ODE-RXN ep11; `20260524T144958Z`, ~22m teacher after pretrain.
- **Result @ best-all ep12**: all **0.571**, bulk **0.528**, wall **2.252** (identical ep2–25), high **0.959**, **r≈0.405**. **High-μ ckpt** @ ep24: high **0.593** (all **0.786**). Best bulk ep4 **0.603**. Preflight median **1.50**.
- **Compare**: vs §76 cold bulk-lock only: all **0.571** vs **0.907** (strict freeze **helps** global/bulk). vs warm leash: all **0.223**, wall **1.92**. vs sentinel @34ep: all **0.307**, wall **1.48**. Train **`L_bio~3×10²`** with `biology=0` — species terms still in `L_Back` graph but no bio/ODE DOF; do not read `L_bio↓` as learning.
- **Readout**: μ-only teacher can fit **bulk/all** from a clean pretrain better than full cold joint, but **wall logMAE is a flat line (~2.25)** — wall head not moving on val despite `W·L_MuLogWall≈7.4`. Likely need wall-aware ckpt, higher wall weight, or unfreeze wall-adjacent DOF; `gate_wall` stuck at floor **0.06**, `gate_all` collapses late.

### 78. Hard gate threshold (`BIOCHEM_MU_TRIGGER_GATE_HARD_THRESH=0.15`) on cold μ-freeze stack: wall unfreezes; all/bulk trade off (2026-05-24)

- **Setup**: §77 stack + **`TRIGGER_GATE_MIN=0`** (cleared after preset) + hard cutoff on bulk/tail `gate` and `wall_gate` / `wall_signal`; `20260524T153126Z`, ~22m teacher.
- **Result @ best-all ep16**: all **0.591**, bulk **0.523**, wall **1.868** (best wall in cold stack; was **2.25** flat in §77), high **1.202**, **r≈0.12**. Ep14: all **0.688**, wall **1.878**. Ep2: all **0.941**, wall **2.004**, high **0.672** (high-μ ckpt policy). `dbg_gate_mean_wall` **~0.99** early train, **`gate_all` ~0.038** val after ep6.
- **Compare**: vs §77 mufreeze: all **0.591** vs **0.571** (similar), **wall 1.87 vs 2.25 frozen** (hard gate **works** for val wall metric). vs warm leash: all **0.223**, wall **1.92**. vs sentinel: all **0.307**, wall **1.48**. High-μ **worse** at best-all ckpt (**1.20** vs **0.96** §77).
- **Readout**: hard threshold likely **stops center viscosity bleed** (viz should be checked: lumen not fully clotted @ late time). Val wall improved but still **~1.87**; checkpoint still all-truth score. Next: viz confirm geometry; Pareto/wall-weighted save; may need lower wall Δ clip or longer schedule.

### 79. Differentiable soft gate (sigmoid steepness 20) replaces `torch.where` hard gate: better early all/high, wall plateau returns, late bulk collapse (2026-05-24)

- **Symptom**: §78 `torch.where` gate suspected of dead gradients below 0.15; viz still showed full-domain clot concern.
- **Fix**: `g * sigmoid(20*(g-0.15))` on bulk/tail + wall gates (`BIOCHEM_MU_TRIGGER_GATE_HARD_STEEPNESS=20`); same cold stack as §77–78 (`data leash`, bulk lock, `StrictMuFreeze`, AE13/ODE11); `20260524T160923Z`, ~26ep teacher.
- **Result @ best-all ep10** (saved ckpt): all **0.758**, bulk **0.776**, wall **2.253** (back to §77 plateau, not §78 **1.87**), high **0.597**, **r≈0.40**. Ep8: all **0.851**; ep6: all **1.354**. **High-μ ckpt** @ ep14: high **0.553** (all **0.887**). **Late regression**: ep22 all **2.012**, ep24 **2.586**, ep25 **2.669**; bulk **r** **-0.44**; train `gate_clot` / `gate_all` **~0.96** from ep6 onward (saturated clot gate).
- **Compare**: vs §78 hard `where`: all **0.758** vs **0.591** @ best (soft **better** all/high), wall **2.25** vs **1.87** (soft **worse** wall). vs §77 mufreeze: all **0.758** vs **0.571** (soft **worse** than no gate at best). vs warm leash: all **0.223**. Preflight median **1.48**.
- **Readout**: restoring gradients **does not** fix wall metric in 26ep; early-stop @ ep10 would avoid saving a ckpt that **overfits bulk clot gate** then blows up. Next: viz `biochem_teacher_best_high_mu.pth` (ep10/14); try **early stop on val all**, lower gate steepness or separate wall objective; revisit §78 wall win with soft wall-only path.

### 80. Viz + gate diagnosis: soft gate on bulk ``gate`` saturates clot path; wall floors were partial red herring (2026-05-24)

- **Viz** (`biochem_teacher_best_high_mu.pth`, patient007): **t=0** Biochem **|u|≈0** vs COMSOL peak **~1.4**; late time **μ₂ trigger → 80** and total **μ ~5–6 Pa·s** fill lumen (COMSOL: trigger **~0**, μ near wall only).
- **Logs**: `gate_wall≈0`, `gate_all`/`gate_clot` **→0.96** from ep6 — soft cutoff on **bulk clot gate** does **not** suppress when `p_gate>0.15` (pass-through); §78 `torch.where` had val **`gate_all≈0.04`**.
- **`TRIGGER_GATE_MIN`**: already cleared by hard-gate hook when thresh set; **`WALL_GATE_MIN=0.08`** still applied in wall path until preset/hook fix.
- **Code fix (next run)**: `BIOCHEM_MU_SOFT_GATE_SCOPE=wall_only` (soft cutoff on `wall_gate` / `wall_signal` only); bulk clot gate uses bio suppressor + `TRIGGER_GATE_MIN=0`; optional `BIOCHEM_MU_GATE_LEARNED_TEMP=1` (`mu_soft_gate_log_temp`); sentinel preset floors **0.0**; default steepness **10**.

### 81. **visc3h** architecture sweep (8 legs, warm post-pretrain, data leash / MU_LOG): logMAE leaderboard ≠ velocity winner (2026-05-24)

- **Setup**: `scripts/run_biochem_visc_velocity_arch_sweep_3h.ps1` / `go_visc3h.ps1` — `outputs/biochem/sweep_visc_velocity_3h/`, 18ep teacher, val/2, `BIOCHEM_REUSE_LAST_PRETRAIN=1`, ~113m total (SPAGNIER).
- **Metric ranking (patient007 val, saved ckpt)**: **L5** all **0.408** high **1.15** (`MU_LOG` isolate, `DETACH=1`, `gate_clot~0.3`); **L1** all **0.917** high **0.59** (leash + **soft wall-only** gate); **L4** all **0.941** high **0.56** (kin LoRA r4); **L6** all **0.995** high **0.517** (**new global `biochem_teacher_best_high_mu.pth`**); **L7** all **0.958** high **0.57** (early-stop 0.65 never fired); **L0** ref all **1.095** high **0.52**; **L3** all **1.15**; **L2** all **1.51** (ReLU wall — **fail**).
- **Wall**: val logMAE **~2.23 on every leg** (including L6 with train `W·L_MuLogWall≈7.5`); wall metric still decoupled from “localized high-μ” goal.
- **Gates**: leash legs still **`gate_clot→0.96`** late; L5 suppressor keeps **`gate_clot~0.2–0.3`** but sacrifices high-μ; L1 bulk **r** can go slightly positive.
- **Readout**: For **viz / velocity**, **L1 confirmed fail** (true `teacher_last`); next **L4 → L6 → L7**; skip **L0** (same stack as failed L1 viz); **L5** metric-only; skip **L2**. Manifest metrics OK; per-leg **`best_high_mu.pth`** often wrong — use **`teacher_last`** unless leg updated global.
- **Archive caveat** (`Save-LegArtifacts`): per-leg `biochem_teacher_best_high_mu.pth` is a copy of **global** high-μ best at leg end, **not** that leg’s weights unless it updated global. Manifest val columns are correct (`run_note`); **`.pth` can be wrong** (e.g. L1 path showed **L0** ckpt: all **1.095**, high **0.52**, `run_note=L0`). For true leg weights use **`biochem_teacher_last.pth`** in the leg folder (end-of-run snapshot) or re-run the leg.

### 82. Viz **L1** (archive bug then true ckpt): soft wall-only gate **fails** velocity + clot localization (2026-05-24)

- **First viz** (`L1/.../biochem_teacher_best_high_mu.pth`): embedded metadata **L0** (all **1.095**, `run_note=L0`) — §81 archive copies **global** high-μ into leg folder; misleading.
- **Second viz** (`L1/.../biochem_teacher_last.pth`, **true L1**): all **0.917**, high **0.583**, `run_note=visc3h_L1_softwall_learn` @ ep16 — **same physics failure** as L0/§80.
- **Viz** (patient007): **t=0** Biochem **|u|≈0** vs COMSOL **~1.2–1.4**; late **μ₂ trigger ~80** **full domain** (COMSOL **~0** bulk, wall spots); total **μ ~5–6 Pa·s** **uniform lumen** (COMSOL wall-localized); **μ₁(Mat)** Biochem **flat** vs COMSOL wall patches — triggers saturated, **no COMSOL-like clots** despite better val logMAE than L0.
- **Readout**: **L1 soft wall-only + data leash is not a velocity/clot fix** at 18ep (train still **`gate_clot≈0.96`**). Val logMAE can improve while flow dies. Next viz: **L4** `teacher_last`, then **L6** `best_high_mu` (global **0.517** high-μ).

### 83. Viz leash+kin (likely **L4** `teacher_last`): t=0 flow weak; **μ₂(FI) global**, **μ₁(Mat) off** — wrong gelation channel (2026-05-24)

- **Viz** (patient007; user report + screenshots): **t=0** Biochem **|u|** slightly above L1 (thin core vs **≈0**) but still **≪ COMSOL** (~1.2–1.5); late **total μ ~4–5 Pa·s uniform** (COMSOL wall-localized); gelation panel **μ₁(Mat) Biochem black** vs COMSOL wall yellow; **μ₂(FI) Biochem ~80 everywhere** vs COMSOL **~0** — model raises viscosity via **FI tail**, not **Mat wall gelation**.
- **Mechanism** (code): `explicit_gelation = mu1_sigmoid(Mat) + mu2_sigmoid(FI)` with **`mu2` capped at `mu_ratio_max` (80)**; soft-gate fix (§80) applies to **learned clot `gate` / wall branch**, **not** FI/Mat sigmoids. Log-μ + leash can fit anchors while species/ODE feed **high FI** bulk-side.
- **Readout**: Dynamic soft-gating + data leash + kin LoRA **does not** restore COMSOL-like **Mat-on-wall / FI-quiet** pattern. **L6** (sentinel wall loss + leash) improves val high-μ (**0.517**) but **same μ₁/μ₂ viz pathology** (§84). Next science: **μ₁-only / wall-masked Mat loss**, or suppress **μ₂** in bulk; do not trust logMAE alone.

### 84. Viz **L6** `sentinel_leash` + train log `20260524T191651Z`: wall loss on, val `gate_wall=0`; `gate_all~0.49` not 0.96 — still μ₂-global (2026-05-24)

- **Ckpt**: `L6_sentinel_leash/biochem_teacher_best_high_mu.pth` — all **0.995**, high **0.517**, ep14 (`run_note=visc3h_L6_sentinel_leash`).
- **Viz**: same as §83 — weak t=0 **u**; late uniform **~4–5 Pa·s** μ; **μ₁(Mat) off**, **μ₂(FI)~80** domain-wide.
- **Structured log**: `outputs/reports/training/biochem/20260524T191651Z/run.jsonl` — val **`dbg_gate_mean_wall=0`** every epoch; **`dbg_gate_mean_all≈0.48–0.50`** (lower than L1 **`~0.96`**); best val ep14 all **0.995** / high **0.517** / wall **2.229**.
- **Console** (visc3h sweep): `W·L_MuLogWall≈7.5`, `W·L_MuLogHigh≈1.8`, `W·L_MuSI=2`, `DETACH_MACRO=0`, floors **0.0** in sentinel preset; train `gate_all~0.48` ep0 (not clot-saturated).
- **Readout**: Sentinel **wall log-μ weighting does not fix** wall-gate starvation on val or COMSOL-like gelation channel; FI/Mat sigmoid path still wrong in rollout.

### 85. `viz_final_mu2_mean` / `clot_frac` read raw FI sigmoid, not effective μ (2026-05-25) — **fixed**

- **Symptom**: **health10h** manifest showed **`viz_final_mu2_mean=80`** and **`clot_frac=1.0`** on **K0** / **S0** / **M1** identically.
- **Cause**: `_compute_slice_viz_health_metrics` and `visualize_pipeline` used raw species sigmoids / COMSOL-style **`μ_b×(μ₁+μ₂)`** regardless of forward ablations.
- **Fix**: **`biochem_explicit_gelation_terms()`** (`gnode_biochem.py`); training viz health uses **effective** μ₁/μ₂; **`clot_frac`** from rollout **`μ_eff`** when explicit gel is off; viz uses **stored `μ_eff` channel** for biochem panels (COMSOL: stored **`μ_eff`** + species sigmoids for trigger rows only).
- **Status**: Re-run sweeps to refresh manifest **`viz_mu2` / `clot_frac`** columns; old rows are stale.

### 86. **health10h** architecture sweep (9 legs, cold K0 pretrain → warm rest, viz-health ranking): **S0** wins scoreboard (2026-05-25)

- **Setup**: `scripts/go_health10h.ps1` → `run_biochem_health_arch_sweep_10h.ps1`; `outputs/biochem/sweep_health_arch_10h/`; per-leg **`BIOCHEM_ARCHIVE_CHECKPOINT_DIR`** (true leg weights); manifest `manifest.jsonl`; console `outputs/reports/training/biochem/health10h_console_20260525_000258.log`. **K0** cold AE+ODE pretrain → `biochem_post_pretrain.pth`; legs **R0–M2** warm-reuse. **22ep** (K0 **8ep**). ~**3h** total (not 10h — short teacher budgets).
- **Ranking** (patient007, **lower `viz_health_score` = better**; manifest best-epoch rows):

| leg | viz_health | val all logMAE | high-μ | viz t0 \|u\| | viz μ₂ mean | viz clot_frac |
|-----|------------|----------------|--------|--------------|-------------|---------------|
| **S0** simple residual (`MU_LOG`, no μ₁/μ₂ mult) | **13.37** | **0.451** | 1.12 | **0.346** | 80 | 1.0 |
| S1 simple + leash | 14.10 | 0.555 | 1.26 | 0.236 | 80 | 1.0 |
| M2 no explicit gel (leash, Δ heads only) | 14.41 | **0.417** | 1.46 | 0.253 | 80 | 1.0 |
| R0 ref leash (visc3h L0 stack) | 14.58 | 0.827 | 1.25 | 0.253 | 80 | 1.0 |
| M1 μ₁-only leash | 14.66 | 0.599 | 1.43 | 0.246 | 80 | 1.0 |
| M0 μ₂ cap=8 + leash | 15.05 | 1.016 | **0.756** | 0.262 | 80 | 1.0 |
| G1 gemini + `MU_LOG` | 15.11 | 1.468 | 0.868 | 0.267 | 80 | 1.0 |
| G0 gemini + leash | 15.13 | 1.465 | 0.864 | 0.260 | 80 | 1.0 |
| K0 Carreau-only (`DATA_KINE`, no clot heads) | 15.17 | 1.463 | 1.16 | 0.251 | 80 | 1.0 |

- **Val highlights**: **S0** ep20 all **0.451** (best logMAE in sweep); **M2** ep18 all **0.417** (best wall **2.15**); **R0** ep04 all **0.827** / high **1.25** (only leg near prior leash tier on all); **K0** flat **~1.463** all (μ not trained — expected). **S0** train: `L_tot~0.9`, `L_kine~1.75`, `gate=n/a` (simple path).
- **Viz-health caveats** (critical):
  - **`viz_final_mu2_mean=80` and `viz_final_clot_frac=1.0` on every leg**, including **K0** (no clot), **S0** (`BIOCHEM_MU_SIMPLE_LOG_RESIDUAL`), **M1** (`DISABLE_MU2`), **M0** (μ₂ cap=8). Cause: `_compute_slice_viz_health_metrics` reads **raw `mu2_sigmoid(FI)`** from rolled-out species, **not** the term that enters `μ_eff` when flags disable/cap explicit gelation (`train_biochem_corrector.py` ~5034–5040). **Do not** use these two columns to compare ablation legs until fixed.
  - **`viz_t0_speed_mean≈0.25–0.35`** on all legs (target healthy **~1.2–1.5**). Score still ranks **S0** first mainly via **lower final logMAE** + slightly higher t0 speed (**0.35**); **kinematic backbone / IC rollout** issue is **shared**, not fixed by architecture leg alone.
- **Gemini / sentinel**: **G0/G1** did **not** beat **S0** on viz score or logMAE; additive Δlogμ + symmetric clip **not** the winning knob in this 22ep budget. **G0** `gate→0` late (good) but val all **~1.46**; **R0** ep04 dip **0.827** then regress — same leash pathology as §74–84.
- **Readout**: **Promote S0** for full viz + optional longer run (leash vs pure `MU_LOG`); **M2** as logMAE/wall runner-up. **Fix viz μ₂/clot_frac** to reflect effective μ path before next sweep sorts on them. **K0** confirms weak t0 flow is **upstream of clot heads** (still **|u|~0.25**). Next: viz **S0**, **M2**, **R0** (baseline), **K0** (kinematic sanity); compare **`μ_eff` panels**, not raw μ₂ debug.

### 87. K0 viz (`health10h` / `K0_carreau_kinematic`): weak flow is **biochem rollout path**, not standalone GINO-DEQ (2026-05-25)

- **Ckpt**: `outputs/biochem/sweep_health_arch_10h/K0_carreau_kinematic/biochem_teacher_best_high_mu.pth` — `run_note=health10h_K0_carreau_kinematic_ep8`, val all **1.463** (flat), ep7.
- **Training reality**: teacher **`L_kine≈1.570` constant** all 8 ep (`DATA_KINE` isolate, kin/bio/ODE/mu frozen) — K0 did **not** materially fit velocity; checkpoint ≈ shared cold pretrain + frozen stack.
- **Temporal inspector (patient007)**: Biochem **|u|≈0** at **t=0** and still **≪ COMSOL** at **t≈9540 s**; **p** nearly flat (~−0.5 ND) vs COMSOL inlet→outlet gradient. **Small drift over time** is expected: forward still **integrates frozen ODE** species between macro knots and recomputes Carreau **γ̇(u,v)** each step — not a steady single-shot solve.
- **Gelation 2×2 panel**: Biochem **μ₂(FI)≈80 domain-wide** but **μ₁ product ≈0** — **misleading for K0**: `visualize_pipeline._biochem_rheology_fields` always plots **raw `mu1_sigmoid`/`mu2_sigmoid` on rolled-out species**, while forward with `BIOCHEM_MU_DISABLE_EXPLICIT_GELATION=1` uses **`μ_eff = μ_kin(γ̇)` only** (no explicit gelation in `μ_eff`). Do **not** read K0 gelation figures as “model clotting.”
- **Fair kinematics comparison**: same run also opens **`Kinematics (GINO-DEQ), steady — patient007`** (`kinematics_best.pth`, one-shot on biochem mesh). If that window shows healthy **|u|~1+** while temporal slider stays weak, the bug is **biochem coupled macro rollout** (DEQ + `mu_encoder` feedback + resting species IC + low-shear Carreau lockup), **not** the kinematics backbone weights.
- **Readout**: K0 **does not** clear the “healthy kin baseline” hypothesis on **biochem rollout**; it only shows clot architecture is off in **`μ_eff`**. Next compare **steady GINO-DEQ figure** vs **Biochem `μ_eff` channel** at final time (Figure 2 dynamic viscosity), then viz **S0** (best scoreboard leg).

### 88. S0 viz (`health10h` / `S0_simple_residual`): better **t=0** rollout, **open-loop ODE blow-up** by first keyframe (2026-05-25)

- **Ckpt**: `…/S0_simple_residual/biochem_teacher_best_high_mu.pth` — ep21, val all **0.451**, `run_note=health10h_S0_simple_residual_ep22`.
- **t=0 (slider 0)**: Biochem **|u|** faint but **non-zero** lumen structure vs K0 **≈0**; **p** still flat vs COMSOL; species / trigger rows **at rest** (black / white) — matches resting prior before first ODE segment.
- **t≈2623 s (slider 1, first post-IC keyframe)**: **Global pathology** — Biochem **FI ~4–5×10³** domain-wide (COMSOL **0**); **μ₂ trigger → cap** (~50 in slider, ~80 in static gelation fig); slider **μ_b×(μ₁+μ₂)** **~6–7.5 Pa·s** everywhere; **|u|→0**, **p** flat. **One macro jump** (~2623 s) with **viz `teacher_forcing_ratio=0`** and **frozen ODE** (`TRAIN_ODE=0` in sweep) — not the TBPTT+anchor-supervised regime that earned **0.451** logMAE.
- **Forward vs display viscosity (S0)**: training uses **`BIOCHEM_MU_SIMPLE_LOG_RESIDUAL=1`** → **`μ_eff = μ_kin × exp(Δlogμ)`**, **no** explicit μ₁/μ₂ in forward. Temporal slider rows 4–6 and Figure 2 **recompute** COMSOL-style **`μ_b×(1+μ₁+μ₂)`** via `_biochem_rheology_fields` on rolled-out species — **overstates** clot when FI explodes open-loop (§85). For S0, also inspect **stored `mu_eff` ND channel** in the rollout tensor (Figure 1 steady kin panel uses GINO-DEQ, not this channel).
- **Steady GINO-DEQ (same run)**: **|u|** inlet core **~1.5–1.7** ND but **mostly dark downstream**; **μ_eff** lumen core **elevated** (yellow ~5–8 ND) — **better than biochem rollout** at all slider times, **not** a perfect COMSOL match. Confirms **kin backbone partially OK**, **coupled rollout + chemistry integration** is the failure mode.
- **Gelation 2×2 (final time)**: same **global μ₂ flood** as K0 — **diagnostic sigmoid on species**, not S0’s trained **`μ_eff`** path.
- **Readout**: **S0 improves first-step kinematics vs K0** but **does not** fix open-loop rollout health; **do not promote** from val logMAE alone. Next: viz **M2**; optional **`VIZ_BIOCHEM_TIME_MODE=dense`** or shorter first Δt; fix viz to plot **effective `μ_eff`** when `SIMPLE_LOG_RESIDUAL` / `DISABLE_EXPLICIT_GELATION` / `DISABLE_MU2`.

### 89. K0 parity fresh viz (`K0_stage_a_parity_fresh`): flow **restored** vs health10h K0, **evolves in time** on open-loop rollout (2026-05-25)

- **Ckpt**: `outputs/biochem/biochem_teacher_best_high_mu.pth` — `run_note=K0_stage_a_parity_fresh`, stored val all **1.4715** / high **1.2325** (teacher best ep06 in `run.jsonl`; ckpt metadata may say ep0).
- **Training**: unchanged from row above — **`L_kine≈2.51` flat** all 8 ep (`DATA_KINE`, frozen kin/bio/ODE); **no** epoch-wise velocity learning.
- **t=0 temporal inspector**: Biochem **|u| ~1.0–1.2** (yellow–green) vs COMSOL **~1.5–2.0** (inlet core red) — **~30–40% amplitude gap**, but **not** the health10h **|u|≈0** pathology. **p** still weaker gradient than COMSOL. **μ_eff / μ₁ rows black** at t=0 (stored rollout channel ~0 or below color floor).
- **Steady GINO-DEQ** (same session): **|u| ~1.4–1.6** at inlet, structured lumen — **closer to COMSOL** than biochem **t=0** one macro step. Fair read: **kinematics_best weights OK**; gap is **biochem coupled macro solve** (Anderson + `mu_encoder` @ t=0, then Carreau feedback), not missing SIREN/width load.
- **t≈9540 s (late slider)**: Biochem shows **narrow high-|u| jet** (local peaks **~1.5–2.0**) vs COMSOL **broad** lumen flow — pattern **wrong** though **local speed can look “stronger”** than t=0. **μ₂(FI) → 80** domain-wide on biochem side; COMSOL **~0** — **diagnostic only** (`DISABLE_EXPLICIT_GELATION=1` → forward **`μ_eff = μ_kin(γ̇)`** only; §87). **Do not** interpret late-time red μ₂ as trained clotting.
- **Why |u| changes with the time slider** (expected in viz, **not** from teacher updates):
  1. **`GNODE_Phase3.forward` re-solves kinematics every macro knot** with updated **`current_mu_eff`** (Carreau from **u,v**) written into `kin_in` channel 13 and **`z_kin_ws` carry-over** between steps.
  2. **Frozen biochem ODE still integrates species** between knots at viz (`teacher_forcing_ratio=0`) — resting prior drifts over **~9.5 ks** → inflated **FI → μ₂ diagnostic** panels even when **μ_eff** path ignores explicit gelation.
  3. **Fast viz subsamples ~12 macro steps** + optional **time extension** past COMSOL `t_final` — late slider may be **extrapolation**, not anchor GT at that second.
  4. **Teacher did not learn this drift** — train loss flat; compare **GINO steady** for “what frozen kin can do” vs **slider** for “open-loop coupled rollout.”
- **Readout**: Stage-A parity fixes **cleared dead flow** (manifest **viz score 2.71** vs **15.17** health10h K0). Remaining: **t=0 amplitude**, **late jet + species blow-up in open-loop viz**. Next: plot **stored `μ_eff` ND channel** (row 2) vs COMSOL; try **`VIZ_BIOCHEM_TIME_MODE=dense`** or anchor-only short rollout; μ work remains **MU_LOG / leash** legs, not more K0 epochs.

### 90. **K1** `delta_mu` + `DATA_KINE` on 4GB (2026-05-25): μ moves when Δμ head trains; OomSafe completes

- **Hypothesis test**: After **K0** parity (flow OK, **μ flat ~1.47**, `USE_DELTA_MU_HEAD=0`, `TRAIN_MU_ENCODER=0`), enable **Carreau × exp(Δlogμ)** (`USE_DELTA_MU_HEAD=1`, `TRAIN_MU_ENCODER=1`, `MU_DISABLE_EXPLICIT_GELATION=1`, `LOSS_ISOLATE=DATA_KINE`, `TEACHER_FORCE_MIN=1.0`, bio/ODE/LoRA frozen).
- **Run A** (`20260525T101349Z`, `TBPTT=12`, `workers=4`): val all **1.335→0.541** ep3; train `L_kine` **2.11→0.81** ep7; **CUDA OOM** ep7 backward (Anderson + μ-path adjoint on RTX 500 4GB).
- **Run B** (`20260525T102611Z`, `go_k1_delta_mu.ps1` OomSafe: `TBPTT=5`, `workers=0`, `kin_ckpt=1`, `RK4=8`): **12 ep complete**; best val all **0.4643** ep11 (ep9 **0.4784**); wall **1.80**, high-μ **1.12**, bulk **r=0.22** ep11 (all-truth **r≈-0.09**); train `L_kine` **2.11→0.55**; viz health **2.63→1.04**, **t0|u| 0.36→0.61**, `clot_frac=0`, `μ1/μ2=0` on effective path.
- **Cause → fix (VRAM only)**: `TBPTT=12` + default `DATALOADER_WORKERS=4` + μ-encoder gradients through macro DEQ — not a physics-flag bug. **Do not** change gelation disable, loss isolate, or TF to fix OOM; shorten TBPTT / zero workers / kin gradient checkpointing (`scripts/go_k1_delta_mu.ps1 -OomSafe`).
- **Readout**: **Pass** on “can learned Δμ + `μ_encoder` fit COMSOL μ under `DATA_KINE` with perfect species (TF=1)?” vs **K0 fail**. Next: **viz** `biochem_teacher_best_high_mu.pth`; optional longer run or `MU_LOG` isolate for wall/high-μ; compare open-loop rollout vs K0; do not read `mu2=80` train debug as forward clotting (`DISABLE_EXPLICIT_GELATION=1`).

### 91. **K2** explicit gelation + step-3 multitask (2026-05-25): completes on 4GB but **regresses** vs K1

- **Setup** (`20260525T105120Z`, `go_k2_physics_triggers_on.ps1`): warm-start intent from K1 ckpt; `MU_DISABLE_EXPLICIT_GELATION=0`, `GELATION_PRIOR_GATE=0`, `USE_DELTA_MU_HEAD=1`, `TRAIN_MU_ENCODER=1`, `COMPLEXITY_STEP=3`, `LOSS_DATA_ONLY=0` (Kendall multitask), `TF=1`, bio/ODE/LoRA frozen, OomSafe `TBPTT=5`. (`run.jsonl` meta: `INIT_FROM_BEST=0` — verify ckpt lineage if comparing to K1 weights.)
- **Preflight**: median **5.77** (vs K1 ~1.45) — explicit **μ₁/μ₂** sigmoids on resting species inflate IC μ error before training.
- **Val (patient007)**: ep0 all **5.58** → best **4.22** ep9 (ep11 **4.22**); wall **3.34**, high-μ **3.10**, **r≈-0.05** flat. **K1** best was **0.464** on same GPU recipe minus gelation/multitask.
- **Train**: `L_tot` **~1.7e3** ep0 → **~7e2** ep9 (physics Kendall sum); `L_kine` **~26→7** (similar scale to K1); `L_bio` ~380 flat (not in backward — frozen). `mu2=80`, `mu1~7–9` in μdbg throughout.
- **Viz health**: score **20.1→18.4**; **t0|u|~0.37–0.40**; **`clot_frac=1.0`**, **`μ₂=80`**, **`μ₁~6–10`** on rollout — global clot channel flood (expected once explicit gelation enters **`μ_eff`** without prior gate / cap).
- **Readout**: **Fail** as a promotion step from K1. Proves step-3 **runs** on 4GB OomSafe but **raw trigger + full PDE backward** does not improve held-out logMAE vs data-only Δμ path. Next: viz K2; try **`GELATION_PRIOR_GATE=1`**, **`MU2_SIGMOID_CAP`**, or **staged** gelation re-enable while keeping **`LOSS_ISOLATE=DATA_KINE` or `MU_LOG`** until val **<1.0**; do not advance corrector on this ckpt.

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
| 2026-05-22 | New wall-controls **Run A** (`sweep_free_wall_a`, RTX500 4GB, teacher-only 25ep): `MU_LOG` isolate, split clipping, `WALL_HEAD_ISOLATE_GEOM=1`, latent320/prior4, `TBPTT=5`, `DETACH=1` | **0.4547** (ep18 best-all) | **1.5478** (best-all ckpt; best wall **1.4669** ep21) | **0.347** | high **1.0757** (best-all ckpt) | Geometry-isolate materially improves wall and reaches near-target boundary band, but high-μ worsens and late-epoch instability persists |
| 2026-05-22 | New wall-controls **Run B** (`sweep_free_wall_b`, P2200 5GB, teacher-only 25ep): `MU_LOG` isolate, split clipping, `WALL_SPATIAL_DECAY=1`, latent320/prior4, `TBPTT=5`, `DETACH=1` | **0.3016** (ep24 best-all) | **2.1246** | **0.444** | high **0.6915** | Spatial decay gives strongest global score in this session and good high-μ recovery, but wall remains far above target and gate_wall stays near-zero after startup |
| 2026-05-22 | Geom-blend+decay retest **Run 1** (`sweep_free_wall_b`, RTX500 4GB): `GEOM_BLEND=0.35`, `WALL_GATE_MIN=0.05`, decay `7.0` floor `0.05`, teacher-only 25ep | **0.5296** (ep15 best-all) | **3.8510** (best-all; frequent **~6.19** regime) | **-0.126** | high **1.0369** (best-all ckpt; high best **0.5681** ep21 in collapse regime) | Regressed vs prior sweep_free_wall_b baseline; wall branch collapsed and gates decayed to near-zero |
| 2026-05-22 | Geom-blend+decay retest **Run 2** (`sweep_free_wall_b`, P2200 5GB): `GEOM_BLEND=0.35`, `WALL_GATE_MIN=0.05`, decay `7.0` floor `0.05`, teacher-only 25ep | **0.3145** (ep24 best-all) | **3.4304** (best-all; alternate regime **~6.61**) | **-0.078** | high **1.0097** (best-all ckpt; high best **0.5254** ep21) | Strong all-truth but unstable bimodal wall dynamics (gate_wall toggles 1.0 <-> ~1.9e-22), so boundary objective remains unsolved |
| 2026-05-22 | Exploratory **Run A** (`sweep_free_wall_a`, RTX500 4GB): `GEOM_BLEND=0.80`, `WALL_GATE_MIN=0.12`, decay `3.0` floor `0.30`, teacher-only 25ep | **0.7295** (ep00 best-all) | **3.8385** | **-0.126** | high **1.8299** | High-geometry blend was too aggressive: run never beat startup checkpoint, wall stayed poor, and late epochs showed unstable learned-wall escalation |
| 2026-05-22 | Exploratory **Run B** (`sweep_free_wall_b`, P2200 5GB): `GEOM_BLEND=0.15`, `WALL_GATE_MIN=0.10`, decay `4.0` floor `0.20`, teacher-only 25ep | **0.4323** (ep09 best-all) | **3.7021** | **-0.141** | high **1.0749** | Softer blend is more stable and improves all-truth vs Run A, but wall/high-μ remain far from target; gate_wall still effectively inactive |
| 2026-05-22 | New budgeted comp-A sweep attempt (`compA_L4_S0_B8_R0`, RTX500 4GB, 8ep): script default profile after spatial-knob merge | n/a (teacher startup OOM) | n/a | n/a | n/a | OOM at first teacher forward (`GINO softmax` path) with `gnode_layers=4`; fixed by adding 4GB-safe runtime defaults + kinematic gradient checkpointing + safer default layer profile (2/3 layers) |
| 2026-05-22 | New budgeted comp-B sweep attempt (`compB_M1_C100_A0_T5`, P2200 5GB, 8ep): first leg with dense ODE backward (`A0`) | n/a (teacher startup OOM) | n/a | n/a | n/a | OOM in ODE/GINO path during first teacher step (dense `odeint` + GAT softmax); fixed by 5GB-safe defaults (allocator/checkpointing, workers/pin tuning), safe TBPTT profile, and adjoint-only budget legs |
| 2026-05-23 | Budgeted comp-A safe sweep (RTX500 4GB, 8 legs, teacher-only 8ep): `layers∈{2,3}`, SIREN/Fourier/LoRA ablations under VRAM-safe defaults | **best 1.4599** (`L3,S0,B8,R0`) | **~2.25–2.43** | **~0.35** | high **~0.91–1.03** | OOM resolved, but all legs remain in teacher plateau basin; best leg is `layers=3` without SIREN/LoRA |
| 2026-05-23 | Budgeted comp-B safe sweep (P2200 5GB, 8 legs, teacher-only 8ep): `MU_LOSS_SCALE×RHEOLOGY_CAP` with adjoint-only safe profile | **best ~1.4937** (`M1,C100` / `M10,C100`) | **~2.29–3.27** | **~0.39–0.43** | high **~0.89–0.98** | OOM resolved; low cap (100) dominates, while cap 500/1000 degrades wall/global strongly in this short detached regime |
| 2026-05-23 | Fair h2h **BestAllArch** (`WALL_SPATIAL_DECAY=1`, SPAGNIER RTX500, 6ep warm-start, `DETACH=1`, ~3m) | **0.6496** (ep04 best) | **3.8468** | **~-0.20** | high **1.1247** | Sequential on one fast laptop (not dual-machine); wall stuck ~3.85; `gate_wall=0`; finished ~6× faster than ~30m budget |
| 2026-05-23 | Fair h2h **BalancedArch** (`WALL_HEAD_ISOLATE_GEOM=1`, SPAGNIER RTX500, 6ep warm-start, `DETACH=1`, ~2m) | **0.5259** (ep04 best) | **1.6610** | **~0.43** | high **1.0205** | Wins fair A/B on all/wall/high at best ckpt; positive `r`; still `gate_wall=0`; ep05 regressed slightly (all 0.5892) |
| 2026-05-23 | Wall-μ₀ A/B **LearnWallMu** (`FORCE_WALL_MU0=0`, geom-isolate, 8ep, ~3m) | **0.4763** (ep06) | **1.7743** | **~0.36** | high **0.6933** | Fix path: wall improves vs forced-μ₀; batch ~6.3m on SPAGNIER |
| 2026-05-23 | Wall-μ₀ A/B **ForceWallMu0** (`FORCE_WALL_MU0=1`, same base, 8ep, ~3m) | **0.4883** (ep06) | **2.4594** (flat) | **0** (wall) | high **0.9204** | Override stunts wall learning; wall logMAE identical every val — confirms hazard |
| 2026-05-23 | ND surface A/B **Baseline** (legacy surface ODEs, geom-isolate, 8ep, blend 0.15, ~3.3m) | **0.7267** (ep04 best) | **1.8845** | **~0.33** | high **0.6194** | Ep07 collapse (all 1.44); less stable than fix arm |
| 2026-05-23 | ND surface A/B **NdSurfaceFix** (`ND_SURFACE_PHYSICS=1`, Da+gate, same base, ~3.3m) | **0.4941** (ep06 best) | **1.8015** | **~0.36** | high **0.9425** (ep07 **0.7687**) | Wins all-truth; marginal wall gain; high-μ worse at best ckpt but stable late |
| 2026-05-23 | Wall-gate A/B **baseline** (geom-isolate, wall/high=2, blend 0.15, 8ep, `DETACH=1`) | **0.3729** (ep07) | **1.7928** (ep04) | **~0.46** | high **0.5558** (ep02) | Best-all strong; wall ~1.74–1.96; `gate_wall=0` |
| 2026-05-23 | Wall-gate A/B **`sweep_free_wall_a`** (intended 8ep; preset forced **25ep**) | **0.3472** (ep20) | **1.5667** (ep20) | **~0.22** | high **0.6777** (ep18) | Beats A on wall at ep08+long run; `gate_wall=0`; `gate_clot` saturates late |
| 2026-05-23 | `wall_ab_fix_8ep` (`sweep_free_wall_a`, geom-isolate; **still 25ep**) | **0.4369** (ep06 best) | **1.6777** (ep06) | **~0.49** | high **0.8202** | Ep08 collapse (all 0.90); worse than baseline 8ep; epoch-cap bug confirmed |
| 2026-05-23 | `wall_ab_fix_8ep_v2` (preset, **8ep OK**; fresh pretrain+LoRA; confounded) | **0.7789** (ep06) | **1.6271** (ep06; ep07 **1.5403**) | **~0.22** | high **1.0421** | All-truth far worse than fair baseline; wall ep07 promising; `gate_wall=0` |
| 2026-05-23 | Fair wall A/B **A baseline** (`Set-FairBase`, 8ep, warm-start) | **0.7615** (ep02 best-all) | **1.5027** (ep04) | **~0.36** | high **0.9370** | Ckpt on all only; wall best ep04 not saved |
| 2026-05-23 | Fair wall A/B **B `sweep_free_wall_a`** (same base, 8ep) | **0.5081** (ep06 best-all) | **1.9513** (ep06) | **~0.30** | high **0.9889** | **Wins fair A/B on saved all**; ep07 high **0.8214**; `gate_wall=0` |
| 2026-05-23 | Wall3h sweep **batch** (fair base, warm-start, `DETACH=1`, `LORA=0`, val/2; SPAGNIER+SILKSPECTRE; `wall_gate_fair_sweep_3h`) | see legs below | — | — | ~62m+76m total; `CLI_TEACHER_EPOCHS` ladder honored |
| 2026-05-23 | Wall3h **baseline** `B0_ep8` (SPAGNIER) | **0.8695** (ep06) | **1.6052** (ep06) | **0.36** | high **2.0640** | `gate_wall=0`; early-stop not used |
| 2026-05-23 | Wall3h **baseline** `B0_ep14` (SPAGNIER) | **0.4577** (ep04) | **1.9106** (ep04) | **0.37** | high **0.6238** (ep08) | `gate_wall=0`; best all early ep04 |
| 2026-05-23 | Wall3h **baseline** `B0_ep20` (SPAGNIER) | **0.5982** (ep19) | **2.0260** (ep19) | **0.42** | high **0.9692** (ep19) | `gate_wall=0`; late improves all, wall drifts up |
| 2026-05-23 | Wall3h **baseline** `B0_ep26` (SPAGNIER) | **0.5069** (ep20) | **1.9737** (ep20) | **0.32** | high **1.1427** (ep20) | `gate_wall=0`; ep24 wall **1.7110** not saved (all ckpt) |
| 2026-05-23 | Wall3h **baseline** `B0_ep30` (SPAGNIER) | **0.4328** (ep29) | **2.0210** (ep29) | **0.34** | high **1.0729** (ep29) | `gate_wall=0`; best all in ladder |
| 2026-05-23 | Wall3h **baseline** `B0_ep34` (SPAGNIER) | **0.4239** (ep33) | **1.9708** (ep33) | **0.31** | high **1.0464** (ep33) | `gate_wall=0`; wall still ~2.0 at best-all |
| 2026-05-23 | Wall3h **baseline Pareto** `B0_ep20_pareto` (SPAGNIER) | **0.7504** (ep02 saved) | **1.5388** (ep04) | **~0.25** | high **0.8646** (ep04) | Pareto froze early; missed ep20+ wall **1.67** |
| 2026-05-23 | Wall3h **`sweep_wall_sentinel`** `WS_ep18` (SPAGNIER) | **0.3185** (ep17) | **1.5479** (ep17) | **0.12** | high **0.9962** (ep17) | **`gate_wall=1.0`** train ep10–17; best all in sweep |
| 2026-05-23 | Wall3h **`sweep_free_wall_a`** `FWa_ep8` (SILKSPECTRE) | **0.9812** (ep00) | **2.5124** (ep00) | **0.37** | high **0.9100** (ep00) | `gate_wall~0` after ep02; poor start |
| 2026-05-23 | Wall3h **`sweep_free_wall_a`** `FWa_ep14` (SILKSPECTRE) | **0.4790** (ep12) | **1.7933** (ep12) | **0.28** | high **0.9023** (ep12) | Beats baseline ep14 on all; `gate_wall~0` |
| 2026-05-23 | Wall3h **`sweep_free_wall_a`** `FWa_ep20` (SILKSPECTRE) | **0.4704** (ep19) | **1.7016** (ep19) | **0.30** | high **0.9414** (ep19) | Beats baseline ep20 on all; wall similar |
| 2026-05-23 | Wall3h **`sweep_free_wall_a`** `FWa_ep26` (SILKSPECTRE) | **0.4116** (ep22) | **1.9674** (ep22) | **0.29** | high **1.1376** (ep22) | Best all @22ep; wall still ~2.0 |
| 2026-05-23 | Wall3h **`sweep_free_wall_a`** `FWa_ep30` (SILKSPECTRE) | **0.4425** (ep22) | **2.1296** (ep22) | **0.45** | high **0.7069** (ep22) | `gate_wall~0`; ep30 all **0.5004** |
| 2026-05-23 | Wall3h **`sweep_free_wall_a`** `FWa_ep34` (SILKSPECTRE) | **0.3422** (ep33) | **2.1341** (ep33) | **0.37** | high **1.0464** (ep33) | **Best all** in sweep; wall not improved |
| 2026-05-23 | Wall3h **`sweep_free_wall_b`** `FWb_ep20` (SILKSPECTRE) | **0.5060** (ep19) | **1.9180** (ep19) | **0.36** | high **0.6895** (ep19) | High-clot penalty preset; wall ~1.9 |
| 2026-05-23 | Wall3h **`sweep_bio_suppressor`** `BIO_ep18` (SILKSPECTRE) | **0.7279** (ep08) | **1.7623** (ep10) | **0.45** | high **0.6834** (ep16) | `gate_wall≈0.06`; clot-tail ok, wall stuck |
| 2026-05-23 | Gate-fix **Arm A batch** (`gate_fix_sweep`, SPAGNIER, fair 18ep) | see legs | — | — | ~43m; `scripts/run_biochem_gate_fix_sweep.ps1` |
| 2026-05-23 | Gate-fix **baseline** (Arm A, fair 18ep) | **0.6227** (ep16) | **1.8311** | **0.31** | high **1.0805** | `gate_wall=0`; wall ep04 dip **1.521** not saved |
| 2026-05-23 | Gate-fix **Fix A** curriculum 12ep (Arm A) | **0.5155** (ep16) | **1.9864** | **0.28** | high **1.2763** | **Best all Arm A**; wall not improved vs baseline |
| 2026-05-23 | Gate-fix **Fix B** bypass w=1.5 (Arm A) | **0.5335** (ep14) | **1.9000** | **0.43** | high **1.0433** | Train `L_tot` ~2×; ep16 val collapse **0.85** |
| 2026-05-23 | Gate-fix **Fix C** pos-init 3.0 (Arm A) | **0.5936** (ep16) | **1.8729** | **0.27** | high **1.0346** | `gate_all` collapse ep12–14; below baseline all |
| 2026-05-23 | Gate-fix **Fix D relu_add** (Arm A) | **0.5091** (ep17) | **1.9673** | **0.43** | high **0.8945** | Strong all + best high-μ trade on SPAGNIER |
| 2026-05-23 | Gate-fix **Fix D siren_add** (Arm A) | **0.5650** (ep12) | **1.9644** | **0.39** | high **0.9566** | Mid pack; ep17 regress **0.91** |
| 2026-05-23 | Gate-fix **Arm B batch** (`gate_fix_sweep`, SILKSPECTRE, fair 18ep) | see legs | — | — | ~33m |
| 2026-05-23 | Gate-fix **sentinel ref** (Arm B, 18ep) | **0.4282** (ep16) | **2.0973** | **0.27** | high **1.2191** | `gate_wall` floor ~0.06; below wall3h **0.3185** |
| 2026-05-23 | Gate-fix **fix_ab** curriculum+bypass (Arm B) | **0.4354** (ep16) | **2.3458** | **0.33** | high **0.5258** | **Best high-μ**; wall worst in batch |
| 2026-05-23 | Gate-fix **fix_ac** curriculum+pos-init (Arm B) | **0.4356** (ep17) | **1.8957** | **0.31** | high **0.8425** | Best wall in batch; ties sentinel all |
| 2026-05-23 | Gate-fix **fix_abc** combo (Arm B) | **0.4752** (ep10) | **1.9142** | **0.40** | high **1.4000** | Combo worse than AB or AC alone |
| 2026-05-23 | Gate-fix **deep 4h** `WS_sentinel` @34ep (Arm A) | **0.2938** (ep32) | **1.5001** | **0.11** | high **0.740** | **New best all** on patient007; `metrics.jsonl` run seg 115 |
| 2026-05-23 | Gate-fix **deep 4h** `A_curriculum` @30ep (Arm A) | **0.3286** (ep29) | **1.8479** | **0.37** | high **0.754** | Fair MU_LOG; no sentinel preset |
| 2026-05-23 | Gate-fix **deep 4h** `D_relu` @40ep (Arm A) | **0.4839** (ep34) | **2.0158** | **0.48** | high **0.626** | Best in relu ladder; vs 18ep **0.509** |
| 2026-05-23 | Gate-fix **deep 4h** `D_relu_tbptt6` @34ep (Arm A) | **0.5173** (ep18) | **2.0142** | **0.40** | high **0.564** | Last leg; overwrites `biochem_teacher_best.pth` |
| 2026-05-23 | Gate-fix **deep 4h** batch Arm B (`gate_fix_deep_4h`, SILKSPECTRE) | (on host B) | — | — | **204m**, 16/16 OK | metrics not in laptop A `metrics.jsonl` |
| 2026-05-24 | Supervised data leash (`WS_sentinel_data_leash_ep26`, sentinel + **`SUPERVISED_DATA_LEASH=1`**, init-from-best, `DATA_ONLY=1`, no isolate, `DETACH=0`, `W_MuSI=2.0`, 26ep ~23m) | **0.2232** (ep14 ckpt) | **1.9247** @ ep14; **1.538** ep20; **1.474** ep22 | **0.267** (ep14); **0.548** ep22 | high **0.470** (ep14); **1.367** ep24 | **Wiring OK** (`L_Back≈L_tot`, `L_bio`→0.05); **all/high beat** sentinel MU_LOG @34ep (**0.307** / **0.746**) but **wall worse** at saved ckpt; `gate_wall` floor ~0.06 (not saturated) |
| 2026-05-24 | Data leash + **bulk surgical lock** (`bulklock_ep26`, `CLIP_BULK=0.05`, bio suppressor floor 0, init-from-best-high-μ, 26ep ~23m) | **0.3531** (ep16) | **2.0649** @ ep16; **1.92** ep22 | **0.287** (ep16) | high **1.256** (ep16); global ckpt kept **0.470** (prior leash) | Bulk subset **0.25** @ ep16; **regressed** vs leash on all/high; wall still **~2.0**; volatile late epochs |
| 2026-05-24 | **Cold** data leash + bulk lock (`bulklock_cold_ep26`, `INIT_FROM_BEST=0`, AE+ODE pretrain, kinematics init, same leash/bulk env, 26ep ~23m, `20260524T140238Z`) | **0.9070** (ep12) | **2.1955** @ ep12; **2.25** ep25 | **0.242** (ep12); **0.321** ep25 | high **0.773** @ ep12; ckpt @ ep22 high **0.702** | Preflight **1.52**; ep18 collapse **1.52** all; **far** from sentinel (**0.307**) / warm leash (**0.223**); wall unchanged; `gate_wall` floor ~0.06 |
| 2026-05-24 | **Cold + strict μ-freeze** (`bulklock_cold_mufreeze_ep26`, leash+bulk+`TRAIN_*=0` teacher, μ-path=22, AE13/ODE11, ~22m, `20260524T144958Z`) | **0.5713** (ep12) | **2.2516** flat ep2–25 | **0.405** (ep12) | high **0.959** @ ep12; ckpt ep24 high **0.593** | vs §76 cold: all **0.57** vs **0.91**; bulk **0.53**; wall **stuck** ~2.25 (no ep17 dip); still **&gt;** warm leash/sentinel on all; `L_bio~300` with `biology=0` |
| 2026-05-24 | **Cold μ-freeze + hard gate 0.15** (`bulklock_cold_mufreeze_hardgate_ep26`, same stack + `HARD_THRESH=0.15`, `TRIGGER_GATE_MIN=0`, 26ep, `20260524T153126Z`) | **0.5906** (ep16) | **1.8678** @ ep16 (§77 wall **2.25** flat); **2.27** ep25 | **0.125** (ep16); **0.334** ep25 | high **1.202** @ ep16; ckpt high **0.672** @ ep02 | Wall **unfreezes** on val; all ~§77; bulk **0.52**; train `gate_wall≈1` early; **viz** for full-domain clot |
| 2026-05-24 | **Cold μ-freeze + soft gate** (`bulklock_cold_mufreeze_hardgate_ep26`, `HARD_THRESH=0.15` sigmoid **steepness=20**, leash+bulk+mufreeze, cold pretrain, 26ep, `20260524T160923Z`) | **0.7579** (ep10 ckpt) | **2.2525** @ ep10 (§78 **1.87**); **2.234** ep25 | **0.400** (ep10); **0.008** ep25 | high **0.597** @ ep10; ckpt high **0.553** @ ep14 | Early peak then **collapse** ep22–25 (**2.67** all); `gate_clot≈0.96` from ep6; beats §78 on all/high at ckpt, **loses** §78 wall gain; **early-stop @ ep10** candidate |
| 2026-05-24 | **visc3h** `L0_mufreeze_ref` (data leash, μ-freeze, bulk lock, 18ep warm pretrain, `20260524T171304Z`) | **1.095** (ep16) | **2.229** | **0.392** (ep16) | high **0.520** | Reference; `gate_clot≈0.94` late; first run false-FAIL (PS exit-code bug, recovered on rerun) |
| 2026-05-24 | **visc3h** `L1_softwall_learn` (leash + **soft gate wall_only** + learned temp) | **0.917** (ep16) | **2.229** | **0.395** | high **0.593** | Val best among leash+arch legs; **viz FAIL** (`teacher_last` ep16): t=0 **u≈0**, μ₂ **~80** global, **~5–6 Pa·s** uniform — **not** a deployable clot model |
| 2026-05-24 | **visc3h** `L2_relu_wall` (ReLU `delta_wall`) | **1.515** (ep17) | **2.229** | **0.344** | high **0.962** | Metrics flat ~1.54 train logMAE; **no gain** vs L0 |
| 2026-05-24 | **visc3h** `L3_wall_decay` (SDF wall decay) | **1.151** (ep14) | **2.228** | **0.378** | high **0.813** | Mid bulk improvement; wall unchanged |
| 2026-05-24 | **visc3h** `L4_kine_lora` (leash + μ-freeze + LoRA r4) | **0.941** (ep16) | **2.254** | **0.396** | high **0.561** | Val strong; **viz FAIL** (`teacher_last`): t=0 **u** weak; **μ₂~80** global, **μ₁(Mat)~0**; uniform **~4–5 Pa·s** μ (§83) |
| 2026-05-24 | **visc3h** `L5_mu_log_suppress` (`MU_LOG` isolate, suppressor, **no leash**, `DETACH=1`) | **0.408** (ep17) | **2.228** | **0.404** | high **1.152** | **Best val all** in sweep; **`gate_clot~0.3`**; **viz velocity** before promote |
| 2026-05-24 | **visc3h** `L6_sentinel_leash` (`sweep_wall_sentinel` + leash) | **0.995** (ep14) | **2.229** | **0.395** | high **0.517** | **`W·L_MuLogWall≈7.5`** train; global high-μ ckpt; **viz FAIL**: t=0 **u** weak; **μ₂~80** / **μ₁~0**; uniform late μ; val **`gate_wall=0`** despite wall loss (`run.jsonl` `20260524T191651Z`) |
| 2026-05-24 | **visc3h** `L7_early_stop` (L1 stack + target all≤0.65) | **0.958** (ep16) | **2.229** | **0.394** | high **0.569** | Early-stop threshold **not hit** (best all 0.958); similar to L1 |
| 2026-05-25 | **health10h** `S0_simple_residual` (`MU_LOG`, `SIMPLE_LOG_RESIDUAL`, 22ep warm) | **0.451** (ep20) | **1.743** | **0.346** viz t0 \|u\| | high **1.12** | **Best viz_health 13.37**; **viz patient007**: t0 rollout **>K0**; slider1 **FI/μ₂ global flood**, **u→0** (§88); steady GINO-DEQ **partial** |
| 2026-05-25 | **health10h** `S1_simple_residual_leash` (S0 + data leash) | **0.555** (ep04) | **1.774** | **0.236** | high **1.26** | viz **14.10**; worse t0 speed than S0 |
| 2026-05-25 | **health10h** `M2_no_explicit_gel` (leash, Δ heads only) | **0.417** (ep18) | **2.148** | **0.253** | high **1.46** | **Best all logMAE** in sweep; viz **14.41** |
| 2026-05-25 | **health10h** `R0_ref_leash` (visc3h L0 stack) | **0.827** (ep04) | **2.556** | **0.253** | high **1.25** | ep04 best then regress; viz **14.58** |
| 2026-05-25 | **health10h** `M1_mu1_only_leash` (`DISABLE_MU2`) | **0.599** (ep12) | **3.855** | **0.246** | high **1.43** | viz **14.66**; wall metric bad |
| 2026-05-25 | **health10h** `M0_mu2_cap_leash` (`MU2_SIGMOID_CAP=8`) | **1.016** (ep10) | **2.643** | **0.262** | high **0.756** | viz **15.05**; high-μ ok, all weak |
| 2026-05-25 | **health10h** `G1_gemini_mu_log` (`sweep_gemini`, `MU_LOG`) | **1.468** (ep16) | **2.428** | **0.267** | high **0.868** | viz **15.11**; Gemini **not** best |
| 2026-05-25 | **health10h** `G0_gemini_leash` (Gemini + leash + sentinel wall w) | **1.465** (ep04) | **2.403** | **0.260** | high **0.864** | viz **15.13**; `gate→0` late |
| 2026-05-25 | **health10h** `K0_carreau_kinematic` (`DATA_KINE`, Carreau-only, 8ep cold) | **1.463** (ep04) | **2.053** | **0.251** | high **1.16** | **No μ train**; viz **15.17**; t0 still weak; **viz**: biochem rollout \|u\|≈0 all t; gelation panel μ₂=80 is **diag-only** (§87) |
| 2026-05-25 | **K0_stage_a_parity_fresh** (`DATA_KINE`, 8ep, SIREN+width+Anderson+warm-start, `DETACH=1`, fresh ckpt) | **1.471** (ep06) | **2.050** | **-0.042** | high **1.232** | Preflight pass; **L_kine≈2.51 flat**; **viz score 2.71**, **t0\|u\|≈0.39**, `flow_trivial=0`; μ flat ep0–7 — architecture parity fixes **flow**, not μ |
| 2026-05-25 | **K0_stage_a_parity_fresh** viz (patient007, `biochem_teacher_best_high_mu.pth`) | **1.471** (ckpt ep0 tag) | **2.050** | **—** | high **1.232** | **§89**: t0 biochem **\|u\|~1.0–1.2** vs COMSOL **~1.5–2**; steady GINO-DEQ **~1.6**; **t≈9540** narrow **jet**; **μ₂=80** diag-only; **μ_eff** row black — open-loop ODE + macro re-DEQ |
| 2026-05-25 | **K1_delta_mu** partial (`DATA_KINE`, Δμ+`TRAIN_MU_ENCODER`, `TF=1`, `DETACH=1`, `TBPTT=12`, 7ep **OOM**) | **0.541** (ep03) | **1.841** | **0.028** bulk | high **1.292** | `L_kine` **2.11→0.81**; same physics as OomSafe; **OOM** ep7 Anderson backward |
| 2026-05-25 | **K1_delta_mu_data_kine** OomSafe (`TBPTT=5`, `workers=0`, `kin_ckpt=1`, `RK4=8`, 12ep, `20260525T102611Z`) | **0.464** (ep11) | **1.797** | **0.218** bulk ep11 | high **1.122** | **§90**: `L_kine` **2.11→0.55**; viz **t0\|u\|≈0.61**, score **1.04**; wall still high; ckpt `biochem_teacher_best_high_mu.pth` |
| 2026-05-25 | **K2_physics_triggers_on** (`COMPLEXITY_STEP=3`, `LOSS_DATA_ONLY=0`, explicit gelation, `GELATION_PRIOR_GATE=0`, Δμ+`TRAIN_MU_ENCODER`, `TF=1`, OomSafe 12ep, `20260525T105120Z`) | **4.222** (ep09) | **3.338** | **-0.050** | high **3.097** | **§91**: preflight **5.77**; **regress vs K1 0.464**; `L_tot` **~700**; viz **clot_frac=1**, **μ₂=80**; 4GB **no OOM** |

---

## References

- Module header: `src/training/train_biochem_corrector.py` (presets, complexity steps).
- Project overview: [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md).
- Corona script (experimental): `scripts/run_biochem_thrombus_corona.ps1`.
- Comprehensive μ script (experimental): `scripts/run_biochem_comprehensive_mu.ps1`.
- Teacher-best checkpoint (after teacher stage): `outputs/biochem/biochem_teacher_best.pth` — load in viz via `python -m src.evaluation.visualize_pipeline` (prefers this over `biochem_best_bio.pth`).
