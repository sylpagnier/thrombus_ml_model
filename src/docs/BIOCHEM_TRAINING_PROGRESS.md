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
| **2+** | Thrombus corona bundle | Step 2 + gelation prior gate + 3-hop corona + phys temp | **Teacher + full corrector** (mixed graphs, pseudo labels) | `BIOCHEM_PRESET=thrombus_corona`, `STOP_AFTER_TEACHER=0` |
| **3** | Full multitask | Kendall sum: PDE + walls + ADR + data heads (not data-only) | Full corrector, LoRA on | `BIOCHEM_COMPLEXITY_STEP=3` → forces `LOSS_DATA_ONLY=0` |
| **Prod** | Long schedules | Step 2 or 3 + long AE/ODE/teacher/corrector | Overnight wall time | `BIOCHEM_PRESET=overnight_step2` (still step-2 loss tier) |

**“All losses”** in code terms = **complexity step 3** (`BIOCHEM_LOSS_DATA_ONLY=0`): physics Kendall terms enter `backward()`, not only metrics.

**“Full run”** in product terms = **`thrombus_corona` + corrector to completion** (`STOP_AFTER_TEACHER=0`), with μ and species stable on val anchors — usually **after** step-2 teacher is healthy, then step 2.5 / corona, then step 3 if VRAM allows.

---

## Where we are now (2026-05)

### Gate checklist

| Gate | Target (teacher) | Status | Notes |
|------|------------------|--------|--------|
| Preflight μ (train anchors, t0→t1) | median logMAE ≲ 2.5 | **Pass** | ~1.43–1.45 |
| Val μ (held-out anchor, e.g. patient007) | logMAE → **0.25** (env target) | **Fail** | ~1.47–1.60 typical; best recent **1.4666** (low-TF probe) |
| Val spatial correlation `r` | ≳ 0.5+ stable | **Partial** | Low-TF run: **0.40 → 0.43** ep0→1 |
| Wall μ logMAE | ≲ 1.5 | **Fail** | ~1.88–2.30; improving with low TF |
| `L_bio` on anchors | Decrease without μ stall | **Pass** | Species fit is easy |
| Phase A: `MU_SI` isolate, TF≈1 | Val logMAE drops | **Fail** | Flat ~1.59; train `L_MuSI` only |
| Phase B: `MU_SI` + low TF | Val logMAE drops | **Partial / stalled** | Stride=1, TBPTT=4: ep0→1 **1.474→1.467**. TBPTT=2, stride=10 val: **flat ~1.49** over 12 ep |

### Distance to full run (honest)

- **Step-2 teacher “done”**: Not yet — need val logMAE materially below **~1.2** (interim) before trusting corrector mix; official target **0.25** is still far.
- **Step 2.5 / thrombus_corona**: **1–2 weeks of iteration** after μ moves reliably under step-2 losses (estimate; GPU-bound).
- **Step 3 (all PDE losses in backward)**: **Blocked** until (1) μ + bio stable at step 2, (2) `DETACH_MACRO_STATE=0` stable without OOM, (3) adjoint not dominating with junk gradients.
- **Overnight / production**: Run only after fast probes pass with `VAL_TIME_STRIDE=10`; confirm once with `stride=1`.

**We are roughly at: late step-2 diagnostics / early step-2 training** — not at corona-full or step 3.

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
- **Fix (in progress)**: Low teacher forcing, higher `W_MuSI`, align loss with logMAE (future code), wall-weighted μ.
- **Status**: Ongoing.

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
- **Status**: **First clear val μ movement** — continue this direction before full multitask.

### 7. Validation slow (stride myth)

- **Symptom**: ~**2100 s** (~35 min) per val with `BIOCHEM_VAL_TIME_STRIDE=1` *and still* ~**2065 s** with **`stride=10`** on patient007 (large graph, DEQ + micro-ODE per retained step).
- **Cause**: Stride reduces **macro time indices**, not node count; each forward remains heavy.
- **Fix**: For iteration use **`BIOCHEM_TEACHER_SKIP_VAL=1`** and watch train `L_tot` / `L_MuSI`, or **`BIOCHEM_MAX_LOAD_VESSELS=1`** / smaller anchor for dev; full val only when needed. Final report: `stride=1` once.
- **Status**: Documented — do not expect 10× speedup from stride alone on this workload.

### 8. Preflight vs training μ cap

- **Symptom**: Preflight median ~1.44 at cap **1.0**; val ~1.51 at cap **80**.
- **Fix (todo)**: Run preflight at same `BIOCHEM_TEACHER_MU_RATIO_MAX` as epoch 0.
- **Status**: Known mismatch.

### 9. Preset overwrites env

- **Symptom**: `thrombus_corona` sets `DATA_ONLY_PHYS_TEMP=1` even if user set `0`.
- **Fix**: Use **`BIOCHEM_STOCK_DEFAULTS=1`** and no preset for μ probes; or re-export vars after preset (preset runs at import).
- **Status**: Documented.

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

---

## Code / architecture backlog (μ)

Ordered by impact:

1. **`L_mu_log`** on all TBPTT timesteps (match val metric).
2. **Multi-step `L_MuSI`** (not only `pred_final`).
3. **Preflight** uses `BIOCHEM_TEACHER_MU_RATIO_MAX`.
4. **`BIOCHEM_GELATION_USE_MODEL_SPECIES`** — decouple μ gelation from TF-injected GT species.
5. **Rheology-only optimizer group** (`learned_clot_penalty`, `mu_encoder`; teacher currently `freeze_lora=True`).
6. **`BIOCHEM_FAST_MU_PROBE=1`** preset: **`SKIP_VAL=1`** or tiny dev graph — not “stride=10 ⇒ fast val” on patient007.

**Increasing complexity order (do not skip):** (A) align train loss with val `logMAE` (`L_mu_log`, multi-step μ) → (B) widen TBPTT + joint **step-2** (`DATA_KINE`+`MU_SI` or clear isolate) → (C) **step 2.5** `L_PhysTemp` → (D) **thrombus_corona** corrector → (E) **step 3** multitask only if stable.

---

## Recommended run profiles

### Fast μ probe (iteration)

```powershell
$env:BIOCHEM_STOCK_DEFAULTS = "1"
$env:BIOCHEM_SKIP_PRETRAIN = "1"
$env:BIOCHEM_STOP_AFTER_TEACHER = "1"
$env:BIOCHEM_LOSS_ISOLATE = "MU_SI"
$env:BIOCHEM_MU_SI_ANCHOR_AUX_WEIGHT = "8.0"
$env:BIOCHEM_TEACHER_MU_RATIO_MAX = "80.0"
$env:BIOCHEM_TEACHER_FORCE_MIN = "0.0"
$env:BIOCHEM_TEACHER_TF_WARMUP_EPOCHS = "4"
$env:BIOCHEM_VAL_TIME_STRIDE = "10"
$env:BIOCHEM_TEACHER_VAL_EVERY = "5"
$env:BIOCHEM_TEACHER_EPOCHS = "12"
$env:BIOCHEM_TBPTT_MAX_WINDOW = "4"   # avoid 2 — too short for μ generalization (see chronicle §10)
$env:BIOCHEM_NO_TEACHER_DEFAULTS = "1"
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

### Full corona run (not yet)

```powershell
.\scripts\run_biochem_thrombus_corona.ps1
# STOP_AFTER_TEACHER=0 → teacher + corrector + pseudo bank
# Only after val mu_log_mae < ~1.2 stable on stride=10
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

---

## References

- Module header: `src/training/train_biochem_corrector.py` (presets, complexity steps).
- Project overview: [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md).
- Corona script: `scripts/run_biochem_thrombus_corona.ps1`.
