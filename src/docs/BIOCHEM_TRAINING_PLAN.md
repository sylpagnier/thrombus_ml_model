# Biochem training plan

Structured roadmap for **Phase 3 biochem** (`train_biochem_corrector.py`): what to prove, in what order, and how to debug by **isolating** objectives before combining them.

**Companion docs (living evidence, not the plan):**

| Doc | Role |
|-----|------|
| [BIOCHEM_TRAINING_PROGRESS.md](BIOCHEM_TRAINING_PROGRESS.md) | Run chronicle, gate checklist, run log table |
| [PASSIVE_KIN_BLOCKER_CHECKLIST.md](PASSIVE_KIN_BLOCKER_CHECKLIST.md) | Kin-blocked passive backlog while `GT_KINE_VEL=1` |
| [CLOT_PHI_BASELINE.md](CLOT_PHI_BASELINE.md) | Wall-local clot phi (downstream of species teacher) |
| [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) | Architecture and data layout |

**Rule of thumb:** Change **one axis per leg** (loss term, mask, init, or LR). Log outcomes to the progress doc after each batch of runs.

### Probe vs promote vs scale

Early Phase I work is **not** about final model quality. It is about **trends**, **pass/fail signals**, and **catching bugs** (wrong mask, preset clobber, flow trivial, loss not wired). Scale training only when a probe matrix has picked a recipe worth promoting.

| Tier | Typical epochs | Time / leg | Purpose | When to run |
|------|----------------|------------|---------|-------------|
| **Probe** | **2-4** | ~10-20 min | Compare curves in `run.jsonl`; gates use **relaxed** thresholds (`--probe`) | Default for I.1 X3-X6, I.2 Y, most mask/LR/isolate ablations |
| **Calibration** | **12-20** | ~1-2 h | One **canonical init** per recipe family (e.g. align locked) | Once per stable recipe; not every ablation |
| **Promote** | eval + dump (+ optional short confirm) | ~30-90 min | Lock ckpt + species dump for clot-phi | Only after probes agree on mask/weights/init |
| **Scale** | 6h ladders, 20ep reruns | hours | Architecture A/B with a **written hypothesis** | When probes cannot separate two designs |

**Do not** treat probe FAIL on saturated inits (FI already ~0.03, flat `L_bio`) as a recipe bug -- reset init to `biochem_teacher_passive_align_locked.pth` each leg. **Do** treat flow trivial, exploding grads, or FI regression vs ep0 as hard fails.

**Default launchers:** `go_passive_x_probe.ps1` (**-Turbo**, target **<30 min**). **Full matrix:** `go_passive_x_iterate.ps1 -Turbo:$false`. **Promote I.1:** `go_passive_x_block_finish.ps1 -Promote`.

**Turbo env** (`_passive_x_block_env.ps1`): 2ep, 3 legs (X4/X5/m3 union), `VAL_TIME_STRIDE=50`, `TEACHER_VAL_EVERY=2`, `PASSIVE_SPECIES_VAL_ONLY=1` (one rollout/anchor, no duplicate mu val). Training code also reuses rollout for species when full val runs.

---

## Two program phases

```mermaid
flowchart LR
  subgraph P1 [Phase I - Teacher on anchors]
    P1a[Plumbing M0]
    P1b[Species X]
    P1c[Physics Y]
    P1d[Joint XY]
    P1e[Viscosity / clots M5]
    P1f[Cross-anchor M8]
  end
  subgraph P2 [Phase II - Future]
    P2a[Synthetic graphs]
    P2b[Pseudo bank]
    P2c[Corrector]
    P2d[Step 3 multitask at scale]
  end
  P1 --> P2
```

| Phase | Goal | Pipeline | Status |
|-------|------|----------|--------|
| **I — Teacher** | Anchor-only teacher that predicts **species (FI/Mat)** and **viscosity / clot fields** well enough to dump reliable anchors and support clot-phi | `STOP_AFTER_TEACHER=1`, GT flow until kin is ready | **Active** — species strong; bulk mu unlock ~0.80; wall/high + viz clots open |
| **II — Synthetic + corrector** | Train on **mixed anchor + synthetic** graphs with pseudo labels from a frozen teacher | `STOP_AFTER_TEACHER=0`, corrector loop | **Not started** — blocked on Phase I teacher quality |

Phase II is intentionally **out of scope** for current iteration. Do not enable `thrombus_corona` or large `STOP_AFTER_TEACHER=0` runs until Phase I promotion criteria are met.

---

## Milestone checklist (M0–M8)

| Milestone | Status | Evidence / gap |
|-----------|--------|----------------|
| **M0 Plumbing** | **Done** | `run.jsonl`, `runs_index.jsonl`, `go_*` launchers, `_python_rc.ps1`, passive/explore env, gate scripts, explore summarize |
| **M1 Supervised species** | **Pass (passive)** / **Partial (clot-phi)** | 20ep align: val FI **2.01->0.029**; **I.1 X block done** (§131): `species_locked` + dump `anchors_stride36_m6`. Gap: clot-phi min F1 not yet on new dump |
| **M2 Passive co-train (step 2a)** | **Pass (20ep)** / **Partial (ramp2)** | Locked align; `L_bio` + masked `ADR_S` co-descent. Gap: global/raw `ADR_S` in combined ramp2 still ~2.26e6 |
| **M3 Analytical ADR alignment** | **Partial** | Masked co-descent **pass** (20ep §126, **6ep minpass** §132 from `phaseB_ramp1`); val FI **0.25** @6ep (target **~0.03** @12ep); **global** ADR in generic ramp2 still open |
| **M4 Predicted kinematics** | **Not solved** | All biochem wins use `GT_KINE_VEL=1`. Stage-A kin is a parallel track |
| **M5 Viscosity / clot teacher** | **Partial** | Mu-unlock **all ~0.80**; isolates **~0.40–0.49** on patient007. Gap: wall/high finetune noop; viz localized clots poor |
| **M6 Corrector / synthetic** | **Not started** | Phase II |
| **M7 Full multitask (step 3)** | **Not solved** | K2 val **~4.2** vs K1 **~0.46** |
| **M8 Cross-anchor gate** | **Partial** | GT-flow round2 **min F1 0.357** (pass 0.34). Gap: retrain protocol + new teacher dumps |

**Current focus (2026-05-30):** **M3** analytical ADR alignment (masked co-descent proven; global/joint + formulation winner still open), then **I.3 XY** bridge from unlock ckpt, then **M5** / **M8** — still on **GT kinematics**.

### M3 block chunks (probe -> cal -> promote)

| Chunk | ID | Tier | Launcher | Ep (turbo / full) | Pass means |
|-------|-----|------|----------|-------------------|------------|
| 0 | M3.0 | audit | `audit_passive_adr_alignment.py --all-formulations` | — | GT mask/residual baselines logged |
| 1 | M3.1 | seed | copy `passive_species_locked` -> `biochem_teacher_best_high_mu.pth` | — | Non-saturated init for ramp1 |
| 2 | M3.2 | probe | `go_phaseB_xy_passive.ps1` ramp1 only | 3 / 3 | `phaseB_ramp1_last.pth`; `L_bio` drops |
| 3 | M3.3 | cal | Phase B ramp2 (+ADR backprop) | 4 / 6 | Masked ADR co-descent in `run.jsonl` (ignore raw global `ADR_S~1e6` alone) |
| 4 | M3.4 | cal | `go_m3_align_probe.ps1` | 6 / 12 | `check_m3_align_gate.py` **PASS** (`union` + `transport_only`) |
| 5 | M3.5 | probe | `go_m3_narrowing_90m.ps1` | 2 / 3 per leg | Rank via `summarize_m3_narrowing.py`; pick residual formulation |
| 6 | M3.6 | probe | `go_m3_adr_alignment_sweep.ps1` | 3 / 6 per leg | `check_m3_alignment_gate.py` on mask/wall legs |
| 7 | M3.7 | promote | `go_passive_lock_align_ckpt.ps1` -> `passive_m3_locked.pth` | — | `check_m3_block_pass.py` **PASS** |

**Already met (do not re-run as success criteria):** 20ep `passive_align_20ep` + §125 `m3_align_transport_union` 12ep both pass **masked** m3 gate. **Still open:** ramp2 global ADR flat; sweep/narrowing winner + locked **M3** teacher.

**Quick probe (~30-45m, trends only):** `go_m3_block_pass.ps1 -Probe` (ramp1/2 x2ep + align x3ep; gate with `--min-epochs 3`).

**Min formal pass (~1-1.5h):** `go_m3_block_pass.ps1 -MinPass` or align-only 6ep from `phaseB_ramp1_last.pth` — **PASS** logged §132 (`m3_align_transport_union_minpass`). Species cal: extend to **12ep** for FI **~0.03**.

**One-liner (full wrap):** `go_m3_block_pass.ps1` (~8-12h) or `-Turbo` (~3-5h); cal+lock only: `-SkipNarrowing -SkipSweep` (~2h).

---

## Isolation framework (X, Y, XY)

Debug by separating **what is supervised in backward** from **what is only logged**.

| Track | Meaning | Typical `LOSS_ISOLATE` / backward | Trainable (usual) |
|-------|---------|-----------------------------------|-------------------|
| **X** | **Species / data bio** — FI, Mat, bulk substance on COMSOL anchors | `PASSIVE`, `DATA_BIO`, or step-2 `L_Data_Bio` in `LOSS_DATA_ONLY` | Bio encoder, decoder, ODE |
| **Y** | **Single physics / mu term** — one knob at a time | `ADR_S`, `ADR_F`, `W_BIO`, `W_PHY`, `MU_LOG`, `MU_SI`, … | Term-dependent; mu legs often **freeze bio** |
| **XY** | **Combination** — joint step-2, bridge, ramps, mu-unlock + species guard | `LOSS_DATA_ONLY=1` + weights; or `PASSIVE` + ADR weight | Multi-group per recipe |

**Always fix flow first:** `BIOCHEM_GT_KINE_VEL=1`, `BIOCHEM_GT_KINE_SKIP_DEQ=1`, `BIOCHEM_TEACHER_FORCE_MIN=1` for passive/GT-flow work.

**Orchestrated ladder:** [scripts/go_passive_explore_6h.ps1](../../scripts/go_passive_explore_6h.ps1) runs X -> Y -> XY legs; [scripts/_passive_explore_base_env.ps1](../../scripts/_passive_explore_base_env.ps1) sets clean env (no `passive_transport` preset clobber).

### Gate scripts (per track)

| Track | Gate | Pass means |
|-------|------|------------|
| X / XY (species+ADR) | `check_m3_align_gate.py` | `L_bio` and masked `ADR_S` ratios; species FI stable. **Caveat:** false FAIL if init already saturated (~2275 `L_bio`) |
| X / XY (species only) | `val_species_fi_log_mae`, train-anchor eval | FI **~0.03** on val + train anchors |
| Y | `check_phase_a_gate.py --mode y --term <TERM>` | Isolated train loss monotone / non-trivial |
| XY bridge | `check_passive_step2_bridge_gate.py` | `passive_step2_bridge=1`, species OK, mu stable |
| XY mu-unlock | `check_passive_mu_unlock_gate.py` | Val all logMAE drop; species FI **~0.03** |

### Metrics that matter (ignore misleading ones)

| Use | Do not use alone |
|-----|------------------|
| Val `mu_log_mae` (all / wall / high-mu), `mu_pearson` | Train `L_kine` for mu success |
| Val `val_species_fi_log_mae`, train-anchor FI/Mat | Train `L_bio` when bio is frozen |
| Train `L_Back` / isolated term under `LOSS_ISOLATE` | Global raw `ADR_S` when mask is clot-band |
| Preflight median logMAE | `L_tot` under step-3 Kendall |

---

## Loss function catalog

**Enforced in code:** [biochem_loss_policy.py](../training/biochem_loss_policy.py) (`BIOCHEM_LEGACY_LOSSES=1` for old sweeps).

**Source of truth (implementation):** `compute_biochem_loss()` and `_biochem_resolve_isolated_loss()` in [train_biochem_corrector.py](../training/train_biochem_corrector.py). **Evidence:** [BIOCHEM_TRAINING_PROGRESS.md](BIOCHEM_TRAINING_PROGRESS.md) run table + chronicle + **Loss policy** section.

### How losses enter `backward`

| Mode | Env | Backprop sum |
|------|-----|--------------|
| **Isolate** | `BIOCHEM_LOSS_ISOLATE=<TERM>` | Single term (or composite below) |
| **Step 2 data-only** | `BIOCHEM_LOSS_DATA_ONLY=1`, no isolate | `L_Data_Kine + L_Data_Bio + W_MuSI*L_MuSI + W_MuLog*...` (+ optional `L_PhysTemp`, passive ADR) |
| **Step 3 multitask** | `BIOCHEM_COMPLEXITY_STEP=3` | Kendall-weighted 8 tasks + aux terms |
| **Passive preset** | `LOSS_ISOLATE=PASSIVE` | `w_bio*L_Data_Bio + w_kine*L_Data_Kine` [+ ADR if `PASSIVE_ADR_BACKPROP=1`] |

### A. Pretrain (before teacher loop)

| Metric | What | Tested? | Performance |
|--------|------|---------|-------------|
| AE recon + latent reg | Huber on normalized bio channels | Yes (default pipeline) | Standard warm-start; skipped when `SKIP_PRETRAIN=1` |
| ODE-RXN mimic | Reaction path on latent | Yes | Plateau-driven; not primary μ/species gate |

### B. Eight Kendall tasks (`DynamicLossWeighter`, step 3)

| # | Metric | Physics | Isolate key | Tested? | Performance (patient007 unless noted) |
|---|--------|---------|-------------|---------|--------------------------------------|
| 0 | `L_ADR_F` | Fast ADR residual | `ADR_F` | **Yes** (Phase A Y, explore `Y_ADR_F`) | Train loss **~1e4 -> ~4e2** @ LR=1e-3, clip=10 (§107); val mu flat **1.3966** |
| 1 | `L_ADR_S` | Slow ADR residual | `ADR_S` | **Yes** (Phase A Y, explore `Y_ADR_S`, m3n) | **Masked** co-descent **0.007->0.0003** (20ep align §126); **global** raw **~2.26e6** flat in ramp2 |
| 2 | `L_W_Bio` | Wall bio flux | `W_BIO` | **Yes** (explore `Y_W_BIO`) | Phase-A gate **WARN** (trivial/rebound); not reliable alone |
| 3 | `L_W_Phy` | Wall physics flux | `W_PHY` | **Yes** (explore `Y_W_PHY`, Phase A) | Train **~0.6->0.1** with clip=10; finicky |
| 4 | `L_B_IO` | Bio in/out | `BIO_IO` | **Partial** (Phase A seed) | Numerical descent; **val mu flat** |
| 5 | `L_mom` | NS momentum residual | `NS_MOM` | **Little** | Dominated by step-3 runs; not isolated success |
| 6 | `L_Data_Kine` | Supervised u,v,p,mu on anchors | `DATA_KINE` | **Yes** (K1, I4, K10e aux) | Val logMAE **~0.47-0.49** with mu-path + delta head (§90) |
| 7 | `L_Data_Bio` | Supervised FI/Mat/species | `DATA_BIO` | **Yes** (passive, I3, 20ep) | Val FI **2.01->0.029**; train `L_bio` **16.6k->225**; **val mu flat** when mu not in loss |

### C. Mu / viscosity aux terms (added outside Kendall index)

| Metric | Isolate / access | Tested? | Performance |
|--------|------------------|---------|-------------|
| `L_MuSI_aux` | `MU_SI` (bundle) | **Yes** | Best **~0.44** ep3 (Phase B / I2); competes with MU_LOG |
| `L_MuLog_aux` (all-truth) | `MU_LOG` | **Yes** | **~0.40-0.49** isolates; passive unlock **0.80**; finetune **noop** |
| `L_MuLog_wall` | `MU_LOG_WALL` | **Yes** (sweeps) | Can move wall to **~2.09** but **hurts** all/high (**~0.66-0.70**) |
| `L_MuLog_high` | `MU_LOG_HIGH` | **Yes** | High-tail **0.94->0.58** isolate; **no** spatial clots; all-truth poor alone |
| `L_MuMSE` | `MU_MSE` / `MU_DATA` | **Yes** (K10d) | Proof path for delta-mu SI head |
| `L_MuLog_adjacent` | part of `K10E` | **Yes** (K10e) | Wall-adjacent band; logMAE **~0.47-0.49**; **viz clots still flat** |
| `L_MuWall_bypass` | in `MU_LOG` | Partial | Used in MU_LOG bundle |
| `L_MuLog_boundary` | step-2 only | Little | Optional boundary weight |
| K10E bundle | `K10E` | **Yes** | Adjacent + bulk delta + small DATA_KINE |

### D. Other aux terms

| Metric | Isolate | Tested? | Performance |
|--------|---------|---------|-------------|
| `L_PhysTemp` | `PHYS_TEMP` | **Yes** (overnight B) | **No val mu gain** vs baseline step-2 |
| `L_KinePrior` | `KINE_PRIOR` | Little | Carreau/clot-risk prior; not main lever |
| `L_Latent_Reg` | `LATENT` | Default on | ODE derivative energy; scaled |
| `L_Visc_Reg` | `VISC` / `VISC_REG` | Little | Curriculum viscosity reg |
| `L_Pseudo` | `PSEUDO` | **No** (corrector) | Phase II — teacher frozen |
| `L_FIGateStart` | `FI_GATE` | Little | FI gate start penalty |
| `L_ResidualSparse` | `RES_SPARSE` | Little | Sparse residual reg |
| Trigger floors/sparse/nuc | env weights | **Yes** (K11/clot6h era) | **Fail** viz: `gate_all` collapses, `clot_frac->0` (§112) |

### E. Composite isolate modes

| Isolate | Composition | Tested? | Performance |
|---------|-------------|---------|-------------|
| `PASSIVE` / `ONE_WAY` | `w_bio*L_Data_Bio + w_kine*L_Data_Kine` [+ ADR] | **Yes** | **ADR in backward early:** grad explode, `L_bio` flat; **`PASSIVE_ADR_BACKPROP=0`:** species pass; masked ADR co-descent @ 20ep |
| Step-2 `LOSS_DATA_ONLY` | data + mu weights (+ bridge ADR) | **Yes** | Bridge: species **0.027**, mu **flat 1.3966** under `mu_ratio_max=1` |
| Step-3 Kendall sum | all 8 tasks | **Yes** (K2) | Val **5.58->4.22** — **regress** vs K1 **0.46** |

### F. Legacy / separate tracks

| Name | Notes | Tested? | Performance |
|------|-------|---------|-------------|
| **K11 clot gate** | Documented sweeps used `LOSS_ISOLATE=K11` (clot BCE); **not** in current `_biochem_resolve_isolated_loss` valid list — verify script before re-run | **Yes** (clot6h) | **Fail** localized viz; wall halo; `gate` collapse |
| **Clot-phi** (`train_clot_phi_simple`) | Separate model on dumped species — not a `train_biochem_corrector` loss | **Yes** | GT-flow **min F1 0.357**; patient007 F1 ~0.48 simple baseline |

### G. Quick map: X / Y / XY vs losses

| Track | Primary losses | Status |
|-------|----------------|--------|
| **X** | `DATA_BIO`, `PASSIVE` (data part) | **Pass** species (~0.03 FI) |
| **Y** | `ADR_*`, `W_*`, `MU_LOG`, `MU_SI`, ... | **Partial** — ADR_S/F and MU_LOG pass; W_BIO/W_PHY finicky; wall/high noop at plateau |
| **XY** | `LOSS_DATA_ONLY`, bridge ADR, unlock+bridge | **Partial** — unlock **0.80** mu; bridge species OK; analytical ADR global still open |

**Not tested / defer:** `PSEUDO`, full corrector mix, step-3 at production quality, teacher without `GT_KINE_VEL`, most trigger/sparse clot priors at scale.

---

## Phase I — Teacher (viscosity + species on anchors)

### I.0 — Plumbing (M0) [done]

- Compact logging, run index, promote/lock ckpt scripts, pytest preflight on launchers.
- **Promotion paths:** `biochem_teacher_passive_align_locked.pth`, `expl6h_*_last.pth`, `biochem_teacher_passive_mu_unlock_best.pth`.

### I.1 — X: Species lane (M1)

**Goal:** COMSOL **FI / Mat** on anchors with **frozen GT flow** and **no clot feedback** (`TEACHER_MU_RATIO_MAX=1`). **Probe goal:** see FI trend / isolate bugs; **promote goal:** dump for clot-phi only after probes pick a recipe.

| Step | ID | Tier | What to test | Launcher | Pass (probe) | Pass (promote) |
|------|-----|------|--------------|----------|----------------|----------------|
| 1 | X0 | probe | GT flow sanity | `go_passive_transport.ps1` (short) | `flow_trivial=0`, `t0|u|~0.96` | — |
| 2 | X1 | **calibration** | Union mask + `transport_only` ADR | `go_m3_align_probe.ps1` -> `go_passive_align_20ep.ps1` | — | `check_m3_align_gate`; val FI **<0.05** |
| 3 | X2 | **calibration** | Lock canonical init | `go_passive_lock_align_ckpt.ps1` | — | `biochem_teacher_passive_align_locked.pth` |
| 4 | X3 | probe (3ep) | Mask ablation | `go_passive_x_iterate.ps1` | `check_passive_x_species_gate --probe`; compare `run.jsonl` | — |
| 5 | X4 | probe (3ep) | `DATA_BIO` vs `PASSIVE` | same | FI not worse vs ep0; note `L_bio` path | — |
| 6 | X5 | probe (3ep) | FI/Mat weights | same | No FI regression vs ep0 | — |
| 7 | X6 | probe | Train-anchor table | `eval_passive_species_anchors.py` on locked init | Informational FI table | — |
| 8 | X7 | **promote** | Dump for clot-phi | `go_passive_x_block_finish.ps1 -Promote` | — | Train anchors FI **<0.04**; `anchors_stride36_m6` |

**Done (calibration):** X1–X2 (20ep align). **Done (probe + promote, 2026-05-30):** X3–X5 + `X_m3_union` turbo 2ep (`go_passive_x_probe.ps1`); promote from `passive_align_locked` -> `biochem_teacher_passive_species_locked.pth` + dump `outputs/biochem/x_block/anchors_stride36_m6/` (6 graphs); `check_passive_x_block_pass.py --require-promote` **PASS**. Probe val FI (trend only): X3 **0.018**, X4/X5/m3 **~0.065**; teacher quality = 20ep cal **~0.03**. X6 confirm skipped (promote uses calibration ckpt). **Re-verify:** `python scripts/check_passive_x_block_pass.py --require-promote`.

**Default session (probe matrix, ~30 min turbo):** `go_passive_x_probe.ps1`. **Promote:** `go_passive_x_block_finish.ps1 -Promote`. **Full pass (probe + promote):** `go_passive_x_block_pass.ps1`.

### I.2 — Y: Isolated terms (M2 / M3 debug)

**Goal:** Prove each **backward term can move** (probe, **3-4 ep**) before XY joint training. Use `check_phase_a_gate.py --mode y` with **trend** interpretation; do not run 8ep+ Y legs unless comparing two architectures.

| Step | ID | Term | Tier | Pass (probe) |
|------|-----|------|------|--------------|
| 1 | Y1 | `ADR_S` + `transport_only` | probe | Masked ADR down or flat FI; no explode |
| 2 | Y2 | `ADR_F` | probe | Term loss moves; watch clip |
| 3 | Y3 | `W_BIO` / `W_PHY` | probe | Non-trivial loss (rebounds OK in probe) |
| 4 | Y4 | `MU_LOG`, bio frozen | probe | Mu metric moves; FI not worse vs ep0 |
| 5 | Y5 | wall/high weights | probe | **Often noop at plateau** -- stop after 3ep; change recipe, do not extend epochs |
| 6 | Y6 | ADR formulation | probe | 3ep m3n-style compare only |

**Done (explore 6h):** Y1–Y4 signals. **Failed / noop:** Y5 finetune at plateau (extend training did not help).

### I.3 — XY: Combine species + physics (M2 joint, M3, M5)

**Goal:** Step-2 teacher — species + modest mu + optional masked ADR — **without** step-3 Kendall.

| Step | ID | Recipe | Launcher | Pass criteria |
|------|-----|--------|----------|---------------|
| 1 | XY0 | Phase B ramp1 (data only) | `go_phaseB_xy_passive.ps1` (ramp1) | `L_Data_Bio` clear drop |
| 2 | XY1 | Ramp2 (+ADR backprop, low weight) | ramp2 leg | Data down; **masked** ADR co-descent (not global ADR alone) |
| 3 | XY2 | **Step-2 bridge** | `go_passive_step2_bridge.ps1` | `check_passive_step2_bridge_gate`; species **~0.03** |
| 4 | XY3 | **Mu-unlock** then bridge | `go_passive_mu_unlock_probe.ps1` -> bridge from **unlock** ckpt | All logMAE **~0.80**; species held |
| 5 | XY4 | Low ADR in backward | explore `XY_adr_low` | Species stable + masked ADR ratio |

**Anti-patterns (do not repeat):**

- `BIOCHEM_PRESET=passive_transport` + mu-unlock (zeros `TRAIN_MU`).
- `BIOCHEM_REUSE_LAST_PRETRAIN=1` after align (clobbers species).
- Judging X legs only with `check_m3_align_gate` from **saturated** locked init without fresh ramp1 init.

**In progress:** XY3 bridge from `expl6h_XY_mu_unlock_last.pth`.

### I.4 — Viscosity / clot fields (M5) — core Phase I outcome

**Goal:** Teacher predicts **viscosity fields** that match COMSOL on anchors — including **localized clot bands** — not only bulk logMAE.

Sub-tracks (still X/Y/XY inside mu):

| Sub | Track | What | Notes |
|-----|-------|------|-------|
| V1 | Y | `MU_LOG` isolate, `DELTA_MU_HEAD`, `mu_ratio_max=20` | **Pass** ~0.80 all-truth on patient007 |
| V2 | Y | Wall / high-mu weighted `MU_LOG` | **Fail** — 8ep finetune flat |
| V3 | XY | Step-2 bridge + modest `W_MuLog` / `W_MuSI` | Species OK; mu may stay capped if `mu_ratio_max=1` |
| V4 | Y | Wall-adjacent / split heads (K4–K10e family) | For **spatial** clots; use when bulk unlock stalls |
| V5 | Viz | `visualize_pipeline --teacher-only`, clot_frac, gate_all | **Fail** on many legs — separate from logMAE |
| V6 | Clot-phi | GT-flow ladders on dumped species | Parallel read on **localization** (F1), not teacher ADR |

**Promotion criteria (teacher "good enough" for Phase II prep):**

1. Species: val + train-anchor FI **~0.03** (passive lane) — **met**.
2. Bulk mu: val all logMAE **<=0.85** on patient007 with unlock recipe — **met** (~0.80).
3. Wall/high: wall logMAE improving OR explicit decision to defer to split-head — **not met**.
4. Viz: localized clot channel qualitatively non-trivial on patient007 — **not met**.
5. Dump: `dump_teacher_species_to_anchors` stable (ladder m6 protocol) — **partial**.
6. Cross-anchor: clot-phi min F1 **>=0.34** with retrain from promoted dump — **partial** (0.357 on old cache).

### I.5 — Cross-anchor clot-phi (M8)

Depends on **I.1 dump** + promoted teacher. See [CLOT_PHI_BASELINE.md](CLOT_PHI_BASELINE.md) and `go_gt_flow_*` scripts. Not a substitute for fixing teacher ADR/mu.

### I.6 — Predicted kinematics (M4) — parallel

When Stage-A `kinematics_best.pth` is trusted: turn off `GT_KINE_VEL`, enable kin LoRA, re-run **X0** and **Y4** smoke before any long XY.

---

## Phase II — Synthetic graphs + corrector (future)

**Not started.** Prerequisites:

| Prerequisite | Phase I item |
|--------------|--------------|
| Stable teacher ckpt | Locked align + unlock and/or bridge winner |
| Species dump quality | I.1 X6 + dump protocol |
| Mu acceptable on anchors | I.4 promotion criteria (at least bulk + species) |
| Step-2 joint stable | I.3 XY2/XY3 without preset clobber |

### Planned stages (outline only)

| Stage | Goal | Env sketch |
|-------|------|------------|
| II.0 | Pseudo-label bank from frozen teacher | `STOP_AFTER_TEACHER=1` generate; audit FI/Mat/mu on synthetics |
| II.1 | Corrector smoke (synthetic-only) | Small graph subset; data loss only |
| II.2 | Anchor + synthetic mix | Curriculum fraction; monitor anchor val |
| II.3 | Step 2.5 temporal | `DATA_ONLY_PHYS_TEMP` on trajectories |
| II.4 | Step 3 multitask | `COMPLEXITY_STEP=3`, only after II.2 stable |
| II.5 | Optional spatial priors | `GELATION_PRIOR_GATE`, corona hops — **one flag at a time** |

Do **not** use `BIOCHEM_PRESET=thrombus_corona` as an entry point.

---

## Suggested run order (next 2–3 sessions)

Probe-first (default):

1. **I.1 X probes:** `go_passive_x_iterate.ps1` (3ep matrix); `summarize_passive_x_block.py`; promote only if mask/iso winner is unclear from logs.
2. **I.3 XY probe:** `go_passive_step2_bridge.ps1` **short** (e.g. 6ep) from `expl6h_XY_mu_unlock_last.pth` -- trend only, not quality chase.
3. **I.2 Y probe:** single term at a time via `go_phase_a_xy_iterate.ps1 -YOnly -EpochsY 3` when debugging ADR/mu wiring.
4. **Promote dump** (`go_passive_x_block_finish.ps1 -Promote` or bridge winner) **once** before clot-phi.
5. **M8 clot-phi:** GT-flow round on that dump; compare to 0.357 -- scale hours only if probe F1 separates recipes.

Avoid until hypothesis is clear: `go_passive_explore_6h.ps1`, 20ep reruns on saturated inits, 8ep finetune at mu plateau.

---

## Complexity vs plan phases

| Code level | Plan phase |
|------------|------------|
| Step 2a passive | I.1 X |
| Phase A/B isolates | I.2 Y, I.3 XY0–XY1 |
| Step 2 bridge | I.3 XY2–XY4 |
| Mu-unlock / K10* | I.4 V1–V4 |
| Step 3 | Phase II.4 |
| Corrector | Phase II.1–II.2 |

Full complexity table: [BIOCHEM_TRAINING_PROGRESS.md](BIOCHEM_TRAINING_PROGRESS.md) (top section).

---

## Document maintenance

- Update **milestone Status** in this file when promotion criteria change materially.
- Append **run evidence** to [BIOCHEM_TRAINING_PROGRESS.md](BIOCHEM_TRAINING_PROGRESS.md) (run table + chronicle), not duplicate long logs here.
- When adding a new `go_*.ps1` leg, add one row to the relevant I.x table and [scripts/README.md](../../scripts/README.md).
