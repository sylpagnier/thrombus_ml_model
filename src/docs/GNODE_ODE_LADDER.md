# GNODE-ODE introduction ladder (rungs 9.0–9.9)

**Goal:** Bring in **GNODE-ODE** component by component — simplest wiring first, **fast iterations** for feasibility/trends, then longer runs only after each gate passes.

**Prerequisites (clot side, already done):** rungs **0–6a** (localized clot-phi on GT flow). **6b** waits on `kinematics_best.pth` retrain.

**Success metric for this track:** clot-phi on **dumped species** + **scatter viz patches** — not global `mu_log_mae` or `viz_final_mu2_mean` alone.

---

## Design rules (fast vs full)

| Fast (trend / feasibility) | Full (promote / lock) |
|----------------------------|------------------------|
| 3–8 teacher epochs | 12–20ep align + lock |
| `BIOCHEM_VAL_TIME_STRIDE=10` | stride 1–2 on val |
| `BIOCHEM_GT_KINE_VEL=1` (+ optional `GT_KINE_SKIP_DEQ=1`) | solved DEQ each step |
| `BIOCHEM_PASSIVE_ADR_BACKPROP=0` until species OK | ADR co-train |
| `dump --time-stride 72 --min-steps 2` | stride 36, min 6 |
| clot-phi **20–30ep** + multi-anchor | 60ep |
| patient007 scatter only | all anchors |

Always: **`BIOCHEM_STOP_AFTER_TEACHER=1`** for teacher-only legs. Check **clot-φ** after every dump.

**Viz helpers:** dot-source [`scripts/_gnode_viz_helpers.ps1`](../../scripts/_gnode_viz_helpers.ps1) — headless PNGs under `outputs/biochem/viz/`; optional interactive slider via `visualize_pipeline`.

---

## What “simplest GNODE-ODE” means here

Full `GNODE_Phase3` = encoder + ODE-RXN + **macro loop** (kine solve + species update + mu path + TBPTT).

We peel the onion:

```text
9.0  No GNODE — steady GINO-DEQ only (Stage A)
9.1  GNODE forward-only — GT [u,v,p], no backward (smoke)
9.2  Pretrain only — AE + ODE-RXN, 3–6ep
9.3  Teacher isolate — L_Data_Bio only, 3ep
9.4  Passive transport — species TBPTT, 8–12ep  (canonical 9)
9.5  Dump -> clot-phi — spatial gate
9.6  + masked ADR backprop — still GT vel
9.7  Mu unlock probe — MU_LOG, bio frozen
9.8  Step-2 bridge — species + modest mu
9.9  Full teacher recipe — clot_band mask, 12ep+
10+  Predicted kine (6b / rung 11)
```

---

## Rung table

| Rung | Adds this component | ~Wall time | Gate (trend) | Launcher / command |
|------|---------------------|------------|--------------|-------------------|
| **K0** | Stage-A **GINO-DEQ** weights | 2–10h | Steady kin on p007: structured flow, rel_L2 not absurd | `go_kinematics_foundation.ps1 -Fresh` or recovery sweep |
| **9.0** | **No GNODE** — DEQ on biochem mesh | **5 min** | `rel_L2(uvp)` OK vs GT; not trivial flow | `visualize_pipeline --steady-kin-only` (interactive) |
| **9.1** | **GNODE load + 1 val forward** | **10–15 min** | Forward completes; `flow_trivial=0`; FI not NaN | `go_gnode91_smoke.ps1` (+ teacher PNG) |
| **9.2** | **AE + ODE-RXN** pretrain | **30–60 min** | Pretrain loss down; no OOM | Pretrain flags; optional `snapshot_biochem_teacher.py` |
| **9.3** | Teacher **DATA_BIO only** | **20–40 min** | Val FI trending down (target **< 0.15** on 3ep) | X-probe matrix + teacher snapshot per leg |
| **9.4** | **Passive transport** (full 9) | **1–2 h** | Val FI **< 0.05**; train anchors ~0.03 | `go_passive_transport.ps1` (auto PNG; `-SkipViz` off) |
| **9.5** | **Species dump -> clot-phi** | dump **1–2h** + train **30 min** | min F1 **>= 0.26**; p007 patches in viz | `go_passive_transport_clotband_focus.ps1` (teacher + mask + clot PNGs) |
| **9.6** | **ADR backprop** on | **+8ep** | `L_bio` stable; masked ADR not exploding | Same + `Invoke-GnodeRungVizCheckup` after promote |
| **9.7** | **Mu unlock** (probe) | **~1 h** | Val all logMAE **< ~0.85** trend | `go_passive_mu_unlock_probe.ps1` + teacher snapshot |
| **9.8** | **Step-2 bridge** | **~1 h** | Species held; mu not worse | `go_passive_step2_bridge.ps1` + teacher snapshot |
| **9.9** | **Full teacher** (clot_band, TBPTT) | **2–4 h** | Clot-phi on dump beats 9.5; viz bands | Long teacher + full clotband_focus viz stack |

**Rung 10 (plan):** replace GT vel with **GINO-DEQ** each step (same as ladder **6b** + full GNODE species). **Rung 11:** corrector / synthetics (Phase II).

---

## Rung 9.1 — GNODE forward smoke

Proves the **graph + GNODE_Phase3** loads on biochem anchors (1 teacher epoch, GT flow).

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_gnode91_smoke.ps1"
# Optional interactive slider after PNG:
powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\go_gnode91_smoke.ps1" -InteractiveViz
```

**Read:** `run.jsonl` `event=val` — `flow_trivial=0`, `val_viz_t0_speed_mean` ~0.9+ with GT vel; `val_viz_health_score` (lower=better, passive mu pinned so score is coarse).

**Viz artifacts:** `outputs/biochem/viz/gnode91_*_patient007.png` (headless: |u|, FI/Mat vs GT at t=0 and t_final).

Manual snapshot on any ckpt:

```powershell
python scripts/snapshot_biochem_teacher.py --checkpoint outputs/biochem/biochem_teacher_last.pth --anchor patient007
```

---

## Visualization per rung (checkup cadence)

| Rung | Auto (launchers) | Manual / optional |
|------|------------------|-------------------|
| **9.0** | — | `python -m src.evaluation.visualize_pipeline --steady-kin-only --anchor patient007` |
| **9.1** | `go_gnode91_smoke.ps1` -> teacher PNG | `-InteractiveViz`; `run.jsonl` scalars |
| **9.2** | — | Snapshot after pretrain if teacher ckpt exists |
| **9.3–9.4** | `go_passive_transport.ps1` -> teacher PNG | `visualize_pipeline --teacher-only` between epochs |
| **9.5** | `go_passive_transport_clotband_focus` -> teacher + mask + clot PNGs | `eval_clot_phi_multi_anchor.py` (metrics) |
| **9.6–9.9** | Re-run snapshot after each promote | Full interactive slider before locking ckpt |

**Headless commands** (from repo root, after dot-sourcing `_gnode_viz_helpers.ps1`):

- `Invoke-BiochemTeacherSnapshot` -> `scripts/snapshot_biochem_teacher.py`
- `Invoke-BiochemTeacherClotbandViz` -> `scripts/snapshot_biochem_teacher_clotband.py` (same 2x2 phi/mu layout as clot-phi)
- `Invoke-ClotPhiScatterViz` -> `viz_clot_phi_simple --plot-mode scatter`
- `Invoke-ClotPhiMaskViz` -> `viz_clot_phi_masks`
- `Invoke-GnodeTeacherInteractiveViz` -> `visualize_pipeline --teacher-only`

**Scalar-only during train:** every val epoch logs `viz_*` fields in `outputs/reports/training/biochem/<run_id>/run.jsonl` (full rollout val unless `BIOCHEM_PASSIVE_SPECIES_VAL_ONLY=1`).

---

## Component map (what each piece does)

| Component | Module / env | Turned on at |
|-----------|--------------|--------------|
| Biochem graph + schema | anchor `.pt` | always |
| Stage-A **GINO-DEQ** | `kinematics_best.pth` | K0, 9.0; inside GNODE macro step |
| **GT velocity** | `BIOCHEM_GT_KINE_VEL=1` | 9.1–9.9 (default for fast track) |
| Skip DEQ (use GT u,v) | `BIOCHEM_GT_KINE_SKIP_DEQ=1` | 9.1–9.3 only |
| Species encoder + **ODE-RXN** | pretrain | 9.2+ |
| **TBPTT** species | `LOSS_ISOLATE=PASSIVE` / DATA_BIO | 9.3+ |
| **ADR** (transport) | passive preset | 9.4+ log; 9.6+ backward |
| **Mu path** | delta head, MU_LOG | 9.7+ |
| **Clot-band mask** | `BIOCHEM_DATA_BIO_MASK_MODE=clot_band` | 9.5, 9.9 |
| Spatial proof | **clot-phi** on dump | every dump |

---

## Fast iteration queue (recommended order)

While **K0** (kinematics retrain) runs or queues:

1. **9.0** — steady kin viz (no train).
2. **9.1** — GNODE smoke (1 forward / 0–1 ep).
3. **9.3** — 3ep DATA_BIO probe (skip 9.2 if `biochem_post_pretrain.pth` exists).
4. **9.4** or shortened **9.5** (8ep teacher + fast dump + clot-phi 30ep).
5. Only if 9.5 passes trend: **9.6 → 9.8** mu/ADR ladder.
6. **9.9** when you need joint teacher quality for Phase II.

**Parallel (no GNODE):** rung **7** (graph MP on clot-phi) if t=200 thickness still weak — does not block 9.x.

**After K0 + 6a ckpt:** **6b** (clot-phi + DEQ loop) before **rung 10** predicted-vel teacher.

---

## Gates vs clot-phi ladder

| GNODE rung | Clot-phi check |
|------------|----------------|
| 9.5+ | `go_clot_phi_from_anchor_dir` or clotband_focus; **min F1 >= 0.26** (rung 5 bar) |
| 9.9 | beat rung **4/6a** on p007 F1 optional; must **see patches** on p007 @ t=200 |
| Never | promote teacher from **K10E bridge-only** for species dump (progress SS136) |

---

## Anti-patterns (this track)

1. Jumping to **12ep K10*** or **thrombus_corona** before 9.5 clot-phi passes.
2. Using **biochem teacher viz** alone as clot proof.
3. **Full species dump** (stride 36, 6 anchors) on every 3ep probe.
4. **6b** before K0 — coupling broken flow to a good clot head wastes time.
5. Treating **mu_log_mae ~0.8** as “clots solved.”

---

## Related docs

- [BIOCHEM_TRAINING_PLAN.md](BIOCHEM_TRAINING_PLAN.md) — milestones M1, M5, isolation X/Y/XY
- [BIOCHEM_TRAINING_PROGRESS.md](BIOCHEM_TRAINING_PROGRESS.md) — passive table SS113–126
- [CLOT_PHI_ROLLOUT.md](CLOT_PHI_ROLLOUT.md) — rung 6a/6b (pre-GNODE coupling)
- [CLOT_PHI_BASELINE.md](CLOT_PHI_BASELINE.md) — mask + eval
