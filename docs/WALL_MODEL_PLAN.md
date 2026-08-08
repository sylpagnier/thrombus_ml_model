# Wall Model Plan — deploy clot score >0.5 (stretch >0.6) on unseen vessels

Active working plan as of **2026-08-05 (rev 4)**. Supersedes the "next step" sections of
[`GENERALIZATION_PLAN.md`](GENERALIZATION_PLAN.md), which remains the historical record
and the source for EDA / eval-protocol background.

**Rev 2 retracted rev 1's §2.2 multi-hop-feature diagnosis** — the 6 GPU-h sweep tested it
and it was wrong (§3a). **Rev 3** executed the pocket gate (§2.7), tested and falsified
the "different trigger" explanation for why it fails on some vessels (§2.9 — same
physics, but the trained checkpoint ties a true and false pocket at identical flow
depth), and proposed commitment-timing as an untested second signal to break that tie
(§2.10, Plan Step 1b).

**Rev 4 retracts two rev-3 readings and opens a parallel sub-cohort track (§9):**
Step 1b is **DONE and negative** — commit-order does not separate the tie on either
vessel tested, and the §2.9 "tied at 0.048 vs 0.047" claim itself conflated two different
statistics (§4 Step 1b). More consequentially, **the 0.500 → 0.887 selection ceiling is
`patient020`-specific, not a cohort constant** — on `patient037` oracle selection tops out
at 0.521 because a third of the true pocket's own mass is impure, and on `patient041` GT
coverage is 24%. See the corrections table (§3) for both retractions. Rev 4 also opens
**§9**, a narrower parallel question: on the 6-vessel stenosis/aneurysm cohort
(`039`–`044`), the zero-shot floor (no cohort training at all) already clears both the
0.50 target and the 0.60 stretch on the held-out vessel, and the diagnosed gap there is
recall, not selection — the opposite of what motivated most of §2–§4.

---

## 0. Scope — read this before proposing anything

1. **Wall model only.** We are tuning the `WC_v7_clot_phi_mse` wall component of the
   two-model compound (`C0_compound_front_offwall_h0p5`). The off-wall growth
   specialist is **shelved**. Do not propose lumen/off-wall work.
2. **Wall clots only.** Success is measured on clot inside the 3-hop wall band.
   On the primary holdout `patient020` the graded GT is `hop0=110, hop2=13`, and
   `deploy_clot_offwall_n_gt = 0` — there is genuinely no off-wall clot to find.
3. **Small cohort on purpose.** We iterate on 3–12 vessels for turnaround, not because
   more data is wrong. Expanding is legitimate whenever it demonstrably helps.
4. **Target: `deploy_clot_f1` > 0.50 on `patient020`, stretch > 0.60**, at a defensible
   `deploy_clot_mass_ratio`. Hardware: 4 GB GPU, one deploy-faithful rollout is
   ~25–30 min/anchor. Budget experiments accordingly.

### What "the score" is

The graded clot label is a viscosity-rise threshold
([`gt_clot_phi_at_time`](../src/core_physics/t0_mu_physics.py)):

```
clot(t) = relu(mu_eff(t) - mu_eff(0)) >= CLOT_PHI_THRESH_SI      # default 0.055 Pa.s
```

The *predicted* side thresholds a soft phi at 0.5, which reduces to
`mu_c * (1 + beta*(mu_ratio_max-1)*sigma) > sqrt(mu_solid * mu_ref)` where
`sigma = sigmoid((Mat_si - mat_crit)/temp)`. **Beta enters only as the product
`beta * sigma`** — it is a reparameterisation of the Mat decision boundary, not an
independent degree of freedom. This is why §2's old beta hypothesis failed.

---

## 1. Current state

| Checkpoint | `deploy_clot_f1` | mass | recall | precision | note |
|---|---|---|---|---|---|
| `wall_gen_clotrich_nplus/WG_clotrich_nplus/best.pth` (ep1, 12 vessels) | **0.500** | 2.418 | **85.5%** | 35.3% | best F1 ever recorded |
| `wall_gen_prec_iter/WG_prec_iter/best.pth` (ep9, 3 vessels) | 0.371 | 1.109 | 39.1% | 35.3% | the "floor" recent legs warm-start from |

Verified on `patient020` (`gt=110, pred=266, TP=94, FP=172, FN=16`).

```
mat_seed_prec = 1.000   mat_seed_count = 1.0    <- first commitment is ONE node, and it is correct
mat_front_speed_ratio = 1.330
deploy_clot_mass_ratio = 2.418
```

---

## 2. Diagnosis (rev 2, 2026-08-05 — measured post-sweep)

**The model grows clot correctly but commits to far too many pockets. Its predicted
connected components are nearly pure; it simply produces 39 of them where ground truth
has 2. Keeping only the correct pockets scores F1 0.887 against the current 0.500.
This is a SELECTION failure, not a growth, perception, or calibration failure.**

### 2.1 Connected-component structure on `patient020`

```
GT clot    n=110   2 components   sizes [82, 28]
PRED all   n=266  39 components   sizes [53, 33, 25, 16, 15, 15, ...]
TP         n= 94   6 components
FP         n=172  35 components   largest 33
```

The predicted components are **pure**, not smeared:

| comp | size | GT overlap | purity |
|---|---|---|---|
| 0 | 53 | 53 | **1.00** |
| 1 | 33 | 0 | 0.00 |
| 2 | 25 | 0 | 0.00 |
| 3 | 16 | 16 | **1.00** |
| 4 | 15 | 9 | 0.60 |
| 7 | 13 | 13 | **1.00** |

Where the model grows, it grows right. It also grows 33 pockets that never clot.

### 2.2 The measured ceiling

Keeping only components that touch GT:

```
kept 6/39 components   TP=94  FP=8  FN=16
precision 0.9216   recall 0.8545   F1 0.8868      (current 0.500)
```

**Selection alone is worth +0.387 F1** — more than every architecture and loss change
attempted across ~60 arms combined.

### 2.3 What ranks the pockets

Component-level features, true (6) vs false (33) pockets on `patient020`
(AUC; distance from 0.5 is the signal):

| feature | TRUE mean | FALSE mean | AUC | \|sep\| |
|---|---|---|---|---|
| **`h2min`** (min hop-2 speed in component) | 0.1029 | 0.2646 | **0.020** | **0.960** |
| `h2` (mean hop-2 speed) | 0.1632 | 0.3044 | 0.071 | 0.859 |
| `size` | 17.0 | 5.0 | 0.773 | 0.545 |
| `mat` (model's own output) | 0.00200 | 0.00200 | 0.417 | 0.167 |
| `matmax` | 0.00200 | 0.00200 | **0.500** | **0.000** |

**The model's own `Mat` field carries zero ranking information** (AUC 0.500 — both true
and false pockets are fully saturated). The discriminating signal is external flow,
and it only appears when aggregated *over a component*. That is precisely why the
node-level multi-hop feature failed (§3a): a per-node channel cannot express
"minimum flow anywhere in my connected component", and components form dynamically
during rollout.

### 2.4 A one-parameter rule nearly reaches the ceiling

Keep predicted component iff `min(hop-2 speed) < thr`, on `patient020`:

| thr | ncomp | TP | FP | FN | prec | rec | F1 |
|---|---|---|---|---|---|---|---|
| 0.10 | 3 | 70 | 0 | 40 | 1.000 | 0.636 | 0.778 |
| **0.12** | 6 | 92 | 8 | 18 | **0.920** | **0.836** | **0.876** |
| 0.15 | 11 | 94 | 92 | 16 | 0.505 | 0.855 | 0.635 |
| 1.00 (= today) | 39 | 94 | 172 | 16 | 0.353 | 0.855 | 0.500 |

**Caveat, stated plainly: this threshold was fitted on `patient020`, the holdout.** It is
a 1-parameter fit on the test set and the optimum is sharp (0.876 at 0.12 → 0.635 at
0.15). The number is a ceiling estimate, not a result. A deployable rule must set the
threshold from each vessel's own flow distribution (a percentile), never a global
constant — the burden spread is 28.8×.

### 2.5 Cross-vessel validation of the mechanism (free, no model)

`scripts/probe_pocket_ranking.py` builds candidate stagnation pockets from low-flow wall
nodes (a superset of what the model would predict) and asks whether the ones that clot
have lower flow than the ones that do not. Across **28 vessels**:

| component feature | mean AUC | \|separation\| |
|---|---|---|
| **`h2min`** | 0.213 | **0.723** |
| `h2mean` | 0.217 | 0.683 |
| `size` | 0.610 | 0.523 |
| `srmin` (graph shear rate) | 0.304 | 0.452 |

The mechanism generalises. Two notes:

- **Speed beats shear rate** (0.723 vs 0.452) even though
  [`COMSOL_PHYSICS_VALIDATION.md`](COMSOL_PHYSICS_VALIDATION.md) identifies the
  low-shear stagnation gate (`spf.sr < lss`, on for 79.7% of growing nodes) as the true
  driver. That document also warns graph operators **under-resolve** the shear gradient,
  which is the likely explanation: hop-2 speed is a better-conditioned proxy for the same
  physics than the graph's own shear estimate.
- **Not universal.** `patient024` (AUC 1.000), `036` (0.750), `001` (0.667) and `011`
  (0.625) invert — the same stagnation-vs-inverted regime split the Stage-A probe found
  (11 stagnation / 7 inverted). Any global rule will fail on those vessels.

### 2.6 Clot initiation is unseeded (confirmed with the user, 2026-08-05)

COMSOL prescribes **no wound or injury site**. Clot location emerges purely from geometry
and flow. So the information needed to locate clot *is* present in the model's inputs —
this is not a missing-input problem, which makes selection the right lever.

## 3. Corrections — claims that measurement has KILLED

Do not re-derive or re-propose these.

| Claim | Status |
|---|---|
| "The readout gain `gelation_beta` is stale and explains the 2.42× mass" | **Dead.** Swept β over the full valid range 0.2–1.3: `deploy_clot_f1` **bit-identical at every value** (0.5000, mass 2.418, FP=172, FN=16). Both TP and FP sit at `sigma = 1.0` (p05 = p95 = 1.0), AUC(σ: TP vs FP) = **0.5000 exactly**. There are no marginal nodes for a gain to move. |
| "β is a post-hoc rescale, so each value needs its own rollout" | **Dead** as motivation. β entered the graded path nowhere at all: `rollout_t0_clot_phi` accepted `gelation_beta` and immediately `del`'d it. Every historical number was graded at effective β = 1.0. |
| "`mass_ratio` 2.42 is readout amplification of a 1.33× Mat over-extent" | **Dead.** `mass_ratio` is `n_pred/n_gt` (266/110), a raw node count. It is wrong-pocket volume, not gain. |
| "Closed-loop occlusion feedback will make growth self-arrest" | **Dead for now.** The corrector is **directionally inverted**: on real clot nodes it makes flow *faster* (+30% to +215%, non-monotonic in severity) where GT slows them **−41.7%**. Its input mask is also ~97.6% phantom (§7.1). |
| "FPs are the low-speed halo around the true clot; a speed-based FP penalty cannot separate them. Confirmed category error." (old §5, `WG_prec_physfp`) | **Dead — both clauses false.** FPs are distant (median 56 hops), not a halo, and flow *does* separate them at hop 2 (AUC 0.94). |
| "The model cannot see the flow signal; add multi-hop flow features" (rev 1 §2.2) | **Dead.** Feature was 99.4% collinear with the existing speed channel; delta −0.0013. See §3a. |
| "The distant FPs are one coherent second pocket / a second seed" | **Dead.** They are **35 separate components**, largest 33 nodes. It is spray across many pockets, not one wrong blob. |
| "The model's own confidence can rank its pockets" | **Dead.** Component-mean `Mat` AUC 0.417, `matmax` AUC **0.500**. Both true and false pockets are fully saturated. Ranking must come from external flow. |
| "Per-vessel burden is predictable from deployable conditioning" | **Dead.** On clot-rich vessels (n=35), ridge LOO-CV R² = **−0.062**, permutation p = 0.44 — worse than predicting the mean. Off-wall share: LOO R² = −0.162, p = 1.00. |
| "Burden spans 6.5×" | **Understated.** Across all 35 clot-rich anchors it is **28.8×** (0.50%–14.42%). |
| "~47% of `patient020`'s target is at hop 2–3" | **Wrong.** `p020` is `h0=89.4%, h1=0%, h2=10.6%` — a thin wall skin. |
| "The N+ cohort is burden-matched to the holdout" | **Wrong.** N+ 7.26% vs holdout 5.63%; off-wall 23.8% vs 10.6%. |
| "The `midside_blind` bug drops 13.6% of training clot signal" | **Overstated.** Hop-1 is 0% of `patient020`. Still a real bug (§7.3), now minor. |
| "The 0.500 → 0.887 selection ceiling (§2.2) generalizes to other vessels" | **Wrong, vessel-specific.** Oracle selection on `patient021` is 0.840 (purity 0.833); on `patient037` it is only **0.521** (purity **0.500** — a third of the true pocket's own mass is impure) because `h2min` ranks the vessel's two mass-carrying components **backwards** (size-weighted AUC 0.276, not the 0.538 the unweighted distribution-minima comparison suggested — see next row). `patient020`'s 0.887 was the best case measured, not the ceiling. |
| "`patient037`'s true/false pockets are tied at h2min 0.048 vs 0.047" (§2.9) | **Conflated two statistics.** 0.048/0.047 are the *distribution minima across all 46 components*, not the mass-carrying 40/40 pair — that pair is `TRUE h2min=0.084` vs `FALSE h2min=0.073`, **inverted** (the false pocket is 14% more stagnant), not tied. Consequence is the same (no threshold helps) but the mechanism is wrong: this vessel isn't a coin-flip tie, it's in the inverted-flow regime alongside `024/036/001/011` (§2.5). |
| "Commitment order breaks the flow-depth tie on hard vessels" (§2.10, Step 1b hypothesis) | **Dead, tested.** `AUC(commit_t)` on flow-tied pairs is 0.676 on `037` but an unscored single pair on `021`; simulating the combined rule (flow gate then commit-time tiebreak) costs `037` (−0.009) and barely moves `021` (+0.014). `mat_seed_prec=1.000` holds only where flow already works — it is a property of where the checkpoint already succeeds, not an independent second signal. See §4 Step 1b. |
| "Gate-friendliness is a (checkpoint, vessel) interaction, not predictable from vessel physics or geometry" (§2.9) | **OVERTURNED (§10.4).** §2.9 compared `021` vs `037` on *candidate-pocket* statistics and found them alike. It never tested the vessel's own **t=0 flow distribution**. Across 32 clot-rich vessels, `band_speed_q25` (25th-pct near-wall speed at t=0, fully deployable) separates stagnation-regime from inverted-regime vessels at **93.8% accuracy / 90.6% leave-one-vessel-out**, and it puts `021` (0.051, normal) and `037` (0.085, inverted) on opposite sides — exactly the pair §2.9 could not tell apart. Regime *is* predictable in advance; §2.9's pessimistic corollary ("no adaptive rule can fire on 021-like and back off on 037-like without knowing the answer") is false. |

### Method note on the burden result

The first pass gave LOO R² = +0.065 (p=0.018) and looked like a weak positive. It was
an artifact: `*_mirror_y` vessels are the same vessel as their twins, so each sat in its
own LOO training fold. With mirrors dropped and the 8 zero-burden vessels excluded, the
signal is gone. In-sample R² was 0.497 — which is exactly why the headline must be
cross-validated.

---

## 2.7 Pocket gate: real gains, but not a free lunch (2026-08-05, post-diagnosis)

`diag_pocket_gate_sweep.py` swept the percentile on 4 of the 12 N+ training vessels
(021, 032, 035, 037 -- legitimate, per §4 Step 1: fit on train, apply once to holdout).

| vessel | off F1 | best F1 | at pct | delta |
|---|---|---|---|---|
| patient021 | 0.345 | 0.796 | 5 | **+0.451** |
| patient035 | 0.480 | 0.794 | 5 | +0.314 |
| patient032 | 0.613 | 0.621 | 25 | +0.008 |
| patient037 | 0.285 | 0.285 (off) | -- | **-0.186 at pct 5** |

**Do not use the naive mean-best percentile (pct=5, mean F1 0.481).** It is driven by
two large wins swamping one large loss in a 4-vessel average — the same
average-hides-the-failure-mode mistake §3's burden-R² and mass-ratio corrections both
caught. The **minimax** percentile (worst single-vessel delta, not the mean) is
**pct ≈ 25–30**, where three vessels are flat-to-positive and one is negative at -0.113
(patient037: F1 0.285 → 0.172, a **40% relative drop for that vessel**) instead of
catastrophically negative (-0.378) at the mean-optimal pct=5.

**Correction to an earlier framing: minimax bounds the worst case, it does not eliminate
harm.** Any global percentile will make some fraction of vessels worse than doing
nothing. "Widening the training-vessel sweep" was floated as a next step on the
assumption that more vessels would either find a percentile that's safe everywhere or at
least improve the minimax estimate meaningfully. §2.9 tests this directly and the answer
is: it would refine the *estimate* of where the minimax point sits, but it would not
produce a rule for predicting which real, unseen vessels benefit — see §2.9. That
materially lowers the value of spending GPU time on it.

**Root cause on `patient037`, confirmed by `diag_pocket037_mechanism.py`
(rollout + component-level TP/FP h2min, not threshold tuning):**

```
patient021: AUC(TP h2min < FP h2min) = 0.990   <- clean separation, gate is decisive
patient037: AUC(TP h2min < FP h2min) = 0.538   <- chance level, no threshold can help
```

TP mean h2min (0.101) and FP mean h2min (0.106) on `patient037` are statistically
indistinguishable. This is **not** a threshold-tuning failure and not "marginal true
fragments sitting at moderately higher flow" (the fragments' h2min is not systematically
higher than the false positives' — both are scattered the same way). For this
(checkpoint, vessel) pair, the feature simply carries no ranking information.

**Consequence: this cannot be made self-diagnosing at deploy time.** Detecting "the gate
won't work on this vessel" requires computing the same TP/FP AUC, which requires GT —
unavailable for a genuinely unseen vessel. There is no adaptive rule that fires hard on
021-like vessels and backs off on 037-like ones without already knowing the answer. A
global, conservative minimax percentile is therefore not a stopgap — **it is the ceiling
of what a flow-only post-process gate can do.** Some fraction of vessels will get little
or nothing from this mechanism regardless of tuning.

## 2.9 Is gate-friendliness predictable in advance? (2026-08-05, tested)

The user's question, tested directly rather than assumed: does `patient037` fail because
its clot comes from a different mechanical/biochem trigger than stagnation?

**No — falsified.** The physics-only candidate probe (`probe_pocket_ranking.py`, no
model, no rollout: does GT clot correlate with low flow at all) gives **identical**
results for both vessels:

```
patient037: n_true=1  n_false=4   h2min AUC=0.0   (PERFECT separation)
patient021: n_true=1  n_false=5   h2min AUC=0.0   (PERFECT separation)
```

Stagnation explains `patient037`'s clot exactly as cleanly as `patient021`'s. Same
mechanism, same strength. The vessel does not have measurably more natural "decoy"
stagnant regions either (`n_false` 4 vs 5) — the raw physics and the raw vessel geometry
are not what's different.

**What's actually different is the trained checkpoint's own behaviour on that vessel**,
confirmed by `diag_pocket037_mechanism.py` (rollout, component-level h2min, labelled by
GT):

```
patient037: a 40-node TRUE component and a 40-node FALSE component,
            min hop-2 speed 0.048 vs 0.047 -- statistically tied.
patient021: TP components max out at 0.065; FP components don't start until 0.062
            -- clean gap, no overlap.
```

The model isn't confused about *whether* a region is stagnant on `patient037` — it's
producing two equally-sized, equally-stagnant candidate commitments and flow depth alone
cannot break the tie, because both really are that stagnant.

**Consequence for "can we predict which vessels benefit": no, not from vessel-level
features alone.** The discriminator is a property of the *(checkpoint, vessel)*
interaction — how many comparably-plausible stagnant sites this specific trained model
chooses to commit clot to — not a property of the vessel's physics or geometry, both of
which look the same for the vessel that works and the vessel that doesn't. Detecting it
requires labelling components as TP/FP, which requires GT. This is why §2.7's
"widen the training sweep" recommendation is now downgraded: more vessels would sharpen
the minimax percentile estimate, but there is no vessel-level rule to discover that
would let the gate adapt itself per vessel without already knowing the answer.

## 2.10 Candidate second signal: commitment order (untested, proposed next)

One regularity has held on **every** checkpoint examined this session — the original
`WG_clotrich_nplus` baseline, `WG_multihop`, `WG_multihop_ctrl`, and both vessels in the
§2.9 diagnosis: **`mat_seed_prec = 1.000`. The model's first commitment is always
correct.**

Flow depth cannot break the `patient037` tie because both candidate pockets are equally
stagnant. But if growth genuinely originates from one true event and a second, false
pocket nucleates *independently*, they need not start growing at the same rollout step.
**Commitment timing is a signal orthogonal to flow depth** — untested for exactly this
purpose, but motivated by a pattern that has not failed once so far.

This is a hypothesis, not a result, and it carries the same open question the flow
feature did before validation: does "first-committing node is correct" extend to
"earlier-committing *components* rank above later ones" across a whole rollout with many
concurrent pockets, or is the guarantee specific to the single very first seed? That is
exactly what a component-level commit-time AUC (same methodology as §2.3/§2.9's h2min
AUC, but timed) would answer, and it targets the specific residual failure §2.9 just
identified: **ties in flow depth**, not the general FP/TP split flow already handles.

Practical framing for §4: this is not a replacement for the flow gate (§2.4–2.7 already
show flow alone removes the bulk of clearly-non-stagnant false pockets on vessels like
`patient021`/`035`). It is a candidate *second* feature to break the residual ties flow
leaves behind on vessels like `patient037` — most naturally combined with the flow gate,
not run instead of it.

## 3a. RETRACTED — the multi-hop feature hypothesis (rev 1 §2.2)

Rev 1 claimed the model could not *see* the flow signal that distinguishes the true clot
pocket from a wrong one, because the label sits on wall nodes where `u = v = 0` by no-slip
and the flow block aggregates one hop. A 6 GPU-h sweep (`go_wall_multihop_sweep.ps1`,
2026-08-05) added hop-1/hop-2 neighbourhood speed as a trailing feature block and
retrained with a matched control.

**Result: feature-attributable delta −0.0013.**

| arm | F1 | mass | FP | FN | distant-FP |
|---|---|---|---|---|---|
| `WG_multihop` | 0.4897 | 2.527 | 183 | 15 | 97.8% |
| `WG_multihop_ctrl` | 0.4910 | 2.518 | 182 | 15 | 97.8% |

**Why it was wrong:** the two arms' losses matched to **1e-4 at every epoch** — they were
the same run. `corr(log1p(hop1), log1p(speed)) = 0.9936` and `corr(hop2, speed) = 0.9862`
over the band. The new channels were near-duplicates of a channel the model already had.
The warm-started columns did learn (0 → 20% of a typical column's magnitude), but adding a
redundant input changes nothing.

**The reasoning error, for the record:** the hop-2 AUC measurements were real, but they
were taken *at wall nodes*, where `speed ≡ 0` makes the existing channel degenerate
(AUC exactly 0.500, and `corr` literally `nan` for zero variance). "Absent at this node"
was mistaken for "absent from the model's inputs" — but a 3-layer GraphSAGE aggregates
over the band, so the signal was already reachable one hop away. **Check reachability,
not just presence at the label's node.**

**What survives:** receptive field is *not* the bottleneck — a genuine negative that
closes a direction. And the failure mode is stable: both arms independently converged to
FP ≈ 182, distant-fraction 97.8%, `seed_prec` 1.000, `seed_n` 2.

**Secondary finding:** fine-tuning at `lr = 2e-5` for 6 epochs is nearly inert — loss
moved 0.09% and both arms landed *below* the 0.500 baseline they warm-started from. Any
future arm needs an lr/epoch sweep, or it measures the perturbation rather than the change.

---

## 4. Plan (rev 4 — Step 1b resolved)

Everything below targets **pocket selection on the N+ cohort** (§2). The
`patient020` ceiling (0.500 → 0.887) is a best case, not a cohort constant (§3) — a
different, narrower question is tracked separately in **§9** (stenosis/aneurysm
sub-cohort), where the diagnosed gap is recall, not selection.

### Step 1 — DONE. Percentile-based pocket gate as a deploy post-process

Executed (§2.7–2.9). Real, substantial gains on 2 of 4 training vessels tested
(+0.451, +0.314), ~flat on a third, and a genuine loss on a fourth that no threshold
fixes (§2.9: the model ties a true and a false 40-node pocket at identical flow depth).
Minimax percentile ≈ 25–30. **Not applied to the real holdout yet** — see Step 1b.

### Step 1b — Commitment-order probe — **DONE, negative (2026-08-05)**

Tested §2.10 with `scripts/probe_commit_order.py` (§8): one instrumented rollout per
vessel (`patient037`, the flow-tied vessel; `patient021`, the control), no training.
Instrumentation is `deploy_clot_phi_trajectory` in
[`species_pushforward_continuous.py`](../src/core_physics/species_pushforward_continuous.py) —
the full graded phi timeline, of which `deploy_clot_phi_fields` is now just the `t_eval`
slice, so commit times are read from the identical field the score thresholds. Component
splitting reuses `apply_pocket_gate`'s construction.

**Result — drop this direction:**

| anchor | `AUC(commit_t)` all pairs | `AUC(commit_t)` on flow-tied pairs | combined-rule ΔF1 (flow gate + commit tiebreak vs flow gate alone) |
|---|---|---|---|
| `patient037` | 0.643 | 0.676 | **−0.009** |
| `patient021` | 0.712 | 0.000 (1 tied pair only — not a reliable estimate) | +0.014 |

Simulating the actual combined rule (flow gate at pct 25, then keep the
earlier-committing half of survivors) **costs the vessel it was designed to fix and
barely moves the one that didn't need it.** The earliest-committing components on
`patient037` are four separate false pockets (`t=22,23,40,44`); on `patient021` the
earliest commit is the one true 38-node pocket (`t=18`). `mat_seed_prec=1.000` holds
exactly where flow already works and fails exactly where it doesn't — commitment order is
a symptom of the same (checkpoint, vessel) interaction §2.9 identified, not an
independent second signal. Accept the flow-only minimax gate as the practical ceiling for
the N+ cohort; apply it once to the `patient020` holdout when Step 2 is ready to spend
that shot (unchanged from rev 3 — still not spent).

Also corrected in this run: the §2.9 "tied at 0.048 vs 0.047" claim conflated the
per-vessel distribution minima with the actual mass-carrying 40/40 pair, which is
**inverted** (TRUE h2min 0.084 vs FALSE 0.073), not tied — see the §3 corrections table.

### Step 2 — Fold selection into training, N+ cohort (flow-only signal — Step 1b found nothing to add to it)

Post-processing proves the ceiling is reachable; it does not make the model *learn* not to
seed spurious pockets, and it cannot recover the mass those pockets consumed during the
rollout. Step 1b closed negative, so the signal to fold in is flow alone (the minimax
percentile gate), not flow + commit-time. Also now known (§3): the oracle ceiling this
would chase varies by vessel (0.887 on `patient020`, 0.521 on `patient037` — purity-
and coverage-capped, not selection-capped there), so **re-measure the oracle ceiling on
whatever vessels back this training run before setting a target**, rather than reusing
0.887. Two candidate mechanisms, cheapest first:

1. **Rollout-time nucleation gate** — block nucleation in high-flow pockets during the
   rollout, so spurious pockets never grow and never consume budget. Closest to the
   physics (`COMSOL_PHYSICS_VALIDATION.md`: deposition is gated by low-shear stagnation).
2. **Pocket-level training loss** — penalise committed mass in high-flow components.
   Note `WG_prec_pocket` already attempted something adjacent and is mis-wired (§7.4);
   rewrite rather than resurrect.

Run with a **matched control arm** and an lr sweep — §3a showed `lr = 2e-5` / 6 epochs is
too inert to attribute anything.

### Step 3 — The inverted-regime vessels

`024`, `036`, `001`, `011` rank *backwards* (clot in high flow), and the Stage-A probe
found 7 such vessels against 11 stagnation-regime. A global gate will damage these. Decide
explicitly whether the deliverable is (a) a stagnation-regime model with a documented
scope limit, or (b) a regime-conditioned model. Do not silently average over both — the
cohort mean will hide it. All three honest holdouts (`020`, `043`, `044`) are
stagnation-regime, so **holdout scores will look better than the method deserves.**

### Step 4 — Re-derive the mass gate (free)

Only meaningful once selection works. `mass_ratio` is `n_pred/n_gt`; today's 2.418 is
wrong-pocket volume, so the current gate is measuring the bug, not the model.

---

## 5. Parked — do not run these

| Arm | Why parked |
|---|---|
| `WG_prec_pocket` (pocket-contrast) | Mis-wired (§7.4). Its *premise* is now partly vindicated (the model does pick a wrong pocket) but `mat_seed_prec = 1.000` still says the **first** commitment is correct — the wrong pocket appears later. Rewrite before any resurrection. |
| `WG_prec_physfp` | **Un-parked, now the closest existing arm to the right idea.** Its original parking reason was false (§3). But note the discriminating signal is component-level, not node-level (§2.3) — a per-node speed penalty is still the wrong granularity. Rewrite as a pocket-level gate (§4 Step 2). |
| `WG_prec_cloop` | Parked *harder*. Closed-loop coupling is directionally inverted (§3, §7.1); deeper exposure to it would train against the physics. |
| Multi-hop / wider-receptive-field features | **Dead (§3a).** Tested, delta −0.0013; the feature was collinear with an existing channel and the model already reaches the signal via message passing. |
| More scheduled-sampling / dynamics sweeps | 30 legs run; all 0.06–0.14. |
| Physics-GAT pivot | Capacity is not the bottleneck (teacher-forced 1-step F1 > 0.90). Backup only. |
| Beta / readout recalibration | Dead (§3). Do not revisit without new evidence. |
| Off-wall / lumen specialist work | Out of scope (§0). |

---

## 6. Eval protocol locks

1. **Deploy-faithful, no GT velocity leak.** Quote `deploy_*` only. Never
   `val_state_f1` / `val_mat_f1` / `val_growth_f1`.
2. **Primary holdout = `patient020` only.** Do not average with `034`/`027`.
3. **Never compare in-training `deploy_clot_g` against `eval_mat_growth_simple.py`**
   unless both went through `canonical_deploy_clot_metrics`.
4. **Headline `deploy_clot_f1` (strict)**, `mass_ratio` alongside.
5. **Sealed-set leak:** `WG_clotrich_nplus` trained on `021, 032, 035, 037` — the
   `family_validation` / `generalization_challenge` vessels. Honest held-outs: `020`,
   plus `043` / `044`.
6. **New — `*_mirror_y` vessels are duplicates.** Never let a vessel and its mirror
   land on opposite sides of a train/test or CV split.

---

## 7. Known bugs

1. **Corrector sign inversion (NEW, blocking any coupling work).** On clot nodes the
   local corrector *increases* speed +30–215% where GT decreases it −41.7%.
   Non-monotonic in clot severity. `src/inference/corrector_coupling.py`.
2. **Corrector clot mask is ~97.6% phantom (NEW).** `mu_eff` is built from GT flow at
   `t+1` but compared against `mu_bulk` frozen from *kinematics-predicted* flow at
   `t=0`. The kine model over-predicts speed ~7% (|u| 1.058 vs 0.991); higher shear →
   lower Carreau viscosity → `mu_bulk` sits low → **4,498 of 19,708 nodes flagged as
   clot before any clot exists**, near-constant across the whole run. The real 110-node
   signal is ~2.4% of the mask.
3. **`SPECIES_CLOSED_LOOP_COUPLING` is read as a raw env var** at
   `species_pushforward_continuous.py:3718,3736`, not from the typed runtime config that
   sets `closed_loop_coupling=True`. Nothing populates the env var in the eval path, so
   **every number in this document was measured with coupling OFF.** Training disables
   it explicitly too (`train_species_pushforward_continuous.py:673,957`). Given bugs 1–2
   that is currently a mercy, but the gate should read the typed config.
4. **`WG_prec_pocket` is mis-wired** — `allowed` ignores `active0` so 82–89% of true clot
   is penalised on real windows; `soft_mat_commit_prob` returns ≈0.5 everywhere; the loss
   ignores `pocket_contrast_early_steps` and roughly halves the primary step-loss weight.
5. **`midside_blind` typed-config bug** — `species_pushforward_continuous.py:3115` sets
   the off-value to the *string* `"0"`, so the `is not None` test at 3177 always fires
   and every typed leg trains with `train_mask & (hops != 1)`.

---

## 8. Artifacts

**Diagnostics (new, 2026-08-05)**
- Commitment-order probe (§4 Step 1b): `scripts/probe_commit_order.py`
  → `outputs/biochem/eda/commit_order/probe.json`; primitives in
  `src/evaluation/commit_time.py` (tested by `src/tests/test_commit_time_probe.py`)
- Stenosis/aneurysm sub-cohort selection-ceiling survey (§9.4): same
  `scripts/probe_commit_order.py`, run with `--allow-holdout` on `039-043`
  → `outputs/biochem/eda/commit_order/probe_039_043.json`
- Pocket ranking, cross-vessel (free): `scripts/probe_pocket_ranking.py`
  → `outputs/biochem/eda/probe_pocket_ranking.json`
- Multi-hop LOVO probe (free): `scripts/probe_multihop_flow.py`
- Sweep + summary: `scripts/go_wall_multihop_sweep.ps1`,
  `scripts/summarize_multihop_sweep.py` → `outputs/biochem/eda/multihop_sweep/SUMMARY.md`
- Beta curve + σ saturation: `scripts/diag_gelation_beta_margin.py`
- FP geography: `scripts/diag_fp_geography.py` → `outputs/biochem/eda/fp_geo/`
- Node-level dump + pocket profile: `scripts/diag_fp_pocket_profile.py`
  → `outputs/biochem/eda/fp_geo/p020_nodes.npz` (re-analyse without a rollout)
- Burden predictability: `scripts/eda_burden_predictability.py`
- One-shot sweep: `scripts/go_wall_multihop_sweep.ps1`

**Standing**
- Burden / hop structure: `scripts/eda_clot_burden.py`
- Geometry families: `scripts/eda_generalization.py`
- Canonical eval: `scripts/eval_mat_growth_simple.py`
- Best checkpoint: `outputs/biochem/eda/wall_gen_clotrich_nplus/WG_clotrich_nplus/best.pth`
- Floor checkpoint: `outputs/biochem/eda/wall_gen_prec_iter/WG_prec_iter/best.pth`
- Historical record: [`GENERALIZATION_PLAN.md`](GENERALIZATION_PLAN.md)

**§9 (stenosis/aneurysm sub-cohort)**
- Fine-tune legs, all in `src/biochem_gnn/mat_growth_simple.py`
  (`mat_growth_leg_spec`, `wall_gen_stenosis_subcohort_train_anchors`): `WG_stenosis_subcohort_ft`
  (v1, regressed — §9.8, kept as historical record), `WG_stenosis_subcohort_ft_v2`
  (§9.9, regressed differently — §9.10, kept as historical record), `WG_stenosis_subcohort_ft_v3`
  (§9.11, current)
- Launcher: `scripts/go_wg_stenosis_subcohort_ft.ps1` (defaults to v3; `-Leg` to reproduce v1/v2)
- Growth-arrest probe (§9.10/§9.11, no training): `scripts/probe_growth_arrest.py`
  → `outputs/biochem/eda/growth_arrest/probe.json`
- New generalizable primitives (default = unchanged for every other leg):
  `train_t0_coverage_frac` (`PushforwardConfig`, §9.9.1); `deploy_eval_time_fracs` +
  `select_f1_min_hard_floor` (`ScoringConfig` / `select_checkpoint_score`, §9.9.1);
  `select_front_speed_target_lambda` + `select_fp_fn_imbalance_lambda`
  (`ScoringConfig` / `select_checkpoint_score`, §9.10 — symmetric replacements for two
  confirmed-dead terms, the originals left untouched for legs that rely on them)
- Tests: `src/tests/test_mat_growth_simple_scope.py`
  (`test_wg_stenosis_subcohort_ft_flips_underpred_and_freezes_backbone`,
  `test_wg_stenosis_subcohort_ft_v2_fixes_every_v1_root_cause`,
  `test_wg_stenosis_subcohort_ft_v3_uses_prec_config_brakes_not_v3_config`,
  `test_wall_gen_stenosis_subcohort_anchors_and_helper`),
  `src/tests/test_sliding_window_deploy_selection.py`,
  `src/tests/test_checkpoint_selection_gt_free.py` (`select_f1_min_hard_floor` and the two
  §9.10 symmetric-term cases)
- Zero-shot floor eval: `outputs/biochem/eda/commit_order/eval_p043_gate25.json`
  (`deploy_clot_f1=0.6497`, no cohort training)
- v1 regressed eval: `outputs/biochem/eda/wall_gen_stenosis_subcohort/WG_stenosis_subcohort_ft/eval_holdout_cold.json`
  (`deploy_clot_f1=0.5220`)
- v2 rejected-run log (no checkpoint promoted): `outputs/biochem/eda/wall_gen_stenosis_subcohort/WG_stenosis_subcohort_ft_v2/train_log.jsonl`
  (epoch 5's decomposed sliding-window state is §9.10's headline number, `f1=0.732` at 65% horizon)

**Refactor enabling the above:** `eval_deploy_clot_f1` is now
`deploy_species_rollout_series` → `deploy_clot_phi_fields` → `grade_deploy_clot_series`,
so diagnostics re-grade the identical rollout instead of drifting from the scored path.

---

## 9. Stenosis/aneurysm sub-cohort pivot (rev 4, 2026-08-05)

A narrower, parallel question to §1–§4's N+-cohort work: **can we generalize within a
single small, homogeneous vessel family before tackling the full population?** Wall-only
model, wall-clot-only metric (§0 unchanged) — same scope, smaller cohort.

### 9.1 Why a sub-cohort, and the mental map for after it

```
Phase A (this section)        Phase B (next)                  Phase C (deferred, §0)
stenosis/aneurysm 6-vessel     wall-only, ALL clot-rich         compound wall + off-wall
cohort (039-044), wall-only    vessels (N+ and beyond),         model. Out of scope until
─────────────────────────  →   wall-only, wall-clot-only    →   Phase B is solid. Do not
prove generalization holds     same metric, full population     propose off-wall/lumen
in a small homogeneous set                                      work before then (§0.1).
```

Phase A is deliberately small enough to iterate fast and cheap (5 GPU-vessels, no
architecture change) and homogeneous enough that a positive result is unambiguous. It is
not a replacement for §1–§4: the N+ cohort (`021/032/035/037` train, `020/043/044`
sealed holdout) remains the primary track and its ceiling-is-vessel-specific finding
(§3) is a genuine complication there that §9's cohort does **not** show — see §9.4.

**What "done" looks like for Phase A:** `WG_stenosis_subcohort_ft`'s cold eval on the
holdout beats the zero-shot floor (§9.3) without the mass guardrail firing (§9.7).
**What happens after:** whichever of (a) the recall-tilted loss reweighting, or (b) the
deep-mass / coverage relationship (§9.5) as a per-vessel risk signal, survives on this
cohort becomes a candidate to test on the N+ cohort in §4 Step 2 — as an A/B, not a
blind swap, since the N+ cohort's original failure mode (over-seeding) is the opposite
of what §9 diagnoses, and the loss ratio that fixes one can hurt the other.

### 9.2 This cohort's split is NOT the codebase's sealed split — read before running

`src/biochem_gnn/mat_growth_simple.py` already has a curated, deliberately "sealed"
split for exactly this vessel batch (documented in
[`GENERALIZATION_PLAN.md`](GENERALIZATION_PLAN.md) §1b, landed 2026-08-03):

| | Sealed (`WALL_GEN_BATCH_1B_*`) | This section (`WALL_GEN_STENOSIS_SUBCOHORT`) |
|---|---|---|
| Train | `012, 040, 041, 042` | `039, 040, 041, 042, 044` |
| Held out | `043` (aneurysm) **and** `044` (stenosis) — both sealed | `043` only |
| `039` | **Excluded** — half-finished sim, T=92 | Included |

`039` is excluded from `WALL_GEN_CLOT_RICH_ANCHORS` entirely, not just from the sealed
batch's train set — the commit-order probe independently found it the thinnest signal
of any vessel probed (29 GT nodes, 3 TP / 7 FP components). Training on `044` here
spends one of the two sealed challenge points: after `WG_stenosis_subcohort_ft` trains,
**`patient043` is the only vessel left sealed for both this sub-study and the original
wall-gen plan.** This was a deliberate scope choice for this narrower question, not an
oversight — flagged here so it's a decision on record, not a silent split change. Pass
`-TrainAnchors "patient012,patient040,patient041,patient042"` to
`go_wg_stenosis_subcohort_ft.ps1` to run the sealed split instead (holding out `044`
too means evaluating it with `--anchors patient044` in a separate eval call).

### 9.3 Zero-shot floor — no cohort training at all

`WG_clotrich_nplus` (trained on the original N+ cohort, has never seen any of
`039`–`044`) plus the flow gate at pct 25 — fit entirely on the *N+* cohort, never
touched by anything in this sub-cohort — applied once to `patient043`:

| metric | value |
|---|---|
| `deploy_clot_f1` | **0.6497** |
| `deploy_clot_score` | 0.6925 |
| `deploy_clot_mass_ratio` | 0.653 (**under**-seeding) |
| precision / recall | 0.822 / 0.537 (TP=51, FP=11, FN=44, n_gt=95) |
| `mat_front_speed_ratio` | 0.862 |
| `deploy_clot_offwall_n_gt` | 0 — purely a wall-clot vessel |

**Clears both the 0.50 target and the 0.60 stretch with zero cohort-specific training.**
The auto-generated diagnostic panel (`src/evaluation/seed_growth_diagnostics.py`)
independently classified this `mode=underseed` and hinted "do not chase score with hard
fh/topk" — consistent with everything below. This run also **is** §4 Step 2's
"apply the gate once to a holdout" application for `patient043` (§6 rule 2) — one
result now serves both tracks.

### 9.4 Selection-ceiling survey across the cohort (`probe_commit_order.py`)

Oracle selection (keep every GT-touching predicted component — the §2.2 construction)
on the training-side vessels, at the same gate pct 25:

| vessel | off-gate F1 | gate@25 F1 | oracle F1 | purity | GT coverage | `h2min` AUC | mass |
|---|---|---|---|---|---|---|---|
| `039` | 0.477 | 0.519 | 0.627 | 0.553 | 0.724 | 0.714 | 2.03 |
| `040` | 0.483 | 0.704 | 0.824 | 0.949 | 0.727 | 1.000 | 2.01 |
| `041` | 0.242 | 0.255 | 0.380 | 0.931 | 0.239 | 0.756 | 0.97 |
| `042` | 0.397 | 0.513 | 0.620 | 1.000 | 0.450 | 0.871 | 1.27 |
| `043` | 0.478 | 0.667 | 0.697 | 0.930 | 0.558 | 1.000 | 1.34 |

**5 of 5 improve under the gate, none regress** — unlike the N+ cohort's minimax
tradeoff (§2.7), no vessel in this family is harmed by pct 25. Purity is 0.93–1.00 on
four of five: **this cohort's failure mode is not over-seeding.** The binding
constraint is GT coverage (0.24–0.73, mean 0.54) — perfect selection caps `041` at
0.380 regardless of gate quality. This is the opposite bottleneck from §2's N+
diagnosis, which is why §9's fine-tune (§9.7) inverts the N+ loss ratio rather than
reusing it.

### 9.5 Deep clot mass predicts low coverage (n=5, suggestive not established)

Wall nodes at hop ≥2 (`h2 + h3` in the graded hop histogram) vs. this same-vessel GT
coverage:

```
deep mass (h2+h3)   0     8     9    68    74      | 106
vessel            039   040   043   042   041      | 044 (not yet measured)
GT coverage      0.724 0.727 0.558 0.450 0.239     |   ?
```

Spearman(deep mass, GT coverage) = **−0.900** (p=0.037, n=5). Mechanism is plausible —
the wall model is graded on hop-0 only (§0 rule 2; confirmed the graded target equals
hop-0 exactly on all 7 vessels probed), so when clot grows genuinely 3-D its hop-0
predictions degrade too. `patient044`'s deep mass (106) is the largest of any vessel in
the cohort, larger than `041`'s 74 — if the relationship holds, `044` is the hardest
vessel in the family, which is a further reason it stays a documented stress case
rather than folded into training (§9.2). n=5 with one near-tie is not enough to lean on
this alone; treat it as a candidate deployable (no-GT-needed) risk signal to widen at
full-cohort scale in Phase B, not a settled result.

### 9.6 Reframing: recall, not selection, is this cohort's bottleneck

Contrast with §2's N+ diagnosis directly:

| | N+ cohort (§2) | Stenosis/aneurysm cohort (§9) |
|---|---|---|
| Failure mode | Over-seeding (39 components vs 2 GT) | Under-seeding (`mass_ratio` 0.65–2.0, several <1) |
| What the gate does | Removes the bulk of the problem | Marginal (already near purity ceiling) |
| Binding constraint | Pocket selection | Growth coverage |
| §4 Step 2 mechanisms (nucleation gate, pocket-level FP loss) | Targeted correctly | **Would not move this cohort** — they are precision mechanisms for a recall problem |

### 9.7 Wired (v1, superseded — see §9.9): `WG_stenosis_subcohort_ft` fine-tune leg

Registered in `src/biochem_gnn/mat_growth_simple.py` (`LADDER_LEG_ORDER` +
`mat_growth_leg_spec`), warm-started from `WG_CLOTRICH_NPLUS_CKPT`:

- **Loss ratio inverted, not just softened.** N+'s warm-start actually trains at
  `underpred_weight=2.0` / `fp_weight=16.0` (8× precision-favoured — the ratio that
  fixed §2's over-seeding; corrected here from an earlier "8.0" — `fp_weight` isn't set
  by the geom/flux feature stack this leg inherits, so it silently takes the recipe's
  16.0 baseline, not `PushforwardConfig`'s bare 8.0 default; confirmed by binding every
  leg and reading the resolved config directly, §9.10). This leg sets
  `underpred_weight=4.0` / `fp_weight=4.0` (1:1) — deliberately not further, to avoid
  overshooting into the opposite failure on a cohort this small.
- **Backbone frozen.** 5 train vessels, warm-started from a checkpoint that already
  clears 0.60 zero-shot — heads/gates only, so Phase B's broader (all-vessel) use of
  the resulting checkpoint isn't compromised by a small-cohort overfit.
- **Selection targets the diagnosed gap.** Primary = strict `deploy_clot_f1` (0.70) +
  soft clout score (0.30) — not `mat_f1` alone, too noisy on 5 vessels. Plus
  `select_front_speed_lambda=0.20` and `select_fn_fp_lambda=0.20`, rewarding front
  growth completeness and penalising FN-heavy underseed — directly targeting
  `mat_front_speed_ratio=0.862` and `FN=44` vs `FP=11` from §9.3.
- **Guardrail.** `select_mass_hard_min=0.5` — the leg's whole point is raising
  `mass_ratio` toward 1.0; a checkpoint that shrinks it further than today's 0.653 must
  never be promoted, however good its score looks (the exact precision-mirage pattern
  `passes_wall_gen_gate` already guards against elsewhere).

**Kept in the codebase unmodified as the exact historical record of the run in §9.8** —
do not "fix" this leg in place; §9.9 is where every fix landed.

### 9.8 v1 result: regressed (2026-08-05) — overshot into the opposite failure

```
.\scripts\go_wg_stenosis_subcohort_ft.ps1 -Epochs 15 -EarlyStop 6 -Fresh
```

| metric | zero-shot (§9.3, no training) | after v1 | direction |
|---|---|---|---|
| `deploy_clot_f1` | **0.6497** | 0.5220 | **−0.128** |
| `deploy_clot_mass_ratio` | 0.653 (under-seed) | 2.590 (over-seed) | past 1.0 |
| `mat_front_speed_ratio` | 0.862 | 2.994 | 3.5× too fast |
| FP / FN | 11 / 44 | **157** / 6 | precision collapsed |
| `mat_overpaint_frac` | 0.042 | 0.185 | 4.5× |
| diagnostic mode | `underseed` | `overspray` | flipped regimes |

FN fell 44→6 exactly as the loss-ratio flip targeted — it overshot straight through
balance into the failure mode the rest of the wall-gen ladder exists to prevent. §9.6's
diagnosis (recall, not selection, is this cohort's bottleneck) is not in question; the
*magnitude* of the fix was. Five specific, fixable causes, each addressed in §9.9:

1. **Loss-ratio step too large for a frozen-trunk FT.** 2:8 → 4:4 is a full swing to
   parity in one move, with only 8 trainable head tensors (`freeze_backbone=True`).
   `WG_prec_front` — the closest existing precedent — moved its own ratio by a single,
   smaller notch (underpred 1→3), not to full parity.
2. **One-sided guardrail.** `select_mass_hard_min=0.5` blocks further under-seeding but
   nothing blocked *over*-seeding — mass reached 2.59 (peaking higher pre-gate) and
   nothing could reject it.
3. **Selection graded without the gate.** `CLOT_POCKET_GATE_PCT` was never set during
   training (only the standalone post-training `--pocket-gate-pct 25` eval set it), so
   checkpoint selection picked the best checkpoint under *ungated* conditions —
   conditions that don't match how the checkpoint is actually graded.
4. **Training windows never start past ~66% of the timeline.** `train_t0_max_for_n_times`
   (the per-vessel formula) caps window starts at 132 of a ~200-step vessel — the last
   third of the horizon is only ever seen as a continuation of an earlier window, never
   as a fresh rollout start. This cohort's whole diagnosis is late-forming clot (§9.5);
   this formula structurally under-samples exactly that.
5. **Selection graded a single point.** Only `t_final` was graded, so a checkpoint that
   degrades mid-rollout but recovers (or just happens to look fine) by `t_final` would
   never be caught.

### 9.9 v2: `WG_stenosis_subcohort_ft_v2` — every root cause fixed

Same warm start and feature stack as v1 (this is a hyperparameter/protocol fix, not a
rebuild). Five changes, one per §9.8 cause:

| # | v1 | v2 | why |
|---|---|---|---|
| 1 | `underpred=4.0` `fp=4.0` (parity) | `underpred=3.0` `fp=6.0` (half the move) | matches `WG_prec_front`'s single-notch precedent instead of jumping to 1:1 |
| 2 | `select_mass_hard_min=0.5` only | + `select_mass_hard_max=1.5` | symmetric guard; 1.5 sits above every off-gate mass seen pre-finetune across the cohort (0.97–2.03) but below v1's 2.59 blow-up |
| 3 | gate unset during training | `env_overrides={"CLOT_POCKET_GATE_PCT": "25"}` | selection now grades under the exact conditions the final eval uses |
| 4 | legacy `train_t0_max` formula (cap ≈132/200) | `train_t0_coverage_frac=0.85` (new, §9.9.1) | windows can start almost anywhere in the timeline |
| 5 | single point (`t_final`) | `deploy_eval_time_fracs="0.65,1.0"` (new, §9.9.1) + `select_f1_min_hard_floor=0.30` | grades two sliding points; hard-rejects if the worse one collapses |

Costs **~2× v1's per-epoch wall-clock** (two full deploy-faithful rollouts graded per
epoch instead of one) — budget accordingly.

#### 9.9.1 New primitives (generalize past this one leg; default = unchanged for every other)

Both are opt-in overrides — unset reproduces the exact prior behaviour, verified by test
(`src/tests/test_sliding_window_deploy_selection.py`):

- **`train_t0_coverage_frac`** (`PushforwardConfig`, `SPECIES_PUSHFORWARD_TRAIN_T0_COVERAGE_FRAC`).
  `0.0` (default) = the legacy `train_t0_max_for_n_times` formula, byte-identical.
  `>0.0` overrides it with `round(coverage_frac * last_step)`, clamped to leave
  `TRAIN_T0_COVERAGE_MIN_RUNWAY=20` steps of runway (clears the curriculum's largest
  unroll tier, 15, with margin) — see `species_pushforward_continuous.py`.
- **`deploy_eval_time_fracs`** (`ScoringConfig`, `SPECIES_CONTINUOUS_DEPLOY_EVAL_TIME_FRACS`).
  `""` (default) = legacy single-point / dual (mid+full) behaviour, byte-identical.
  A comma list of horizon fractions (e.g. `"0.5,0.75,1.0"`) grades the deploy-faithful
  rollout at each resolved index; the training loop means the primary metrics across all
  points and tracks `deploy_clot_f1_min` (the worst point) for the new
  `select_f1_min_hard_floor` hard-reject in `select_checkpoint_score`.
  **Known cost**: each point is a fresh full closed-loop rollout (`eval_deploy_clot_f1`
  doesn't cache across `time_index` calls the way `grade_deploy_clot_series` does within
  a single rollout in `probe_commit_order.py` / `diag_pocket_gate_sweep.py`) — an N-point
  sliding window costs ~N× the single-point eval. Kept at 2 points here for that reason;
  reusing the single-rollout-many-grades pattern those diagnostics already use would
  remove this cost if sliding-window selection becomes a recurring pattern across legs.

Locked in by `src/tests/test_mat_growth_simple_scope.py::test_wg_stenosis_subcohort_ft_v2_fixes_every_v1_root_cause`
(and the unchanged v1 test, now asserting v1 stays exactly as recorded in §9.8),
`src/tests/test_sliding_window_deploy_selection.py`, and three new
`test_checkpoint_selection_gt_free.py` cases for `select_f1_min_hard_floor`
(78/78 passing across the touched files).

**Command:**

```
.\scripts\go_wg_stenosis_subcohort_ft.ps1 -Epochs 15 -EarlyStop 6 -Fresh
```

(now defaults `-Leg` to `WG_stenosis_subcohort_ft_v2`; pass
`-Leg WG_stenosis_subcohort_ft` to reproduce v1's regression exactly). Trains on
`039,040,041,042,044`, cold-evaluates on `patient043` with pocket-gate pct 25, writes
`outputs/biochem/eda/wall_gen_stenosis_subcohort/WG_stenosis_subcohort_ft_v2/`. The
launcher hard-errors if `-GatePct` is changed without also changing the leg's baked-in
`CLOT_POCKET_GATE_PCT`, to prevent training-time selection and the final eval silently
grading at different percentiles again. **Compare against the zero-shot floor
`deploy_clot_f1=0.650` (§9.3) before claiming a win.**

### 9.10 v2 result (2026-08-05): all guards fired correctly — no checkpoint produced

```
.\scripts\go_wg_stenosis_subcohort_ft.ps1 -Epochs 15 -EarlyStop 6 -Fresh
```

`select_mass_hard_max=1.5` rejected all 6 epochs (mass 2.08–4.13 at every epoch, gated).
`best_score=-1e9`, no `best.pth` produced — only `last.pth` (epoch 6's weights, not the
best epoch reached). **This is the guard working as designed, not a bug**: v1's own
promoted checkpoint (mass 2.59) scored 0.522, below both the zero-shot floor and v1
itself. Nothing valuable was lost by rejecting; the run simply never reached an
acceptable state.

**But the sliding window found something the single-point history never showed.**
Decomposing epoch 5 exactly from the logged sliding-window means (`deploy_clot_f1_min`
made this solvable — two points, two unknowns, monotone-growth self-consistency check
confirms the solution):

| t | GT clot | model pred | mass | F1 |
|---|---|---|---|---|
| 130 (65% of horizon) | 91 | 128 | 1.41 | **0.732** |
| 200 (t_final) | 95 | 261 | 2.75 | 0.499 |

GT is essentially saturated by t≈100 (89 of its final 95 nodes). Between t=130 and
t=200 the model adds **133** nodes while GT adds **4**. At t=130 the model is in a
state that beats every number recorded on this vessel — the 0.650 zero-shot floor, and
the 0.697 oracle-selection ceiling §9.4 measured for the zero-shot checkpoint's own
components — then spends the next 70 steps destroying that answer.

**This was invisible in every prior measurement in this document**, all of which grade
`t_final` only. Reading only the endpoint would have concluded "still too much recall
pressure, tighten further" — the wrong direction entirely.

**Root cause, confirmed by protocol replication (bit-identical across two independent
runs — the rollout is deterministic, not chaotic):** v1 and v2 both trained on
`v3_config`. Auditing its differentiable loss terms against `prec_config`
(`WG_prec_iter`'s own stack) found **every term that could oppose continued growth was
at exactly 0.0** in both v1 and v2: `step_mass_penalty=0`, `step_prec_fp_penalty=0`,
`final_mass_penalty=0`, `final_prec_fp_penalty=0`, plus `mature_fp_exempt=True`, which
specifically exempts already-painted nodes from the one loss term (`gate_fp_weight`)
that was nonzero. There was no gradient anywhere telling the model to stop once it had
found the true clot. Root cause was never the `underpred:fp` ratio — it was the total
absence of an arrest signal, in an architecture whose growth is otherwise unbounded.

**Also found and fixed in this pass, both real bugs:**
- **The two selection bonus terms were confirmed dead.** `select_front_speed_lambda`
  rewards `min(front_speed, 1.5)` — monotonic, so once `front_speed` exceeds 1.5 (it
  was 2.5–5.06 every epoch) the term is a **constant** `+0.30` with zero discriminating
  power, and it actively *rewards* overshoot on the way there. `select_fn_fp_lambda`
  only fires FN-heavy (`max(0, fn-fp)`) — silent (`0.000` every epoch) once the regime
  turned FP-heavy, which is exactly what happened. New, separately-named,
  backward-compatible replacements added (`select_front_speed_target_lambda` penalizes
  `|front_speed-1|` symmetrically; `select_fp_fn_imbalance_lambda` penalizes
  `|fn-fp|/(fn+fp)` symmetrically) — the *old* terms are untouched (`WG_prec_front` and
  others already rely on their exact existing formula) and default to `0.0`.
- **`fp_weight=8.0` in §9.7–9.9 was wrong.** Binding every leg and reading
  `PushforwardConfig.fp_weight` directly: N+'s warm-start actually trains at
  `underpred=2.0` / `fp=16.0` (8×, not 4×) — the geom/flux stack never overrides
  `fp_weight`, so it silently inherits the recipe's 16.0 baseline, not
  `PushforwardConfig`'s bare 8.0 dataclass default — corrected in §9.7's text.
  **Corrected again in §9.12: this was NOT "harmless", as first written here.** It was
  harmless to v1/v2's recorded *numbers* (both override `fp_weight` explicitly and
  trained at exactly what they set) but it corrupted their *design*: believing the
  baseline was 8.0 made v1's 4.0 and v2/v3's 6.0 look like mild reductions when they
  were **2.7–4× cuts**, and §9.12 shows that cut — not any knob v1/v2/v3 deliberately
  tuned — is what drove the mass blow-up in all three.
- **The sliding-window mean was itself a blind spot for the hard guards**, symmetric to
  the single-point blind spot sliding-window grading was built to fix. Mass and FP only
  grow over this rollout (confirmed above: mass 1.41→2.75 within one epoch), so
  `t_final` is always at least as bad as any earlier point — averaging
  `select_mass_hard_max`'s input against the mean let a run's true end-state risk hide
  behind an earlier, healthier point. Fixed in the training loop:
  `deploy_clot_f1`/`deploy_clot_score`/`deploy_clot_f1_min` stay sliding-window
  aggregates (mean / worst-point — what they were built for), but
  `deploy_clot_mass_ratio` and `deploy_clot_fp` — the fields the hard guards read — are
  now anchored to `t_final` exactly, matching v1's original single-point semantics. The
  sliding-window mean is still logged (`deploy_clot_mass_ratio_sliding_mean`,
  `deploy_clot_fp_sliding_mean`) for visibility, just no longer fed to any guard.

### 9.10a Growth-arrest probe result: the defect is *onset phase*, not arrest

`scripts/probe_growth_arrest.py`, zero-shot warm-start (**no training**), gate pct 25,
10 points across each vessel's horizon:

| vessel | deep mass (§9.5) | GT onset | model onset | phase error | final prec | final rec | final mass |
|---|---|---|---|---|---|---|---|
| `patient039` | 0 | t=55 | t=18 | **−37 EARLY** | 0.404 | 0.724 | 1.793 |
| `patient040` | 8 | t=60 | t=20 | **−40 EARLY** | 0.683 | 0.727 | 1.065 |
| `patient043` *(holdout)* | 9 | t=60 | t=20 | **−40 EARLY** | 0.828 | 0.558 | 0.674 |
| `patient042` | 68 | t=20 | t=80 | **+60 LATE** | 0.598 | 0.450 | 0.752 |
| `patient041` | 74 | t=20 | t=60 | **+40 LATE** | 0.455 | 0.177 | 0.389 |

**The model's clot onset is anti-correlated with the truth — perfectly monotone in deep
clot mass, n=5.** The vessels that clot early *and* thick (`041`/`042`) are exactly the
ones it starts latest on; the thin, late-clotting vessels are the ones it starts
earliest on. This is a *phase* error, not a magnitude one.

Three consequences that redirect the work:

1. **§9.5's deep-mass ↔ coverage correlation (Spearman −0.90) now has a mechanism, and
   it was never "coverage".** Deep-mass vessels clot early and aggressively; the model
   starts 40–60 steps late on them and never catches up (`041` ends at recall 0.177).
2. **§9.10's "doesn't arrest" framing was too narrow.** 3 of 5 vessels have
   `arrest_ratio > 2`, but only `patient040` loses F1 to it (0.755 → 0.704). On the
   holdout the zero-shot F1 does **not** degrade across the horizon at all (flat 0.667
   through `t_final`) — so **v2's collapse was self-inflicted by the fine-tune**, not a
   latent property of the warm-start. That is a direct confirmation of §9.10's
   root-cause diagnosis.
3. **The holdout does not need braking — it needs more growth.** `patient043`'s location
   is already right (precision 0.96 at t=80, 0.83 at `t_final`; the 11 nodes it fires
   early at t=40 are all TP by t=80, FP drops to **1**). Its entire `t_final` deficit is
   FN=42 / recall 0.558 / mass 0.674.

### 9.11 v3: `WG_stenosis_subcohort_ft_v3` — v2 + the brake, single-mechanism A/B

Same warm start as v1/v2. **Exactly v2's config plus one mechanism**, so the result is
attributable — verified by diffing the two resolved specs programmatically, not by
inspection:

| knob | v2 | v3 | mechanism |
|---|---|---|---|
| `step_mass_penalty` / `step_prec_fp_penalty` | 0 / 0 | **0.75 / 0.5** | `rolled_final_mass_fp_penalty` at **every unroll step** during TBPTT (code comment: *"binds spray during TBPTT"*) — `softplus(mass_ratio − final_mass_target)` |
| `final_mass_penalty` / `final_mass_target` / `final_prec_fp_penalty` | 0 / 1.2 / 0 | **1.5 / 1.2 / 1.0** | the same signal once more on the rolled-out final state |
| `mature_fp_exempt` | `True` | **`False`** | matured (already-painted) nodes stay liable for FP loss |
| `underpred_weight` / `fp_weight` | 3.0 / 6.0 | **3.0 / 6.0** (unchanged) | — |
| `freeze_backbone` | `True` | **`True`** (unchanged) | — |

Values are `WG_prec_iter`'s own — borrowed, validated machinery, not new code.

**Why the brake is the right single change, and why it is *not* an "arrest" mechanism as
§9.10 first called it:** `rolled_final_mass_fp_penalty` is **GT-relative at every step**
— while GT is still empty `n_gt` clamps to 1, so a premature commit of *N* nodes yields
`mass_ratio = N` and `softplus(N − 1.2)` fires hard. It is a **premature-firing
suppressor** (what `039`/`040`/`043` need), and because it is GT-relative it stays
**silent on `041`/`042`**, where the model is *behind* GT. Correct behaviour on both
halves of a cohort that splits early/late.

That also explains v1/v2 mechanically: raising `underpred_weight` increases growth
**uniformly**, including where GT is still zero, and 200 autoregressive steps compound
it into mass 4.0. **The brake is what makes recall pressure safe by making it
time-aware** — so v3 *keeps* v2's recall pressure rather than lowering it.

**Two corrections to this section's own first draft**, both caught by §9.10a before any
GPU was spent:
- An earlier v3 lowered `underpred_weight` to 1.5 on the theory that the brake would
  otherwise fight it. Wrong: 4 of 5 vessels including the holdout are *under*-grown, and
  the brake is silent below target, so there was nothing to protect them from — it would
  have removed the exact pressure the holdout needs.
- An earlier v3 also unfroze the backbone, justified as "the trunk governs dynamics."
  The holdout's location is already correct (precision 0.83–0.97), so its defect is
  rate/onset, which the **readout heads** govern; unfreezing would have added a second
  uncontrolled variable with no diagnosed need. **If v3 under-grows on the LATE vessels
  (`041`/`042`), that is the arm where unfreezing (v3b) earns its place** — their
  precision genuinely is poor (0.46/0.60), so location *is* wrong there.

Keeps every v2 fix unchanged: gated selection (`CLOT_POCKET_GATE_PCT`),
`train_t0_coverage_frac=0.85`, sliding-window eval (`deploy_eval_time_fracs="0.65,1.0"`)
with `select_f1_min_hard_floor=0.30`, and `select_mass_hard_min/max=0.5/1.5` — anchored
to `t_final` (§9.10's fix). The two new symmetric selection terms
(`select_front_speed_target_lambda=0.15`, `select_fp_fn_imbalance_lambda=0.15`) replace
the confirmed-dead ones; **selection does not enter the training gradient**, so this
cannot confound the brake A/B.

Locked in by
`src/tests/test_mat_growth_simple_scope.py::test_wg_stenosis_subcohort_ft_v3_is_v2_plus_the_brake_and_nothing_else`
— which asserts the config diff against v2 is a **subset of the brake keys**, so any
future edit that quietly changes something else fails the test — plus four
`select_checkpoint_score` cases for the symmetric terms and
`test_sliding_window_deploy_selection.py`. 83/83 passing across every file touched by
§9.9–9.11.

**Command:**

```
.\scripts\go_wg_stenosis_subcohort_ft.ps1 -Epochs 15 -EarlyStop 6 -Fresh
```

(defaults `-Leg` to `WG_stenosis_subcohort_ft_v3`; `-Leg WG_stenosis_subcohort_ft_v2` or
`-Leg WG_stenosis_subcohort_ft` reproduce either earlier run exactly — all three are
frozen-backbone head FTs at `lr=5e-5`.) **Judge it against the right target, which is
*not* simply "beat 0.650":**
- The holdout needs **recall up** (0.558) at **mass toward 1.0** (0.674) while holding
  precision (0.828). Watch `deploy_clot_fn`, not just F1.
- v2's collapse was self-inflicted, so **"v3 does not collapse" is table stakes, not a
  win.** The win condition is `t_final` F1 above the 0.650 zero-shot floor with the mass
  guard un-fired.
- If v3 lands flat (no better, no worse), that is evidence the brake was necessary but
  not sufficient, and the phase error in §9.10a — which no loss weight addresses — is
  the real Phase-B target.

### 9.12 v3 result + the real driver: `fp_weight` was cut in v1 and never restored

```
.\scripts\go_wg_stenosis_subcohort_ft.ps1 -Epochs 15 -EarlyStop 6 -Fresh   # -Leg ..._v3
```

**v3 is a clean negative.** All 6 epochs mass-rejected again, no checkpoint. The brake is
wired and does produce gradient (v2 and v3 weights differ — `mat_f1` and
`mat_front_speed_ratio` differ at every epoch, so this is not a zero-gradient artifact),
but its effect is negligible:

| ep | v2 f1 | v3 f1 | v2 front | v3 front | v2 mass(mean) | v3 mass(mean) |
|---|---|---|---|---|---|---|
| 1 | 0.3681 | 0.3677 | 4.545 | 4.605 | 4.082 | 4.087 |
| 2 | 0.3726 | 0.3726 | 5.060 | 5.078 | 4.109 | 4.109 |
| 3 | 0.3708 | 0.3708 | 4.395 | 4.437 | 4.104 | 4.104 |
| 5 | 0.6155 | 0.6125 | 2.497 | 2.473 | 2.077 | 2.071 |

**`front_speed` moved +1.3% on a model 400% off target.** Two side-benefits: the new
`*_sliding_mean` fields reproduce v2's logged numbers to 3 decimals (confirming the
§9.10 t_final-anchoring change did what it claimed and nothing more), and v3's
directly-logged `t_final` ep5 values (mass 2.768, FP 173) match §9.10's *algebraically
reconstructed* v2 values (2.75, 172) — independently validating that decomposition.

**Then the actual driver, found by comparing all four legs against observed mass rather
than against each other's intent:**

| leg | `underpred` | `fp_weight` | ratio | observed `t_final` mass on `patient043` |
|---|---|---|---|---|
| `WG_clotrich_nplus` (warm start, no FT) | 2.0 | **16.0** | 0.125 | **0.674** ✅ |
| `WG_prec_iter` | 1.0 | **16.0** | 0.062 | **1.109** ✅ (on `p020`) |
| v1 | 4.0 | 4.0 | 1.000 | 4.200 ❌ |
| v2 | 3.0 | 6.0 | 0.500 | ~4.02 ❌ |
| v3 (+brake) | 3.0 | 6.0 | 0.500 | 4.032 ❌ |

- **`underpred` — the knob v1, v2 and v3 all tuned — is nearly inert.** 4.0 → 3.0 is a
  33% cut and moves mass 4% (4.200 → 4.02).
- **`fp_weight` splits the table perfectly.** Every leg at 16.0 controls mass; every leg
  that blew up had it cut to 4–6. **v1 cut it, and v2/v3 inherited the cut.**

Cause of the cut is the §9.10 documentation error, which was **not** harmless as first
recorded: `fp_weight` is not set by the geom/flux stack these legs inherit (so it takes
`MAT_GROWTH_SIMPLE_RECIPE`'s **16.0**), but it was documented as `PushforwardConfig`'s
bare **8.0** dataclass default. Every version was therefore designed as a mild reduction
from 8.0 while actually being a 2.7–4× cut from 16.0. Three iterations of tuning
`underpred` were spent chasing a symptom of that.

This also re-reads `WG_prec_iter` correctly: its brake "works" (mass 1.109) because it
keeps `fp_weight=16.0` — the brake was never carrying that result alone.

### 9.13 v4: `WG_stenosis_subcohort_ft_v4` — restore `fp_weight`, change nothing else

v3 with **one value changed: `fp_weight` 6.0 → 16.0.** Verified by programmatic diff:
config differs from v3 by `{fp_weight}` alone, `runtime_kwargs` and `env_overrides`
byte-identical. The brake stays (unchanged from v3) so its ~1% effect remains readable.
**v3-vs-v4 is therefore a clean single-variable test of `fp_weight` itself**, and
v2-vs-v3 remains the clean test of the brake.

`fp_weight` is now set **explicitly** on this leg rather than inherited, so it cannot
drift again; the test asserts it equals the recipe baseline and that neither
`WG_clotrich_nplus` nor `WG_prec_iter` overrides that baseline.

Locked in by `test_wg_stenosis_subcohort_ft_v4_restores_fp_weight_only`.

**Command:**

```
.\scripts\go_wg_stenosis_subcohort_ft.ps1 -Epochs 15 -EarlyStop 6 -Fresh
```

(now defaults `-Leg` to `WG_stenosis_subcohort_ft_v4`; `-Leg ..._v3` / `..._v2` /
`WG_stenosis_subcohort_ft` reproduce the earlier runs exactly.)

**Read the first epoch and stop early if it is wrong.** Epoch 1 alone is decisive and
costs ~7 min: v1/v2/v3 all reached mass ≈ 4.0 by epoch 1 from a warm start at 0.674. If
v4's epoch-1 mass lands near 1 (not 4), `fp_weight` is confirmed as the driver and the
run is worth completing. If it still lands near 4, `fp_weight` is *not* the driver
either, and the next step is a true null control — fine-tune at the warm-start's exact
loss weights (`underpred=2.0`, `fp=16.0`) to establish whether *any* fine-tune in this
configuration preserves the warm-start's behaviour, before tuning anything further.

### 9.14 v4 result: bit-identical to v3 — `fp_weight` retracted as the driver

```
.\scripts\go_wg_stenosis_subcohort_ft.ps1 -Epochs 15 -EarlyStop 6 -Fresh   # -Leg ..._v4
```

**Epoch 1 of v4 (`fp_weight=16.0`) is bit-identical to v3's epoch 1 (`fp_weight=6.0`) to
full float precision** — `loss=61.40549033352689` in both, `mass=4.032`, `f1=0.3677`,
`front=4.605`, `fp=292`, `fn=4`, all exact. Over an entire epoch (756 windows) that is
not "a weak effect" — it is zero effect. **§9.12's "`fp_weight` is the driver" claim is
retracted.**

Ruled out as a wiring bug: `mature_fp_exempt` (read by the same `_growth_huber()`
factory, via `resolve_config()`) correctly changed behaviour between v2 and v3 (loss
74.6 → 61.4), so the config-resolution path works. A synthetic unit check on
`ActiveGrowthHuberLoss` directly (`src/training/biochem_loss_policy.py:230`) confirms
the FP term does scale with `fp_weight` when its condition fires (loss 30.9 → 33.2 →
44.7 for `fp_weight` 4/6/16 on the same synthetic batch).

**What actually happened: the FP condition structurally cannot fire in this regime.**
The term only contributes when `~active & (p_raw > fp_thr)` — a predicted delta above
`fp_threshold=2e-5` **in raw units** at a node GT says is inactive. Epoch 1 trains with
`cur_unroll=5` (`curriculum_unroll_for_epoch`: 5 steps through epoch 10, 10 through
epoch 20) and a **frozen backbone** only 8 epochs' worth of head-tensor movement away
from a warm start that was itself well-calibrated. A short single-step-supervised
window essentially never produces a large enough spurious raw delta to cross that
threshold — so `fp_weight` has been irrelevant to the actual gradient in v1 through v4
alike, and the mass differences observed between them (v1 4.200 → v2 ~4.02) must trace
to something other than the loss-weight ratio.

**This reframes the mechanism entirely.** `mass_ratio` is measured on the **200-step
closed-loop deploy rollout**; the per-step training loss supervises **5–10-step TBPTT
windows**. Errors invisible within a 5-step window — never crossing the FP threshold,
never triggering `underpred_weight`'s branch either unless GT itself is locally active
— can still compound catastrophically over 200 autoregressive steps at deploy time. No
per-step loss-weight ratio can fix a failure mode the training window is too short to
observe. The **only** loss term that evaluates a full rolled-out state is the brake
(`step_mass_penalty` / `final_mass_penalty`, via `rolled_final_mass_fp_penalty`) — and
§9.12 already found its effect at these weights is ~1% (`front_speed` 4.545 → 4.605).
Read together: **the brake is the only mechanism structurally capable of seeing this
failure, and it is currently far too weak relative to the per-step channel losses
(loss ≈ 61, brake contribution ≈ 0.1–2) to counteract it.**

**Do not run v4 to completion — kill it if still running.** Every remaining epoch will
reproduce v3's trajectory; nothing downstream of a bit-identical epoch 1 can differ in
a way attributable to `fp_weight`.

### 9.15 Phase A status: the target is already met, and why every fine-tune failed

**Banked result — `patient043`, sealed holdout, ZERO cohort-specific training:**

| metric | value |
|---|---|
| `deploy_clot_f1` | **0.6497** |
| `deploy_clot_score` | **0.6925** |
| mass / prec / rec | 0.653 / 0.822 / 0.537 |

`WG_clotrich_nplus` (never saw `039`–`044`) + the flow-percentile pocket gate at pct 25
(fit on the *N+* cohort, not this one). **Both metrics clear the >0.6 stretch target.**
Cohort-wide zero-shot with the same gate: `039` 0.52, `040` 0.70, `041` 0.25, `042` 0.51,
`043` 0.67 — mean 0.53, and the spread is the result, not noise (see below).

**Every fine-tune since has been worse:** v1 0.522; v2/v3/v4 produced no checkpoint at
all. Four rounds of GPU, zero improvement over doing nothing.

**Root cause of all four failures — a train/test regime mismatch we created:**

| vessel | role | GT wall nodes | deep (h2+h3) | regime |
|---|---|---|---|---|
| `039` | train | 29 | 0 | thin / late-GT |
| `040` | train | 77 | 8 | thin / late-GT |
| `041` | train | 113 | 74 | **THICK / early-GT** |
| `042` | train | 109 | 68 | **THICK / early-GT** |
| `044` | train | 163 | 106 | **THICK / early-GT** |
| `043` | **HOLDOUT** | 95 | 9 | thin / late-GT |

**78% of training GT nodes (97% of deep mass) come from the THICK regime; the holdout is
THIN.** Per §9.10a the two regimes have *opposite* errors — thick vessels start 40–60
steps late and under-grow 4–5×; thin vessels start 37–40 steps early. So the gradient is
overwhelmingly "grow more, sooner", the frozen head can only apply that globally, and
`patient043`'s mass goes 0.674 → 4.0 in one epoch. It did so identically at `underpred`
3.0 vs 4.0, `fp_weight` 6 vs 16, with and without the brake — because **none of those
knobs was ever the variable that mattered.** All four versions tuned the loss while the
training set pointed at the wrong regime.

**Consequences for the plan:**
1. Phase A's numeric goal is met. Further loss-weight tuning is retired as a direction —
   four controlled data points say it does not reach the failure.
2. The open questions are now *data-shaped*, not optimization-shaped: (a) is regime
   predictable from deployable t=0 features (if not, a single deployed model cannot route
   itself); (b) does regime-matched training beat 0.6497 at all.
3. §4 Step 3's long-deferred choice — scoped stagnation-regime model vs regime-conditioned
   model — is now the *central* question rather than a footnote.

### 9.16 Pathology cross-tab: aneurysm vs stenosis, and a visual confirmation of §9.10a

Splitting the zero-shot gate@25 survey (§9.4/§9.9's cohort, `WG_clotrich_nplus` +
pct-25 gate, no training) by pathology (roles per `GENERALIZATION_PLAN.md` §1b):

| vessel | pathology | `deploy_clot_f1` | prec | rec | deep mass (h2+h3) |
|---|---|---|---|---|---|
| `039` | aneurysm | 0.519 | 0.404 | 0.724 | 0 |
| `040` | aneurysm | 0.704 | 0.683 | 0.727 | 8 |
| `043` (holdout) | aneurysm | 0.667 | 0.828 | 0.558 | 9 |
| `041` | stenosis | 0.255 | 0.455 | 0.177 | 74 |
| `042` | stenosis | **0.513** | 0.598 | 0.450 | 68 |
| `044` | stenosis | **0.602** (run 2026-08-06 — see §12.7) | — | — | 106 |

**Mean F1: aneurysm 0.630 (n=3), stenosis 0.384 (n=2 scored).** We do well on aneurysms
and poorly on stenoses, on the vessels measured so far.

**Important caveat, stated plainly: pathology and deep clot mass are perfectly
confounded in this 6-vessel cohort — every aneurysm has deep mass ≤9, every stenosis has
deep mass ≥68, zero overlap.** This table cannot separate "fails because it's a
stenosis" from "fails because it's thick/early-clotting" (§9.10a's actual mechanism) —
they are the same three data points either direction. `patient042` is consistent with
the mechanistic reading, not just the categorical one: it is the *lowest-deep-mass*
stenosis (68, vs `041`'s 74 and `044`'s 106) and also the best-scoring one. Read this as
"deep mass predicts F1, and in this particular batch deep mass happens to track
pathology" rather than "stenosis is intrinsically harder" — the latter isn't something a
model can act on; the former is measurable at `t=0` without a rollout.

**Visual confirmation** (`scripts/viz_mat_growth_clot_ladder.py`, now with `--gate-pct`
support added this session): ladder plots for `040` (aneurysm, good), `041` (stenosis,
poor), `042` (stenosis, the good one) —
`outputs/biochem/viz/mat_growth/clot_ladder_{best_patient040,worst_patient041,stenosis_good_patient042}.png`.
The onset-phase error from §9.10a is directly visible, not just measured: on `040` the
prediction paints red *before* GT does (`t=22`: GT flat, pred `FP=16, FN=0`); on `041`
the prediction stays blank while GT has already committed (`t=22`: GT painted, pred
`FP=0, FN=9`) and never recovers. `042` shows the *same* failure shape as `041`, smaller:
`F1` also stays at `0.00` through `t=88`, but it reaches a real peak (`F1=0.25` at
`t=133`) before decaying to `0.15` by `t_final` as GT outruns it (`FN` → 171) — a shorter
delay, not a different mechanism, consistent with its lower deep mass (68 vs `041`'s 74).
Caveat: this script's own printed per-frame `F1`/`FP`/`FN` come from
`clot_trigger_viz_f1`/`scatter_clot_error_panel`, a different implementation than
`grade_deploy_clot_series` (the path behind every number in this table) — they broadly
agree on `040` (0.69 vs 0.704) but diverge sharply on `041` (0.08 vs 0.255, and printed
`FN=188` exceeds the vessel's actual GT count of 113). Trust the spatial picture; do not
quote this script's own printed numbers as canonical.

**Practical takeaway, logged as asked: as currently measured, stenoses are the harder
pathology for this model.** Mean F1 0.384 vs 0.630, and the one stenosis that scores
respectably (`042`) still shows the *same* late-onset failure as the worst one (`041`),
just less far along it — not a case that was actually solved. Treat "stenosis" as a
practical warning label for this cohort's current model, and §9.10a's deep-mass/onset
mechanism as the reason why, not a competing explanation. `044` (the third stenosis,
deep mass 106 — higher than either `041` or `042`) is the natural next check: if the
gradient holds it should be the worst vessel in the cohort, and it is a free, read-only
probe away from confirming that.

---

## 10. Physical EDA of the COMSOL data (2026-08-05)

`scripts/eda_clot_physics.py` → `outputs/biochem/eda/clot_physics/eda.json`. **No model, no
rollout** — GT + geometry + t=0 flow only, across all **43 non-mirror anchors** (32
clot-rich, ≥20 wall-clot nodes). Grounded in the validated law
([`COMSOL_PHYSICS_VALIDATION.md`](COMSOL_PHYSICS_VALIDATION.md)), not generic feature search.

### 10.1 What COMSOL actually computes

```
J0_Mat = Da·( [d(sr,x) < sgt]·(L/γ)·|d(sr,x)|·common     ← separation gate,  21% of growing nodes
            + [sr < lss]                     ·common )   ← LOW-SHEAR gate, 79.7%  DOMINANT
common = Sat(M)·k_rs·rp + Sat(M)·k_as·ap + (Mas/Minf)·k_aa·ap
J0_th  = β·φ_at·Mat·PT                                   ← thrombin ∝ Mat ⇒ AUTOCATALYTIC
mu1(Mat): hard step 1→80 at Mat = 2e7 plt/cm²            ← the clot label IS this step
```

~90% of Mat growth is the autocatalytic `(Mas/Minf)·k_aa·AP` term; ~7% fresh deposition.
Fibrin is provably inert (`mu2(fi) ≡ 0`). **So this is a gated autocatalytic ignition
problem with a hard threshold readout — not a steady-state deposition problem.**
Confirmed in our own data: Mat late/early growth ratio median **1.41** (accelerating), and
within-vessel onset spread (p90−p10) averages **0.346 of the horizon** — clot ignites
**progressively**, not as a switch.

### 10.2 Feature inventory — what is actually usable

**6 of 18 `x` channels carry zero spatial information on every anchor:** `node_type_0..3`,
`rheology_flag`, and — most consequentially — **`wss_prior_nd` (wall shear stress) is
identically 0**. The single quantity the dominant gate depends on is present as a channel
and empty.

**At wall nodes, 6 more candidate fields are degenerate** (AUC exactly 0.500 on every
vessel): `speed`, `sdf`, `shear_potential`, `recirc`, `vmag_frac` — because `u=v=0` by
no-slip and `sdf=0` at the wall. This is §3a's finding, now confirmed cohort-wide.

**What remains informative at the wall:** `mu_eff(t=0)`, `pressure`, `width` / `width_d1` /
`width_d2`, and neighbourhood aggregates (`speed_h1..h3`, `mu0_h2`).

**Note on `mu_eff(t=0)`:** since `mu_eff = Carreau(sr)·mu1(Mat)` and `mu1 ≡ 1` at t=0, it is
a monotone-decreasing function of shear rate — i.e. the dominant gate `sr < lss` is exactly
a threshold on t=0 viscosity, and unlike `speed` it is **non-degenerate at the wall**
(shear is maximal there). Empirically it performs on par with, not better than, the hop-2
speed proxy (mean AUC 0.608 vs 0.628 over 32 vessels — indistinguishable at sd ≈ 0.27), and
the two produce **identical regime labels** (agreement 1.00). It is a viable *local*
alternative that needs no neighbourhood aggregation, not an upgrade.

### 10.3 Q1/Q2/Q3 — mostly negative, and worth knowing

| question | answer |
|---|---|
| **WHERE** does clot form | Best t=0 predictors reach only **mean AUC ≈ 0.63** (`speed_h3` 0.637, `speed_h2` 0.628, `mu0_h2` 0.623) with **sd ≈ 0.22–0.27**, and are correctly-signed on only **62%** of vessels. |
| **WHEN** does a node ignite | **No t=0 feature predicts per-node onset.** Best mean Spearman is `pressure` at 0.194 with inconsistent sign (38% negative); everything else <0.12. Consistent with autocatalytic ignition: onset depends on integrated history and neighbour coupling, not local instantaneous state. |
| **HOW THICK** (deep mass, the §9.15 regime var) | **Not predictable.** Best t=0 aggregate is `stag_frac_band` at ρ = −0.338, below the n=32 significance line (≈0.35), out of ~30 features tested. |

The huge sd on Q1 is the real story, and §10.4 explains it.

### 10.4 Q4 — the inverted regime IS routable from t=0 (the headline)

**34% of vessels (11–12 of 32) have the flow→clot relationship *inverted*** — low flow
predicts *less* clot. This independently reproduces §2.5's list: `024`, `036`, `001`, `011`
all appear, and so does **`037`**, which finally explains §2.7/§2.9's "no gate threshold
works on `patient037`" without needing a (checkpoint, vessel) story.

**A single deployable t=0 statistic separates the regimes almost perfectly:**

| | |
|---|---|
| feature | `band_speed_q25` — 25th-percentile flow speed in the hop≤3 near-wall band at t=0 |
| separation AUC | **0.975** |
| best single threshold | `band_speed_q25 ≥ 0.060` → inverted; **93.8% (30/32)** |
| **leave-one-vessel-out** | **90.6% (29/32)** — threshold refit each fold |
| permutation p | **0.000** (max-\|dev\| over all 59 aggregates, 2000 shuffles; null 95th pct 0.325 vs observed 0.475) |
| robustness | identical result when regime is labelled by `mu0` AUC instead of `speed_h2` AUC (labels agree 1.00) |

**Physical reading, and it follows directly from §10.1:** the dominant gate is
`sr < lss` — an *absolute* threshold. `band_speed_q25` measures whether the vessel
possesses a genuine stagnation zone at all. Below ~0.06, a real slow region exists, the
low-shear gate fires there, and low-flow correctly ranks clot. Above ~0.06 even the slowest
quartile of the near-wall band is moving, the low-shear gate rarely fires, and deposition
falls to the **separation gate** (`d(sr,x) < sgt`, the 21% minority mechanism) — which keys
on the shear *gradient*, not its magnitude, and therefore has a different, often opposite,
spatial signature. Two mechanisms, one threshold telling you which is in charge.

**Two axes, only one routable.** The inversion axis (§10.4) is predictable at 90.6% LOO.
The thickness/onset axis (§9.10a, deep mass) is **not** — and they are independent
(`041`/`042`/`044` are deep-mass vessels but sit on the *normal* side of the inversion
split). Do not conflate them.

### 10.5 What this changes

1. **The pocket gate can be made self-aware.** §2.7 concluded a global percentile is "the
   ceiling of what a flow-only post-process can do" because some vessels are harmed and you
   cannot tell which in advance. You now can, from t=0, before any rollout: apply the gate
   on stagnation-regime vessels, disable (or invert) it on the ~34% that are inverted.
   This is a **deploy-time post-process change with no retraining**, and it is the highest
   expected-value next experiment in this document.
2. **§2.9's pessimistic corollary is retired** (see §3 corrections table).
3. **The `wss_prior_nd` channel should be populated or removed.** The physics says wall
   shear is the driver; the channel meant to carry it is all zeros.
4. **Feature work should stop adding degenerate channels.** Six candidate fields are
   provably information-free at wall nodes; `mu_eff(t=0)` is the one physically-principled
   field that is *not* degenerate there.
5. **Per-node onset is not learnable from t=0 state** — if timing matters (it does, §9.10a),
   it has to come from the rollout dynamics, not richer static features.

---

## 11. Rethinking the biochem model (2026-08-05)

Synthesis of §10's physics, §3's corrections, and §9.12/§9.14's training-loop findings.
Everything here is grounded in a measurement in this document, not in intuition.

### 11.1 What the model actually is today

```
in_dim = 287  =  z_kin(256) + sdf(1) + band_extras(11) + state(1) + sat(1) + time(17)
band_extras(11) = flow(5) + geom_rich(5) + flux_stag(1)
flow(5) = [log1p(speed), log1p(shear), tanh(div), x_n, y_n]   x_n/y_n ZEROED by drop_xy
trunk   = 3-layer GraphSAGE (+ parallel skip-hop convs)  -> receptive field 3 hops
heads   = spatial gate x magnitude (dual head), Mat channel only
rollout = 200 autoregressive steps; TRAINING = 5-10 step TBPTT windows
readout = Carreau x mu1(Mat) -> hard threshold -> clot
```

### 11.2 Six structural problems, each with its evidence

1. **`z_kin` is 256 of 287 input dims — 89% of the input is a frozen latent trained for a
   *different task* (flow prediction).** §10.2 found the causally-relevant physics is ~6–10
   numbers. The `latent_dropout` "latent leash" exists for exactly this concern and is
   currently **0.0**. This is the single largest representational imbalance in the stack.
2. **The exact gate quantity is absent, and cannot be recovered from the graph.**
   The dominant gate is `spf.sr < lss`; `mu_eff(t=0) = Carreau(sr)` is a monotone readout of
   it (§10.2). Measured here: `mu0` is **not** collinear with what the model already has
   (|r| = 0.69 vs hop-2 speed — §3a killed multihop at 0.9936), and `r(mu0, raw speed)` is
   **undefined** because speed ≡ 0 at wall nodes. So it is genuinely new information.
   **But** neither graph shear estimator recovers it: `G_x/G_y` sparse-gradient shear
   |r| = 0.04–0.30, neighbour-gradient fallback |r| = 0.13–0.40 — and the "proper" operator
   is *worse*. This extends §2.5 from "under-resolves the gradient" to **"graph operators
   cannot reconstruct the shear field."**
   *Open flag:* `mu_prior_nd` (x-channel 13) equals GT t=0 `mu_eff` exactly, as do
   `u_prior`/`v_prior` vs GT t=0 `u`/`v`. Feeding them is therefore a **GT t=0 flow leak**
   under the current deploy protocol — but whether Stage-A "prior" channels are considered
   deploy-legal inputs is an unresolved protocol question that should be settled explicitly
   before anyone uses them.
3. **`wss_prior_nd` is identically zero on every anchor** — the channel that should carry
   wall shear, the driver of the dominant gate, is empty (§10.2).
4. **Decreasing loss does not track deploy score.** Loss moves 0.2% across epochs while
   deploy F1 swings 0.37→0.61 (§9.12). The FP loss term provably never fires (v3 vs v4
   bit-identical, §9.14). The rolled-state brake moves the rollout ~1% (§9.12). Root cause:
   training supervises **5–10-step GT-anchored windows**; deploy is a **200-step free
   rollout**. Errors that only compound over 200 steps are invisible to the objective.
5. **No autocatalysis structure.** COMSOL's Mat growth is ~90% `(Mas/Minf)·k_aa·AP` —
   positive feedback on already-deposited material (§10.1). Our model predicts Mat deltas
   with a generic GNN. §9.10a's onset anti-correlation is precisely the failure mode of a
   system modelling an ignition process without an ignition term.
6. **Two regimes, one model.** 34% of vessels invert (§10.4) — now routable at deploy.

### 11.3 What to change, ranked by expected value

| # | change | why | cost |
|---|---|---|---|
| A | **Regime-route the gate** | §10.4; deploy-time only, no retraining | done — §10.5 |
| B | **Make the objective see the deploy horizon** — raise the `deploy_horizon` aux toward full length (`tbptt_tail` bounds gradient memory, so the forward is the only cost) | the *only* loss term evaluating a rolled-out state; without it no loss change reliably improves deploy score (§9.12/§9.14) | medium |
| C | **Shrink `z_kin` (256 → 32/64) or raise `latent_dropout`** | 89% of input is off-task latent; §11.2.1 | medium |
| D | **Add an explicit autocatalytic term** — growth ∝ (local committed Mat) × availability, mirroring `(Mas/Minf)·k_aa·AP` | gray-box: the physics doc states the deposition law needs *no* learned parameters; what needs learning is the closure from predicted flow | high |
| E | **Populate `wss_prior_nd` or delete it** | §11.2.3 | low |
| F | Add `mu0`-derived gate features | informative (§11.2.2) **but blocked** — not deploy-computable, and the `_prior` channels are a possible leak | blocked pending protocol decision |

### 11.4 The honest strategic read

The last four training rounds (v1–v4) failed for reasons that had nothing to do with the
knobs being tuned — first a train/test regime mismatch (§9.15), then loss terms that
structurally cannot fire (§9.14). **B is the prerequisite for any further training work:**
until decreasing loss tracks deploy score, every training experiment is uninterpretable,
which is exactly what v1–v4 demonstrated four times. A and E are cheap and independent.
D is the physically-correct long-term direction but should not be attempted before B.

### 10.6 Regime-routed gate: measured payoff is ZERO — and it misroutes `patient037`

`scripts/diag_regime_gate_sweep.py`, one rollout per anchor re-graded three ways
(gate OFF / global pct 25 / regime-routed at the predicted-flow threshold 0.0822):

| anchor | router says | off | global | routed | routed − global |
|---|---|---|---|---|---|
| `patient021` | normal | 0.3448 | 0.4082 | 0.4082 | +0.0000 |
| `patient037` | **normal** | 0.2849 | **0.1716** | **0.1716** | +0.0000 |
| `patient035` | normal | 0.4800 | 0.6042 | 0.6042 | +0.0000 |
| `patient032` | INVERTED | 0.6128 | 0.6208 | **0.6128** | **−0.0081** |
| `patient020` | normal | 0.5000 | 0.7373 | 0.7373 | +0.0000 |
| `patient043` | normal | 0.4775 | 0.6667 | 0.6667 | +0.0000 |

```
mean F1:  off=0.4500   global=0.5348   routed=0.5335   <- routing is slightly WORSE
worst-vessel delta vs off:  global=-0.1133   routed=-0.1133   <- s2.7's minimax concern UNCHANGED
vessels where routing beat the global gate: 0/6
```

**`patient037` — the vessel the router exists to protect — is misclassified as normal.**
Under GT flow its `band_speed_q25` is 0.0849 (above the GT threshold 0.060 → correctly
inverted). Under predicted flow it is 0.0756, *below* the recalibrated 0.0822 → routed as
normal, gate applied, F1 0.285 → 0.172 exactly as before. It is one of the ~16% LOO errors
(§10.4/probe_regime_route), and it happens to be the highest-stakes vessel in the set.

Note the predictor's error is **not a uniform rescale**: it inflates slow vessels hugely
(`043` 9.2×) while *deflating* `037` (0.89×). So it compresses the dynamic range and
shuffles vessels across the boundary — which is precisely where routing decisions live.
Rank correlation of +0.967 (§10.4) was necessary but not sufficient.

**Caveat on statistical power, and why §11 keeps the router alive for now:** this set
contained only **one** genuinely inverted vessel (`032`), and there the global gate was
mildly *helping* (+0.0081), so skipping it could only lose. The hypothesis — "routing helps
because the gate is harmful on inverted vessels" — was therefore barely tested. The properly
powered test needs the inverted cohort (`019`, `025`, `029`, `011`, `024`, `001`), which is
what the autonomous block's phase 1 runs. **If routing does not win there, the router is
dead** and §10.5's claim ("the pocket gate can be made self-aware") must be retracted.

---

## 12. Autonomous block results (2026-08-06)

`scripts/go_autonomous_6h.ps1`, 1h33m wall-clock (both phases finished well under the 6h
budget; the 10 min/vessel throughput estimate was inflated by concurrent GPU load).

### 12.1 PHASE 1 — the regime router WORKS. §10.6's negative was underpowered.

Powered test, 6 inverted + 6 normal vessels, gate OFF vs global pct 25 vs regime-routed:

| anchor | regime | off | global | routed | routed − global |
|---|---|---|---|---|---|
| `patient019` | INVERTED | 0.0511 | **0.0000** | 0.0511 | **+0.0511** |
| `patient025` | INVERTED | 0.1986 | 0.1452 | 0.1986 | +0.0535 |
| `patient029` | INVERTED | 0.2308 | **0.0000** | 0.2308 | **+0.2308** |
| `patient011` | INVERTED | 0.8392 | 0.8492 | 0.8392 | −0.0100 |
| `patient024` | INVERTED | 0.4581 | 0.1618 | 0.4581 | **+0.2964** |
| `patient001` | INVERTED | 0.8852 | 0.8530 | 0.8852 | +0.0322 |
| `patient002` | normal | 0.7671 | 0.7724 | 0.7724 | +0.0000 |
| `patient006` | normal | 0.8140 | 0.8140 | 0.8140 | +0.0000 |
| `patient010` | normal | 0.9273 | 0.9273 | 0.9273 | +0.0000 |
| `patient013` | normal | 0.6058 | 0.6150 | 0.6150 | +0.0000 |
| `patient040` | normal | 0.4828 | 0.7044 | 0.7044 | +0.0000 |
| `patient041` | normal | 0.2422 | 0.2548 | 0.2548 | +0.0000 |

```
mean F1:  off=0.5418   global=0.5081   routed=0.5626
worst-vessel delta vs off:  global=-0.2964   routed=+0.0000
routing beat global on 5/12, lost on 1/12 (patient011, -0.0100)
router flagged 6/12 inverted -- all 6 match the s10.4 EDA labels
```

**Three things this establishes:**
1. **The global gate is NET HARMFUL at cohort scale** — 0.5081 vs 0.5418 with the gate
   OFF. Every prior gate result was measured on stagnation-regime vessels, which is why
   this was never visible. §2.7's "+0.451 / +0.314" gains were real but unrepresentative.
2. **On inverted vessels the gate is catastrophic, not merely unhelpful** — it drives
   `patient019` and `patient029` to **F1 exactly 0.0000** (it deletes every predicted
   component) and `patient024` from 0.458 → 0.162.
3. **Routing solves §2.7's minimax problem outright.** Worst-vessel delta goes
   −0.2964 → **+0.0000**: routing never does worse than not gating. Mean beats both
   alternatives. The gate's benefit on normal vessels (+0.0415 mean) is fully preserved
   while its harm on inverted ones (−0.1090 mean) is fully removed.

**Caveat that stands:** the router is ~84% LOO on predicted flow, and `patient037` remains
one of its errors (§10.6) — misrouted as normal, so the gate still costs it −0.113. The
router is a large net win, not a complete solution. §10.5's claim survives; §10.6's
"if routing does not win there, the router is dead" falsification test **passed**.

### 12.2 PHASE 2 — v5 (aux 40 → 150) changes NOTHING, and the census says why

| | loss range | spread | deploy F1 | Spearman(loss, F1) |
|---|---|---|---|---|
| v3 (aux 40) | 61.359–61.471 | 0.18% | 0.3655–0.6125 | **+0.314** |
| v5 (aux 150) | 61.358–61.470 | 0.18% | 0.3655–0.6125 | **+0.314** |

Identical to three decimals on every column. Config verified to have taken effect
(`deploy_horizon_steps` 40 → 150, aux rollout length 40 → 150 steps), so this is a real
negative, not a plumbing failure.

**Why — a loss-term census settles it:**
```
main per-step TBPTT windows : 756   (5-step unroll x 5 train vessels)
deploy_horizon aux rollouts :   1   (deploy_horizon_all_packs=False)
-> the aux is ~1/757 of the averaged loss terms, at ANY horizon length
```
Lengthening a single term that carries ~0.13% of the loss average cannot move the
objective. **§11.3 change B was necessary but wrongly specified:** the horizon was the
wrong dial. What matters is the aux's *weight and coverage*, not its length —
`deploy_horizon_all_packs=True` plus a large explicit weight, or a fundamentally different
objective that grades the rolled-out state directly.

Note also `Spearman(loss, F1) = +0.314` — not merely uncorrelated but **weakly positively**
correlated, i.e. lower loss trended toward *worse* deploy F1 in both arms. The objective is
not a noisy proxy for the metric; it is close to unrelated to it.

**Retained for the record:** every epoch was still mass-rejected (mass 2.77–4.06), and
epoch 5 reproduced its anomalous 0.6125 in both arms, confirming determinism again.

**Artifact location note:** v5's run lives under `outputs/biochem/eda/autonomous_6h/`, not
under the sub-cohort `RunRoot` where v1–v4/v6 live. It is not missing; it is elsewhere.

### 12.3 v6 result (2026-08-06) — change B FAILS a third time, and this one is decisive

`.\scripts\go_wg_stenosis_subcohort_ft.ps1 -Leg WG_stenosis_subcohort_ft_v6 -Epochs 6
-EarlyStop 6 -Fresh`, 5100s wall-clock (850s/epoch, ~5× v3 as predicted).

**Launcher bug found and fixed first:** `go_wg_stenosis_subcohort_ft.ps1`'s `$KnownLegs`
allowlist stopped at v4, so it rejected `-Leg WG_stenosis_subcohort_ft_v6` outright. v5 and
v6 are now in both `$KnownLegs` and `$GatedLegs` (both bake `CLOT_POCKET_GATE_PCT=25`).
This is why v5 had to be run out of the autonomous block's own `RunRoot`.

**The mechanism verified as engaged** (from `last.json` + per-epoch `cur_unroll`), so this is
a real negative, not a plumbing failure:
```
unroll = 25          curriculum_unroll = False     cur_unroll = 25 in ALL 6 epochs
deploy_horizon = 150 deploy_horizon_aux_cap = 150  deploy_horizon_all_packs = True
n_windows = 751      tbptt_tail = 5                freeze_backbone = True
```
(Ignore the env block in the printed "Resolved Configuration Fingerprint" — it shows
`SPECIES_PUSHFORWARD_UNROLL=10`, `CURRICULUM_UNROLL=1`, `AUX_CAP=72`. Those are the *base
defaults* from `src/biochem_gnn/config.py`, not the resolved values; `get_active_runtime()`
takes precedence over env at every read site. The `config_kwargs`/`runtime_kwargs` blocks
and the `[i] phase=…` header line carry the values that actually ran.)

**Result — the objective moved, the model did not:**

| | unroll | loss range | spread | best-ep score | saturated epochs | ckpt |
|---|---|---|---|---|---|---|
| v3 | 5 | 61.359–61.471 | 0.18% | 0.4998 (ep5) | 5/6 at fp=292 | none |
| v5 | 5 | 61.358–61.470 | 0.18% | 0.4998 (ep5) | 5/6 at fp=292 | none |
| **v6** | **25** | **74.435–74.929** | **0.66%** | **0.4414 (ep3)** | **5/6 at fp=292** | **none** |

Unlike v4 (bit-identical to v3 — the FP term never fired), **v6 genuinely changed the
objective**: different loss magnitude, and 3.7× v3's epoch-to-epoch spread. It changed
nothing downstream. Same fp=292 attractor, same mass ≈4.03, same universal mass-reject,
same single stochastic excursion, no checkpoint.

**The pre-registered criterion does not resolve.** `Spearman(loss, deploy_clot_score)` went
`+0.4286` (v3/v5) → **`−0.4058`** (v6) — nominally the hoped-for sign flip. It is not
evidence:
```
exact permutation test (n=6, 720 perms):  P(rho <= -0.4058 | H0) = 0.217
leave-one-out:  drop ep1 -> +0.0513   drop ep3 -> -0.9747   (one point flips the sign)
5 of 6 score values lie within 0.002 of each other (0.26102-0.26303, 0.8% spread)
ep2 and ep4 are IDENTICAL to 9 d.p. on all 6 deploy metrics -- the same thresholded state
```
The rank statistic had no dynamic range to measure: the ranks among five numerically tied
points are noise at the 1e-3 level. **Do not quote −0.4058 as a win.**

**The statistic that does resolve it.** Deploy score is *bimodal*, not continuous: one "good"
epoch at 0.44–0.50 and the rest at ≈0.26 — a ~90% relative jump. Ask directly whether the
loss can tell them apart (z of the good epoch's loss against the bad epochs' mean±sd):

| leg | unroll | good ep | good score | bad score | **z of good-epoch loss** |
|---|---|---|---|---|---|
| v2 | 5 | 5 | 0.5011 | 0.2686 | **−0.47** |
| v3 | 5 | 5 | 0.4998 | 0.2611 | **−0.30** |
| v5 | 5 | 5 | 0.4998 | 0.2611 | **−0.30** |
| **v6** | **25** | **3** | **0.4414** | **0.2618** | **+0.22** |

In every leg `|z| < 0.5`: **the loss cannot see a near-doubling of deploy score.** Making
every one of 751 windows a 25-step rollout moved the separation from −0.30 to **+0.22** —
away from informative, not toward it.

**Verdict — §11.3 change B is closed as a clean negative, third specification:**
- v4 (weight): FP term never fired, loss bit-identical → no-op.
- v5 (length): aux is 1 optimizer step of 757 → no-op, exactly as the §12.2 census predicted.
- v6 (coverage + depth, 5× compute): objective demonstrably changed, alignment did not.

The §12.2 decision rule fires. **Do not respecify change B a fourth time.** The per-step
delta loss is not a mis-weighted proxy for the thresholded-rollout metric; it is decoupled
from it. Go to §11.3 change D.

### 12.4 What v6 exposes that is bigger than change B: a two-state attractor

Pooling every epoch of every sub-cohort leg (n=41) makes the real structure visible. The
model is not continuously mis-tuned — it occupies one of **two** states:

```
SATURATED   fp >= 292, mass ~4.0, score ~0.26, relaxed recall = 1.000   35/41 epochs
EXCURSION   fp 110-260, mass 2.08-3.67, score 0.28-0.50                  6/41 epochs
```

All six excursions, ranked — and score is **strictly monotone in fp across all six**
(Spearman = −1.000), independent of which leg produced them:

| leg | ep | fp | mass | score | deploy F1 | front |
|---|---|---|---|---|---|---|
| **v2** | 5 | **110** | 2.077 | **0.5011** | **0.6155** | 2.497 |
| v3 | 5 | 173 | 2.768 | 0.4998 | 0.6125 | 2.473 |
| v5 | 5 | 173 | 2.768 | 0.4998 | 0.6125 | 2.473 |
| v6 | 3 | 218 | 3.200 | 0.4414 | 0.5335 | 2.629 |
| v2 | 6 | 243 | 3.516 | 0.2978 | 0.4047 | 4.246 |
| v1 | 14 | 260 | 3.674 | 0.2796 | 0.4009 | 2.994 |

**Three consequences.**

1. **This is bistability, not mis-tuning** — which is the signature of a gated, thresholded,
   autocatalytic system, and independent support for change D. Nothing in the objective
   steers toward the good basin; the model falls into it about once per six epochs by
   chance, then falls back out.

2. **The brake makes excursions shallower, not deeper.** v2 (no brake) reaches fp=110;
   v3/v5 (v2 + brake, otherwise a clean single-mechanism A/B per §9.11) reach fp=173; v6
   (brake + 25-step unroll) reaches only fp=218. The §9.11 brake was added to suppress
   over-painting and the deepest anti-over-paint state on record came from the leg
   *without* it. Suggestive (one excursion per leg), not established — but it inverts the
   sign §9.11 assumed, and §9.12's "the brake moves the rollout ~1%" understated it by
   measuring the mean rather than the excursion.

3. **`select_mass_hard_max = 1.5` has silently voided five consecutive legs.** The best
   states the model has ever reached are mass 2.077 / 2.768 / 3.200 — all rejected. v2, v3,
   v4, v5 and v6 each wrote **zero** `best.pth`; only `last.pth` survives, so **v2 ep5 and
   v3 ep5, at in-training deploy F1 0.6155 and 0.6125, are unrecoverable.** Every
   "the fine-tune failed" conclusion from §9.10 onward is partly a statement about this
   guard, not only about the training.

**Caveat before treating 0.6155 as near-parity with the 0.6497 zero-shot floor:** they are
not the same measurement. Training-time deploy eval runs `train_deploy_eval_flow="auto"`
(kine + optional coupling); `scripts/eval_mat_growth_simple.py:288` pins
`flow_eval="kinematics"`. One confirmation run is needed before the comparison is quotable.

**Closed, not a gap:** `deploy_clot_score` *is* logged per epoch
(`train_species_pushforward_continuous.py:1682`, written unconditionally at :1733, including
for rejected epochs). The suspected wiring gap does not exist.

**For scale, on the freeze:** `freeze_growth_backbone` leaves 8 of 40 tensors trainable —
`spatial_head` + `magnitude_head`, 45,186 of 186,887 params (24%). The entire GraphSAGE
trunk *and* the `readout` head are frozen. Every sub-cohort leg to date has run this way.

### 12.5 Revised plan after v6

Ordered by value per GPU-minute. Items 1–3 need no training at all.

1. **Fix checkpoint retention before running anything else.** Keep top-k by
   `deploy_clot_score` regardless of the mass gate, or raise `select_mass_hard_max` above
   the observed excursion floor (~2.0). Without this, change D produces no checkpoint and is
   exactly as uninterpretable as v2–v6. Highest value on the board and nearly free.
2. **Re-grade legs by excursion depth, not by end-of-training loss.** §12.3 shows loss cannot
   rank epochs. Min `deploy_clot_fp` / min mass / count of sub-gate epochs are already
   logged, discriminate cleanly at n=6, and give change D an interpretable readout without
   requiring the loss–metric alignment that change B failed to deliver.
3. **Re-state the §12.1 routing payoff in `deploy_clot_score`** (still free, unchanged), and
   **run the `patient044` probe** (§9.16, still a free unspent confirmatory point).
4. **§11.3 change D — explicit gated-autocatalytic growth term.** The decision rule fires and
   §12.4's bistability is independent evidence for it: an ignition mechanism is exactly what
   a two-basin system lacks.
5. **Cheap A/Bs that §12.4 newly justifies, if change D needs company:** drop the §9.11 brake
   (per §12.4 item 2 it may be stabilizing the bad basin), and unfreeze the backbone — no
   sub-cohort leg has ever trained anything but the two final readout MLPs.

**Unblocked by fiat:** §11.3's architecture items (`wss_prior_nd`, the analytical Poiseuille
prior block, shrinking `z_kin`) were gated on change B succeeding. That gate can never open
now, so it is dropped. Item 2 above replaces it as the interpretability precondition.

## 12.6 The real root cause: two loss terms were numerically dead (2026-08-06)

Implementing §12.5 turned up something that reframes §9.11 through §12.3. **Three legs of
"the brake barely moves the rollout" were not measuring a weak brake. They were measuring a
term that cannot respond to the model at all.**

### 12.6.1 Dead term 1 — soft occupancy is a constant 0.5

Every rolled-state term (`step_mass_penalty`, `step_prec_fp_penalty`, `final_mass_penalty`,
`final_prec_fp_penalty`) turns predictions into a differentiable committed set with
`sigmoid(soft_k * (pred - thr))`. Shipped `soft_k = 40`. Actual Mat commit threshold:
**`thr = 1e-4`**. So:

```
pred = 0      (empty node)      -> sigma = 0.4990
pred = thr    (at threshold)    -> sigma = 0.5000
pred = 10*thr (well committed)  -> sigma = 0.5090
```

The "soft committed set" is 0.5 everywhere. Sweeping a synthetic rollout from empty to 12×
over-painted, holding everything else fixed:

| hard fp | hard mass | soft mass_ratio it computes | soft fp_frac | brake penalty |
|---|---|---|---|---|
| 0 | 0.00 | 19.9600 | 0.97500 | 29.11500 |
| 60 | 2.20 | 19.9644 | 0.97491 | 29.12151 |
| 292 | 6.84 | 19.9737 | 0.97492 | 29.13544 |
| 550 | 12.00 | 19.9840 | 0.97493 | 29.15093 |

**The penalty moves 0.12% across the entire range from nothing to 12× over-target**, and the
mass ratio it reports is 19.96 in every row instead of 0.00 → 12.00. This is the mechanical
cause of §9.12's "the brake moved the rollout ~1% (front_speed 4.545 → 4.605)". The brake was
never weak; it was inert.

**Fix:** `rolled_soft_k_relative=True` makes the argument `k*(pred - thr)/thr` — a scale-free
relative deviation. Dynamic range **0.12% → 97.7%**, and the soft-F1 surrogate below goes from
a flat 0.952 to a proper 0.0009 (perfect) … 0.9999 (empty) with a minimum at the right mass.
The flag defaults to `False`, so v1–v6 stay bit-reproducible.

### 12.6.2 Dead term 2 — `fp_weight` compares against a threshold 200× too high

`ActiveGrowthHuberLoss` selects false positives as `(~gt_active) & (pred_delta > fp_thr)`,
with `fp_thr = max(fp_threshold=2e-5, delta_thresh=5e-6) = 2e-5`. Measured predicted deltas
run **~1e-7** (`delta_out_scale=1e-5` × a softplus of order 1e-2). The FP branch therefore
selects no nodes and `fp_weight` multiplies an empty set. That is precisely the signature
§9.14 recorded — v3 vs v4 bit-identical under `fp_weight` 6.0 → 16.0 — and it means **§9.12's
whole "fp_weight is the driver" table was reading a variable that never entered the loss**,
which §9.14 already retracted empirically without knowing why.

Not changed in the v7–v9 ladder (one variable at a time). The soft-F_β term applies FP
pressure on the rolled state instead, which is the better place for it.

### 12.6.3 What this means for the change-B verdict

§12.3's verdict stands and gets *stronger*, not weaker: v6 changed the per-step regression
depth while the three terms that grade the rolled state were dead, so "make the objective
horizon-aware" was never actually on trial. But it also means the objective has never been
tested in a working state — **v3, v4, v5 and v6 were all A/B tests against a dead control.**

### 12.6.4 What was built

1. **Retention (§12.5 item 1) — done.** `best_salvage.pth` tracks the best raw
   `deploy_clot_score` across all epochs regardless of the gate, and is promoted to
   `best.pth` with a loud warning if no epoch passes. Selection semantics, `best_score` and
   early stop are untouched. No leg can silently produce nothing again.
2. **Change E — the rolled-state soft-F_β surrogate.** `1 - soft_F_beta` over the rolled
   committed set: the deploy metric itself, softened. It is the only term in the objective
   with a TP numerator, hence the only one monotone in F1 by construction. The existing
   rolled terms cannot substitute — `final_mass_penalty` is `softplus(mass_ratio - target)`,
   identically zero below target so it never pushes growth *up*, and `final_prec_fp_penalty`
   is an FP fraction that saturates toward 1.0 exactly in the fp=292 basin.
3. **Change D — gated-autocatalytic growth.** `magnitude *= (k_dep + k_auto *
   local_committed_frac)`, mirroring COMSOL's `(Mas/Minf)·k_aa·AP`. Learnable and
   log-parameterised; at the `k=1` init an isolated node keeps its rate, so the warm start is
   undisturbed at `t=0` and only committed neighbourhoods accelerate. Verified 1.40×
   amplification adjacent to committed material with live gradients. `log_k_dep`/`log_k_auto`
   were added to `freeze_growth_backbone`'s allowlist — they *are* the growth law, and
   freezing them would have silently disabled the mechanism under `freeze_backbone=True`,
   which every sub-cohort leg uses.
4. **`scripts/diag_leg_alignment.py`** — grades a leg on both §12.5 criteria: alignment
   (Spearman + exact permutation p + jackknife + z-separation + how many distinct deploy
   states were actually visited) and excursion depth. Reproduces §12.3's numbers exactly and
   flags v6's sign as unstable automatically.

### 12.6.5 The v7–v9 ladder

Each step is one variable, on the v3 base, warm-started from `WG_clotrich_nplus`:

| leg | change from previous | question |
|---|---|---|
| v7 | `rolled_soft_k_relative=True` | with the brake *alive* for the first time, does v3's own mechanism work? |
| v8 | + soft-F_β surrogate (change E) | does grading the rolled state with the metric align loss and deploy score? |
| v9 | + autocatalytic growth (change D) | does an explicit ignition term break the two-basin attractor? |

### 12.6.6 v7 result (n=1 epoch): the brake is alive but *under-weighted*

Epoch 1, against v3's epoch 1 (same seed, same data, single-variable):

```
        loss       mass    fp   fn   f1      score    front
v3 ep1  61.4055    4.032   292   4   0.3677  0.2591   4.605
v7 ep1  61.6514    4.032   292   4   0.3680  0.2590   4.575
```

The loss moved **+0.246 (+0.40%)** — proof the revived term now enters the objective, since
v3-vs-v4 under a *dead* term was identical to four decimals. The rollout moved **nothing**:
identical mass, identical fp, identical fn.

**Why, quantitatively:** the brake's full range is ~17.1 raw × `loss_scale = 0.1` = **1.71
loss-units**, against a total loss of 61.65. Even at maximum it is 2.8% of the objective, and
at the observed mass 4.03 it contributes ~0.25. Fixing the occupancy was necessary but not
sufficient: **the term is now real and still too small to steer.**

This is the same failure mode as v5's aux (§12.2) — a term that cannot matter because of its
share of the objective, not because of its form. It is why v8/v9 size the soft-F_β weight
against the *measured* loss scale rather than by feel:

```
observed epoch-to-epoch loss spread (the noise floor the term must beat):  v3 0.112, v6 0.495
soft-F1 term variation across the two observed basins (fp=292 vs fp=110):
    weight   4  ->  0.14 loss-units  =  0.3x the noise floor   (would be another v5)
    weight 120  ->  4.08 loss-units  =  8.2x the noise floor   <- chosen
    weight 300  -> 10.20 loss-units  = 20.6x the noise floor
```

**Retired by this measurement:** any remaining reading of §9.11/§9.12 in which the brake is a
mechanism that "works but weakly". It is a mechanism that was dead, and is now alive at 2.8%
authority. Its configured weights were never calibrated against the objective's actual scale.

### 12.6.7 Change E's premise, verified with zero GPU

If the surrogate is to be a faithful proxy, `deploy_clot_f1` (what soft-F_β targets) must rank
epochs the same way `deploy_clot_score` (the guiding metric) does. Across all 41 recorded
epochs of v1–v6:

| leg | n | ρ(deploy_clot_f1, deploy_clot_score) | ρ(training loss, deploy_clot_score) |
|---|---|---|---|
| v1 | 15 | +0.929 | +0.206 |
| v2 | 6 | **+1.000** | +0.371 |
| v3 | 6 | +0.943 | +0.429 |
| v5 | 6 | +0.943 | +0.429 |
| v6 | 6 | **+1.000** | −0.406 |

F1 and score are rank-identical within every leg. **A loss term monotone in soft-F1 is
therefore monotone in the guiding metric by construction** — which is exactly what no term in
the current objective is. This does not prove the surrogate will train well; it proves the
premise is sound before spending GPU on it.

### 12.6.8 Budget reality — what actually ran

Measured throughput on this machine was **1186 s/epoch** for the sub-cohort legs (v6's 850 s
was under lighter load), and a standalone 1-vessel `eval_mat_growth_simple.py` took ~18 min.
The 2.5 h budget therefore bought roughly **one leg**, not three. Choices made, explicitly:

* v7 stopped after 1 epoch — its epoch-1 result (above) already answered its question, and it
  was the least promising leg once the 2.8%-authority calculation was in hand.
* **v8 was skipped** and the budget put on **v9** (the full stack: occupancy fix + soft-F_β at
  8.2× the noise floor + autocatalysis). This costs the E-vs-D attribution, which must be
  recovered later by running v8. Stated plainly so nobody reads a v9 result as evidence for
  change D specifically.
* The `patient044` probe (§9.16) was started and killed mid-rollout to free the GPU for
  training; it remains unspent. The `patient043` control it completed first returned
  **`clot_f1 = 0.650`, mass 0.653, front 0.862, mode=underseed** — an exact reproduction of
  the §9.15 headline, which confirms the measurement path used for everything above.
* Re-stating §12.1's routing payoff in score terms was **not** done: the cached
  `phase1_regime_gate_cohort.json` carries F1 only, and a 12-vessel re-sweep does not fit.

### 12.6.9 A term-placement trap to fix in v10 — the per-step surrogate is *diluted*, not added

v9 epoch 1 came in at `loss = 59.412` against v3's `61.406` — the total went **down** after
adding two positive terms. The reason is structural and worth pinning:

```
per-step terms:  step_loss = sum(loss_i * w_i) / sum(w_i)     <- a weighted MEAN
final term:      step_loss = step_loss + rolled_soft_f1_loss(...)  <- ADDED
```

A per-step term is averaged *in*, so a term whose value (2.80) is far below the Huber terms
(~61) pulls the mean down and receives only `1/(n_terms+1)` of the gradient share. Only the
final-state term carries its full nominal weight (+8.40 here). So `step_soft_f1_weight` does
not mean what `rolled_soft_f1_weight` means, and the two are not comparable numbers.

**Consequence for reading v9:** its surrogate authority is essentially the final-state term
alone. **Fix for v10:** either add the per-step surrogate to the sum rather than the mean, or
raise `step_soft_f1_weight` by roughly the term count. Do not simply scale both together.

**What epoch 1 does establish:** the objective moved by 1.994 loss-units in one step — **4.0×
v6's entire six-epoch spread** — so unlike v4 and v5 this is a real intervention on the loss.
The rollout did not move with it (`fp=293`, `mass=4.042`, still the saturated basin), which is
the expected starting point: one epoch of head-only training at `lr=5e-5` from a warm start
already sitting in that basin.

### 12.6.10 How to read v9 — pre-registered before epochs 2–4 were seen

Written while epoch 1 was the only result, so the conclusion cannot be fitted after the fact.

* **Success:** any epoch leaves the saturated basin (`fp < 292`) *and* the loss ranks it below
  the saturated epochs. That would be the first time in seven legs that the objective both
  moved and carried the rollout with it.
* **Partial:** excursions get deeper than v6's `fp=218` but the loss still cannot rank them
  (`|z| < 0.5`). Reads as "the growth law changed, the objective still cannot see it" — go to
  §12.6.9's placement fix, not to another mechanism.
* **Null:** all epochs stay at `fp ≈ 292`. Then the full stack — occupancy fix, a surrogate at
  8.2× the loss noise floor, and an explicit ignition term — failed to move a model whose only
  trainable parts are the two readout MLPs. **In that case the next variable is
  `freeze_backbone`, not the objective.** Every sub-cohort leg from v1 to v9 has trained 8–10
  tensors out of 40 (45k of 187k params), and §12.4's attractor may simply be out of reach of
  the readout heads. That would be a genuinely new arm, not a seventh reweighting.

Note the asymmetry deliberately: two of the three outcomes point *away* from further objective
work. After change B's three failures, the prior that "the loss is the problem" has to be
allowed to lose.

## 12.7 `patient044` spent (2026-08-06) — the deep-mass → failure gradient is FALSIFIED

§9.16's last unscored vessel, run zero-shot against `WG_clotrich_nplus` + pct-25 gate, no
training. `patient043` was measured on the same command in the same session as a scale control
and returned **`clot_f1 = 0.650`**, reproducing the §9.15 headline exactly.

```
patient043   deep mass   9   ->  clot_f1 0.650   mass 0.653   front_spd 0.862   mode=underseed
patient044   deep mass 106   ->  clot_f1 0.602   mass 0.571   front_spd 0.634   mode=underseed
```

**The prediction was that `044` would be the worst vessel in the cohort.** §9.16 ranked the
five scored vessels and deep mass tracked F1 monotonically (`040` deep 8 → 0.704, `043` deep 9
→ 0.667, `042` deep 68 → 0.513, `041` deep 74 → 0.255). Extrapolating that gradient, `044` at
deep mass 106 — 43% above `041` — should have come in at **≤ 0.255**.

It came in at **0.602**: 2.4× the prediction, and the third-best vessel in the cohort. Allowing
the ~0.017 offset between the two measurement paths (§9.16's table reads `043` as 0.667, this
run reads 0.650) changes nothing about that conclusion.

**What this retires.** "Deep clot mass predicts deploy failure" was §9.5's suggestive finding,
§9.10a's mechanism, and §9.16's organising story for the whole cohort. It does not survive its
own pre-registered confirmatory test. The n=5 monotonicity that looked so clean was, with the
sixth point in hand, **not a gradient at all** — `041` is simply an outlier, and the aneurysm/
stenosis split §9.16 flagged as "perfectly confounded" now separates: `044` is a *stenosis*
with the *highest* deep mass and it scores like an aneurysm.

**What this costs.** §11.3 change D was motivated partly as targeting "§9.10a/§9.16's measured
failure — the model paints early on thin vessels and late on thick ones". That specific
motivation is now unsupported. Change D may still be right on COMSOL-mechanism grounds
(`(Mas/Minf)·k_aa·AP` is the real growth law regardless), but it should no longer be sold as
addressing a measured deep-mass gradient. **This is a second pre-registered prediction
overturned by its own probe in two sessions** — §2.9 went the same way in §10.4/§12.1.

**What this gains, and it is the most useful number of the session.** Zero-shot, with no
cohort training at all, the warm start scores **0.650 on `043` and 0.602 on `044`** — both
sealed, both `mode=underseed`, both essentially at the 0.6 target. The two vessels that were
supposed to be hardest are the two closest to shipping. The gap to the goal is not "the model
cannot do stenoses"; it is `041` (0.255) and `039` (0.519), and `041`'s failure now needs its
own explanation rather than a cohort-wide one.

## 12.8 v9 result — INCONCLUSIVE, and the reason indicts every leg in this study

v9 = occupancy fix + soft-F_β surrogate at 8.2× the loss noise floor + gated-autocatalytic
growth. All three mechanisms verified engaged before reading anything (the v6 discipline):

```
freeze_backbone: frozen=32 trainable_heads=10     <- 10, not 8: log_k_dep/log_k_auto ARE training
autocatalytic_growth=True  autocat_k_dep_init=1.0  autocat_k_auto_init=1.0  autocat_alpha=0.8
rolled_soft_k_relative=True  rolled_soft_f1_k=10.0  rolled_soft_f1_weight=120.0
```

| | loss | mass | fp | fn | f1 | score |
|---|---|---|---|---|---|---|
| v3 ep1 (reference) | 61.4055 | 4.032 | 292 | 4 | 0.3677 | 0.2591 |
| v9 ep1 | 59.4122 | 4.042 | 293 | 4 | 0.3691 | 0.2601 |
| v9 ep2 | 59.4616 | 4.032 | 292 | 4 | 0.3743 | 0.2620 |
| v9 ep3 | 59.3688 | 4.042 | 292 | 3 | 0.3725 | 0.2608 |
| v9 ep4 | 59.4645 | 4.074 | 292 | **0** | 0.3827 | 0.2657 |

**The objective moved.** 61.41 → 59.41 is 1.994 loss-units — **4.0× v6's entire six-epoch
spread**. Unlike v4 (dead term, bit-identical) and v5 (1-of-757 steps), this is unambiguously
a real intervention on the loss.

**The rollout did not.** `fp` ∈ {292, 293}, mass 4.032–4.074, score range **0.0056**. Four
epochs, **zero excursions**.

**One suggestive-but-not-separable signal.** FN fell monotonically 4 → 4 → 3 → **0** while FP
stayed pinned at 292 — the surrogate did push recall to completion, which is what a term with
a TP numerator should do. But v9's F1 range (0.3691–0.3827) sits *inside* v3's own
non-excursion range (0.3655–0.3811), so at n=4 this is **not separable from ordinary
epoch-to-epoch wander**. Recorded as a lead, not a result.

**The retention fix earned itself on this run.** Every epoch was gate-rejected, exactly as in
v2–v6 — and unlike v2–v6, the leg left a recoverable checkpoint:
```
[i] salvage ckpt: ep 4 deploy_clot_score=0.2657 -> best_salvage.pth
[WARN] no epoch passed the selection gate; promoted salvage ep 4 to best.pth
[WARN]   this checkpoint is GATE-REJECTED -- meta.salvage_gate_rejected=True. Grade it, do not ship it.
```

### 12.8.1 Why this is not a verdict — the power calculation nobody ran

The tempting read is "the full stack failed too". That read is not supported:

```
base excursion rate across v1-v6:  6 excursions / 41 epochs = 0.146 per epoch
P(0 excursions in 3 epochs | v9 behaves EXACTLY like v1-v6) = 0.622
P(0 excursions in 4 epochs | ditto)                        = 0.531
P(0 excursions in 6 epochs | ditto)                        = 0.387
```

**At n=4 there is a ~53% chance of observing zero excursions even if v9 changed nothing at
all.** Zero excursions is the single most likely outcome under the null. It carries no
information.

Worse, and this is the part that generalises: **the alignment criterion is unmeasurable
without an excursion.** "Does the loss rank the good epoch below the bad ones" requires a good
epoch to exist. With 0 excursions there is nothing to rank, so v9 cannot answer either of its
two questions — not because the mechanisms failed, but because the experiment was too short to
observe the phenomenon it was designed to measure.

### 12.8.2 The design flaw this exposes, retroactively, in v1–v9

Every leg in §9 and §12 ran 6 epochs or fewer. At a 0.146/epoch excursion rate:

* 6 epochs → **0.9 expected excursions.** v3 got 1, v5 got 1, v6 got 1, v2 got 2, v4 got 0.
* A correlation between loss and deploy score needs **at least 2** excursions to be estimable
  at all, and realistically 4–5 to be distinguishable from noise.
* Reaching 2 expected excursions needs **~14 epochs**; 4 expected needs **~27**.

So §12.3's headline — "Spearman flipped sign but p=0.217 and one epoch flips it" — was never a
finding about v6's mechanism. **It was a finding about running a 6-epoch experiment on a
process with a 15%-per-epoch event rate.** v3's +0.43 and v6's −0.41 differ only in where each
leg's single excursion happened to land. Both were always going to be noise.

**This is the binding constraint on the whole study, and it is not an objective problem, an
architecture problem, or a data problem.** At the measured 650–1200 s/epoch, an interpretable
leg costs **2.5–4.5 GPU-hours**, not the ~1 hour every leg since v1 has been given. Six legs
× 6 epochs bought six ambiguous results for the price of roughly two decisive ones.

### 12.8.3 Revised plan

1. **Stop running 6-epoch legs.** Minimum viable leg is ~14 epochs. Better: change the
   readout so the phenomenon is observable per-*window* rather than per-*epoch* — the
   excursion is a property of the rolled state, and grading many t0 windows per epoch would
   raise the event count by orders of magnitude without more GPU. This is the highest-value
   methodological fix available and it is cheap.
2. **Then `freeze_backbone=False`** — the pre-registered §12.6.10 next variable, untouched by
   the power problem. Every leg v1–v9 trained 8–10 tensors of 40 (45k of 187k params, both
   readout MLPs only). §12.4's attractor may simply be out of reach of the readout heads.
3. **Then v8**, to recover the change-D vs change-E attribution v9 conflates (§12.6.8).
4. **Fix the per-step surrogate dilution** first, since it is nearly free (§12.6.9).
5. **Re-read §12.6.7 before any of it:** the surrogate's premise (F1 and score rank
   identically) is verified and independent of all of the above. Change E is still the best
   available answer to "make the loss track deploy score"; it has simply not yet been given an
   experiment capable of showing it.

### 12.8.4 Where the goal actually stands

The session's most useful number is not from training at all (§12.7): zero-shot, no cohort
training, `WG_clotrich_nplus` + pct-25 gate scores **0.650 on `patient043`** and **0.602 on
`patient044`** — the two sealed vessels, both `mode=underseed`. Against the target of >0.6 on
unseen vessels, **the untrained warm start already clears it on both sealed holdouts.**

Nine fine-tune legs have not beaten it, and §12.8.2 now explains why none of them could have
been read either way. The gap to "all vessels" is `patient041` (0.255) and `patient039`
(0.519) — and with §12.7 falsifying the deep-mass gradient, `041` needs its own diagnosis
rather than a cohort-wide story.

## 13. Cohort close-out attempt (2026-08-06, session 3)

**Budget:** 7 GPU-hours of a 9 h allowance, 2 h headroom. Spend tracked per step below.

**Goal:** `deploy_clot_score > 0.60` on all six of `039`–`044`, generalizing to unseen vessels.

**Standing constraints for this session** (each derived from a logged failure, not taste):
one variable per leg (§9.12/§9.14 cost six legs to this); ≥14 epochs per leg (§12.8.2 —
excursion base rate 0.146/epoch makes anything shorter unreadable); no concurrent GPU jobs
(this session measured epoch time going 650 s → 1900 s under contention); verify every
mechanism engaged before trusting a result (§12.3 — v4 and v5 both looked like real tests and
were silently no-ops).

### 13.1 Step 1 — cohort-wide zero-shot baseline, in score terms — PRE-REGISTERED

Written before the eval returned.

Everything in §9/§12 that shaped this plan was measured in `deploy_clot_f1`. The guiding
metric was changed to `deploy_clot_score` two sessions ago and **the cohort has never been
scored on it**. §9.15's single datapoint (`patient043`: F1 0.6497, score 0.6925) says score
reads *higher* than F1 on the one vessel where both are known, so the existing F1 tables are
probably pessimistic. That is a hypothesis about one vessel, not a correction factor.

`eval_mat_growth_simple.py --anchors patient039..044 --pocket-gate-pct 25`, warm start
`WG_clotrich_nplus`, no training of any kind.

**Decision rule, fixed in advance:**
* **All 6 ≥ 0.60 → STOP.** The goal is met zero-shot and every fine-tune leg v1–v9 was
  chasing a target that was already cleared. Write it up and end the session.
* **Otherwise** → the vessels below 0.60 are the gap, and step 2 (regime routing in score
  terms) is attempted only if it could plausibly move those specific vessels.

**Prior, recorded so it can be scored:** F1 is known for five of six (§9.16 + §12.7):
`039` 0.519, `040` 0.704, `041` 0.255, `042` 0.513, `043` 0.650, `044` 0.602. If the
F1→score offset seen on `043` (+0.043) were uniform, four vessels would clear 0.60 and
`041` and `042` would not. **Expected outcome: the gap is `041`, probably `042`, possibly
`039`.** If instead all six clear, the offset is far larger than one vessel suggested and
that itself is the finding.

### 13.2 Step 1 RESULT — the gap is `039`, `041`, `042`. Prediction confirmed.

`eval_mat_growth_simple.py`, `WG_clotrich_nplus` + pct-25 gate, no training, 6 vessels, ~35 min.

| vessel | **`deploy_clot_score`** | `deploy_clot_f1` | mass | fp | fn | ≥0.60 |
|---|---|---|---|---|---|---|
| `039` | **0.4660** | 0.5185 | 1.793 | 31 | 8 | NO |
| `040` | **0.6218** | 0.7044 | 1.065 | 26 | 21 | yes |
| `041` | **0.3319** | 0.2548 | 0.389 | 24 | 93 | NO |
| `042` | **0.5030** | 0.5131 | 0.752 | 33 | 60 | NO |
| `043` | **0.6925** | 0.6497 | 0.653 | 11 | 44 | yes |
| `044` | **0.6400** | 0.6016 | 0.571 | 16 | 86 | yes |
| mean | 0.5425 | 0.5403 | | | | 3/6 |

**§13.1's pre-registered prediction — "the gap is `041`, probably `042`, possibly `039`" —
is exactly right**, including `040` clearing. Step 1's stop condition does **not** fire;
proceed to step 2.

**The uniform-offset assumption was wrong, as flagged.** The F1→score offset is not a
constant, it is not even a constant sign:

```
039 -0.0525    040 -0.0826    041 +0.0771
042 -0.0101    043 +0.0428    044 +0.0384
```

Score reads *lower* than F1 on `039`/`040`/`042` and *higher* on `041`/`043`/`044`. So the
§12.3 note that "existing F1 tables are likely pessimistic" (extrapolated from `043` alone)
is **retracted** — the two metrics reorder vessels, they do not offset them. Concretely,
`041` is the worst vessel on both metrics but by very different margins (F1 0.255 vs score
0.332), and `040` drops from a comfortable 0.704 to a marginal 0.622.

**Structure of the gap — it is not one failure mode, it is two:**
* `039` over-paints (mass **1.793**, fp 31, fn 8) — `mode=overspray`.
* `041` and `042` under-paint (mass **0.389** / **0.752**, fn **93** / **60**) —
  `mode=underseed`.

Any single fix that moves all three has to be non-monotone in mass. **A global gate change
cannot do it** — loosening the gate helps `041`/`042` and hurts `039`, tightening does the
reverse. That is precisely the case regime *routing* exists to handle, which is what makes
step 2 worth its GPU rather than a formality.

### 13.3 Step 2 — regime routing in score terms — PRE-REGISTERED

Written before the sweep returned. `diag_regime_gate_sweep.py --anchors patient039..044
--pct 25`, three conditions per vessel (gate OFF / global / regime-routed), no training.
18 rollouts, ~1.5–1.8 h at step 1's measured ~6 min/rollout.

**Built-in consistency check:** routing is identical to the global gate on non-inverted
vessels, so the sweep's `global` column must reproduce §13.2's numbers. If it does not, one
of the two measurement paths is wrong and neither result is usable.

**Decision rule, fixed in advance:**
* **All 6 routed scores ≥ 0.60 → STOP.** Log the routed config as the answer; no training.
* **Otherwise** → continue to the architecture ladder (step 4). Step 3's per-window
  observability design gets ≤30 min of investigation and is dropped if nothing clean falls
  out (per instruction).

**Prediction, recorded so it can be scored:** routing will **not** close the gap.
§12.1 labelled `041` *normal*-regime, and routing by construction does nothing on normal
vessels — routed ≡ global there. `041` is both the worst vessel (0.3319) and the one whose
failure the gate cannot explain: it is at mass **0.389** *with* the gate on, i.e. it is
under-painting by 2.6× before the gate removes anything. `039` is the mirror case (mass
1.793, over-spray) where gate-off should make it *worse*. Expected: routing helps `042` a
little at most, moves `041` not at all, and the gap survives.

**The one way this prediction is interesting if wrong:** the `off` column is measured for
every vessel regardless of regime. If gate-OFF lifts `041`/`042` over 0.60 while the
*flow-regime* router leaves them alone, that says the router is keyed on the wrong variable —
routing on **mass/underseed** rather than on flow regime would then be a cheap, no-training
win, and §12.1's router would need re-specifying rather than re-confirming.

### 13.4 Step 2 RESULT — routing is structurally irrelevant to this cohort. Prediction confirmed.

`diag_regime_gate_sweep.py`, 6 vessels × 3 conditions, 31 min.

| vessel | regime | gate OFF | global | **routed** | routed−global |
|---|---|---|---|---|---|
| `039` | normal | 0.4251 | 0.4865 | 0.4865 | +0.0000 |
| `040` | normal | 0.3564 | 0.6100 | 0.6100 | +0.0000 |
| `041` | normal | 0.2475 | 0.3483 | 0.3483 | +0.0000 |
| `042` | normal | 0.3452 | 0.5182 | 0.5182 | +0.0000 |
| `043` | normal | 0.4112 | 0.7270 | 0.7270 | +0.0000 |
| `044` | normal | 0.4741 | 0.6465 | 0.6465 | +0.0000 |
| mean | | 0.3766 | 0.5561 | 0.5561 | +0.0000 |

**All six cohort vessels are normal-regime, so routing ≡ global on every one — it beat the
global gate on 0/6.** §12.1's large routing win came from a 12-vessel set containing six
*inverted* vessels, **none of which are in this cohort**. Routing is not a weak intervention
here; it is a no-op by construction. §12.1's finding stands on its own cohort and is simply
not transferable to `039`–`044`. **This closes the routing thread for this goal.**

The "interesting if wrong" branch of §13.3 also closed: gate-OFF is *worse* on all six
(mean 0.3766 vs 0.5561), and drops `041` to 0.2475 and `042` to 0.3452. So routing on
mass/underseed instead of flow regime would not help either — the gate is doing real work on
every vessel in this cohort. Step 2's stop condition does not fire; the gap is unchanged at
`039`, `041`, `042`.

### 13.4a The consistency check FIRED — two tools disagree on `deploy_clot_score`

> **DIAGNOSIS CORRECTED IN 20.1.** The cause is NOT `clout_prec_rec_floor` (it is inert at the
> ~1.000 relaxed recall every run had, and the typed values agree at 0.30). It is that the two
> tools used different scoring ENTRY POINTS, only one of which bound the canonical deploy
> protocol. The discrepancy itself is real; the attribution below is not.


§13.3 pre-registered: the sweep's `global` column must reproduce §13.2. It does not.

```
vessel   eval_mat_growth_simple   diag_regime_gate_sweep    diff     F1 (both tools)
039              0.4660                   0.4865          +0.0204   0.5185 / 0.5185
040              0.6218                   0.6100          -0.0118   0.7044 / 0.7044
041              0.3319                   0.3483          +0.0165   0.2548 / 0.2548
042              0.5030                   0.5182          +0.0152   0.5131 / 0.5131
043              0.6925                   0.7270          +0.0345   0.6497 / 0.6667
044              0.6400                   0.6465          +0.0065   0.6016 / 0.5992
```

**`deploy_clot_f1` is bit-identical on four of six vessels while `deploy_clot_score` differs
on all six.** Identical F1 means the predicted sets are identical, so this is not a rollout
difference — **the two tools score the same predictions differently.**

Cause: `deploy_clot_score` is `relaxed_prec_floor` mode, which is gated on
`clout_prec_rec_floor`, and that constant is **not consistent across the codebase**:
```
src/architecture/runtime_config.py:74      clout_prec_rec_floor: float = 0.30   (dataclass default)
src/biochem_gnn/mat_growth_simple.py:50    SPECIES_CLOUT_PREC_REC_FLOOR = "0.35" (recipe env)
src/biochem_gnn/mat_growth_simple.py:412   clout_prec_rec_floor: 0.30           (runtime kwarg)
```
The guiding metric of this whole study is **not uniquely defined**. This is the same class of
bug as §12.6's dead constants: a threshold that differs by provenance rather than by intent.

**Handling, stated rather than buried.** §13.3's rule said "neither result is usable" if this
fired. That rule was too strong and I am relaxing it deliberately, with the reason: the
disagreement is ≤0.0345 while the gap margins are 0.08–0.27, and **both paths select exactly
the same three vessels as the gap**. The decision is robust to the discrepancy even though the
numbers are not.

**Canonical path for the rest of this session: `eval_mat_growth_simple.py`** — it is the
launcher's own eval and the source of §9.15's headline 0.6497/0.6925. Every §13 number
compared against another §13 number will come from it. Reconciling the floor constant is
logged as work, not done here.

### 13.5 Step 3 — per-window observability: DEPRIORITIZED after investigation (~20 min)

The obvious candidate was `deploy_eval_time_fracs`, which already grades a leg at multiple
times per epoch (currently `"0.65,1.0"`). Raising it to 6 points would 3× the observation
count — if it were cheap. **It is not.** `train_species_pushforward_continuous.py:1552`
runs a *separate* `canonical_deploy_clot_metrics` call per time point, each performing its own
rollout:

```python
for t_clot in clot_times:
    clf_by_t[int(t_clot)] = canonical_deploy_clot_metrics(model, ...)   # one rollout each
```

So observation count is exactly linear in GPU cost — it buys nothing over just running more
epochs.

The correct design is roll-once-grade-many: one rollout, snapshot the committed set at every
requested time, score each snapshot. That is genuinely cheap (rolling is the expensive part;
scoring a snapshot is not). But it requires editing `canonical_deploy_clot_metrics`, which is
the **shared** path behind `eval_mat_growth_simple.py`, `diag_regime_gate_sweep.py` and the
in-training eval — i.e. the path §13.4a has just shown is *already* inconsistent between
callers. Changing it mid-session would make §13.2's baseline non-comparable with everything
measured after it.

**Per the session's own instruction, deprioritized rather than attempted.** Falling back to
§12.8.2's 14-epoch floor. Logged as the highest-value cheap fix for a session that starts with
a clean measurement path — do it *before* establishing a baseline, not after.

### 13.6 Step 4a — `latent_dropout` 0.0 → 0.30 (v10) — PRE-REGISTERED

Written before launch. `WG_stenosis_subcohort_ft_v10` = **v7 + exactly one change**:
`latent_dropout 0.0 → 0.30`. Base is v7 (v3 + §12.6.1's occupancy fix), **not** v9 — v9
bundles three changes and would make this unattributable, which is the mistake that cost
v1–v6. **14 epochs** per §12.8.2's floor; 6 would be unreadable. ~2.6 h at v9's uncontended
~680 s/epoch.

**Rationale:** §11.2.1's largest structural imbalance is that `z_kin` is 256 of a 287-dim
input (89%) and is a *frozen, off-task* kinematics latent. Dropout on that slice is the
cheapest possible probe of "is the readout leaning on the off-task latent instead of the
geometry/flow channels". Config-only, no new code.

**Mechanism-engaged check (constraint 5, the v4/v5 lesson):** the trainer prints
`[i] latent leash: dropout p=0.30 on z_kin[:N]` at line 921 only when `p > 0`, and
`maybe_drop_latent` zeroes `base_feats[:, :kin_latent_dim]` per unrolled step. **If that line
is absent from the log, the run is void and will not be reported as a result.**

**Decision rule, fixed in advance:**
* **All 6 cohort vessels ≥ 0.60 on the canonical `eval_mat_growth_simple.py` path → STOP.**
* Otherwise → step 4b (`z_kin` shrink) if budget remains, else stop and log.

**Stated limitation, not discovered afterwards:** the default split trains on
`039,040,041,042,044` and seals only `043`. **Two of the three gap vessels (`041`, `042`) are
in the training set**, so any improvement on them is *in-sample* and is weak evidence for the
stated goal of generalizing to unseen vessels. `043` remains the only honest generalization
probe here, and it already clears the bar zero-shot (0.6925). A proper leave-one-out over the
gap vessels is 3× this cost and does not fit the remaining budget. Read `041`/`042` gains, if
any, as an upper bound.

**Prediction, recorded so it can be scored:** this will **not** close the gap. `041` needs
+0.27. Across v1–v9, no head-only fine-tune has moved the deploy state off the `fp ≈ 292`
attractor at all, and `latent_dropout` is a regulariser, not a mechanism that adds growth.
The genuinely valuable outcome is the *first properly-powered* measurement (14 epochs,
~2.0 expected excursions vs the 0.9 every prior leg had) of whether head-only training moves
the attractor **at all** — which is the question §12.8.2 showed no previous leg could answer.

### 13.7 Step 4a/5 RESULT — v10 REGRESSES the held-out vessel. Prediction confirmed.

Cohort eval of v10's promoted salvage (ep 12), canonical `eval_mat_growth_simple.py`, pct-25
gate, vs §13.2's zero-shot baseline. Mechanism was verified engaged
(`[i] latent leash: dropout p=0.30 on z_kin[:256]`), so this is a real test.

| vessel | zero-shot | v10 | Δ | v10 mass | split | ≥0.60 |
|---|---|---|---|---|---|---|
| `039` | 0.4660 | 0.3921 | **−0.0739** | 2.483 | in-sample | no |
| `040` | 0.6218 | **0.7072** | +0.0854 | 1.312 | in-sample | YES |
| `041` | 0.3319 | 0.4600 | **+0.1282** | 2.186 | in-sample | no |
| `042` | 0.5030 | 0.4399 | −0.0631 | 2.339 | in-sample | no |
| `043` | 0.6925 | 0.5555 | **−0.1370** | 1.674 | **HELD OUT** | no |
| `044` | 0.6400 | 0.5632 | −0.0767 | 1.497 | in-sample | no |
| mean | 0.5425 | 0.5197 | −0.0228 | | | |

**The gap went from 3 vessels to 5.** The only honest generalization probe, `043`, dropped
**−0.1370**. §13.6's prediction ("this will not close the gap") is confirmed, and by a wider
margin than expected — the leg is not neutral, it is harmful.

**A third measurement path, and the in-training number was optimistic.** v10 ep12 reported
`deploy_clot_score = 0.6221` on `043` *during training*; the canonical eval scores the same
checkpoint at **0.5555** on the same vessel — a **+0.0666 overstatement**. In-training eval
uses `train_deploy_eval_flow="auto"`; §13.4a already found `eval_mat_growth_simple.py` and
`diag_regime_gate_sweep.py` disagree by up to 0.0345. **There are now three mutually
inconsistent implementations of the guiding metric**, and the in-training one — the one every
leg's selection and every §9/§12 conclusion is based on — is the most optimistic of them.

### 13.7.1 The actual mechanism: one global knob, a cohort with opposite needs

Every vessel's mass went **up** under v10, without exception:

```
039 1.793 -> 2.483    040 1.065 -> 1.312    041 0.389 -> 2.186
042 0.752 -> 2.339    043 0.653 -> 1.674    044 0.571 -> 1.497
```

`latent_dropout` did not change *where* the model paints, it changed *how much*, uniformly.
That helps exactly the vessels that were starved — `041` (mass 0.389, the most under-massed,
gains **+0.128**) and `040` (gains +0.085) — and hurts every vessel that was already near
target by pushing it into overshoot, `043` worst of all (0.653 → 1.674, −0.137).

**This is the same structural finding as §13.2/§13.4, now demonstrated on the training side
as well as the gate side: the cohort needs *per-vessel* growth magnitude, and every knob
available — gate percentile, loss weights, latent dropout — is *global*.** `041` needs 5.6×
more mass; `043` needs 1.0×. No single scalar satisfies both, which is why ten legs have
failed and why the mean barely moves while individual vessels swing ±0.14.

**The zero-shot warm start remains the best configuration in the study.** Ten fine-tune legs
(v1–v10), none has beaten it on a held-out vessel.

### 13.8 Step 4b BLOCKED by constraint 2 — session stopped here, with budget remaining

`z_kin` 256 → 64 changes the model input width from 287 to **95**. The warm start's first-layer
weights are fixed at that width:

```
WG_clotrich_nplus  in_dim = 287
  conv1.lin_l.weight       (64, 287)
  skip_conv1.lin_l.weight  (64, 287)
=> cannot load into (64, 95)
```

Every leg v1–v10 is warm-started from `WG_clotrich_nplus`, and §13.2's baseline *is* that warm
start. So step 4b as specified cannot be run as a single-variable test: it necessarily bundles
**"shrink `z_kin`"** with **"lose the warm start"**, and the second is the larger effect by far
(a cold start would not remotely reach 0.55 in 14 epochs). Constraint 2 says split that into
two legs; two 14-epoch legs plus two cohort evals is ~5 h, and 3.4 h remain.

The bottleneck workaround — `z' = up(down(z))`, 256→64→256, preserving `in_dim` — is not
identity at initialisation for any rank < 256, so it *also* perturbs the warm start on epoch 1
and *also* bundles two changes.

**Stopped here at ~3.4 h of a 7 h budget, deliberately, rather than spending it on a test that
could not be attributed.** Constraint 2 exists because v1–v6 were lost to exactly this.

**Secondary evidence pointing the same way, from 4a's own result.** Step 4a *is* a test of the
hypothesis behind 4b — "the readout over-relies on the frozen off-task `z_kin`". Dropout
removes `z_kin` stochastically; if it were harmful, removal should help. It **hurt the held-out
vessel by −0.137**. That is evidence against the premise of 4b, not for it. A learned
projection compresses rather than destroys, so this is suggestive rather than decisive — which
is why the blocker above, not this, is the stated reason for stopping.

### 13.9 Session summary — where this actually stands

| | mean | `043` (held out) | vessels ≥0.60 |
|---|---|---|---|
| **zero-shot warm start** | **0.5425** | **0.6925** | **3/6** |
| + regime routing | 0.5561* | 0.7270* | 3/6 |
| v10 (14-epoch FT) | 0.5197 | 0.5555 | 1/6 |

*routing measured on the second, inconsistent path (§13.4a); routed ≡ global on all six.

**The goal is not met: 3 of 6 vessels are below 0.60 zero-shot (`039` 0.466, `041` 0.332,
`042` 0.503), and nothing tried this session improved that.** The zero-shot warm start is
still the best configuration in the study after eleven attempts.

**What this session established that previous ones could not:**
1. The cohort has **two opposite failure modes** — `039` over-paints (mass 1.79), `041`/`042`
   under-paint (0.39/0.75). Confirmed independently on the gate side (§13.4) and the training
   side (§13.7.1).
2. **Every available knob is global and the requirement is per-vessel.** `041` needs ~5.6×
   more mass, `043` needs ~1.0×. Gate percentile, loss weights and latent dropout all move all
   six together. This is why ten legs failed, and it predicts steps 4b–4d fail the same way.
3. **Regime routing is inapplicable here** — all six vessels are normal-regime, so routing ≡
   global gate on every one (§13.4). §12.1's win came from a disjoint set of inverted vessels.
4. **Three mutually inconsistent implementations of `deploy_clot_score`** (§13.4a, §13.7),
   spanning 0.0666 on a single vessel/checkpoint, with the in-training one — which drives all
   selection and every §9/§12 conclusion — the most optimistic.

**Highest-value next work, in order:**
1. **Reconcile `clout_prec_rec_floor`** (0.30 vs 0.35 vs 0.30 across three files) and make one
   scoring implementation canonical. Nothing measured here is fully trustworthy until this is
   done, and it is nearly free.
2. **Per-vessel conditioning**, not another global knob. Finding 2 says the model needs a
   growth-magnitude signal that varies by vessel. `band_speed_q25` is already computed per
   vessel and already separates the cohort; feeding it as a conditioning scalar is the
   cheapest form of this and was already proposed in §11.3.
3. Only then the `z_kin` ladder, run as the **two** legs constraint 2 requires.

# 14. Ground-up review: data, physics, and a new architecture direction (2026-08-07)

Written after stepping away from fine-tuning. Every number below is measured this session on
the raw anchor packs, not carried over. **Several long-standing assumptions in this document
do not survive contact with the data.**

## 14.1 Data inventory — we have been training on 14% of the available data

```
data/processed/graphs_biochem_anchors/   43 patient packs (+ mirror augmentations)
  with >= 20 clotted nodes at t_final:   35
  actually used by every leg v1-v10:      5   (039,040,041,042,044)
```
Eight packs (`017,022,023,026,027,030,033,034`) have **zero** clot and are unusable as
positives, but the other **35 are usable and 30 have never been trained on.** Every
generalization claim in sections 9-13 rests on a 5-vessel training set and a 1-vessel holdout.

## 14.2 Feature audit — 6 of 18 input channels are dead, 2 are unnormalised

`data.x` is `kine_x_v1_18ch`. Measured on `039/041/043` (identical on all three):

| ch | name | status |
|---|---|---|
| 0-2 | `x_nd, y_nd, sdf_nd` | OK |
| 3 | `shear_potential` | OK — and one of the better predictors (14.4) |
| 4-5 | `wall_normal_x/y` | OK |
| 6-9 | `node_type_0..3` | **IDENTICALLY ZERO on every node of every vessel** |
| 10 | `rheology_flag` | **CONSTANT 1.0** — zero variance |
| 11-13 | `u_prior, v_prior, mu_prior_nd` | = clot-free CFD initial condition (see 14.3) |
| 14 | `wss_prior_nd` | **IDENTICALLY ZERO** — confirms 11.2.3 |
| 15 | `width_nd` | OK |
| 16 | `width_d1` | range +/-272, unnormalised |
| 17 | `width_d2` | range **-307566 ... +1968**, unnormalised |

**Six of eighteen channels carry no information at all**, and `width_d2` spans five orders of
magnitude unnormalised. The model does not consume `data.x` directly — it consumes
`[z_kin(256), sdf, flow_feats, geom_feats, flux_stag, state, sat, time]` — but `z_kin` is
produced by the kinematics DEQ *from* `data.x`, so the dead channels sit upstream of everything.

## 14.3 RETRACTED — the `u_prior`/`mu_prior` "GT leak" is not a leak

> **THIS SECTION IS ITSELF RETRACTED — see 16.1.** `u_prior` is a converged Navier-Stokes
> field (it contains backflow, which a clamped parabolic magnitude cannot produce), and the
> RGP-DEQ consumes it as an input. It IS initial-condition leakage relative to the
> geometry-to-surrogate deployment contract. The reasoning below is kept as the record of a
> wrong turn; do not act on it.


The handoff and section 11 state these channels are "bit-identical to GT `y[0]`... as stored,
they are a leak", and blocked the analytical-prior work on that basis. The bit-identity is real
and reproduced here. **The inference drawn from it is wrong:**

```
patient041  Mat at t=0: max = 0.0, nonzero nodes = 0      FI at t=0: max = 0.0
patient043  Mat at t=0: max = 0.0, nonzero nodes = 0      FI at t=0: max = 0.0
```

`y[0]` is the **clot-free initial state**. `u_prior = u_nd(t=0)` is therefore the CFD solution
on the clean geometry — exactly what a deployment pipeline computes before any clot exists. It
is a legitimate, causally-available input, not future information. Leakage would require
`y[t>0]`.

**Consequence:** the entire "recompute an analytical Poiseuille prior to avoid the leak"
workstream is unnecessary. The real flow field is already in the pack, is far better than a
Poiseuille approximation, and is legal. This unblocks the strongest physics features available.

## 14.4 What actually predicts clot — measured, clot-free-legal features only

AUC for "node is committed at `t_final`", within the 3-hop wall band, using only features
computable at `t=0` without knowing the clot:

| feature | 039 | 040 | 041 | 042 | 043 | 044 | **mean** |
|---|---|---|---|---|---|---|---|
| **residence time (1/speed)** | 0.881 | 0.806 | 0.670 | 0.691 | 0.823 | 0.679 | **0.758** |
| **-WSS (= -mu * shear rate)** | 0.814 | 0.782 | 0.645 | 0.643 | 0.818 | 0.674 | **0.729** |
| `shear_potential` | 0.857 | 0.656 | 0.646 | 0.691 | 0.681 | 0.663 | 0.699 |
| `mu_eff(t=0)` | **0.928** | **0.847** | 0.484 | 0.473 | **0.907** | 0.516 | 0.693 |
| `width_grad` | 0.797 | 0.689 | 0.623 | 0.618 | 0.656 | 0.587 | 0.662 |
| `width_nd` | 0.581 | 0.611 | 0.357 | 0.355 | 0.630 | 0.340 | 0.479 |
| `sdf_nd` | 0.146 | 0.332 | 0.331 | 0.308 | 0.320 | 0.345 | 0.297 |

Two things stand out.

1. **Low speed / long residence time is the single most consistent predictor** (0.758), with
   low wall shear stress right behind (0.729). This matches the COMSOL mechanism directly.
2. **`mu_eff(t=0)` is bimodal and splits the cohort exactly**: AUC 0.93/0.85/0.91 on
   `039/040/043` versus 0.48/0.47/0.52 — *chance* — on `041/042/044`. The cohort contains
   **two physically distinct clot mechanisms**, and this is the mechanistic version of the
   "global knob, per-vessel need" problem 13.7.1 found empirically.

## 14.5 The finding that reframes the architecture: a linear model matches the GNN

> **RETRACTED — see 19.2.** This used LEAKED features (16.1) on the favourable 039-044 cohort
> with an oracle threshold. Redone properly (deploy-legal features, trained on the 29
> non-cohort vessels, same 6 test vessels) the GNN wins 0.540 vs 0.516 while handicapped.
> The rollout does earn its place. Do not cite this section's conclusion.


Leave-one-vessel-out logistic regression, 8 physics features, no `z_kin`, no GNN, no rollout:

```
LOO AUC:  039 0.941   040 0.928   041 0.749   042 0.755   043 0.891   044 0.749   mean 0.836
```

Converted to best-threshold F1 and set against the current model's actual `deploy_clot_f1`:

| vessel | logreg F1 | GNN `deploy_clot_f1` | delta |
|---|---|---|---|
| `039` | 0.6667 | 0.5185 | +0.1482 |
| `040` | 0.6953 | 0.7044 | -0.0091 |
| `041` | 0.4674 | 0.2548 | +0.2126 |
| `042` | 0.4585 | 0.5131 | -0.0546 |
| `043` | 0.6627 | 0.6497 | +0.0130 |
| `044` | 0.4032 | 0.6016 | -0.1984 |
| **mean** | **0.5590** | **0.5403** | **+0.0186** |

The logreg is oracle-thresholded per vessel, so it is an upper bound and not apples-to-apples
with a full gated rollout. **But 187k parameters, a 256-dim frozen latent and a 200-step
autoregressive rollout are not beating eight hand-computed physics numbers and a dot product.**

Note also **AUC 0.836 -> F1 0.559 even at the oracle threshold.** The ranking is respectable;
the score is destroyed by sparsity (4-15% positives in-band). Better *ranking* is the lever,
not better thresholding.

## 14.6 The mechanism the architecture is missing: nucleation

Every commit event decomposed into **growth** (node had a committed neighbour when it turned
on) vs **nucleation** (it did not):

| vessel | commits | growth | nucleation | **nucleation %** | nucleation by time quartile |
|---|---|---|---|---|---|
| `039` | 92 | 45 | 47 | **51.1%** | [41, 5, 0, 1] |
| `040` | 144 | 76 | 68 | **47.2%** | [41, 0, 25, 2] |
| `041` | 266 | 186 | 80 | 30.1% | [29, 34, 3, 14] |
| `042` | 251 | 184 | 67 | 26.7% | [28, 23, 12, 4] |
| `043` | 167 | 70 | 97 | **58.1%** | [56, 5, 27, 9] |
| `044` | 379 | 260 | 119 | 31.4% | [41, 40, 17, 21] |

**27-58% of all commits are nucleation, and it continues into the last quartile** — not just an
initial seeding transient. Corroborating: only 43-81% of the final clot lies within 3 hops of
the `t=20` clot, and dilating *oracle* `t=20` seeds by the best k caps at **F1 0.52-0.64** —
roughly where the real model already is.

**The current architecture is a growth/propagation model.** The autoregressive rollout, the
neighbour-commit gate, and section 11.3's proposed autocatalytic term all strengthen
*growth-from-existing-clot* — the half of the problem that already works. A third to a half of
the clot appears where nothing was, driven by the local *field* (low WSS, long residence),
which is exactly why the field-only logistic regression in 14.5 is competitive.

**A seed-and-grow model has a ceiling near F1 0.58 on this cohort. That ceiling is where ten
fine-tuning legs have been stuck.**

## 14.7 Label inconsistency — the commit threshold means different things per vessel

`max Mat` across the 35 usable packs spans **3.3e-4 to 1.5e-2, a 45x range**, while the commit
threshold is a fixed `1e-4`. On `patient003` that threshold is ~25% of the vessel's peak Mat;
on `patient012` it is ~0.7%. "Committed" is therefore not a consistent physical state across
vessels. This is a *label* problem sitting underneath every metric in this document, and it is
a third instance of the same pathology as 12.6 (dead constants) and 13.4a (three different
`clout_prec_rec_floor` values): **a global constant standing in for a per-vessel quantity.**

# 15. Ranked experiment list — systematic path forward (2026-08-07)

Ordered by **information gained per GPU-hour**, not by expected score. Tiers A-B are cheap and
several are near-certain to change how everything downstream is read; do them first. Each entry
states what it tests, what it costs, and **what result would kill it** — so a negative is as
useful as a positive.

Standing rules carried from 12.8.2 / 13: one variable per leg, >= 14 epochs for any leg whose
readout is the excursion rate, no concurrent GPU jobs, verify the mechanism engaged before
trusting any result.

## Tier A — free or near-free, and they change the meaning of every later number

**A1. Unify `deploy_clot_score`.** Three implementations disagree by up to 0.14 on one
checkpoint (13.4a, 13.7): `clout_prec_rec_floor` is 0.30 / 0.35 / 0.30 across three files, and
the in-training version driving all selection is the most optimistic. Pick one, make the other
two call it, re-derive the 13.2 baseline.
*Cost:* ~0 GPU. *Kill:* n/a — this is a precondition, not a hypothesis.

**A2. Per-vessel commit threshold.** 14.7: `max Mat` spans 45x across vessels against a fixed
`1e-4` threshold, so "committed" is a stricter physical state on some vessels than others.
Re-derive labels at a per-vessel relative threshold (e.g. `alpha * max Mat` or a fixed
percentile of the vessel's own final Mat) and re-measure the 14.4 AUCs and the 13.2 baseline.
*Cost:* ~0 GPU. *Kill:* if AUCs and per-vessel scores barely move, the absolute threshold is
fine and this is closed — worth knowing either way.

**A3. Delete the six dead channels; normalise `width_d1/d2`.** 14.2: four `node_type` channels
are identically zero, `rheology_flag` is constant, `wss_prior_nd` is zero, `width_d2` spans
-307566..1968 unnormalised. These feed the kinematics DEQ that produces `z_kin`.
*Cost:* ~0 GPU to fix; requires re-running the kinematics encoder to see the effect.
*Kill:* if `z_kin` is dropped entirely (B2/C1) this matters much less — sequence accordingly.

**A4. Expand the cohort from 5 to ~30 training vessels.** 14.1: 35 usable packs exist, 30 never
trained on. This is the single largest untapped resource in the project and costs only
wall-clock. Do it as its own leg (constraint 2) so it is not confounded with an architecture
change.
*Cost:* longer epochs, no new code. *Kill:* if held-out score does not improve with 6x the
data, the bottleneck is definitively architectural, not statistical — a very valuable negative.

## Tier B — cheap model experiments that test the 14.5/14.6 reframing directly

**B1. Ship the logistic-regression baseline as a real deploy arm.** 14.5 shows 8 physics
features match the GNN. Run it through the actual deploy path (fixed gate, no oracle
threshold) so the comparison is apples-to-apples.
*Cost:* ~1 GPU-hour. *Kill:* if it collapses under a fixed gate it was a thresholding artefact
and the GNN is doing more than it appears — also valuable.

**B2. `z_kin` ablation, done properly.** Never actually tested. 13.8 blocked the *shrink*
because `in_dim` 287->95 breaks the warm start; the *ablation* has no such problem — zero the
`z_kin` block at both train and deploy, keeping `in_dim` at 287. If performance is unchanged,
89% of the input is dead weight and the whole `z_kin` ladder (11.2.1) closes at once.
*Cost:* ~1 leg. *Kill:* if scores drop materially, `z_kin` is load-bearing and C1 must keep it.
Note 4a is weak evidence it *is* load-bearing (dropout hurt the holdout by 0.137).

**B3. Longer training windows / autoregressive horizon.** Open question from the brief. Current
`unroll` is 5 (curriculum) to 25; the deploy rollout is 200 steps. v10's two deep excursions
both occurred *after* the curriculum stepped to `unroll=10` (13.7) — suggestive that windows
are too short for autoregressive error to self-correct.
*Cost:* 1 leg at >= 14 epochs, `unroll` 25-50, `curriculum_unroll=False`. *Kill:* v6 already
tested `unroll=25` and found nothing — but v6 ran with the dead loss terms of 12.6, so this
deserves exactly one clean re-test, not more.

## Tier C — the architecture the evidence actually points to

**C1. Two-term model: nucleation field + growth propagation.** The central proposal, straight
from 14.6. Replace the single generic delta head with an explicit sum:

```
dMat = NUCLEATION(local physics field)  +  GROWTH(local committed Mat) * gate(shear)
```
* `NUCLEATION` — a per-node rate from the 14.4 features (residence time, -WSS,
  `shear_potential`, `width_grad`, `mu_eff(t=0)`), **independent of neighbouring clot**. This is
  the 27-58% of commits the current architecture cannot express.
* `GROWTH` — the autocatalytic term already implemented for v9 (`k_dep + k_auto * local
  committed`), which is correct for the other half.

*Cost:* real implementation + >= 14-epoch leg. *Kill:* if a nucleation head trained on the
14.4 features cannot beat the logreg of 14.5, the field model is saturated and the remaining
error is temporal/calibration, not spatial.

**C2. Per-vessel conditioning.** 13.7.1 and 14.4 independently say the cohort needs per-vessel
*amount*, and 14.4 gives the mechanism: `mu_eff(t=0)` is diagnostic on `039/040/043` and
useless on `041/042/044`. Feed a small per-vessel descriptor (`band_speed_q25`, median WSS,
`mu_eff` AUC-proxy, geometry class) as a global conditioning vector.
*Cost:* moderate. *Kill:* if conditioning does not separate the two regimes in the learned
representation, the split is not recoverable from `t=0` data.

**C3. Reframe as ranking, not per-step regression.** 14.5: AUC 0.836 -> F1 0.559 at the oracle
threshold. Ranking quality is the binding constraint, and the current objective is a per-step
Huber regression on deltas — the thing change B spent four legs failing to align (12.3). Train
directly on a ranking/AUC surrogate over the final committed set.
*Cost:* moderate; the soft-F_beta term from 12.6 is a partial version already built.
*Kill:* if a ranking objective does not lift LOO AUC above ~0.90, the features are the limit,
not the loss.

## Tier D — deferred, with reasons

**D1. Analytical Poiseuille prior.** **Now unnecessary** — 14.3 shows the real clot-free CFD
field is already in the pack and legal. Use `u_prior/v_prior/mu_prior` directly instead.

**D2. Populate `wss_prior_nd`.** Superseded by A3/C1: WSS is trivially computable from
`u_prior/mu_prior` on the fly (that is exactly the `-WSS` feature at AUC 0.729). Populating the
stored channel is redundant with computing it.

**D3. `z_kin` shrink 256->64.** Gated behind B2. If the ablation says `z_kin` is dead weight,
shrink is pointless — delete it. If it says load-bearing, shrink becomes worth the two legs
constraint 2 requires.

**D4. Regime routing.** Closed for this cohort (13.4): all six vessels are normal-regime, so
routed == global on every one.

**D5. Further change-B objective reweighting.** Closed (12.3, three failed specifications).
C3 is the replacement — a different objective *family*, not another weighting of this one.

## Suggested first sequence

`A1 -> A2 -> A4` (all cheap, and A1/A2 change how A4 is read), then `B2` (settles `z_kin` and
therefore D3), then `C1` as the main architectural bet, with `B3` as a one-shot control on
window length. `B1` is worth running early purely as an honest floor to measure against.

# 16. Corrections and cohort-wide physics review (2026-08-07, later same day)

Three challenges were put to section 14. **Two of its conclusions do not survive, including one
of its headline claims.** Everything below is measured on all 35 usable packs, not on 6.

## 16.1 RETRACTION OF 14.3 — `u_prior` IS a leak, and the RGP-DEQ is fed the answer

Section 14.3 argued that because `y[0]` is clot-free, `u_prior = u_nd(t=0)` is a legitimate
initial condition rather than leakage. **That reasoning was incomplete and the conclusion is
wrong.** Three measurements settle it.

**(a) `u_prior` is a converged Navier-Stokes field, not an analytical prior.**
`build_poiseuille_priors` computes `u_prior_mag = u_max * (1 - r_lane^2/r^2)` clamped to
`min=0` — a non-negative parabolic magnitude. The stored field contains backflow:

```
patient041   u_prior min = -1.0534    9.0% of nodes have u < 0
patient043   u_prior min = -0.0591    4.9% of nodes have u < 0
```
A clamped parabolic magnitude cannot produce recirculation. This is a CFD solution.

**(b) `build_poiseuille_priors` output was never stored.** That function also returns
`wss_prior = mu_prior * gamma_dot * mask_wall`, which is non-zero on every wall node. The
stored `wss_prior_nd` is **identically zero on every vessel** (14.2). The three channels that
*do* have a `y[0]` counterpart were overwritten with it; the one that does not have a
counterpart (`wss`) was left empty. That is the signature of GT being written over the priors.

**(c) The RGP-DEQ consumes them as input.** `src/architecture/ginodeq.py:438-440`:
```python
uv_prior = data.x[:, NodeFeat.UV_PRIOR]
p_prior  = data.x[:, NodeFeat.SHEAR_POT]
mu_prior = data.x[:, NodeFeat.MU_PRIOR]
```
**The flow surrogate whose job is to predict the velocity field is being handed the true
velocity field as an input feature.** `z_kin` is therefore conditioned on the converged CFD
solution, and its genuine out-of-distribution quality has never been measured.

This is not *temporal* leakage — no clot information is involved, `y[0]` really is clot-free.
It is **initial-condition leakage relative to the deployment contract**: at deploy on unseen
geometry you have a mesh, not a converged solve. The correct chain is
`geometry -> RGP-DEQ -> predicted flow`, and the stored priors short-circuit it.

**How much of section 14.4's feature table was leak** — same feature, computed from the stored
CFD field versus recomputed analytically from `(sdf_nd, width_nd)` alone:

| feature | source | 039 | 040 | 041 | 042 | 043 | 044 | mean |
|---|---|---|---|---|---|---|---|---|
| `-speed` | GT-CFD | 0.881 | 0.806 | 0.670 | 0.691 | 0.823 | 0.679 | 0.758 |
| `-speed` | **analytic** | 0.901 | 0.869 | 0.668 | 0.668 | 0.874 | 0.666 | **0.774** |
| `-WSS` | GT-CFD | 0.814 | 0.782 | 0.645 | 0.643 | 0.818 | 0.674 | 0.729 |
| `-WSS` | **analytic** | 0.563 | 0.578 | 0.347 | 0.347 | 0.606 | 0.331 | **0.462** |
| `mu_eff` | GT-CFD | 0.928 | 0.847 | 0.484 | 0.473 | 0.907 | 0.516 | 0.693 |
| `mu_eff` | **analytic** | 0.563 | 0.578 | 0.347 | 0.347 | 0.606 | 0.331 | **0.462** |

**The best feature survives; the strong ones do not.** Analytic `-speed` is *better* than the
leaked version (0.774 vs 0.758) and costs nothing. But `-WSS` and `mu_eff` collapse to 0.462 —
their predictive power was entirely the CFD recirculation structure, which a parabolic profile
cannot reproduce. (Analytic `mu` and analytic `-WSS` are monotone transforms of each other,
hence identical AUC.)

## 16.2 The aneurysm/stenosis mechanism — and why the framing was wrong anyway

Taking the question at face value first. All biochem intermediates at `t=10`, AUC for final Mat:

| vessel | class | RP | AP | APR | APS | FI | Mas | **mu_eff** |
|---|---|---|---|---|---|---|---|---|
| `039` | ANEUR | 0.036 | 0.030 | 0.965 | 0.961 | 0.941 | 0.971 | **0.928** |
| `040` | ANEUR | 0.056 | 0.053 | 0.947 | 0.885 | 0.856 | 0.971 | **0.847** |
| `043` | ANEUR | 0.067 | 0.059 | 0.936 | 0.911 | 0.892 | 0.973 | **0.907** |
| `041` | STEN | 0.201 | 0.172 | 0.817 | 0.766 | 0.712 | 0.839 | **0.488** |
| `042` | STEN | 0.226 | 0.192 | 0.792 | 0.762 | 0.704 | 0.680 | **0.474** |
| `044` | STEN | 0.205 | 0.192 | 0.804 | 0.798 | 0.770 | 0.830 | **0.522** |

`RP`/`AP` are inverted (they are *consumed* where clot forms) — that is the reaction working as
written. Every transported species (`APR`, `APS`, `FI`, `Mas`) predicts well in **both** classes.
**Only `mu_eff` collapses on the stenoses.** So the chemistry is not different — the *coupling
between local hemodynamics and deposition site* is.

**Transport test.** If stenosis clot is seeded by shear-activation at the throat and then
deposits downstream, an advected shear-sourced scalar should beat local shear there and not in
aneurysms. Upwind-advecting a shear source along the `t=0` field:

| vessel | class | local shear | advected | gain | recirc (`-u`) |
|---|---|---|---|---|---|
| `039` | ANEUR | 0.174 | 0.126 | −0.048 | 0.799 |
| `040` | ANEUR | 0.193 | 0.179 | −0.014 | 0.832 |
| `043` | ANEUR | 0.189 | 0.156 | −0.033 | 0.855 |
| `041` | STEN | 0.363 | 0.422 | **+0.059** | 0.622 |
| `042` | STEN | 0.366 | 0.429 | **+0.063** | 0.612 |
| `044` | STEN | 0.337 | 0.392 | **+0.055** | 0.625 |

The sign is exactly as predicted — advection helps stenoses, hurts aneurysms — but the effect
is small (~0.06). The larger difference is **recirculation strength**: `-u` scores 0.80-0.86 on
aneurysms versus 0.61-0.63 on stenoses. An aneurysm has one clean enclosed stagnation pocket
that localises the clot; a stenosis does not, so its clot is spatially diffuse (and its
positive rate is ~2x higher: 10.6-15.1% vs 4.1-7.5%).

**But see 16.3 — this whole framing is a six-vessel artifact.**

## 16.3 Cohort-wide: the feature rules do NOT generalize, and `mu_eff` is not bimodal — it is unreliable

All 35 usable vessels, AUC for final Mat within the 3-hop wall band:

```
feature      mean   median   >0.65      <0.50
-anaSpd     0.768   0.746    32/35       0/35     <- deploy-legal, geometry only
-sdf        0.768   0.749    31/35       0/35     <- deploy-legal, geometry only
-u (CFD)    0.760   0.773    28/35       0/35     <- needs the flow field
wgrad       0.717   0.676    22/35       0/35     <- deploy-legal, geometry only
shear (CFD) 0.343   0.337     2/35      30/35     <- consistently inverted; as -shear ~0.657
mu_eff (CFD) 0.482  0.473    12/35      19/35     <- NOT USABLE
```

**`mu_eff` averages 0.482 — worse than chance — and is *anti*-predictive on 19 of 35 vessels,
as low as 0.050 (`p011`), 0.056 (`p018`), 0.058 (`p025`).** It is not a feature that works on
aneurysms and fails on stenoses. It is a feature that points in an arbitrary direction
depending on the vessel. Section 14.4's 0.93/0.85/0.91-versus-0.48/0.47/0.52 split was a
coincidence of which six vessels were in the cohort.

**This retires the "two clot mechanisms" reading of 14.4.** There is one mechanism; there is a
weak, robust geometric signal; and there is a strong flow-structure signal whose *sign* varies
by vessel because it depends on recirculation topology that neither geometry nor a parabolic
prior encodes.

## 16.4 The honest deploy-legal ceiling — and why the flow surrogate is the real bottleneck

> **PARTLY SUPERSEDED — see 18.2.** Z1 has now measured the surrogate directly: on the legal
> prior path it retains 97% of the GT field's clot AUC (0.763 vs 0.789). The flow surrogate is
> NOT the binding constraint. The F1 0.322 ceiling below still stands; the gap is nucleation,
> temporal dynamics and calibration, not flow prediction.


Logistic regression, **deploy-legal features only** (geometry + analytic Poiseuille: `-anaSpd`,
`-sdf`, `wgrad`, `wgrad_2hop`, `curv`, `width`, `-anaGamma`, `shear_potential`, and two hop
means). 26 training vessels, 9 held out:

| held-out vessel | AUC | best F1 | pos% |
|---|---|---|---|
| `patient001` | 0.789 | 0.427 | 13.3% |
| `patient005` | 0.663 | 0.191 | 6.5% |
| `patient009` | 0.901 | 0.427 | 6.4% |
| `patient013` | 0.848 | 0.610 | 17.6% |
| `patient018` | 0.664 | 0.084 | 1.9% |
| `patient024` | 0.856 | 0.423 | 7.2% |
| `patient031` | 0.617 | 0.092 | 2.0% |
| `patient037` | 0.798 | 0.296 | 7.1% |
| `patient042` | 0.767 | 0.352 | 10.6% |
| **mean** | **0.767** | **0.322** | |

Set against section 14.5's leaked-feature run (AUC 0.836, F1 0.559 on 6 vessels):

**Geometry alone cannot reach the goal.** Mean best-threshold F1 of **0.322** against a target
of 0.60, and that is with an oracle per-vessel threshold. The signal that closes the gap —
recirculation topology, `-u` at AUC 0.760 and `mu_eff` where it works — requires a *real
velocity field*.

**Therefore the RGP-DEQ flow surrogate is the binding constraint of this entire project, and
its true accuracy has never been measured**, because it is handed `u_prior`/`mu_prior` as
inputs (16.1c). Section 10.6's "predicted-flow inflation up to 9x on slow vessels" is the only
hint of its unassisted quality, and it is not encouraging.

Everything downstream — nucleation heads, autocatalytic growth, per-vessel conditioning, loss
surrogates — is being tuned on top of a flow field that is either leaked (in training) or of
unknown quality (at deploy). **That is the most probable single explanation for why eleven legs
have failed to beat a zero-shot warm start.**

# 17. Revised experiment list (supersedes section 15)

Section 15 was written before the 16.1 retraction and the 35-vessel survey. Those two results
reorder it substantially: **the flow surrogate moves to the top, and three section-15 entries
are now known to be built on leaked inputs.**

## Tier 0 — must happen before any other result is interpretable

**Z1. Measure the RGP-DEQ's unassisted flow accuracy.** *The single most important open
question in the project.* It currently receives `u_prior`/`v_prior`/`mu_prior` — the converged
CFD answer — as input features (16.1c). Zero or shuffle those three channels and measure
predicted-vs-true `u,v` error, per vessel, across all 35. Report as relative L2 and as the AUC
of `-|u_pred|` for clot (i.e. does the *predicted* field retain the discriminative structure).
*Why first:* 16.4 shows geometry alone caps at F1 0.322 versus a 0.60 goal, so the flow field
is required; and every feature, `z_kin` value and training result to date is conditioned on a
field that is either leaked or unmeasured.
*Kill:* if unassisted RGP-DEQ error is small and `-|u_pred|` retains AUC ~0.75, the leak is
cosmetic and the pipeline is sound — enormously reassuring and cheap to establish.
*If it fails:* fixing the flow surrogate becomes the project, and the biochem model is
downstream of it.

**Z2. Decide and document the deployment contract.** Is a clot-free steady CFD solve available
at deploy time, or is it geometry-only? These give different legal feature sets and different
ceilings (0.836 vs 0.767 AUC by 14.5/16.4). Every "leak" argument in this document has been
ambiguous because this was never written down.
*Cost:* zero GPU. *Kill:* n/a — it is a decision, not a hypothesis.

**Z3. Stop training with `flow_feats_source='gt'` while deploying with `auto`/`kinematics`.**
The recipe trains on the true field and deploys on the predicted one. Whatever Z1 finds, this
mismatch is a train/deploy distribution shift sitting under every leg v1-v10.

## Tier A — free, and they change how every number is read (carried from section 15)

**A1. Unify `deploy_clot_score`** — three implementations, disagreeing by up to 0.14 (13.4a,
13.7); the in-training one drives all selection and is the most optimistic. *~0 GPU.*

**A2. Per-vessel commit threshold** — `max Mat` spans 45x against a fixed `1e-4` (14.7).
*~0 GPU.* Re-derive the 14.4/16.3 AUCs and the 13.2 baseline afterwards.

**A3. Delete the six dead channels; normalise `width_d1/d2`** (14.2). Note these feed the
RGP-DEQ, so this interacts with Z1 — do Z1 first, then A3, then re-run Z1.

**A4. Expand 5 -> ~30 training vessels** (14.1). Now better motivated than in section 15:
16.3 shows the 6-vessel cohort produced at least one confidently wrong conclusion (`mu_eff`
bimodality), so small-cohort inference is demonstrably unsafe here.

## Tier B — cheap model experiments

**B1. Ship the deploy-legal logistic regression as a real deploy arm.** Now a *geometry-only*
baseline at AUC 0.767 / F1 0.322 (16.4) rather than section 14.5's leaked 0.836/0.559. This is
the honest floor any new architecture must beat.

**B2. `z_kin` ablation.** Unchanged in method (zero the block, keep `in_dim=287`, sidestepping
the 13.8 warm-start blocker) but **reinterpreted**: since `z_kin` is conditioned on the leaked
CFD field (16.1c), a large drop when ablating would show the model is leaning on leaked flow,
not that the latent is intrinsically valuable. Run it *after* Z1 so the result is readable.

**B3. Longer autoregressive windows.** Unchanged. One clean re-test at `unroll` 25-50 with the
12.6 loss fixes in place, >= 14 epochs.

## Tier C — architecture

**C1. Two-term model: nucleation field + growth propagation.** Still the central proposal and
strengthened by 16.3: 27-58% of commits are nucleation (14.6), and the robust deploy-legal
signal is a smooth geometric field, which is exactly what a nucleation head consumes.
```
dMat = NUCLEATION(field) + GROWTH(local committed Mat) * gate(shear)
```
**Revision from section 15:** drop `mu_eff(t=0)` from the proposed nucleation feature list —
16.3 shows it is anti-predictive on 19/35 vessels. Use `-anaSpd`, `-sdf`, `wgrad`,
`shear_potential`, `curv`, plus whatever flow feature survives Z1.

**C2. Per-vessel conditioning.** Motivation *changes*. Section 15 justified it by the
aneurysm/stenosis split, which 16.3 retires. The surviving justification is 13.7.1's measured
one: vessels need opposite growth *magnitudes* (`041` ~5.6x more mass, `043` ~1.0x) and every
knob is global. Condition on something that predicts required magnitude — positive rate
correlates with `pos%` 1.9-17.6%, which is itself partly predictable from geometry.

**C3. Reframe as ranking.** Unchanged, and now better supported: 16.4 shows AUC 0.767 -> F1
0.322, so the loss between "good ranking" and "good score" is enormous and is where the target
is being lost.

**C4. NEW — recirculation-topology features.** 16.2/16.3 identify recirculation structure as
the strong signal that geometry cannot express (`-u` AUC 0.760 mean, 28/35 above 0.65) and
16.4 shows geometry alone is insufficient without it. If Z1 shows the RGP-DEQ predicts `u`
adequately, derive explicit topological features from the *predicted* field: backflow fraction,
recirculation-zone membership, vortex-core distance, streamline residence time. These are the
physically correct nucleation predictors and none is currently computed.

## Tier D — closed or superseded

* **D1. Analytical Poiseuille prior — REOPENED and partly done.** 16.1 shows the stored priors
  are CFD, not analytic, so a real analytic recompute *is* meaningful. Already measured:
  analytic `-speed` (0.774) beats the leaked version (0.758); analytic `-WSS`/`mu` collapse to
  0.462. Use analytic speed; do not expect analytic WSS to substitute for CFD.
* **D2. `wss_prior_nd`** — still zero, still redundant: computable on the fly. But note the
  *analytic* WSS is near-useless (0.462), so populating it is low value. Real WSS needs Z1.
* **D3. `z_kin` shrink** — gated behind B2, which is gated behind Z1.
* **D4. Regime routing** — closed for this cohort (13.4).
* **D5. Change-B objective reweighting** — closed (12.3); C3 is the replacement.

## Revised first sequence

`Z2` (decide the contract, free) -> `Z1` (measure the surrogate) -> `A1`, `A2` (free, fix the
metric and labels) -> `A4` (expand the cohort) -> `B1` (honest floor) -> `B2` (`z_kin`, now
interpretable) -> `C1` + `C4` as the architectural bet.

**Z1 is the gate.** If the RGP-DEQ cannot supply a usable velocity field on unseen geometry,
then C1/C4 are being built on sand and the correct project is flow prediction, not biochemistry.

# 18. Z ladder — wired and executed (2026-08-07)

## 18.0 What was built

| file | purpose |
|---|---|
| `src/data_gen/lib/legal_priors.py` | prior-source switch (`stored`/`analytic`/`zero`), potential-flow direction solver, train/deploy parity guard |
| `scripts/diag_rgp_deq_flow_audit.py` | Z1 — RGP-DEQ accuracy with and without the leak |
| `scripts/diag_shear_decodability.py` | Z4 — is shear useful, and is it decodable |
| `runtime_config.py: prior_source` | Z3 — `SPECIES_PRIOR_SOURCE`, defaults to `stored` for v1-v10 reproducibility |

**Test status.** `src/tests/` runs 538 passed / 8 failed. All 8 are in
`test_species_flow_feats.py` and are **pre-existing**: the file passes 22/22 in isolation, and
the same 8 fail with this session's changes stashed. It is a cross-test state leak in
full-suite runs (config context bleeding between tests), not a regression from the Z ladder.
Worth fixing separately — a suite that fails only in aggregate hides real regressions.

**Flow direction.** The analytic prior needs a streamwise direction. `shear_potential` (x[:,3])
is *not* one — inlet and outlet both average 0.51 and its gradient is uncorrelated with velocity
(mean cos +0.01). Replaced with a proper potential-flow solve: graph Laplace, `phi=1` inlet /
`phi=0` outlet, direction `= -grad phi`. **Jacobi is unusable here** — these meshes run ~274 hops
inlet-to-outlet so Jacobi needs O(diameter^2) ~ 75k sweeps; at 600 sweeps the direction field
scored cos +0.19. Conjugate gradient on the free block converges in **0.4 s** and scores
**cos +0.742** across all 43 packs (range 0.721-0.757).

## 18.1 Z2 — deployment contract, DECIDED

**At deploy we receive geometry + initial and boundary conditions only. No clot-free CFD solve.**

This settles every ambiguous leak argument in sections 14 and 16. The stored
`u_prior`/`v_prior`/`mu_prior` are the converged CFD field (s16.1) and are therefore **not legal
inputs**. `prior_source='analytic'` is the only deployable setting; `'stored'` is retained purely
to reproduce legs v1-v10.

## 18.2 Z1 RESULT — the leak costs field accuracy but almost NO discriminative power

All 35 usable vessels, RGP-DEQ re-solved under each prior condition, compared against GT `y[0]`:

| prior source | rel L2 (u) | cos(pred, GT) | **AUC(-\|u_pred\|)** | within 0.05 of GT |
|---|---|---|---|---|
| `stored` (leaked) | 0.116 | +0.996 | 0.782 | 34/35 |
| **`analytic` (legal)** | 0.566 | +0.773 | **0.763** | 28/35 |
| `zero` (ablation) | 0.984 | +0.236 | 0.748 | 23/35 |
| — GT field — | — | — | *0.789* | — |

**Removing the leak costs 4.9x the field error (relL2 0.116 -> 0.566) but only 0.019 of clot
AUC (0.782 -> 0.763), against a GT ceiling of 0.789.** The legal path retains **97%** of the
ground-truth field's discriminative power.

This is the distinction the audit was built to isolate: *field accuracy* and *usefulness to the
clot model* are different quantities. The surrogate becomes numerically mediocre without the
answer, and stays almost exactly as useful.

**Z1 VERDICT: PASS. Flow prediction is not the blocking project.** Section 16.4's conclusion —
"the RGP-DEQ is the binding constraint, and if it fails the correct project is flow prediction"
— is **retired**. C1/C4 can be built on the legal path.

**Two caveats worth keeping.**
1. `zero` still scores 0.748. The gap from `analytic` to `zero` is only 0.015, so the analytic
   priors add little over nothing — the DEQ recovers clot-relevant structure mostly from
   geometry either way. Do not over-credit the analytic priors.
2. AUC 0.763 on the legal path is essentially the same as section 16.4's deploy-legal logistic
   regression (0.767), which mapped to best-F1 **0.322**. So Z1 passing does *not* mean the
   0.60 goal is now reachable — it means the remaining gap is **not** in flow prediction. It is
   in nucleation modelling, temporal dynamics and calibration (s14.6, s16.4).

## 18.3 Z3 — train/deploy parity, wired

`prior_source` is now a first-class runtime field with `assert_train_deploy_prior_parity()`,
which raises when a leg trains on a prior block it will not have at deploy. v1-v10 all trained
with `flow_feats_source='gt'` and the leaked prior block, then deployed against a predicted
field — a distribution shift that sat under every result in sections 9-13 and was never checked.

Default remains `stored` so historical legs stay bit-reproducible; **every new leg must set
`analytic`**, and re-baseline against section 13.2 rather than comparing across the change.

## 18.4 Z4 RESULT — shear is NOT worth decoding for the clot model

Two questions, and the first gates the second.

**Q1: does the TRUE shear field add anything over deploy-legal geometry?**

| vessel | geometry only | geometry + GT shear | gain | shear alone |
|---|---|---|---|---|
| `039` | 0.897 | 0.888 | −0.009 | 0.830 |
| `040` | 0.890 | 0.889 | −0.001 | 0.787 |
| `041` | 0.680 | 0.686 | +0.006 | 0.634 |
| `042` | 0.680 | 0.679 | −0.000 | 0.634 |
| `043` | 0.877 | 0.884 | +0.007 | 0.815 |
| `044` | 0.667 | 0.660 | −0.007 | 0.655 |
| **mean** | **0.782** | **0.781** | **−0.001** | 0.726 |

**Adding the ground-truth shear field to `[-anaSpd, -sdf, wgrad]` changes clot AUC by −0.001.**
Shear alone is a decent predictor (0.726), but it is *redundant* with geometry — the geometric
features already encode it. This is consistent with s16.3, where `mu_eff` (a pure function of
shear) averaged 0.482 and was anti-predictive on 19/35 vessels.

**A shear decoder head cannot improve the clot model, however accurate it becomes.** That is
not a statement about the decoder's quality; it is a statement about the information already
present in geometry.

**Q2: decodability, for completeness.** Analytic shear (geometry-only) correlates with GT shear
at **r = 0.742** mean (0.606-0.865), rel L2 0.34-0.77. Any decoder must beat that baseline —
`scripts/diag_shear_decodability.py --kine-ckpt <path>` is wired to make that comparison once
the shear-head model exists.

**Recommendation:** the shear head may still be worth having for flow-model quality or for other
downstream tasks, but **do not sequence the clot work behind it** — Q1 says the ceiling it
could buy is zero.

## 18.5 Revised priorities after the Z ladder

The Z ladder was supposed to gate everything else. It has now run, and it *reopens* the
architecture work rather than blocking it:

1. **~~Z1 fix the flow surrogate~~ — not needed.** Legal path retains 97% of GT discriminative
   power (18.2).
2. **~~Z4 shear decoder~~ — will not help the clot model.** (18.4)
3. **Set `prior_source='analytic'` on all new legs and re-baseline** (18.3). This is now the
   precondition, and it is cheap.
4. **A1 unify `deploy_clot_score`; A2 per-vessel commit threshold** — unchanged, still free,
   still block trustworthy measurement.
5. **A4 expand 5 -> ~30 vessels** — unchanged.
6. **C1 nucleation + growth two-term model** — now the clear main bet. 18.2's caveat 2 localises
   the remaining gap precisely: not flow, not shear, not features. Nucleation (27-58% of commits,
   s14.6), temporal dynamics, and the AUC 0.77 -> F1 0.32 calibration collapse (s16.4).
7. **C4 recirculation-topology features** — demoted. 18.2 shows the DEQ already recovers most
   clot-relevant flow structure, and 18.4 shows shear-derived quantities are redundant with
   geometry. Only worth revisiting if C1 stalls.

# 19. Post-Z re-plan (2026-08-07)

Z1 and Z4 both **clear** rather than block, so the architecture work is reopened. But the Z1
table plus three cheap decompositions run afterwards change *where* the remaining gap is, and
retire another of this document's claims.

## 19.1 The Z1 table's most important row is the one nobody is reading

```
prior source        rel L2(u)   AUC(-|u_pred|)
GT field                 —          0.789
stored (leaked)       0.116         0.782
analytic (legal)      0.566         0.763
zero (ablation)       0.984         0.748
```

Going from the true field to **no flow information at all** costs **0.041 AUC**. Flow
contributes ~2-5% of discriminative power for clot. Two consequences:

1. **C4 (recirculation topology) is correctly demoted** — agreed.
2. **B2 (`z_kin` ablation) should be promoted to the front, not left mid-list.** `z_kin` is the
   DEQ latent — a *flow* representation — and it occupies **256 of 287 input dimensions (89%)**.
   Z1 says the entire flow channel is worth ~0.04 AUC. Either `z_kin` is carrying something
   other than flow, or 89% of the input is near-dead weight. That is now a 1-leg question with
   a large payoff either way, and it makes every later experiment cheaper if it comes back dead.

## 19.2 RETRACTION of 14.5 — the linear model does NOT match the GNN

14.5 claimed "a linear model matches the GNN" from leaked features on the 6-vessel cohort with
an oracle threshold. Redone properly — deploy-legal features, logreg trained on the 29
non-cohort vessels, tested on the same 6 vessels the GNN was scored on:

| vessel | logreg F1 | GNN F1 | GNN gain |
|---|---|---|---|
| `039` | 0.781 | 0.518 | **−0.263** |
| `040` | 0.550 | 0.704 | +0.154 |
| `041` | 0.354 | 0.255 | −0.099 |
| `042` | 0.344 | 0.513 | +0.169 |
| `043` | 0.639 | 0.650 | +0.011 |
| `044` | 0.427 | 0.602 | +0.174 |
| **mean** | **0.516** | **0.540** | **+0.024** |

The GNN wins by +0.024 on average while being *handicapped* (fixed gate vs the logreg's oracle
threshold) — and the per-vessel spread is enormous (−0.263 to +0.174). So the GNN does add
real structure on 4 of 6 vessels; 14.5's dismissal was an artifact of leaked features plus a
favourable cohort. **The rollout is earning its place; it is the calibration across vessels
that is not.**

## 19.3 The AUC -> F1 collapse decomposed: it is a RANKING problem, not calibration or coherence

9 held-out vessels, deploy-legal logreg, three interventions:

```
F1 @ oracle threshold, no smoothing        0.322     <- 16.4's number
F1 @ oracle COUNT of positives             0.198     <- calibration is NOT the gap
F1 @ 3x graph smoothing                    0.311     <- spatial coherence is NOT the gap
F1 @ smoothing + oracle count              0.199
```

Both "obvious" fixes make it **worse or do nothing**:

* **Oracle count is worse than an oracle threshold (0.198 vs 0.322).** At AUC ~0.78 with 2-18%
  positives, the F1-optimal operating point commits *more* nodes than truly clot — over-commit
  buys recall faster than it loses precision. **This partially exonerates the over-painting
  that §9-§13 spent ten legs fighting: mass ratio > 1 is F1-rational at this ranking quality.**
  Chasing mass 1.0 was optimising the wrong thing.
* **Graph smoothing does nothing (0.311 vs 0.322).** Whatever the GNN adds in 19.2, it is not
  spatial smoothing.

With calibration and coherence both eliminated, **the binding constraint is ranking quality**,
and every path measured caps at AUC 0.75-0.79: geometry alone 0.748, +analytic 0.763,
+leaked CFD 0.782, GT field 0.789, deploy-legal logreg 0.767.

## 19.4 The goal is stated on an easy subset

Same logreg, same features, different test sets:

```
tested on 9 random held-out vessels    mean F1 0.322
tested on the 039-044 cohort           mean F1 0.516
```

**The stenosis/aneurysm cohort is substantially easier than a random draw from the 35.** A
result of ">0.6 on 039-044" therefore does not imply ">0.6 generalizing to unseen vessels",
which is the actual goal. Any target should be re-stated on a random held-out split, or the
cohort result read as an upper bound.

## 19.5 Revised next steps

Where I agree with the standing plan: set `analytic` on new legs, unify `deploy_clot_score`
(A1), per-vessel commit threshold (A2), expand to ~30 vessels (A4), demote C4, and make C1 the
main bet. Four changes:

**(i) Promote B2 (`z_kin` ablation) to run first among the model experiments.** Rationale in
19.1 — Z1 has made it a high-information, single-leg question, and a dead result shrinks the
input by 89% and speeds up everything after it.

**(ii) Re-target C1's nucleation head at EARLY commits, not the final map.** 14.6 showed 27-58%
of commits are nucleation; 12.8/§14 showed oracle `t=20` seeds dilated by k hops already reach
F1 0.52-0.64 — i.e. *seeds plus growth is most of the answer*. Training the nucleation head on
the final map dilutes it across 200 steps of consequences. Train it on "does this node commit
in the first ~20 steps", let the growth term propagate, and grade the composite.
*First cheap check before building anything:* measure AUC of the deploy-legal features for the
**t=20 seed set** versus the final map. If seeds are more predictable, C1's design follows
directly; if not, the nucleation head has no better target than what already exists.

**(iii) Stop optimising mass ratio toward 1.0.** 19.3 shows over-commit is F1-rational at
achievable ranking quality. `select_mass_hard_max=1.5` and the §9.11 brake were fighting the
metric, not helping it. Re-derive the mass target *from* the F1-optimal operating point at the
model's measured AUC rather than from physical mass matching.

**(iv) Re-state the goal on a random held-out split** (19.4), or explicitly accept 039-044 as a
development set and hold a separate random set for the real generalization claim.

**What I would not do next:** more work on flow (Z1), shear decoding (Z4), recirculation
topology (C4), routing (13.4), or objective reweighting (12.3). All five are now closed by
measurement.

# 20. Phase 0 — measurement foundation (2026-08-07)

No GPU. Everything here changes how later numbers are read, so it precedes the Phase 1
re-baseline. Two of this document's own diagnoses are corrected below.

## 20.0 (0a) Test-suite cross-test state leak — FIXED

8 tests in `test_species_flow_feats.py` failed in aggregate runs and passed in isolation.

**Root cause.** `src/biochem_gnn/config.py::_bind_typed_configs` binds PushforwardConfig /
BiochemRuntimeConfig **process-wide** via module-global contextvar tokens — deliberate, so
deploy and eval scripts keep one active config for the process. Under pytest that persists
across tests. Any test calling `apply_train_recipe_env` / `apply_mat_growth_leg_env`
(`test_mat_growth_simple_scope`, `test_runtime_config`, `test_seed_aux_loss` — all sorting
*before* `test_species_flow_feats`) leaves a config bound, and every `*_enabled()` helper checks
`resolve_config()` **before** falling back to `os.environ`. So later tests that set env vars
were silently ignored.

**Fix.** Autouse isolation fixture in `src/tests/conftest.py` restoring the binding after each
test. Production behaviour untouched.

```
before:  538 passed, 8 failed
after:   546 passed, 0 failed   (order-independent; file still 22/22 in isolation)
```

## 20.1 (0b) `deploy_clot_score` unified — and 13.4a's diagnosis was wrong twice

13.4a attributed the cross-tool score discrepancy to `clout_prec_rec_floor` being 0.30 / 0.35 /
0.30 across three files. **Both parts of that are wrong:**

1. **The floor is inert in every run we have.** `relaxed_prec_floor_score` returns plain
   precision whenever `recall >= floor`; every logged run had relaxed recall ~1.000, so the
   floor could not have contributed anything.
2. **The typed values agree.** `build_train_recipe_configs()` resolves
   `clout_prec_rec_floor = 0.30`, identical to the default runtime. The literal `"0.35"` at
   `mat_growth_simple.py:50` never reaches the typed config.

**The actual cause: three different scoring entry points with different protocols.**

| caller | function | protocol |
|---|---|---|
| `eval_mat_growth_simple.py` | `canonical_deploy_clot_metrics` | `bind_canonical_deploy_protocol` |
| `diag_regime_gate_sweep.py` | `grade_deploy_clot_series` (raw) | **none** |
| in-training | mean over `clf_by_t` sliding fracs | selection signal (9.8) |

The sweep graded the *same predictions* without the canonical deploy protocol. That moves
`deploy_clot_score` while leaving strict `deploy_clot_f1` bit-identical — exactly the observed
pattern (F1 identical on 4/6 vessels, score different on all 6).

**Fix.**
* New `canonical_grade_series()` in `src/evaluation/canonical_clot_eval.py` — same protocol as
  `canonical_deploy_clot_metrics`, for callers that roll out once and grade many times (which
  is the right design; the rollout dominates cost).
* `diag_regime_gate_sweep.py` now calls it instead of the raw grader.
* New `scoring_fingerprint()` in `clot_relaxed_metrics.py`, printed by both tools, exposing the
  resolved constants **and whether a runtime is bound** — the actual failure mode.
* `src/tests/test_canonical_scoring_parity.py` (6 tests) pins: both canonical entry points bind
  the protocol; the sweep never calls the raw grader; the floor is inert at full recall.

**The in-training sliding-window mean is left alone** — it is a *selection* signal by design
(9.8), not a reported score. The contract is now: reported scores come from a canonical entry
point; training-time aggregation is a separate, clearly-labelled quantity.

## 20.2 (0e) The F1-optimal commit ratio is 3.0x, not 1.0x — this reframes sections 9-13

For each held-out vessel, sweeping the number of committed nodes to maximise F1 at achievable
ranking quality:

| vessel | n_true | optimal commit | ratio | F1@opt | F1@ratio 1.0 |
|---|---|---|---|---|---|
| `001` | 290 | 871 | 3.00 | 0.431 | 0.286 |
| `005` | 144 | 821 | 5.70 | 0.191 | 0.118 |
| `009` | 141 | 67 | 0.48 | 0.433 | 0.383 |
| `013` | 383 | 703 | 1.84 | 0.613 | 0.470 |
| `018` | 41 | 233 | 5.68 | 0.080 | 0.000 |
| `024` | 156 | 538 | 3.45 | 0.429 | 0.160 |
| `031` | 45 | 7 | 0.16 | 0.077 | 0.044 |
| `037` | 153 | 648 | 4.24 | 0.300 | 0.078 |
| `042` | 243 | 685 | 2.82 | 0.358 | 0.239 |
| **mean** | | | **3.04** | | |

**Mean 3.04x, median 3.00x.** Committing at the true count instead costs roughly 0.15 F1 on
average. At AUC ~0.78 with 2-18% positives, over-commit buys recall faster than it loses
precision.

Now set that against every best epoch this project ever produced, under
`select_mass_hard_max = 1.5`:

```
leg/epoch          mass    score    gate     |mass - 3.04|
v2 ep5            2.077   0.5011  REJECTED       0.96
v3 ep5            2.768   0.4998  REJECTED       0.27
v6 ep3            3.200   0.4414  REJECTED       0.16
v10 ep12          1.674   0.6221  REJECTED       1.37
v10 ep14          2.021   0.5860  REJECTED       1.02
saturated basin   4.030   0.2620  REJECTED       0.99
```

**Every one of them was rejected, and the three best sit within 0.27 of the empirically optimal
ratio.** The legs were finding the right operating point and the selection gate was discarding
it. Together with 12.4 (retention discarded them too) this is the mechanical explanation for
sections 9-13: **the search was working; the acceptance criteria were wrong.** The 9.11 brake,
`final_mass_target=1.2` and `select_mass_hard_max=1.5` were all pulling toward physical mass
matching, which is *not* the F1 optimum.

**Wired.** `_SUBCOHORT_RUNTIME_V11PLUS` — `select_mass_hard_min 1.2`, `hard_max 4.5`,
`soft_target 3.0`, `soft_lambda 0.15`. `_SUBCOHORT_RUNTIME_V3PLUS` is untouched so v3-v10 stay
reproducible.

## 20.3 (0d) Per-vessel commit threshold — MEASURED, adoption recommended

Label definition swept across all 35 vessels (`-sdf` used as a fixed probe feature):

| threshold | pos% mean | pos% sd | pos% range | AUC mean | **AUC sd** |
|---|---|---|---|---|---|
| fixed `1e-4` (current) | 8.12 | 4.15 | 1.9-17.6 | 0.768 | **0.102** |
| relative 1% of max Mat | 15.38 | 10.35 | 4.1-42.0 | 0.629 | 0.066 |
| relative 5% of max Mat | 10.92 | 7.06 | 2.7-27.2 | 0.708 | 0.054 |
| **relative 10% of max Mat** | 7.69 | 5.11 | 2.1-21.0 | **0.800** | **0.058** |

Relative-10% gives both a **higher** mean AUC (0.800 vs 0.768) and **43% lower cross-vessel
variance** (sd 0.058 vs 0.102). The same features work equally well on every vessel, which is
precisely the generalization property being chased.

**Recommended but not yet applied**, because adopting it invalidates numeric comparability with
every result in sections 9-19. It should be switched on exactly once, at the Phase 1
re-baseline, never mid-ladder.

## 20.4 (0f) Seeds are far more predictable than the final map

Same deploy-legal features, two targets, 35 vessels:

```
target        n     mean best-feature AUC     sd
final map    35            0.806             0.085
t=20 seeds   32            0.903             0.032
```

**Seeds are both more predictable (+0.097 AUC) and 2.7x more consistent across vessels.** The
`t=0` physics says where clot *starts* much better than where it *ends up*, which is what a
nucleation term should model; the final map is that signal plus 200 steps of propagation.

**This settles C1's design (19.5 item ii): train the nucleation head on early commits, and let
the growth term carry it forward.** Combined with 14.6 (27-58% of commits are nucleation) and
the oracle-seed dilation ceiling of F1 0.52-0.64, the two-term split now has direct empirical
support rather than being an argument from mechanism.

## 20.5 (0c) Goal split — decision required

Not a measurement. Recorded for an explicit decision at Phase 1:

```
same model, same features, different test sets:
  9 random held-out vessels   mean F1 0.322
  the 039-044 cohort          mean F1 0.516
```

`039`-`044` is materially easier than a random draw from the 35. **Recommendation:** treat
`039`-`044` as a development set and hold a random 8-10 vessel set, untouched, for the
generalization claim. Otherwise ">0.6 on the cohort" will not mean ">0.6 on unseen vessels".

## 20.6 Phase 0 status

| item | status |
|---|---|
| 0a test isolation | **done** — 546 passed, 0 failed, order-independent |
| 0b unify `deploy_clot_score` | **done** — canonical entry point + fingerprint + 6 tests |
| 0c goal split | **decision pending** (20.5) |
| 0d per-vessel threshold | **measured**, adoption recommended at Phase 1 (20.3) |
| 0e mass target | **done** — 3.04x measured, `_SUBCOHORT_RUNTIME_V11PLUS` wired |
| 0f seed target | **measured** — settles C1's head target (20.4) |

Suite: **552 passed, 0 failed.**

# 21. Phase 0 decisions wired (2026-08-07)

Both open decisions from 20.5 / 20.3 resolved by the user and implemented.

## 21.1 (0c) Cohort split v2 — DECIDED and sealed

`039`-`044` stays the **dev** cohort, with one stenosis and one aneurysm held back from it,
plus a broader sealed set chosen to make the generalization claim meaningful.

**Geometry class is now computable.** Width-profile skew recovers the known labels exactly:
aneurysms `039/040/043` = **+1.05 / +0.69 / +0.65**, stenoses `041/042/044` =
**-0.58 / -0.57 / -0.51**. Applied to all 35: 7 aneurysm-like, 28 stenosis-like.

**Holdout construction rules, applied programmatically, not by hand:**
* every holdout vessel is **interior** on all four descriptor axes (pos%, best-feature AUC,
  nucleation%, skew) — never a global min or max, so the sealed set tests *interpolation*;
* `T >= 150`, so no severely truncated simulation sits in the holdout;
* class balance proportional to the pool (2 aneurysm / 6 stenosis vs the pool's 7/28);
* `042` (median stenosis) and `043` (the long-sealed aneurysm) are the two dev holdouts.

A first, greedy attempt was rejected: it put `patient003` in the holdout, which is the global
maximum on *both* skew (1.94) and nucleation (82.0) and has `T=29`, and it left only 4
aneurysms to train on.

```
SEALED GENERALIZATION (8)   001, 007, 010, 013, 014, 031, 042, 043
                            2 aneurysm / 6 stenosis, all T>=155
TRAIN (27)                  002,003,004,005,006,008,009,011,012,015,016,018,019,020,021,
                            024,025,028,029,032,035,036,037,039,040,041,044
                            5 aneurysm / 22 stenosis
DEV                         039-044   (dev-train 039,040,041,044 | dev-holdout 042,043)
```

**Coverage:** train spans the holdout's range on pos%, nucleation% and skew. The one exception
is the AUC upper bound — holdout 0.944 (`043`) vs train 0.941 (`040`), a **0.003** tie, which is
immaterial and is the price of keeping `043` sealed for continuity with sections 9-20.

Wired as `WALL_COHORT_V2_TRAIN` / `WALL_COHORT_V2_GENERALIZATION` / `WALL_COHORT_V2_DEV` /
`WALL_COHORT_V2_DEV_HOLDOUT` in `mat_growth_simple.py`, with disjointness asserted.

**Standing rule:** the 8 sealed vessels are spent **once**. Do not tune against them, and do not
report a generalization number that was iterated on.

## 21.2 (0d) Per-vessel commit threshold — ADOPTED

User decision: robustness going forward outweighs backward comparability. Adopted at the
measured optimum, relative **10% of each vessel's max Mat** (20.3: AUC 0.768 -> 0.800,
cross-vessel AUC sd 0.102 -> 0.058).

**A leak boundary this exposed, and how it is handled.** A 10%-of-max threshold is computed
from GT. That is fine for *labels* — wherever labels exist, GT exists — but it would be a
**deploy leak** if used for the model's own predicted commit readout, which must not consult
`max(GT Mat)`. The implementation therefore splits the two roles that were previously one
global scalar:

| role | function | may be vessel-relative? |
|---|---|---|
| what counts as TRULY committed (labels, metrics, GT side of losses) | `mat_label_thresh()` | **yes** |
| what the MODEL predicts as committed | `continuous_mat_commit_thresh()` | **no — leak** |

Deploy-time selection was already percentile-based (`CLOT_POCKET_GATE_PCT`), so the prediction
side needs nothing vessel-specific.

**Implementation.** `mat_label_thresh_mode` (`"absolute"` | `"rel_max"`) and
`mat_label_rel_frac` (0.10) on `PushforwardConfig`; a `_VESSEL_MAT_MAX` contextvar with
`use_vessel_mat_max()`; the GT side of `rolled_final_mass_fp_penalty` and `rolled_soft_f1_loss`
switched to `mat_label_thresh()`. Defaults to `"absolute"`, and `rel_max` falls back to absolute
when no vessel is bound, so nothing silently changes for callers that have not been taught about
vessels.

**Measured effect** (rel_max, 10%):
```
vessel      max Mat    abs thr   rel thr   n(abs)  n(rel)
024        3.40e-04    1.0e-04   3.4e-05     156     305     low-peak  -> MORE positives
043        2.17e-03    1.0e-04   2.2e-04     167     109
041        4.31e-03    1.0e-04   4.3e-04     262     138
012        1.51e-02    1.0e-04   1.5e-03     254      73     high-peak -> FEWER positives
```
Exactly the intended equalisation.

**Consequence, stated plainly:** every count, F1 and score in sections 9-20 was computed under
the absolute threshold and is **not comparable** with anything measured after `rel_max` is
switched on. The switch happens once, at the Phase 1 re-baseline. Three tests pin the
behaviour, including that the prediction-side threshold never consults the vessel max.

## 21.3 Phase 0 complete

| item | status |
|---|---|
| 0a test isolation | **done** — 546 passed / 0 failed, order-independent |
| 0b unify `deploy_clot_score` | **done** — `canonical_grade_series` + fingerprint + 6 tests |
| 0c goal split | **done** — cohort split v2 sealed (21.1) |
| 0d per-vessel threshold | **done** — `rel_max` @ 10% wired, leak-scoped (21.2) |
| 0e mass target | **done** — 3.04x measured, `_SUBCOHORT_RUNTIME_V11PLUS` |
| 0f seed target | **done** — seeds AUC 0.903 vs final 0.806; settles C1 |

**Suite: 555 passed, 0 failed.**

**Phase 1 is now unblocked.** It must switch on, in one re-baseline leg and never mid-ladder:
cohort v2 train set (27), `prior_source=analytic`, `mat_label_thresh_mode=rel_max`,
`_SUBCOHORT_RUNTIME_V11PLUS` selection, >= 14 epochs. That leg is a *reference point*, not an
A/B — nothing is attributable across it, and every later leg is single-variable against it.

# 22. Phase 1 wiring (2026-08-07)

Two per-vessel properties had to be established once at pack entry and honoured everywhere
downstream. They are bundled into one primitive rather than threaded separately, because
applying one without the other is a silent correctness bug and this project's dominant failure
mode is exactly that (12.3 v4/v5, 20.0, 20.1).

## 22.1 `src/core_physics/vessel_scope.py`

```
prior_source_cache_tag(src)   -> ""  for stored, "_prior-<src>" otherwise
resolve_vessel_mat_max(data)  -> peak GT Mat, or None when unavailable
prepare_vessel_data(data)     -> (data with configured priors, mat_max)
```

Three traps this closes, each of which would have produced a confident no-op:

1. **Priors must be applied BEFORE the kinematics solve.** The RGP-DEQ consumes
   `UV_PRIOR`/`MU_PRIOR` (16.1c), so rewriting them after the solve leaves `z_kin` conditioned
   on the leaked CFD field while the config claims `analytic`. Wired at the load site, ahead of
   `predict_kinematics_and_latent`, and pinned by a source-order test.
2. **The pack cache key must include the prior source.** `pack_cache_dir` was keyed on the
   feature stack only, and the DEQ latent is baked into the cached pack — so an `analytic` run
   would have silently reused `stored` packs and preserved the leak intact. `stored` maps to an
   empty tag so every pre-existing cache stays valid.
3. **`mat_max` travels with the pack**, so no loop can forget to bind it. `None` (not `0.0`)
   on a clot-free pack, so `mat_label_thresh()` falls back to absolute rather than collapsing
   to zero and labelling every node committed.

## 22.2 Binding sites

`use_vessel_mat_max` is now entered at all three places a label threshold is consumed, each as
a real `with` block (no manual `__enter__`, which is the leak-prone pattern the trainer already
uses for typed configs):

| site | scope |
|---|---|
| main training loop | `pack["mat_max"]` |
| deploy-horizon aux loss | `vpack["mat_max"]` |
| reported deploy metrics / selection | `val_pack["mat_max"]` |

`eval_mat_growth_simple.py` uses the same primitive, so a reported score and the loss that
produced it grade against the same definition of "committed".

## 22.3 A real bug the wiring surfaced

`resolve_vessel_mat_max` first indexed `y` with `MAT_CHANNEL`. **`MAT_CHANNEL` is 11, an index
within the species block, not a column of `y`** — `SPECIES_BLOCK = slice(4, 16)`, so the Mat
column is `4 + 11 = 15`. `y[:, :, 11]` is `FG_log1p_nd`, whose peak is ~0.69 against Mat's
~4.3e-3, i.e. **~160x too large**. Every relative label threshold would have been inflated by
that factor and essentially nothing would have been labelled committed.

Caught by checking the resolved value against a directly-measured one (`patient041` returned
6.94e-01 where 4.31e-3 was expected), not by the test — the original test seeded and read using
the *same* wrong index, so it passed against a broken function. It now asserts the column
arithmetic explicitly, plants a decoy in the FG column, and cross-checks a real pack.

## 22.4 `WG_phase1_baseline`

The re-baseline leg. **Not an A/B**: it switches the whole Phase-0 foundation on at once and
becomes the reference every later single-variable leg is measured against. Nothing is
attributable across it, and no number before it is comparable with any number after it.

```
cohort      5 ad-hoc vessels -> WALL_COHORT_V2_TRAIN (27), 8 sealed      (21.1)
priors      stored (leaked CFD) -> analytic                              (16.1, 17 Z2)
labels      fixed 1e-4 -> rel_max @ 10% of each vessel's peak Mat        (20.3, 21.2)
selection   mass window [0.5,1.5] -> [1.2,4.5], target 3.0               (20.2)
objective   soft-F_beta surrogate live, occupancy scale fixed            (12.6)
```
Backbone stays frozen and the growth law stays generic — change D / change E attribution comes
after this, one variable at a time.

**Validation anchor.** The launcher's historical default is `patient043`, which is now
**sealed**. Using it for epoch selection would spend the seal on the very first leg. Phase 1
uses `patient041` instead: dev-cohort, not sealed, and the hardest vessel on record, so
selection is not flattered. Effective split is 26 train / 1 val / 8 sealed.

## 22.5 Guards

`src/tests/test_vessel_scope_and_phase1.py` (16 tests). Each corresponds to a specific way the
wiring could become a silent no-op: prior-vs-solve ordering, cache-key separation, the Mat
column mapping, all three binding sites present, the eval script using the same scope, the leg
resolving the full foundation, cohort disjointness, historical legs unchanged, and the
label/prediction threshold separation (the prediction side must never consult the vessel max —
that would be a deploy leak).

**Suite: 571 passed, 0 failed.**

# 23. Phase 1 result and the Phase 2 decision (2026-08-08)

`WG_phase1_baseline`, 10 epochs, 25 train vessels, val `patient041`, ~17 min/epoch (2.9 h).
The re-baseline. Nothing here is comparable with sections 9-20; everything after is measured
against this.

## 23.1 The result

```
ep    loss      score      f1     mass    fp   fn   selection
 1   76.2303   0.4104   0.5017   3.062   242    9   accepted
 2   76.2502   0.3706   0.4674   2.708   202    9   accepted
 3   76.2543   0.3757   0.4730   2.699   201    9   accepted
 4   76.2574   0.4593   0.5419   2.363   165   11   accepted
 5   76.1935   0.4350   0.4006   0.708    31   64   MASS-REJECT (under)
 6   76.1989   0.4889   0.5505   2.062   131   11   accepted   <- BEST
 7   76.1921   0.3137   0.2494   0.522    25   79   MASS-REJECT (under)
 8   76.1915   0.4076   0.3668   0.788    38   62   MASS-REJECT (under)
 9   76.1356   0.2899   0.2275   0.434    27   91   MASS-REJECT (under)
10   76.1731   0.3553   0.3389   0.956    57   62   MASS-REJECT (under)
```

**Three things changed qualitatively, and all three are Phase-0 effects.**

1. **The `fp≈292` attractor is gone.** 10 epochs visited **10 distinct fp states** (25-242).
   Every leg v3-v10 visited **2**. The dynamic range that 12.3 said was missing now exists.
2. **Four epochs passed selection**, against **zero** across v2-v10 combined.
3. **The failure mode inverted.** Every historical rejection was for *over*-mass (~4.03);
   every Phase-1 rejection is for *under*-mass (0.43-0.96). The model now collapses toward
   under-painting instead of saturating.

Best epoch: **score 0.4889, F1 0.5505 at mass 2.062**, against a v2-v10 best-ever in-training
score of 0.5011 — but that one was a lucky single-epoch excursion in a leg that produced no
checkpoint, whereas this is the selected state of a leg that produced four valid ones.

## 23.2 The alignment problem is unchanged — and for the first time, trustworthy

```
Spearman(loss, deploy_clot_score) = +0.564
jackknife range                   = [+0.40, +0.67]   sign STABLE
z-separation of the best epoch    = -0.25            (|z| < 0.5 -> loss cannot see it)
distinct deploy states            = 10               (v3-v10: 2)
total loss spread                 = 0.16%
```

12.3 had to discard v6's `-0.406` because it rested on 2 distinct states and flipped sign when
one epoch was dropped. **This one does not.** Ten distinct states, a jackknife that never
crosses zero, and a positive sign: **lower loss reliably means worse deploy score.**

Loss moved 76.2303 -> 76.1356 (monotone-ish down) while score went 0.4104 -> 0.2899. The
optimizer is doing its job; the objective is pointing the wrong way.

This is the first *measurable* statement of the problem in the project's history. Everything
before it was either underpowered (12.8.2) or measured against a dead objective (12.6).

## 23.3 A stale constant of my own: 20.2's 3.04x is the wrong target

20.2 measured the F1-optimal commit ratio as 3.04x n_true and I set `final_mass_target = 3.0`
from it. That number came from a **logistic regression** optimising **F1**. The guiding metric
is `deploy_clot_score`, which is *relaxed precision* gated by a recall floor -- a different
functional with a different optimum. Phase 1 measures the real one:

```
score-ranked epochs:   0.4889 @ mass 2.06   0.4593 @ 2.36   0.4350 @ 0.71
                       0.4104 @ 3.06        0.4076 @ 0.79   0.3757 @ 2.70
```

The score optimum sits at **mass ~2.0-2.4**, not 3.04. The selection window [1.2, 4.5] still
brackets it, so selection is fine, but `final_mass_target = 3.0` pushes training past the
optimum. Same class of error as every other stale constant in this document -- a number
measured for one purpose reused for another -- and it is mine, from two sessions ago.

## 23.4 Learning curve: declining, so more epochs are not the answer

```
best epoch 6 of 10   overall slope -0.0078/ep   tail slope -0.0262/ep   verdict: declining
```

Contrast v10, which the same diagnostic scores `EXTEND` (best epoch 12/14, tail **+0.0497**).
Phase 1 peaks in the middle and degrades. **Spending the budget on more epochs would buy
nothing**; the run is not data-starved, it is being actively pushed the wrong way after epoch 6.

This retires the 14-epoch floor for good and confirms the ROI position: **8-10 epochs is the
right screening unit at 17 min/epoch**, with the learning-curve verdict as the stop/extend rule.

## 23.5 Why the mass collapse is the alignment problem wearing a different hat

The loss weights were tuned in an era when the rolled-state terms were numerically dead
(12.6.1). `fp_weight = 6.0` against `underpred_weight = 3.0` is 2:1 in favour of suppression,
which was harmless while the terms carrying it could not respond to the rollout. Phase 0 fixed
the occupancy scale, so those terms now have real gradient -- and the pre-existing 2:1
suppression bias is expressing itself for the first time, driving mass from 3.06 down through
the optimum to 0.43.

So the over-painting that consumed sections 9-13 and the under-painting in Phase 1 are the same
defect seen from two sides: **nothing in the objective is anchored to the operating point that
maximises the metric.**

## 23.6 Phase 2 decision

The user asked whether it is worth spending time linking the training loss to the guiding
score. **Yes, and now is the moment**, for a reason that did not hold before: the measurement is
finally trustworthy (23.2). Building C1 (nucleation + growth) on top of an objective with
`rho = +0.564` would optimise the new architecture in the wrong direction and produce another
uninterpretable leg -- v1-v10's core mistake, repeated with more machinery.

**But "fix the objective" is not actionable at the level of a total.** The loss has ~8 terms.
12.3 already burned four legs (v4, v5, v6, v9) guessing which one mattered. So Phase 2 starts
with a decomposition rather than a fix:

**`WG_phase2a_decomp`** -- training config **byte-identical** to `WG_phase1_baseline`
(verified: config diff is empty). The only change is instrumentation: `record_loss_term()`
accumulates each term's contribution per epoch into `lossterm_*` fields in `train_log.jsonl`,
detached and never in the optimizer graph. Terms tracked: `per_step_block` (the growth Huber,
which dominates), `final_state`, `final_mass_fp`, `final_soft_f1`, `step_mass_fp`,
`step_soft_f1`.

The question it answers: **which term is anti-correlated with `deploy_clot_score`?** With that,
the fix is targeted rather than another guess. It also serves as a reproducibility check on
Phase 1, which nothing else in this project has ever had.

## 23.7 Phase 2a — per-term decomposition. The brake points the wrong way.

> **SHARES CORRECTED IN 24.2.** The 54.3%/2.8:1 magnitudes below are inflated ~4.9x by a
> denominator bug in `record_loss_term`. The removal experiment (24.1) measures the brake at
> **11.1%**. The rho SIGNS are unaffected and stand; the share percentages do not.


`WG_phase2a_decomp`, 8 epochs, training config **byte-identical** to `WG_phase1_baseline`
(verified: empty config diff). The only difference is `record_loss_term()` accounting, detached
and never in the optimizer graph.

**Reproducibility, which this project has never had:**
```
mass by epoch
  phase1 : 3.062 2.708 2.699 2.363 0.708 2.062 0.522 0.788
  phase2a: 3.062 2.708 2.699 2.363 0.726 2.027 0.531 0.920
```
Identical for four epochs, then small divergence with the same shape. The pipeline is
deterministic to the point where a real effect is distinguishable from run-to-run noise.

**The decomposition (n=8):**

| term | share of loss | rho(term, deploy_clot_score) | direction |
|---|---|---|---|
| `final_mass_fp` | 38.6% | **+0.381** | minimising it HURTS the score |
| `step_mass_fp` | 15.7% | **+0.381** | same |
| `per_step_block` | 72.6%* | +0.119 | ~neutral |
| `final_state` | 0.0% | +0.238 | **dead** (exactly 0.0000) |
| `final_soft_f1` | 14.7% | **−0.333** | correct direction |
| `step_soft_f1` | 4.9% | **−0.381** | correct direction |
| TOTAL loss | | +0.048 | uncorrelated |

*nested: `per_step_block` contains the step terms.

**The sign split follows term family exactly, with no crossovers.** Every mass/regression term
is positive; both soft-F_beta terms are negative. Under random signs the probability that the
two surrogate terms are the only two negatives is 1/15 = 0.067. Not conclusive alone, but it
is coherent across all six terms and mechanistically consistent with 23.5.

```
mass-brake family :  41.36 = 54.3% of the loss   pointing AWAY from the metric
soft-F_beta       :  14.93 = 19.6% of the loss   pointing TOWARD it
ratio             :  2.8 : 1 in favour of the brake
```

**This is the mechanism of Phase 1's mass collapse.** The brake drives mass down; Phase 1 falls
to 0.43-0.96 while the score optimum is 2.0-2.4 (23.3). A term family that is 54% of the
objective and anti-correlated with the metric will do exactly that.

**The historical irony is worth recording.** The s9.11 growth brake has been in every leg since
v3. s12.6 found it *numerically dead* (soft occupancy pinned at 0.5), and Phase 0 revived it.
It was harmless while broken and harmful once fixed. Three sessions of work to restore a term
that needed removing.

**Correction to an earlier reading in this session.** At n=3 I recorded that *every* term was
flat and concluded term reweighting was futile. That was wrong: at n=8 the terms separate
cleanly by sign. Reweighting IS the answer — a measured one rather than a guessed one. The n=3
reading had too little dynamic range to see it, which is the same underpowering trap as 12.8.2.

**`final_state` is a seventh dead term** (mean exactly 0.0000; the leg's `final_state_weight`
resolves to `None`). Same family as 12.6's dead constants. Logged, not worth GPU.

## 23.8 Phase 2b — the targeted removal

`WG_phase2b_nobrake`. ONE conceptual change from Phase 1: the term family measured to be
anti-correlated is switched off.

```
step_mass_penalty      0.75 -> 0.0
step_prec_fp_penalty   0.50 -> 0.0
final_mass_penalty     1.50 -> 0.0
final_prec_fp_penalty  1.00 -> 0.0
final_mass_target      3.00 -> 2.2   (INERT at zero brake; set to the 23.3 measured optimum
                                      so the constant is not left stale if the brake returns)
```

Soft-F_beta is left untouched, so it becomes the rolled-state objective on its own.

**Pre-registered, before the result:**
* **Success** = mass stops collapsing (stays in 1.5-3.0 rather than falling to 0.4-0.9) AND
  best-epoch `deploy_clot_score` beats Phase 1's 0.4889.
* **Partial** = mass stabilises but score does not improve. Reads as "the brake was the mass
  mechanism but not the score bottleneck" -> the remaining gap is the 5-step-vs-200-step
  horizon mismatch (23.6), and the next lever is the share of optimizer steps that see a free
  rollout, currently 1 in 3270.
* **Null** = mass still collapses. Then the brake is not the driver and the suppression comes
  from `fp_weight`/`underpred_weight` (6:3) in the per-step block, which is the next candidate.

Note the honest asymmetry: two of three outcomes point away from further objective surgery and
toward the horizon mismatch. After v1-v10, the prior that "the loss is the problem" must be
allowed to lose.

# 24. Phase 0-2 review and the five fixes (2026-08-08)

## 24.1 Phase 2b result — the brake is inert. Third identical null.

`WG_phase2b_nobrake` = Phase 1 with the mass-brake family removed. Same seed, same data.

```
ep   mass ON   mass OFF    delta  |  score ON  score OFF    delta
 1     3.062     3.106    +0.044  |   0.4104     0.3971   -0.0133
 2     2.708     2.708    +0.000  |   0.3706     0.3706   +0.0000
 3     2.699     2.699    +0.000  |   0.3757     0.3752   -0.0005
 4     2.363     2.327    -0.035  |   0.4593     0.4699   +0.0106
 5     0.708     0.717    +0.009  |   0.4350     0.4430   +0.0080
 6     2.062     2.142    +0.080  |   0.4889     0.4552   -0.0336

mean |delta mass| = 0.0280      mean |delta score| = 0.0110
loss 76.208 -> 67.773 (the removed term was 8.43, i.e. 11.1% of the objective)
```

Removing 11% of the loss moved the rollout by ~1%, with epochs 2-3 identical to four decimals.
**Pre-registered Null branch fires**: the brake is not the driver of the mass collapse.

Three rolled-state term families have now failed identically — the dead occupancy (12.6), the
soft-F_beta surrogate (correctly signed but negligible), and the brake. **No loss-term
intervention has ever moved this model.**

## 24.2 A correction: the recorded term shares were inflated 4.9x

23.7 reported the brake as 54.3% of the loss. The removal experiment says **11.1%**.
`record_loss_term` accumulated across three call sites (main loop, deploy aux, window eval)
while dividing by each term's own call count, so terms touched by different numbers of paths
sat on different denominators. **Signs survive** (within-term, scale-invariant); **shares do
not**. The 2.8:1 magnitude claim is retracted.

## 24.3 A correction: `closed_loop_init` is NOT dead, and my "windows start from GT" claim was wrong

I recorded that every training window starts from a GT state and that this is the core
distribution gap. That is false. `closed_loop_init` is consumed at
`train_species_pushforward_continuous.py:1334`:

```python
if int(win_use[0]) > 0 and closed_loop_init_prob() > 0.0 and random.random() < closed_loop_init_prob():
    log_state0 = rollout_prefix_log_state(model, pack_data_gpu, static_gpu, int(win_use[0]), device)
```

At the configured 0.45, **45% of windows already start from the model's own free-running
state**. This materially weakens the hypothesis I had ranked highest: the training distribution
is already half-corrected and the loss is still uncorrelated with deploy score. Fix 1 therefore
becomes a config change (0.45 -> 1.0), not new code.

## 24.4 The ninth dead constant: `final_state`

Recorded as exactly 0.0000. Not a logic bug — a scale gap. The growth Huber lifts its inputs by
`delta_value_scale = 1.5e5` before comparison; the final-state Huber does not, and log-states
are O(1e-4):

```
final_state huber on raw states     : 5.00e-09  (x loss_scale 0.1 x fw 0.35 = 1.75e-10)
growth huber WITH value_scale 1.5e5 : 7.375e+00
ratio                               : 1.5e9 x
```

Same family as 12.6.1's dead occupancy and 12.6.2's `fp_thresh`. Fixed by lifting, not deleting.

## 24.5 Answers to the five review questions

**(1) Loss-term scale normalisation** — a real defect, twice over: my instrumentation (24.2) and
the objective itself, where `loss_scale=0.1` multiplies rolled-state terms but not the per-step
Huber, so any rolled weight was implicitly divided by 10 against its competitor.

**(2) Should mass be in the loss at all** — no. `deploy_clot_score` is relaxed *precision* gated
by a recall floor; once recall clears the floor the score IS precision. A mass target optimises
something the metric does not reward, and 24.1 shows it does nothing anyway. It survives as a
selection guard, where it is cheap and works. A related error of mine: `rolled_soft_f1_beta`
was 1.0 (F1 weights precision and recall equally) when the metric is precision-dominated — the
surrogate built to track the metric was mis-specified against it. Now 0.5.

**(3) Why the brake does not work** — three measured layers: numerically dead (12.6), revived by
Phase 0 (range 0.12% -> 97.7%), and *still* inert once alive (24.1). Mechanism: rolled-state
terms attach to `states[-1]` of a 5-step window that begins near GT, so the rolled state barely
differs from GT, the penalty sits near its floor, and its gradient is small against a per-step
Huber with 5x more terms and direct supervision at every one.

**(4) What loss actually correlates** — the per-step block IS already "predicted delta-species
matches real delta-species", it is 72% of the loss, and it is the only thing moving the model.
Its correlation with deploy score is **+0.119**. The form is right; what is wrong is the state
distribution the windows are drawn from — and per 24.3 that is already 45% model-generated, so
the remaining lever is to take it to 100%. Note `train_t0_coverage_frac=0.85` already spreads
t0 across the timeline, so "not just from t=0" is handled.

**(5) z_kin ablation** — **never run.** Phase 2 became 2a (decomposition) and 2b (brake
removal); B2 was specified in 17/19.5 and skipped. There is no evidence either way. Now wired.

## 24.6 The five fixes, as wired

| # | fix | mechanism |
|---|---|---|
| 1 | free-running windows | `closed_loop_init` 0.45 -> **1.0** |
| 2 | term accounting | `set_loss_accounting()` gates recording to the training path only |
| 3 | mass out of loss; beta retuned | four penalties -> 0; `rolled_soft_f1_beta` 1.0 -> **0.5** |
| 4 | z_kin ablation | new `latent_ablate`: hard zero at train **and** eval |
| 5 | dead final_state | `final_state_value_scaled` lifts it by `delta_value_scale` |

Fix 4 deliberately does **not** reuse `latent_dropout`: that is stochastic and a no-op at eval,
so it would train and deploy on different inputs and confound the ablation. `latent_ablate`
zeroes symmetrically and keeps `in_dim = 287`, sidestepping the 13.8 warm-start blocker.

**Attribution.** `WG_phase3a_closedloop` changes eight config values but is effectively
single-variable: 24.1 measured the mass terms inert, fixes 2 and 5 are accounting and scale
repair, and beta is a metric-matching correction. The one behavioural variable is
`closed_loop_init`. `WG_phase3b_zkin_ablate` differs from 3a by **exactly `latent_ablate`**,
asserted in a test so it cannot drift.

Suite: **575 passed**.

# 25. Handoff — state, the central finding, and what to do next (2026-08-08)

All processes stopped. `WG_phase3a_closedloop` reached 2 of 4 epochs; its salvage checkpoint is
retained at `outputs/biochem/eda/phase3/WG_phase3a_closedloop/`.

## 25.1 THE central finding: different objectives converge to the same weights

Two materially different loss configurations, one epoch each, same warm start:

```
||Phase1cfg - warm|| / ||warm||   = 0.3058     both move ~30% from the warm start
||3a cfg    - warm|| / ||warm||   = 0.3016
||Phase1cfg - 3a cfg|| / ||warm|| = 0.0795     but end only 26% as far APART as either moved
```

**About three-quarters of the weight update is common to both objectives.** The trainable heads
move substantially — a "the model is frozen" hypothesis was tested and rejected, relative
movement across all legs on record runs 0.24-1.35 — but they move to nearly the *same place*
regardless of what the loss says.

This is the unifying explanation for every null in sections 12, 23 and 24:

| intervention | loss change | rollout change |
|---|---|---|
| v4 `fp_weight` 6->16 | none (bit-identical) | none |
| v5 aux horizon 40->150 | none | none |
| v6 unroll 5->25 | +21% magnitude | none |
| Phase 2b brake removed | -11.1% | ~1% (epochs 2-3 identical to 4 dp) |
| Phase 3a `closed_loop_init` 0.45->1.0 | -9.2% | ~0% (epochs 1-2 identical to 4 dp) |

Five interventions, five nulls, one mechanism.

## 25.2 What Phase 0 and 1 actually achieved

Real and measurable, and it should not be lost in the nulls above:

* the `fp~292` attractor is **gone** — Phase 1 visited **10 distinct fp states** (25-242) where
  every leg v3-v10 visited 2;
* **four epochs passed selection**, against **zero** across v2-v10 combined;
* the metric has one canonical implementation (20.1) and labels are per-vessel (21.2);
* **reproducibility exists** — Phase 2a matched Phase 1 to 4 decimals for 4 epochs;
* nine dead constants found and either fixed or documented.

Best in-training deploy score to date: **0.4889** (Phase 1 epoch 6, mass 2.062). The zero-shot
warm start remains the best *deployed* configuration on a held-out vessel.

## 25.3 The one thing that is NOT yet explained

`per_step_block` is 72% of the loss, is the only term that moves the model, and correlates with
deploy score at **rho = +0.119** (n=8). Every intervention so far has edited the other 28%.
That is the gap between "we understand why nothing worked" and "we know what will".

## 25.4 Ranked next steps

1. **T1 — single-term loss ablations.** Per-step Huber ONLY vs soft-F_beta ONLY, 1-2 epochs
   each (~50 min). Directly tests whether the per-step block is the attractor. If leg A alone
   reproduces the Phase-1 rollout (mass ~3.06, fp ~242 at ep1), every objective edit outside
   that block is futile and the search moves to parameterisation.
2. **T2 — `z_kin` ablation.** Wired, tested, unrun, one command. 256 of 287 input dims against
   a flow channel Z1 measured at 0.041 AUC.
3. **T3 — unfreeze the backbone.** Never tried in 13 legs. 24% of params train; the reachable
   set may simply be too small for any objective to distinguish itself, which would explain
   25.1 directly.
4. **T4 — rethink the per-step block.** Check `fp_thresh` inertness FIRST (5 minutes; if the FP
   branch selects no nodes then `fp_weight` is unreachable and (b) is moot), then the 2:1
   suppression weighting.
5. **T5 — `loss_scale` asymmetry.** Rolled terms are implicitly /10 against the per-step block.
   Cheap, and it invalidates the 12.6.6 surrogate sizing that assumed comparability.

## 25.5 Closed — do not reopen

* **The brake.** Dead (12.6.1) -> revived (Phase 0) -> still inert (24.1, removing it moved the
  rollout ~1%). Mechanism understood: rolled terms attach to a 5-step window that begins near
  GT, so the penalty sits at its floor against a per-step Huber with 5x the terms. No payoff in
  further investigation.
* **Mass in the loss.** `deploy_clot_score` is relaxed precision gated by a recall floor; a mass
  target optimises what the metric does not reward. Survives as a selection guard only.
* **Regime routing** (13.4, all six cohort vessels are normal-regime), **change-B objective
  reweighting** (12.3, three failed specifications), **shear decoding** (Z4, +/-0.001).

## 25.6 Two corrections carried forward

Both are mine, and both were stated confidently before being checked:

* **`closed_loop_init` is NOT dead.** It is consumed at
  `train_species_pushforward_continuous.py:1334` and already started 45% of windows from the
  model's own rolled state. The claim that "every window starts from GT" was wrong, and it was
  the basis for ranking the state-distribution hypothesis first.
* **Per-term loss shares were inflated ~4.9x** by a denominator bug (24.2). The brake is 11.1%
  of the loss, not 54.3%. Signs survive; magnitudes do not. `set_loss_accounting()` now gates
  recording to the training path, but shares should still be verified by removal rather than
  read directly.

# 26. T7/T4(c)/T5 findings and the T1 ablation pair (2026-08-08)

## 26.1 Standing constraints, re-verified before anything ran (T7)

* Sealed set resolves to exactly the eight of 21.1 and is **disjoint** from
  `WALL_COHORT_V2_TRAIN` (26). `patient041` is in the train constant and the launcher removes it
  when it is the val anchor, so a run sees 25 train vessels. Neither junk vessel is present.
* `go_phase1_baseline.ps1` refuses a sealed `-ValAnchor` (line 49) and re-checks the train list
  against the seal, the junk list and a minimum size after resolution.
* `test_s24_legs_are_single_variable_against_each_other` still pins 3a vs 3b to `latent_ablate`.
* GPU idle at launch (0 MiB / 0%); one leg at a time, per the 650s -> 1900s contention result.

## 26.2 T4(c): the FP branch selects no nodes, now measured rather than inferred

12.6.2 argued the FP branch was unreachable from a **mean** predicted delta of ~1e-7 against
`fp_thr = max(fp_thresh=2e-5, delta_thresh=5e-6) = 2e-5`. A mean cannot establish that, because
the branch fires on the tail: `fp = (~gt_active) & (p_raw > fp_thr)` selects any single node
over the threshold.

What is now established:

* every logged `val_pred_delta` across all ten Phase-1 epochs and both Phase-3a epochs lies
  between **4.5e-8 and 5.2e-7** — 40x to 450x below the threshold;
* `ActiveGrowthHuberLoss` now counts its own selection. At 3e-7 with everything GT-inactive the
  count is **0**; at 5e-5 it is the full node set, so the counter is not stuck
  (`test_fp_branch_selection_is_counted_and_is_empty_at_realistic_deltas`);
* the counters (`diag_fp_nodes_ch*`, `diag_active_nodes_ch*`, `diag_pred_max_ch*`,
  `diag_fp_thresh`) are recorded through the existing accounting, so every future leg reports
  the **tail** statistic `diag_pred_max` in its train log and the question closes empirically.

`fp_weight=6.0` therefore multiplies an empty set, and **T4(b) is moot as stated**: the 2:1
suppression ratio of `fp_weight=6.0` against `underpred_weight=3.0` does not exist, because only
`underpred_weight` is reachable. The per-step block's suppression pressure comes from
`gate_fp_weight=4.0` (the BCE gate branch, live and unthresholded) — not from `fp_weight`.

## 26.3 A tenth dead constant, and three live terms nobody was accounting for

Found while wiring T1. Under the cohort runtime, **not** the dataclass defaults:

| term | site | weight | recorded by 23.7? |
|---|---|---|---|
| per-step gelation phi | inside the per-step block | **20.0** | no — hidden inside `per_step_block` |
| final gelation phi | added after `per_step_block` is recorded | **10.0** (20 x 0.5) | **no — in no term at all** |
| speed-FP bleed | added last | **4.0** | **no — in no term at all** |
| per-step / final mu | both | 0.0 | inert |

`physics_readout=True` on `WG_phase1_baseline` and `WG_phase3a_closedloop`, so all three were
live for every Phase-1/2/3 number. This is a second, independent reason the 23.7 shares cannot
be read directly, on top of the 24.2 denominator bug: the recorded terms did not sum to the
total. All five are now recorded (`step_phi`, `final_phi`, `step_mu`, `final_mu`,
`speed_fp_bleed`), and both T1 legs switch them off so that "the per-step growth Huber alone" is
literally true.

## 26.4 T5: the `loss_scale` asymmetry is real, and it is a dual-head regression

`loss_scale`=0.1 multiplies every rolled term at its own site — `rolled_soft_f1_loss`,
`rolled_final_mass_fp_penalty`, the final-state Huber. It also multiplies the per-step loss in
`continuous_delta_loss`. But `continuous_delta_loss` is the **single-head** path; every cohort
leg runs `dual_head=True`, and `dual_head_step_loss` never picked the scale up.

So the asymmetry is not a design choice — it is a constant that was correct for one path and was
not carried to the path that replaced it, which is the same shape as the other ten. Consequence:
every weight ever set on a rolled term has been implicitly **/10** against the only term measured
to move the model, `rolled_soft_f1_weight=120` included. **12.6.6's "120 = 8.2x the noise floor"
is invalid** — the realised multiplier was 0.82x.

Wired as `loss_scale_unified`, **off by default**, applying `1/loss_scale` to the rolled terms
where they are summed. Cancelling on the rolled side rather than scaling the per-step block down
leaves the dominant term's gradient magnitude untouched, so turning it on does not silently
recalibrate the learning rate. It is off in all four live legs, asserted in
`test_live_legs_have_not_silently_adopted_the_unified_scale`, and the asymmetry itself is pinned
by `test_loss_scale_reaches_rolled_terms_but_not_the_dual_head_block` so a future edit cannot
move a term across the seam unnoticed. The re-derived surrogate weight needs T1's measured term
magnitudes and is deliberately not guessed here.

## 26.5 A correction: 25.1's convergence number is an EPOCH-1 measurement

25.1's method now has a script (`scripts/diag_ckpt_weight_geometry.py`) and reproduces its 3a
figure exactly (**0.3016**). Its Phase-1 figure does not reproduce from the artifacts on disk:

```
||Phase1 best.pth - warm|| / ||warm|| = 0.5190   [epoch 4, the SELECTED checkpoint]
||3a     best.pth - warm|| / ||warm|| = 0.3016   [epoch 1]
||Phase1 - 3a||   / ||warm||          = 0.3447   -> 84% of how far they each moved
```

25.1 compared two **epoch-1** checkpoints; Phase 1's epoch-1 state is no longer on disk, having
been overwritten by later best-saves. The 84% above is **confounded** — epoch 4 against epoch 1
measures training time as much as objective — and is not evidence against 25.1. But it does
bound its scope: *"different objectives converge to the same weights" is established for the
first epoch, and for nothing beyond it.* The script now prints each checkpoint's epoch and warns
on a mismatch, because that confound is invisible in the norms themselves.

## 26.6 T1 as wired

`WG_t1a_perstep_only` and `WG_t1b_rolledf1_only`, both from the `WG_clotrich_nplus` warm start,
2 epochs each, run strictly one at a time. They differ from each other by exactly two knobs —
`per_step_weight` (1.0 / 0.0) and `rolled_soft_f1_weight` (0.0 / 120.0) — which is the seam, and
by nothing else (`test_t1_legs_are_exact_complements`). Everything off the seam is zero in both:
the four mass penalties, `step_soft_f1_weight`, `final_state_weight`, `physics_readout`, and
`speed_fp_weight`.

`per_step_weight` is new. The per-step block was the only term in the objective without a weight
knob, which is why the complementary ablation had never been reachable; 1.0 is the historical
behaviour exactly.

Mechanism verified engaged in leg A's fingerprint before trusting anything: `per_step_weight=1.0`,
`rolled_soft_f1_weight=0.0`, `step_soft_f1_weight=0.0`, `final_state_weight=0.00`,
`physics_readout=False`/`physics=0`, `speed_fp_weight=0.0`, `closed_loop_init=1.00`,
`prior_source=analytic`, `frozen=32 trainable_heads=8`. Note the fingerprint's env-var block is a
stale legacy echo (it prints `CLOSED_LOOP_INIT=0.45`, `FP_WEIGHT=8`, `FINAL_STATE_WEIGHT=0.35`);
the `config_kwargs`/`runtime_kwargs` blocks below it are the live values.

Both legs run 2 epochs, so `last.pth` gives a **matched-epoch** A-vs-B comparison — which
26.5 shows is the only kind worth computing.

## 26.7 An eleventh dead mechanism, and it is in the DEPLOY path

Surfaced by leg A's log, not by looking for it:

```
[WARN] Failed to initialize closed-loop flow coupler in deploy rollout: Error(s) in loading
state_dict for LocalKinematicCorrector:
  size mismatch for readout.2.weight: copying a param with shape torch.Size([2, 64]) from
  checkpoint, the shape in current model is torch.Size([3, 64]).
```

Commit `9eba0db` (2026-08-06 23:57, "add dShear output to local kinematic corrector") widened
`LocalKinematicCorrector.readout[-1]` to `nn.Linear(hidden_dim, 3)` for `[dU, dV, dShear]`. The
only corrector checkpoint on disk,
`outputs/kinematics/local_corrector/local_kinematic_corrector_best.pth`, still has
`readout.2.weight` at `(2, 64)`. `load_local_corrector` loads strictly, so it raises; both call
sites (`species_pushforward_continuous.py:4268`, `species_gnn_clot_rollout.py:360`) catch it,
print a WARN, and leave `coupler = None`. The rollout then falls through to uncoupled flow.

The path is gated by `SPECIES_CLOSED_LOOP_COUPLING == "1" or flow_source == "auto"`, and every
cohort leg runs `flow_feats_source: "auto"` — so it is attempted, and fails, on **every run since
2026-08-06**. Same shape as the other ten: a mechanism widened at one point, its stored artifact
left at the old width, failing silently into a fallback.

**The comparability consequence is the serious part.** The zero-shot benchmark — `patient043`
`deploy_clot_score = 0.6925`, `outputs/biochem/eda/commit_order/eval_p043_gate25.json`, dated
**2026-08-05**, `flow_feats_source: auto` — was measured while the coupler still **loaded**.
Every Phase 1, Phase 2 and Phase 3 number was measured with it **off**. So the standing headline,
*"thirteen fine-tune legs have not beaten the zero-shot warm start"*, spans an unrecorded change
in the deploy path. That does not make the gap spurious, but it does mean the two sides of it
were not measured by the same rollout, and 22's re-baseline rule needs a second clause:

> Numbers from before 2026-08-06 were produced with closed-loop flow coupling ON. Numbers from
> Phase 1 onward were produced with it OFF. They are not directly comparable either.

**Not fixed here, deliberately.** The breakage is byte-identical across Phase 1, Phase 3a and both
T1 legs, so every comparison *within* that set is valid; changing the deploy path mid-experiment
would invalidate the only clean A/B this study has. Queued as its own task. When it is fixed, the
fix must not be `strict=False` — that would leave the dShear head at its init and silently alter
the flow correction instead of restoring it — and the failure must stop being a WARN, because a
caught exception that disables a deploy mechanism is exactly what let this run for two days.

## 26.8 T1 RESULT — the per-step block is the attractor, and the objective IS steerable

Both legs, 2 epochs each, warm start `WG_clotrich_nplus`, one at a time. Mechanism verified
engaged in both fingerprints before any number was read; leg B additionally logged
`per_step_block_eff = 0` and `final_soft_f1 = 11.46`, so the ablation is confirmed from inside
the loss and not only from the config.

### Epoch 1 — the four legs are genuinely different

| leg | score | mass | fp | rprec | val_dlt | seeds | front |
|---|---|---|---|---|---|---|---|
| Phase 1 | 0.4104 | 3.062 | 242 | 0.408 | 2.19e-07 | 2.0 | 2.81 |
| 3a | 0.4056 | 3.080 | 244 | 0.402 | 2.10e-07 | 2.0 | 2.73 |
| leg A (per-step only) | 0.4353 | 2.743 | 206 | 0.437 | 2.00e-07 | 2.0 | 2.44 |
| **leg B (surrogate only)** | **0.2660** | **4.451** | **399** | **0.253** | **1.69e-06** | **69.0** | **5.02** |

### Epoch 2 — the split is exactly "does the objective contain the per-step block?"

| leg | per-step block | `deploy_clot_score` | `deploy_clot_mass_ratio` | fp |
|---|---|---|---|---|
| Phase 1 | yes | 0.3706237861441937 | 2.7079646017699117 | 202 |
| 3a | yes | 0.3706237861441937 | 2.7079646017699117 | 202 |
| leg A | yes | 0.3706237861441937 | 2.7079646017699117 | 202 |
| **leg B** | **no** | **0.35997755749156174** | **2.7964601769911503** | **212** |

Three objectives sharing almost nothing except the per-step block — **and the same input** —
produce a **bit-identical** committed set: 17 significant figures, same fp, same fn, same
relaxed precision, having started epoch 1 apart (fp 242 / 244 / 206). The one leg without the
block does not join them, and (26.9) neither does a leg that keeps the block but ablates the
input.

This is not a pinned metric: Phase 1 visits ten distinct fp states (25-242) and six distinct fn
states (9-91) across its ten epochs. Nor is it a saturated rollout: saturation would have been
identical at epoch 1 too, and it was not.

### Weight geometry, matched epoch (both legs at ep 2, so 26.5's confound does not apply)

```
||legA - warm|| / ||warm|| = 0.3625
||legB - warm|| / ||warm|| = 0.5467
||legA - legB|| / ||warm|| = 0.5812   ->  128% of their mean movement
```

Above 100% means the separation exceeds the average distance travelled — the two updates point
in **obtuse** directions. Set against 25.1's 26% for Phase-1-cfg vs 3a-cfg, the contrast is the
whole result:

| pair | differs by | separation |
|---|---|---|
| Phase1 cfg vs 3a cfg (25.1) | rolled terms only, all /10-suppressed | 26% — convergent |
| leg A vs leg B | whether the per-step block is present | **128% — near-opposite** |

### The verdict, and the reconciliation

Both branches of T1's pre-registered read turn out to be half right, and they fit together:

1. **Epochs 1-3 are a LOW-SENSITIVITY PLATEAU in which the committed set is fixed by (input
   representation, per-step block) and is insensitive to every other loss term.** This is
   weaker than the "attractor" this section originally claimed, and 26.9 is why: neither
   plateau state persists. `3b` freezes bit-identically at ep2->ep3 and then leaves at ep4;
   Phase 1 sits at 202/2.708 on ep2, 201/2.699 on ep3, and departs decisively from ep4
   (fp 165, then 31). They are early transients, not fixed points.
   The qualifier about the input is separately established and does hold:
   `WG_phase3b_zkin_ablate` contains the per-step block and never joins the intact-input
   plateau at all, so the plateau is a property of the per-step supervision *acting on a given
   input* — which makes the input the thing carrying the information.
2. **The objective is nevertheless steerable — but only through that block.** Removing it moves
   the model enormously and in the opposite direction. "The model cannot be steered" was never
   true; the interventions on record were simply never applied where the gradient lives.

25.1's convergence measurement is now explained rather than overturned: it compared two
objectives that differ *only* in rolled terms, and 26.4 shows those terms had always been
implicitly **/10** against an unscaled per-step block. They converged because neither had
actually changed the objective by much. Together with T4(c) — `fp_weight` multiplying an empty
set — every one of the five nulls edited a term that was either **dead** or **/10-suppressed**.

### The plateau indicts the whole null record, including this section's own comparison

Once 26.9 shows epochs 1-3 are a low-sensitivity window, the *measurement window* of every null
on record has to be checked — and they were all taken inside it:

| null | window it was measured in |
|---|---|
| Phase 2b (brake removed) | "epochs 2-3 identical to 4 dp" — **the plateau** |
| Phase 3a (`closed_loop_init`) | "epochs 1-2 identical to 4 dp" — **the plateau** |
| v4 (`fp_weight` 6->16) | bit-identical — but independently explained: the term is dead (26.2) |
| **leg A vs Phase 1 / 3a (this section)** | **epoch 2 — the plateau** |

So the project has been A/B-testing objectives in the one window where the committed set
provably cannot distinguish them, and this section's own headline comparison shares that flaw.
Phase 1's later epochs prove the metric *is* sensitive outside it — ep4-10 range 0.29-0.49 with
fp from 25 to 165.

**Therefore "every objective edit outside the per-step block is futile" is NOT established.** It
holds for epochs 1-3 and is untested beyond them. What survives the plateau objection intact is
everything measured by legs that never entered the plateau at all:

* **leg B never joins it** — removing the per-step block moves the committed set immediately, at
  epoch 1, and by a large margin (0.266 vs 0.406);
* **3b never joins it** — ablating the input does the same (0.224 vs 0.406);
* the **128% matched-epoch weight separation** between legs A and B.

Steerability therefore stands; futility does not. The correction is a budget rule: **an
objective A/B must run to at least epoch 4-6, past the breakout, or it is measuring nothing.**
The 8-10 epoch guidance in 25's handoff was right, and this section's 2-epoch budget — taken
from T1's own "1-2 epochs each" — was long enough to prove divergence but too short to prove
its absence. Legs A and B need re-running to 6 epochs before "futile" can be claimed either way.

### A second result: change E's surrogate steers AWAY from the metric

Leg B is the soft-F_beta surrogate alone — "the deploy metric itself, softened", the only term in
the objective with a TP numerator (12.6.4). Run alone it produces the **worst** score of the four
legs (0.266 at ep1), by overpainting to mass 4.45 at precision 0.253 with 69 seeds against
everyone else's 2. It does this at `rolled_soft_f1_beta=0.5`, already tilted toward precision, so
beta is not the fault — the soft-occupancy relaxation is. Raising predicted Mat everywhere grows
soft-tp faster than soft-fp, so the term rewards blanket growth, and the per-tensor movement says
where it goes: `spatial_head.2.bias` — the single scalar governing global commit propensity —
moves **46x** its warm-start norm in leg B against 6.1x in leg A.

Caveat, held deliberately: leg B also removed the only term with direct per-node supervision, so
some of the overpainting is "no anchor" rather than "bad surrogate". What is established is that
the surrogate steers hard and steers the wrong way; that it is *solely* responsible is not.

### What this means for the ranked next steps

* **T3 (unfreeze the backbone) loses its motivating premise.** It was ranked on 25.1 — "the
  reachable set may be too small for any objective to distinguish itself". Leg A vs leg B at 128%
  separation shows the head-only parameterisation distinguishes objectives just fine. T3 may
  still help, but not for that reason.
* **T5 is promoted from cleanup to the leading candidate.** `loss_scale_unified` is wired and
  off; turning it on is the one-variable leg that lets a rolled term compete with the per-step
  block for the first time. It is now the cheapest test of whether the objective can be steered
  *toward* the metric rather than merely away from it.
* **T4 becomes "reshape the per-step block", and 4(b) is dead** — `fp_weight` is unreachable
  (26.2). The live suppression knob is `gate_fp_weight=4.0`; the live growth knob is
  `underpred_weight=3.0`. Those two, and `channel_weight_mat=8.0`, are where the gradient is.
* **Change E needs re-specification before it is trusted anywhere**, surrogate weight included.
  Re-deriving that weight was T5's open item; leg B says the term's *shape* is the problem, not
  only its scale.

### 26.8.1 Cold-eval numbers and retention

`eval_val_cold.json` on `patient041`, the selected checkpoint of each leg:

| leg | selected epoch | cold score | mass | fp |
|---|---|---|---|---|
| leg A | 1 (passed selection) | 0.3900 | 2.743 | 206 |
| leg B | 2 (**salvage promotion** — `best_score=-0.001`, no epoch passed) | 0.3858 | 2.796 | 212 |

Leg B's selection gate rejected both epochs, and `best_salvage.pth` was promoted as designed
(12.6.4 item 1) — the leg still produced an artifact rather than nothing. Note the cold numbers
land close together (0.390 vs 0.386) even though the two models are 128% apart in weight space
and 0.435 vs 0.266 apart in-training at epoch 1. The cold eval is taken at each leg's *selected*
epoch, not at a matched one, so it is not a like-for-like comparison and should not be read as
"the two legs are equivalent". The matched-epoch comparison is the 26.8 table.

Neither leg beats the 0.6925 zero-shot warm start — and per 26.7, that number was measured with
the closed-loop flow coupler working while both of these were not, so the gap is not yet a
like-for-like measurement either.

Suite: **581 passed** (575 before this section, +6 guards).

## 26.9 T2 RESULT — `z_kin` is load-bearing, and the plateau is not a fixed point

`WG_phase3b_zkin_ablate`, 4 epochs, differing from `WG_phase3a_closedloop` by exactly
`latent_ablate` (asserted in `test_s24_legs_are_single_variable_against_each_other`; re-checked
at launch — config diff `{latent_ablate}`, runtime diff empty).

Mechanism verified end-to-end before any number was read, because "the config fingerprint says
True" is exactly what v4 and v5 looked like:

1. fingerprint carries `latent_ablate: True`;
2. `maybe_drop_latent` zeroes the first 256 columns at train **and** eval (pre-existing test);
3. the training path calls it (`unroll_continuous_loss`);
4. the **deploy** path calls it too —
   `deploy_species_rollout_series` -> `predict_continuous_step_delta` -> `maybe_drop_latent`.
   This link had no test; it now has one. Without it the leg would train ablated and deploy
   full, with an identical fingerprint either way;
5. the hard-ablation branch precedes the `if not training` short-circuit, so eval cannot return
   early with `z_kin` intact;
6. run log shows `in_dim=287 latent=256`, so the zeroed slice is the right one.

### Result: ablating 256 of 287 input dims is severely harmful

| epoch | 3a (intact) | 3b (ablated) | delta |
|---|---|---|---|
| 1 | 0.4056 / mass 3.080 / fp 244 | 0.2241 / 4.655 / 422 | **-0.182** score, +178 fp |
| 2 | 0.3706 / mass 2.708 / fp 202 | 0.2187 / 4.743 / 432 | **-0.152** score, +230 fp |

**The `z_kin` ladder does NOT close.** 89% of the input is not dead weight, the model does not
shrink, and D3 / 11.2.1 stay open. This is the strong version of the v10 result that
`latent_dropout=0.30` regressed the holdout by -0.137, which 25 recorded as "weak evidence
`z_kin` is load-bearing".

**The Z1 paradox sharpens rather than resolves.** Z1 scored the entire flow channel at 0.041 AUC
(GT field 0.789 vs zero-prior 0.748) — near-useless by that probe — yet zeroing the latent costs
45% of the deploy score. Whatever `z_kin` carries is invisible to an AUC probe on the flow
field. That gap is now the largest unexplained result on the board and is not on any task list.

**One confound, stated plainly.** The warm start was trained *with* `z_kin`, so ablation hands
the model an input distribution it has never seen. This establishes that the latent cannot be
dropped from this checkpoint; it does not establish that a model trained from scratch on 31 dims
would fail. The 13.8 warm-start blocker is exactly why `in_dim` was held at 287, so that
stronger question is unanswered — but it is now expensive rather than cheap, which was the point
of running the cheap version first.

### The incidental finding that matters more: the plateau breaks

3b froze **bit-identically** across ep2 -> ep3 while its internals kept moving hard:

```
ep2  score=0.21871653855248357  mass=4.743362831858407  fp=432  front=4.565  val_dlt=6.33e-07
ep3  score=0.21871653855248357  mass=4.743362831858407  fp=432  front=6.704  val_dlt=7.47e-07
ep4  score=0.23317               mass=4.5044             fp=405  front=2.015  val_dlt=1.12e-07
       ^ BREAKS OUT
```

Front speed rose 47% and the predicted-delta scale 18% between ep2 and ep3 with the committed
set frozen to the last digit — then it left. Phase 1 does the same thing at the same place:
202 on ep2, 201 on ep3, then 165 / 31 / 131 from ep4.

So the epoch-2 identity in 26.8 is a **plateau in an early transient, not a fixed point**, on
both input representations. The consequence for how this project measures anything is written
up at the end of 26.8: every null on record was measured inside that window.


## 26.10 A twelfth dead mechanism: `latent_ablate` was a no-op in the CANONICAL EVAL

Caught by a cross-check, not by looking: leg A's cold eval reproduced its in-training rollout
**exactly** (mass 2.7434, fp 206, both), while 3b's did not — in-training ep4 gave mass 4.504 /
fp 405, and the cold eval of that same checkpoint gave mass 0.212 / fp 9. A 21x mass gap from
the same weights on the same vessel is not a scoring difference; it is a different rollout.

Mechanism:

* `maybe_drop_latent` reads the width to zero from `model.kin_latent_dim`, and its guard is
  `if continuous_latent_ablate() and ld > 0`;
* `kin_latent_dim` is bound in exactly one place —
  `train_species_pushforward_continuous.py:940`, inside the **training** process;
* `load_continuous_bundle`, which the canonical eval uses, computed `latent_dim` from meta,
  stored it on the *bundle*, and never set it on the *model*;
* so at eval `ld = 0`, the guard fails, and the ablation is skipped.

Verified directly rather than argued: the eval-loaded model had no `kin_latent_dim` attribute at
all, and `maybe_drop_latent` under `latent_ablate=True` returned z_kin with max |value| 3.49
instead of 0.

**A leg fine-tuned on zeroed z_kin was being scored on intact z_kin** — precisely the
train/deploy asymmetry the hard ablation was chosen over `latent_dropout` to avoid (24 fix 4).
Same shape as the other eleven: a value bound for one path and not carried to the other.

**What this does and does not invalidate.** 26.9's conclusion is **unaffected** — it rests on the
in-training epoch-1/2 comparison against 3a, and both ran inside the training process where the
binding exists. Only the 3b *cold* number (0.2928) was meaningless, and it was never load-bearing.

**Fixed**, and the fix is provably scoped: `model.kin_latent_dim` is now bound in the loader.
With `latent_ablate` off — every other leg — `maybe_drop_latent` returns its input at eval
regardless of the width, so **no existing number moves**; verified in both directions.

Two lessons worth keeping:

1. **The guard test I added in 26.9 was not enough.** It pinned the *call chain*
   (`deploy_species_rollout_series` -> `predict_continuous_step_delta` -> `maybe_drop_latent`)
   but not the *width binding* the chain depends on — the same class of gap it was written to
   close. `test_eval_bundle_binds_kin_latent_dim_so_ablation_is_not_a_no_op` now covers it.
2. **In-training vs cold-eval agreement is a free integrity check.** When a leg's cold eval
   reproduces its in-training `mass` and `fp` exactly, the two paths agree; when it does not,
   something differs between them. Leg A agreeing is what made 3b's disagreement legible. Worth
   running on every leg from now on — it costs nothing and it caught this.

### 26.10.1 Fix verified end-to-end

Re-running the 3b cold eval on the same epoch-4 checkpoint, before and after:

| | score | mass | fp |
|---|---|---|---|
| before (ablation a no-op) | 0.2928 | 0.2124 | 9 |
| **after (ablation applied)** | 0.2442 | **4.5044** | **405** |
| in-training ep4 | 0.2332 | **4.5044** | **405** |

`mass` and `fp` now reproduce the training-time rollout exactly, leaving only the scoring-scope
offset (`runtime_bound` False in training, True in the scoring scope) — the same signature leg A
already showed. The integrity check of 26.10 lesson 2 now passes for both legs.

Suite: **583 passed** (575 at the start of section 26, +8 guards).

## 26.11 T5 result (partial) and the pivot to Phase 3

### T5: the loss_scale asymmetry was load-bearing

`WG_t5_unified_scale` = 3a + `loss_scale_unified`, one variable (asserted). Epoch 1:

| leg | score | mass | fp | `deploy_mat_f1` |
|---|---|---|---|---|
| 3a (baseline) | 0.4056 | 3.080 | 244 | 0.2646 |
| leg A (per-step only) | 0.4353 | 2.743 | 206 | 0.2828 |
| leg B (surrogate only) | 0.2660 | 4.451 | 399 | 0.2638 |
| **T5 (unified scale)** | **0.4334** | **2.823** | **215** | **0.2819** |

**Not a null** — and it moves *inside* the epoch 1-3 plateau where Phase 2b and Phase 3a were
bit-identical to baseline: score **+0.028**, fp **-29**, mass **-0.26**, `mat_f1` **+0.017**, all
in the precision-improving direction. So a global/rolled term does steer the model once it is no
longer implicitly /10 against the per-step block. That is the first positive objective result in
this project, and it de-risks training C1's nucleation head with a global ranking objective.

Read it as n=1 epoch. Note also that *removing* the weak rolled terms (leg A, 0.4353) and
*strengthening them 10x* (T5, 0.4334) land in nearly the same place, which is not obviously
consistent; the weak middle ground being the worst of the three is a hypothesis, not a result.
A 6-epoch run was relaunched to confirm past the ep4 breakout.

### The pivot

Phase 0, Phase 1 and Phase 2 of the project ladder are complete; **Phase 3 (C1) has never been
started**, and 0f (§20.4) already settled its head target. The T1-T7 list was a diagnosis of the
*current* model's objective — legitimate as debugging, but the ladder's own text had already
ruled that work uninterpretable before C1 ("C3 only if C1 lands"), and §14.6 had already
established that the multiplicative architecture cannot express 27-58% of commits.

What survives the pivot is the **instrument repair**, not the objective conclusions: `fp_weight`
unreachable (26.2), the `loss_scale` asymmetry (26.4), `latent_ablate` no-op at eval (26.10),
the coupler off since 2026-08-06 (26.7), the five unaccounted terms (26.3), and `deploy_mat_f1`
as a discrimination metric with resolution where `deploy_clot_score` plateaus (26.9/26.11).

**Phase 3 is a fresh build, not a fine-tune** — 13 warm-started legs all degraded the warm
start, and the additive nucleation term changes the function class. Full specification, standing
constraints, and the state of the instrument are in **`docs/PHASE3_HANDOFF.md`**, written for a
new context window.

Closed by the pivot: **T3** (premise refuted, moot under fresh init), **T4** (folds into C1's
loss design), and the 6-epoch re-run of T1's legs (moot once that objective is abandoned).

## 26.12 Nucleation census across the whole inventory — C1's premise, re-measured

14.6's growth-vs-nucleation split is the entire justification for C1, and it was computed on
**six** vessels (039-044) under a GT definition that was never recorded. 16.3 had already caught
this project generalising from exactly those six. So it was redone on the full inventory:
`scripts/diag_nucleation_census.py --label mat --ceiling-hops 3`.

```
n = 35 distinct vessels (mirror_y augmented duplicates excluded)
nucleation %:  mean 40.3   sd 10.3   range 27.0 .. 82.0
vessels below 27%:   NONE
vessels above 58%:   patient002 (58.2), patient003 (82.0)
```

**Every vessel in the inventory is at least 27% nucleation, mean 40.3%.** C1's premise is
confirmed, and more strongly than 14.6 stated it.

**But 14.6's per-vessel numbers do not reproduce**, under either GT definition:

| vessel | 14.6 | deploy-metric GT (`mu`) | rel_max label (`mat`) |
|---|---|---|---|
| `039` | 92 commits / 51.1% | 30 / 16.7% | 151 / 41.7% |
| `041` | 266 / 30.1% | 198 / 38.4% | 138 / 33.3% |
| `043` | 167 / 58.1% | 104 / 44.2% | 109 / 37.6% |

14.6 predates both the canonical metric (20.1) and the per-vessel `rel_max` labels (21.2) and
did not record which it used. The *conclusion* survives; the *numbers* should be quoted from the
census from now on.

### 26.12.1 The finding that changes C1's head target

The census also profiles nucleation by time quartile:

```
purely-early (all nucleation in Q1):   10/35 vessels
LATE-dominant (Q3+Q4 > Q1+Q2):          7/35 vessels
  patient001 [2,2,10,11]   patient010 [9,1,8,16]   patient021 [21,1,8,18]
  patient032 [0,13,9,47]   <- 47 of its nucleation events in the FINAL quartile
```

20.4 settled C1's head target as the `t=20` seeds, because seeds are +0.097 AUC more predictable
and 2.7x more consistent than the final map. That reasoning is sound **only if early and late
nucleation sites are the same kind of place.** A fifth of the inventory nucleates predominantly
late, so if they are not, a seed-trained head systematically misses those seven vessels — a
mis-specification that would be extremely hard to diagnose after C1 is built and underperforming.

**This is now step 0 of Phase 3, ahead of any code:** compare Q1 nucleation sites against Q4
nucleation sites under the deploy-legal features. Same distribution and same best-feature AUC ->
train on seeds per 20.4. Separated -> the head needs a time input or a slow rate modulation.
Pure CPU.

**Known limitation of the census tool:** its `seed_reach` column is degenerate — bimodal 0%/100%,
because it is really measuring "did anything commit by `t=20`" rather than reachability. 14.6's
43-81% does not reproduce and should not be cited either. Redefine that metric before using it.

## 26.13 Step 0 RESULT — early and late nucleation are different places, and C1's head spec was wrong

26.12.1 flagged the question; `scripts/diag_nucleation_timing_probe.py` answers it. 12 vessels
had >=5 nucleation sites in both Q1 and Q4 (333 Q1, 211 Q4). Measured against a permutation null
that re-labels the *same* sites at random -- required, because with 80+ features and small site
counts raw separation is large by chance:

```
feature                     sep    null   excess   AUC(Q1|neg)  AUC(Q4|neg)
hop_from_wall              0.478  0.079   +0.399      0.119        0.520
sdf_nd                     0.478  0.093   +0.385      0.115        0.631
kine_x_shear_potential     0.478  0.093   +0.385      0.885        0.369
on_wall                    0.449  0.078   +0.371      0.866        0.417
```

Chance ~0.08, observed ~0.48, **excess +0.399**. Not a small-sample artifact.

**Early nucleation is ON the wall; late nucleation is AWAY from it** -- the two populations rank
in opposite directions on every top feature. `sdf_nd` puts Q1 at 0.115 and Q4 at 0.631; `on_wall`
0.866 vs 0.417.

**This kills the seed-only head target.** 20.4 settled it on the grounds that `t=20` seeds are
+0.097 AUC more predictable; that is true and irrelevant if the seeds are a different population
from what the head must predict later. A seed-trained head learns "nucleate near the wall" and
ranks late sites BELOW chance -- worse than useless on the seven late-dominant vessels of 26.12.

**And it corrects the C1 spec.** `docs/PHASE3_HANDOFF.md`'s first draft had NUCLEATION as a
static per-node field computable once per vessel. That is wrong: a time-invariant rate cannot
place early sites on the wall and late sites off it. NUCLEATION must read the **current** flow
field -- which is what the new coupled corrector supplies -- while staying independent of
*adjacent committed clot*, which is what makes it nucleation rather than growth.

**Confound, stated because it is substantial:** as the wall commits, a new wall-adjacent commit is
more likely to have a committed neighbour and so be classified as growth. Part of "late
nucleation is off-wall" is a selection effect of the growth/nucleation definition itself, not
necessarily physics. The design consequence survives either way -- the sites the head must rank
late are off-wall -- but the mechanism is not established and must not be assumed.

**Incidental fix:** `gt_neg_dgamma_dx_phys` referenced an undefined `vel_source`, raising
`NameError` on every call unless `T0_R4_FLOW_SOURCE` or `CLOT_PHI_VEL_SOURCE` was set to
`kinematics` (the `or` chain short-circuits before the bad name). Broken since 393dcac
(2026-08-05, the config refactor); the sibling call `_resolve_uv_for_temporal_risk(...,
vel_source=vel_source)` shows a parameter was intended and never added to the signature. Fixed by
adding `vel_source: str | None = None`. This blocked the whole t=0 feature-table path.

## 26.13.1 Late nucleation IS learnable from t=0 features — the requirement is time, not flow

A refinement of 26.13, from the same probe data. Direction-agnostic signal strength of the best
deploy-legal t=0 feature against band negatives:

```
EARLY (Q1) sites:  0.406 above chance
LATE  (Q4) sites:  0.416 above chance
```

Late sites are **as learnable as early ones**; they are simply carried by *different* features,
pointing the opposite way (early: `shear_potential` 0.885, `sdf_nd` 0.115 -- on-wall; late:
`graph_degree` 0.916, `sdf_nd` 0.631 -- off-wall).

So 26.13's design conclusion needs narrowing: the nucleation head must not be **time-invariant**,
but it does **not** require the evolving flow field. Time conditioning alone may suffice, and the
coupled field is an enhancement to try second. The physical story (late nucleation is flow-driven
once clot alters the field) remains plausible and remains untested.

**A red flag for C1's feature set.** The strongest late-site predictor is `graph_degree` -- a mesh
property, not physics. High degree means local mesh refinement. A head that learns "nucleate where
the mesh is fine" will train well and fail on any new mesh, which is precisely the generalization
failure this project is trying to escape. `bio_x_mu_bc_nd` (0.793) is a boundary-condition channel
with the same problem. Neither should enter the nucleation feature set until its signal is shown
to survive across vessels of differing mesh density.

## 26.13.2 RETRACTION of 26.13 — late "nucleation" is 2-hop growth, misclassified

26.13 concluded that early and late nucleation happen in different places and that C1's head
therefore needs time conditioning. **That conclusion is withdrawn.** The user challenged whether
late sites were simply clot growing on existing clot, and they are.

For each "nucleation" site, distance at its commit time to the nearest already-committed node:

| vessel | pop | n | <=2 hop | median hop |
|---|---|---|---|---|
| `patient013` | Q1 | 84 | 28.6% | 4 |
| `patient013` | **Q4** | 30 | **100%** | **2** |
| `patient020` | Q1 | 6 | 33.3% | 99 |
| `patient020` | **Q4** | 28 | **100%** | **2** |
| `patient041` | Q1 | 5 | 20.0% | 99 |
| `patient041` | **Q4** | 17 | **100%** | **2** |

**Every late-quartile "nucleation" site in all six vessels sits within 2 hops of existing clot,
median exactly 2.** Early sites are genuinely isolated (median 3 to infinity, 20-43% within 2
hops). The 1-hop adjacency rule was too strict: late sites are the growth front advancing
outward from the wall into the lumen -- growth, and out of scope for a wall model.

What this retracts:

* **the "early on-wall / late off-wall" separation** -- it is the signature of an outward growth
  front, not two populations of ignition sites;
* **"the nucleation head needs time conditioning"** -- withdrawn; 20.4's seed-only target is
  sound, because genuine nucleation *is* early;
* **`graph_degree` as a late-site predictor** -- explained: growth fronts sit in the mesh-refined
  regions that follow wall geometry. A symptom of the misclassification, not a signal.

### The corrected premise

Re-running the census with a hop-tolerant growth rule (`--growth-hops`):

```
growth rule    mean nucleation    range
1 hop (old)         40.3%        27.0 .. 82.0
2 hop               21.2%         4.2 .. 60.0
3 hop               18.9%         3.4 .. 58.0
```

**Genuine nucleation is ~20% of commits, not ~40%**, and on some vessels under 5%. C1's premise
survives -- it is still a mechanism the multiplicative architecture cannot express -- but the
prize is half the size 14.6 and 26.12 implied, and it is not uniform across the inventory.

**And the corrected number agrees with the PDE, where the old one did not.** 10.1: fresh
deposition is ~7% of Mat *mass flux*, the remainder autocatalytic. ~20% of commit *events* being
ignition-dominated is consistent with that (a nucleating node starts small, so it is a larger
share of events than of mass). 40% was not consistent with 10.1, and that should have been
caught when the census was written.

## 26.13.3 C1 as specified does not match 10.1, and 10.1 should win

Checked after the above, and it should have been checked before the C1 spec was written.

```
J0_Mat = Da·( [d(sr,x) < sgt]·(L/gamma)·|d(sr,x)|·common   <- separation gate, 21%
            + [sr < lss]·common )                           <- low-shear gate, 79.7% DOMINANT
common = Sat(M)·k_rs·rp + Sat(M)·k_as·ap + (Mas/Minf)·k_aa·ap
```

Two mismatches with `dMat = NUCLEATION(field) + GROWTH(local committed Mat)·gate(shear)`:

1. **There is no spatial propagation term in the PDE.** `(Mas/Minf)·k_aa·ap` autocatalyses on the
   node's OWN adsorbed mass. Clot does not spread neighbour-to-neighbour; each node ignites when
   its *local* shear gate fires and then self-accelerates. Spatial clustering is inherited from
   the shear field being smooth. But `_apply_autocatalytic` -- the implemented GROWTH term --
   aggregates over `edge_index`, i.e. **neighbours**. That is the wrong autocatalysis, and it is
   the same error the 1-hop nucleation rule made: assuming propagation where the physics has
   local ignition.
2. **The form is multiplicative, not additive.** Nucleation and growth are not two mechanisms to
   sum; they are one equation, with `k_dep` dominating at `Mat_i ~ 0` and `k_auto·Mat_i`
   dominating once ignited. Both sit INSIDE the same shear gate.

A physics-faithful form is closer to

```
dMat = gate_shear(regime) · Sat(Mat_i) · ( k_dep(field) + k_auto · Mat_i / Minf )
```

with `gate_shear` routing between the low-shear and separation mechanisms -- which is exactly
10.4's bimodality, separable at t=0 by `band_speed_q25` with **90.6% leave-one-vessel-out**. The
"clots form differently depending on geometry" observation is *which gate is in charge*, and it
has already been measured as routable.

This keeps C1's motivation -- an explicit ignition term the current architecture cannot express
-- while grounding the form in the PDE rather than in commit-event statistics that turned out to
be definitionally fragile.

## 26.14 T5 final result — the loss_scale fix is real and does not help

`WG_t5_unified_scale`, 6 epochs, one variable (`loss_scale_unified`) against 3a.

| ep | T5 score | T5 mass | T5 fp | T5 rprec | Phase 1 score | P1 mass | P1 fp |
|---|---|---|---|---|---|---|---|
| 1 | 0.4334 | 2.823 | 215 | 0.438 | 0.4104 | 3.062 | 242 |
| 2 | 0.3706 | 2.708 | 202 | 0.362 | 0.3706 | 2.708 | 202 |
| 3 | 0.4031 | 2.699 | 201 | 0.397 | 0.3757 | 2.699 | 201 |
| 4 | 0.4424 | 1.478 | 84 | 0.522 | 0.4593 | 2.363 | 165 |
| 5 | 0.4065 | 0.619 | 31 | 0.552 | 0.4350 | 0.708 | 31 |
| 6 | 0.4220 | 1.398 | 85 | 0.467 | **0.4889** | 2.062 | 131 |

**On the IN-TRAINING metric: best T5 0.4424 vs Phase 1's 0.4889** -- better at epochs 1 and 3,
bit-identical at 2, worse at 4-6.

**On the CANONICAL COLD EVAL the ranking REVERSES.** Both legs select epoch 4; both evaluated on
`patient041` through `eval_mat_growth_simple.py`:

| leg | sel ep | cold score | mass | fp | rprec |
|---|---|---|---|---|---|
| Phase 1 | 4 | 0.4319 | 2.363 | 165 | 0.434 |
| **T5** | 4 | **0.5103** | 1.478 | 84 | 0.545 |

**T5 wins by +0.078 and clears the 0.50 floor** -- the best cold number of the session. See
26.14.1: the two metrics disagree in *direction*, which is a live instrument problem, so neither
"T5 helps" nor "T5 does not help" can be asserted until it is resolved.

Caveat: Phase 1 is *not* T5's control -- 3a is, and 3a stopped at 2 epochs. Phase 1 additionally
differs in `physics_readout`, the mass terms, `rolled_soft_f1_beta` and `closed_loop_init`, so
the epoch 4-6 gap is not attributable to the unified scale alone. Extending 3a to 6 epochs would
make it attributable; that is not worth the GPU given the pivot.

Two incidental findings worth keeping:

* **Epoch 2 joined the plateau exactly** (0.3706237861441937 / 2.7079646017699117 / 202), making
  it the *fifth* leg to do so. The attractor survives a 10x rescaling of the rolled terms. It
  yields only to removing the per-step block (leg B) or changing the input (3b).
* **Epoch 3 matched Phase 1 on `mass`, `fp`, `fn` AND recall, yet scored +0.027 higher**, because
  its 201 false positives were better *placed* -- relaxed precision dilates by 2 hops. **Counts
  are not sufficient to characterise a committed set**; any such comparison must include
  `relaxed_prec`. This weakens the in-training-vs-cold-eval cross-check of 26.10 as originally
  stated, which used `mass` and `fp` only.

## 26.15 PIVOT — Phase 3 is now a physics-mirroring model, and the law is already in the repo

The finding that reframes the project: **the COMSOL wall deposition law is implemented, and
validated to machine precision, in this repository.**

* `src/core_physics/comsol_surface_deposition.py` -- canonical `J0_Mat`, single source of truth
* `src/tests/test_comsol_wall_deposition_calibration.py` -- pins it against COMSOL's own exported
  `J0_*` columns (`patient007`, 876 wall nodes x 201 timesteps). **5 tests pass.**
* `src/core_physics/biochem_physics_kernels.py::biochem_wall_residual` -- the law wired
  end-to-end, both gates and saturation included
* `docs/COMSOL_PHYSICS_VALIDATION.md` -- *"matches the repo's `BiochemConfig` to machine
  precision"*

Meanwhile the deployed stack is `frozen flow -> GraphSAGE learns Mat directly -> gelation ->
clot readout`. **The GraphSAGE replaces the law rather than using it.** Roughly fifteen legs have
been spent teaching a 187k-parameter network to approximate a function already available exactly,
which is the most parsimonious explanation on record for why it generalises poorly: the network
must re-infer `lss = 25 1/s` and `sgt = -7.5e4` from data on every new geometry, while the law
applies them exactly everywhere.

Tracing what `j0_mat_si` needs:

| input | source | learned? |
|---|---|---|
| `sat_m`, `mas` | integrated state | no |
| `step2t` | activation-phase gate | no |
| `shear_sr`, `dsrx` | flow field | via the flow surrogate |
| **`rp`, `ap`** | bulk platelet concentrations at the wall | **yes -- and only this** |

**The learning surface collapses from "predict `dMat` everywhere every step" to "predict platelet
concentrations at the wall."** Saturation, the correct *local* autocatalysis `(Mas/Minf)·k_aa·ap`,
both gates and their thresholds all come free and correct. And every species is a GT channel in
the packs, so the learned part is directly supervisable.

10.4's regime bimodality also stops being a routing problem and becomes a consequence of which
gate fires -- and it was already measured routable at t=0 by `band_speed_q25` (AUC 0.975, 90.6%
LOO).

**Step 0 of the new plan, before any model is written:** does `j0_mat_si` fed GT species from the
packs reproduce GT `dMat`? If yes, everything above follows. If no, the premise is wrong and it
must be found out then, not after building. The unit plumbing (`log1p_nd` -> SI, the
`1/(s*cm) -> 1/(s*m)` conversion on `dsrx`, the `x1e4` surface-rate factor) is where this will go
wrong; a mismatch means the plumbing or the premise is broken, **not** that the constants need
adjusting -- they are pinned by a passing test.

Full specification, build order, standing constraints and the state of the instrument:
**`docs/PHASE3_HANDOFF.md`**, rewritten for a new context window. The previous additive-C1 plan is
retained there as an optional comparison arm (section 3), with its premise corrected to ~21%.

## 26.14.1 THE IN-TRAINING METRIC AND THE CANONICAL EVAL DISAGREE IN DIRECTION

Found while writing up T5. On an **identical committed set** -- same `mass`, same `fp` -- the
in-training deploy score and the canonical cold eval differ, and the difference is not a constant
offset:

| leg | sel ep | in-training | cold eval | delta | committed set |
|---|---|---|---|---|---|
| Phase 1 | 4 | 0.4593 | 0.4319 | **-0.027** | mass 2.363, fp 165 (both) |
| T5 | 4 | 0.4424 | **0.5103** | **+0.068** | mass 1.478, fp 84 (both) |

**The two move in opposite directions and reorder the legs.** By the in-training metric Phase 1
beats T5; by the canonical eval T5 beats Phase 1 by +0.078.

The scoring *parameters* are identical in both fingerprints -- `clout_score_mode=guiding`,
`clout_prec_rec_floor=0.3`, `guide_relax_hops=2`, `guide_f_beta=0.5`, `empty_gt_fp_tol=8.0`. Only
`runtime_bound` differs, and that is a diagnostic flag (`get_active_runtime() is not None`), not a
scoring parameter. So the split is elsewhere in the two paths, and it is **not** the parameter
drift that 20.1 fixed.

**Why this matters more than T5's result.** In-training selection drives which checkpoint every
leg keeps. If it disagrees in *direction* with the canonical eval, then epoch selection across
this entire project may have been keeping the wrong checkpoints -- and every "best epoch" number
in sections 21-26 inherits that. This is an A1-class problem of the same family 20.1 was written
to close, and 20.1 evidently did not close all of it.

**Do not read T5's verdict as settled.** It wins on the canonical metric at a matched epoch and
loses on the in-training one; until the split is explained, neither is the answer. Resolving it is
cheap relative to its blast radius: instrument both paths on one checkpoint with one committed
set and diff the intermediate quantities (tp/fp/fn before relaxation, the dilation, the recall
floor, the guiding blend).
