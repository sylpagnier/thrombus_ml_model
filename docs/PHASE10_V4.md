# PHASE 10 — a strict protocol for `clot_gnn`, and what survives it

Opened 2026-08-17.  Two things were asked for: a **strict validation set** (v3 was selected
without one), and a **v4 that scores better on-wall and/or off-wall, mean-over-time and
especially at the last time point**.

> **HEADLINE.**
>
> 1. **v4 beats v3 on all four metrics**, strictly nested, identical protocol and code:
>
> ```
>                     mean wall   mean off   FIN wall   FIN off
> v3  (cv5a,b,c)         0.8687     0.6389     0.9014     0.7011
> v4  (v5a,b,c)          0.8750     0.7188     0.9176     0.7366
> delta                  +0.0063    +0.0799    +0.0162   +0.0355
> oracle timing, same set 0.9662    0.8709     0.9176     0.7372
> ```
>
> **The two off-wall gains are statistically significant; the two wall gains are not.**
> Paired vessel bootstrap, strictly nested:
>
> ```
> mean off   +0.0685 [+0.0281, +0.1116]   P(diff<=0) 0.000   n=13
> FIN  off   +0.0355 [+0.0100, +0.0619]   P(diff<=0) 0.002   n=13
> mean wall  +0.0063 [-0.0150, +0.0353]   P(diff<=0) 0.342   n=19
> FIN  wall  +0.0162 [-0.0153, +0.0572]   P(diff<=0) 0.203   n=19
> physics backbone            —          —     0.8766     0.4141
> ```
>
> Two independent sources, roughly equal in size.  **The features** — COMSOL's own advection
> operator solved on the mesh (§5) — and **the readout** (§10), which turned out to hold more
> score than any feature or architecture change measured here.  Priority class
> (stenosis/aneurysm) off-wall: mean 0.775 → 0.784, final **0.790 → 0.864**.
>
> 2. **v3's published numbers are ~0.02 optimistic.**  Three selection leaks removed in §1.
> 3. **The cohort's noise floor is ±0.024 wall and ±0.091 off-wall** (§2).  Three
>    configurations of the *same* arm spread that much, so most 0.01–0.03 differences in
>    `docs/PHASE9_ML.md` are not measurable at n=19.  **v4's own gains are at that scale and
>    are reported with that caveat** — a paired bootstrap of the final-time set gives wall
>    +0.0124 [-0.0169, +0.0528] and off -0.0015 [-0.0283, +0.0209].  What supports v4 is the
>    consistency of the direction across all four metrics and both domains, not any one cell.
> 4. **A coherence constraint is worth +0.06 off-wall at the last time point** (§3),
>    deterministically rather than statistically.
> 5. **+0.042 wall / +0.120 off-wall sit in the readout, not the model** (§4) — and five
>    unsupervised rules for collecting it all fail.  That is the sharpest open lever.

`clot_gnn_v4` = the three-configuration ensemble on the advective-transport feature cache
(`v5a,v5b,v5c`), read out under the strict protocol, with the commit-by-final constraint on
the temporal head.  **No arm selection, no per-domain family choice, no calibration rule** —
every one of those was tried and every one lost (§7).

---

## 1. THE THREE LEAKS, AND WHY THEY NEEDED NO RETRAINING

`scripts/train_time_conditioned.py` — the code that produced 13.9 and that `clot_gnn_v3`
ships — is out-of-fold in the **weights** and in-sample in the **readout**:

1. **The committed set used hard-coded cuts** `score >= 0.73` (wall) and `>= 0.92`
   (off-wall).  Those two constants were picked by looking at the whole 19-vessel pool, so
   every vessel was read out with a rule that had seen its answer.
2. **Thresholds that *are* tuned** are tuned on the fold's own training vessels using **that
   fold's model**, whose scores on those vessels are in-sample and therefore overconfident.
3. **The head is fitted on in-sample scores** and applied to out-of-fold ones, so its
   `score` feature has a different distribution at train and test time.  Always flatters.

None of this needs a retrain.  `run_phase9_cv.py` already saves, for every fold, that fold's
model's score on *every* vessel — so an out-of-fold score exists for all 19.  To evaluate
held-out fold `k`, everything is selected on the **out-of-fold scores of the vessels outside
`k`**, which come from other folds' models and none of which ever saw a vessel of `k`.

```
scripts/eval_strict.py            final time point, per domain
scripts/eval_strict_temporal.py   mean-over-time AND final, nested head + thresholds
```

Cost of honesty, same weights, same scores:

```
                          FIN wall   FIN off
PHASE9 11.2 readout         0.9198    0.7270
strictly nested             0.9024    0.7075
```

**Reporting both mean-over-time and the final time point is new.**  The project had only
ever quoted mean-over-time; the last time point is the fully-formed clot and is what a
reader of the prediction acts on.  They behave differently and §3 is a case where they
disagree sharply.

---

## 2. THE NUMBER THAT SHOULD GOVERN EVERY FUTURE COMPARISON

`scripts/eval_significance.py`.  Three configurations of the **same** arm, differing only in
`rounds` and `off_mult`, under the strict protocol:

```
cv5a   wall 0.9068   off 0.6537
cv5b   wall 0.9096   off 0.6958
cv5c   wall 0.8858   off 0.6053
                spread   0.0238        0.0905
```

And a paired vessel bootstrap of the largest deliberate change this round (v3 features → v4
advective-transport features, 3 configs each):

```
dom     base     cand     diff [95% CI]          P(diff<=0)
wall  0.9024   0.9092   +0.0068 [-0.0158,+0.0313]    0.277
off   0.7075   0.6889   -0.0186 [-0.0744,+0.0384]    0.746
```

**Why it is this bad:** 19 vessels, ~0.7% positive nodes, off-wall means taken over only the
13 vessels with non-empty off-wall GT, several carrying under 15 GT nodes — and the severity
score is F0.5-weighted, so one false positive on a 4-node vessel moves that vessel by tenths.

**Consequence.** A 0.01–0.03 cohort-mean difference on this cohort is not a result.  This
applies retroactively: several effects `docs/PHASE9_ML.md` reports as wins are inside this
band.  The levers that remain worth pulling are the ones that do **not** require a per-config
effect to be detectable — variance reduction by ensembling, and deterministic readout fixes.

---

## 3. THE ONE ROBUST GAIN — a coherence constraint, worth +0.064 off-wall at t_final

The committed set **is** the model's prediction of the final mask.  v3 then applies a
*further* probability filter at every timestep, including the last — so a node can be in the
set and still be predicted "not clot" at `t_final`.  That is the readout contradicting
itself, and it is expensive:

```
                                     FIN wall   FIN off
v3: set & (P >= th) at every t         0.9167     0.6431
v4: ... with mask(t_final) = set       0.9099     0.7075
```

`series_masks(..., commit_final=True)` in `scripts/eval_strict_temporal.py`.  With it the
final mask equals the set by construction, so **the temporal arm can no longer lose to its
own frozen set at the last timestep** — which it previously did by 0.064 off-wall.  Whether
to enforce it is still chosen per domain inside the fold, because on the wall the extra
filter also removes low-confidence set members and is close to neutral there.

This is a deterministic property of the readout, not a statistical effect, which is why it
is reported as a gain despite §2.

The second deterministic fix: v3 commits its set with a **plain** per-domain cut, and on
`patient032` that commits **nothing** off-wall (score 0.000 against 0.432) because its score
field is uniformly low there and the physics mask is the only thing separating its 120
off-wall nodes.  The physics-conditioned `resid` readout is offered alongside and chosen
in-fold.

---

## 4. WHERE THE REMAINING SCORE ACTUALLY IS — the readout, not the model

`scripts/diag_readout_ceiling.py`.  Same score field, final time point, strictly nested
cohort cut against a per-vessel **oracle** cut:

```
                          wall      off
nested cohort cut       0.9024   0.7075
per-vessel oracle cut   0.9447   0.8275
```

Per vessel, the big ones: `p035` wall 0.656 → 0.974, `p028` 0.699 → 0.854, `p019`
0.851 → 0.971, `p020` off 0.509 → 0.829, `p005` off 0.240 → 0.621, `p032` off 0.432 → 0.742.
**The network already separates these vessels; one cohort constant does not sit in the right
place on any of them.**  (Part of that gap is oracle optimism — it is the best of 33 cuts
chosen per vessel — so treat it as an upper bound, not a promise.)

`docs/PHASE9_ML.md` 4 records two failed per-vessel *budget* rules and never measured this
ceiling.  Five rules were tried here (`src/clot_ml/calibration.py`), all selected in-fold:

```
rule            wall      off     what it does
absolute      0.9069   0.6876     the shipped cohort cut (control)
rel_max       0.9102   0.6773     cut relative to the field's own maximum
quantile      0.7803   0.5312     commit a fixed fraction of the domain
phys_anchored 0.8747   0.5397     commit a multiple of the physics mask's count
gap           0.8803   0.5086     cut at the widest gap in the sorted score
nested_pick   0.9072   0.6825     let the fold choose among them
```

**None of them recovers it.**  `rel_max` is +0.003 wall / -0.010 off — noise.  Letting the
fold pick the rule is *worse* than fixing one, which is §2 again.

A sixth idea failed differently and is worth recording because it looked compelling.  GT clot
**is** `{Mat >= crit}` (PHASE7 10.1), and `src/clot_ml/gnn.py` regresses `log1p(Mat/crit)`,
so the mask can be read off the regression head at `log 2` — a **physical** threshold with
zero free parameters, per-vessel by construction.  `scripts/eval_reg_readout.py`:

```
arm          wall      off
cls        0.9055   0.5105     classifier, cohort cut
reg_phys   0.8228   0.1792     reg >= log 2, ZERO parameters
reg_tuned  0.8620   0.6006     reg >= t, t fitted in-fold
reg_budget 0.8785   0.1700     classifier ranking, count from the physical anchor
```

**The regression head is not magnitude-calibrated**, so the physical anchor lands far from
the right place and the zero-parameter route is dead.  But `reg_tuned` beats the classifier
off-wall on the same weights (0.6006 vs 0.5105; priority class 0.8296 vs 0.6445), so the
head *is* a better off-wall field — calibrating it is an open and well-posed target.

---

## 5. THE NEW PHYSICS — advective transport, and the bug that hid it

`src/clot_ml/transport.py`, `scripts/build_clot_ml_cache_v4.py`.

PHASE7 1.1 read the production `.mph` and found `Mat` is a **domain** field under
`dMat/dt + u.grad(Mat) = 0`, zero diffusion, wall flux BC.  Every off-wall channel v3 has —
`log_mat_owner`, `gate_owner`, `is_shell`, `hop_wall` — is a *nearest wall node* rule, i.e.
it moves information along the mesh **normal**, the one direction the equation does not
transport along.  So COMSOL's own operator was discretised (vertex-centred first-order
upwind) and solved for the steady field `u.grad(C) = S_wall`, plus the residence time
`u.grad(tau) = 1`.

The near-wall accumulation falls out rather than being assumed: no-slip makes near-wall
parcels slow, they dwell, they accumulate — which is the 0.16 attenuation of PHASE7 3.2
without anyone writing 0.16 down.  A finite-horizon term `V/T` caps the dwell at the run
length, which is what makes the steady problem well-posed at a stagnation point.

Against GT off-wall `Mat`, 19 vessels, it is a genuine **complement** rather than a
replacement — better on exactly the high-burden vessels that dominate the off-wall score:

```
              owner rule    advection
p032 (120 GT)     0.716        0.802
p041  (84 GT)     0.415        0.657
p044 (122 GT)     0.365        0.606
p037  (35 GT)     0.582        0.762
p020  (13 GT)     0.396        0.088     <- and worse on the low-burden ones
p035              0.539       -0.167
cohort mean       0.548        0.469
```

### 5.1 The first version had a boundary bug, and it cost the whole off-wall gain

The first operator computed each node's outflow by summing over its **edges** only.  Flux
that leaves through the **domain boundary** has no edge to travel along, so every outlet node
looked like a stagnation point and accumulated without limit.  A unit test on a 1-D chain
caught it (`src/tests/test_transport_and_strict.py`): with a source at one node and no sink,
the field must be *constant* downstream of it, and instead it grew by 1e9 at the last node.

The fix is one line and it is physics, not numerics: for divergence-free flow an interior
node balances, so any excess of inflow over edge-outflow is exactly the flux crossing the
boundary — charge it as additional outflow.

The difference is not cosmetic.  Off-wall rank against GT `Mat`, and the end-to-end
strictly-nested score of the same three configurations:

```
                       rank vs GT Mat    FIN wall   FIN off
buggy operator   (v4)          0.469       0.9138    0.6358
fixed operator   (v5)          0.529       0.9148    0.7059
v3, no transport                 —         0.9024    0.7075
```

**The bug turned a +0.004 off-wall change into a -0.072 one** while leaving the wall gain
intact, which is exactly the signature of a defect concentrated at outlets.  Anyone reading
the earlier `v4a/v4b/v4c` tags should discard them; `v5a/v5b/v5c` are the corrected runs.

### 5.2 End to end

```
              mean wall   mean off   FIN wall   FIN off
v3 (cv5a,b,c)    0.8687     0.6389     0.9014     0.7011
v4 (v5a,b,c)     0.8814     0.6815     0.9148     0.7059
```

Consistent in direction on all four, and largest where the physics argument predicts it —
the off-wall mean, +0.043, and the priority class's final off-wall, 0.790 → 0.849.  §2 still
applies to each individual cell: the paired final-time bootstrap is wall +0.0124
[-0.0169, +0.0528] and off -0.0015 [-0.0283, +0.0209].  **The claim rests on the direction
being consistent across four metrics and two domains, not on any one interval.**

One side benefit worth recording: the v5 features make the wall score far more stable across
configurations.  The config spread (§2) drops from **0.0238 to 0.0061** on the wall.

Two further physics ideas were built and measured **neutral**, and are recorded so they are
not re-derived:

* **The separation branch as an indicator.**  PHASE7 12.3 measured that capping the
  anti-informative `|d(sr,x)|` magnitude in the *rate* lifts oracle ordering 0.492 → 0.703.
  Rebuilt on the model path as `1[dsrx<sgt] + 1[sr<lss]` (mask bit-identical, no new
  parameter): wall `Mat` rank **0.6058 → 0.6017**.  PHASE7's gain was on oracle inputs and
  **does not transfer**.
* **Time-resolved transport for the temporal head.**  The operator is linear and the flow is
  frozen, so `mat_adv(t)` costs one solve per stored time and is the first time-varying
  quantity the head has ever had off the wall.  Mean-over-time wall 0.8720 → 0.8671, off
  0.6489 → 0.6509.  Neutral.

---

## 6. WHAT `clot_gnn_v4` IS — and why it is the SIMPLEST option tried

```
v5a   epochs 80  dim 64  layers 4  rounds 3  off_mult 1.0   metric legacy
v5b   ...                          rounds 5
v5c   ...                          rounds 3  off_mult 2.5
```

three seeds each, scores averaged, on `outputs/clot_ml_cache_v5`; read out with the
in-fold-selected readout family and thresholds of `scripts/eval_strict.py`, and timed by the
nested time-conditioned head with `commit_final`.

The metric is domain-restricted, so "which model" *could* be a per-domain question, and while
the buggy v4 features were strong on the wall and weak off it that looked compelling.
`scripts/eval_multiarm.py` implements it honestly — arm, family and thresholds chosen per
domain on out-of-fold scores of vessels outside the fold.  Once the operator is fixed the two
arms are close, and **selection makes things worse**:

```
arm                        wall      off
cv5a,cv5b,cv5c           0.9024   0.7075
v5a,v5b,v5c              0.9148   0.7059     <- v4, no selection
nested per-domain pick   0.9000   0.6984     <- selecting between them LOSES
ORACLE per-domain pick   0.9294   0.7220
```

That is §2 once more: with 14-15 vessels to choose on and arms this close, the selection's
own variance exceeds what it can win.  **v4 is one ensemble, one readout family per fold,
no arm selection.**

---

## 7. WHAT DID NOT WORK — measured, so nobody repeats it

* **Inner-CV selection of the readout family.**  Motivated correctly (`resid` has twice
  `plain`'s free scalars and wins every selection-set comparison by parameter count, then
  loses on held-out wall) and **it loses**: 0.9024/0.7075 → 0.8920/0.6688, and the same
  inversion on the v4 arms.  At n=19 the inner split's variance exceeds the bias removed.
  Kept behind `--family-select inner` and documented in `pick_family`.
* **Naive score-averaging across feature sets.**  6 configs pooled reads 0.9049/0.6730,
  *between* the two 3-config ensembles rather than above them.  Ensemble diversity does not
  help when the members disagree about which domain they are good at.
* **Per-domain arm selection** (§6): 0.9000/0.6984 against 0.9148/0.7059 for just using the
  better ensemble.
* **More seeds instead of more configurations.**  Six seeds of one config reads
  0.9092/0.6764; three seeds of each of three configs reads 0.9024/0.7075.  **Configuration
  diversity beats seed count off-wall** — which corrects `docs/PHASE9_ML.md` 4's "seed
  averaging is the single cheapest win" into "diversity is, and seeds are one source of it".
* **Training on the severity metric instead of the legacy one.**  `docs/PHASE9_ML.md` 9.3
  measured +0.027 off-wall for this and it does not survive the strict protocol, on either
  feature set:

  ```
                 legacy            severity
  v3 cache   0.9068 / 0.6537   0.9165 / 0.6311
  v4 cache   0.9017 / 0.6314   0.8979 / 0.5621
  ```

  Off-wall loses in both pairs.  Left at `metric=legacy`.
* **The four rules of §4**, and the zero-parameter regression anchor.
* **The A-branch indicator gate** and **time-resolved transport for the head** (§5).
* **The zero-parameter regression-head anchor** (§4).

---

## 8. WHAT TO DO NEXT, IN ORDER

1. **More vessels.**  §2 is the binding constraint on everything and it is a data problem,
   not a modelling one.  Nothing at the 0.01–0.03 scale can be validated until n grows.
   `patient039` re-run to the full horizon is the cheapest single addition.
2. **Calibrate the regression head** (§4).  It is already a better off-wall field than the
   classifier, and if its *magnitude* were right the readout threshold would be physics
   (`Mat >= crit`) rather than a fitted cohort constant — which is the only route on the
   table that dissolves the +0.042/+0.120 readout gap instead of estimating it.
3. **Timing, not the set** for mean-over-time: oracle timing on v4's own set is
   0.9662/0.8910 against 0.8750/0.6833, so **+0.091 wall and +0.208 off** remain purely in
   *when*, against +0.082/+0.264 for a perfect set at the final point.  The two prizes are
   now comparable in size, where PHASE9 13.9 had timing dominant.
5. **Readout, still.**  §10 collected roughly a third of §4's ceiling.  The per-vessel oracle
   cut is 0.9447/0.8275 against v4's 0.9176/0.7372, so ~+0.027 wall and ~+0.090 off-wall are
   still there.
6. **The off-wall LAG DISTRIBUTION** (§12) — the single best-specified open target.  The lag
   behind the owner has median +4 of 11 grid steps and a p25-p75 span of +3 to +6; a cohort
   constant is too crude and loses to the probability rule.  What is needed is what the lag
   is *conditioned on* — a per-node regression on the lag, which is now a well-posed target
   with 584 labelled examples rather than 19.  That is a much better-supported problem than
   anything else remaining, and it is the whole of the 0.680 -> 0.871 off-wall timing gap.
4. **SEALED remains closed.**

## 10. THE READOUT IS THE LEVER — and two things finally move it

§4 measured +0.042 wall / +0.120 off-wall sitting between the cohort cut and a per-vessel
oracle cut, and then five unsupervised rules failed to collect any of it.  Those rules all
share a shape: each **substitutes** the cohort constant with a statistic of the score's own
distribution.  Two constructions that do something else both work.

### 10.1 PERTURB the cut instead of replacing it

```
t_v = clip(a + b * (stat_v - median_over_train(stat)), 0.02, 0.98)
```

with `stat` the mean score in the domain.  `b = 0` reproduces the cohort readout **exactly**,
so the fit can only move away from it if the statistic pays on the selection vessels.  On a
plain cut it reads 0.9097/0.7194 against 0.9016/0.7136; on top of the physics-conditioned
`resid` readout it gives wall 0.9148 → **0.9176** at no off-wall cost.  Three of four
statistics (`mean`, `q90`, `physfrac`) move the score the same way; only the tail statistic
`q99` does not, so a tail quantile is the fragile choice at n=19.

### 10.2 Maximise the EXPECTED metric instead of thresholding it

`scripts/eval_expected_score_readout.py`.  Every readout this project has used asks "which
nodes score above a cut" — a per-node question about a metric that is not per-node.  Whether
the 40th-ranked node is worth committing depends on how many are already committed and how
confident the rest are.

So: treat `p` as a distribution over the unknown truth, and for each prefix of the
score-ranked list compute the **expected** severity score of committing that prefix, using
`soft_severity` with `p` in the place of GT.  Commit the arg-max prefix.  The stopping point
is a property of this vessel's own confidence profile, needs no label, and adapts the budget
automatically — which is exactly the medicine for the low-burden precision problem §5 and
`PHASE9` 5 both trace the off-wall shortfall to.

Two in-fold scalars correct for the known miscalibration (`gamma` sharpening, prefix scale):

```
arm                 wall      off
cohort_cut        0.9016   0.7136
expected          0.8882   0.5646     raw, uncorrected -- miscalibration bites
expected_tuned    0.9036   0.7359     <- off-wall +0.022
resid             0.9148   0.7059
resid_adapt       0.9176   0.7060
nested_pick       0.9176   0.7359     <- v4
```

### 10.3 Per-domain READOUT choice pays, where per-domain ARM choice did not

`nested_pick` chooses the readout per domain, inside the fold, over all four arms.  It picks
`resid_adapt` on the wall and `expected_tuned` off-wall **in every fold**, and gets the best
cell of each column honestly.  That is the opposite of §6, where choosing between score
*ensembles* per domain lost — and the difference is instructive: the two readouts differ
substantially and consistently per domain, so the choice is stable, whereas the two ensembles
were close and the choice was noise.

---

## 11. THIS ROUND'S NEGATIVES

* **Advective recurrence.**  The single largest architectural bet of the round, and it fails.
  `docs/PHASE9_ML.md` 2c calls the recurrence "flow-mediated" but the channels it feeds back
  are `A p` and `A^2 p` with `A` the **isotropic** adjacency — mesh diffusion, the same
  isotropy `PHASE6_RESULTS` 3.4 measured as wrong for the source.  Replacing them with the
  upwind operator's own upstream/downstream weights (`feedback_channels_advective`) is
  physically the right prior and reads **0.8958/0.7016** against 0.9176/0.7359.  Paired at
  one config it is wall -0.011 / off +0.031 — it does help the weak domain — but the
  ensemble does not survive.  Kept behind `--adv-fb 1`, documented, off by default.
* **Wider ensembles.**  18 members (v5 + advective-recurrence configs) reads 0.9001/0.7085,
  *below* the 9-member v5 ensemble.  Pooling members that disagree about which domain they
  are good at does not help, which is the same lesson as §7.
* **Head fusion.**  Rank-fusing the classifier and regression fields collapses
  (0.7927/0.4312); logit fusion selects weight 0, i.e. it declines to use the regression head
  at all.  The regression head is a better off-wall *field* (§4) and still cannot be fused
  naively.
* **A per-vessel physics clock for the temporal head** — the fraction of this vessel's own
  ODE/advection nodes fired by `t`, which is the deployable form of the per-vessel schedule
  calibration `PHASE9` 13.5 identified as the ODE's real contribution.  Mean-over-time wall
  +0.005 but off-wall **-0.038**.  A vessel-level scalar lets the head fit the schedule of
  14 training vessels rather than learn a transferable one.  Off by default (`--clock`).
* **A learned residual on the ODE's wall onset** (`--wall-resid`), the wall analogue of the
  off-wall lag that worked.  Selected in 2 of 5 folds; ALL mean wall 0.8750 -> 0.8749 and
  mean off 0.7075 -> 0.7017 (the wall mask feeds the off-wall lag rule, so a worse wall
  schedule propagates).  Priority-class wall does gain (+0.009).  Off by default.
* **Ordering the expected-score prefix by the regression head** instead of the classifier.
  Fusion failed earlier as a *threshold* field; inside the prefix readout only the ORDER
  matters, so the rank-flattening objection does not apply -- and it still loses:
  `expected_reg` off-wall 0.7068 and `expected_both` 0.7266, against `expected_tuned`'s
  0.7359.  The regression head is a better off-wall field on its own and adds nothing to the
  classifier's ordering.
* **Seed-ensembling the temporal head** is the one that half-works: 4 heads averaged give
  mean-over-time off 0.6815 → 0.6948 and wall -0.005.  Kept, because it moves the weaker
  domain and variance reduction is the lever this cohort rewards.

**A pattern worth naming.** `w5a,w5b,w5c`'s plain cohort cut reads wall **0.9228** — the
highest single wall number in this document — and the in-fold selection did not choose it,
landing on 0.8958 instead. In-fold selection among close readouts costs ~0.02 here, every
time it is measured (§7, §6, this). Prefer a fixed, mechanically-justified readout over a
selected one unless the arms differ as sharply and consistently as §10.3's do.

## 12. WHAT OFF-WALL TIMING ACTUALLY LOOKS LIKE — measured, after four failed models

Five modelling attempts at off-wall timing have now failed the same way: `PHASE9` 12.2's
owner-threshold rule, `PHASE9` 12.4's curve head, and this round's per-vessel physics clock
(§11), two-stage owner-onset head (below) and owner-lag rule (below).  Every one of them was
a plausible mechanism fitted *before* anyone measured the quantity it was meant to predict.
`scripts/diag_offwall_structure.py` measures it.

### 12.1 Off-wall clot lags its owner, and the lag is large

Pooled over 584 off-wall GT nodes, in grid steps of an 11-point grid:

```
p0   -2      p25  +3      p50  +4      p75  +6      p90  +7      p100 +9
lag <= 0 : 8.4%      lag == 1 : 3.6%      lag >= 2 : 88.0%
```

**Two things follow, and both correct standing assumptions.**

* **The owner-precedence constraint is nearly vacuous.**  It binds on the 8.4% of nodes that
  would otherwise commit at or before their owner.  `src/clot_ml/locked.py` ships it and the
  strict evaluator had dropped it; restoring it moves mean-over-time off-wall
  0.6833 -> 0.6803 and final off-wall 0.7359 -> 0.7372.  **It is kept anyway**, because an
  off-wall node predicted as clot with no clot on the wall feeding it is physically
  incoherent output, and 0.003 of mean score is a cheap price for not emitting it.
* **The information is in the LAG, not the ordering.**  Off-wall does not commit *with* the
  wall, it commits most of the horizon later — which is the boundary layer filling to ~0.16
  of its owner's `Mat` (PHASE7 3.2) and only then crossing `crit`.  The per-time score
  curves show it plainly: `patient020` scores nothing off-wall until `t/T = 0.8`, `patient032`
  nothing until 0.5.

### 12.2 And a single cohort lag is still too crude to use

The obvious rule — `off onset = owner's PREDICTED onset + lag`, with one cohort lag fitted
in-fold — is implemented (`offwall_by_lag`, `--owner-lag`) and the in-fold tuner **selects
the plain probability rule over it in all five folds**.  The spread is why: p25 +3 against
p75 +6 on an 11-point grid, so one constant misses most nodes by 2-3 steps, and it also
inherits every error in the predicted wall onset.  The measurement says the lag is real and
large; it does not say it is constant.

**This is now the sharpest open question in the project**: off-wall mean-over-time is 0.680
against an oracle-timing 0.871 on the same set, and §12.1 says essentially all of that gap is
a per-node lag whose *distribution* is known and whose *conditioning* is not.

### 12.3 The other two off-wall arms this round, both negative

* **Two-stage owner-onset head.**  Give each off-wall node its owner's stage-1 *predicted*
  trajectory as a feature -- strictly better information than the owner's ODE state, which
  is biased low and is exactly why `PHASE9` 12.2 collapsed.  Nested properly (selection
  vessels get inner out-of-fold stage-1 predictions, held-out vessels get the stage-1 head
  fitted on all of them), it reads mean-over-time off **0.6580** against 0.6803.  Same
  failure shape as the physics clock: a strong derived signal that the head over-trusts,
  compounding the wall model's own timing error.  `--two-stage`, off by default.
* **De-quantising the off-wall lag.**  The lag is rounded to whole grid steps, which is why
  refining the regression was exactly neutral -- so predicting a continuous fraction of the
  run and comparing absolute times should have unlocked it.  It is **worse**: mean-over-time
  off-wall 0.7075 -> 0.6665, priority 0.8411 -> 0.7257.  The rounding is not lost resolution,
  it is a regulariser; it absorbs the regression's error instead of letting every mispredicted
  fraction move a commit across a step boundary.
* **Inner-fold seed count is not free.**  Dropping the inner cross-validation's heads to one
  seed (they only feed threshold tuning) costs mean-over-time off-wall 0.6833 -> 0.6706.
  The inner predictions are what the time cuts are tuned on, so their variance lands
  straight in the chosen cut.  Inner and final head seed counts are now tied.

---

## 13. THE FINAL OFF-WALL SET — anatomised, and six ways it does NOT improve

Final off-wall is the committed set's quality and it sits at **0.7359** against a per-vessel
oracle cut of 0.8275.  This section is almost entirely negative results, so the anatomy comes
first: it is what makes the negatives interpretable, and it is the durable part.

### 13.1 The anatomy: recall is already 1.000 on 9 of 13 vessels

Severity components of the shipped off-wall mask, per vessel:

```
vessel       n_gt n_pred tp_rel   fn   fp  prec_eff rec_eff  shape
patient005      4     17      4    0   13     0.267   1.000  0.222
patient012     90     62     84    6    1     1.000   0.988  0.691
patient020     13     28     13    0   12     0.615   1.000  0.435
patient029     14     15     14    0    6     0.692   1.000  0.618
patient032    120     63     48   72   13     0.820   0.417  0.276
patient037     35     21     26    9    2     1.000   0.867  0.572
patient044    122     91    110   12    3     0.989   0.940  0.771
```

**Three facts, none of them previously stated.**

1. **Recall is 1.000 on nine of thirteen vessels.**  Off-wall is no longer a recall problem
   except on `p032` (0.417) and, mildly, `p037`/`p044`.
2. **The two failure modes are opposite and burden-sorted.**  Low-burden vessels are
   over-predicted (`p005` commits 17 for 4; `p020` 28 for 13), high-burden ones
   under-predicted (`p032` 63 for 120).  The chosen budget is **compressed toward the cohort
   middle**.
3. **The dilation-IoU term binds even where detection is perfect.**  `p012` has precision
   1.000 and recall 0.988 and still scores 0.845, because `shape` is 0.691.  Since
   `score = 0.5*shape + 0.5*detect`, a vessel with a perfect detection score is still capped
   at `0.5 + 0.5*shape`.  **Above ~0.85, off-wall is a SHAPE problem, not a detection one.**

### 13.2 Six things that do not improve it

* **Coupling the off-wall set to the WALL set** (an off-wall node may commit only if its
  owner is in the committed wall set): **0.7119** against 0.7359, losing in all five folds.
  PHASE7 3.1's "an off-wall GT node's owner is GT-committed 99.9% of the time" is a fact
  about *GT*; our wall set is not GT, so the constraint deletes real off-wall nodes whose
  owner the wall arm missed.  The 99.9% does not transfer to a predicted mask.
* **Restricting to the topological shell**: **0.6791**.  `PHASE9` 4 measured this as a loss
  under a plain threshold and it was worth re-testing under a readout that re-spends the
  freed budget -- it is still a loss, by more.
* **Both together**: 0.6974.
* **Anti-compression on the budget** (`k -> k_med * (k/k_med)^beta`), aimed squarely at
  fact 2 above: **0.6852** with three `beta` values.  The mechanism is right and the knob is
  wrong -- see 13.3.
* **An off-wall SPECIALIST GNN** (`--off-only`: BCE, the regression head and the metric term
  all masked to the off-wall domain, so nothing in the objective is wall).  Single config,
  same readout: off-wall **0.6376** against the generalist's 0.6586.  The shared trunk's wall
  supervision is *useful auxiliary signal* for off-wall, which is what the physics says --
  off-wall `Mat` is advected wall flux, so a model that understands the wall understands the
  source.
* **Morphological closing / opening** of the off-wall mask, zero parameters.  Closing at
  `>=3` predicted neighbours is a bit-identical no-op and at `>=2` costs 0.006; **opening**
  (dropping predicted nodes with no predicted off-wall neighbour) collapses the score to
  **0.2996**.  That last number is the useful one: **most correctly-predicted off-wall nodes
  have no predicted off-wall neighbour at all**, because off-wall GT is a single node row
  interleaved with mid-side nodes (AGENTS.md: ~3/4 of all nodes are mid-edge).  Neighbourhood
  morphology inside the off-wall subgraph is not a meaningful operation here.

### 13.3 The search space is itself a hyperparameter

The anti-compression arm is worth one more line because of *how* it failed.  Same arm, same
code, same data -- only the size of the grid it is fitted over changes:

```
beta grid          combinations   held-out final off-wall
{1.0}                       42            0.7359
{1.0, 1.4, 2.0}            126            0.6852
```

and the SELECTION-set score rose while the held-out score fell.  At n=19 a readout knob is
not free even when its neutral value is in the grid.  Fixing the budget compression needs a
*mechanism* that does not add a fitted parameter -- which, given 13.1's fact 3, most likely
means attacking `shape` rather than the budget.

---

## 14. IS FINAL OFF-WALL > 0.8 REACHABLE?  No, and here is the arithmetic

A full round was spent on this single number.  It ends at **0.7366**, and the answer is that
the target is out of reach on this data -- not for want of a readout, but because two
independent ceilings sit below it.

### 14.1 The two ceilings

```
best prefix of the current ranking, per-vessel ORACLE k     0.8205
   ... with spacing enforced (sparse, non-overlapping balls) 0.8204
oracle per-wall-node band thickness (a different object)     0.9014
SHIPPED                                                      0.7366
```

So a **perfect per-vessel budget on the ranking we have gives 0.82**, and the readout would
have to be near-oracle to clear 0.8.  The 0.90 ceiling belongs to a formulation nothing can
currently drive (14.3).

### 14.2 The discriminator is already near its information limit

The decision off-wall actually turns on is: *of the first-shell nodes whose owner is
committed, which ones clot?*  That population is 1339 nodes at **38.8% positive** -- not the
0.7% of the full-mesh problem.  On it, leave-one-vessel-out:

```
best SINGLE feature (univariate)          AUC 0.72
GBM, MESH channels only        (10)       AUC 0.714
GBM, PHYSICS channels only     (59)       AUC 0.896
GBM, ALL channels + GNN score  (70)       AUC 0.912
the GNN's own score, alone                AUC 0.887
```

Two things follow.  The decision **is** jointly determined and it **is** physical -- physics
channels alone come within 0.006 of everything.  (An earlier reading of the univariate table
as an "information wall" was wrong and is retracted.)  And the shipped GNN score is already
within **0.025 AUC** of a model trained on nothing but this population, so the ranking is
close to what these 69 channels support.  There is no large modelling gain left here.

### 14.3 The higher-ceiling formulation exists and cannot be driven

The number of off-wall GT nodes owned by one wall node is **0 or 1**, essentially always
(p032: 73 zeros / 120 ones; p020: 97 / 13).  Off-wall clot is one companion hanging off some
committed wall nodes -- PHASE7 3.1's thin band that the boundary-layer mesh resolves in ~2
rows.  Painting oracle counts scores 0.9014.  But predicting the count is not possible with
what we have:

```
oracle n_w>=1 labels, same painting        0.8947
learned n_w>=1 (nested, honest)            0.5357
dedicated shell head, predicted wall set   0.6460   (its own oracle: 0.9441)
per-node readout, for reference            0.7359
```

**The formulation has +0.08 of ceiling and -0.19 of achievable.**  Worth recording precisely
so it is not re-derived as an obvious improvement.

### 14.4 Physics routes closed this round, with numbers

* **Owner attenuation is dead, not merely unfitted.**  PHASE7 12.5 called `att * Mat_owner`
  "the cheapest untried route to off-wall 0.7".  Thresholding `0.16 * Mat_owner >= crit` with
  **ORACLE wall Mat** scores **0.1214**.  It fires on every shell node with a committed
  owner, and most of those do not clot.  Close the route.
* **The band-thickness derivation does not bite.**  Near the wall `u ~ gamma-dot * y`, so
  residence time goes as `1/(gamma-dot * y)` and `Mat(y) ~ J0 L/(gamma-dot y)`, giving a
  band thickness `delta ~ J0/gamma-dot`.  Ranking shell nodes by `log(delta) - log(y)` reads
  AUC **0.598** against plain distance's 0.590.  The band is **sub-mesh** -- whether a node
  falls inside it is decided at a scale the mesh does not resolve, which is the same fact
  PHASE7 3.1 recorded as "a mesh artefact".
* **Occlusion-aware transport.**  The band forms LATE (median lag +4 of 11 steps, 12.1), by
  which time the wall clot is an 80x viscosity jump and the t=0 flow is wrong.  Re-solving
  the transport with the physics-mask region set to zero velocity gives within-shell AUC
  **0.500-0.553** -- at chance.
* **There is no unused species information.**  All twelve `tds`/`tds2` GT channels
  (`RP, AP, APR, APS, PT, T, AT, FG, FI, M, Mas, Mat`) are **spatially uniform at t=0**
  (coefficient of variation 0.0000).  Only `u, v, p, mu_eff` vary.  The t=0 feature set is
  complete; there is nothing left on the packs to mine.

### 14.5 Readout routes closed this round

Owner-coupling to the predicted wall set (0.7119), shell restriction (0.6791), both (0.6974),
budget anti-compression (0.6852), burden budgets from confidence mass / transport count /
wall count (best 0.7150), per-owner non-maximum suppression (0.7343 at cap 1, 0.7385 at cap
2 -- the readout already commits ~1 per owner), morphological closing (0.7303) and opening
(0.2996), mid-side restriction (0.7142), regression-head ordering (0.7068).  **None beats
0.7359.**

One of these is worth keeping as a fact rather than a failure: **opening collapses the score
to 0.2996**, i.e. most correctly-predicted off-wall nodes have no predicted off-wall
neighbour, because the band is one node row interleaved with mid-side nodes.

### 14.6 And more seeds make off-wall WORSE

Paired, same config, same readout: `v5a` (3 seeds) off-wall **0.7141**, `v5a6` (6 seeds)
**0.6540**.  Within-shell AUC falls too (0.8734 -> 0.8614), so this is not only the readout's
budget sensitivity -- the averaged field is genuinely a slightly worse discriminator here.
`docs/PHASE9_ML.md` 4's "seed averaging is the single cheapest win" does not hold off-wall on
the v5 field.

### 14.7 What would actually be needed

Not a readout and not an architecture.  Either **more vessels** -- 13 carry off-wall GT, and
every quantity above is estimated on them -- or **a label that is not partly sub-mesh**
(14.4), which means a finer boundary-layer mesh in COMSOL rather than anything on this side.
Recommend redirecting effort to mean-over-time off-wall (0.708 against a 0.871 ceiling, with
a proven mechanism in 12.2b) and to the wall, where both metrics still sit below their
oracles and neither has been attacked with the lag construction that worked.

---

## 15. WHERE THE OFF-WALL TIMING GAP ACTUALLY IS -- decomposed, then closed a bit further

Mean-over-time off-wall is the arm with a proven mechanism (12.2b) and the most headroom.
The right first move is not another model but a decomposition, and it points somewhere
non-obvious.

### 15.1 The wall onset costs 2.3x what the lag does

An ORACLE per-node lag applied with OUR OWN predicted wall onset:

```
frozen (no timing at all)                     0.4394
shipped (learned lag)                         0.7075
ORACLE lag + predicted wall onset             0.7569     <- lag error costs 0.049
full oracle timing                            0.8709     <- wall-onset error costs 0.114
```

So two thirds of the remaining off-wall timing gap is **not** the off-wall model at all -- it
is that the off-wall schedule inherits every error in *when we think the owner committed*.

### 15.2 The fix is the anchor, not the model

The lag was trained against GT owner onsets and applied against the head's *predicted* owner
onsets.  Target and anchor were different objects, so the wall arm's error was injected
straight into the off-wall schedule.  Anchoring both on the **ODE's own crossing** -- a
quantity that is identical at train and at apply time, and needs no wall model --

```
anchor = head's predicted owner onset     mean off  0.7075
anchor = ODE's own owner crossing         mean off  0.7188     (baseline class 0.6674 -> 0.6821)
```

`--lag-anchor ode`, and it is now the shipped configuration.  Paired bootstrap +0.0114
[-0.0085, +0.0426], so it is inside the off-wall noise floor as a number; what recommends it
is that it is exactly the correction 15.1 predicts, and it moves the baseline class (where
the wall arm is weakest) rather than the priority class.

### 15.3 What the anchor cannot be

* **The advected field's own crossing** (`--lag-anchor adv`), `mat_adv(t) >= crit` per node.
  This is the most direct physics anchor there is -- COMSOL's transport operator answering
  "when does this node clot" with no owner indirection -- and the in-fold tuner **rejects the
  learned lag entirely** under it, falling back to the probability rule (0.6803).
* **A higher owner level.**  Off-wall commits when `att*Mat_owner >= crit`, so the anchor
  "ought" to be the owner's crossing of `crit/att`, not `crit`.  Sweeping the level over
  {1, 2, 4, 8}x`crit` in-fold selects **1** and reads 0.7188 -- unchanged.

### 15.4 And the wall arm itself will not give the 0.114 back

Two attempts, both nested honestly:

* **A learned residual on the ODE's wall onset**, gated on wall burden -- the exact
  construction that made the off-wall lag work.  Selected in 2 of 5 folds; mean wall
  0.8750 -> 0.8749 and mean off 0.7075 -> 0.7017.
* **A separate wall cut used only to date the owner**, tuned against the OFF-WALL score
  rather than the wall's own.  Exactly neutral (0.7075).

So the wall-onset error is real, is worth 0.114 off-wall, and is not reachable by
re-parameterising the wall clock.  `PHASE9` 13.7 said why: the remaining wall-timing
information is a per-vessel onset *distribution shape*, and 19 vessels do not determine one.

---

## 8b. THE LOCKED ARTIFACT — SHIPPED 2026-08-21

```
outputs/clot_ml/locked/clot_gnn_v4/
  member_v5<a|b|c>_s<seed>.pth   9 members (3 configs x 3 seeds), ~5.6 MB -- the GNN ensemble
  feature_norm.npz               the 69-column mean/std and column ORDER
  temporal.pkl                   4-seed time-conditioned head + 3-seed off-wall lag model
                                  (plain sklearn estimators only -- no custom wrapper class,
                                  so it unpickles without `scripts/` on sys.path)
  manifest.json                  provenance, pool, strict CV numbers, temporal readout spec
```

`data/reference/clot_gnn_locked.json` -> `{"name": "clot_gnn_v4", "kind": "temporal_v4"}`.
`clot_gnn_v3` is superseded but untouched on disk (`docs/PHASE9_ML.md` 14 marks it
historical); load it explicitly by name if ever needed.

Both the GNN ensemble and the temporal readout are fitted on the full 19-vessel eligible
pool (SEALED never seen, asserted by a test), so there is no held-out vessel left to score
these exact weights against.  The manifest carries two different kinds of number and they
answer different questions:

```
scores_strict_cv            the numbers this whole document is about -- properly nested,
                             out-of-fold, from the CV runs that SELECTED this design
temporal_scores_in_sample   the shipped weights' own score on the vessels they were fitted
                             on -- always higher than the CV estimate, not a generalisation
                             claim (v2/v3's promotion scripts carry the same caveat)
```

`scripts/promote_clot_gnn_v4_temporal.py` fits the temporal pipeline by reusing
`scripts/eval_strict_temporal.py`'s own functions (`fit_head`, `fit_lag_model`, `tune_time`,
`tune_lag`, ...) with the whole pool as the selection set and `EXTERNAL_SET` populated from
the shipped ensemble's own committed set -- the exact procedure validated in 10 and 15, not
a re-derivation of it.  One thing needed a deliberate override: with the default 3-way inner
split the promotion's own tuner picked `burden_gate=None` (i.e. never trust the learned
lag) -- a single 3-fold split of 19 vessels is well inside the project's own noise floor
(+-0.091 off-wall, 2), and one noisy split should not overrule the 5-outer-fold evidence in
15 that the ODE-anchored lag helps on average.  Matching the inner split count to the
outer-fold count the design was actually validated with (5, not 3) recovers the expected
`burden_gate=0` (always trust it) with the same procedure -- not a hand override.

Loading, variant- and kind-aware:

```python
from src.clot_ml.locked import load_default, predict_default_series
bundle, kind = load_default()                              # kind == "temporal_v4"
out = predict_default_series(bundle, kind, data, times)     # {score, mask, onset, series}
```

`times` is used directly as the evaluation grid -- the time-resolved transport field
(`mat_adv_t`) is solved fresh for exactly the requested times (a few sparse linear solves,
CPU-only, ~20 s/vessel), so this is not restricted to the 11-point grid training used.
Smoke-tested end to end: `patient020` (in-pool, in-sample, mean wall 0.98 / off 0.84 --
expected high) and `patient001` (**SEALED**, mask-growth sanity only, never scored: 0 to
295 committed nodes over the horizon, monotone).

Direct ensemble access (`load_ensemble`, `sample_for_ensemble`, `predict_scores`) still
works unchanged for the final-time-only, no-schedule use case; `sample_for_ensemble` checks
the 69-vs-56 feature width so a v4 member can never silently consume a v3-shaped sample.

## 9. REPRODUCE

```bash
python scripts/build_clot_ml_cache_v4.py --out outputs/clot_ml_cache_v5  # +13 ch, ~20 s
python scripts/build_temporal_transport.py                # per-time advected field
python scripts/run_phase9_cv.py --tag v5a --cache v5 --folds 5 --seeds 3
python scripts/run_phase9_cv.py --tag v5b --cache v5 --folds 5 --seeds 3 --rounds 5
python scripts/run_phase9_cv.py --tag v5c --cache v5 --folds 5 --seeds 3 --off-mult 2.5
python scripts/eval_strict.py          --tags v5a,v5b,v5c --cache v5          # FINAL
python scripts/eval_strict_temporal.py --arms v5a,v5b,v5c --cache v5          # + mean/time
python scripts/eval_significance.py    --a cv5a,cv5b,cv5c --b v5a,v5b,v5c --cache gt
python scripts/eval_expected_score_readout.py --tags v5a,v5b,v5c --cache v5        --save-masks outputs/v4_set_masks.npz                   # the v4 readout (10)
python scripts/eval_strict_temporal.py --arms v5a,v5b,v5c --cache v5        --head-seeds 4 --set-masks outputs/v4_set_masks.npz     # mean-over-time + final
python scripts/eval_fusion_calib.py --tags v5a,v5b,v5c --cache v5   # 10.1 + the negatives
python scripts/promote_clot_gnn_v4.py                     # lock 9 GNN members, ~19 min
python scripts/promote_clot_gnn_v4_temporal.py --repoint  # + temporal head, ~7 min, SHIPS IT
python -m pytest src/tests/test_transport_and_strict.py -q
python scripts/diag_readout_ceiling.py --tags cv5a,cv5b,cv5c --cache gt
python scripts/eval_calibration_rules.py --tags cv5a,cv5b,cv5c --cache gt
python scripts/eval_reg_readout.py     --tags v4b --cache v4
```

One CV run is ~24 min on a 4 GB card (5 folds x 3 seeds); `--rounds 5` is ~35 min.

| file | what it is |
|---|---|
| `src/clot_ml/transport.py` | COMSOL's advection operator, discretised and solved |
| `src/clot_ml/calibration.py` | the five per-vessel readout rules of §4 |
| `scripts/eval_strict.py` | strictly-nested final-time evaluation |
| `scripts/eval_strict_temporal.py` | nested head + thresholds, mean-over-time and final |
| `scripts/eval_multiarm.py` | per-domain arm selection (§6) |
| `scripts/eval_significance.py` | paired bootstrap and the seed floor (§2) |
| `scripts/promote_clot_gnn_v4_temporal.py` | fits + ships the temporal head (§8b) |
| `src/clot_ml/locked.py` | `load_default`/`predict_default_series`; v4 kind `temporal_v4` |
