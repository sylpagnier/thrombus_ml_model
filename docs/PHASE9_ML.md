# PHASE 9 — a physics-informed GNN for the full-mesh clot map

Opened 2026-08-17. Scope: **wall deploy > 0.9 and off-wall deploy > 0.7**, domain-restricted,
FIT / DEV protocol, **SEALED never opened**, given the **GT flow field at t=0** as an input.

> **RESULT.** Three of the four targets are met.
>
> | | FIT (out-of-fold) | DEV (held out) | target |
> |---|---|---|---|
> | wall deploy | **0.8998** | 0.8918 | > 0.9 |
> | off-wall deploy | 0.6145 | **0.8058** | > 0.7 |
> | full mesh | 0.8458 | 0.8604 | — |
>
> against the zero-parameter physics backbone at FIT 0.8584 / 0.3651 and DEV 0.8901 / 0.5051.
> **Off-wall is where the model earns its place: +0.25 FIT, +0.30 DEV.** Wall is at target on
> FIT and 0.008 short on DEV. FIT off-wall is the one clear miss, and §5 says why.
>
> A wall-specialised ensemble reaches **FIT wall 0.9040 / DEV 0.8951** if the off-wall arm is
> allowed to be a different model (legitimate — the metric is domain-restricted), at
> off-wall FIT 0.5946 / DEV 0.8168. Both readings are reported; neither is cherry-picked
> across the FIT/DEV boundary.

---

## 1. WHAT IS BEING PREDICTED, AND WHY IT IS ONE OBJECT

`docs/PHASE7_FINDINGS.md` §10.1 established that GT clot **is** `{Mat >= 2e7}` everywhere —
0.0% of clot sits below the platelet gelation step and 0.19% of high-`Mat` nodes are not
clot, on both the wall and off it. Fibrin is inert. So there is no "wall model" and "lumen
arm": there is **one scalar field and one threshold**, and every arm the project has built
(gates, hop growth, speed thresholds, owner attenuation, flux/residence, D=0 characteristics)
is a hand-written surrogate for it.

The model therefore predicts **one per-node score over the whole mesh** and is read out per
domain. Two heads share the trunk: a classifier (what the score uses) and a regression head
on `log1p(Mat/crit)` whose **additive base is the physics backbone's own `Mat`**, zero-init,
so an untrained network *is* the physics.

## 2. WHAT MAKES IT PHYSICS-INFORMED — four choices, each traceable to a measurement

**(a) The loss is the metric.** `src/clot_ml/softmetric.py` is a differentiable copy of
`0.5*dilation_IoU + 0.5*relaxed_F0.5`, computed **separately per domain**, using a noisy-OR
soft dilation that is exact on hard masks (pinned by a test). `PHASE6_RESULTS` §15.3 showed
the score is a cliff that per-node losses cannot see; every previous ML attempt here trained
on something the score does not reward. Turning this term on is worth roughly +0.13 off-wall
FIT at fixed everything else (0.43 -> 0.56 in the first paired run).

**(b) Anisotropic message passing.** Every edge carries the t=0 velocity projected onto it,
and each layer aggregates upstream and downstream neighbourhoods **separately**.
`PHASE6_RESULTS` §3.4 measured that isotropic mesh smoothing of the source makes the fit
*worse* — the non-locality is advective. An isotropic GNN encodes the wrong prior.

**(c) Flow-mediated recurrence, and only that kind.** `PHASE6_RESULTS` §21.3 is explicit
that **contact-mediated** autoregression is dead (two phases, two failures): clot appears
where the *field* is bad, not where clot already is. What is alive is flow-mediated creep.
So the recurrence feeds back occlusion, not adjacency:

```
p_k  ->  [ p_self , p_owner_wall , 1-hop mean , 2-hop mean ]  ->  p_{k+1}
```

`p_owner_wall` is the 0.16 attenuation law of `PHASE7` §3.2 written as an input channel.
Weights are shared across rounds, round 0 is seeded with the physics mask, and `rounds=1`
collapses bit-identically to the feed-forward model (pinned by a test). Truncated BPTT keeps
peak memory at one round — a 4 GB card cannot hold three at `dim=96`.

**Recurrence is the single largest architectural win:** FIT wall 0.873 -> 0.893 and DEV
off-wall 0.631 -> 0.791, same seeds, same everything else.

**(d) The physics as prior, not competitor.** The backbone's mask is an input feature, the
additive base of the regression head, and an optional blend term in the readout. The
backbone alone still beats the network on individual vessels (p035 wall 0.943 vs 0.711 in an
early run), so "do not lose to what you started from" has to be structural.

## 3. THE PROTOCOL, AND WHY THE FIRST NUMBERS WERE FICTION

FIT n=16, DEV n=3 (`src/core_physics/wall_cohort_splits.py`; 039 is T=92 and drops out),
SEALED closed. **Every FIT number is out-of-fold** (4-fold grouped CV over FIT), and readout
parameters are tuned inside each fold on that fold's own training vessels.

This matters more than usual: the first GBM read **FIT wall 0.90 / off 0.92 in-sample** and
**DEV 0.83 / 0.46**. With 19 vessels and ~0.7% positive nodes, an in-sample FIT number is
worthless.

Honest baselines under that protocol:

```
arm                          FIT wall   FIT off   DEV wall   DEV off
physics backbone               0.8584    0.3651     0.8901    0.5051
logistic regression            0.7540    0.3504     0.7922    0.4702
gradient boosting              0.8243    0.5015     0.8393    0.4479
GNN ensemble (this phase)      0.8998    0.6145     0.8918    0.8058
```

The repo's standing kill criterion — *beat ridge/GBM on the same features* — is satisfied on
all four numbers, which is the first time that has happened here.

## 4. WHAT DID **NOT** WORK (measured, so nobody repeats it)

* **Per-vessel budget from the physics mask size** (`topk`, `topk_resid`). FIT off 0.406-0.427
  against 0.428 for a plain threshold. The physics mask size does not track off-wall burden.
* **Per-vessel budget from the model's own confidence mass** (`expected`, `k = a * sum(p)`).
  FIT off 0.437 vs 0.428. Marginal, and unstable on DEV.
* **Confining off-wall predictions to the topological species shell.** FIT off **0.534 vs
  0.597** — it costs more recall than it buys precision, because the shell misses real clot
  on p005/p006/p016 (their shell-oracles are only 0.61/0.66/0.64).
* **Bigger models and longer training.** `dim=96, layers=6, 120 epochs` reads FIT off 0.428;
  `dim=64, layers=4, 80 epochs` with 3 seeds reads **0.615**. Capacity is not the constraint;
  variance is.
* **`off_mult = 5.0`** on the metric loss. Worse than 2.5 on both domains.
* **`pos_weight = 60` at `dim=96`** diverged outright (FIT 0.36).

**Seed averaging is the single cheapest win in the whole phase** — 3 seeds took FIT off-wall
from 0.43 to 0.61. Any single-seed number on this cohort should be distrusted.

## 5. WHERE THE REMAINING FIT OFF-WALL GAP IS, AND WHETHER IT IS REACHABLE

Off-wall score tracks off-wall GT **count**, because the score is F0.5-weighted and a single
false positive is fatal on a vessel with four true nodes:

```
vessel   off GT   shell-oracle   model      vessel   off GT   shell-oracle   model
p005          4         0.608    0.15       p029         14         0.948    0.00-0.71
p040          9         0.835    0.87       p016         32         0.640    0.50-0.70
p020         13         0.824    0.33-0.68  p037         35         0.887    0.52-0.67
p021         14         0.820    0.49-0.66  p012         90         0.910    0.38-0.71
p006         12         0.658    0.80-0.91  p032        120         0.988    0.42-0.72
                                            p044        122         0.940    0.78-0.85
```

The shell-oracle (`shell & (GT Mat_self >= crit)`) means FIT **0.8186** / DEV **0.9005**,
reproducing `PHASE7` §10.7 exactly. So FIT off-wall 0.7 is reachable — but p005 alone caps
the mean at 0.61 even with a perfect shell, and the four smallest-count vessels contribute
most of the shortfall. **The gap is concentrated in vessels with under ~15 off-wall nodes,
and it is a precision problem, not a mechanism problem.**

DEV (040/041/044) has 9/84/122 off-wall nodes and reaches 0.806 — consistent with that
reading, and a caution that DEV n=3 is too small to certify the target.

## 6. HOW TO REPRODUCE

```bash
python scripts/build_clot_ml_cache.py --flow gt          # ~3 min, 19 vessels
python scripts/train_clot_ml.py --arms physics,logreg,gbm
python scripts/run_phase9.py --tag rec3s  --epochs 80 --dim 64 --layers 4 --rounds 3 --seeds 3
python scripts/run_phase9.py --tag rec5s  --epochs 80 --dim 64 --layers 4 --rounds 5 --seeds 3
python scripts/run_phase9.py --tag rec3s6 --epochs 80 --dim 64 --layers 4 --rounds 3 --seeds 6
python scripts/run_phase9.py --tag rec3o  --epochs 80 --dim 64 --layers 4 --rounds 3 --seeds 3 --off-mult 2.5
python scripts/eval_readouts.py --tags rec3s,rec5s,rec3s6,rec3o --readouts thresh --verbose
python -m pytest src/tests/test_clot_ml.py -q
```

Training and evaluation are deliberately separated: `run_phase9.py` writes, for every fold,
that fold's model's score on every vessel, so readout experiments cost seconds instead of
retraining. One config is ~11-20 min on a 4 GB card.

| file | what it is |
|---|---|
| `src/clot_ml/features.py` | 56 per-node features from t=0 geometry/flow/species + the backbone |
| `src/clot_ml/gnn.py` | anisotropic message passing, two heads, physics as the regression base |
| `src/clot_ml/recurrent.py` | the flow-mediated feedback channels |
| `src/clot_ml/softmetric.py` | the differentiable deploy score |
| `src/clot_ml/fastscore.py` | exact fast copy of the metric (0.25 ms/call), pinned by a test |
| `src/clot_ml/readouts.py` | score -> mask, including the separable per-domain tuning |
| `src/clot_ml/protocol.py` | out-of-fold FIT, single-fit DEV |
| `outputs/phase9_log.jsonl` | every arm ever run, with config |

## 7. WHAT TO DO NEXT

1. **Certify on more vessels before trusting either target.** DEV n=3 and FIT off-wall n=10.
   The FIT/DEV disagreement on which off-wall model is best (`small` wins FIT at 0.610 and
   reads 0.623 on DEV; `rec3o` reads 0.595 FIT and 0.817 DEV) is larger than the effects
   being chased. This is the highest-value next step and it needs data, not modelling.
2. **Attack small-count off-wall vessels specifically** (§5). A precision-first readout
   conditioned on predicted burden, or an explicit "does this vessel have off-wall clot at
   all" gate — `PHASE6_RESULTS` §20.4 already priced a per-vessel binary at +0.032.
3. **Then, and only then, the deployable arm.** Everything here is on GT t=0 flow, as scoped.
   `PHASE7` §12.5 measured the drop to RGP-DEQ `u0_pred` at **-0.34 wall**, which is larger
   than everything in this document combined and is a Stage-A quality bug, not a physics one.
4. **SEALED remains closed.** Spend it once, on a frozen configuration, after (1).

---

## 8. THE LOCKED ARTIFACT — `clot_gnn_v2` (supersedes `clot_gnn_v1`, kept alongside it)

`clot_gnn_v1` (below) was fitted on the OLD FIT split -- 16 baseline vessels, **zero**
aneurysms or stenoses in training. Once the geometry confound in the split was found (§11),
that became the wrong thing to ship: `clot_gnn_v2` is fitted on the FULL 19-vessel eligible
pool using the three configurations validated by geometry-stratified 5-fold CV (§11.2), so
the shipped weights have actually seen the priority class.

```
outputs/clot_ml/locked/clot_gnn_v2/     9 members (3 configs x 3 seeds), ~6 MB
data/reference/clot_gnn_locked.json     repointed to v2; v1 files untouched
```

```python
from src.clot_ml.locked import load_ensemble, predict_scores
ens = load_ensemble()                 # follows the pointer -> now clot_gnn_v2
score = predict_scores(ens, sample)
```

**There is no vessel left to score v2's shipped weights against out-of-fold** -- it trains
on the whole pool by design. The manifest instead carries the CV numbers that *selected*
these three configurations (§11.2, `CV_SCORES_OUT_OF_FOLD` in
`scripts/promote_clot_gnn_v2.py`), which is the honest generalisation estimate:

```
                 ALL wall  ALL off   baseline wall  baseline off  priority wall  priority off
severity CV       0.9198    0.7270      0.9202         0.6935        0.9177        0.8388
```

Loaded and re-scored end to end from the saved v2 weights (in-sample, since the training
pool has no holdout): ALL wall 0.9595 / off 0.8422, priority wall 0.9729 / off 0.9366 --
consistent in direction with the CV numbers and not to be quoted as a generalisation claim.

Promotion script: `scripts/promote_clot_gnn_v2.py` (resumable, same pattern as v1's).

---

### 8.1 `clot_gnn_v1` (superseded, kept for reference)

```
outputs/clot_ml/locked/clot_gnn_v1/     15 members (4 configs x seeds), 9 MB
  member_<cfg>_s<seed>.pth              weights + the config that produced them
  feature_norm.npz                      the FIT feature mean/std and column order
  manifest.json                         provenance, splits, scores
data/reference/clot_gnn_locked.json     the pointer other code reads
```

Every member is fitted on **all of FIT**; DEV and SEALED were never seen (asserted by a
test). Load and run it by name:

```python
from src.clot_ml.locked import load_ensemble, predict_scores
ens = load_ensemble()                 # follows data/reference/clot_gnn_locked.json
score = predict_scores(ens, sample)   # [N] per-node probability
```

Re-scored end to end from the saved weights, thresholds tuned on FIT:

```
                 FIT wall  FIT off   DEV wall  DEV off
legacy metric     0.9625*   0.8144     0.8989   0.7411
severity metric   0.9663*   0.8387     0.9058   0.7634
```

`*` FIT is **in-sample** for the locked members. The honest out-of-fold FIT numbers are §0.
DEV is genuinely held out and clears **both** targets under the severity metric.

Promotion is resumable (`scripts/promote_clot_gnn.py`) — a CUDA fault lost 13 members once.

---

## 9. THE METRIC, REDESIGNED — `clot_severity_score`

`src/clot_ml/severity_metric.py`. The shipped score got the *shape* half right: the 2-hop
relaxation is exactly "off by a couple of nodes is fine". What it got wrong is that a miss
is a **rate**, so

```
missing  5 of  15 off-wall nodes   ->  recall 0.667
missing 50 of 150 off-wall nodes   ->  recall 0.667      <- same number, different failure
```

and, worse, low-burden vessels are punished hardest: on a 4-node vessel one false positive
costs more than 30 do on a 120-node one. §5 traced the off-wall shortfall to exactly those
vessels.

### 9.1 The change, in one line

**An absolute grace of a few nodes, capped at a fraction of the true burden.**

```
tau_eff    = min(tau_abs, rho * n_gt)                     tau_abs = 5, rho = 0.25
recall_eff = min(1, TP_relaxed / max(n_gt - tau_eff, 1))
```

`tau_abs` says a handful of nodes either way is not a real failure; `rho` stops that grace
from swallowing a small vessel whole. Precision gets its own, much smaller grace relative to
what was *predicted*, so spraying is still punished in proportion to the spray. Worked:

```
n_gt =  15, found 10  ->  recall_eff 0.889
n_gt = 150, found 100 ->  recall_eff 0.690
n_gt =   4, found  1  ->  recall_eff 0.333     <- cannot be gamed by predicting almost nothing
```

`score = 0.5*dilation_IoU + 0.5*F_0.5(precision_eff, recall_eff)` — the same structure, so
the change is auditable, and `tau_abs = rho = 0` reproduces the old score **exactly**
(pinned by a test). A differentiable copy (`soft_severity`) is wired into the trainer as
`--metric severity`, so "the loss is the metric" survives the redesign.

### 9.2 Properties, all pinned by tests

Predicting nothing scores 0 at every burden (the empty-prediction hole, inverted); adding a
true positive never lowers the score and a false positive never raises it; flooding the mesh
scores < 0.25; the soft form matches the hard one to 2e-3 on binary input; the config is
frozen and serialisable.

### 9.3 What it changes on the real cohort — deliberately little

```
                 FIT wall  FIT off   DEV wall  DEV off
legacy            0.8860    0.6104     0.8899   0.7963
severity          0.9098    0.6369     0.8998   0.8031
```

And it is **insensitive to its own tolerances** — sweeping `tau_abs` 0 -> 12 and `rho`
0.15 -> 0.35 moves FIT off-wall only 0.627 -> 0.641:

```
tau_abs      0      3      5      8     12
FIT off   0.627  0.636  0.637  0.639  0.641
```

That is the result worth keeping. **The severity idea is right in principle and is not where
this model's loss comes from.** The vessels that drag the off-wall mean (p005 0.11, p012
0.38, p032 0.39) are not "missed a few" — they are predictions in the *wrong place*, which
the new metric correctly still punishes. p005 moves 0.106 -> 0.119, not to 0.7.

The one large per-vessel move is p028 wall 0.594 -> 0.773, a genuine "missed a handful on a
55-node vessel" case, which is exactly what the redesign was for.

---

## 10. GEOMETRY CLASSES — measured, not hard-coded

`src/clot_ml/geometry_class.py`. Stenoses and aneurysms are the priority class. Rather than
hard-code the designated list, the class is measured from the mesh's own lumen width, over
wall nodes more than 12 hops from the inlet/outlet, locally smoothed along the wall:

```
aneurysm   bulge     = p98(width)/median >= 2.00   ->  039 (3.48), 043 (2.83), 040 (2.57)
stenosis   narrowing = p2 (width)/median <= 0.40   ->  041 (0.281), 042 (0.292), 044 (0.323)
baseline   everything else; nearest are 016 (bulge 1.68) and 012 (narrowing 0.441)
```

**That is the designated set exactly — 6 of 6, both sub-classes right, with margin** (a test
pins it). Both cuts sit in a real gap, so the rule should transfer to an unlabelled vessel.

`width_nd` is **unusable on 9 of 34 vessels** (001/010/011 read a constant 1.0 with a 10x
spike; 003/004/005/006/007/008 read ~0.12). Those return `"unknown"` and fall back to the
explicit designation rather than being confidently mislabelled. Fixing that channel is a
small, well-defined data job that would complete the classifier.

### 10.1 The consequence that matters most

**DEV (040/041/044) is entirely priority-class; FIT is entirely baseline.**

So every FIT-vs-DEV difference in this document is confounded with a geometry-class
difference, and neither split can certify the other. Concretely: the model reads DEV off-wall
0.80 and FIT off-wall 0.64, and that gap is *not* evidence it generalises — it is a
comparison between six aneurysm/stenosis vessels and sixteen baseline ones. Two of the
priority vessels (042, 043) are in SEALED.

**Recommendation: re-cut the splits so both classes appear on both sides**, before spending
SEALED. As it stands the priority class has n=3 outside SEALED, which is not enough to
certify a tiebreaker on.

---

## 11. THE SPLITS, RE-CUT BY GEOMETRY — and what stenosis/aneurysm performance actually is

`src/clot_ml/geometry_splits.py`, `scripts/run_phase9_cv.py`, `scripts/eval_by_class.py`.

### 11.1 Why a fixed re-cut is impossible, and what replaces it

Of the 19 eligible vessels outside SEALED, exactly **three** are priority class:
`040` (aneurysm), `041` (stenosis), `044` (stenosis). `039` is an aneurysm but T = 92 and a
truncated run is a different quantity; `042` (stenosis) and `043` (aneurysm) are in SEALED
and stay there.

**With one non-SEALED aneurysm, no fixed FIT/DEV cut can put an aneurysm on both sides.**
Put `040` in FIT and aneurysm generalisation is never measured; put it in DEV and the model
never trains on an aneurysm. That is a property of the data, not of the split.

So the fixed cut is replaced by **geometry-stratified 5-fold** over the whole eligible pool.
Each priority vessel lands in a different fold, so every one of them is *trained on* in four
folds and *measured* in one:

```
fold 0  012 021 032 040[aneurysm]     trains on 041,044  -> no aneurysm in training
fold 1  016 024 035 041[stenosis]     trains on 040,044  -> both classes
fold 2  018 025 036 044[stenosis]     trains on 040,041  -> both classes
fold 3  005 019 028 037               trains on all three
fold 4  006 020 029                   trains on all three
```

Four of five folds train on both classes; every vessel gets an out-of-fold score.

### 11.2 Results, out-of-fold, severity metric (3-model ensemble, `cv5a+cv5b+cv5c`)

```
group                  n | wall model  wall physics | off model  off physics   target
ALL                   19 |   0.9198      0.8766     |  0.7270     0.4141      0.9 / 0.7
baseline              16 |   0.9202      0.8709     |  0.6935     0.3819
PRIORITY (sten+aneu)   3 |   0.9177      0.9067     |  0.8388     0.5214
  aneurysm             1 |   0.9762      0.9702     |  0.8417     0.2893
  stenosis             2 |   0.8885      0.8750     |  0.8373     0.6374
```

**Both targets are met out-of-fold, overall and on the priority class.** Under the legacy
metric the same ensemble reads ALL wall 0.9099 / off 0.7195, so this does not depend on the
metric change.

### 11.3 The FIT-vs-DEV "generalisation gap" was the geometry confound, and it inverts

§0 reported DEV off-wall 0.806 against FIT 0.615 and flagged it as uncertifiable. It was not
generalisation: DEV *was* the priority class. With the confound removed, the same pattern
appears as a **class** effect — priority 0.8388 against baseline 0.6935 — i.e. the
stenosis/aneurysm vessels are, if anything, **easier** off-wall, not harder. They carry more
off-wall clot (84/122 nodes on the stenoses), and the off-wall score is precision-limited on
low-burden vessels (§5).

### 11.4 What the aneurysm number is and is not

`patient040` is held out in fold 0, whose training set contains `041` and `044` — **both
stenoses, no aneurysm**. So its wall 0.9762 / off 0.8417 is *cross-class* generalisation: the
model had never seen an aneurysm and scored the best wall number in the cohort on one.

That is genuinely encouraging and it is **n = 1**. It cannot support a tiebreaker. The
stenosis number (n = 2) is measured with one other stenosis in training, which is the weaker
claim but the better-supported one.

**To measure aneurysm generalisation properly there are exactly two options:** re-run
`patient039` to the full 30000 s horizon, or open SEALED (which holds `042` stenosis and
`043` aneurysm). The first is cheap and is the recommendation; the second spends the only
clean held-out evidence the project has.

### 11.5 Where the remaining miss is

`baseline` off-wall 0.6935 is the one sub-target still short, and it is two vessels:
`p005` (4 off-wall GT nodes, 0.240) and `p032` (120 nodes, 0.432). Everything else in the
baseline group is 0.63-0.90. `p005` is the precision-on-tiny-burden problem of §5; `p032` is
a large-burden failure and is the more interesting one to chase.

---

## 12. INTERMEDIATE STATES — the timing prize, re-measured under the fixed metric

### 12.1 The metric was the problem, again

PHASE6's "every timing arm nets zero" was measured under a deploy score whose
empty-prediction cliff (predict nothing = 1.0 while GT is empty) dominated mean-over-time.
`SeverityScorer` returns **NaN** on empty GT, so those timesteps are skipped and the cliff
is gone.  All 19 vessels have GT-empty grid points (median 3 of 11), so this is not
academic.  With it fixed (`scripts/eda_temporal_metric.py`):

```
 t/T   GT frac | predict nothing | ORACLE FINAL mask replayed
0.10     0.008 |      0.0000     |        0.1148
0.30     0.166 |      0.0000     |        0.4232
0.50     0.592 |      0.0000     |        0.8246
1.00     1.000 |      0.0000     |        1.0000
                        mean-over-time:    0.8224
```

Both degenerate extremes are punished, and **a frozen-mask model tops out at 0.822** even
with the perfect final answer.  **No new metric is needed** -- mean-over-time deploy score,
per domain, GT-empty skipped.

### 12.2 Timing is the dominant prize, not a null result

`scripts/eda_timing_prize.py` / `scripts/eval_temporal_arms.py`, mean-over-time:

```
arm             wall      off     what it is
frozen        0.7921   0.5015     the locked GNN mask, replayed  (ships today)
ode_set       0.8247   0.5015     the ODE's own set AND timing
gnn_ode       0.8579   0.5015     GNN set, ODE timing on the wall   <- FREE +0.066
gnn_ode_off   0.8579   0.4899     ... + off-wall owner-threshold timing
oracle        0.9897   1.0000     perfect timing
```

Priority class: frozen 0.8056 -> `gnn_ode` **0.8603**.

* **Crude timing beats a perfect frozen set.** `ode_set` (0.8247) beats the *oracle final
  mask* replayed (0.8190): committing everything at t=0 is worse than committing the wrong
  things at roughly the right times.
* **The GNN set + the ODE's timing is worth +0.066 on the wall for zero training**, and it
  beats the ODE's own set, so the two halves are separable -- the GNN supplies *where*, the
  ODE supplies *when*.  Shipped as `src/clot_ml/temporal.py`.
* **Off-wall timing does not work.** The owner-threshold rule (an off-wall node crosses when
  its owner reaches `crit/att`) is right in principle -- the attenuation is stable in time
  (0.004 -> 0.003 over the horizon) -- but scores 0.4899 against frozen's 0.5015, and a
  sweep over `att` 0.16 / 0.30 / 0.50 / 0.80 reads 0.490 / 0.494 / 0.459 / 0.510.  The best
  value is the degenerate one where off-wall fires with its owner.  Cause: the ODE's `Mat`
  is biased low (ratio 0.602 at the shipped `da_scale`), so `crit/att` is simply unreachable
  and the rule collapses.  **Off-wall timing is gated on the same Mat-magnitude bottleneck
  as everything else off-wall.**

Remaining: wall 0.858 -> 0.990 (+0.13), off-wall 0.490 -> 1.000 (+0.51).

### 12.3 The non-locality is partly a local operator, but NOT a clean mass matrix

PHASE7 9.2 read a 0.310 rank ceiling for a per-node ODE on perfect inputs and called it
missing transport.  At the wall `u = 0` and `D_Mat = 0`, so no physical mechanism moves
`Mat` between wall nodes; the suspect was COMSOL's **consistent mass matrix** plus crosswind
stabilisation, which would make the coupling a fixed *local linear* operator rather than a
transport closure.  `scripts/eda_mass_matrix.py` tests it both ways:

```
                                   local    +1-hop mix
forward   f ~ a*Mat + b*(A Mat)   R2 -0.923   -0.778     b/a positive on 12/18 vessels
inverse   rank(flux, GT Mat)      0.308        0.402     mean optimal smoothing w = 0.71
```

* The local rank of 0.308 **reproduces PHASE7's 0.310**, so the pipelines agree.
* **Smoothing the flux integral over 1-hop wall neighbours lifts the rank +0.094** (~30%
  relative).  Local neighbour information the per-node ODE discards is demonstrably there.
* **But the mass-matrix form does not fit.**  Forward R2 stays strongly negative and `b/a`
  flips sign on 6 of 18 vessels with magnitudes from -1.8 to +2.1.  A two-term `M` is not
  the right object.

**Conclusion: let a GNN learn the local operator; do not hard-code `M^-1`.**  The prior
(local message passing on top of the physics flux integral) is validated and worth ~+0.09
rank; the specific derivation is not.  Note also that the vessels with *negative* rank
(p019, p024, p025, p028, p036) are the saturated ones -- a single relation across all nodes
is wrong, and regime is a real latent variable.

### 12.4 The curve head: FEASIBILITY CHECKED, and it does not ship

Before building the monotone curve head proposed in 12.3, three things had to be true.
`scripts/eda_curve_head_feasibility.py` and `scripts/eda_curve_param_predictability.py`
measure all three.  Two hold; the third does not.

**(1) The ceiling is timing, not the set.**  Perfect timing on OUR OWN locked-GNN set:

```
arm            wall      off
frozen       0.7921   0.5015
oracle_time  0.9875   0.9715     <- on the GNN set, not the GT set
```

Essentially the global oracle (0.9897 / 1.0000).  **The committed set is not the limit.**
The whole remaining prize -- +0.14 wall, +0.47 off-wall -- is timing.

**(2) The two-phase family is expressive enough.**  `Mat(t) = r0*min(t,c) + r1*max(t-c,0)`,
with `r0` fixed from t=0 physics and `(c, r1)` fitted on GT: trajectory R2 **0.809**, onset
rank vs GT **+0.742** (against the ODE's +0.642).  Scored end to end with GT-fitted
parameters it reaches **off-wall 0.8257** against frozen's 0.5015.

Two side results worth keeping.  `r0` alone -- pure first-order deposition, no autocatalysis
-- **never reaches `crit` at all** in the horizon, so it scores 0.0000: the autocatalytic
ramp is not a correction, it is the entire mechanism.  But `r0` *orders* onset as well as
the full ODE does (rho +0.640 vs +0.642), so ordering is available free from t=0 while
scale is not.

**(3) The parameters are NOT predictable from t=0.**  GBM on the same 56 features,
leave-one-vessel-out:

```
arm            wall      off      what is oracle
frozen       0.7921   0.5015
predicted    0.7840   0.3879     nothing        <- LOSES to doing nothing
oracle_c     0.7715   0.4021     c
oracle_r1    0.7109   0.6442     r1
gt_params    0.7890   0.8257     both

out-of-fold   rho(c_hat, c) = +0.468     rho(r1_hat, r1) = +0.188
```

* **The deployable head is worse than frozen on both domains.**  Do not build it as
  specified.
* **`r1` is the blocker, not `c`.**  Handing it `r1` recovers 0.388 -> 0.644 off-wall;
  handing it `c` recovers only 0.388 -> 0.402.
* **`rho(r1_hat, r1) = 0.188` is `rho_corner = 0.193` again.**  The late accumulation rate
  and the `Mat` magnitude are the same unknown wearing different clothes.
* The family also does not help the **wall** even with perfect parameters (0.7890 against
  the ODE's 0.8579), so wall timing should stay with the ODE.

### 12.5 What this settles

Every route to off-wall timing tried so far -- owner attenuation (12.2), the two-phase curve
head (12.4), direct late-rate regression (rho 0.04 from t=0) -- terminates at the same
place: **predicting the `Mat` magnitude field, currently rho ~ 0.19 on species-carrying
nodes.**  It is not three problems, it is one.

So the honest plan is:

1. **Ship the wall timing win** (`src/clot_ml/temporal.py`, +0.066 wall, zero training).
2. **Do not build a temporal head.**  A new time parameterisation cannot route around a
   magnitude problem; three now have failed the same way.
3. **Attack the magnitude field directly.**  It is the single lever: it unlocks off-wall
   mask (PHASE7 10.2), off-wall timing (12.4), and the remaining wall timing at once.  The
   one validated architectural hint is 12.3 -- local message passing on the physics flux
   integral is worth +0.094 rank, and the operator should be learned, not derived.

---

## 13. THE MAGNITUDE BOTTLENECK WAS ALREADY BEATEN — and a learned onset head works

### 13.1 12.5 was wrong: `rho_corner` is 0.592, not 0.193

PHASE7 quotes `rho_corner = 0.193` -- the rank of predicted vs GT wall `Mat` on
species-carrying corner nodes -- as *the* bottleneck, and 12.5 concluded every timing route
terminates there.  That number is for the **physics ODE**.  The GNN's magnitude field was
never measured, because the deploy score reads its classifier and nobody looked at the
regression head.  Measured out-of-fold on the geometry-stratified folds:

```
                              rho_corner (out-of-fold)
physics ODE  Mat                       0.311
locked GNN                             0.592      priority class 0.733
```

(In-sample, the never-used regression head reads 0.619 and the classifier 0.601.)

**The magnitude field is roughly twice as good as the docs assume.**  12.5's "it is one
problem and we are stuck at 0.19" is retracted.

### 13.2 Rescaling the ODE by that magnitude FAILS, for a structural reason

The obvious synthesis -- rescale each node's ODE trajectory so its final value matches the
GNN's magnitude, then take crossings -- is worse than doing nothing
(`scripts/eda_rescaled_ode.py`, wall **0.701** against the ODE's 0.842).

The reason is worth recording: **where the ODE flashes, the trajectory jumps from ~0 to
large in a single step, so changing the threshold cannot move the crossing time.**  Timing
spread must come from a per-node *ordering*, not a per-node *threshold*.  Any future
"calibrate the magnitude and re-threshold" idea dies here.

### 13.3 What works: a learned onset head over the physics orderings

`scripts/train_onset_head.py`.  Three independent orderings of onset already exist and were
never combined -- `r0` the t=0 physics rate (rank +0.640), the ODE's crossing (+0.642), and
the GNN magnitude (0.592).  A static per-node regression on the 56 features plus those three,
leave-one-vessel-out, predicts GT onset; the predicted **ordering** is then mapped onto the
ODE's own time distribution, so the schedule stays physical and only the order is learned.

```
MEAN-OVER-TIME, out-of-fold, same GNN set (n=19)
arm            wall       off
frozen       0.7953    0.4209
ode          0.8422    0.4209
learned      0.8547    0.5369        onset rank +0.755 vs the ODE's +0.642
oracle       0.9705    0.8396

priority class (n=3)
frozen       0.8009    0.4481
ode          0.7886    0.4481        <- the ODE HURTS here
learned      0.8657    0.7060        <- +0.26 off-wall
```

* **The first off-wall timing arm that beats frozen**: +0.116 overall, **+0.26 on the
  priority class**, where every previous attempt lost.
* It also beats the ODE on the wall (+0.013) and rescues the priority class where the raw
  ODE is a regression (0.7886 -> 0.8657).
* No recurrence and no backprop through the stiff ODE -- the two things that killed the
  earlier attempts.  This is the ordering/calibration split PHASE7 7.2 predicted: learn the
  order, take the schedule from physics.

**One bug worth recording** because it looked like a null result: ranking the prediction over
all ~15k nodes instead of within the mask puts every masked node at the bottom of the global
order, maps them all to the earliest ODE time, and collapses the arm to `frozen` exactly
(0.7953) *despite* a rank of +0.755.  Rank within the mask.

### 13.4 Status and what is left

```
                    wall      off
shipped (12.2)    0.8579    0.5098        in-sample locked weights
learned head      0.8547    0.5369        out-of-fold
oracle timing     0.9705    0.8396        out-of-fold ceiling
```

The learned head is not yet promoted -- it is validated out-of-fold but the shipped
artifact still carries ODE-only timing.  Promoting it means fitting the onset regressor on
the full pool and stamping it into `clot_gnn_v2` alongside the existing temporal block.

Remaining: **+0.12 wall, +0.30 off-wall** to the timing ceiling.

### 13.5 Pushing v2: three physics constraints, all negative, and the decomposition that explains why

`scripts/train_onset_head_v2.py`.  v1's per-vessel breakdown showed the arm was
**bit-identical to frozen** on p020/p021/p029/p032/p037, so three physically-motivated
constraints were added.  All three lose:

```
arm                      wall        off
frozen                 0.7953     0.4209
learn + ODE schedule   0.8547     0.5369     <- v1, still the best
cohort growth curve    0.7734     0.5010
  + follow-owner       0.7734     0.4898
  + 1-hop smoothing    0.7806     0.4924
oracle                 0.9705     0.8396
```

Replacing the ODE's onset histogram with a **cohort growth curve** costs -0.081 wall.  The
lesson is precise: the ODE's contribution is **not its ordering** (rank +0.642, worse than
the learned +0.755) -- it is its **per-vessel time calibration**.  A pooled cohort curve
forces every vessel onto the same growth shape and throws that away.  `follow-owner` and
1-hop smoothing then add nothing on top.

### 13.6 Ordering is done; the schedule is the whole remaining lever  [ARM WAS MISLABELLED -- see 13.8]

Crossing the two halves independently, out-of-fold:

```
arm                    wall        off
frozen               0.7953     0.4209
learn + ODE sched    0.8547     0.5369
GT order + ODE sched 0.8522     0.4116     <- PERFECT ordering, no better
learn + GT sched     0.8908     0.6619     <- schedule fixed, ordering as-is
oracle               0.9705     0.8396
```

**Perfect ordering buys nothing** (-0.003 wall, -0.126 off).  Fixing only the schedule buys
**+0.036 wall / +0.125 off**.  So the +0.755 ordering we already have is sufficient, and
every further ordering effort is wasted until the schedule improves.  (Ordering does pay
*after* that: oracle - learn+GTsched is still +0.080 / +0.178.)

### 13.7 But the schedule is not one or two numbers

If the schedule were a location/spread problem it would be a 19-sample regression.  It is
not.  Using **oracle** per-vessel scalars to shift or affine-map the ODE schedule onto GT's
median and IQR:

```
arm                    wall        off
learn + ODE sched    0.8547     0.5369
+ oracle shift (1)   0.8070     0.5471     <- worse on wall
+ oracle affine (2)  0.8280     0.5659     <- still worse
+ full GT quantiles  0.8908     0.6619
```

Both **hurt the wall even with oracle parameters.**  The ODE schedule's *shape* is doing the
work; matching GT's median and IQR destroys it.  The remaining +0.036/+0.125 therefore lives
in the higher moments of the per-vessel onset distribution -- a full quantile function per
vessel, from 19 vessels.

**Verdict for this round.** v1 (`learn + ODE schedule`) stands as the best temporal arm and
is what should be promoted.  The next real gain requires predicting a per-vessel onset
*distribution shape*, which is a genuinely harder target than anything attempted here and
should not be started without more vessels.  Do not spend further effort on onset ordering,
on cohort-average schedules, or on 1-2 moment schedule corrections -- all three are measured
dead.

### 13.8 CORRECTION to 13.6, and the pivot that broke the plateau

**13.6's "GT order" arm was mislabelled.** It passed the **ODE's** onset as the ordering,
not GT's, so it measured the plain ODE arm and "perfect ordering buys nothing" was never
actually tested.  Re-run correctly (`scripts/eda_onset_decomposition.py`):

```
arm                    wall        off
frozen               0.7953     0.4209
ode                  0.8422     0.4209
learn_rank           0.8547     0.5369
learn_abs            0.8369     0.5222      learned onset used directly, no quantile map
learn_first          0.8452     0.5418      ODE schedule shifted to GT's FIRST commit
gtorder_ode          0.8568     0.5457      <- the REAL perfect-ordering test
learn_gtsched        0.8908     0.6619
oracle               0.9705     0.8396
```

The conclusion **survives the correction**: perfect ordering with the ODE schedule is worth
only +0.002 wall / +0.009 off over the learned ordering.  Also new: `learn_abs` is *worse*
than rank-mapping, so the ODE schedule is carrying real information, and `learn_first`
(PHASE6 15.4's best single lever under the old metric) is now marginal.

### 13.9 Dropping the ORDER x SCHEDULE factorisation

Every arm above factorises as "rank the nodes, then map the ranks onto some reference time
distribution".  With ordering saturated and the schedule uncarryable by moments or by a
pooled curve, the factorisation itself was the constraint.

`scripts/train_time_conditioned.py` drops it: **time becomes an input feature** and the model
predicts `P(node is clot at time t)` directly, so the schedule emerges from the features
instead of being imposed.  Physics stays in the formulation rather than in a post-hoc map --
the ODE's own answer at time `t` and the t=0 rate `r0` are features, so the model can agree
with the physics schedule where it is right and depart where it is not, and predictions are
made **monotone in time** by a cumulative maximum (clot does not un-clot; the production law
has no sink).

```
arm                    wall        off
frozen               0.7953     0.4209
learn_rank  (13.3)   0.8547     0.5369
timecond_mono        0.8838     0.5959
timecond_tuned       0.8845     0.6110      <- per-domain thresholds, tuned on train folds
learn_gtsched        0.8908     0.6619      <- ORACLE-schedule reference
oracle               0.9705     0.8396

priority class (n=3)
learn_rank           0.8657     0.7060
timecond_tuned       0.9047     0.7943      <- BEATS the oracle-schedule arm (0.7864)
learn_gtsched        0.9144     0.7864
```

* **wall +0.030 and off-wall +0.074 over `learn_rank`**, out-of-fold.
* It closes **92% of the wall gap and 93% of the off-wall gap** to the oracle-schedule
  reference -- i.e. it recovers nearly all of the schedule information that no imposed
  reference could supply, with no oracle.
* On the **priority class it exceeds the oracle-schedule arm** (off 0.7943 vs 0.7864) and
  takes wall past **0.90**.

Cumulative against the mask that shipped before any temporal work: **wall 0.7953 -> 0.8845
(+0.089), off-wall 0.4209 -> 0.6110 (+0.190).**

Remaining to true oracle: +0.086 wall, +0.229 off-wall.

---

## 14. THE LOCKED ARTIFACT — `clot_gnn_v3` (supersedes `clot_gnn_v2`, kept alongside v1/v2)

`clot_gnn_v3` ships the time-conditioned model from 13.9.  It does not retrain the GNN: it
reuses the locked `clot_gnn_v2` ensemble as the committed SET (v2's classifier score is one
of its input features) and adds a small gradient-boosted head that reads
`{v2's 56 features, ODE onset time, t=0 rate r0, v2 score, query time, whether the ODE has
fired by that time} -> P(clot now)`, trained on the full 19-vessel eligible pool.

```
outputs/clot_ml/locked/clot_gnn_v3/     model.pkl (classifier) + manifest.json, ~200 KB
data/reference/clot_gnn_locked.json     repointed to v3; v1/v2 untouched on disk
```

```python
from src.clot_ml.locked import load_default, predict_default_series
bundle, kind = load_default()                       # follows the pointer; kind="temporal_v3"
out = predict_default_series(bundle, kind, data, times)   # {score, mask, onset, series}
```

`scripts/predict_clot_temporal.py` now calls this dispatcher instead of the v2-only path, so
it automatically runs whichever generation is shipped without the caller choosing.

**Two physical constraints are enforced on the raw per-time output**
(`enforce_owner_and_monotone`, pure and unit-tested without loading any weights):
predictions are forced monotone in time (the production law has no sink), and an off-wall
node is never predicted clot before its owner wall node is.

**There is no vessel left to score v3 against out-of-fold** -- like v2, it trains on the
whole pool by design.  The manifest carries the CV numbers that *selected* this design
(13.9), which is the honest generalisation estimate:

```
                 ALL wall  ALL off   priority wall  priority off
out-of-fold CV    0.8845    0.6110       0.9047        0.7943
```

Smoke-tested end to end from the saved v3 artifact on `patient020` (in-pool, so in-sample
and expected to read high: wall 0.9907 / off 0.9365) and on the SEALED `patient001` (mask
only, growing monotonically 0 -> 202 wall / 0 -> 3 off nodes over the horizon -- SEALED is
never scored).

Promotion script: `scripts/promote_clot_gnn_v3.py` (deterministic; loads v2 rather than
retraining it).  `test_v3_manifest_is_consistent_and_excludes_sealed` and
`test_locked_ensemble_manifest_is_consistent` both check the artifact actually on disk.
