# THE STANDARD DEBUG VIZ — read this before building a new one

Written 2026-08-16. If a human says "viz", "visualize", or "let's look at this vessel" for
any wall/lumen clot model in this repo, **extend this template, do not start from a blank
canvas.** It was arrived at over several iterations (a static PNG gallery, an overlay
diff, a two-line-per-chart temporal report) and each iteration was replaced because it was
harder to read than the one before it — the shape below is the one that survived.

Reference implementation (zero-parameter physics model): `scripts/gen_offwall_temporal_data.py`
(data) + `scripts/build_offwall_temporal_artifact.py` (HTML/JS). Reference implementation
for a learned model with a real held-out set, including the FIT/DEV badging in point 7
below: `scripts/gen_gnn_viz_data.py` + `scripts/build_gnn_temporal_artifact.py`
(`clot_gnn_v1`). Reference implementation for a learned model with NO held-out set left
(point 8 below) and genuine per-timestep predictions in both windows:
`scripts/gen_v3_temporal_data.py` + `scripts/build_v3_temporal_artifact.py` (`clot_gnn_v3`).
Run the relevant pair, then publish the resulting `outputs/*.html` file as an Artifact
(Claude Code's hosted-page tool). Note this is unrelated to
[docs/PUBLISHING.md](PUBLISHING.md), which is this repo's git tracking policy — nothing
under `outputs/` is meant to be committed.

## The shape

1. **Two plain windows, side by side: Model | Ground truth.** Not an overlay, not a
   TP/FN/FP diff. Diffing was tried first and was harder to read at a glance than two
   plain pictures on the same clock — a human can eyeball two patterns faster than they
   can decode four overlaid colors on one.
2. **A time slider with play/pause**, scrubbing through the vessel's real simulated
   seconds (not an abstract frame index). Both windows update together.
3. **Synced zoom/pan across both windows.** One shared `{k, panX, panY}` transform state;
   scroll to zoom (centered on cursor), drag to pan, pinch on touch, double-click or a
   Reset button to snap back. Reset on vessel change (different geometry), preserved
   across time frames (the point is to zoom into a region and watch it evolve).
4. **Shape encodes location, color encodes identity.** Circle = wall node, square =
   off-wall/lumen node. One color per window (not per class) — model window uses the
   accent hue, GT window uses a second hue — and off-wall nodes fade toward a lighter/
   desaturated version of that same hue as their wall-normal distance increases
   (`physics_lumen_model.wall_normal_projection`, normalised by ~1.5 median edge
   lengths so the real off-wall clot shell spans the visible gradient instead of
   clustering at one end).
5. **The canonical deploy score, live, at every timestep — not just the final mask.**
   Domain-restricted: run `compute_clot_relaxed_metrics` + `clot_score_from_deploy_dict`
   twice, once with both prediction and GT zeroed outside the wall, once zeroed outside
   the wall (i.e. off-wall only). Show the current numbers as big stat cells and as a
   curve with a synced time cursor. Never report one blended full-mesh number when two
   domain scores are cheap and more diagnostic.
6. **State plainly which curves are diagnostic vs shipped.** The lumen arm has no time
   axis in the shipped model (`grow_into_lumen` is a static geometric rule) — the lumen
   onset curve in the reference build is a diagnostic extension (min-time flood fill from
   the wall onset field), verified to reproduce the shipped final mask before timing is
   attached. Say so in-page, every time. This project's credibility rests on never
   letting a diagnostic quietly read as a shipped result — see `docs/PHASE3_RESULTS.md`
   §8/§9 for why that discipline exists.
7. **For any learned model, badge every vessel by split and lead with generalization.**
   Read the split straight from the model's own manifest/protocol (e.g.
   `data/reference/clot_gnn_locked.json`'s `fit_anchors`/`dev_anchors`), never hand-guess
   it, and bake the split into the JSON payload itself (a `"split"` field per vessel) so
   the page can't silently drift out of sync with the data. Order held-out vessels first,
   badge every tab, open the page on a held-out vessel by default, and if you show any
   in-sample number for context, label it as such next to the honest out-of-fold one —
   never let a trained-on vessel or an in-sample score pass as evidence of generalization.
   A SEALED set (if the model has one) is never opened for a viz — assert it in code, the
   way `scripts/gen_gnn_viz_data.py` does.
8. **If the model has NO held-out set left, say so louder, don't fake one.** `clot_gnn_v3`
   trains on its entire eligible pool by design (its held-out evidence lives only in a
   k-fold CV run whose per-vessel predictions aren't retrievable from the shipped
   artifact) — every vessel reachable locally is in-sample. In that case: (a) do not
   badge tabs FIT/DEV — badge by something real and orthogonal, like geometry class; (b)
   put the trustworthy number — the CV metric from the model's own manifest — in its own
   section ahead of the tabs, explicitly labeled as the one to trust, with the in-sample
   numbers you're about to show flagged as optimistic; (c) never call picking "generalizes
   better" vessels for the panel a substitute for actually holding data out. See
   `docs/PHASE9_ML.md` §11–14 for why the FIT/DEV split was abandoned in the first place —
   a fixed cut couldn't put the one available aneurysm on both sides of it.

## Design tokens (reuse, don't reinvent)

Light/dark-aware CSS custom properties, already in `build_offwall_temporal_artifact.py`:
`--bg --surface --surface-2 --ink --muted --line --accent` for the page chrome;
`--model-c/--model-far` and `--gt-c/--gt-far` for the two windows' near/far gradient
pairs; `--score-wall/--score-off` for the two score-curve colors. Serif for headings
(`ui-serif, "Iowan Old Style", ...`), system sans for body, mono with tabular-nums for
every number. No webfonts embedded — the payload is already images-worth of JSON, keep
CSS lean.

## Extending to a new model or vessel set

- `VESSELS` in `gen_offwall_temporal_data.py` is a `(anchor, flow_arm)` list — `flow_arm`
  is `"pred"` where the pack has `u0_pred` (deployable), else `"gt"`. Pick vessels with
  `scripts/find_offwall_vessels.py` if the question involves off-wall clot.
  `MAX_BG_POINTS` caps the faint background context layer per vessel.
- Swapping in a different predictor: replace the calls into
  `scripts/predict_wall_clot.py` (`predict_wall_clot`, `predict_wall_onset`) with whatever
  the new model's mask/onset functions are; keep the output shapes
  (`frame_gt_wall/model_wall/gt_lumen/model_lumen`, `score_wall/score_offwall` at full
  timestep resolution) so the build script needs no changes.
- Payload size: budget under ~1.5 MB per artifact (current build is ~900 KB for 5
  vessels). Full-resolution score curves cost one `compute_clot_relaxed_metrics` call per
  timestep per domain (~26 ms each measured on a 17k-node vessel) — cheap enough to not
  need frame-subsampling; only the *spatial* frames are subsampled to 13 stops.

## What NOT to do

- Don't ship a static PNG gallery for anything with a time dimension — it was the first
  iteration here and got replaced.
- Don't score a combined wall+lumen prediction against a combined GT and call it "the
  score" without also breaking it out by domain — a model can be right on the wall and
  wrong off it (or vice versa) and a blended number hides which.
- Don't skip the honesty callouts. If a curve, threshold, or timing value was built for
  the visualization rather than shipped in the model, the page says so, in the page, not
  just in this doc.
