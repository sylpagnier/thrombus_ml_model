"""Assemble outputs/viz_generalization/*.png + metrics into a standalone HTML report."""
from __future__ import annotations

import base64
import json
from pathlib import Path

SCRATCH = Path(r"C:\Users\pgssy\AppData\Local\Temp\claude\C--Users-pgssy-thrombus-ml-model\27b9191e-449c-42c9-b376-54782ca8e919\scratchpad")
IMG_JSON = Path("scratch_b64.json")

VESSELS = [
    # id, split, gt_score, gt_pred_n, gt_gt_n, pred_score, pred_pred_n, has_pred
    dict(a="patient014", note="best case", gt_score=0.9916, gt_pred=185, pred_score=0.9741, pred_pred=188, gt_n=181),
    dict(a="patient001", note=None, gt_score=0.9651, gt_pred=194, pred_score=0.9163, pred_pred=156, gt_n=183),
    dict(a="patient007", note="validated against raw COMSOL export", gt_score=0.9485, gt_pred=205, pred_score=0.8627, pred_pred=215, gt_n=226),
    dict(a="patient010", note=None, gt_score=0.8998, gt_pred=44, pred_score=0.8776, pred_pred=42, gt_n=54),
    dict(a="patient031", note="smallest clot, n=19", gt_score=0.8126, gt_pred=31, pred_score=0.8341, pred_pred=16, gt_n=19),
    dict(a="patient013", note="largest arm-B drop", gt_score=0.9457, gt_pred=220, pred_score=0.6753, pred_pred=113, gt_n=228),
    dict(a="patient042", note="no flow surrogate for this pack", gt_score=0.7313, gt_pred=56, pred_score=None, pred_pred=None, gt_n=109),
    dict(a="patient043", note="no flow surrogate for this pack; previous best on record was 0.6925 here", gt_score=0.9796, gt_pred=91, pred_score=None, pred_pred=None, gt_n=95),
]


def main() -> None:
    imgs = json.loads(IMG_JSON.read_text())

    cards = []
    for v in VESSELS:
        a = v["a"]
        gt_img = imgs.get(f"{a}_gt", "")
        pred_img = imgs.get(f"{a}_pred", "")
        note_html = f'<p class="card-note">{v["note"]}</p>' if v["note"] else ""

        pred_block = ""
        if v["pred_score"] is not None:
            pred_block = f"""
            <figure class="panel">
              <img src="data:image/png;base64,{pred_img}" alt="{a} predicted-flow clot map" loading="lazy" />
              <figcaption>
                <span class="tag tag-pred">deployable</span>
                <span class="score">{v['pred_score']:.3f}</span>
                <span class="meta">{v['pred_pred']} pred nodes</span>
              </figcaption>
            </figure>"""
        else:
            pred_block = """
            <figure class="panel panel-empty">
              <div class="empty-slot">flow surrogate<br/>not available<br/>for this pack</div>
              <figcaption>
                <span class="tag tag-pred tag-disabled">deployable</span>
                <span class="meta">—</span>
              </figcaption>
            </figure>"""

        cards.append(f"""
        <article class="vessel-card">
          <header class="vessel-head">
            <h3>{a}</h3>
            <span class="split-badge">sealed &mdash; never trained on</span>
          </header>
          {note_html}
          <div class="panel-row">
            <figure class="panel">
              <img src="data:image/png;base64,{gt_img}" alt="{a} GT-flow clot map" loading="lazy" />
              <figcaption>
                <span class="tag tag-gt">with GT t=0 flow</span>
                <span class="score">{v['gt_score']:.3f}</span>
                <span class="meta">{v['gt_pred']} pred nodes</span>
              </figcaption>
            </figure>
            {pred_block}
          </div>
          <div class="vessel-foot">
            <span>{v['gt_n']} GT clot nodes</span>
          </div>
        </article>""")

    html = TEMPLATE.replace("<!--CARDS-->", "\n".join(cards))
    out = Path("outputs/phase3_generalization_report.html")
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out}  ({out.stat().st_size/1024/1024:.2f} MB)")


TEMPLATE = r"""<title>Phase 3 &mdash; Sealed-Set Generalization</title>
<style>
:root {
  --bg: #f2f4f6;
  --surface: #ffffff;
  --surface-2: #eaeef1;
  --ink: #16232b;
  --muted: #55707d;
  --line: #dde3e7;
  --accent: #1f6f78;
  --accent-ink: #ffffff;
  --tp: #1f8a4c;
  --fn: #c8362f;
  --fp: #b3730f;
  --tp-bg: #e4f3ea;
  --fn-bg: #fbe7e5;
  --fp-bg: #faedd9;
  --shadow: 0 1px 2px rgba(22,35,43,0.06), 0 8px 24px -12px rgba(22,35,43,0.18);
  --serif: ui-serif, "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif;
  --sans: -apple-system, BlinkMacSystemFont, "Segoe UI", "Helvetica Neue", Arial, sans-serif;
  --mono: ui-monospace, "SF Mono", "Cascadia Code", "Roboto Mono", Consolas, monospace;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0d1417;
    --surface: #141c20;
    --surface-2: #192226;
    --ink: #e7edef;
    --muted: #93a9b0;
    --line: #263136;
    --accent: #52b7c1;
    --accent-ink: #08191b;
    --tp: #3fcb78;
    --fn: #ff7a70;
    --fp: #f0b246;
    --tp-bg: #133622;
    --fn-bg: #3a1a17;
    --fp-bg: #3a2c12;
    --shadow: 0 1px 2px rgba(0,0,0,0.4), 0 12px 28px -14px rgba(0,0,0,0.6);
  }
}
:root[data-theme="dark"] {
  --bg: #0d1417; --surface: #141c20; --surface-2: #192226; --ink: #e7edef; --muted: #93a9b0;
  --line: #263136; --accent: #52b7c1; --accent-ink: #08191b;
  --tp: #3fcb78; --fn: #ff7a70; --fp: #f0b246;
  --tp-bg: #133622; --fn-bg: #3a1a17; --fp-bg: #3a2c12;
  --shadow: 0 1px 2px rgba(0,0,0,0.4), 0 12px 28px -14px rgba(0,0,0,0.6);
}
:root[data-theme="light"] {
  --bg: #f2f4f6; --surface: #ffffff; --surface-2: #eaeef1; --ink: #16232b; --muted: #55707d;
  --line: #dde3e7; --accent: #1f6f78; --accent-ink: #ffffff;
  --tp: #1f8a4c; --fn: #c8362f; --fp: #b3730f;
  --tp-bg: #e4f3ea; --fn-bg: #fbe7e5; --fp-bg: #faedd9;
  --shadow: 0 1px 2px rgba(22,35,43,0.06), 0 8px 24px -12px rgba(22,35,43,0.18);
}

* { box-sizing: border-box; }
body {
  background: var(--bg);
  color: var(--ink);
  font-family: var(--sans);
  line-height: 1.5;
  -webkit-font-smoothing: antialiased;
}
main {
  max-width: 1180px;
  margin: 0 auto;
  padding: 3.5rem 1.5rem 6rem;
}

/* ---- header ---- */
.eyebrow {
  font-family: var(--mono);
  font-size: 0.72rem;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--accent);
  margin: 0 0 0.9rem;
}
h1 {
  font-family: var(--serif);
  font-weight: 600;
  font-size: clamp(1.9rem, 3.4vw, 2.7rem);
  line-height: 1.12;
  letter-spacing: -0.01em;
  text-wrap: balance;
  margin: 0 0 0.9rem;
  max-width: 20ch;
}
.lede {
  font-size: 1.05rem;
  color: var(--muted);
  max-width: 62ch;
  margin: 0 0 2.4rem;
}
.lede strong { color: var(--ink); font-weight: 600; }

/* ---- score strip ---- */
.score-strip {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 1px;
  background: var(--line);
  border: 1px solid var(--line);
  border-radius: 10px;
  overflow: hidden;
  margin-bottom: 1rem;
  box-shadow: var(--shadow);
}
.score-cell {
  background: var(--surface);
  padding: 1.25rem 1.4rem;
}
.score-cell .label {
  font-family: var(--mono);
  font-size: 0.68rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 0.5rem;
}
.score-cell .value {
  font-family: var(--mono);
  font-variant-numeric: tabular-nums;
  font-size: 2rem;
  font-weight: 600;
  letter-spacing: -0.02em;
}
.score-cell .value small {
  font-size: 0.95rem;
  font-weight: 500;
  color: var(--muted);
}
.score-cell.hero .value { color: var(--accent); }
.bar-track {
  margin-top: 0.6rem;
  height: 5px;
  background: var(--surface-2);
  border-radius: 3px;
  overflow: hidden;
}
.bar-fill { height: 100%; background: var(--accent); border-radius: 3px; }
.threshold-note {
  font-family: var(--mono);
  font-size: 0.68rem;
  color: var(--muted);
  margin-top: 0.45rem;
}
.strip-caption {
  font-size: 0.82rem;
  color: var(--muted);
  margin: 0 0 3rem;
}

/* ---- legend ---- */
.legend {
  display: flex;
  flex-wrap: wrap;
  gap: 1.4rem;
  align-items: center;
  padding: 0.95rem 1.2rem;
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 8px;
  margin-bottom: 2.2rem;
  font-size: 0.85rem;
}
.legend-item { display: flex; align-items: center; gap: 0.5rem; }
.swatch { width: 11px; height: 11px; border-radius: 50%; flex: none; }
.swatch.tp { background: var(--tp); }
.swatch.fn { background: var(--fn); }
.swatch.fp { background: var(--fp); }
.swatch.tn { background: var(--muted); opacity: 0.55; }

/* ---- section heads ---- */
h2 {
  font-family: var(--serif);
  font-weight: 600;
  font-size: 1.4rem;
  margin: 0 0 0.35rem;
  letter-spacing: -0.005em;
}
.section-note {
  color: var(--muted);
  font-size: 0.92rem;
  max-width: 66ch;
  margin: 0 0 1.6rem;
}

/* ---- vessel gallery ---- */
.gallery {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
  gap: 1.3rem;
}
.vessel-card {
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 1.3rem 1.3rem 1.1rem;
  box-shadow: var(--shadow);
}
.vessel-head {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 0.8rem;
  margin-bottom: 0.2rem;
}
.vessel-head h3 {
  font-family: var(--mono);
  font-size: 1rem;
  font-weight: 600;
  margin: 0;
}
.split-badge {
  font-family: var(--mono);
  font-size: 0.66rem;
  letter-spacing: 0.04em;
  color: var(--muted);
  white-space: nowrap;
}
.card-note {
  font-size: 0.8rem;
  color: var(--muted);
  font-style: italic;
  margin: 0.15rem 0 0.9rem;
}
.panel-row {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0.8rem;
}
.panel {
  margin: 0;
  border: 1px solid var(--line);
  border-radius: 8px;
  overflow: hidden;
  background: var(--surface-2);
}
.panel img { display: block; width: 100%; height: auto; }
.panel-empty { display: flex; flex-direction: column; }
.empty-slot {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  text-align: center;
  min-height: 160px;
  font-size: 0.76rem;
  color: var(--muted);
  padding: 1rem;
}
.panel figcaption {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.55rem 0.7rem;
  border-top: 1px solid var(--line);
  background: var(--surface);
  flex-wrap: wrap;
}
.tag {
  font-family: var(--mono);
  font-size: 0.62rem;
  letter-spacing: 0.03em;
  text-transform: uppercase;
  padding: 0.18rem 0.45rem;
  border-radius: 4px;
  background: var(--surface-2);
  color: var(--muted);
}
.tag-gt { color: var(--accent-ink); background: var(--accent); }
.tag-pred { color: var(--ink); background: var(--surface-2); border: 1px solid var(--line); }
.tag-disabled { opacity: 0.55; }
.score {
  font-family: var(--mono);
  font-variant-numeric: tabular-nums;
  font-weight: 700;
  font-size: 0.92rem;
  margin-left: auto;
}
.meta {
  font-family: var(--mono);
  font-size: 0.68rem;
  color: var(--muted);
  white-space: nowrap;
}
.vessel-foot {
  margin-top: 0.7rem;
  font-family: var(--mono);
  font-size: 0.7rem;
  color: var(--muted);
}

/* ---- footer note ---- */
.foot-note {
  margin-top: 3rem;
  padding-top: 1.6rem;
  border-top: 1px solid var(--line);
  font-size: 0.82rem;
  color: var(--muted);
  max-width: 72ch;
}
.foot-note code {
  font-family: var(--mono);
  background: var(--surface-2);
  padding: 0.08rem 0.35rem;
  border-radius: 4px;
  font-size: 0.85em;
}

@media (max-width: 620px) {
  .panel-row { grid-template-columns: 1fr; }
}
</style>

<main>
  <p class="eyebrow">Phase 3 &middot; Thrombus wall-clot model &middot; sealed-set check</p>
  <h1>Zero learned parameters, spent once on eight vessels never trained on.</h1>
  <p class="lede">
    The wall-clot readout below is COMSOL's own deposition law integrated on a corrected
    t&#8209;0 shear field &mdash; no network, no checkpoint. <strong>Deploy score</strong> is the
    project's canonical wall-masked metric (relaxed precision/recall, guiding blend), scored
    against a target of <strong>0.60</strong>.
  </p>

  <div class="score-strip">
    <div class="score-cell hero">
      <div class="label">Sealed &middot; with GT t=0 flow</div>
      <div class="value">0.909</div>
      <div class="bar-track"><div class="bar-fill" style="width:90.9%"></div></div>
      <div class="threshold-note">8 / 8 vessels &ge; 0.60</div>
    </div>
    <div class="score-cell hero">
      <div class="label">Sealed &middot; deployable flow</div>
      <div class="value">0.857</div>
      <div class="bar-track"><div class="bar-fill" style="width:85.7%"></div></div>
      <div class="threshold-note">6 / 6 vessels &ge; 0.60 &mdash; 2 packs lack a flow surrogate</div>
    </div>
    <div class="score-cell">
      <div class="label">Previous best on record</div>
      <div class="value">0.693 <small>one favourable vessel</small></div>
      <div class="bar-track"><div class="bar-fill" style="width:69.3%; background:var(--muted); opacity:.6"></div></div>
      <div class="threshold-note">GNN 0.540 &middot; logreg 0.516 &middot; mean over 34, not one</div>
    </div>
  </div>
  <p class="strip-caption">
    &ldquo;With GT t=0 flow&rdquo; is a bandaid this project intends to remove &mdash; it needs a
    real flow solve per new vessel. &ldquo;Deployable&rdquo; uses only a pretrained kinematic
    network's predicted flow (<code>u0_pred</code>/<code>v0_pred</code>), geometry, and boundary
    conditions &mdash; nothing GT.
  </p>

  <h2>Sealed set, node by node</h2>
  <p class="section-note">
    All eight vessels below are the project's sealed set: never trained on, never tuned
    against, spent once. Each map shows every wall node, colored by outcome against the
    ground-truth clot at final time.
  </p>

  <div class="legend">
    <div class="legend-item"><span class="swatch tp"></span> correct clot</div>
    <div class="legend-item"><span class="swatch fn"></span> missed clot</div>
    <div class="legend-item"><span class="swatch fp"></span> false positive</div>
    <div class="legend-item"><span class="swatch tn"></span> correct: no clot</div>
  </div>

  <div class="gallery">
<!--CARDS-->
  </div>

  <p class="foot-note">
    Model: <code>src/core_physics/physics_wall_model.py</code>. Gates from
    <code>comsol_surface_deposition.py</code> on a moving-least-squares shear
    reconstruction (<code>mls_gradient.py</code>), grown along the wall graph into
    low-shear neighbours. Three scalars (stencil width, admission threshold, growth hops)
    fit on <code>WALL_COHORT_V2_TRAIN</code> only. Full derivation in
    <code>docs/PHASE3_RESULTS.md</code>.
  </p>
</main>
"""

if __name__ == "__main__":
    main()
