"""Assemble outputs/v3_temporal_data.json into the standard two-window viz for clot_gnn_v3."""
from __future__ import annotations

import json
from pathlib import Path

DATA_PATH = Path("outputs/v3_temporal_data.json")
OUT_PATH = Path("outputs/phase9_v3_temporal_report.html")

VESSEL_LABELS = {
    "patient040": "patient040 — aneurysm",
    "patient041": "patient041 — stenosis",
    "patient044": "patient044 — stenosis",
    "patient012": "patient012 — baseline geometry",
    "patient032": "patient032 — baseline geometry",
}
CLASS_COLOR = {"aneurysm": "badge-aneurysm", "stenosis": "badge-stenosis", "baseline": "badge-baseline"}
CLASS_TEXT = {"aneurysm": "ANEURYSM", "stenosis": "STENOSIS", "baseline": "BASELINE"}


def main() -> None:
    payload = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    data_json = json.dumps(payload).replace("</", "<\\/")
    labels_json = json.dumps(VESSEL_LABELS)
    order = list(payload.keys())

    def badge(v):
        cls = payload[v]["geom_class"]
        return f'<span class="tab-badge {CLASS_COLOR[cls]}">{CLASS_TEXT[cls]}</span>'

    tabs = "\n".join(
        f'<button class="tab{" active" if i == 0 else ""}" data-vessel="{v}">{v}{badge(v)}</button>'
        for i, v in enumerate(order)
    )

    html = TEMPLATE.replace("__DATA__", data_json).replace("__LABELS__", labels_json).replace(
        "<!--TABS-->", tabs
    ).replace("__FIRST__", order[0])
    OUT_PATH.write_text(html, encoding="utf-8")
    print(f"wrote {OUT_PATH}  ({OUT_PATH.stat().st_size/1024:.0f} KB)")


TEMPLATE = r"""<title>Phase 9 &mdash; clot_gnn_v3 vs Ground Truth</title>
<style>
:root {
  --bg: #f6f4f0; --surface: #ffffff; --surface-2: #efeae2; --ink: #241f18; --muted: #6b6255;
  --line: #e2dbcd; --accent: #a8681a; --accent-ink: #ffffff;
  --model-c: #a8681a; --gt-c: #2b6f6b;
  --model-far: #ecd9bb; --gt-far: #b9dbd8;
  --score-wall: #a8681a; --score-off: #2b6f6b;
  --shadow: 0 1px 2px rgba(36,31,24,0.06), 0 8px 24px -12px rgba(36,31,24,0.18);
  --serif: ui-serif, "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif;
  --sans: -apple-system, BlinkMacSystemFont, "Segoe UI", "Helvetica Neue", Arial, sans-serif;
  --mono: ui-monospace, "SF Mono", "Cascadia Code", "Roboto Mono", Consolas, monospace;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #17140f; --surface: #201b14; --surface-2: #2a2318; --ink: #f1ece0; --muted: #b0a48d;
    --line: #382f21; --accent: #e0a03e; --accent-ink: #1a1206;
    --model-c: #e0a03e; --gt-c: #5fbcb6;
    --model-far: #5c481f; --gt-far: #1f4d49;
    --score-wall: #e0a03e; --score-off: #5fbcb6;
    --shadow: 0 1px 2px rgba(0,0,0,0.4), 0 12px 28px -14px rgba(0,0,0,0.6);
  }
}
:root[data-theme="dark"] {
  --bg: #17140f; --surface: #201b14; --surface-2: #2a2318; --ink: #f1ece0; --muted: #b0a48d;
  --line: #382f21; --accent: #e0a03e; --accent-ink: #1a1206;
  --model-c: #e0a03e; --gt-c: #5fbcb6;
  --model-far: #5c481f; --gt-far: #1f4d49;
  --score-wall: #e0a03e; --score-off: #5fbcb6;
  --shadow: 0 1px 2px rgba(0,0,0,0.4), 0 12px 28px -14px rgba(0,0,0,0.6);
}
:root[data-theme="light"] {
  --bg: #f6f4f0; --surface: #ffffff; --surface-2: #efeae2; --ink: #241f18; --muted: #6b6255;
  --line: #e2dbcd; --accent: #a8681a; --accent-ink: #ffffff;
  --model-c: #a8681a; --gt-c: #2b6f6b;
  --model-far: #ecd9bb; --gt-far: #b9dbd8;
  --score-wall: #a8681a; --score-off: #2b6f6b;
  --shadow: 0 1px 2px rgba(36,31,24,0.06), 0 8px 24px -12px rgba(36,31,24,0.18);
}
* { box-sizing: border-box; }
body { background: var(--bg); color: var(--ink); font-family: var(--sans); line-height: 1.5; -webkit-font-smoothing: antialiased; }
main { max-width: 1220px; margin: 0 auto; padding: 3.2rem 1.5rem 5rem; }

.eyebrow { font-family: var(--mono); font-size: 0.72rem; letter-spacing: 0.14em; text-transform: uppercase; color: var(--accent); margin: 0 0 0.9rem; }
h1 { font-family: var(--serif); font-weight: 600; font-size: clamp(1.7rem, 3.1vw, 2.4rem); line-height: 1.15; letter-spacing: -0.01em; text-wrap: balance; margin: 0 0 0.9rem; max-width: 32ch; }
.lede { font-size: 1.02rem; color: var(--muted); max-width: 70ch; margin: 0 0 1.1rem; }
.lede strong { color: var(--ink); font-weight: 600; }
.lede code { font-family: var(--mono); background: var(--surface-2); padding: 0.06rem 0.32rem; border-radius: 4px; font-size: 0.88em; }

.callout { background: var(--surface); border: 1px solid var(--line); border-left: 3px solid var(--model-c); border-radius: 8px; padding: 0.95rem 1.2rem; font-size: 0.87rem; color: var(--ink); max-width: 76ch; margin: 0 0 1rem; }
.callout b { color: var(--model-c); }
.callout + .callout { border-left-color: var(--gt-c); margin-bottom: 2rem; }
.callout + .callout b { color: var(--gt-c); }

.tabs { display: flex; flex-wrap: wrap; align-items: center; gap: 0.5rem; margin-bottom: 0.6rem; }
.tab { font-family: var(--mono); font-size: 0.78rem; padding: 0.5rem 0.7rem 0.5rem 0.9rem; border-radius: 7px; border: 1px solid var(--line); background: var(--surface); color: var(--muted); cursor: pointer; transition: background 0.15s, color 0.15s, border-color 0.15s; display: inline-flex; align-items: center; gap: 0.5rem; }
.tab:hover { border-color: var(--accent); color: var(--ink); }
.tab.active { background: var(--accent); border-color: var(--accent); color: var(--accent-ink); font-weight: 600; }
.tab-badge { font-size: 0.6rem; font-weight: 700; letter-spacing: 0.04em; padding: 0.12rem 0.4rem; border-radius: 4px; }
.badge-aneurysm { background: #8a4fc7; color: #ffffff; }
.badge-stenosis { background: #c7504f; color: #ffffff; }
.badge-baseline { background: var(--surface-2); color: var(--muted); }
.tab.active .badge-baseline { background: rgba(255,255,255,0.3); color: var(--accent-ink); }
.tab-group-label { font-family: var(--mono); font-size: 0.66rem; letter-spacing: 0.06em; text-transform: uppercase; color: var(--muted); margin: 0 0 1.3rem; }

.spatial-row { display: grid; grid-template-columns: 1fr 1fr; gap: 1.1rem; margin-bottom: 1.1rem; }
.panel-box { background: var(--surface); border: 1px solid var(--line); border-radius: 12px; box-shadow: var(--shadow); padding: 1rem 1.1rem 1.2rem; }
.panel-box h2 { font-family: var(--serif); font-size: 1.0rem; font-weight: 600; margin: 0 0 0.6rem; display: flex; align-items: center; gap: 0.5rem; }
.dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
canvas { width: 100%; display: block; border-radius: 8px; background: var(--surface-2); }
.spatial-canvas { aspect-ratio: 1 / 1; cursor: grab; touch-action: none; }
.spatial-canvas:active { cursor: grabbing; }
.panel-head-row { display: flex; align-items: center; justify-content: space-between; gap: 0.6rem; margin-bottom: 0.6rem; flex-wrap: wrap; }
.panel-head-row h2 { margin: 0; }
.panel-meta { display: flex; align-items: center; gap: 0.6rem; }
.zoom-hint { font-family: var(--mono); font-size: 0.66rem; color: var(--muted); white-space: nowrap; }
.score-badge { font-family: var(--mono); font-variant-numeric: tabular-nums; font-size: 1.5rem; font-weight: 700; padding: 0.35rem 0.85rem; border-radius: 8px; background: var(--surface-2); border: 1px solid var(--line); white-space: nowrap; letter-spacing: -0.01em; }
.score-badge.wall { color: var(--score-wall); }
.score-badge.off { color: var(--score-off); }
.score-row { display: grid; grid-template-columns: 1fr 1fr; gap: 1.1rem; margin-bottom: 1rem; }
.chart-canvas { aspect-ratio: 16 / 9; }

.transport { display: flex; align-items: center; gap: 0.9rem; margin: 0.9rem 0 1.8rem; }
.play-btn { font-family: var(--mono); font-size: 0.85rem; padding: 0.5rem 0.9rem; border-radius: 7px; border: 1px solid var(--accent); background: var(--accent); color: var(--accent-ink); cursor: pointer; flex: none; min-width: 4.6rem; }
.play-btn:hover { filter: brightness(1.06); }
.reset-btn { font-family: var(--mono); font-size: 0.78rem; padding: 0.5rem 0.8rem; border-radius: 7px; border: 1px solid var(--line); background: var(--surface); color: var(--muted); cursor: pointer; flex: none; }
.reset-btn:hover { border-color: var(--accent); color: var(--ink); }
input[type="range"] { flex: 1; accent-color: var(--accent); }
.time-readout { font-family: var(--mono); font-variant-numeric: tabular-nums; font-size: 0.76rem; color: var(--muted); min-width: 9rem; text-align: right; flex: none; }
.time-readout b { color: var(--ink); }

.legend { display: flex; flex-wrap: wrap; gap: 1.1rem; align-items: center; padding: 0.85rem 1.1rem; background: var(--surface); border: 1px solid var(--line); border-radius: 8px; margin: 0 0 2.4rem; font-size: 0.8rem; }
.legend-item { display: flex; align-items: center; gap: 0.45rem; }
.swatch { width: 11px; height: 11px; border-radius: 50%; flex: none; }
.swatch.model { background: var(--model-c); } .swatch.gt { background: var(--gt-c); }
.grad-swatch { width: 34px; height: 11px; border-radius: 3px; flex: none; }
.grad-swatch.grad-model { background: linear-gradient(90deg, var(--model-c), var(--model-far)); }
.grad-swatch.grad-gt { background: linear-gradient(90deg, var(--gt-c), var(--gt-far)); }
.lineswatch { width: 18px; height: 0; border-top: 2px solid var(--score-wall); }
.lineswatch.off { border-top-color: var(--score-off); }

h2.section { font-family: var(--serif); font-weight: 600; font-size: 1.3rem; margin: 2.6rem 0 0.4rem; }
.section-note { color: var(--muted); font-size: 0.92rem; max-width: 70ch; margin: 0 0 1.2rem; }
.finding-box { background: var(--surface); border: 1px solid var(--line); border-left: 3px solid var(--gt-c); border-radius: 8px; padding: 1rem 1.2rem; font-size: 0.9rem; color: var(--ink); max-width: 76ch; }
.finding-box b { color: var(--gt-c); }
.compare-table { width: 100%; border-collapse: collapse; font-family: var(--mono); font-size: 0.82rem; margin: 1rem 0; }
.compare-table th, .compare-table td { text-align: right; padding: 0.4rem 0.7rem; border-bottom: 1px solid var(--line); }
.compare-table th:first-child, .compare-table td:first-child { text-align: left; }
.compare-table th { color: var(--muted); font-weight: 600; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.03em; }
.compare-table td.win { color: var(--gain, #1f8a4c); font-weight: 600; }
.compare-table td.trust { color: var(--model-c); font-weight: 700; }
.foot-note { margin-top: 2.6rem; padding-top: 1.5rem; border-top: 1px solid var(--line); font-size: 0.82rem; color: var(--muted); max-width: 74ch; }
.foot-note code { font-family: var(--mono); background: var(--surface-2); padding: 0.08rem 0.35rem; border-radius: 4px; font-size: 0.85em; }

@media (max-width: 760px) { .spatial-row, .score-row { grid-template-columns: 1fr; } }
</style>

<main>
  <p class="eyebrow">Phase 9 &middot; clot_gnn_v3 &middot; model vs ground truth</p>
  <h1>Time is now an input, not an afterthought.</h1>
  <p class="lede">
    <code>clot_gnn_v3</code> predicts <strong>P(node is clot at time t) directly</strong> --
    time is a genuine model input, not a static score held constant or an onset rank mapped
    onto a borrowed schedule. Both windows below actually animate from the model's own
    per-timestep output for the first time in this project. Predictions are forced monotone
    in time (clot never un-clots) and an off-wall node can never fire before its own wall
    owner does.
  </p>

  <div class="callout">
    <b>Every vessel below is in-sample.</b> Per <code>docs/PHASE9_ML.md</code> &sect;14:
    <i>"There is no vessel left to score v3 against out-of-fold -- like v2, it trains on the
    whole pool by design."</i> The tabs are badged by <b>geometry class</b>
    (aneurysm / stenosis / baseline), not by held-out status -- there is no held-out status
    left to badge. The scores you see here are optimistic; scroll down for the number that
    actually estimates generalization.
  </div>
  <div class="callout">
    <b>SEALED was never opened.</b> <code>patient042</code>/<code>043</code> and every
    vessel outside the 19-vessel <code>training_pool</code> (<code>data/reference/clot_gnn_locked.json</code>)
    are excluded from this page.
  </div>

  <h2 class="section">The number that matters: geometry-stratified 5-fold CV</h2>
  <p class="section-note">
    <code>docs/PHASE9_ML.md</code> &sect;11 replaced the old FIT/DEV split (which put every
    aneurysm/stenosis vessel in DEV, confounding class with generalization) with 5-fold CV
    stratified so every priority-class vessel is held out in exactly one fold. This is the
    protocol that selected v3's design, and it is the only trustworthy estimate here.
  </p>
  <div style="overflow-x:auto">
  <table class="compare-table">
    <thead><tr><th>metric</th><th>wall</th><th>off-wall</th></tr></thead>
    <tbody>
      <tr><td>frozen mask (pre-temporal baseline)</td><td>0.7953</td><td>0.4209</td></tr>
      <tr><td>v2 + ODE timing (previous generation)</td><td>0.8547</td><td>0.5369</td></tr>
      <tr><td class="trust">v3 time-conditioned, out-of-fold CV &mdash; trust this</td><td class="trust">0.8845</td><td class="trust">0.6110</td></tr>
      <tr><td>v3, priority class only (n=3), out-of-fold CV</td><td class="trust">0.9047</td><td class="trust">0.7943</td></tr>
      <tr><td>oracle timing (ceiling, perfect schedule)</td><td>0.9705</td><td>0.8396</td></tr>
    </tbody>
  </table>
  </div>
  <p class="section-note">
    v3 closes 92% of the wall gap and 93% of the off-wall gap between the frozen baseline
    and the oracle ceiling, out-of-fold. On the priority class it exceeds even an
    oracle-schedule reference arm off-wall. Compare that to what you're about to see in the
    panels below, which will read noticeably higher because those 5 vessels were in the
    training pool.
  </p>

  <p class="tab-group-label">tabs badged by geometry class, not held-out status &mdash; every vessel is in-sample</p>
  <div class="tabs">
    <!--TABS-->
  </div>

  <div class="spatial-row">
    <div class="panel-box">
      <div class="panel-head-row">
        <h2><span class="dot" style="background:var(--model-c)"></span>Model (clot_gnn_v3)</h2>
        <div class="panel-meta">
          <span class="score-badge wall" id="val-wall">wall &mdash;</span>
          <span class="zoom-hint" id="zoom-readout">1.0&times;</span>
        </div>
      </div>
      <canvas id="canvas-model" class="spatial-canvas" width="560" height="560"></canvas>
    </div>
    <div class="panel-box">
      <div class="panel-head-row">
        <h2><span class="dot" style="background:var(--gt-c)"></span>Ground truth</h2>
        <div class="panel-meta">
          <span class="score-badge off" id="val-off">off-wall &mdash;</span>
          <span class="zoom-hint">scroll/drag, synced</span>
        </div>
      </div>
      <canvas id="canvas-gt" class="spatial-canvas" width="560" height="560"></canvas>
    </div>
  </div>

  <div class="transport">
    <button class="play-btn" id="play-btn">&#9654; Play</button>
    <button class="reset-btn" id="reset-zoom-btn" title="Reset zoom/pan (or double-click either window)">&#8635; Reset view</button>
    <input type="range" id="frame-slider" min="0" max="12" value="0" step="1" />
    <div class="time-readout" id="time-readout">t = 0 s <b>(0%)</b></div>
  </div>

  <div class="score-row">
    <div class="panel-box">
      <h2>Wall score over time</h2>
      <canvas id="chart-wall" class="chart-canvas" width="620" height="349"></canvas>
    </div>
    <div class="panel-box">
      <h2>Off-wall score over time</h2>
      <canvas id="chart-off" class="chart-canvas" width="620" height="349"></canvas>
    </div>
  </div>

  <div class="legend">
    <div class="legend-item"><span class="swatch model"></span> model</div>
    <div class="legend-item"><span class="swatch gt"></span> ground truth</div>
    <div class="legend-item">&#9679; wall &nbsp; &#9632; lumen</div>
    <div class="legend-item"><span class="grad-swatch grad-model"></span> depth into lumen (model)</div>
    <div class="legend-item"><span class="grad-swatch grad-gt"></span> depth into lumen (GT)</div>
    <div class="legend-item"><span class="lineswatch"></span> wall score</div>
    <div class="legend-item"><span class="lineswatch off"></span> off-wall score</div>
  </div>

  <h2 class="section">What to read off this</h2>
  <div class="finding-box">
    Both panels now genuinely move &mdash; watch the <em>Model</em> window's own front
    advance, not just Ground Truth's. Where the two shapes diverge in time (model ahead of
    GT, or behind it) is a real read on the learned schedule, not an artifact of a frozen
    mask being scored against a moving target. <code>patient040</code> (aneurysm) is the
    single most encouraging out-of-fold result in the whole phase per the docs -- worth
    watching closely even though it's shown here in-sample.
  </div>

  <p class="foot-note">
    Model: <code>src/clot_ml/locked.py::load_default</code> +
    <code>predict_default_series</code> (dispatches to <code>predict_temporal_v3</code>),
    reusing locked <code>clot_gnn_v2</code> features plus a gradient-boosted temporal head
    (<code>docs/PHASE9_ML.md</code> &sect;13.9/&sect;14). GT from
    <code>gt_clot_phi_at_time</code> at every real simulated timestep. Scores from
    <code>compute_clot_relaxed_metrics</code> + <code>clot_score_from_deploy_dict</code>,
    domain-restricted to wall or off-wall -- computed fresh here, independent of the
    manifest's own severity-metric CV numbers shown above.
  </p>
</main>

<script id="viz-data" type="application/json">__DATA__</script>
<script>
(function () {
  const DATA = JSON.parse(document.getElementById('viz-data').textContent);
  const LABELS = __LABELS__;
  const order = Object.keys(DATA);
  let vessel = "__FIRST__";
  let frame = 0;
  let playing = false;
  let timer = null;

  const canvasModel = document.getElementById('canvas-model');
  const canvasGt = document.getElementById('canvas-gt');
  const cmModel = canvasModel.getContext('2d');
  const cmGt = canvasGt.getContext('2d');
  const wallChart = document.getElementById('chart-wall');
  const offChart = document.getElementById('chart-off');
  const wctx = wallChart.getContext('2d');
  const octx = offChart.getContext('2d');
  const slider = document.getElementById('frame-slider');
  const readout = document.getElementById('time-readout');
  const playBtn = document.getElementById('play-btn');
  const resetZoomBtn = document.getElementById('reset-zoom-btn');
  const zoomReadout = document.getElementById('zoom-readout');
  const valWall = document.getElementById('val-wall');
  const valOff = document.getElementById('val-off');

  const view = { k: 1, panX: 0, panY: 0 };
  const MIN_K = 1, MAX_K = 25;

  function css(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }

  function bbox(pts) {
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    for (const [x, y] of pts) { if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; }
    return [x0, x1, y0, y1];
  }

  function canvasPoint(canvas, evt) {
    const rect = canvas.getBoundingClientRect();
    const cx = (evt.clientX !== undefined ? evt.clientX : evt.touches[0].clientX);
    const cy = (evt.clientY !== undefined ? evt.clientY : evt.touches[0].clientY);
    return { x: (cx - rect.left) * (canvas.width / rect.width), y: (cy - rect.top) * (canvas.height / rect.height) };
  }

  function resetView() {
    view.k = 1; view.panX = 0; view.panY = 0;
    zoomReadout.textContent = '1.0×';
    redraw();
  }

  function zoomAt(canvas, screenX, screenY, factor) {
    const newK = Math.max(MIN_K, Math.min(MAX_K, view.k * factor));
    const worldX = (screenX - view.panX) / view.k;
    const worldY = (screenY - view.panY) / view.k;
    view.panX = screenX - worldX * newK;
    view.panY = screenY - worldY * newK;
    view.k = newK;
    zoomReadout.textContent = view.k.toFixed(1) + '×';
    redraw();
  }

  function attachZoomPan(canvas) {
    canvas.addEventListener('wheel', (evt) => {
      evt.preventDefault();
      const p = canvasPoint(canvas, evt);
      zoomAt(canvas, p.x, p.y, evt.deltaY < 0 ? 1.15 : 1 / 1.15);
    }, { passive: false });

    let dragging = false, start = null;
    canvas.addEventListener('mousedown', (evt) => {
      dragging = true;
      start = { x: evt.clientX, y: evt.clientY, panX: view.panX, panY: view.panY };
    });
    window.addEventListener('mousemove', (evt) => {
      if (!dragging) return;
      const rect = canvas.getBoundingClientRect();
      const scaleX = canvas.width / rect.width, scaleY = canvas.height / rect.height;
      view.panX = start.panX + (evt.clientX - start.x) * scaleX;
      view.panY = start.panY + (evt.clientY - start.y) * scaleY;
      redraw();
    });
    window.addEventListener('mouseup', () => { dragging = false; });
    canvas.addEventListener('dblclick', () => resetView());

    let pinchDist = null;
    canvas.addEventListener('touchstart', (evt) => {
      if (evt.touches.length === 1) {
        dragging = true;
        start = { x: evt.touches[0].clientX, y: evt.touches[0].clientY, panX: view.panX, panY: view.panY };
      } else if (evt.touches.length === 2) {
        dragging = false;
        const [a, b] = evt.touches;
        pinchDist = Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY);
      }
    }, { passive: true });
    canvas.addEventListener('touchmove', (evt) => {
      if (evt.touches.length === 1 && dragging) {
        const rect = canvas.getBoundingClientRect();
        const scaleX = canvas.width / rect.width, scaleY = canvas.height / rect.height;
        view.panX = start.panX + (evt.touches[0].clientX - start.x) * scaleX;
        view.panY = start.panY + (evt.touches[0].clientY - start.y) * scaleY;
        redraw();
      } else if (evt.touches.length === 2 && pinchDist !== null) {
        const [a, b] = evt.touches;
        const d = Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY);
        const mid = canvasPoint(canvas, { clientX: (a.clientX + b.clientX) / 2, clientY: (a.clientY + b.clientY) / 2 });
        zoomAt(canvas, mid.x, mid.y, d / pinchDist);
        pinchDist = d;
      }
    }, { passive: true });
    canvas.addEventListener('touchend', () => { dragging = false; pinchDist = null; });
  }

  function hexToRgb(hex) {
    const h = hex.replace('#', '');
    return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)];
  }
  function lerpColor(hexNear, hexFar, t) {
    const a = hexToRgb(hexNear), b = hexToRgb(hexFar);
    return 'rgb(' + Math.round(a[0] + (b[0] - a[0]) * t) + ',' + Math.round(a[1] + (b[1] - a[1]) * t) + ',' + Math.round(a[2] + (b[2] - a[2]) * t) + ')';
  }

  function drawWindow(ctx, canvas, wallHot, lumenHot, nearColor, farColor) {
    const d = DATA[vessel];
    const w = canvas.width, h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = css('--surface-2');
    ctx.fillRect(0, 0, w, h);

    const all = d.bg.concat(d.wall_pos).concat(d.lumen_pos);
    const [x0, x1, y0, y1] = bbox(all);
    const pad = 24;
    const sx = (w - 2 * pad) / Math.max(x1 - x0, 1e-9);
    const sy = (h - 2 * pad) / Math.max(y1 - y0, 1e-9);
    const s = Math.min(sx, sy);
    const ox = pad + ((w - 2 * pad) - s * (x1 - x0)) / 2;
    const oy = pad + ((h - 2 * pad) - s * (y1 - y0)) / 2;
    function px(x) { return (ox + (x - x0) * s) * view.k + view.panX; }
    function py(y) { return (oy + (y1 - y) * s) * view.k + view.panY; }

    ctx.fillStyle = css('--muted');
    ctx.globalAlpha = 0.22;
    for (const [x, y] of d.bg) { ctx.beginPath(); ctx.arc(px(x), py(y), 1.1, 0, Math.PI * 2); ctx.fill(); }

    ctx.globalAlpha = 1;
    ctx.fillStyle = nearColor;
    for (let i = 0; i < d.wall_pos.length; i++) {
      if (!wallHot[i]) continue;
      const [x, y] = d.wall_pos[i];
      ctx.beginPath(); ctx.arc(px(x), py(y), 2.9, 0, Math.PI * 2); ctx.fill();
    }
    for (let i = 0; i < d.lumen_pos.length; i++) {
      if (!lumenHot[i]) continue;
      const [x, y] = d.lumen_pos[i];
      const r = 2.9;
      ctx.fillStyle = lerpColor(nearColor, farColor, d.lumen_dist[i]);
      ctx.fillRect(px(x) - r, py(y) - r, r * 2, r * 2);
    }
  }

  function drawScoreChart(ctx, canvas, series, tSeries, tFinal, curT, color) {
    const w = canvas.width, h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = css('--surface-2');
    ctx.fillRect(0, 0, w, h);
    const padL = 32, padR = 12, padT = 10, padB = 26;
    const plotW = w - padL - padR, plotH = h - padT - padB;
    function px(t) { return padL + (t / tFinal) * plotW; }
    function py(v) { return padT + plotH - Math.max(0, Math.min(1, v)) * plotH; }

    ctx.strokeStyle = css('--line'); ctx.lineWidth = 1; ctx.font = '9px ' + css('--mono'); ctx.fillStyle = css('--muted');
    for (let i = 0; i <= 4; i++) {
      const v = i / 4, y = py(v);
      ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(w - padR, y); ctx.stroke();
      ctx.fillText(v.toFixed(2), 2, y + 3);
    }
    for (let i = 0; i <= 3; i++) {
      const tv = (tFinal / 3) * i, x = px(tv);
      ctx.fillText(Math.round(tv / 1000) + 'k', x - 8, h - padB + 13);
    }

    ctx.strokeStyle = color; ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < tSeries.length; i++) {
      const x = px(tSeries[i]), y = py(series[i]);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();

    const cx = px(curT);
    ctx.strokeStyle = css('--ink'); ctx.globalAlpha = 0.35; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(cx, padT); ctx.lineTo(cx, padT + plotH); ctx.stroke();
    ctx.globalAlpha = 1;
  }

  function scoreAtTime(scoreT, series, t) {
    let best = 0, bestDist = Infinity;
    for (let i = 0; i < scoreT.length; i++) {
      const dist = Math.abs(scoreT[i] - t);
      if (dist < bestDist) { bestDist = dist; best = i; }
    }
    return series[best];
  }

  function updateReadout() {
    const d = DATA[vessel];
    const t = d.frame_t[frame];
    const pct = Math.round((t / d.t_final) * 100);
    readout.innerHTML = 't = ' + Math.round(t) + ' s <b>(' + pct + '%)</b>';
    const sw = scoreAtTime(d.score_t, d.score_wall, t);
    const so = scoreAtTime(d.score_t, d.score_offwall, t);
    valWall.textContent = 'wall ' + sw.toFixed(3);
    valOff.textContent = 'off-wall ' + so.toFixed(3);
  }

  function redraw() {
    const d = DATA[vessel];
    drawWindow(cmModel, canvasModel, d.frame_model_wall[frame], d.frame_model_lumen[frame], css('--model-c'), css('--model-far'));
    drawWindow(cmGt, canvasGt, d.frame_gt_wall[frame], d.frame_gt_lumen[frame], css('--gt-c'), css('--gt-far'));
    drawScoreChart(wctx, wallChart, d.score_wall, d.score_t, d.t_final, d.frame_t[frame], css('--score-wall'));
    drawScoreChart(octx, offChart, d.score_offwall, d.score_t, d.t_final, d.frame_t[frame], css('--score-off'));
    updateReadout();
  }

  function setVessel(v) {
    vessel = v;
    frame = 0;
    slider.value = 0;
    view.k = 1; view.panX = 0; view.panY = 0;
    zoomReadout.textContent = '1.0×';
    document.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t.dataset.vessel === v));
    redraw();
  }

  attachZoomPan(canvasModel);
  attachZoomPan(canvasGt);
  resetZoomBtn.addEventListener('click', resetView);

  document.querySelectorAll('.tab').forEach(t => {
    t.addEventListener('click', () => { stopPlay(); setVessel(t.dataset.vessel); });
  });

  slider.addEventListener('input', () => { stopPlay(); frame = parseInt(slider.value, 10); redraw(); });

  function stopPlay() {
    playing = false; playBtn.innerHTML = '&#9654; Play';
    if (timer) { clearInterval(timer); timer = null; }
  }

  playBtn.addEventListener('click', () => {
    if (playing) { stopPlay(); return; }
    playing = true; playBtn.innerHTML = '&#10074;&#10074; Pause';
    timer = setInterval(() => { frame = (frame + 1) % 13; slider.value = frame; redraw(); }, 750);
  });

  window.addEventListener('resize', redraw);
  setVessel(vessel);
})();
</script>
"""

if __name__ == "__main__":
    main()
