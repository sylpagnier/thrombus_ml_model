"""Assemble outputs/offwall_temporal_data.json into a two-window (Model / GT) time-lapse report."""
from __future__ import annotations

import json
from pathlib import Path

DATA_PATH = Path("outputs/offwall_temporal_data.json")
OUT_PATH = Path("outputs/phase6_offwall_temporal_report.html")

VESSEL_LABELS = {
    "patient012": "patient012 — most off-wall clot in the cohort (48%)",
    "patient044": "patient044 — largest absolute off-wall count",
    "patient042": "patient042 — sealed, never trained on",
    "patient007": "patient007 — sealed, deployable flow, COMSOL-validated",
    "patient032": "patient032 — deployable flow, lumen arm finds nothing",
}


def main() -> None:
    payload = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    data_json = json.dumps(payload).replace("</", "<\\/")
    labels_json = json.dumps(VESSEL_LABELS)
    order = list(payload.keys())

    tabs = "\n".join(
        f'<button class="tab{" active" if i == 0 else ""}" data-vessel="{v}">{v}</button>'
        for i, v in enumerate(order)
    )

    html = TEMPLATE.replace("__DATA__", data_json).replace("__LABELS__", labels_json).replace(
        "<!--TABS-->", tabs
    ).replace("__FIRST__", order[0])
    OUT_PATH.write_text(html, encoding="utf-8")
    print(f"wrote {OUT_PATH}  ({OUT_PATH.stat().st_size/1024:.0f} KB)")


TEMPLATE = r"""<title>Phase 6 &mdash; Model vs Ground Truth Over Time</title>
<style>
:root {
  --bg: #f2f4f6; --surface: #ffffff; --surface-2: #eaeef1; --ink: #16232b; --muted: #55707d;
  --line: #dde3e7; --accent: #1f6f78; --accent-ink: #ffffff;
  --model-c: #1f6f78; --gt-c: #b3453e;
  --model-far: #a8d6cf; --gt-far: #e8b3a6;
  --score-wall: #1f6f78; --score-off: #6a4fb8;
  --shadow: 0 1px 2px rgba(22,35,43,0.06), 0 8px 24px -12px rgba(22,35,43,0.18);
  --serif: ui-serif, "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif;
  --sans: -apple-system, BlinkMacSystemFont, "Segoe UI", "Helvetica Neue", Arial, sans-serif;
  --mono: ui-monospace, "SF Mono", "Cascadia Code", "Roboto Mono", Consolas, monospace;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0d1417; --surface: #141c20; --surface-2: #192226; --ink: #e7edef; --muted: #93a9b0;
    --line: #263136; --accent: #52b7c1; --accent-ink: #08191b;
    --model-c: #52b7c1; --gt-c: #ef7f77;
    --model-far: #285a5c; --gt-far: #6e3934;
    --score-wall: #52b7c1; --score-off: #a68bef;
    --shadow: 0 1px 2px rgba(0,0,0,0.4), 0 12px 28px -14px rgba(0,0,0,0.6);
  }
}
:root[data-theme="dark"] {
  --bg: #0d1417; --surface: #141c20; --surface-2: #192226; --ink: #e7edef; --muted: #93a9b0;
  --line: #263136; --accent: #52b7c1; --accent-ink: #08191b;
  --model-c: #52b7c1; --gt-c: #ef7f77;
  --model-far: #285a5c; --gt-far: #6e3934;
  --score-wall: #52b7c1; --score-off: #a68bef;
  --shadow: 0 1px 2px rgba(0,0,0,0.4), 0 12px 28px -14px rgba(0,0,0,0.6);
}
:root[data-theme="light"] {
  --bg: #f2f4f6; --surface: #ffffff; --surface-2: #eaeef1; --ink: #16232b; --muted: #55707d;
  --line: #dde3e7; --accent: #1f6f78; --accent-ink: #ffffff;
  --model-c: #1f6f78; --gt-c: #b3453e;
  --model-far: #a8d6cf; --gt-far: #e8b3a6;
  --score-wall: #1f6f78; --score-off: #6a4fb8;
  --shadow: 0 1px 2px rgba(22,35,43,0.06), 0 8px 24px -12px rgba(22,35,43,0.18);
}
* { box-sizing: border-box; }
body { background: var(--bg); color: var(--ink); font-family: var(--sans); line-height: 1.5; -webkit-font-smoothing: antialiased; }
main { max-width: 1220px; margin: 0 auto; padding: 3.2rem 1.5rem 5rem; }

.eyebrow { font-family: var(--mono); font-size: 0.72rem; letter-spacing: 0.14em; text-transform: uppercase; color: var(--accent); margin: 0 0 0.9rem; }
h1 { font-family: var(--serif); font-weight: 600; font-size: clamp(1.7rem, 3.1vw, 2.4rem); line-height: 1.15; letter-spacing: -0.01em; text-wrap: balance; margin: 0 0 0.9rem; max-width: 28ch; }
.lede { font-size: 1.02rem; color: var(--muted); max-width: 70ch; margin: 0 0 1.1rem; }
.lede strong { color: var(--ink); font-weight: 600; }
.lede code { font-family: var(--mono); background: var(--surface-2); padding: 0.06rem 0.32rem; border-radius: 4px; font-size: 0.88em; }

.callout { background: var(--surface); border: 1px solid var(--line); border-left: 3px solid var(--score-off); border-radius: 8px; padding: 0.95rem 1.2rem; font-size: 0.87rem; color: var(--ink); max-width: 76ch; margin: 0 0 1rem; }
.callout b { color: var(--score-off); }
.callout + .callout { border-left-color: var(--accent); margin-bottom: 2rem; }
.callout + .callout b { color: var(--accent); }

.tabs { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 1.4rem; }
.tab { font-family: var(--mono); font-size: 0.78rem; padding: 0.5rem 0.9rem; border-radius: 7px; border: 1px solid var(--line); background: var(--surface); color: var(--muted); cursor: pointer; transition: background 0.15s, color 0.15s, border-color 0.15s; }
.tab:hover { border-color: var(--accent); color: var(--ink); }
.tab.active { background: var(--accent); border-color: var(--accent); color: var(--accent-ink); font-weight: 600; }

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
.reset-btn {
  font-family: var(--mono); font-size: 0.78rem; padding: 0.5rem 0.8rem; border-radius: 7px;
  border: 1px solid var(--line); background: var(--surface); color: var(--muted); cursor: pointer; flex: none;
}
.reset-btn:hover { border-color: var(--accent); color: var(--ink); }
input[type="range"] { flex: 1; accent-color: var(--accent); }
.time-readout { font-family: var(--mono); font-variant-numeric: tabular-nums; font-size: 0.76rem; color: var(--muted); min-width: 9rem; text-align: right; flex: none; }
.time-readout b { color: var(--ink); }

.legend { display: flex; flex-wrap: wrap; gap: 1.1rem; align-items: center; padding: 0.85rem 1.1rem; background: var(--surface); border: 1px solid var(--line); border-radius: 8px; margin: 0 0 2.4rem; font-size: 0.8rem; }
.legend-item { display: flex; align-items: center; gap: 0.45rem; }
.swatch { width: 11px; height: 11px; border-radius: 50%; flex: none; }
.swatch.sq { border-radius: 2px; }
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
.foot-note { margin-top: 2.6rem; padding-top: 1.5rem; border-top: 1px solid var(--line); font-size: 0.82rem; color: var(--muted); max-width: 74ch; }
.foot-note code { font-family: var(--mono); background: var(--surface-2); padding: 0.08rem 0.35rem; border-radius: 4px; font-size: 0.85em; }

@media (max-width: 760px) { .spatial-row, .score-row { grid-template-columns: 1fr; } }
</style>

<main>
  <p class="eyebrow">Phase 6 &middot; wall + lumen clot model &middot; model vs ground truth</p>
  <h1>Two windows, one clock, the score at every step.</h1>
  <p class="lede">
    Scrub the slider or hit play: <strong>left is the model's prediction</strong>,
    <strong>right is what actually happened</strong> &mdash; no overlay, no diffing, just
    two plain pictures on the same clock. Circles are wall nodes, squares are off-wall
    (lumen) nodes. Below, the deploy score at that instant, computed separately on the
    wall domain and the off-wall domain.
  </p>

  <div class="callout">
    <b>What "wall score" / "off-wall score" mean here.</b> Same canonical scoring function
    (relaxed precision/recall + dilation IoU, guiding blend) run twice: once with both
    prediction and ground truth restricted to wall nodes, once restricted to off-wall
    nodes. This is a different number from the earlier full-mesh report, which scored one
    prediction against the whole vessel at once &mdash; here the two domains are judged
    independently, so a vessel can score well on wall timing and poorly off-wall (or the
    reverse) without one dragging the other down.
  </div>
  <div class="callout">
    <b>The off-wall (lumen) curve is a diagnostic, not a shipped quantity.</b>
    <code>grow_into_lumen</code> has no time axis on its own; each lumen node here inherits
    the earliest onset time of whichever wall/lumen neighbour admitted it. Verified to
    reproduce the shipped final mask exactly before timing was attached.
  </div>

  <div class="tabs">
    <!--TABS-->
  </div>

  <div class="spatial-row">
    <div class="panel-box">
      <div class="panel-head-row">
        <h2><span class="dot" style="background:var(--model-c)"></span>Model</h2>
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
    The wall score curve uses the shipped, AP-closure-calibrated onset model
    (<code>predict_wall_onset</code>). The off-wall score curve inherits whatever timing
    the diagnostic lumen extension produces, which itself inherits from the wall curve --
    so when the wall score dips, watch whether the off-wall score dips with it a few
    frames later. <code>patient032</code>'s off-wall score sits at zero the entire run:
    the lumen arm never admits a node there, which is now visible as a flat line rather
    than a single missing final-count number.
  </div>

  <p class="foot-note">
    Model: wall mask + onset from <code>scripts/predict_wall_clot.py::predict_wall_onset</code>
    (AP closure <code>C=62.42</code>, <code>da_scale=40</code>). Lumen mask from
    <code>src/core_physics/physics_lumen_model.py::grow_into_lumen</code>
    (<code>hops=2</code>, <code>speed&lt;0.2&times;u_ref</code>); lumen onset from
    <code>scripts/gen_offwall_temporal_data.py::lumen_onset_bfs</code>. Scores from
    <code>compute_clot_relaxed_metrics</code> + <code>clot_score_from_deploy_dict</code>,
    domain-restricted to wall or off-wall. GT from <code>gt_clot_phi_at_time</code> at
    every real simulated timestep.
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

  // Shared zoom/pan, so both windows always frame the same region. `k` is the extra
  // zoom multiplier on top of the fit-to-vessel base transform; panX/panY are canvas-
  // pixel offsets. Reset on vessel change (different geometry), kept across time frames.
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
    return {
      x: (cx - rect.left) * (canvas.width / rect.width),
      y: (cy - rect.top) * (canvas.height / rect.height),
    };
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

    // touch: one-finger pan, two-finger pinch zoom
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
    const r = Math.round(a[0] + (b[0] - a[0]) * t);
    const g = Math.round(a[1] + (b[1] - a[1]) * t);
    const bl = Math.round(a[2] + (b[2] - a[2]) * t);
    return 'rgb(' + r + ',' + g + ',' + bl + ')';
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
