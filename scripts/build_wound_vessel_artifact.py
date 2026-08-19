"""Assemble outputs/wound_vessel_temporal_data.json into the standard two-window report."""
from __future__ import annotations

import json
from pathlib import Path

DATA_PATH = Path("outputs/wound_vessel_temporal_data.json")
OUT_PATH = Path("outputs/wound_vessel_temporal_report.html")


def main() -> None:
    payload = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    if not payload:
        raise SystemExit("[ERR] empty payload; run scripts/gen_wound_vessel_viz_data.py first")
    data_json = json.dumps(payload).replace("</", "<\\/")
    order = list(payload.keys())
    tabs = "\n".join(
        f'<button class="tab{" active" if i == 0 else ""}" data-vessel="{v}">{v}</button>'
        for i, v in enumerate(order)
    )
    html = (
        TEMPLATE.replace("__DATA__", data_json)
        .replace("<!--TABS-->", tabs)
        .replace("__FIRST__", order[0])
        .replace("__N_FRAMES__", str(len(next(iter(payload.values()))["frame_t"])))
    )
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(html, encoding="utf-8")
    print(f"[save] {OUT_PATH}  ({OUT_PATH.stat().st_size / 1024:.0f} KB)")


TEMPLATE = r"""<title>Wound vessels &mdash; wall identity vs GT clot</title>
<style>
:root {
  --bg: #f2f4f6; --surface: #ffffff; --surface-2: #eaeef1; --ink: #16232b; --muted: #55707d;
  --line: #dde3e7; --accent: #1f6f78; --accent-ink: #ffffff;
  --model-c: #1f6f78; --gt-c: #b3453e;
  --model-far: #a8d6cf; --gt-far: #e8b3a6;
  --score-wall: #1f6f78; --score-off: #6a4fb8;
  --wound-c: #c47a12; --inlet-c: #2f6f3e; --outlet-c: #3d5a99;
  --warn: #b3453e;
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
    --wound-c: #e0a24a; --inlet-c: #6ec287; --outlet-c: #7ea0e0;
    --warn: #ef7f77;
    --shadow: 0 1px 2px rgba(0,0,0,0.4), 0 12px 28px -14px rgba(0,0,0,0.6);
  }
}
* { box-sizing: border-box; }
body { background: var(--bg); color: var(--ink); font-family: var(--sans); line-height: 1.5; -webkit-font-smoothing: antialiased; }
main { max-width: 1220px; margin: 0 auto; padding: 3.2rem 1.5rem 5rem; }
.eyebrow { font-family: var(--mono); font-size: 0.72rem; letter-spacing: 0.14em; text-transform: uppercase; color: var(--accent); margin: 0 0 0.9rem; }
h1 { font-family: var(--serif); font-weight: 600; font-size: clamp(1.7rem, 3.1vw, 2.4rem); line-height: 1.15; letter-spacing: -0.01em; text-wrap: balance; margin: 0 0 0.9rem; max-width: 32ch; }
.lede { font-size: 1.02rem; color: var(--muted); max-width: 70ch; margin: 0 0 1.1rem; }
.lede strong { color: var(--ink); font-weight: 600; }
.lede code { font-family: var(--mono); background: var(--surface-2); padding: 0.06rem 0.32rem; border-radius: 4px; font-size: 0.88em; }
.callout { background: var(--surface); border: 1px solid var(--line); border-left: 3px solid var(--warn); border-radius: 8px; padding: 0.95rem 1.2rem; font-size: 0.87rem; color: var(--ink); max-width: 76ch; margin: 0 0 1rem; }
.callout b { color: var(--warn); }
.callout.ok { border-left-color: var(--accent); }
.callout.ok b { color: var(--accent); }
.tabs { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 1.4rem; }
.tab { font-family: var(--mono); font-size: 0.78rem; padding: 0.5rem 0.9rem; border-radius: 7px; border: 1px solid var(--line); background: var(--surface); color: var(--muted); cursor: pointer; }
.tab:hover { border-color: var(--accent); color: var(--ink); }
.tab.active { background: var(--accent); border-color: var(--accent); color: var(--accent-ink); font-weight: 600; }
.tab.warn { border-color: var(--warn); }
.stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.7rem; margin: 0 0 1.2rem; }
.stat { background: var(--surface); border: 1px solid var(--line); border-radius: 8px; padding: 0.7rem 0.85rem; }
.stat .k { font-family: var(--mono); font-size: 0.66rem; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); }
.stat .v { font-family: var(--mono); font-variant-numeric: tabular-nums; font-size: 1.15rem; font-weight: 700; margin-top: 0.15rem; }
.stat.warn .v { color: var(--warn); }
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
.score-badge { font-family: var(--mono); font-variant-numeric: tabular-nums; font-size: 1.15rem; font-weight: 700; padding: 0.35rem 0.75rem; border-radius: 8px; background: var(--surface-2); border: 1px solid var(--line); white-space: nowrap; }
.score-badge.wall { color: var(--score-wall); }
.score-badge.off { color: var(--score-off); }
.score-row { display: grid; grid-template-columns: 1fr 1fr; gap: 1.1rem; margin-bottom: 1rem; }
.chart-canvas { aspect-ratio: 16 / 9; }
.transport { display: flex; align-items: center; gap: 0.9rem; margin: 0.9rem 0 1.8rem; }
.play-btn { font-family: var(--mono); font-size: 0.85rem; padding: 0.5rem 0.9rem; border-radius: 7px; border: 1px solid var(--accent); background: var(--accent); color: var(--accent-ink); cursor: pointer; flex: none; min-width: 4.6rem; }
.reset-btn { font-family: var(--mono); font-size: 0.78rem; padding: 0.5rem 0.8rem; border-radius: 7px; border: 1px solid var(--line); background: var(--surface); color: var(--muted); cursor: pointer; flex: none; }
.reset-btn:hover { border-color: var(--accent); color: var(--ink); }
input[type="range"] { flex: 1; accent-color: var(--accent); }
.time-readout { font-family: var(--mono); font-variant-numeric: tabular-nums; font-size: 0.76rem; color: var(--muted); min-width: 9rem; text-align: right; flex: none; }
.time-readout b { color: var(--ink); }
.legend { display: flex; flex-wrap: wrap; gap: 1.1rem; align-items: center; padding: 0.85rem 1.1rem; background: var(--surface); border: 1px solid var(--line); border-radius: 8px; margin: 0 0 2.4rem; font-size: 0.8rem; }
.legend-item { display: flex; align-items: center; gap: 0.45rem; }
.swatch { width: 11px; height: 11px; border-radius: 50%; flex: none; }
.swatch.sq { border-radius: 2px; }
.swatch.model { background: var(--model-c); } .swatch.gt { background: var(--gt-c); }
.swatch.wound { background: var(--wound-c); } .swatch.inlet { background: var(--inlet-c); } .swatch.outlet { background: var(--outlet-c); }
.grad-swatch { width: 34px; height: 11px; border-radius: 3px; flex: none; }
.grad-swatch.grad-model { background: linear-gradient(90deg, var(--model-far), var(--model-c)); }
.grad-swatch.grad-gt { background: linear-gradient(90deg, var(--gt-c), var(--gt-far)); }
.lineswatch { width: 18px; height: 0; border-top: 2px solid var(--score-wall); }
.lineswatch.off { border-top-color: var(--score-off); }
h2.section { font-family: var(--serif); font-weight: 600; font-size: 1.3rem; margin: 2.6rem 0 0.4rem; }
.finding-box { background: var(--surface); border: 1px solid var(--line); border-left: 3px solid var(--gt-c); border-radius: 8px; padding: 1rem 1.2rem; font-size: 0.9rem; color: var(--ink); max-width: 76ch; }
.finding-box b { color: var(--gt-c); }
.foot-note { margin-top: 2.6rem; padding-top: 1.5rem; border-top: 1px solid var(--line); font-size: 0.82rem; color: var(--muted); max-width: 74ch; }
.foot-note code { font-family: var(--mono); background: var(--surface-2); padding: 0.08rem 0.35rem; border-radius: 4px; font-size: 0.85em; }
@media (max-width: 760px) { .spatial-row, .score-row, .stats { grid-template-columns: 1fr 1fr; } }
</style>

<main>
  <p class="eyebrow">Wound patient vessels &middot; data health &middot; wall identity vs GT clot</p>
  <h1>Can we see the wound on the wall?</h1>
  <p class="lede">
    Two windows, one clock. <strong>Left is wall identity</strong>
    (<code>mask_wound</code> when it exists). <strong>Right is ground-truth clot</strong>
    from COMSOL <code>mu_eff</code> growth. Circles are wall, squares are lumen.
    Inlet/outlet are the small filled ticks so you can orient the vessel.
  </p>

  <div class="callout">
    <b>The wound mask is empty on these packs.</b>
    Each graph has a <code>mask_wound</code> tensor, but it is all-false.
    The meshes have no Gmsh line tags, there is no
    <code>wound_patientXXX_wound.txt</code>, and the COMSOL selection labeled
    <code>wound</code> (<code>sel1</code> in the <code>.mph</code>) never made it
    into the graph. Until that export is re-pulled, we cannot point at a wall
    node and say &ldquo;this is the wound&rdquo; from the pack itself.
  </div>
  <div class="callout ok">
    <b>Left window fallback is diagnostic, not the wound selection.</b>
    Because the mask is empty, the left window colours wall nodes by
    <code>Mat</code> at the current time (normalised by that vessel&rsquo;s
    final wall max). Wound physics dumps Mat through <code>WoundFlux_9spec</code>,
    so a compact Mat patch is a useful hint &mdash; it is not
    <code>data.mask_wound</code>.
  </div>

  <div class="tabs">
    <!--TABS-->
  </div>
  <div class="stats">
    <div class="stat" id="stat-wound"><div class="k">wound nodes</div><div class="v" id="val-wound">-</div></div>
    <div class="stat"><div class="k">wall / inlet / outlet</div><div class="v" id="val-bounds">-</div></div>
    <div class="stat"><div class="k">final clot (wall / lumen)</div><div class="v" id="val-clot">-</div></div>
    <div class="stat"><div class="k">mass-flux imbalance</div><div class="v" id="val-flux">-</div></div>
  </div>

  <div class="spatial-row">
    <div class="panel-box">
      <div class="panel-head-row">
        <h2><span class="dot" style="background:var(--model-c)"></span><span id="left-title">Wall identity</span></h2>
        <div class="panel-meta">
          <span class="score-badge wall" id="val-left">-</span>
          <span class="zoom-hint" id="zoom-readout">1.0&times;</span>
        </div>
      </div>
      <canvas id="canvas-model" class="spatial-canvas" width="560" height="560"></canvas>
    </div>
    <div class="panel-box">
      <div class="panel-head-row">
        <h2><span class="dot" style="background:var(--gt-c)"></span>Ground truth clot</h2>
        <div class="panel-meta">
          <span class="score-badge off" id="val-off">-</span>
          <span class="zoom-hint">scroll/drag, synced</span>
        </div>
      </div>
      <canvas id="canvas-gt" class="spatial-canvas" width="560" height="560"></canvas>
    </div>
  </div>

  <div class="transport">
    <button class="play-btn" id="play-btn">&#9654; Play</button>
    <button class="reset-btn" id="reset-zoom-btn" title="Reset zoom/pan (or double-click either window)">Reset view</button>
    <input type="range" id="frame-slider" min="0" max="12" value="0" step="1" />
    <div class="time-readout" id="time-readout">t = 0 s <b>(0%)</b></div>
  </div>

  <div class="score-row">
    <div class="panel-box">
      <h2>Wall clot fraction over time</h2>
      <canvas id="chart-wall" class="chart-canvas" width="620" height="349"></canvas>
    </div>
    <div class="panel-box">
      <h2>Off-wall clot fraction over time</h2>
      <canvas id="chart-off" class="chart-canvas" width="620" height="349"></canvas>
    </div>
  </div>

  <div class="legend">
    <div class="legend-item"><span class="swatch wound"></span> wound mask (empty here)</div>
    <div class="legend-item"><span class="grad-swatch grad-model"></span> wall Mat diagnostic</div>
    <div class="legend-item"><span class="swatch gt"></span> GT clot, wall</div>
    <div class="legend-item"><span class="swatch sq gt"></span> GT clot, lumen</div>
    <div class="legend-item"><span class="swatch inlet"></span> inlet</div>
    <div class="legend-item"><span class="swatch outlet"></span> outlet</div>
    <div class="legend-item">&#9679; wall &nbsp; &#9632; lumen</div>
    <div class="legend-item"><span class="lineswatch"></span> wall clot frac</div>
    <div class="legend-item"><span class="lineswatch off"></span> off-wall clot frac</div>
  </div>

  <h2 class="section">What to read off this</h2>
  <div class="finding-box" id="finding">
    If the left window lights up a compact wall patch as time advances, that is
    where Mat is accumulating &mdash; the wound-flux signature, not a stored
    wound mask. The right window should start empty and grow a clot that
    overlaps that patch. A healthy extract has inlet/outlet on opposite ends,
    a connected wall, and a non-zero final clot. None of these packs currently
    satisfy &ldquo;we know which wall nodes are the wound.&rdquo;
  </div>

  <p class="foot-note">
    Graphs: <code>data/processed/graphs_biochem_anchors/wound_patient00{1,2,3}.pt</code>.
    GT clot from <code>gt_clot_phi_at_time</code> (growth of capped <code>mu_eff</code>).
    Mat from <code>y</code> channel <code>Mat</code>. Identity from
    <code>mask_wound</code> / <code>mask_wall</code> / <code>mask_inlet</code> /
    <code>mask_outlet</code>. This page is a data-health check, not a model score.
    To actually populate the wound mask, re-pull the COMSOL <code>wound</code>
    selection into <code>*_wound.txt</code> and re-extract.
  </p>
</main>

<script id="viz-data" type="application/json">__DATA__</script>
<script>
(function () {
  const DATA = JSON.parse(document.getElementById('viz-data').textContent);
  const order = Object.keys(DATA);
  let vessel = "__FIRST__";
  let frame = 0;
  let playing = false;
  let timer = null;
  const nFrames = DATA[vessel].frame_t.length;

  const canvasModel = document.getElementById('canvas-model');
  const canvasGt = document.getElementById('canvas-gt');
  const cmModel = canvasModel.getContext('2d');
  const cmGt = canvasGt.getContext('2d');
  const wallChart = document.getElementById('chart-wall');
  const offChart = document.getElementById('chart-off');
  const wctx = wallChart.getContext('2d');
  const octx = offChart.getContext('2d');
  const slider = document.getElementById('frame-slider');
  slider.max = String(nFrames - 1);
  const readout = document.getElementById('time-readout');
  const playBtn = document.getElementById('play-btn');
  const resetZoomBtn = document.getElementById('reset-zoom-btn');
  const zoomReadout = document.getElementById('zoom-readout');
  const valLeft = document.getElementById('val-left');
  const valOff = document.getElementById('val-off');
  const leftTitle = document.getElementById('left-title');

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
    zoomReadout.textContent = '1.0x';
    redraw();
  }

  function zoomAt(canvas, screenX, screenY, factor) {
    const newK = Math.max(MIN_K, Math.min(MAX_K, view.k * factor));
    const worldX = (screenX - view.panX) / view.k;
    const worldY = (screenY - view.panY) / view.k;
    view.panX = screenX - worldX * newK;
    view.panY = screenY - worldY * newK;
    view.k = newK;
    zoomReadout.textContent = view.k.toFixed(1) + 'x';
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
    const r = Math.round(a[0] + (b[0] - a[0]) * t);
    const g = Math.round(a[1] + (b[1] - a[1]) * t);
    const bl = Math.round(a[2] + (b[2] - a[2]) * t);
    return 'rgb(' + r + ',' + g + ',' + bl + ')';
  }

  function worldMap(canvas, d) {
    const w = canvas.width, h = canvas.height;
    const all = d.bg.concat(d.wall_pos).concat(d.lumen_pos).concat(d.inlet_pos).concat(d.outlet_pos);
    const [x0, x1, y0, y1] = bbox(all);
    const pad = 24;
    const sx = (w - 2 * pad) / Math.max(x1 - x0, 1e-9);
    const sy = (h - 2 * pad) / Math.max(y1 - y0, 1e-9);
    const s = Math.min(sx, sy);
    const ox = pad + ((w - 2 * pad) - s * (x1 - x0)) / 2;
    const oy = pad + ((h - 2 * pad) - s * (y1 - y0)) / 2;
    return {
      px: (x) => (ox + (x - x0) * s) * view.k + view.panX,
      py: (y) => (oy + (y1 - y) * s) * view.k + view.panY,
    };
  }

  function drawPorts(ctx, map, d) {
    ctx.globalAlpha = 1;
    ctx.fillStyle = css('--inlet-c');
    for (const [x, y] of d.inlet_pos) {
      ctx.beginPath(); ctx.arc(map.px(x), map.py(y), 3.4, 0, Math.PI * 2); ctx.fill();
    }
    ctx.fillStyle = css('--outlet-c');
    for (const [x, y] of d.outlet_pos) {
      ctx.beginPath(); ctx.arc(map.px(x), map.py(y), 3.4, 0, Math.PI * 2); ctx.fill();
    }
  }

  function drawLeft(ctx, canvas) {
    const d = DATA[vessel];
    const w = canvas.width, h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = css('--surface-2');
    ctx.fillRect(0, 0, w, h);
    const map = worldMap(canvas, d);

    ctx.fillStyle = css('--muted');
    ctx.globalAlpha = 0.18;
    for (const [x, y] of d.bg) { ctx.beginPath(); ctx.arc(map.px(x), map.py(y), 1.1, 0, Math.PI * 2); ctx.fill(); }

    ctx.globalAlpha = 1;
    const mat = d.frame_mat_wall[frame];
    const near = css('--model-c'), far = css('--model-far');
    const woundC = css('--wound-c');
    for (let i = 0; i < d.wall_pos.length; i++) {
      const [x, y] = d.wall_pos[i];
      if (d.wall_is_wound[i]) {
        ctx.fillStyle = woundC;
        ctx.beginPath(); ctx.arc(map.px(x), map.py(y), 3.3, 0, Math.PI * 2); ctx.fill();
        continue;
      }
      const t = d.left_mode === 'wound_mask' ? 0.15 : Math.max(0, Math.min(1, mat[i]));
      ctx.fillStyle = lerpColor(far, near, t);
      ctx.globalAlpha = d.left_mode === 'wound_mask' ? 0.35 : (0.25 + 0.75 * t);
      ctx.beginPath(); ctx.arc(map.px(x), map.py(y), 2.6, 0, Math.PI * 2); ctx.fill();
      ctx.globalAlpha = 1;
    }
    drawPorts(ctx, map, d);
  }

  function drawGt(ctx, canvas) {
    const d = DATA[vessel];
    const w = canvas.width, h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = css('--surface-2');
    ctx.fillRect(0, 0, w, h);
    const map = worldMap(canvas, d);

    ctx.fillStyle = css('--muted');
    ctx.globalAlpha = 0.18;
    for (const [x, y] of d.bg) { ctx.beginPath(); ctx.arc(map.px(x), map.py(y), 1.1, 0, Math.PI * 2); ctx.fill(); }

    ctx.globalAlpha = 0.22;
    ctx.fillStyle = css('--gt-c');
    for (const [x, y] of d.wall_pos) {
      ctx.beginPath(); ctx.arc(map.px(x), map.py(y), 1.7, 0, Math.PI * 2); ctx.fill();
    }
    ctx.globalAlpha = 1;
    const wallHot = d.frame_gt_wall[frame];
    const lumenHot = d.frame_gt_lumen[frame];
    ctx.fillStyle = css('--gt-c');
    for (let i = 0; i < d.wall_pos.length; i++) {
      if (!wallHot[i]) continue;
      const [x, y] = d.wall_pos[i];
      ctx.beginPath(); ctx.arc(map.px(x), map.py(y), 2.9, 0, Math.PI * 2); ctx.fill();
    }
    const near = css('--gt-c'), far = css('--gt-far');
    for (let i = 0; i < d.lumen_pos.length; i++) {
      if (!lumenHot[i]) continue;
      const [x, y] = d.lumen_pos[i];
      const r = 2.9;
      ctx.fillStyle = lerpColor(near, far, d.lumen_dist[i]);
      ctx.fillRect(map.px(x) - r, map.py(y) - r, r * 2, r * 2);
    }
    drawPorts(ctx, map, d);
  }

  function drawScoreChart(ctx, canvas, series, tSeries, tFinal, curT, color) {
    const w = canvas.width, h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = css('--surface-2');
    ctx.fillRect(0, 0, w, h);
    const padL = 36, padR = 12, padT = 10, padB = 26;
    const plotW = w - padL - padR, plotH = h - padT - padB;
    const ymax = Math.max(0.05, ...series);
    function px(t) { return padL + (t / tFinal) * plotW; }
    function py(v) { return padT + plotH - Math.max(0, Math.min(1, v / ymax)) * plotH; }

    ctx.strokeStyle = css('--line'); ctx.lineWidth = 1; ctx.font = '9px ' + css('--mono'); ctx.fillStyle = css('--muted');
    for (let i = 0; i <= 4; i++) {
      const v = ymax * i / 4, y = py(v);
      ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(w - padR, y); ctx.stroke();
      ctx.fillText(v.toFixed(2), 2, y + 3);
    }
    for (let i = 0; i <= 3; i++) {
      const tv = (tFinal / 3) * i, x = px(tv);
      ctx.fillText(Math.round(tv) + 's', x - 8, h - padB + 13);
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
    const sw = scoreAtTime(d.score_t, d.clot_wall_frac, t);
    const so = scoreAtTime(d.score_t, d.clot_off_frac, t);
    valLeft.textContent = d.left_mode === 'wound_mask'
      ? ('wound ' + d.n_wound)
      : ('Mat max ' + d.mat_wall_max.toExponential(1));
    valOff.textContent = 'wall ' + sw.toFixed(3) + ' / off ' + so.toFixed(3);
    document.getElementById('val-wound').textContent = d.n_wound + (d.flags.wound_known ? '' : '  (missing)');
    document.getElementById('stat-wound').classList.toggle('warn', !d.flags.wound_known);
    document.getElementById('val-bounds').textContent = d.n_wall + ' / ' + d.n_inlet + ' / ' + d.n_outlet;
    document.getElementById('val-clot').textContent = d.n_clot_wall_final + ' / ' + d.n_clot_off_final;
    const flux = d.flux_imbalance;
    document.getElementById('val-flux').textContent = flux == null ? '-' : (100 * flux).toFixed(2) + '%';
    leftTitle.textContent = d.left_mode === 'wound_mask' ? 'Wound mask' : 'Wall Mat (diagnostic)';
  }

  function redraw() {
    const d = DATA[vessel];
    drawLeft(cmModel, canvasModel);
    drawGt(cmGt, canvasGt);
    drawScoreChart(wctx, wallChart, d.clot_wall_frac, d.score_t, d.t_final, d.frame_t[frame], css('--score-wall'));
    drawScoreChart(octx, offChart, d.clot_off_frac, d.score_t, d.t_final, d.frame_t[frame], css('--score-off'));
    updateReadout();
  }

  function setVessel(v) {
    vessel = v;
    frame = 0;
    slider.value = 0;
    view.k = 1; view.panX = 0; view.panY = 0;
    zoomReadout.textContent = '1.0x';
    document.querySelectorAll('.tab').forEach(tab => {
      tab.classList.toggle('active', tab.dataset.vessel === v);
    });
    redraw();
  }

  document.querySelectorAll('.tab').forEach(tab => {
    const d = DATA[tab.dataset.vessel];
    if (d && !d.flags.wound_known) tab.classList.add('warn');
    tab.addEventListener('click', () => { stopPlay(); setVessel(tab.dataset.vessel); });
  });

  attachZoomPan(canvasModel);
  attachZoomPan(canvasGt);
  resetZoomBtn.addEventListener('click', resetView);
  slider.addEventListener('input', () => { stopPlay(); frame = parseInt(slider.value, 10); redraw(); });

  function stopPlay() {
    playing = false; playBtn.innerHTML = '&#9654; Play';
    if (timer) { clearInterval(timer); timer = null; }
  }

  playBtn.addEventListener('click', () => {
    if (playing) { stopPlay(); return; }
    playing = true; playBtn.innerHTML = '&#10074;&#10074; Pause';
    timer = setInterval(() => {
      frame = (frame + 1) % nFrames;
      slider.value = frame;
      redraw();
    }, 750);
  });

  window.addEventListener('resize', redraw);
  setVessel(vessel);
})();
</script>
"""


if __name__ == "__main__":
    main()
