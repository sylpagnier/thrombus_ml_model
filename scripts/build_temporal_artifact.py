"""Assemble outputs/temporal_viz_data.json into an interactive time-lapse HTML report."""
from __future__ import annotations

import json
from pathlib import Path

DATA_PATH = Path("outputs/temporal_viz_data.json")
OUT_PATH = Path("outputs/phase3_temporal_report.html")

VESSEL_LABELS = {
    "patient043": "patient043 — previous project best (0.6925)",
    "patient014": "patient014 — closest timing match",
    "patient001": "patient001",
    "patient007": "patient007 — validated against raw COMSOL export",
    "patient013": "patient013 — largest deployable-arm gap",
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


TEMPLATE = r"""<title>Phase 3 &mdash; Clot Growth Over Time</title>
<style>
:root {
  --bg: #f2f4f6; --surface: #ffffff; --surface-2: #eaeef1; --ink: #16232b; --muted: #55707d;
  --line: #dde3e7; --accent: #1f6f78; --accent-ink: #ffffff;
  --tp: #1f8a4c; --fn: #c8362f; --fp: #b3730f;
  --tp-bg: #e4f3ea; --fn-bg: #fbe7e5; --fp-bg: #faedd9;
  --gt-line: #1f6f78; --model-line: #b3730f;
  --shadow: 0 1px 2px rgba(22,35,43,0.06), 0 8px 24px -12px rgba(22,35,43,0.18);
  --serif: ui-serif, "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif;
  --sans: -apple-system, BlinkMacSystemFont, "Segoe UI", "Helvetica Neue", Arial, sans-serif;
  --mono: ui-monospace, "SF Mono", "Cascadia Code", "Roboto Mono", Consolas, monospace;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0d1417; --surface: #141c20; --surface-2: #192226; --ink: #e7edef; --muted: #93a9b0;
    --line: #263136; --accent: #52b7c1; --accent-ink: #08191b;
    --tp: #3fcb78; --fn: #ff7a70; --fp: #f0b246;
    --tp-bg: #133622; --fn-bg: #3a1a17; --fp-bg: #3a2c12;
    --gt-line: #52b7c1; --model-line: #f0b246;
    --shadow: 0 1px 2px rgba(0,0,0,0.4), 0 12px 28px -14px rgba(0,0,0,0.6);
  }
}
:root[data-theme="dark"] {
  --bg: #0d1417; --surface: #141c20; --surface-2: #192226; --ink: #e7edef; --muted: #93a9b0;
  --line: #263136; --accent: #52b7c1; --accent-ink: #08191b;
  --tp: #3fcb78; --fn: #ff7a70; --fp: #f0b246;
  --tp-bg: #133622; --fn-bg: #3a1a17; --fp-bg: #3a2c12;
  --gt-line: #52b7c1; --model-line: #f0b246;
  --shadow: 0 1px 2px rgba(0,0,0,0.4), 0 12px 28px -14px rgba(0,0,0,0.6);
}
:root[data-theme="light"] {
  --bg: #f2f4f6; --surface: #ffffff; --surface-2: #eaeef1; --ink: #16232b; --muted: #55707d;
  --line: #dde3e7; --accent: #1f6f78; --accent-ink: #ffffff;
  --tp: #1f8a4c; --fn: #c8362f; --fp: #b3730f;
  --tp-bg: #e4f3ea; --fn-bg: #fbe7e5; --fp-bg: #faedd9;
  --gt-line: #1f6f78; --model-line: #b3730f;
  --shadow: 0 1px 2px rgba(22,35,43,0.06), 0 8px 24px -12px rgba(22,35,43,0.18);
}
* { box-sizing: border-box; }
body { background: var(--bg); color: var(--ink); font-family: var(--sans); line-height: 1.5; -webkit-font-smoothing: antialiased; }
main { max-width: 1180px; margin: 0 auto; padding: 3.2rem 1.5rem 5rem; }

.eyebrow { font-family: var(--mono); font-size: 0.72rem; letter-spacing: 0.14em; text-transform: uppercase; color: var(--accent); margin: 0 0 0.9rem; }
h1 { font-family: var(--serif); font-weight: 600; font-size: clamp(1.7rem, 3.1vw, 2.4rem); line-height: 1.15; letter-spacing: -0.01em; text-wrap: balance; margin: 0 0 0.9rem; max-width: 26ch; }
.lede { font-size: 1.02rem; color: var(--muted); max-width: 68ch; margin: 0 0 2.2rem; }
.lede strong { color: var(--ink); font-weight: 600; }
.lede code { font-family: var(--mono); background: var(--surface-2); padding: 0.06rem 0.32rem; border-radius: 4px; font-size: 0.88em; }

.tabs { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 1.4rem; }
.tab {
  font-family: var(--mono); font-size: 0.78rem; padding: 0.5rem 0.9rem; border-radius: 7px;
  border: 1px solid var(--line); background: var(--surface); color: var(--muted); cursor: pointer;
  transition: background 0.15s, color 0.15s, border-color 0.15s;
}
.tab:hover { border-color: var(--accent); color: var(--ink); }
.tab.active { background: var(--accent); border-color: var(--accent); color: var(--accent-ink); font-weight: 600; }

.stage {
  display: grid; grid-template-columns: 1.05fr 1fr; gap: 1.1rem; margin-bottom: 1rem;
}
.panel-box { background: var(--surface); border: 1px solid var(--line); border-radius: 12px; box-shadow: var(--shadow); padding: 1rem 1.1rem 1.2rem; }
.panel-box h2 { font-family: var(--serif); font-size: 1.02rem; font-weight: 600; margin: 0 0 0.7rem; }
canvas { width: 100%; display: block; border-radius: 8px; background: var(--surface-2); }
#spatial-canvas { aspect-ratio: 1 / 1; }
#chart-canvas { aspect-ratio: 4 / 3; }

.transport {
  display: flex; align-items: center; gap: 0.9rem; margin-top: 1.5rem;
}
.play-btn {
  font-family: var(--mono); font-size: 0.85rem; padding: 0.5rem 0.9rem; border-radius: 7px;
  border: 1px solid var(--accent); background: var(--accent); color: var(--accent-ink); cursor: pointer;
  flex: none; min-width: 4.6rem;
}
.play-btn:hover { filter: brightness(1.06); }
input[type="range"] { flex: 1; accent-color: var(--accent); }
.time-readout {
  font-family: var(--mono); font-variant-numeric: tabular-nums; font-size: 0.78rem; color: var(--muted);
  min-width: 11rem; text-align: right; flex: none;
}
.time-readout b { color: var(--ink); }

.legend { display: flex; flex-wrap: wrap; gap: 1.3rem; align-items: center; padding: 0.85rem 1.1rem; background: var(--surface); border: 1px solid var(--line); border-radius: 8px; margin: 1.3rem 0 2.4rem; font-size: 0.82rem; }
.legend-item { display: flex; align-items: center; gap: 0.5rem; }
.swatch { width: 11px; height: 11px; border-radius: 50%; flex: none; }
.swatch.tp { background: var(--tp); } .swatch.fn { background: var(--fn); } .swatch.fp { background: var(--fp); } .swatch.tn { background: var(--muted); opacity: 0.55; }
.lineswatch { width: 18px; height: 0; border-top: 2px solid var(--gt-line); }
.lineswatch.model { border-top: 2px dashed var(--model-line); }

h2.section { font-family: var(--serif); font-weight: 600; font-size: 1.3rem; margin: 2.6rem 0 0.4rem; }
.section-note { color: var(--muted); font-size: 0.92rem; max-width: 68ch; margin: 0 0 1.2rem; }
.finding-box {
  background: var(--surface); border: 1px solid var(--line); border-left: 3px solid var(--model-line);
  border-radius: 8px; padding: 1rem 1.2rem; font-size: 0.9rem; color: var(--ink); max-width: 74ch;
}
.finding-box b { color: var(--model-line); }

.foot-note { margin-top: 2.6rem; padding-top: 1.5rem; border-top: 1px solid var(--line); font-size: 0.82rem; color: var(--muted); max-width: 72ch; }
.foot-note code { font-family: var(--mono); background: var(--surface-2); padding: 0.08rem 0.35rem; border-radius: 4px; font-size: 0.85em; }

@media (max-width: 760px) { .stage { grid-template-columns: 1fr; } }
</style>

<main>
  <p class="eyebrow">Phase 3 &middot; Thrombus wall-clot model &middot; growth over time</p>
  <h1>Same endpoint, different path to get there.</h1>
  <p class="lede">
    The deployed model (gate + graph growth) has no time axis &mdash; it emits one final
    mask, which is what scored 0.86&ndash;0.91 on the sealed set. To see growth over time
    at all, this uses the alternate physics arm: COMSOL's autocatalytic surface ODE,
    integrated forward through the vessel's real simulated seconds with the shear gates
    <strong>frozen at t=0</strong>. Scrub the slider or hit play to watch both the true
    simulation (<code>GT</code>) and this integration (<code>model</code>) grow, node by
    node, on the same clock.
  </p>

  <div class="tabs">
    <!--TABS-->
  </div>

  <div class="stage">
    <div class="panel-box">
      <h2 id="spatial-title">Wall map</h2>
      <canvas id="spatial-canvas" width="600" height="600"></canvas>
    </div>
    <div class="panel-box">
      <h2>Committed wall nodes over time</h2>
      <canvas id="chart-canvas" width="600" height="450"></canvas>
    </div>
  </div>

  <div class="transport">
    <button class="play-btn" id="play-btn">&#9654; Play</button>
    <input type="range" id="frame-slider" min="0" max="12" value="0" step="1" />
    <div class="time-readout" id="time-readout">t = 0 s <b>(0%)</b></div>
  </div>

  <div class="legend">
    <div class="legend-item"><span class="swatch tp"></span> both agree: clot</div>
    <div class="legend-item"><span class="swatch fn"></span> GT clot, model not yet</div>
    <div class="legend-item"><span class="swatch fp"></span> model clot, GT not yet</div>
    <div class="legend-item"><span class="swatch tn"></span> neither</div>
    <div class="legend-item"><span class="lineswatch"></span> GT (actual growth)</div>
    <div class="legend-item"><span class="lineswatch model"></span> model (ODE, t0 gates)</div>
  </div>

  <h2 class="section">What this actually shows</h2>
  <p class="section-note">
    The deployed model's 0.86&ndash;0.91 sealed score is a <strong>final-time-only</strong>
    metric &mdash; it was never asked to get the timing right, only the terminal map.
  </p>
  <div class="finding-box">
    <b>The terminal footprint is close; the growth curve is not.</b> Across these five
    vessels the ODE-integrated model reaches most of its final committed count within the
    first 15&ndash;20% of the horizon, then goes flat. GT spreads onset across
    70&ndash;90% of the horizon on four of the five. The gates are frozen at t=0 and every
    gated node shares the same deposition rate, so once a node ignites nothing slows it
    down or staggers its neighbours &mdash; the real vessel's flow narrows as the clot
    grows and throttles the rate; this integration cannot see that. <code>patient014</code>
    is the exception (onset spread 0.19 vs 0.24) and is the only vessel where the two
    curves visually track each other.
  </div>

  <p class="foot-note">
    Model: <code>scripts/diag_ignition_timing.py</code> /
    <code>integrate_mat_trajectory</code> (an extension of
    <code>src/core_physics/physics_wall_model.py::integrate_mat</code> that keeps every
    timestep instead of only the final one). Gates from t=0 GT flow via
    <code>mls_gradient.py</code>, held fixed for the whole rollout &mdash; this is a
    diagnostic arm, not the deployed model. GT growth is
    <code>gt_clot_phi_at_time</code> evaluated at each of the vessel's real simulated
    timesteps (150s spacing, 201 steps, 30000s horizon).
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

  const spatialCanvas = document.getElementById('spatial-canvas');
  const chartCanvas = document.getElementById('chart-canvas');
  const sctx = spatialCanvas.getContext('2d');
  const cctx = chartCanvas.getContext('2d');
  const slider = document.getElementById('frame-slider');
  const readout = document.getElementById('time-readout');
  const title = document.getElementById('spatial-title');
  const playBtn = document.getElementById('play-btn');

  function css(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  function bbox(pts) {
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    for (const [x, y] of pts) {
      if (x < x0) x0 = x; if (x > x1) x1 = x;
      if (y < y0) y0 = y; if (y > y1) y1 = y;
    }
    return [x0, x1, y0, y1];
  }

  function drawSpatial() {
    const d = DATA[vessel];
    const w = spatialCanvas.width, h = spatialCanvas.height;
    sctx.clearRect(0, 0, w, h);
    sctx.fillStyle = css('--surface-2');
    sctx.fillRect(0, 0, w, h);

    const all = d.bg.concat(d.wall_pos);
    const [x0, x1, y0, y1] = bbox(all);
    const pad = 24;
    const sx = (w - 2 * pad) / Math.max(x1 - x0, 1e-9);
    const sy = (h - 2 * pad) / Math.max(y1 - y0, 1e-9);
    const s = Math.min(sx, sy);
    const ox = pad + ((w - 2 * pad) - s * (x1 - x0)) / 2;
    const oy = pad + ((h - 2 * pad) - s * (y1 - y0)) / 2;
    function px(x) { return ox + (x - x0) * s; }
    function py(y) { return oy + (y1 - y) * s; } // flip so +y is up, matches static report

    sctx.fillStyle = css('--muted');
    sctx.globalAlpha = 0.28;
    for (const [x, y] of d.bg) {
      sctx.beginPath();
      sctx.arc(px(x), py(y), 1.1, 0, Math.PI * 2);
      sctx.fill();
    }
    sctx.globalAlpha = 1;

    const gt = d.frame_gt[frame], model = d.frame_model[frame];
    const tp = css('--tp'), fn = css('--fn'), fp = css('--fp'), tn = css('--muted');
    for (let i = 0; i < d.wall_pos.length; i++) {
      const g = gt[i], m = model[i];
      let color, r;
      if (g && m) { color = tp; r = 2.6; }
      else if (g && !m) { color = fn; r = 2.6; }
      else if (!g && m) { color = fp; r = 2.6; }
      else { color = tn; r = 1.5; }
      sctx.globalAlpha = (g || m) ? 1 : 0.5;
      sctx.fillStyle = color;
      const [x, y] = d.wall_pos[i];
      sctx.beginPath();
      sctx.arc(px(x), py(y), r, 0, Math.PI * 2);
      sctx.fill();
    }
    sctx.globalAlpha = 1;
  }

  function drawChart() {
    const d = DATA[vessel];
    const w = chartCanvas.width, h = chartCanvas.height;
    cctx.clearRect(0, 0, w, h);
    cctx.fillStyle = css('--surface-2');
    cctx.fillRect(0, 0, w, h);

    const padL = 44, padR = 14, padT = 14, padB = 34;
    const plotW = w - padL - padR, plotH = h - padT - padB;
    const tMax = d.t_final;
    const yMax = Math.max(Math.max(...d.count_gt), Math.max(...d.count_model)) * 1.12;
    function px(t) { return padL + (t / tMax) * plotW; }
    function py(v) { return padT + plotH - (v / yMax) * plotH; }

    cctx.strokeStyle = css('--line');
    cctx.lineWidth = 1;
    cctx.font = '10px ' + css('--mono');
    cctx.fillStyle = css('--muted');
    for (let i = 0; i <= 4; i++) {
      const v = (yMax / 4) * i;
      const y = py(v);
      cctx.beginPath(); cctx.moveTo(padL, y); cctx.lineTo(w - padR, y); cctx.stroke();
      cctx.fillText(Math.round(v), 4, y + 3);
    }
    for (let i = 0; i <= 3; i++) {
      const tv = (tMax / 3) * i;
      const x = px(tv);
      cctx.fillText(Math.round(tv / 1000) + 'k', x - 8, h - padB + 14);
    }

    function line(series, color, dashed) {
      cctx.strokeStyle = color;
      cctx.lineWidth = 2;
      cctx.setLineDash(dashed ? [5, 4] : []);
      cctx.beginPath();
      for (let i = 0; i < d.count_t.length; i++) {
        const x = px(d.count_t[i]), y = py(series[i]);
        if (i === 0) cctx.moveTo(x, y); else cctx.lineTo(x, y);
      }
      cctx.stroke();
      cctx.setLineDash([]);
    }
    line(d.count_gt, css('--gt-line'), false);
    line(d.count_model, css('--model-line'), true);

    const curT = d.frame_t[frame];
    const cx = px(curT);
    cctx.strokeStyle = css('--ink');
    cctx.globalAlpha = 0.35;
    cctx.lineWidth = 1;
    cctx.beginPath(); cctx.moveTo(cx, padT); cctx.lineTo(cx, padT + plotH); cctx.stroke();
    cctx.globalAlpha = 1;
  }

  function updateReadout() {
    const d = DATA[vessel];
    const t = d.frame_t[frame];
    const pct = Math.round((t / d.t_final) * 100);
    const gtN = d.frame_gt[frame].filter(Boolean).length;
    const mN = d.frame_model[frame].filter(Boolean).length;
    readout.innerHTML = 't = ' + Math.round(t) + ' s <b>(' + pct + '%)</b> &middot; GT ' + gtN + ' &middot; model ' + mN;
  }

  function redraw() {
    drawSpatial();
    drawChart();
    updateReadout();
  }

  function setVessel(v) {
    vessel = v;
    frame = 0;
    slider.value = 0;
    title.textContent = 'Wall map — ' + (LABELS[v] || v);
    document.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t.dataset.vessel === v));
    redraw();
  }

  document.querySelectorAll('.tab').forEach(t => {
    t.addEventListener('click', () => { stopPlay(); setVessel(t.dataset.vessel); });
  });

  slider.addEventListener('input', () => {
    stopPlay();
    frame = parseInt(slider.value, 10);
    redraw();
  });

  function stopPlay() {
    playing = false;
    playBtn.innerHTML = '&#9654; Play';
    if (timer) { clearInterval(timer); timer = null; }
  }

  playBtn.addEventListener('click', () => {
    if (playing) { stopPlay(); return; }
    playing = true;
    playBtn.innerHTML = '&#10074;&#10074; Pause';
    timer = setInterval(() => {
      frame = (frame + 1) % 13;
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
