"""Assemble outputs/viz_phase6_offwall/*.png + metrics into a standalone HTML report."""
from __future__ import annotations

import base64
from pathlib import Path

IMG_DIR = Path("outputs/viz_phase6_offwall")
OUT_PATH = Path("outputs/phase6_offwall_report.html")

# (anchor, flow, off_gt, n_gt, score_wall, score_full, offwallF1_wall, offwallF1_full, note)
ROWS = [
    ("patient012", "gt",   90, 186, 0.5945, 0.6743, 0.000, 0.762, "train &middot; most off-wall clot in the cohort"),
    ("patient044", "gt",  122, 285, 0.6293, 0.7123, 0.000, 0.681, "train &middot; largest absolute off-wall count"),
    ("patient042", "gt",   78, 187, 0.6085, 0.6570, 0.000, 0.636, "sealed &middot; never trained on"),
    ("patient007", "pred", 99, 325, 0.7364, 0.7948, 0.000, 0.653, "sealed &middot; deployable flow &middot; validated against raw COMSOL export"),
    ("patient032", "pred",120, 313, 0.7547, 0.7547, 0.000, 0.000, "train &middot; deployable flow &middot; the lumen arm found nothing here"),
]


def main() -> None:
    cards = []
    for anchor, flow, off_gt, n_gt, sw, sf, ow, of, note in ROWS:
        img_b64 = base64.b64encode((IMG_DIR / f"{anchor}.png").read_bytes()).decode("ascii")
        delta = sf - sw
        flow_label = "deployable (predicted flow)" if flow == "pred" else "GT t=0 flow (bandaid)"
        cards.append(f"""
        <article class="vessel-card">
          <header class="vessel-head">
            <h3>{anchor}</h3>
            <span class="flow-badge {'flow-pred' if flow == 'pred' else 'flow-gt'}">{flow_label}</span>
          </header>
          <p class="card-note">{note}</p>
          <img class="vessel-img" src="data:image/png;base64,{img_b64}" alt="{anchor} wall vs wall+lumen" loading="lazy" />
          <div class="metric-row">
            <div class="metric">
              <span class="metric-label">off-wall share of GT clot</span>
              <span class="metric-value">{off_gt}<span class="metric-sub">/{n_gt} nodes ({100*off_gt/n_gt:.0f}%)</span></span>
            </div>
            <div class="metric">
              <span class="metric-label">full-mesh score</span>
              <span class="metric-value">{sw:.3f} <span class="arrow">&rarr;</span> <b class="{'gain' if delta > 0.0005 else 'flat'}">{sf:.3f}</b>
                <span class="metric-sub">({'+' if delta >= 0 else ''}{delta:.4f})</span></span>
            </div>
            <div class="metric">
              <span class="metric-label">off-wall relaxed F1</span>
              <span class="metric-value">{ow:.3f} <span class="arrow">&rarr;</span> <b class="{'gain' if of > ow + 0.0005 else 'flat'}">{of:.3f}</b></span>
            </div>
          </div>
        </article>""")

    html = TEMPLATE.replace("<!--CARDS-->", "\n".join(cards))
    OUT_PATH.write_text(html, encoding="utf-8")
    print(f"wrote {OUT_PATH}  ({OUT_PATH.stat().st_size/1024:.0f} KB)")


TEMPLATE = r"""<title>Phase 6 &mdash; Off-Wall Clot</title>
<style>
:root {
  --bg: #f2f4f6; --surface: #ffffff; --surface-2: #eaeef1; --ink: #16232b; --muted: #55707d;
  --line: #dde3e7; --accent: #1f6f78; --accent-ink: #ffffff;
  --tp: #1f8a4c; --fn: #c8362f; --fp: #b3730f;
  --gain: #1f8a4c; --shadow: 0 1px 2px rgba(22,35,43,0.06), 0 8px 24px -12px rgba(22,35,43,0.18);
  --serif: ui-serif, "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif;
  --sans: -apple-system, BlinkMacSystemFont, "Segoe UI", "Helvetica Neue", Arial, sans-serif;
  --mono: ui-monospace, "SF Mono", "Cascadia Code", "Roboto Mono", Consolas, monospace;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0d1417; --surface: #141c20; --surface-2: #192226; --ink: #e7edef; --muted: #93a9b0;
    --line: #263136; --accent: #52b7c1; --accent-ink: #08191b;
    --tp: #3fcb78; --fn: #ff7a70; --fp: #f0b246; --gain: #3fcb78;
    --shadow: 0 1px 2px rgba(0,0,0,0.4), 0 12px 28px -14px rgba(0,0,0,0.6);
  }
}
:root[data-theme="dark"] {
  --bg: #0d1417; --surface: #141c20; --surface-2: #192226; --ink: #e7edef; --muted: #93a9b0;
  --line: #263136; --accent: #52b7c1; --accent-ink: #08191b;
  --tp: #3fcb78; --fn: #ff7a70; --fp: #f0b246; --gain: #3fcb78;
  --shadow: 0 1px 2px rgba(0,0,0,0.4), 0 12px 28px -14px rgba(0,0,0,0.6);
}
:root[data-theme="light"] {
  --bg: #f2f4f6; --surface: #ffffff; --surface-2: #eaeef1; --ink: #16232b; --muted: #55707d;
  --line: #dde3e7; --accent: #1f6f78; --accent-ink: #ffffff;
  --tp: #1f8a4c; --fn: #c8362f; --fp: #b3730f; --gain: #1f8a4c;
  --shadow: 0 1px 2px rgba(22,35,43,0.06), 0 8px 24px -12px rgba(22,35,43,0.18);
}
* { box-sizing: border-box; }
body { background: var(--bg); color: var(--ink); font-family: var(--sans); line-height: 1.5; -webkit-font-smoothing: antialiased; }
main { max-width: 1100px; margin: 0 auto; padding: 3.4rem 1.5rem 5rem; }

.eyebrow { font-family: var(--mono); font-size: 0.72rem; letter-spacing: 0.14em; text-transform: uppercase; color: var(--accent); margin: 0 0 0.9rem; }
h1 { font-family: var(--serif); font-weight: 600; font-size: clamp(1.8rem, 3.2vw, 2.5rem); line-height: 1.15; letter-spacing: -0.01em; text-wrap: balance; margin: 0 0 0.9rem; max-width: 24ch; }
.lede { font-size: 1.02rem; color: var(--muted); max-width: 68ch; margin: 0 0 1.1rem; }
.lede strong { color: var(--ink); font-weight: 600; }
.lede code { font-family: var(--mono); background: var(--surface-2); padding: 0.06rem 0.32rem; border-radius: 4px; font-size: 0.88em; }

.callout {
  background: var(--surface); border: 1px solid var(--line); border-left: 3px solid var(--accent);
  border-radius: 8px; padding: 0.95rem 1.2rem; font-size: 0.88rem; color: var(--ink);
  max-width: 74ch; margin: 0 0 2.6rem;
}
.callout b { color: var(--accent); }

.gallery { display: flex; flex-direction: column; gap: 1.5rem; }
.vessel-card { background: var(--surface); border: 1px solid var(--line); border-radius: 12px; box-shadow: var(--shadow); padding: 1.3rem 1.4rem 1.4rem; }
.vessel-head { display: flex; align-items: baseline; justify-content: space-between; gap: 0.8rem; margin-bottom: 0.15rem; }
.vessel-head h3 { font-family: var(--mono); font-size: 1.05rem; font-weight: 600; margin: 0; }
.flow-badge { font-family: var(--mono); font-size: 0.66rem; letter-spacing: 0.02em; padding: 0.2rem 0.5rem; border-radius: 5px; white-space: nowrap; }
.flow-pred { background: var(--tp-bg, var(--surface-2)); color: var(--accent); border: 1px solid var(--accent); }
.flow-gt { background: var(--surface-2); color: var(--muted); border: 1px solid var(--line); }
.card-note { font-size: 0.82rem; color: var(--muted); font-style: italic; margin: 0.1rem 0 0.9rem; }
.vessel-img { width: 100%; height: auto; display: block; border-radius: 8px; border: 1px solid var(--line); margin-bottom: 1rem; }

.metric-row { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.8rem; }
.metric { background: var(--surface-2); border-radius: 8px; padding: 0.65rem 0.85rem; }
.metric-label { display: block; font-family: var(--mono); font-size: 0.62rem; letter-spacing: 0.03em; text-transform: uppercase; color: var(--muted); margin-bottom: 0.3rem; }
.metric-value { font-family: var(--mono); font-variant-numeric: tabular-nums; font-size: 0.95rem; font-weight: 600; }
.metric-value b.gain { color: var(--gain); }
.metric-value b.flat { color: var(--ink); }
.metric-sub { font-size: 0.72rem; color: var(--muted); font-weight: 400; margin-left: 0.2rem; }
.arrow { color: var(--muted); margin: 0 0.15rem; }

.legend-note { display: flex; flex-wrap: wrap; gap: 1.2rem; align-items: center; padding: 0.85rem 1.1rem; background: var(--surface); border: 1px solid var(--line); border-radius: 8px; margin: 1.6rem 0 0; font-size: 0.8rem; color: var(--muted); }
.legend-note b { color: var(--ink); }

.foot-note { margin-top: 2.6rem; padding-top: 1.5rem; border-top: 1px solid var(--line); font-size: 0.82rem; color: var(--muted); max-width: 72ch; }
.foot-note code { font-family: var(--mono); background: var(--surface-2); padding: 0.08rem 0.35rem; border-radius: 4px; font-size: 0.85em; }
</style>

<main>
  <p class="eyebrow">Phase 6 &middot; wall + lumen clot model &middot; off-wall generalization</p>
  <h1>The wall isn't the whole clot.</h1>
  <p class="lede">
    Up to <strong>48% of a vessel's ground-truth clot forms off the wall</strong>, in the
    stagnant lumen just behind committed wall tissue &mdash; and every score this project
    reported before now was computed on a wall-only subset that excluded it entirely. The
    Phase&nbsp;6 model (<code>scripts/predict_wall_clot.py</code>) adds a second, still
    zero-learned-parameter arm: it propagates the wall prediction into adjacent lumen nodes
    wherever local flow speed is low enough to let platelets settle. Below, five vessels
    with substantial off-wall clot, each scored on the <strong>full mesh</strong>, wall arm
    alone vs. wall + lumen.
  </p>

  <div class="callout">
    <b>What's genuinely new here</b> is not the wall prediction &mdash; that's the same
    t=0 gates + graph growth from Phase 3. It's the second propagation step: take the wall
    mask, and admit an off-wall neighbour within 2 graph hops when its t=0 speed is below
    <code>0.2&times;u_ref</code>. Two scalars, fit once on train, unchanged since.
  </div>

  <div class="gallery">
    <!--CARDS-->
  </div>

  <p class="legend-note">
    <span><b>&#9679; circle</b> = wall node</span>
    <span><b>&#9632; square</b> = off-wall (lumen) node</span>
    <span style="color:var(--tp)">&#9679; green = correct</span>
    <span style="color:var(--fn)">&#9679; red = missed</span>
    <span style="color:var(--fp)">&#9679; amber = false positive</span>
  </p>

  <p class="foot-note">
    Model: <code>predict_wall_clot(data, bio, flow=..., lumen=True)</code> &mdash; wall arm
    from <code>src/core_physics/physics_wall_model.py</code>, lumen arm from
    <code>src/core_physics/physics_lumen_model.py::grow_into_lumen</code>
    (<code>LUMEN_HOPS=2</code>, <code>LUMEN_SPEED=0.2</code>). Scored full-mesh via
    <code>compute_clot_relaxed_metrics_full_mesh</code>, not the wall-masked metric this
    project reported through Phase 3/5 &mdash; see <code>docs/PHASE6_RESULTS.md</code> &sect;20.3.
    <code>patient032</code> is included deliberately as a case where the lumen arm found
    nothing: it does not help every vessel, and PHASE6_RESULTS records its default
    threshold as net negative before recalibration (&sect;21.1).
  </p>
</main>
"""

if __name__ == "__main__":
    main()
