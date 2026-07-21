"""Presentation diagram for biochem_deploy species + clot architecture."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle, Rectangle


NAVY = "#1a2a4a"
GOLD = "#b8956a"
LIGHT_GOLD = "#f5efe6"
PALE_BLUE = "#eef2f8"
WHITE = "#ffffff"
GRAY = "#6b7280"
GREEN = "#2d6a4f"


def _box(ax, xy, w, h, text, *, fc=PALE_BLUE, ec=NAVY, fontsize=9, bold=False):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.4,
        edgecolor=ec,
        facecolor=fc,
        zorder=2,
    )
    ax.add_patch(patch)
    weight = "bold" if bold else "normal"
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=NAVY,
        weight=weight,
        zorder=3,
        wrap=True,
    )
    return patch


def _arrow(ax, start, end, *, color=NAVY, style="-|>", lw=1.3):
    arr = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=12,
        linewidth=lw,
        color=color,
        zorder=1,
    )
    ax.add_patch(arr)


def _mini_graph(ax, cx, cy, r=0.55):
    nodes = [
        (cx - 0.35, cy + 0.2),
        (cx + 0.35, cy + 0.2),
        (cx, cy - 0.25),
        (cx - 0.15, cy + 0.45),
        (cx + 0.15, cy + 0.45),
    ]
    edges = [(0, 1), (0, 2), (1, 2), (3, 0), (4, 1), (3, 4)]
    for i, j in edges:
        ax.plot(
            [nodes[i][0], nodes[j][0]],
            [nodes[i][1], nodes[j][1]],
            color=GRAY,
            lw=0.8,
            zorder=1,
        )
    for i, (x, y) in enumerate(nodes):
        active = i in (0, 3)
        ax.add_patch(
            Circle(
                (x, y),
                0.08,
                facecolor=GOLD if active else WHITE,
                edgecolor=NAVY,
                lw=1.0,
                zorder=2,
            )
        )


def draw(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 7.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7.5)
    ax.axis("off")
    fig.patch.set_facecolor(WHITE)

    ax.text(
        7.0,
        7.05,
        "Biochem deploy architecture",
        ha="center",
        va="center",
        fontsize=20,
        color=NAVY,
        family="serif",
        weight="bold",
    )

    # --- Row 1: inputs ---
    _box(ax, (0.4, 5.2), 2.4, 1.2, "PMGP-DEQ\n(frozen latent z_kin)", fc=LIGHT_GOLD, bold=True)
    _box(ax, (3.2, 5.2), 2.2, 1.2, "Wall-band mesh\n+ SDF features", fc=LIGHT_GOLD)
    _box(ax, (5.8, 5.2), 2.6, 1.2, "Restricted nucleation mask\nwall | 1-hop from commits", fc="#fff8ef", ec=GOLD, bold=True)

    _mini_graph(ax, 9.4, 5.8)
    ax.text(9.4, 5.15, "masked subgraph", ha="center", fontsize=8, color=GRAY)

    _arrow(ax, (2.8, 5.8), (3.2, 5.8))
    _arrow(ax, (5.4, 5.8), (5.8, 5.8))
    _arrow(ax, (8.4, 5.8), (8.85, 5.8))

    # --- Row 2: GraphSAGE trunk ---
    _box(ax, (1.0, 3.35), 1.5, 0.9, "SAGE L1", fc=PALE_BLUE)
    _box(ax, (2.8, 3.35), 1.5, 0.9, "SAGE L2", fc=PALE_BLUE)
    _box(ax, (4.6, 3.35), 1.5, 0.9, "SAGE L3", fc=PALE_BLUE)
    ax.text(3.55, 4.55, "3-layer GraphSAGE on restricted nucleation mask", ha="center", fontsize=10, color=NAVY, weight="bold")

    _arrow(ax, (9.4, 5.2), (1.75, 4.25))
    _arrow(ax, (2.5, 3.8), (2.8, 3.8))
    _arrow(ax, (4.3, 3.8), (4.6, 3.8))

    # skip connection
    ax.plot([3.3, 3.3, 6.2], [5.2, 2.95, 2.95], color=GOLD, lw=1.0, ls="--", zorder=1)
    ax.text(4.8, 3.05, "skip concat", ha="center", fontsize=8, color=GOLD)

    # --- Row 2: dual head ---
    _box(ax, (6.2, 3.15), 2.3, 1.3, "Spatial gate\nsigmoid(logits)", fc="#e8f4ec", ec=GREEN)
    _box(ax, (8.8, 3.15), 2.3, 1.3, "Magnitude delta\nsoftplus(|dFI|, |dMat|)", fc="#e8f4ec", ec=GREEN)
    ax.text(8.15, 4.55, "Dual MLP head", ha="center", fontsize=10, color=GREEN, weight="bold")

    _arrow(ax, (6.1, 3.8), (6.2, 3.8))
    _arrow(ax, (8.5, 3.8), (8.8, 3.8))

    # multiply
    ax.add_patch(Circle((11.6, 3.8), 0.22, facecolor=WHITE, edgecolor=NAVY, lw=1.2, zorder=2))
    ax.text(11.6, 3.8, "x", ha="center", va="center", fontsize=12, color=NAVY, weight="bold")
    _arrow(ax, (8.5, 3.5), (11.38, 3.65))
    _arrow(ax, (11.1, 3.8), (11.38, 3.8))

    _box(ax, (12.1, 3.35), 1.5, 0.9, "d species\n(FI, Mat)", fc=PALE_BLUE, bold=True)

    _arrow(ax, (11.82, 3.8), (12.1, 3.8))

    # --- Row 3: time unroll ---
    ax.add_patch(
        Rectangle((0.5, 1.55), 12.8, 1.35, linewidth=1.2, edgecolor=GOLD, facecolor="#fffdf9", zorder=0)
    )
    ax.text(0.75, 2.65, "Autoregressive time unroll  t -> t+dt -> ... -> T", fontsize=9, color=GOLD, weight="bold")

    _box(ax, (0.8, 1.75), 2.0, 0.95, "Species state\nlog(FI), log(Mat)", fc=WHITE)
    _box(ax, (3.1, 1.75), 2.4, 0.95, "gelation_beta\n(global Mat scale)", fc=LIGHT_GOLD)
    _box(ax, (5.9, 1.75), 2.8, 0.95, "Analytical closure\nCarreau mu + nucleation phi", fc=LIGHT_GOLD, bold=True)
    _box(ax, (9.1, 1.75), 1.8, 0.95, "Clot map phi", fc=WHITE, bold=True)
    _box(ax, (11.2, 1.75), 1.9, 0.95, "Huber loss\nvs GT deltas", fc="#fde8e8", ec="#9b2c2c")

    _arrow(ax, (12.85, 3.8), (1.8, 2.7))
    _arrow(ax, (2.8, 2.22), (3.1, 2.22))
    _arrow(ax, (5.5, 2.22), (5.9, 2.22))
    _arrow(ax, (8.7, 2.22), (9.1, 2.22))
    _arrow(ax, (10.9, 2.22), (11.2, 2.22))

    # feedback loop
    ax.annotate(
        "",
        xy=(0.8, 2.0),
        xytext=(12.0, 1.55),
        arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1.0, connectionstyle="arc3,rad=-0.35"),
    )
    ax.text(6.5, 1.15, "state feeds next step (mask updates from predicted phi)", ha="center", fontsize=8, color=GRAY)

    # legend bullets matching slide
    bullets = [
        "Learned: GraphSAGE trunk + dual MLP (spatial gate x magnitude)",
        "Physics: species -> mu -> clot via fixed analytical equations",
        "Train: Huber on species deltas, unrolled over COMSOL timeline",
    ]
    for i, b in enumerate(bullets):
        ax.plot([0.55, 0.75], [0.75 - 0.28 * i, 0.75 - 0.28 * i], color=GOLD, lw=3, solid_capstyle="round")
        ax.text(0.9, 0.75 - 0.28 * i, b, ha="left", va="center", fontsize=9, color=NAVY)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[save] {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/biochem/viz/biochem_architecture_diagram.png"),
    )
    args = p.parse_args()
    draw(args.out.resolve())


if __name__ == "__main__":
    main()
