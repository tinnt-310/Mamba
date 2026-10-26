"""Generate publication-quality RGB--Geometry architecture figures.

Run from any directory with::

    python figures/draw_architecture.py

The PDF and SVG files remain vector graphics. PNG files are exported at
600 dpi. All layout and style parameters are collected near the top.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


# ---------------------------------------------------------------------------
# Global style and layout parameters
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path(__file__).resolve().parent
PNG_DPI = 600
FONT_FAMILY = "STIXGeneral"
FONT_SIZE = 8.2
SMALL_FONT_SIZE = 7.0
TINY_FONT_SIZE = 6.2
TITLE_FONT_SIZE = 9.0
LINE_WIDTH = 1.05
ARROW_WIDTH = 0.95
BLOCK_HEIGHT = 0.72
CORNER_RADIUS = 0.09
STAGE_GAP = 1.95
MARGINS = 0.18

COLORS = {
    "rgb": "#F8E5D6",       # warm peach
    "geometry": "#DFEEE8",  # muted green
    "computed": "#E8F0F3",  # gray-blue
    "fusion": "#DDEAF5",    # light blue
    "cross": "#F7EFCB",     # light yellow
    "shared": "#E8E1F0",    # muted lavender
    "tensor": "#F7F7F4",
    "container": "#F6F8FA",
    "ink": "#25282B",
    "muted": "#5C6268",
    "white": "#FFFFFF",
}

mpl.rcParams.update(
    {
        "font.family": FONT_FAMILY,
        "font.size": FONT_SIZE,
        "mathtext.fontset": "stix",
        "axes.linewidth": 0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    }
)


# ---------------------------------------------------------------------------
# Reusable drawing primitives
# ---------------------------------------------------------------------------
def make_canvas(width: float, height: float, xlim: tuple[float, float],
                ylim: tuple[float, float]):
    fig, ax = plt.subplots(figsize=(width, height), facecolor="white")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    return fig, ax


def draw_block(ax, x, y, w, h, label, *, fill="tensor", fontsize=FONT_SIZE,
               rounded=True, linewidth=LINE_WIDTH, linestyle="-", zorder=3,
               text_color="ink", padding=0.05):
    color = COLORS.get(fill, fill)
    if rounded:
        patch = FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f"round,pad={padding},rounding_size={CORNER_RADIUS}",
            facecolor=color, edgecolor=COLORS["ink"], linewidth=linewidth,
            linestyle=linestyle, zorder=zorder,
        )
    else:
        patch = Rectangle(
            (x, y), w, h, facecolor=color, edgecolor=COLORS["ink"],
            linewidth=linewidth, linestyle=linestyle, zorder=zorder,
        )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=fontsize, color=COLORS.get(text_color, text_color),
            linespacing=1.05, zorder=zorder + 1)
    return patch


def draw_container(ax, x, y, w, h, label, *, fill="container",
                   label_side="top", fontsize=SMALL_FONT_SIZE):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.04,rounding_size={CORNER_RADIUS}",
        facecolor=COLORS.get(fill, fill), edgecolor=COLORS["muted"],
        linewidth=LINE_WIDTH, linestyle=(0, (4, 3)), zorder=0,
    )
    ax.add_patch(patch)
    if label_side == "top":
        tx, ty, ha, va = x + 0.14, y + h - 0.12, "left", "top"
    else:
        tx, ty, ha, va = x + 0.14, y + 0.12, "left", "bottom"
    ax.text(tx, ty, label, ha=ha, va=va, fontsize=fontsize,
            fontweight="bold", color=COLORS["muted"], zorder=5)
    return patch


def draw_arrow(ax, start, end, *, label=None, label_offset=(0, 0.12),
               color="ink", linewidth=ARROW_WIDTH, style="-|>",
               connectionstyle="arc3", zorder=2, mutation_scale=8.5):
    arrow = FancyArrowPatch(
        start, end, arrowstyle=style, mutation_scale=mutation_scale,
        linewidth=linewidth, color=COLORS.get(color, color),
        connectionstyle=connectionstyle, shrinkA=1.5, shrinkB=1.5,
        zorder=zorder,
    )
    ax.add_patch(arrow)
    if label:
        mx = (start[0] + end[0]) / 2 + label_offset[0]
        my = (start[1] + end[1]) / 2 + label_offset[1]
        ax.text(mx, my, label, ha="center", va="center",
                fontsize=SMALL_FONT_SIZE, color=COLORS["ink"], zorder=6)
    return arrow


def draw_poly_arrow(ax, points: Iterable[tuple[float, float]], *, label=None,
                    label_at=None, color="ink", linewidth=ARROW_WIDTH,
                    linestyle="-", zorder=1):
    points = list(points)
    for a, b in zip(points[:-2], points[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=COLORS.get(color, color),
                linewidth=linewidth, linestyle=linestyle, zorder=zorder)
    draw_arrow(ax, points[-2], points[-1], color=color, linewidth=linewidth,
               zorder=zorder)
    if label and label_at:
        ax.text(*label_at, label, ha="center", va="center",
                fontsize=SMALL_FONT_SIZE, color=COLORS["ink"], zorder=6)


def draw_orthogonal_arrow(ax, points: Iterable[tuple[float, float]], **kwargs):
    """Draw a connector made only from horizontal and vertical segments."""
    points = list(points)
    if len(points) < 2:
        raise ValueError("An orthogonal connector needs at least two points")
    for start, end in zip(points, points[1:]):
        if start[0] != end[0] and start[1] != end[1]:
            raise ValueError(f"Non-orthogonal connector segment: {start} -> {end}")
    return draw_poly_arrow(ax, points, **kwargs)


def draw_operator(ax, x, y, symbol, *, radius=0.17, fill="white",
                  fontsize=FONT_SIZE):
    circle = Circle((x, y), radius, facecolor=COLORS.get(fill, fill),
                    edgecolor=COLORS["ink"], linewidth=LINE_WIDTH, zorder=4)
    ax.add_patch(circle)
    ax.text(x, y, symbol, ha="center", va="center", fontsize=fontsize,
            color=COLORS["ink"], fontweight="bold", zorder=5)
    return circle


def draw_feature_map(ax, x, y, w, h, label, *, fill="tensor",
                     fontsize=SMALL_FONT_SIZE):
    color = COLORS.get(fill, fill)
    offsets = (0.10, 0.05, 0.0)
    for idx, off in enumerate(offsets):
        rect = Rectangle((x + off, y + off), w, h, facecolor=color,
                         edgecolor=COLORS["ink"], linewidth=0.8,
                         alpha=0.82 + 0.09 * idx, zorder=2 + idx)
        ax.add_patch(rect)
    ax.text(x + w / 2 + 0.04, y + h / 2 + 0.03, label,
            ha="center", va="center", fontsize=fontsize, zorder=6)


def draw_tied_marker(ax, x, y, text="shared weights"):
    ax.text(x, y, text, ha="center", va="center", fontsize=TINY_FONT_SIZE,
            color=COLORS["muted"], style="italic", zorder=7)


def export_figure(fig, stem: str):
    for suffix in ("pdf", "svg"):
        fig.savefig(OUTPUT_DIR / f"{stem}.{suffix}", bbox_inches="tight",
                    pad_inches=MARGINS, facecolor="white")
    fig.savefig(OUTPUT_DIR / f"{stem}.png", dpi=PNG_DPI,
                bbox_inches="tight", pad_inches=MARGINS, facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1: Overall architecture
# ---------------------------------------------------------------------------
def draw_main_architecture():
    fig, ax = make_canvas(7.15, 4.45, (0, 15.6), (0, 9.5))

    # Conceptual regions.
    draw_container(ax, 0.18, 6.75, 15.15, 2.28,
                   "RGB feature extraction", fill="#FFF9F4")
    draw_container(ax, 0.18, 0.32, 15.15, 2.45,
                   "Deterministic geometry + learned geometry encoder",
                   fill="#F5FAF8", label_side="top")
    draw_container(ax, 3.85, 3.18, 11.48, 3.08,
                   "", fill="#F5F8FB")
    ax.text(14.18, 6.04, "Scale-aligned\nMS2Fusion",
            ha="center", va="center", fontsize=SMALL_FONT_SIZE,
            fontweight="bold", color=COLORS["muted"], linespacing=0.9)

    # Inputs and compact preprocessing.
    draw_block(ax, 0.42, 7.37, 1.10, 0.76, "RGB\nimage", fill="rgb")
    draw_block(ax, 0.35, 1.40, 1.12, 0.56, "Depth\n$D(u,v)$", fill="geometry",
               fontsize=TINY_FONT_SIZE)
    draw_block(ax, 0.35, 0.66, 1.12, 0.48, "Intrinsics $K$", fill="computed",
               fontsize=SMALL_FONT_SIZE)
    draw_operator(ax, 1.76, 1.40, "+", radius=0.17)
    draw_orthogonal_arrow(
        ax, [(1.47, 1.68), (1.76, 1.68), (1.76, 1.57)]
    )
    draw_orthogonal_arrow(
        ax, [(1.47, 0.90), (1.76, 0.90), (1.76, 1.23)]
    )
    draw_block(ax, 2.02, 0.96, 1.62, 0.88,
               "Geometry prep.\n$XYZ, N, M$", fill="computed",
               fontsize=5.7)
    draw_orthogonal_arrow(ax, [(1.93, 1.40), (2.02, 1.40)])

    stage_x = [4.35 + i * STAGE_GAP for i in range(5)]
    stage_names = ["Stem", "Stage 1", "Stage 2", "Stage 3", "Stage 4"]
    reductions = [2, 4, 8, 16, 32]
    rgb_channels = [64, 256, 512, 1024, 2048]
    geo_channels = [32, 48, 96, 128, 192]

    # RGB and geometry encoders, with stage-to-stage flow.
    for i, x in enumerate(stage_x):
        draw_block(ax, x, 7.32, 1.18, 0.80,
                   f"{stage_names[i]}\n$R_{i}$  /{reductions[i]}", fill="rgb",
                   fontsize=SMALL_FONT_SIZE)
        draw_block(ax, x, 1.01, 1.18, 0.94,
                   f"{stage_names[i]}\n$G_{i}$  /{reductions[i]}\n{geo_channels[i]} ch", fill="geometry",
                   fontsize=TINY_FONT_SIZE)
        ax.text(x + 0.59, 8.28, f"{rgb_channels[i]} ch", ha="center",
                va="center", fontsize=TINY_FONT_SIZE, color=COLORS["muted"])
        if i:
            draw_orthogonal_arrow(
                ax, [(stage_x[i - 1] + 1.18, 7.72), (x, 7.72)]
            )
            draw_orthogonal_arrow(
                ax, [(stage_x[i - 1] + 1.18, 1.48), (x, 1.48)]
            )

    draw_orthogonal_arrow(
        ax, [(1.52, 7.75), (stage_x[0], 7.75)],
        label="SE-ResNet-50", label_at=(2.94, 7.98)
    )
    draw_orthogonal_arrow(
        ax, [(3.64, 1.40), (3.98, 1.40), (3.98, 1.48),
             (stage_x[0], 1.48)]
    )

    # One aligned projection/fusion unit per scale.
    for i, x in enumerate(stage_x):
        cx = x + 0.38
        draw_block(ax, cx - 0.38, 5.52, 0.76, 0.48,
                   "RGB $1\\times1$", fill="rgb", fontsize=5.4)
        draw_block(ax, cx - 0.38, 3.43, 0.76, 0.48,
                   "Geo $1\\times1$", fill="geometry", fontsize=5.4)
        draw_block(ax, cx - 0.50, 4.26, 1.00, 0.66,
                   f"MS2\nFusion-{i}", fill="fusion", fontsize=TINY_FONT_SIZE)
        draw_orthogonal_arrow(
            ax, [(x + 0.59, 7.32), (x + 0.59, 6.24),
                 (cx, 6.24), (cx, 6.00)],
            label=f"$R_{i}$", label_at=(x + 0.79, 6.68)
        )
        draw_orthogonal_arrow(
            ax, [(cx, 5.52), (cx, 4.92)],
            label=f"$R'_{i}$", label_at=(cx + 0.17, 5.22)
        )
        draw_orthogonal_arrow(
            ax, [(x + 0.59, 1.95), (x + 0.59, 3.15),
                 (cx, 3.15), (cx, 3.43)],
            label=f"$G_{i}$", label_at=(x + 0.79, 2.62)
        )
        draw_orthogonal_arrow(
            ax, [(cx, 3.91), (cx, 4.26)],
            label=f"$G'_{i}$", label_at=(cx + 0.17, 4.08)
        )
        draw_feature_map(ax, cx + 0.66, 4.31, 0.38, 0.50,
                         f"$\\mathcal{{F}}_{i}$", fill="fusion",
                         fontsize=TINY_FONT_SIZE)
        draw_orthogonal_arrow(
            ax, [(cx + 0.50, 4.59), (cx + 0.66, 4.59)]
        )

    export_figure(fig, "fig_main_architecture")


# ---------------------------------------------------------------------------
# Figure 2: Geometry branch and encoder details
# ---------------------------------------------------------------------------
def draw_geometry_branch():
    fig, ax = make_canvas(7.15, 4.75, (0, 15.5), (0, 10.0))

    draw_container(ax, 0.15, 5.35, 10.95, 4.30,
                   "A. Deterministic metric geometry", fill="#F5F9FA")
    draw_container(ax, 11.30, 5.35, 4.05, 4.30,
                   "B. Learned representation", fill="#F5FAF8")

    # Deterministic preprocessing chain.
    draw_block(ax, 0.42, 7.75, 1.32, 0.70, "Metric depth\n$D(u,v)$", fill="geometry")
    draw_block(ax, 0.42, 6.55, 1.32, 0.62, "Intrinsics $K$", fill="computed")
    draw_operator(ax, 2.15, 7.49, "+", radius=0.18)
    draw_arrow(ax, (1.74, 8.02), (1.98, 7.58))
    draw_arrow(ax, (1.74, 6.86), (2.00, 7.39))

    draw_block(ax, 2.58, 7.02, 2.25, 1.28,
               "Back-projection\n$X=(u-c_x)Z/f_x$\n$Y=(v-c_y)Z/f_y$\n$Z=D(u,v)$",
               fill="computed", fontsize=SMALL_FONT_SIZE)
    draw_arrow(ax, (2.33, 7.49), (2.58, 7.63))
    draw_feature_map(ax, 5.18, 7.23, 0.82, 0.92, "$XYZ$", fill="computed")
    draw_arrow(ax, (4.83, 7.66), (5.18, 7.66))

    draw_block(ax, 6.38, 6.93, 2.25, 1.46,
               "Central differences\n$\\mathbf{d}_x=P_{u+1,v}-P_{u-1,v}$\n$\\mathbf{d}_y=P_{u,v+1}-P_{u,v-1}$\n$\\mathbf{n}=\\mathrm{norm}(\\mathbf{d}_x\\times\\mathbf{d}_y)$",
               fill="computed", fontsize=TINY_FONT_SIZE)
    draw_arrow(ax, (6.10, 7.66), (6.38, 7.66))
    draw_feature_map(ax, 9.00, 7.23, 0.82, 0.92, "$N$", fill="computed")
    draw_arrow(ax, (8.63, 7.66), (9.00, 7.66))
    draw_operator(ax, 10.35, 7.66, "C", radius=0.19, fill="cross")
    draw_arrow(ax, (9.92, 7.66), (10.16, 7.66))
    draw_poly_arrow(ax, [(1.08, 7.75), (1.08, 5.77), (10.35, 5.77),
                         (10.35, 7.47)], label="$XYZ,\ M$",
                    label_at=(7.2, 5.93), color="muted", linewidth=0.8)

    draw_block(ax, 11.62, 7.00, 1.55, 1.35,
               "7-channel\ngeometry\n$[X,Y,Z,$\n$N_x,N_y,N_z,M]$",
               fill="geometry", fontsize=SMALL_FONT_SIZE)
    draw_arrow(ax, (10.54, 7.66), (11.62, 7.66))
    draw_block(ax, 13.52, 7.02, 1.45, 1.30,
               "Dense Geometry\nEncoder", fill="geometry",
               fontsize=SMALL_FONT_SIZE)
    draw_arrow(ax, (13.17, 7.66), (13.52, 7.66))
    ax.text(14.25, 6.66, "$G_0, G_1, G_2, G_3, G_4$",
            ha="center", va="center", fontsize=SMALL_FONT_SIZE)
    ax.text(14.25, 6.35, "reductions /2, /4, /8, /16, /32",
            ha="center", va="center", fontsize=TINY_FONT_SIZE,
            color=COLORS["muted"])

    # Dense encoder detail.
    draw_container(ax, 5.45, 0.28, 9.90, 4.58,
                   "C. DenseGeometryEncoder detail", fill="#F5FAF8")
    encoder_y = 3.15
    enc_blocks = [
        (5.72, 0.82, "$[B,7,$\n$H,W]$", "tensor"),
        (6.76, 1.25, "Stem\n$7\\times7$, s=2\n$7\\rightarrow32$; GN; SiLU\n$G_0$: 32, /2", "geometry"),
        (8.23, 1.02, "MaxPool\n$3\\times3$, s=2", "geometry"),
        (9.47, 1.16, "ResBlock 1\n$32\\rightarrow48$, s=1\n$G_1$: 48, /4", "geometry"),
        (10.85, 1.16, "ResBlock 2\n$48\\rightarrow96$, s=2\n$G_2$: 96, /8", "geometry"),
        (12.23, 1.16, "ResBlock 3\n$96\\rightarrow128$, s=2\n$G_3$: 128, /16", "geometry"),
        (13.61, 1.38, "ResBlock 4\n$128\\rightarrow192$, s=2\n$G_4$: 192, /32", "geometry"),
    ]
    previous = None
    for x, w, label, fill in enc_blocks:
        draw_block(ax, x, encoder_y - 0.50, w, 1.00, label, fill=fill,
                   fontsize=5.35)
        if previous is not None:
            draw_arrow(ax, previous, (x, encoder_y))
        previous = (x + w, encoder_y)
    # Residual-block inset.
    draw_container(ax, 0.15, 0.28, 4.95, 4.58,
                   "D. GeometryResidualBlock", fill="#FAFAF7")
    draw_block(ax, 0.42, 2.78, 0.68, 0.54, "$x$", fill="tensor")
    draw_block(ax, 1.42, 2.72, 0.88, 0.66, "Conv\n$3\\times3$", fill="geometry",
               fontsize=TINY_FONT_SIZE)
    draw_block(ax, 2.58, 2.72, 0.78, 0.66, "GN\nSiLU", fill="geometry",
               fontsize=TINY_FONT_SIZE)
    draw_block(ax, 3.64, 2.72, 0.88, 0.66, "Conv\n$3\\times3$ + GN", fill="geometry",
               fontsize=TINY_FONT_SIZE)
    draw_arrow(ax, (1.10, 3.05), (1.42, 3.05))
    draw_arrow(ax, (2.30, 3.05), (2.58, 3.05))
    draw_arrow(ax, (3.36, 3.05), (3.64, 3.05))
    draw_operator(ax, 4.55, 2.05, "+", radius=0.17, fill="cross")
    draw_poly_arrow(ax, [(4.52, 3.05), (4.72, 3.05), (4.72, 2.24),
                         (4.64, 2.16)])
    draw_block(ax, 1.42, 1.14, 2.22, 0.62,
               "Identity or Conv $1\\times1$", fill="tensor",
               fontsize=TINY_FONT_SIZE)
    draw_poly_arrow(ax, [(0.76, 2.78), (0.76, 1.45), (1.42, 1.45)])
    draw_arrow(ax, (3.64, 1.45), (4.39, 1.96))
    draw_block(ax, 3.77, 0.67, 1.02, 0.55, "SiLU → $y$", fill="geometry",
               fontsize=TINY_FONT_SIZE)
    draw_arrow(ax, (4.55, 1.88), (4.35, 1.22))

    ax.text(0.35, 5.08,
            "$M$: finite positive-depth mask; undefined normals are zeroed",
            ha="left", va="center", fontsize=TINY_FONT_SIZE,
            color=COLORS["muted"])

    export_figure(fig, "fig_geometry_branch")


# ---------------------------------------------------------------------------
# Figure 3: MS2Fusion branches, CP-SSM, and SP-SSM
# ---------------------------------------------------------------------------
def draw_fusion_block():
    fig, ax = make_canvas(7.15, 5.15, (0, 15.7), (0, 11.2))

    draw_container(ax, 0.18, 7.48, 5.35, 3.35,
                   "A. Input embeddings and shared local mixing", fill="#FAFAFA")
    draw_container(ax, 5.82, 5.90, 7.60, 4.93,
                   "B. CP-SSM: cross-parametric readout", fill="#FFFDF4")
    draw_container(ax, 5.82, 0.45, 7.60, 4.95,
                   "C. SP-SSM: jointly shared parameters", fill="#F8F6FB")
    draw_container(ax, 13.68, 0.45, 1.80, 10.38,
                   "D. Output heads", fill="#F7F9FA")

    # Four explicit streams: modality-specific r/g and tied shared s_r/s_g.
    lane_y = [10.00, 7.78]
    for y, symbol, fill in zip(lane_y, ["$R_i$", "$G_i$"], ["rgb", "geometry"]):
        draw_feature_map(ax, 0.43, y - 0.30, 0.55, 0.62, symbol, fill=fill)
    draw_block(ax, 1.47, 9.72, 1.12, 0.56, "RGB $1\\times1$", fill="rgb",
               fontsize=TINY_FONT_SIZE)
    draw_block(ax, 1.47, 7.50, 1.12, 0.56, "Geo $1\\times1$", fill="geometry",
               fontsize=TINY_FONT_SIZE)
    draw_block(ax, 2.92, 9.72, 1.45, 0.56, "DWConv + SiLU", fill="fusion",
               fontsize=TINY_FONT_SIZE)
    draw_block(ax, 2.92, 7.50, 1.45, 0.56, "DWConv + SiLU", fill="fusion",
               fontsize=TINY_FONT_SIZE)
    for y in lane_y:
        draw_arrow(ax, (1.08, y), (1.47, y))
        draw_arrow(ax, (2.59, y), (2.92, y))

    # Shared input projections are tied between modalities.
    draw_block(ax, 1.47, 8.89, 1.12, 0.52, "Shared $1\\times1$*", fill="shared",
               fontsize=TINY_FONT_SIZE)
    draw_block(ax, 2.92, 8.89, 1.45, 0.52, "DWConv + SiLU", fill="shared",
               fontsize=TINY_FONT_SIZE)
    draw_block(ax, 1.47, 8.13, 1.12, 0.52, "Shared $1\\times1$*", fill="shared",
               fontsize=TINY_FONT_SIZE)
    draw_block(ax, 2.92, 8.13, 1.45, 0.52, "DWConv + SiLU", fill="shared",
               fontsize=TINY_FONT_SIZE)
    draw_poly_arrow(ax, [(1.08, 10.00), (1.26, 10.00), (1.26, 9.15), (1.47, 9.15)],
                    color="muted", linewidth=0.75)
    draw_poly_arrow(ax, [(1.08, 7.78), (1.26, 7.78), (1.26, 8.39), (1.47, 8.39)],
                    color="muted", linewidth=0.75)
    draw_arrow(ax, (2.59, 9.15), (2.92, 9.15), color="muted", linewidth=0.75)
    draw_arrow(ax, (2.59, 8.39), (2.92, 8.39), color="muted", linewidth=0.75)
    for y, symbol in [(10.00, "$r$"), (9.15, "$s_r$"),
                      (8.39, "$s_g$"), (7.78, "$g$")]:
        ax.text(4.73, y, symbol, ha="center", va="center", fontsize=FONT_SIZE)
        draw_arrow(ax, (4.37, y), (5.02, y), color="muted" if "s" in symbol else "ink")

    # CP-SSM: separate parameterization and exchanged C readouts.
    draw_block(ax, 6.17, 9.38, 1.70, 0.68,
               "RGB parameterizer\n$(\\Delta_r,B_r,C_r)$", fill="rgb",
               fontsize=TINY_FONT_SIZE)
    draw_block(ax, 6.17, 6.72, 1.70, 0.68,
               "Geo parameterizer\n$(\\Delta_g,B_g,C_g)$", fill="geometry",
               fontsize=TINY_FONT_SIZE)
    draw_arrow(ax, (5.02, 10.00), (6.17, 9.72))
    draw_arrow(ax, (5.02, 7.78), (6.17, 7.06))

    draw_block(ax, 8.48, 9.10, 2.48, 1.18,
               "RGB selective scan\ninput $r$; $A_r,D_r$\n$\\Delta_r,B_r,\\mathbf{C_g}$",
               fill="cross", fontsize=SMALL_FONT_SIZE)
    draw_block(ax, 8.48, 6.50, 2.48, 1.18,
               "Geometry selective scan\ninput $g$; $A_g,D_g$\n$\\Delta_g,B_g,\\mathbf{C_r}$",
               fill="cross", fontsize=SMALL_FONT_SIZE)
    draw_arrow(ax, (7.87, 9.72), (8.48, 9.72))
    draw_arrow(ax, (7.87, 7.06), (8.48, 7.06))
    draw_poly_arrow(ax, [(7.02, 9.38), (7.02, 8.17), (8.05, 8.17),
                         (8.48, 7.45)], label="$C_r$",
                    label_at=(7.64, 8.33), color="muted", linewidth=0.8)
    draw_poly_arrow(ax, [(7.02, 7.40), (7.02, 8.63), (8.05, 8.63),
                         (8.48, 9.36)], label="$C_g$",
                    label_at=(7.64, 8.78), color="muted", linewidth=0.8)
    draw_block(ax, 11.43, 9.32, 1.35, 0.75, "$F_i^{r,c}$\nRGB cross", fill="cross",
               fontsize=SMALL_FONT_SIZE)
    draw_block(ax, 11.43, 6.72, 1.35, 0.75, "$F_i^{g,c}$\nGeo cross", fill="cross",
               fontsize=SMALL_FONT_SIZE)
    draw_arrow(ax, (10.96, 9.69), (11.43, 9.69))
    draw_arrow(ax, (10.96, 7.09), (11.43, 7.09))

    # SP-SSM: one shared parameterization from sr + sg, distinct scan inputs.
    draw_operator(ax, 6.46, 4.25, "+", radius=0.18, fill="shared")
    draw_poly_arrow(ax, [(5.02, 9.15), (5.28, 9.15), (5.28, 4.38),
                         (6.28, 4.30)], color="muted", linewidth=0.8)
    draw_poly_arrow(ax, [(5.02, 8.39), (5.48, 8.39), (5.48, 4.12),
                         (6.28, 4.20)], color="muted", linewidth=0.8)
    draw_block(ax, 7.02, 3.77, 2.28, 0.96,
               "Shared parameterizer\ninput $s_r+s_g$\noutput $(\\Delta_s,B_s,C_s)$",
               fill="shared", fontsize=TINY_FONT_SIZE)
    draw_arrow(ax, (6.64, 4.25), (7.02, 4.25))
    draw_tied_marker(ax, 8.06, 3.55, "one shared $A_s,D_s$ and one parameter set")

    draw_block(ax, 9.62, 3.96, 1.92, 0.92,
               "Shared scan\ninput $r$", fill="shared",
               fontsize=SMALL_FONT_SIZE)
    draw_block(ax, 9.62, 1.37, 1.92, 0.92,
               "Shared scan\ninput $g$", fill="shared",
               fontsize=SMALL_FONT_SIZE)
    draw_arrow(ax, (9.30, 4.25), (9.62, 4.42))
    draw_poly_arrow(ax, [(8.92, 3.77), (8.92, 1.83), (9.62, 1.83)],
                    color="muted", linewidth=0.8)
    draw_poly_arrow(ax, [(5.02, 10.00), (5.62, 10.00), (5.62, 5.62),
                         (9.28, 5.62), (9.62, 4.66)],
                    color="ink", linewidth=0.85)
    draw_poly_arrow(ax, [(5.02, 7.78), (5.68, 7.78), (5.68, 0.92),
                         (9.28, 0.92), (9.62, 1.59)],
                    color="ink", linewidth=0.85)
    draw_block(ax, 11.95, 4.05, 0.95, 0.72, "$F_i^{r,s}$\nRGB shared", fill="shared",
               fontsize=TINY_FONT_SIZE)
    draw_block(ax, 11.95, 1.47, 0.95, 0.72, "$F_i^{g,s}$\nGeo shared", fill="shared",
               fontsize=TINY_FONT_SIZE)
    draw_arrow(ax, (11.54, 4.42), (11.95, 4.42))
    draw_arrow(ax, (11.54, 1.83), (11.95, 1.83))

    # Output heads. Cross heads are modality-specific; shared heads are tied.
    outputs = [
        (9.69, "$F_i^{r,c}$", "RGB LN\n+ Conv $1\\times1$", "rgb"),
        (7.09, "$F_i^{g,c}$", "Geo LN\n+ Conv $1\\times1$", "geometry"),
        (4.42, "$F_i^{r,s}$", "Shared LN\n+ Conv $1\\times1$", "shared"),
        (1.83, "$F_i^{g,s}$", "Shared LN\n+ Conv $1\\times1$", "shared"),
    ]
    for y, symbol, label, fill in outputs:
        draw_block(ax, 13.94, y - 0.36, 1.24, 0.72, label, fill=fill,
                   fontsize=TINY_FONT_SIZE)
        draw_feature_map(ax, 15.31, y - 0.25, 0.28, 0.48, symbol,
                         fill=fill, fontsize=TINY_FONT_SIZE)
        draw_arrow(ax, (15.18, y), (15.31, y))
    draw_arrow(ax, (12.78, 9.69), (13.94, 9.69))
    draw_arrow(ax, (12.78, 7.09), (13.94, 7.09))
    draw_arrow(ax, (12.90, 4.42), (13.94, 4.42))
    draw_arrow(ax, (12.90, 1.83), (13.94, 1.83))
    ax.text(14.58, 5.42, "shared heads tied",
            ha="center", va="center", fontsize=TINY_FONT_SIZE,
            color=COLORS["muted"], style="italic")

    ax.text(7.80, 0.16,
            "* Shared $1\\times1$ weights are tied; one local DWConv is reused. CP exchanges only $C$; SP shares $(\\Delta,B,C,A,D)$.",
            ha="center", va="center", fontsize=TINY_FONT_SIZE,
            color=COLORS["muted"])

    export_figure(fig, "fig_fusion_block")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    draw_main_architecture()
    draw_geometry_branch()
    draw_fusion_block()
    print(f"Generated architecture figures in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
