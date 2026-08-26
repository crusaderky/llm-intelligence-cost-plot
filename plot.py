"""
Scatter plots: Cost per Task (USD, linear) vs Artificial Analysis Intelligence Index.

Built with matplotlib, saved as SVG and PNG. Two plots are generated:
- all models with intelligence above fixed threshold
- all models with cost per task below fixed threshold

The figures are deliberately very wide so the cost gap between the cheap
models and the frontier models is dramatic.

Dots are colored by publisher (colors replicated from artificialanalysis.ai)
and a legend lists only the publishers present in each plot.

Label placement is automatic: the script measures each label's real rendered
size and tries a list of candidate positions around its dot, keeping the
first that collides with nothing already placed. So MODELS only needs
publisher / name / price / index -- just add rows and re-run.
"""

import math

import matplotlib

matplotlib.use("agg")  # Agg gives us a measurable renderer; we still save SVG
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FormatStrFormatter, MultipleLocator

# --------------------------------------------------------------------------
# Publisher colors, replicated from artificialanalysis.ai
# --------------------------------------------------------------------------

PUBLISHERS = {
    "Alibaba": "#ff7018",
    "Anthropic": "#cc785c",
    "DeepSeek": "#2243e6",
    "Google": "#34A853",
    "InclusionAI": "#4fb5ff",
    "Kimi": "#047AFE",
    "Meta": "#0089f4",
    "OpenAI": "#1f1f1f",
    "SpaceXAI": "#736cd3",
    "Tencent": "#66b7fb",
    "Xiaomi": "#fb6d25",
    "Z AI": "#1c7ff8",
}

# --------------------------------------------------------------------------
# Data: (publisher, model, price per task USD, AA Intelligence Index)
# --------------------------------------------------------------------------

# Weighted by population, May 2026 (USD/KWh)
# https://www.eia.gov/electricity/monthly/epm_table_grapher.php?t=epmt_5_6_a
US_ELECTRICITY_PRICE = 0.2049


def local_cost_per_task(
    tok_per_task: float, tok_per_sec: float, watts: float = 350
) -> float:
    sec_per_task = tok_per_task / tok_per_sec
    joules_per_task = watts * sec_per_task
    kwh_per_task = joules_per_task / 3_600_000
    cost_per_task = kwh_per_task * US_ELECTRICITY_PRICE
    # Finger-in-the-air overhead to account for prefill
    return cost_per_task * 1.2


MODELS = [
    ("InclusionAI", "Ling-3.0-Tiny (⚡)", local_cost_per_task(50676, 200), 25),
    ("Alibaba", "Qwen3.6-35B-A3B (⚡)", local_cost_per_task(30591, 150), 32),
    ("Meta", "Muse Glimmer (⚡)", local_cost_per_task(11993, 124), 35),
    (
        "Alibaba",
        "Qwen3.8-27B (non-reasoning ⚡)",
        local_cost_per_task(17589, 67),
        34.8,
    ),
    ("Alibaba", "Qwen3.8-27B (low ⚡)", local_cost_per_task(26040, 67), 42.9),
    ("Alibaba", "Qwen3.8-27B (medium ⚡)", local_cost_per_task(31215, 67), 44.5),
    ("Alibaba", "Qwen3.8-27B (xhigh ⚡)", local_cost_per_task(47166, 67), 52.0),
    ("DeepSeek", "DeepSeek V4 Flash 0731", 0.11, 51.7),
    ("DeepSeek", "DeepSeek V4 Flash 0731 (OpenRouter)", 0.11 * 0.18 / 0.66, 51.7),
    ("DeepSeek", "DeepSeek V4 Flash Vision", 0.11, 51.5),
    ("DeepSeek", "DeepSeek V4 Pro 0813", 0.25, 53.2),
    ("Tencent", "Hy3", 0.0358, 42.2),
    ("Google", "Gemini 3.7 Flash", 0.40, 56.0),
    ("Meta", "Muse Spark 1.2", 0.40, 56.8),
    ("Z AI", "GLM-5.3-Flash", 0.087, 57.4),
    # ("Z AI", "GLM-5.2", 0.445, 52.3),
    # ("Z AI", "GLM-5.2 (OpenRouter)", 0.445 * 2.20 / 4.40, 52.3),
    ("Z AI", "GLM-5.3", 0.68, 59.8),
    ("Z AI", "GLM-5.3 (OpenRouter Sep'26)", 0.68 * 2.20 / 4.40, 59.8),
    ("Alibaba", "Qwen3.8 Max", 1.09, 58.0),
    ("Kimi", "Kimi K3", 0.84, 60.2),
    ("SpaceXAI", "Grok 4.6 (low)", 0.22, 52.0),
    ("SpaceXAI", "Grok 4.6 (high)", 0.84, 61.0),
    ("Xiaomi", "MiMo-V2.5", 0.0104, 38.03),
    ("OpenAI", "GPT-5.6 Luna (low)", 0.0088, 33.85),
    ("OpenAI", "GPT-5.6 Luna (medium)", 0.0113, 38.90),
    ("OpenAI", "GPT-5.6 Luna (high)", 0.0216, 46.95),
    ("OpenAI", "GPT-5.6 Luna (xhigh)", 0.0316, 50.05),
    ("OpenAI", "GPT-5.6 Luna (max)", 0.0471, 52.32),
    ("OpenAI", "GPT-5.6 Terra (max)", 0.51, 56.5),
    ("OpenAI", "GPT-5.6 Sol (medium)", 0.30, 55.6),
    ("OpenAI", "GPT-5.6 Sol (high)", 0.43, 57.3),
    ("OpenAI", "GPT-5.6 Sol (xhigh)", 0.64, 59.0),
    ("OpenAI", "GPT-5.6 Sol (max)", 0.96, 61.0),
    ("Anthropic", "Claude Opus 4.7", 2.23, 55.0),
    ("Anthropic", "Claude Opus 4.8", 2.03, 57.5),
    ("Anthropic", "Claude Sonnet 5 (max)", 1.72, 55.2),
    ("Anthropic", "Claude Opus 5 (medium)", 0.72, 59),
    ("Anthropic", "Claude Opus 5 (high)", 1.23, 61.5),
    ("Anthropic", "Claude Opus 5 (xhigh)", 1.80, 62.5),
    ("Anthropic", "Claude Opus 5 (max)", 2.34, 63.0),
    ("Anthropic", "Claude Fable 5", 3.14, 62.0),
]

HIGH_INTELLIGENCE_THRESHOLD = 50
LOW_COST_THRESHOLD = 0.1

# The two plots to generate:
# (title, filter, x tick step, x tick format, band side, file stem)
# "band side" is the plot edge that coincides with the green band's edge.
PLOTS = [
    (
        "Intelligence vs. Cost per Task (High Intelligence)",
        lambda m: m[3] >= HIGH_INTELLIGENCE_THRESHOLD,
        0.10,
        "$%.2f",
        "bottom",
        "intelligence_vs_cost",
    ),
    (
        "Intelligence vs. Cost per Task (Low Cost)",
        lambda m: m[2] <= LOW_COST_THRESHOLD,
        0.005,
        "$%.3f",
        "top",
        "intelligence_vs_cost_local",
    ),
]

# --- knobs -----------------------------------------------------------------
FIG_W, FIG_H = 26, 14  # inches
DPI = 100
DOT_SIZE = 110
LABEL_SIZE = 13
PAD_PX = 4  # breathing room added around each label's bbox
LEADER_COLOR = "#9aa1ad"
LEADER_MIN = 8  # draw a leader once the label sits this far off the dot
CROWD_X = 200  # px window used to decide a point is "in a cluster"
CROWD_Y = 60
CROWD_OFFSET = 23  # clustered labels sit at least this far out (points),
# so their leader lines are long enough to follow

# Candidate label positions: (dx, dy) in points, plus alignment.
# Ordered by preference -- first collision-free one wins.
CANDIDATES = [
    (10, 0, "left", "center"),
    (-10, 0, "right", "center"),
    (0, 10, "center", "bottom"),
    (0, -10, "center", "top"),
    (10, 9, "left", "bottom"),
    (10, -9, "left", "top"),
    (-10, 9, "right", "bottom"),
    (-10, -9, "right", "top"),
    (0, 24, "center", "bottom"),
    (0, -24, "center", "top"),
    (10, 23, "left", "bottom"),
    (10, -23, "left", "top"),
    (-10, 23, "right", "bottom"),
    (-10, -23, "right", "top"),
    (0, 38, "center", "bottom"),
    (0, -38, "center", "top"),
    (10, 37, "left", "bottom"),
    (10, -37, "left", "top"),
    (-10, 37, "right", "bottom"),
    (-10, -37, "right", "top"),
    (0, 52, "center", "bottom"),
    (0, -52, "center", "top"),
    (10, 51, "left", "bottom"),
    (10, -51, "left", "top"),
    (-10, 51, "right", "bottom"),
    (-10, -51, "right", "top"),
    (0, 66, "center", "bottom"),
    (0, -66, "center", "top"),
    (0, 80, "center", "bottom"),
    (0, -80, "center", "top"),
    (10, 79, "left", "bottom"),
    (10, -79, "left", "top"),
    (-10, 79, "right", "bottom"),
    (-10, -79, "right", "top"),
]


def _pad(bb, pad=PAD_PX):
    return (bb.x0 - pad, bb.y0 - pad, bb.x1 + pad, bb.y1 + pad)


def _overlap_area(a, b):
    dx = min(a[2], b[2]) - max(a[0], b[0])
    dy = min(a[3], b[3]) - max(a[1], b[1])
    return dx * dy if dx > 0 and dy > 0 else 0.0


def place_labels(ax, fig, points, marker_r_px, extra_obstacles=()):
    """points: [(name, x, y)] in data coords. Adds annotations, auto-placed.

    extra_obstacles: additional (x0, y0, x1, y1) display-pixel boxes that
    labels must not overlap (e.g. the legend).
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes_box = _pad(ax.get_window_extent(renderer), 0)

    # Dots are sacred: a label is never allowed to sit on top of a marker
    # unless every candidate position is worse (see scoring below).
    dot_boxes = []
    for _, x, y in points:
        px, py = ax.transData.transform((x, y))
        dot_boxes.append(
            (px - marker_r_px, py - marker_r_px, px + marker_r_px, py + marker_r_px)
        )

    # The rest (legend, previously placed labels) is ranked below dot overlap.
    obstacles = list(extra_obstacles)

    # Place the most crowded points first -- they have the fewest good options.
    disp = [ax.transData.transform((x, y)) for _, x, y in points]

    def crowding(i):
        xi, yi = disp[i]
        return sum(
            1
            for j, (xj, yj) in enumerate(disp)
            if j != i and abs(xi - xj) < CROWD_X and abs(yi - yj) < CROWD_Y
        )

    crowd = [crowding(i) for i in range(len(points))]
    order = sorted(range(len(points)), key=lambda i: (-crowd[i], -points[i][1]))

    for i in order:
        name, x, y = points[i]
        best = None  # (penalty, dx, dy, ha, va, bbox)
        # In a cluster, a label touching its dot is ambiguous no matter what, so
        # only consider the far slots -- that buys a visible leader line.
        # Right of the point is always tried first, then left; the stacked
        # far slots are the fallback once those collide.
        cands = CANDIDATES
        if crowd[i]:
            near = [
                c for c in CANDIDATES if abs(c[1]) < CROWD_OFFSET and c[3] == "center"
            ]
            far = [c for c in CANDIDATES if abs(c[1]) >= CROWD_OFFSET]
            cands = near + far
        for dx, dy, ha, va in cands:
            ann = ax.annotate(
                name,
                (x, y),
                textcoords="offset points",
                xytext=(dx, dy),
                ha=ha,
                va=va,
                fontsize=LABEL_SIZE,
                color="#1f2328",
                zorder=4,
            )
            bb_raw = ann.get_window_extent(renderer)
            ann.remove()
            bb = _pad(bb_raw)

            # Rank: spilling outside the axes is worst, then touching a marker,
            # then how much of it, then overlap with the legend/other labels.
            # Spilling is judged on the unpadded extent -- the pad only guards
            # collisions between labels, text may approach the frame closely.
            spill = (
                bb_raw.x0 < axes_box[0]
                or bb_raw.x1 > axes_box[2]
                or bb_raw.y0 < axes_box[1]
                or bb_raw.y1 > axes_box[3]
            )
            dot_pen = sum(_overlap_area(bb, o) for o in dot_boxes)
            pen = sum(_overlap_area(bb, o) for o in obstacles)
            score = (spill, dot_pen > 0, dot_pen, pen)

            if not any(score):  # collision-free position
                best = (score, dx, dy, ha, va, bb)
                break
            if best is None or score < best[0]:
                best = (score, dx, dy, ha, va, bb)

        _, dx, dy, ha, va, bb = best
        ax.annotate(
            name,
            (x, y),
            textcoords="offset points",
            xytext=(dx, dy),
            ha=ha,
            va=va,
            fontsize=LABEL_SIZE,
            color="#1f2328",
            zorder=4,
        )
        # Leader line from the dot to the nearest edge of its label. Always drawn
        # for points in a cluster (where proximity alone is ambiguous), and for
        # any label that ended up well away from its dot.
        px, py = ax.transData.transform((x, y))
        anchor_x = max(bb[0], min(px, bb[2]))  # closest point on the label bbox
        anchor_y = max(bb[1], min(py, bb[3]))
        dist = math.hypot(anchor_x - px, anchor_y - py)

        if crowd[i] or dist > marker_r_px + LEADER_MIN:
            # start at the edge of the marker, not its centre
            if dist > 1e-6:
                ux, uy = (anchor_x - px) / dist, (anchor_y - py) / dist
                sx_, sy_ = px + ux * marker_r_px, py + uy * marker_r_px
            else:
                sx_, sy_ = px, py
            # Convert back to data coords: pixel-space artists don't survive the
            # SVG renderer's own coordinate space, data coords do.
            inv = ax.transData.inverted()
            (x0, y0), (x1, y1) = inv.transform([(sx_, sy_), (anchor_x, anchor_y)])
            ax.add_line(
                Line2D(
                    [x0, x1],
                    [y0, y1],
                    lw=0.8,
                    color=LEADER_COLOR,
                    zorder=2,
                    clip_on=False,
                )
            )
        obstacles.append(bb)


def make_plot(title, models, xtick_step, xtick_format, band, y_lim, stem):
    xs = [m[2] for m in models]
    ys = [m[3] for m in models]
    colors = [PUBLISHERS[m[0]] for m in models]

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=DPI)

    ax.scatter(
        xs,
        ys,
        s=DOT_SIZE,
        c=colors,
        zorder=3,
    )

    # axes (set limits before placing labels -- placement uses pixel positions)
    ax.set_xlim(0, max(xs) * 1.03)
    # y_lim: floor/ceil of the points with 0.2 of slack, except on the band
    # side, which snaps exactly to the band edge (computed in main).
    y_lo, y_hi = y_lim
    ax.set_ylim(y_lo, y_hi)
    if band is not None:
        # band: the same (y_lo, y_hi) range on every plot, clipped to the axis
        b_lo = max(band[0], y_lo)
        b_hi = min(band[1], y_hi)
        if b_hi > b_lo:
            ax.add_patch(
                Rectangle(
                    (0, b_lo),
                    LOW_COST_THRESHOLD,
                    b_hi - b_lo,
                    facecolor="#22c55e",
                    edgecolor="none",
                    alpha=0.12,
                    zorder=0,
                )
            )
    ax.xaxis.set_major_locator(MultipleLocator(xtick_step))
    ax.xaxis.set_major_formatter(FormatStrFormatter(xtick_format))
    ax.yaxis.set_major_locator(MultipleLocator(1))

    ax.set_xlabel("Cost per Task (USD)", fontsize=15, fontweight="bold", labelpad=12)
    ax.set_ylabel(
        "Artificial Analysis Intelligence Index",
        fontsize=15,
        fontweight="bold",
        labelpad=12,
    )
    ax.set_title(
        title,
        fontsize=20,
        fontweight="bold",
        loc="left",
        pad=18,
    )

    ax.grid(True, color="#e6e8ec", linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=12, colors="#5b6270")

    # Legend: one entry per publisher actually present in this plot.
    present = [p for p in PUBLISHERS if any(m[0] == p for m in models)]
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=DOT_SIZE**0.5,
            markerfacecolor=PUBLISHERS[p],
            label=p,
        )
        for p in present
    ]
    legend = ax.legend(
        handles=handles,
        loc="upper left",
        fontsize=LABEL_SIZE,
        framealpha=0.9,
        edgecolor="#d0d4da",
        handletextpad=0.6,
        borderpad=0.6,
    )

    fig.tight_layout()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    legend_box = _pad(legend.get_window_extent(renderer))
    marker_r_px = (DOT_SIZE**0.5) / 2 / 72 * DPI + 2
    place_labels(
        ax,
        fig,
        [(m[1], m[2], m[3]) for m in models],
        marker_r_px,
        extra_obstacles=(legend_box,),
    )

    fig.savefig(f"{stem}.svg", format="svg", bbox_inches="tight")
    fig.savefig(f"{stem}.png", format="png", bbox_inches="tight")
    plt.close(fig)


def main():
    models_by_stem = {}
    for title, filt, step, fmt, band_side, stem in PLOTS:
        models = [m for m in MODELS if filt(m)]
        if not models:
            raise SystemExit(f"no models match the filter for {stem}")
        models_by_stem[stem] = models

    # Green band (x $0-$LOW_COST_THRESHOLD): the points that appear on both plots (cheap
    # AND smart). Its edges coincide with the band-side axis edge of each plot: the
    # floor of the high-int plot's points and the ceil of the low-cost plot's points.
    # Identical on both plots.
    band = (
        math.floor(min(m[3] for m in models_by_stem["intelligence_vs_cost"])),
        math.ceil(max(m[3] for m in models_by_stem["intelligence_vs_cost_local"])),
    )
    for title, filt, step, fmt, band_side, stem in PLOTS:
        models = models_by_stem[stem]
        y_lo = math.floor(min(m[3] for m in models)) - 0.2
        y_hi = math.ceil(max(m[3] for m in models)) + 0.2
        if y_hi == y_lo:  # degenerate: all points on one level
            y_hi = y_lo + 1
        if band_side == "bottom":
            y_lo = band[0]
        elif band_side == "top":
            y_hi = band[1]
        make_plot(title, models, step, fmt, band, (y_lo, y_hi), stem)


if __name__ == "__main__":
    main()
