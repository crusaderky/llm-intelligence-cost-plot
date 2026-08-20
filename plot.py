"""
Scatter plot: Cost per Task (USD, linear) vs Artificial Analysis Intelligence Index.

Built with matplotlib, saved as SVG. The figure is deliberately very wide so the
cost gap between the cheap models and the frontier models is dramatic.

Label placement is automatic: the script measures each label's real rendered size
and tries a list of candidate positions around its dot, keeping the first that
collides with nothing already placed. So DATA only needs name / price / index --
just add rows and re-run.
"""

import math

import matplotlib

matplotlib.use("agg")  # Agg gives us a measurable renderer; we still save SVG
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, MultipleLocator

# --------------------------------------------------------------------------
# Data: (model, price per task USD, AA Intelligence Index)
# --------------------------------------------------------------------------

US_ELECTRICITY_PRICE = 0.1844  # Avg residential electricity price; USD/KWh


def local_cost_per_task(
    tok_per_task: float, tok_per_sec: float, watts: float = 350
) -> float:
    sec_per_task = tok_per_task / tok_per_sec
    joules_per_task = watts * sec_per_task
    kwh_per_task = joules_per_task / 3_600_000
    cost_per_task = kwh_per_task * US_ELECTRICITY_PRICE
    # Finger-in-the-air overhead to account for prefill
    return cost_per_task * 1.2


DATA = [
    # ("Ling-3.0-Tiny (⚡)", local_cost_per_task(50676, 200), 25),
    # ("Qwen3.6-35B-A3B (⚡)", local_cost_per_task(30591, 150), 32),
    # ("Muse Glimmer (⚡)", local_cost_per_task(11993, 124), 35),
    ("Qwen3.8-27B (⚡)", local_cost_per_task(47166, 67), 52),
    ("GPT-5.6 Luna", 0.050, 52),
    ("DeepSeek V4 Flash 0731", 0.110, 52),
    ("DeepSeek V4 Flash 0731 (3pp)", 0.110 * 0.18 / 0.66, 52),
    ("DeepSeek V4 Pro 0813", 0.250, 53),
    ("Gemini 3.7 Flash", 0.400, 56),
    ("Muse Spark 1.2", 0.400, 57),
    ("GLM-5.3", 0.680, 60),
    ("GLM-5.3 (3pp)", 0.680 * 2.20 / 4.40, 60),
    ("Qwen3.8 Max", 1.090, 58),
    ("Kimi K3", 0.840, 60),
    ("Grok 4.6", 0.840, 61),
    ("GPT-5.6 Sol", 1.230, 61),
    ("Claude Opus 5", 2.340, 63),
]

# --- knobs -----------------------------------------------------------------
FIG_W, FIG_H = 26, 14  # inches
DPI = 100
DOT_COLOR = "#d97757"
DOT_EDGE = "#8c4a2f"
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
]


def _pad(bb, pad=PAD_PX):
    return (bb.x0 - pad, bb.y0 - pad, bb.x1 + pad, bb.y1 + pad)


def _overlap_area(a, b):
    dx = min(a[2], b[2]) - max(a[0], b[0])
    dy = min(a[3], b[3]) - max(a[1], b[1])
    return dx * dy if dx > 0 and dy > 0 else 0.0


def place_labels(ax, fig, points, marker_r_px):
    """points: [(name, x, y)] in data coords. Adds annotations, auto-placed."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes_box = _pad(ax.get_window_extent(renderer), 0)

    # Dots are obstacles too, so labels never sit on top of a marker.
    obstacles = []
    for _, x, y in points:
        px, py = ax.transData.transform((x, y))
        obstacles.append(
            (px - marker_r_px, py - marker_r_px, px + marker_r_px, py + marker_r_px)
        )

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
            bb = _pad(ann.get_window_extent(renderer))
            ann.remove()

            penalty = sum(_overlap_area(bb, o) for o in obstacles)
            # Penalise anything spilling outside the axes, too.
            if (
                bb[0] < axes_box[0]
                or bb[2] > axes_box[2]
                or bb[1] < axes_box[1]
                or bb[3] > axes_box[3]
            ):
                penalty += 1e6

            if penalty == 0:
                best = (0, dx, dy, ha, va, bb)
                break
            if best is None or penalty < best[0]:
                best = (penalty, dx, dy, ha, va, bb)

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


def main():
    xs = [d[1] for d in DATA]
    ys = [d[2] for d in DATA]

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=DPI)

    ax.scatter(
        xs,
        ys,
        s=DOT_SIZE,
        color=DOT_COLOR,
        edgecolors=DOT_EDGE,
        linewidths=1.5,
        zorder=3,
    )

    # axes (set limits before placing labels -- placement uses pixel positions)
    ax.set_xlim(0, max(xs) * 1.03)
    ax.set_ylim(min(ys) - 2, max(ys) + 2)
    ax.xaxis.set_major_locator(MultipleLocator(0.10))
    ax.xaxis.set_major_formatter(FormatStrFormatter("$%.2f"))
    ax.yaxis.set_major_locator(MultipleLocator(1))

    ax.set_xlabel("Cost per Task (USD)", fontsize=15, fontweight="bold", labelpad=12)
    ax.set_ylabel(
        "Artificial Analysis Intelligence Index",
        fontsize=15,
        fontweight="bold",
        labelpad=12,
    )
    ax.set_title(
        "Intelligence vs. Cost per Task",
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

    fig.tight_layout()
    marker_r_px = (DOT_SIZE**0.5) / 2 / 72 * DPI + 2
    place_labels(ax, fig, DATA, marker_r_px)

    fig.savefig("intelligence_vs_cost.svg", format="svg", bbox_inches="tight")
    fig.savefig("intelligence_vs_cost.png", format="png", bbox_inches="tight")


if __name__ == "__main__":
    main()
