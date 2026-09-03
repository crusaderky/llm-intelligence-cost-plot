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

from __future__ import annotations

import math
import os
from typing import NamedTuple

import matplotlib

matplotlib.use("agg")  # Agg gives us a measurable renderer; we still save SVG
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.path import Path as MplPath
from matplotlib.ticker import FormatStrFormatter, MultipleLocator

# Publisher colors, replicated from artificialanalysis.ai
PUBLISHERS = {
    "Alibaba": "#ff7018",
    "Anthropic": "#cc785c",
    "Apodex": "#30d8d1",
    "DeepSeek": "#2243e6",
    "Google": "#34A853",
    "InclusionAI": "#4fb5ff",
    "Meta": "#0089f4",
    "Moonshot AI": "#047AFE",
    "OpenAI": "#1f1f1f",
    "Ornith AI": "#dddddd",
    "SpaceXAI": "#736cd3",
    "Tencent": "#66b7fb",
    "Xiaomi": "#fb6d25",
    "Z AI": "#1c7ff8",
}


class LocalHardware(NamedTuple):
    name: str
    peak_power_draw: int  # Watts under max load
    idle_power_draw: int  # Watts when idling


RTX3090 = LocalHardware("RTX 3090", 350, 43)
STRIX_HALO = LocalHardware("Strix Halo 128GB", 170, 11)


class ModelPrice:
    input: float
    output: float
    cache_read: float

    def __init__(
        self, input: float, output: float, cache_read: float, total: float | None = None
    ):
        if total is not None:
            scale = total / (input + output + cache_read)
            input *= scale
            output *= scale
            cache_read *= scale
        self.input = input
        self.output = output
        self.cache_read = cache_read

    def __mul__(self, factor: float) -> ModelPrice:
        return ModelPrice(
            input=self.input * factor,
            output=self.output * factor,
            cache_read=self.cache_read * factor,
        )


class Model(NamedTuple):
    publisher: str
    name: str
    intelligence: float
    cost_per_task: float

    @classmethod
    def local(
        cls, publisher, name, intelligence, tok_per_task, tok_per_sec, hardware=RTX3090
    ):
        # Weighted by population, May 2026 (USD/KWh)
        # https://www.eia.gov/electricity/monthly/epm_table_grapher.php?t=epmt_5_6_a
        US_ELECTRICITY_PRICE = 0.2049
        sec_per_task = tok_per_task / tok_per_sec
        power_draw = hardware.peak_power_draw - hardware.idle_power_draw
        joules_per_task = power_draw * sec_per_task
        kwh_per_task = joules_per_task / 3_600_000
        cost_per_task = kwh_per_task * US_ELECTRICITY_PRICE
        # Finger-in-the-air overhead to account for prefill
        cost_per_task *= 1.2

        return cls(
            publisher, f"{name} ({hardware.name} ⚡)", intelligence, cost_per_task
        )

    @classmethod
    def reduced_price(
        cls,
        publisher: str,
        name: str,
        intelligence: float,
        nominal_cost_per_task: ModelPrice,
        nominal_price: ModelPrice,
        cheapest_price: ModelPrice,
    ) -> Model:
        nc = nominal_cost_per_task
        np = nominal_price
        cp = cheapest_price

        input_cost = nc.input * cp.input / np.input
        output_cost = nc.output * cp.output / np.output
        cache_read_cost = nc.cache_read * cp.cache_read / np.cache_read

        cost_per_task = input_cost + output_cost + cache_read_cost
        print(
            f"Reduced from ${nc.input + nc.output + nc.cache_read:.4f} to "
            f"${cost_per_task:.4f}: {name}"
        )
        return cls(publisher, name, intelligence, cost_per_task)

    @property
    def trains_on_your_data(self) -> bool:
        return "[TRAIN]" in self.name

    @property
    def not_publicly_available(self) -> bool:
        return "[UNAVAILABLE]" in self.name


MODELS = [
    Model.local("InclusionAI", "Ling-3.0-Tiny", 25, 50676, 200),
    Model.local("Alibaba", "Qwen3.6-35B-A3B", 32, 30591, 150),
    Model.local("Ornith AI", "Ornith-1.5-35B-A3B", 32 * 1.15, 30591, 110),
    Model.local("Meta", "Muse Glimmer", 35, 11993, 124),
    Model.local("Alibaba", "Qwen3.8-27B (non-reasoning)", 34.8, 17589, 67),
    Model.local("Alibaba", "Qwen3.8-27B (low)", 42.9, 26040, 60),
    Model.local("Alibaba", "Qwen3.8-27B (medium)", 44.5, 31215, 60),
    Model.local("Alibaba", "Qwen3.8-27B (xhigh)", 52.0, 47166, 60),
    Model.local("Alibaba", "Qwen3.8-Flash-Next", 55.7, 61494, 25, hardware=STRIX_HALO),
    Model("Alibaba", "Qwen3.8-Flash-Next", 55.7, 0.097),
    Model.reduced_price(
        "Alibaba",
        "Qwen3.8-Max",
        58.0,
        ModelPrice(input=0.30, output=0.23, cache_read=0.38),
        nominal_price=ModelPrice(input=2.00, output=6.00, cache_read=0.25),
        cheapest_price=ModelPrice(input=2.00, output=6.00, cache_read=0.20),
    ),
    # Model.reduced_price(
    #     "Alibaba",
    #     "Qwen3.8-Max-0902",
    #     58.0 * 1.146,
    #     ModelPrice(input=0.30, output=0.23, cache_read=0.38),
    #     nominal_price=ModelPrice(input=2.00, output=6.00, cache_read=0.25),
    #     cheapest_price=ModelPrice(input=2.00, output=6.00, cache_read=0.25),
    # ),
    Model.reduced_price(
        "DeepSeek",
        "DeepSeek V4 Flash 0731",
        51.7,
        ModelPrice(input=0.0327, output=0.06, cache_read=0.02, total=0.112),
        nominal_price=ModelPrice(input=0.44, output=1.32, cache_read=0.014),
        cheapest_price=ModelPrice(input=0.065, output=0.16, cache_read=0.013),
    ),
    # Model.reduced_price(
    #     "DeepSeek",
    #     "DeepSeek V4 Flash Vision Exp",
    #     51.5,
    #     ModelPrice(input=0.0327, output=0.06, cache_read=0.02, total=0.117),
    #     nominal_price=ModelPrice(input=0.44, output=1.32, cache_read=0.014),
    #     cheapest_price=ModelPrice(input=0.22, output=0.66, cache_read=0.007),
    # ),
    Model("DeepSeek", "DeepSeek V4 Pro 0813", 53.2, 0.25),
    Model.reduced_price(
        "Tencent",
        "Hy3",
        42.2,
        ModelPrice(input=0.0109, output=0.0142, cache_read=0.01, total=0.0358),
        nominal_price=ModelPrice(input=0.136, output=0.554, cache_read=0.136 * 0.25),
        cheapest_price=ModelPrice(input=0.0825, output=0.33, cache_read=0.02063),
    ),
    # Model.reduced_price(
    #     "Tencent",
    #     "Hy4 preview",
    #     57.4 * 51.5 / 51.1,
    #     ModelPrice(input=0.09, output=0.18, cache_read=0.41),
    #     nominal_price=ModelPrice(input=1.40, output=4.40, cache_read=0.26),
    #     cheapest_price=ModelPrice(input=0.834, output=2.501, cache_read=0.042),
    # ),
    Model.reduced_price(
        "Meta",
        "Muse Spark 1.3 (xhigh)",
        60.78,
        ModelPrice(input=0.18, output=0.13, cache_read=0.24),
        nominal_price=ModelPrice(input=1.25, output=4.25, cache_read=0.15),
        cheapest_price=ModelPrice(input=1.25, output=4.25, cache_read=0.15),
    ),
    Model.reduced_price(
        "Meta",
        "Muse Spark 1.3 (max) [UNAVAILABLE]",
        62.09,
        ModelPrice(input=0.18, output=0.13, cache_read=0.24) * (35221 / 30543),
        nominal_price=ModelPrice(input=1.25, output=4.25, cache_read=0.15),
        cheapest_price=ModelPrice(input=1.25, output=4.25, cache_read=0.15),
    ),
    Model.reduced_price(
        "Meta",
        "Muse Spark 1.3 (xhigh) [TRAIN]",
        60.78,
        ModelPrice(input=0.18, output=0.13, cache_read=0.24),
        nominal_price=ModelPrice(input=1.25, output=4.25, cache_read=0.15),
        cheapest_price=ModelPrice(input=0.10, output=0.20, cache_read=0.002),
    ),
    Model.reduced_price(
        "Z AI",
        "GLM-5.3-Flash (high)",
        # scaled intelligence and output toks (estimate)
        57.4 * 28.01 / 28.99,
        ModelPrice(input=0.011, output=0.03, cache_read=0.05, total=0.087)
        * (70610 / 138690),
        nominal_price=ModelPrice(input=0.15, output=0.50, cache_read=0.03),
        cheapest_price=ModelPrice(input=0.075, output=0.25, cache_read=0.015),
    ),
    Model.reduced_price(
        "Z AI",
        "GLM-5.3-Flash (max)",
        57.4,
        ModelPrice(input=0.011, output=0.03, cache_read=0.05, total=0.087),
        nominal_price=ModelPrice(input=0.15, output=0.50, cache_read=0.03),
        cheapest_price=ModelPrice(input=0.075, output=0.25, cache_read=0.015),
    ),
    Model.reduced_price(
        "Z AI",
        "GLM-5.3",
        59.8,
        ModelPrice(input=0.09, output=0.18, cache_read=0.41),
        nominal_price=ModelPrice(input=1.40, output=4.40, cache_read=0.26),
        cheapest_price=ModelPrice(input=1.20, output=4.00, cache_read=0.12),
    ),
    Model.reduced_price(
        "Moonshot AI",
        "Kimi K3",
        60.2,
        ModelPrice(input=0.20, output=0.38, cache_read=0.25, total=0.84),
        nominal_price=ModelPrice(input=3.00, output=15.00, cache_read=0.30),
        cheapest_price=ModelPrice(input=2.85, output=14.25, cache_read=0.285),
    ),
    Model("Xiaomi", "MiMo-V2.5", 38.03, 0.0104),
    # Model("Google", "Gemini 3.8 Flash (low)", 51.75, 0.239),
    # Model("Google", "Gemini 3.8 Flash (medium)", 56.64, 0.410),
    # Model("Google", "Gemini 3.8 Flash (high)", 58.68, 0.577),
    Model("Google", "Gemini 3.8 Flash", 58.68, 0.577),
    # Model("SpaceXAI", "Grok 4.6 (low)", 52.0, 0.25),
    # Model("SpaceXAI", "Grok 4.6 (high)", 61.0, 0.94),
    Model("SpaceXAI", "Grok 4.6", 61.0, 0.94),
    Model("OpenAI", "GPT-5.4 (Mar '26)", 53, 1.12),
    Model("OpenAI", "GPT-5.5 (Apr '26)", 56, 1.19),
    Model("OpenAI", "GPT-5.6 Luna (low)", 33.85, 0.0088),
    Model("OpenAI", "GPT-5.6 Luna (medium)", 38.90, 0.0113),
    Model("OpenAI", "GPT-5.6 Luna (high)", 46.95, 0.0216),
    Model("OpenAI", "GPT-5.6 Luna (xhigh)", 50.05, 0.0316),
    Model("OpenAI", "GPT-5.6 Luna (max)", 52.32, 0.0471),
    # Model("OpenAI", "GPT-5.6 Terra (max)", 56.5, 0.51),
    # Model("OpenAI", "GPT-5.6 Sol (medium)", 55.6, 0.30),
    Model("OpenAI", "GPT-5.6 Sol (high)", 57.3, 0.43),
    Model("OpenAI", "GPT-5.6 Sol (xhigh)", 59.0, 0.64),
    Model("OpenAI", "GPT-5.6 Sol (max)", 61.0, 0.96),
    Model("Anthropic", "Claude Opus 4.7 (Apr '26)", 55.0, 2.23),
    Model("Anthropic", "Claude Opus 4.8 (May '26)", 57.5, 2.03),
    Model("Anthropic", "Claude Haiku 4.5", 29.9, 0.218),
    Model("Anthropic", "Claude Sonnet 5", 55.2, 1.72),
    Model("Anthropic", "Claude Opus 5 (low)", 52.5, 0.43),
    Model("Anthropic", "Claude Opus 5 (medium)", 58.74, 0.724),
    Model("Anthropic", "Claude Opus 5 (high)", 61.48, 1.226),
    Model("Anthropic", "Claude Opus 5 (xhigh)", 62.53, 1.80),
    Model("Anthropic", "Claude Opus 5 (max)", 63.06, 2.336),
    Model("Anthropic", "Claude Fable 5 (Jun '26)", 62.0, 3.14),
    Model("Anthropic", "Claude Fable 5.1 (low)", 58.64, 0.773),
    Model("Anthropic", "Claude Fable 5.1 (medium)", 60.48, 1.000),
    Model("Anthropic", "Claude Fable 5.1 (high)", 62.49, 1.429),
    Model("Anthropic", "Claude Fable 5.1 (xhigh)", 64.81, 2.649),
    Model("Anthropic", "Claude Fable 5.1 (max)", 65.66, 3.687),
]

HIGH_INTELLIGENCE_THRESHOLD = 51
LOW_COST_THRESHOLD = 0.05

# The two plots to generate:
# (title, filter, x tick step, x tick format, band side, file stem)
# "band side" is the plot edge that coincides with the green band's edge.
PLOTS = [
    (
        "Intelligence vs. Cost per Task (High Intelligence)",
        lambda m: m.intelligence >= HIGH_INTELLIGENCE_THRESHOLD,
        0.10,
        "$%.2f",
        "bottom",
        "high_intelligence",
    ),
    (
        "Intelligence vs. Cost per Task (Low Cost)",
        lambda m: m.cost_per_task <= LOW_COST_THRESHOLD,
        0.005,
        "$%.3f",
        "top",
        "low_cost",
    ),
    (
        "Intelligence vs. Cost per Task (All Models)",
        lambda m: True,
        0.10,
        "$%.2f",
        None,
        "all_models",
    ),
]

# --- knobs -----------------------------------------------------------------
FIG_W, FIG_H = 26, 14  # inches
DPI = 100
DOT_SIZE = 110
LABEL_SIZE = 13
PAD_PX = 4  # breathing room added around each label's bbox
LEADER_COLOR = "#9aa1ad"
UNAVAILABLE_COLOR = "#4b5563"  # dark grey for [UNAVAILABLE] labels
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


# Colored vector icons that stand in for the ⚡ emoji: matplotlib cannot render
# color emoji, so without this ⚡ draws as a thin black outline.
ICON_SIZE = {"bolt": 15, "mask": 20}  # px
ICON_GAP = 4  # px between a label's text and its icon
ICON_COLORS = {
    "bolt": ("#fbbf24", "#b45309"),  # amber fill, dark edge
    "mask": ("#1f2937", "#09090b"),  # black fill, blacker edge
}


def _icon_path(pts):
    """Path from (x, y) vertices, recentered on the bbox of its control points."""
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    verts = [(x - cx, y - cy) for x, y in pts]
    codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(pts) - 1)
    return MplPath(verts + [(0.0, 0.0)], codes + [MplPath.CLOSEPOLY])


# Thief mask: a black domino with swept-up wing tips and two almond eye
# holes, cut out under the nonzero fill rule (eye holes wound counterclockwise
# against the clockwise outline). Also rendered once to thief_mask.png for the
# README.
#
# Mask outline details:
_MASK_PTS = [
    (0.00, 0.60),  # left wing tip
    (0.10, 0.88),  # left wing peak
    (0.34, 0.78),
    (0.50, 0.72),  # nose-bridge dip
    (0.66, 0.78),
    (0.90, 0.88),  # right wing peak
    (1.00, 0.60),  # right wing tip
    (0.95, 0.32),  # right lower wing
    (0.66, 0.26),
    (0.50, 0.34),  # nose dip
    (0.34, 0.26),
    (0.05, 0.32),  # left lower wing
]
_MASK_CENTER = (
    (min(p[0] for p in _MASK_PTS) + max(p[0] for p in _MASK_PTS)) / 2,
    (min(p[1] for p in _MASK_PTS) + max(p[1] for p in _MASK_PTS)) / 2,
)


def _eye_pts(ecx, ecy, tilt):
    """Eye-hole octagon, counterclockwise (opposite winding to the mask outline
    so it cuts out a hole under the nonzero fill rule)."""
    pts = []
    for i in range(8):
        a = i * math.pi / 4
        x, y = 0.15 * math.cos(a), 0.10 * math.sin(a)
        x, y = (
            x * math.cos(tilt) - y * math.sin(tilt),
            x * math.sin(tilt) + y * math.cos(tilt),
        )
        pts.append((ecx + x - _MASK_CENTER[0], ecy + y - _MASK_CENTER[1]))
    return pts


def _mask_path():
    mask = _icon_path(_MASK_PTS)
    verts, codes = mask.vertices.tolist(), mask.codes.tolist()
    for ecx, ecy, tilt in ((0.30, 0.56, -0.22), (0.70, 0.56, 0.22)):
        eye = _eye_pts(ecx, ecy, tilt)
        verts += eye + [eye[0]]
        codes += (
            [MplPath.MOVETO] + [MplPath.LINETO] * (len(eye) - 1) + [MplPath.CLOSEPOLY]
        )
    return MplPath(verts, codes)


ICON_PATHS = {
    "bolt": _icon_path(
        [
            (0.65, 1.00),
            (0.10, 0.42),
            (0.42, 0.42),
            (0.28, 0.00),
            (0.92, 0.58),
            (0.57, 0.58),
        ]
    ),
    "mask": _mask_path(),
}


def _split_icon(name):
    """Split the marker out of a model name.

    Returns (left, icon, right, strike): the text before the marker, the icon
    key (None, "bolt", or "mask"), the text after it, and whether the name
    carried [UNAVAILABLE] (rendered dark grey with a strikethrough, no icon).
    The icon is drawn between the two text halves, so e.g. "(RTX 3090 ⚡)"
    keeps its parentheses. Models that train on your data are marked with
    [TRAIN] in the data; the plots render them with the thief mask icon.
    """
    for marker, icon in (("⚡", "bolt"), ("[TRAIN]", "mask")):
        idx = name.find(marker)
        if idx >= 0:
            left = name[:idx].rstrip()
            right = name[idx + len(marker) :].lstrip()
            return left, icon, right, False
    idx = name.find("[UNAVAILABLE]")
    if idx >= 0:
        left = name[:idx].rstrip()
        right = name[idx + len("[UNAVAILABLE]") :].lstrip()
        return left, None, right, True
    return name, None, "", False


def _pad(bb, pad=PAD_PX):
    if hasattr(bb, "x0"):  # Bbox
        bb = (bb.x0, bb.y0, bb.x1, bb.y1)
    return (bb[0] - pad, bb[1] - pad, bb[2] + pad, bb[3] + pad)


def _overlap_area(a, b):
    dx = min(a[2], b[2]) - max(a[0], b[0])
    dy = min(a[3], b[3]) - max(a[1], b[1])
    return dx * dy if dx > 0 and dy > 0 else 0.0


def _text_width(ax, renderer, text):
    """Width of text in display pixels at LABEL_SIZE."""
    t = ax.text(0, 0, text, fontsize=LABEL_SIZE)
    w = t.get_window_extent(renderer).width
    t.remove()
    return w


def _strike_text(ax, text, renderer, color=UNAVAILABLE_COLOR, zorder=4):
    """Make a placed Text artist dark grey with a strikethrough. matplotlib
    has no native strikethrough, so the strike is a line across the text's
    measured extent (data coords, like the leader lines, so it survives SVG
    export). zorder must sit above whatever covers the text (the legend frame
    is zorder 5)."""
    text.set_color(color)
    bb = text.get_window_extent(renderer)
    yc = (bb.y0 + bb.y1) / 2
    inv = ax.transData.inverted()
    (x0, y0), (x1, y1) = inv.transform([(bb.x0, yc), (bb.x1, yc)])
    ax.add_line(
        Line2D(
            [x0, x1],
            [y0, y1],
            lw=1.0,
            color=color,
            zorder=zorder,
            clip_on=False,
        )
    )


def place_labels(ax, fig, points, marker_r_px, extra_obstacles=()):
    """points: [(left, right, icon, strike, x, y)] in data coords -- icon is
    None, "bolt", or "mask"; the icon is drawn as a colored vector marker
    between the left and right text halves, so e.g. "(RTX 3090 ⚡)" keeps its
    parens. strike=True (the [UNAVAILABLE] tag) renders the text dark grey
    with a strikethrough and no icon. Adds annotations, auto-placed.

    Placement runs in rounds: a greedy sequential pass, then repair rounds in
    which every label re-chooses its spot around everyone else's position, so
    a crowded plot doesn't end up with cascading overlaps.

    extra_obstacles: additional (x0, y0, x1, y1) display-pixel boxes that
    labels must not overlap (e.g. the legend).
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes_box = _pad(ax.get_window_extent(renderer), 0)

    # Dots are sacred: a label is never allowed to sit on top of a marker
    # unless every candidate position is worse (see scoring below).
    dot_boxes = []
    for _, _, _, _, x, y in points:
        px, py = ax.transData.transform((x, y))
        dot_boxes.append(
            (px - marker_r_px, py - marker_r_px, px + marker_r_px, py + marker_r_px)
        )

    # Place the most crowded points first -- they have the fewest good options.
    disp = [ax.transData.transform((x, y)) for _, _, _, _, x, y in points]

    def crowding(i):
        xi, yi = disp[i]
        return sum(
            1
            for j, (xj, yj) in enumerate(disp)
            if j != i and abs(xi - xj) < CROWD_X and abs(yi - yj) < CROWD_Y
        )

    crowd = [crowding(i) for i in range(len(points))]
    order = sorted(range(len(points)), key=lambda i: (-crowd[i], -points[i][5]))

    def choose(i, obstacles):
        """Pick the best candidate position for label i. Returns
        (score, bbox, left_x0, vc, icon_cx, right_x0, w_right); the bbox and
        positions are display pixels."""
        left, right, icon, _strike, x, y = points[i]
        best = None
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
                left,
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

            w_left = bb_raw.width
            h = bb_raw.height
            vc = (bb_raw.y0 + bb_raw.y1) / 2
            raw = (bb_raw.x0, bb_raw.y0, bb_raw.x1, bb_raw.y1)
            left_x0 = bb_raw.x0
            icon_cx = right_x0 = 0.0
            w_right = 0.0
            if icon is not None:
                s = ICON_SIZE[icon]
                extra = ICON_GAP + s + ICON_GAP
                if right:
                    w_right = _text_width(ax, renderer, right)
                    extra += w_right
                if ha == "right":
                    left_x0 = bb_raw.x0 - extra
                elif ha == "center":
                    left_x0 = bb_raw.x0 - extra / 2
                icon_cx = left_x0 + w_left + ICON_GAP + s / 2
                right_x0 = icon_cx + s / 2 + ICON_GAP
                half_h = max(h / 2, s / 2)
                raw = (
                    left_x0,
                    vc - half_h,
                    right_x0 + w_right,
                    vc + half_h,
                )
            bb = _pad(raw)

            # Rank: spilling outside the axes is worst, then touching a marker,
            # then how much of it, then overlap with the legend/other labels.
            # Spilling is judged on the unpadded extent -- the pad only guards
            # collisions between labels, text may approach the frame closely.
            spill = (
                raw[0] < axes_box[0]
                or raw[2] > axes_box[2]
                or raw[1] < axes_box[1]
                or raw[3] > axes_box[3]
            )
            dot_pen = sum(_overlap_area(bb, o) for o in dot_boxes)
            pen = sum(_overlap_area(bb, o) for o in obstacles)
            score = (spill, dot_pen > 0, dot_pen, pen)

            if not any(score):  # collision-free position
                return (score, bb, left_x0, vc, icon_cx, right_x0, w_right)
            if best is None or score < best[0]:
                best = (score, bb, left_x0, vc, icon_cx, right_x0, w_right)
        return best

    # Round 0: greedy sequential pass (already-placed labels are obstacles).
    placed = [None] * len(points)
    obstacles = list(extra_obstacles)
    for i in order:
        b = choose(i, obstacles)
        placed[i] = b
        obstacles.append(b[1])

    # Repair rounds: every label re-chooses with everyone else's previous
    # position as obstacles, so no single greedy stumble cascades.
    for _ in range(2):
        old = [p[1] for p in placed]
        new = [None] * len(points)
        for i in order:
            obs = list(extra_obstacles) + [old[j] for j in range(len(points)) if j != i]
            new[i] = choose(i, obs)
        placed = new

    inv = ax.transData.inverted()
    for i in order:
        left, right, icon, strike, x, y = points[i]
        _, bb, left_x0, vc, icon_cx, right_x0, _ = placed[i]
        color = UNAVAILABLE_COLOR if strike else "#1f2328"
        ((tx, ty),) = inv.transform([(left_x0, vc)])
        ax.text(
            tx,
            ty,
            left,
            ha="left",
            va="center",
            fontsize=LABEL_SIZE,
            color=color,
            zorder=4,
        )
        if icon is not None:
            fill, edge = ICON_COLORS[icon]
            ((ix, iy),) = inv.transform([(icon_cx, vc)])
            ax.plot(
                [ix],
                [iy],
                marker=ICON_PATHS[icon],
                markersize=ICON_SIZE[icon] * 72 / DPI,
                markerfacecolor=fill,
                markeredgecolor=edge,
                markeredgewidth=1.0,
                linestyle="none",
                zorder=4,
                clip_on=False,
            )
        if right:
            ((rx, ry),) = inv.transform([(right_x0, vc)])
            ax.text(
                rx,
                ry,
                right,
                ha="left",
                va="center",
                fontsize=LABEL_SIZE,
                color=color,
                zorder=4,
            )
        if strike:
            # Strikethrough across the text (the bbox pad is not part of the
            # text -- bb was padded by PAD_PX on each side).
            (sx0, sy0), (sx1, sy1) = inv.transform(
                [(bb[0] + PAD_PX, vc), (bb[2] - PAD_PX, vc)]
            )
            ax.add_line(
                Line2D(
                    [sx0, sx1],
                    [sy0, sy1],
                    lw=1.0,
                    color=color,
                    zorder=4,
                    clip_on=False,
                )
            )
        # Leader line from the dot to the nearest edge of its label. Always
        # drawn for points in a cluster (where proximity alone is ambiguous),
        # and for any label that ended up well away from its dot.
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


def make_plot(title, models, xtick_step, xtick_format, band, y_lim, stem):
    xs = [m.cost_per_task for m in models]
    ys = [m.intelligence for m in models]
    colors = [PUBLISHERS[m.publisher] for m in models]

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=DPI)

    ax.scatter(
        xs,
        ys,
        s=DOT_SIZE,
        c=colors,
        zorder=3,
    )

    # Faint dotted Pareto frontier: max intelligence for each cost. Models that
    # train on your data ([TRAIN] in the name) are excluded: they are the same
    # offers at providers that train on your data, not separate models. Models
    # that are not publicly available ([UNAVAILABLE]) are excluded too: their
    # cost is an estimate, not a real offer.
    pts = sorted(
        (
            (m.cost_per_task, m.intelligence)
            for m in models
            if not m.trains_on_your_data and not m.not_publicly_available
        ),
        key=lambda p: (p[0], -p[1]),
    )
    frontier_x, frontier_y = [], []
    best_int = -float("inf")
    for x, y in pts:
        if y > best_int:
            best_int = y
            frontier_x.append(x)
            frontier_y.append(y)
    if len(frontier_x) > 1:
        ax.plot(
            frontier_x,
            frontier_y,
            linestyle=":",
            color="#7a7f8a",
            linewidth=1.2,
            alpha=0.9,
            zorder=2,
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
    present = [p for p in PUBLISHERS if any(m.publisher == p for m in models)]
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
    if any("⚡" in m.name for m in models):
        fill, edge = ICON_COLORS["bolt"]
        handles.append(
            Line2D(
                [0],
                [0],
                marker=ICON_PATHS["bolt"],
                linestyle="none",
                markersize=10,
                markerfacecolor=fill,
                markeredgecolor=edge,
                markeredgewidth=0.9,
                label="Local electricity cost",
            )
        )
    if any(m.trains_on_your_data for m in models):
        fill, edge = ICON_COLORS["mask"]
        handles.append(
            Line2D(
                [0],
                [0],
                marker=ICON_PATHS["mask"],
                linestyle="none",
                markersize=12,
                markerfacecolor=fill,
                markeredgecolor=edge,
                markeredgewidth=0.9,
                label="Trains on your data",
            )
        )
    have_unavailable = any(m.not_publicly_available for m in models)
    if have_unavailable:
        handles.append(
            Line2D(
                [],
                [],
                linestyle="none",
                label="Not publicly available",
            )
        )
    legend = ax.legend(
        handles=handles,
        loc="lower right",
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
    if have_unavailable:
        for t in legend.get_texts():
            if t.get_text() == "Not publicly available":
                _strike_text(ax, t, renderer, zorder=6)  # above the legend frame
                break
    marker_r_px = (DOT_SIZE**0.5) / 2 / 72 * DPI + 2
    place_labels(
        ax,
        fig,
        [
            (left, right, icon, strike, m.cost_per_task, m.intelligence)
            for m in models
            for left, icon, right, strike in [_split_icon(m.name)]
        ],
        marker_r_px,
        extra_obstacles=(legend_box,),
    )

    os.makedirs("plots", exist_ok=True)
    fig.savefig(f"plots/{stem}.svg", format="svg", bbox_inches="tight")
    fig.savefig(f"plots/{stem}.png", format="png", bbox_inches="tight")
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
        math.floor(min(m.intelligence for m in models_by_stem["high_intelligence"])),
        math.ceil(max(m.intelligence for m in models_by_stem["low_cost"])),
    )
    for title, filt, step, fmt, band_side, stem in PLOTS:
        models = models_by_stem[stem]
        y_lo = math.floor(min(m.intelligence for m in models)) - 0.2
        y_hi = math.ceil(max(m.intelligence for m in models)) + 0.2
        if y_hi == y_lo:  # degenerate: all points on one level
            y_hi = y_lo + 1
        if band_side == "bottom":
            y_lo = band[0]
        elif band_side == "top":
            y_hi = band[1]
        make_plot(title, models, step, fmt, band, (y_lo, y_hi), stem)


if __name__ == "__main__":
    main()
