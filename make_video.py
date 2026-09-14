"""
Generate a 5s/10fps video: the high-intelligence plot transitioning between
the intelligence-index-v4.1 and intelligence-index-v4.3 tags.

Frame 0 renders exactly plots/high_intelligence.svg as of v4.1; frame 49 the
same plot as of v4.3 (both verified byte-identical to the committed SVGs).
In between, ymin, the xtick positions, the green band's edges, and every
point's x/y move linearly between the two tag values. Points that only
exist in v4.3 fade in; xticks that only exist in v4.3 fade out; the Pareto
frontier is recomputed per frame from the interpolated all-models set (both
tags list the same models, in the same order).

Usage: pixi run python make_video.py
Writes high_intelligence_v4.1_to_v4.3.mp4 to the repo root; intermediate
SVG/PNG frames go to a temp dir. Requires ffmpeg.
"""

from __future__ import annotations

import contextlib
import difflib
import importlib.util
import io
import math
import struct
import subprocess
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("agg")
matplotlib.rcParams["svg.hashsalt"] = "llm-intelligence-cost-plot"
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FormatStrFormatter, MultipleLocator

REPO = Path(__file__).resolve().parent
TAG_A = "intelligence-index-v4.1"
TAG_B = "intelligence-index-v4.3"
STEM = "high_intelligence"
N_FRAMES = 50  # 5 s @ 10 fps
FPS = 10
OUT_MP4 = REPO / f"{STEM}_v4.1_to_v4.3.mp4"
FRAME_DIR = Path(tempfile.mkdtemp(prefix="hi_video_"))


def _git_show(rel: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(REPO), "show", rel], capture_output=True, check=True
    ).stdout


def load_tag(tag: str) -> object:
    """Load a tag's plot.py as a module (its data plus rendering code)."""
    path = FRAME_DIR / f"plot_{tag.replace('-', '_')}.py"
    path.write_bytes(_git_show(f"{tag}:plot.py"))
    name = f"plot_{tag.replace('-', '_')}"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):  # reduced_price chatter
        spec.loader.exec_module(mod)
    return mod


class FramePoint:
    """A model at its interpolated position for one frame. Same attribute
    surface as plot.py's Model that the renderer touches, plus alpha.
    PseudoModel is a FramePoint without publisher/alpha, for the frontier."""

    def __init__(
        self,
        name: str,
        x: float,
        y: float,
        publisher: str | None = None,
        alpha: float = 1.0,
    ):
        self.name = name
        self.cost_per_task = x
        self.intelligence = y
        self.publisher = publisher
        self.alpha = alpha

    @property
    def trains_on_your_data(self) -> bool:
        return "[TRAIN]" in self.name

    @property
    def not_publicly_available(self) -> bool:
        return "[UNAVAILABLE]" in self.name


PseudoModel = FramePoint


def extract(P) -> dict:
    """The high-intelligence plot's frame parameters, computed exactly as
    plot.py's main() computes them for this tag's data."""
    hi = next(p for p in P.PLOTS if p[5] == STEM)
    lo = next(p for p in P.PLOTS if p[5] == "low_cost")
    hi_models = [m for m in P.MODELS if hi[1](m)]
    lo_models = [m for m in P.MODELS if lo[1](m)]
    band_lo = math.floor(min(m.intelligence for m in hi_models))
    band_hi = math.ceil(max(m.intelligence for m in lo_models))
    y_lo = band_lo  # the high plot's band side is "bottom"
    y_hi = math.ceil(max(m.intelligence for m in hi_models)) + 0.2
    xlim_hi = max(m.cost_per_task for m in hi_models) * 1.03
    ticks = [
        float(v)
        for v in MultipleLocator(hi[2]).tick_values(0, xlim_hi)
        if 0.0 <= v <= xlim_hi
    ]
    return {
        "models": P.MODELS,
        "hi_models": hi_models,
        "title": hi[0],
        "fmt": hi[3],
        "band_lo": band_lo,
        "band_hi": band_hi,
        "y_lo": y_lo,
        "y_hi": y_hi,
        "xlim_hi": xlim_hi,
        "low_cost": P.LOW_COST_THRESHOLD,
        "ticks": ticks,
    }


def frontier_of(models) -> list[tuple[float, float]]:
    """Pareto frontier: same computation as make_plot, max intelligence for
    each cost over ALL models, excluding [TRAIN] and [UNAVAILABLE]."""
    pts = sorted(
        (
            (m.cost_per_task, m.intelligence)
            for m in models
            if not m.trains_on_your_data and not m.not_publicly_available
        ),
        key=lambda p: (p[0], -p[1]),
    )
    out: list[tuple[float, float]] = []
    best = -float("inf")
    for x, y in pts:
        if y > best:
            best = y
            out.append((x, y))
    return out


def place_frame_labels(
    P, ax, fig, points, marker_r_px, extra_obstacles=(), window=None
):
    """Auto-place labels, like plot.py's place_labels, with per-point alpha.

    points: [(left, right, icon, strike, x, y, alpha)] in data coords.
    Points with alpha == 0 take no part in placement (they are invisible
    this frame) and are not drawn, so frame 0 places exactly the v4.1
    point set and frame 49 exactly the v4.3 one. Points whose dot is
    outside the window (the current axis range) are skipped as well: their
    dot is clipped away, and placing their label in the far margin would
    only draw text outside the plot and inflate the tight bbox.
    window: (x_min, x_max, y_min, y_max) in data coords.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes_box = P._pad(ax.get_window_extent(renderer), 0)

    # Dots are sacred: a label is never allowed to sit on top of a marker
    # unless every candidate position is worse (see scoring below).
    dot_boxes = []
    for p in points:
        if p[6] <= 0:
            continue
        px, py = ax.transData.transform((p[4], p[5]))
        dot_boxes.append(
            (px - marker_r_px, py - marker_r_px, px + marker_r_px, py + marker_r_px)
        )

    def _inside(p):
        return (
            window is None
            or window[0] <= p[4] <= window[1]
            and window[2] <= p[5] <= window[3]
        )

    active = [i for i in range(len(points)) if points[i][6] > 0 and _inside(points[i])]
    disp = {i: ax.transData.transform((points[i][4], points[i][5])) for i in active}

    def crowding(i):
        xi, yi = disp[i]
        return sum(
            1
            for j in active
            if j != i
            and abs(xi - disp[j][0]) < P.CROWD_X
            and abs(yi - disp[j][1]) < P.CROWD_Y
        )

    # Place the most crowded points first -- they have the fewest good options.
    crowd = {i: crowding(i) for i in active}
    order = sorted(active, key=lambda i: (-crowd[i], -points[i][5]))

    def choose(i, obstacles):
        """Pick the best candidate position for label i. Returns
        (score, bbox, left_x0, vc, icon_cx, right_x0, w_right); the bbox and
        positions are display pixels."""
        left, right, icon, _strike, x, y, _a = points[i]
        best = None
        # In a cluster, a label touching its dot is ambiguous no matter what, so
        # only consider the far slots -- that buys a visible leader line.
        # Right of the point is always tried first, then left; the stacked
        # far slots are the fallback once those collide.
        cands = P.CANDIDATES
        if crowd[i]:
            near = [
                c
                for c in P.CANDIDATES
                if abs(c[1]) < P.CROWD_OFFSET and c[3] == "center"
            ]
            far = [c for c in P.CANDIDATES if abs(c[1]) >= P.CROWD_OFFSET]
            cands = near + far
        for dx, dy, ha, va in cands:
            ann = ax.annotate(
                left,
                (x, y),
                textcoords="offset points",
                xytext=(dx, dy),
                ha=ha,
                va=va,
                fontsize=P.LABEL_SIZE,
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
                s = P.ICON_SIZE[icon]
                extra = P.ICON_GAP + s + P.ICON_GAP
                if right:
                    w_right = P._text_width(ax, renderer, right)
                    extra += w_right
                if ha == "right":
                    left_x0 = bb_raw.x0 - extra
                elif ha == "center":
                    left_x0 = bb_raw.x0 - extra / 2
                icon_cx = left_x0 + w_left + P.ICON_GAP + s / 2
                right_x0 = icon_cx + s / 2 + P.ICON_GAP
                half_h = max(h / 2, s / 2)
                raw = (
                    left_x0,
                    vc - half_h,
                    right_x0 + w_right,
                    vc + half_h,
                )
            bb = P._pad(raw)

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
            dot_pen = sum(P._overlap_area(bb, o) for o in dot_boxes)
            pen = sum(P._overlap_area(bb, o) for o in obstacles)
            score = (spill, dot_pen > 0, dot_pen, pen)

            if not any(score):  # collision-free position
                return (score, bb, left_x0, vc, icon_cx, right_x0, w_right)
            if best is None or score < best[0]:
                best = (score, bb, left_x0, vc, icon_cx, right_x0, w_right)
        return best

    # Round 0: greedy sequential pass (already-placed labels are obstacles).
    placed = {i: None for i in active}
    obstacles = list(extra_obstacles)
    for i in order:
        b = choose(i, obstacles)
        placed[i] = b
        obstacles.append(b[1])

    # Repair rounds: re-place every label, in crowd order, against the latest
    # positions of the others (Gauss-Seidel style). Iterate until no label
    # collides with anything, or give up after a fixed number of rounds.
    for _round in range(10):
        for i in order:
            obs = list(extra_obstacles) + [placed[j][1] for j in active if j != i]
            placed[i] = choose(i, obs)
        if not any(
            P._overlap_area(placed[i][1], placed[j][1]) > 4 * P.PAD_PX * P.PAD_PX
            for i in active
            for j in active
            if j > i
        ):
            break

    inv = ax.transData.inverted()
    for i in order:
        left, right, icon, strike, x, y, a = points[i]
        _, bb, left_x0, vc, icon_cx, right_x0, _ = placed[i]
        if a < 1.0:
            color = to_rgba(P.UNAVAILABLE_COLOR if strike else "#1f2328", a)
        else:
            color = P.UNAVAILABLE_COLOR if strike else "#1f2328"
        ((tx, ty),) = inv.transform([(left_x0, vc)])
        ax.text(
            tx,
            ty,
            left,
            ha="left",
            va="center",
            fontsize=P.LABEL_SIZE,
            color=color,
            zorder=4,
        )
        if icon is not None:
            fill, edge = P.ICON_COLORS[icon]
            if a < 1.0:
                fill, edge = to_rgba(fill, a), to_rgba(edge, a)
            ((ix, iy),) = inv.transform([(icon_cx, vc)])
            ax.plot(
                [ix],
                [iy],
                marker=P.ICON_PATHS[icon],
                markersize=P.ICON_SIZE[icon] * 72 / P.DPI,
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
                fontsize=P.LABEL_SIZE,
                color=color,
                zorder=4,
            )
        if strike:
            # Strikethrough across the text (the bbox pad is not part of the
            # text -- bb was padded by PAD_PX on each side).
            (sx0, sy0), (sx1, sy1) = inv.transform(
                [(bb[0] + P.PAD_PX, vc), (bb[2] - P.PAD_PX, vc)]
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

        if crowd[i] or dist > marker_r_px + P.LEADER_MIN:
            # start at the edge of the marker, not its centre
            if dist > 1e-6:
                ux, uy = (anchor_x - px) / dist, (anchor_y - py) / dist
                sx_, sy_ = px + ux * marker_r_px, py + uy * marker_r_px
            else:
                sx_, sy_ = px, py
            # Convert back to data coords: pixel-space artists don't survive the
            # SVG renderer's own coordinate space, data coords do.
            (x0, y0), (x1, y1) = inv.transform([(sx_, sy_), (anchor_x, anchor_y)])
            leader_color = to_rgba(P.LEADER_COLOR, a) if a < 1.0 else P.LEADER_COLOR
            ax.add_line(
                Line2D(
                    [x0, x1],
                    [y0, y1],
                    lw=0.8,
                    color=leader_color,
                    zorder=2,
                    clip_on=False,
                )
            )


def make_frame(
    P,
    title: str,
    items: list[FramePoint],
    xticks: list[float],
    xtick_alphas: list[float],
    y_lo: float,
    y_hi: float,
    xlim_hi: float,
    band: tuple[float, float, float],
    frontier: list[tuple[float, float]],
    xtick_format: str,
    k: int,
) -> None:
    """plot.py's make_plot for the high-intelligence plot, parameterized for
    a mid-transition frame. items: [FramePoint] at their interpolated
    positions. band: (band_lo, band_hi, band_x_max), clipped to the y range
    exactly as make_plot does. At the frame-0/49 endpoints every value
    equals the corresponding tag's, so the output SVGs are byte-identical to
    the committed ones."""
    models = items

    fig, ax = plt.subplots(figsize=(P.FIG_W, P.FIG_H), dpi=P.DPI)

    # Only points visible this frame are drawn; at the endpoints that is
    # exactly the tag's point set, with no alpha kwarg, like the original.
    visible = [m for m in models if m.alpha > 0]
    if min(m.alpha for m in visible) < 1.0:
        ax.scatter(
            [m.cost_per_task for m in visible],
            [m.intelligence for m in visible],
            s=P.DOT_SIZE,
            c=[P.PUBLISHERS[m.publisher] for m in visible],
            alpha=[m.alpha for m in visible],
            zorder=3,
        )
    else:
        ax.scatter(
            [m.cost_per_task for m in visible],
            [m.intelligence for m in visible],
            s=P.DOT_SIZE,
            c=[P.PUBLISHERS[m.publisher] for m in visible],
            zorder=3,
        )

    if len(frontier) > 1:
        ax.plot(
            [p[0] for p in frontier],
            [p[1] for p in frontier],
            linestyle=":",
            color="#7a7f8a",
            linewidth=1.2,
            alpha=0.9,
            zorder=2,
        )

    ax.set_xlim(0, xlim_hi)
    ax.set_ylim(y_lo, y_hi)
    if band is not None:
        band_lo, band_hi, band_x = band
        # band: the same (band_lo, band_hi) range on every plot, clipped to
        # the axis
        b_lo = max(band_lo, y_lo)
        b_hi = min(band_hi, y_hi)
        if b_hi > b_lo:
            ax.add_patch(
                Rectangle(
                    (0, b_lo),
                    band_x,
                    b_hi - b_lo,
                    facecolor="#22c55e",
                    edgecolor="none",
                    alpha=0.12,
                    zorder=0,
                )
            )
    # Ticks fully faded out this frame are dropped entirely (a tick with
    # alpha 0 must not exist at all -- frame 49 must equal v4.3 exactly).
    shown = [(x, a) for x, a in zip(xticks, xtick_alphas) if a > 0]
    formatter = FormatStrFormatter(xtick_format)
    ax.set_xticks([x for x, _ in shown])
    ax.set_xticklabels([formatter(x, None) for x, _ in shown])
    ax.xaxis.set_major_formatter(formatter)
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
    present = [p for p in P.PUBLISHERS if any(m.publisher == p for m in visible)]
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=P.DOT_SIZE**0.5,
            markerfacecolor=P.PUBLISHERS[p],
            label=p,
        )
        for p in present
    ]
    if any("⚡" in m.name for m in visible):
        fill, edge = P.ICON_COLORS["bolt"]
        handles.append(
            Line2D(
                [0],
                [0],
                marker=P.ICON_PATHS["bolt"],
                linestyle="none",
                markersize=10,
                markerfacecolor=fill,
                markeredgecolor=edge,
                markeredgewidth=0.9,
                label="Local electricity cost",
            )
        )
    if any(m.trains_on_your_data for m in visible):
        fill, edge = P.ICON_COLORS["mask"]
        handles.append(
            Line2D(
                [0],
                [0],
                marker=P.ICON_PATHS["mask"],
                linestyle="none",
                markersize=12,
                markerfacecolor=fill,
                markeredgecolor=edge,
                markeredgewidth=0.9,
                label="Trains on your data",
            )
        )
    have_unavailable = any(m.not_publicly_available for m in visible)
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
        fontsize=P.LABEL_SIZE,
        framealpha=0.9,
        edgecolor="#d0d4da",
        handletextpad=0.6,
        borderpad=0.6,
    )

    fig.tight_layout()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    legend_box = P._pad(legend.get_window_extent(renderer))
    if have_unavailable:
        for t in legend.get_texts():
            if t.get_text() == "Not publicly available":
                P._strike_text(ax, t, renderer, zorder=6)
                break
    marker_r_px = (P.DOT_SIZE**0.5) / 2 / 72 * P.DPI + 2
    place_frame_labels(
        P,
        ax,
        fig,
        [
            (left, right, icon, strike, m.cost_per_task, m.intelligence, m.alpha)
            for m in models
            for left, icon, right, strike in [P._split_icon(m.name)]
        ],
        marker_r_px,
        extra_obstacles=(legend_box,),
        window=(0, xlim_hi, y_lo, y_hi),
    )

    # Fading xticks (only the v4.1-only ones): dim their marks, labels and
    # gridlines.
    if min(a for _, a in shown) < 1.0:
        for label, (_, a) in zip(ax.xaxis.get_ticklabels(), shown):
            if a < 1.0:
                label.set_alpha(a)
        for tickline, (_, a) in zip(ax.xaxis.get_ticklines("major"), shown):
            if a < 1.0:
                tickline.set_alpha(a)
        pos = {round(x, 9): a for x, a in shown}
        for g in ax.get_xgridlines():
            a = pos.get(round(g.get_xdata()[0], 9))
            if a is not None and a < 1.0:
                g.set_alpha(a)

    fig.savefig(
        FRAME_DIR / f"frame_{k:02d}.svg",
        format="svg",
        bbox_inches="tight",
        metadata={"Date": None},
    )
    fig.savefig(FRAME_DIR / f"frame_{k:02d}.png", format="png", bbox_inches="tight")
    plt.close(fig)


def lerp(a: float, b: float, t: float) -> float:
    # Exact at the endpoints: a + (b - a) * 1.0 is not guaranteed to be
    # bitwise b in floating point, and frame 0/49 must equal the tags.
    if t == 0.0:
        return a
    if t == 1.0:
        return b
    return a + (b - a) * t


def _check_frame(tag: str, svg_name: str, expected: bytes) -> None:
    got = (FRAME_DIR / svg_name).read_bytes()
    if got != expected:
        diff = "\n".join(
            difflib.unified_diff(
                expected.decode().splitlines(),
                got.decode().splitlines(),
                f"{tag} (committed)",
                f"{svg_name} (rendered)",
                lineterm="",
            )
        )
        raise SystemExit(f"frame {svg_name} differs from {tag}:\n{diff[:4000]}")
    print(f"  {svg_name} byte-identical to {tag}")


def _png_size(p: Path) -> tuple[int, int]:
    d = p.read_bytes()[:24]
    return struct.unpack(">II", d[16:24])


def main() -> None:
    A = load_tag(TAG_A)
    B = load_tag(TAG_B)
    a = extract(A)
    b = extract(B)

    full_names = [m.name for m in a["models"]]
    assert full_names == [m.name for m in b["models"]], (
        "the two tags list different models; per-point interpolation needs "
        "matching names"
    )
    models_a = {m.name: m for m in a["models"]}
    models_b = {m.name: m for m in b["models"]}
    hi_names_a = {m.name for m in a["hi_models"]}
    hi_names_b = [m.name for m in b["hi_models"]]  # superset, in list order
    assert set(hi_names_b) >= hi_names_a, "a v4.1-only point would need fading out"

    ticks_a, ticks_b = a["ticks"], b["ticks"]
    n_common = min(len(ticks_a), len(ticks_b))

    print(
        f"ymin {a['y_lo']} -> {b['y_lo']}; y-max {a['y_hi']} -> {b['y_hi']}; "
        f"x-max {a['xlim_hi']:.4f} -> {b['xlim_hi']:.4f}; "
        f"band x {a['low_cost']} -> {b['low_cost']}, y "
        f"{a['band_lo']}..{a['band_hi']} -> {b['band_lo']}..{b['band_hi']}; "
        f"xticks {len(ticks_a)} -> {len(ticks_b)}; "
        f"high points {len(hi_names_a)} -> {len(hi_names_b)}"
    )

    for k in range(N_FRAMES):
        t = k / (N_FRAMES - 1)

        pos = {
            name: (
                lerp(models_a[name].cost_per_task, models_b[name].cost_per_task, t),
                lerp(models_a[name].intelligence, models_b[name].intelligence, t),
            )
            for name in full_names
        }
        frontier = frontier_of([PseudoModel(name, *pos[name]) for name in full_names])
        items = [
            FramePoint(
                name,
                pos[name][0],
                pos[name][1],
                publisher=models_b[name].publisher,
                alpha=1.0 if name in hi_names_a else t,
            )
            for name in hi_names_b
        ]
        xticks = [lerp(ticks_a[i], ticks_b[i], t) for i in range(n_common)]
        xtick_alphas = [1.0] * n_common
        xticks += [ticks_a[n_common + i] for i in range(len(ticks_a) - n_common)]
        xtick_alphas += [1.0 - t] * (len(ticks_a) - n_common)
        xticks += [ticks_b[n_common + i] for i in range(len(ticks_b) - n_common)]
        xtick_alphas += [t] * (len(ticks_b) - n_common)

        make_frame(
            B,
            b["title"],
            items,
            xticks,
            xtick_alphas,
            lerp(a["y_lo"], b["y_lo"], t),
            lerp(a["y_hi"], b["y_hi"], t),
            lerp(a["xlim_hi"], b["xlim_hi"], t),
            (
                lerp(a["band_lo"], b["band_lo"], t),
                lerp(a["band_hi"], b["band_hi"], t),
                lerp(a["low_cost"], b["low_cost"], t),
            ),
            frontier,
            b["fmt"],
            k,
        )
        print(f"frame {k:02d}/49 done (t={t:.3f})")

    _check_frame(TAG_A, "frame_00.svg", _git_show(f"{TAG_A}:plots/{STEM}.svg"))
    _check_frame(TAG_B, "frame_49.svg", _git_show(f"{TAG_B}:plots/{STEM}.svg"))

    # bbox_inches="tight" trims by a pixel or so depending on content;
    # pad all frames (white, right/bottom -- the background is white) to one
    # common even size before encoding. Scaling would distort the frames.
    sizes = [_png_size(FRAME_DIR / f"frame_{k:02d}.png") for k in range(N_FRAMES)]
    w = max(s[0] for s in sizes)
    if w % 2:
        w += 1
    h = max(s[1] for s in sizes)
    if h % 2:
        h += 1
    norm = FRAME_DIR / "norm"
    norm.mkdir()
    for k in range(N_FRAMES):
        src = FRAME_DIR / f"frame_{k:02d}.png"
        dst = norm / f"frame_{k:02d}.png"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(src),
                "-vf",
                f"pad={w}:{h}:0:0:white",
                str(dst),
            ],
            check=True,
        )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            str(FPS),
            "-i",
            str(norm / "frame_%02d.png"),
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            "17",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(OUT_MP4),
        ],
        check=True,
    )
    print(f"wrote {OUT_MP4} ({N_FRAMES} frames, {FPS} fps, {w}x{h})")


if __name__ == "__main__":
    main()
