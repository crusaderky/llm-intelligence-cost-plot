"""
Scatter plots: Price per Task (USD, linear) vs Artificial Analysis Intelligence Index.

Built with matplotlib, saved as SVG and PNG. Four plots are generated:
- all models with intelligence above a fixed threshold
- all models with price per task below a fixed threshold
- all models
- all datacenter models, as horizontal bars sorted by intelligence, showing the
  price change vs AA's published price

The figures are deliberately very wide so the cost gap between the cheap
models and the frontier models is dramatic.

The "price per task" is the cost of one Artificial Analysis Intelligence
Index task, priced the way each class of model actually gets served.

Datacenter models are priced from what OpenRouter's customers pay for the
same token mix AA benchmarks. AA publishes the task's cost per task split by
token type -- uncached input (including cache writes), cache reads, and
output (reasoning + answer) -- together with the sticker price per token each
stream was billed at; dividing one by the other recovers how many tokens of
each kind the task uses:

    input tokens  = AA $/task[input] / AA sticker[$/token, input]
                  + AA $/task[cached input] / AA sticker[$/token, cached input]
    output tokens = AA $/task[output] / AA sticker[$/token, output]

Both AA figures are stored as PricedTokens(input_, output, cached_input) — the
cost one in USD per task, the sticker one in USD per 1M tokens. A model AA
publishes no breakdown for stores the bare total as a float instead of the
split.

OpenRouter's prices for those tokens come from the trailing week rather than
the volatile spot quote: every endpoint instance that serves the model is
priced at its mean effective price over those days, and the instances are
averaged again weighted by the share of tokens each one served (see
.agents/skills/refresh-models). Free endpoints are left out. That gives one
synthetic OR price pair per model, in $/M tokens:

    price per task = (input tokens x OR input $/M
                      + output tokens x OR output $/M) / 1e6

OR's effective input price is itself cache-hit-discounted, so it is applied
to the whole input side. A model OR carries no price chart for, or one AA
publishes no breakdown for, falls back to AA's posted cost per task.

Local (electricity-powered) models have no sticker price: their price per task
is the electricity needed to run the task on consumer hardware, counting AA's
uncached input tokens through prefill and its output tokens through decode.
Cache reads cost the local rig nothing and are left out. Models AA does not
break down get an estimated input token count (LOCAL_INPUT_TOKEN_RATIO).

Dots are colored by publisher (colors replicated from artificialanalysis.ai)
and a legend lists only the publishers present in each plot.

Label placement is automatic: the script measures each label's real rendered
size and tries a list of candidate positions around its dot, keeping the
first that collides with nothing already placed. A label is also required to
respect its dot's height order relative to the labels beside it, so a stack of
labels never reads as the wrong model, and a final pass slides every label to
the nearest free height above or below its own dot so nothing stays parked a
slot away from its point -- see place_labels. So MODELS only needs publisher /
name / intelligence / provider type plus the raw cost inputs -- just add rows
and re-run.
"""

from __future__ import annotations

import argparse
import math
import os
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal, NamedTuple

import matplotlib

matplotlib.use("agg")  # Agg gives us a measurable renderer; we still save SVG
# Deterministic SVG output: hash salt makes element IDs (clip paths, glyph
# defs) reproducible instead of uuid-derived, so re-running on unchanged data
# produces a byte-identical SVG that git sees as clean.
matplotlib.rcParams["svg.hashsalt"] = "llm-intelligence-cost-plot"
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.path import Path as MplPath
from matplotlib.ticker import FormatStrFormatter, MultipleLocator, StrMethodFormatter

# --- knobs -----------------------------------------------------------------
FIG_W, FIG_H = 26, 14  # inches
DPI = 100
VERBOSE = False  # set by --verbose: report residual label overlaps to stderr
DOT_SIZE = 110
LABEL_SIZE = 13
PAD_PX = 4  # breathing room added around each label's bbox
LEADER_COLOR = "#9aa1ad"
EGG_SIZE = 60  # points, for the easter eggs: the axis is FIG_H (1008pt)
# tall, so one shout is about an eighth of its height
EGG_COLOR = "#5b6270"  # the tick-label grey, only bigger
EGG_ALPHA = 0.25
UNAVAILABLE_COLOR = "#4b5563"  # dark grey for unavailable labels
LEADER_MIN = 8  # draw a leader once the label sits this far off the dot
LEADER_CLEAR_PX = 8  # ... and keep this much daylight between a leader line and
# every other model's dot: a leader grazing a marker reads as pointing at it
CROWD_X = 200  # px window used to decide a point is "in a cluster"
CROWD_Y = 60
CROWD_OFFSET = 23  # clustered labels sit at least this far out (points),
# so their leader lines are long enough to follow
CROWD_NEAR = 15  # ... except for the mid-height slots at +-16pt: a stacked
# neighbour often needs exactly that much clearance, and rounding up to 23 is
# what walked whole clusters into far-away stacks

# Order rule (see place_labels misreads()): a label may not stack above the
# label of a lower dot, or the reader pairs the upper label with the upper dot
# and reads the wrong model. It applies only to pairs that the eye reads as a
# column: dots closer in height than ORDER_TIE_PX are a tie (either stacking is
# honest), dots more than ORDER_DOT_DX apart horizontally are paired with their
# label by proximity, and labels more than ORDER_X_PX apart horizontally are
# read as separate entries. A stricter, global version of the rule is
# unsatisfiable wherever dots sit closer together than a label is tall.
ORDER_TIE_PX = 4
ORDER_DOT_DX = 80
ORDER_X_PX = 10

# Settle pass (see place_labels settle_pass()): after the repair rounds, each
# label slides toward its own dot to the nearest height that is still clean, so
# a label never stays parked a slot away from its dot just because a neighbour
# held the closer spot at the time it was placed. SETTLE_MAX_PX caps how far
# that search looks (a label further than this from its dot is stranded for a
# reason the repair rounds already found).
SETTLE_MAX_PX = 200

# Bottom of the high-intelligence plot
HIGH_INTELLIGENCE_THRESHOLD = 31
# Right edge of the low cost plot
LOW_COST_THRESHOLD = 0.60


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
    # A step between "touching the dot" and "a full leader-length away":
    # stacked neighbours often need exactly this much clearance, and rounding
    # up to 23pt is what walked whole clusters into far-away stacks.
    (0, 16, "center", "bottom"),
    (0, -16, "center", "top"),
    (10, 15, "left", "bottom"),
    (10, -15, "left", "top"),
    (-10, 15, "right", "bottom"),
    (-10, -15, "right", "top"),
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
    # Wide horizontal slots: last resort when a dense cluster leaves no
    # vertical room, e.g. two dots at the same intelligence level.
    (26, 0, "left", "center"),
    (-26, 0, "right", "center"),
    (26, 9, "left", "bottom"),
    (26, -9, "left", "top"),
    (-26, 9, "right", "bottom"),
    (-26, -9, "right", "top"),
    (26, 23, "left", "bottom"),
    (26, -23, "left", "top"),
    (-26, 23, "right", "bottom"),
    (-26, -23, "right", "top"),
    # Very tall verticals and wide diagonals: escape hatches for dense
    # clusters where every closer slot is taken.
    (0, 94, "center", "bottom"),
    (0, -94, "center", "top"),
    (10, 93, "left", "bottom"),
    (10, -93, "left", "top"),
    (-10, 93, "right", "bottom"),
    (-10, -93, "right", "top"),
    (26, 37, "left", "bottom"),
    (26, -37, "left", "top"),
    (-26, 37, "right", "bottom"),
    (-26, -37, "right", "top"),
    (26, 51, "left", "bottom"),
    (26, -51, "left", "top"),
    (-26, 51, "right", "bottom"),
    (-26, -51, "right", "top"),
    # Even taller verticals: the reference-cost axis squeezes a dozen models
    # into a handful of pixels at the left edge, and labels need to queue
    # several intelligence units away from their dots.
    (0, 108, "center", "bottom"),
    (0, -108, "center", "top"),
    (10, 107, "left", "bottom"),
    (10, -107, "left", "top"),
    (-10, 107, "right", "bottom"),
    (-10, -107, "right", "top"),
    (0, 122, "center", "bottom"),
    (0, -122, "center", "top"),
    (10, 121, "left", "bottom"),
    (10, -121, "left", "top"),
    (-10, 122, "right", "bottom"),
    (-10, -122, "right", "top"),
    (0, 136, "center", "bottom"),
    (0, -136, "center", "top"),
    # Wide horizontal slots: jump the whole left-edge dot/label pile instead
    # of threading it.
    (40, 0, "left", "center"),
    (-40, 0, "right", "center"),
    (54, 0, "left", "center"),
    (-54, 0, "right", "center"),
    (54, 23, "left", "bottom"),
    (54, -23, "left", "top"),
    (-54, 23, "right", "bottom"),
    (-54, -23, "right", "top"),
    # Taller queues still: when the mid-height bands (other models' labels)
    # are full, the label has to climb to empty plot space.
    (0, 150, "center", "bottom"),
    (0, -150, "center", "top"),
    (10, 149, "left", "bottom"),
    (10, -149, "left", "top"),
    (-10, 150, "right", "bottom"),
    (-10, -150, "right", "top"),
    (0, 164, "center", "bottom"),
    (0, -164, "center", "top"),
    (10, 163, "left", "bottom"),
    (10, -163, "left", "top"),
    (-10, 164, "right", "bottom"),
    (-10, -164, "right", "top"),
    (0, 178, "center", "bottom"),
    (0, -178, "center", "top"),
    (10, 177, "left", "bottom"),
    (10, -177, "left", "top"),
    (-10, 178, "right", "bottom"),
    (-10, -178, "right", "top"),
    (0, 192, "center", "bottom"),
    (0, -192, "center", "top"),
    (10, 191, "left", "bottom"),
    (10, -191, "left", "top"),
    (-10, 192, "right", "bottom"),
    (-10, -192, "right", "top"),
    (0, 206, "center", "bottom"),
    (0, -206, "center", "top"),
    (10, 205, "left", "bottom"),
    (10, -205, "left", "top"),
    (-10, 206, "right", "bottom"),
    (-10, -206, "right", "top"),
    (70, 0, "left", "center"),
    (-70, 0, "right", "center"),
    (70, 23, "left", "bottom"),
    (70, -23, "left", "top"),
    (-70, 23, "right", "bottom"),
    (-70, -23, "right", "top"),
]

# Colored vector icons that stand in for the ⚡ emoji: matplotlib cannot render
# color emoji, so without this ⚡ draws as a thin black outline.
ICON_SIZE = {"bolt": 15, "mask": 20}  # px
ICON_GAP = 4  # px between a label's text and its icon
ICON_COLORS = {
    "bolt": ("#fbbf24", "#b45309"),  # amber fill, dark edge
    "mask": ("#1f2937", "#09090b"),  # black fill, blacker edge
}

# Publisher colors, replicated from artificialanalysis.ai
PUBLISHERS = {
    "Accio": "#43674c",
    "Alibaba": "#ff7018",
    "Anthropic": "#cc785c",
    "Apodex": "#30d8d1",
    "DeepSeek": "#2243e6",
    "Google": "#34A853",
    "InclusionAI": "#4fb5ff",
    "Institute of Foundation Models": "#1521a9",
    "Liquid AI": "#000000",
    "Meta": "#0089f4",
    "MiniMax": "#EB3568",
    "Mistral": "#f32e02",
    "Moonshot AI": "#047AFE",
    "OpenAI": "#1f1f1f",
    "OpenBMB": "#3B62EC",
    "Ornith AI": "#dddddd",
    "Prism-ML": "#d6409f",
    "SpaceXAI": "#736cd3",
    "StepFun": "#32f4e6",
    "Tencent": "#66b7fb",
    "Xiaomi": "#fb6d25",
    "Z AI": "#1c7ff8",
}


class LocalHardware(NamedTuple):
    name: str
    idle_power_draw: int | None = None  # Watts when idling
    peak_power_draw: int | None = None  # Watts under max load
    price: int | None = None  # Up-front expense

    @property
    def delta_power_draw(self) -> int:
        assert self.idle_power_draw is not None
        assert self.peak_power_draw is not None
        return self.peak_power_draw - self.idle_power_draw

    def cluster(self, count: int, extra_price: int = 0) -> LocalHardware:
        if self.price is None and extra_price == 0:
            price = None
        else:
            price = (self.price or 0) * count + extra_price
        return LocalHardware(
            f"{self.name} x{count}",
            self.idle_power_draw * count if self.idle_power_draw is not None else None,
            self.peak_power_draw * count if self.peak_power_draw is not None else None,
            price,
        )


RTX3090 = LocalHardware("RTX 3090", 43, 350)
STRIX_HALO = LocalHardware("Strix Halo", 15, 140, price=3100)  # Bosgame
GORGON_HALO = LocalHardware("Gorgon Halo", 15, 140, price=6200)  # Bosgame
DGX_SPARK = LocalHardware("DGX Spark", price=7900)  # Asus Ascent GX10

# Residential electricity, weighted by population, May 2026 (USD/KWh)
# https://www.eia.gov/electricity/monthly/epm_table_grapher.php?t=epmt_5_6_a
US_ELECTRICITY_PRICE = 0.2049


class PricedTokens(NamedTuple):
    """The three token streams a benchmark task is billed on.

    Doubles as AA's cost-per-task breakdown (USD per task), as the sticker
    price those streams were billed at (USD per 1M tokens), and -- after a
    division -- as the task's token counts. Field order is the de-facto
    standard one; `input_` carries the trailing underscore so it does not
    shadow the builtin.

    A cached_input of 0.0 means "no cache": either AA reports no cache reads at
    all, or it publishes no cache-read price for the model, and the task burned
    no cached tokens.
    """

    input_: float  # uncached input tokens, cache writes included
    output: float  # reasoning + final answer tokens
    cached_input: float  # cache reads (cache hits)


class LocalSpeed(NamedTuple):
    """Throughput of a local model on its rig, in tokens/second."""

    prefill: float  # input tokens/s
    decode: float  # output tokens/s


# AA publishes no cost-per-task breakdown for some local models, so their
# uncached input token count is estimated as this multiple of the output token
# count. 3.85 was that multiple's median over the models AA does break down
# when it was picked; across the 54 models it breaks down today the median is
# 3.89, and the constant is left alone rather than retuned so the local rows'
# prices stay put across refreshes. The real ratio is all over the place - 1.98
# on GLM-5.3-Flash (max), 94.75 on Gemini 3.8 Flash, which AA bills for
# close to a hundred cache-write tokens per answer - so this is a coarse middle
# guess, not a per-model fit.
LOCAL_INPUT_TOKEN_RATIO = 3.85


class ProviderType(Enum):
    """How a model's price per task is obtained."""

    LOCAL = "local"
    DATACENTER = "datacenter"


@dataclass
class Model:
    """One model on the plots. Only the first four fields are positional.

    The cost is not computed at construction time: the raw inputs are stored
    and price_per_task() derives the displayed figure on demand. The
    constructor does validate the inputs each provider type needs: LOCAL
    models must have a measured local_speed, DATACENTER models must have
    AA's aa_cost_per_task.
    """

    publisher: str
    name: str
    intelligence: float
    provider_type: ProviderType
    # AA's output tokens per benchmark task (also the task size the local
    # electricity cost is computed from).
    aa_tok_per_task: float = field(kw_only=True)
    # Datacenter models: what one AA task costs, and how it splits. A
    # PricedTokens (USD per task) is AA's split by token type; a bare float is
    # the total only, for the models AA publishes no breakdown for. The split,
    # divided by aa_sticker_price, is what turns AA's dollar breakdown into
    # token counts (see the module docstring); the total is the fallback price
    # and the delta plot's baseline.
    aa_cost_per_task: PricedTokens | float | None = field(default=None, kw_only=True)
    aa_sticker_price: PricedTokens | None = field(default=None, kw_only=True)
    or_slug: str | None = field(default=None, kw_only=True)
    # Recorded for reference only: neither figure prices the plot any more.
    # or_session_cost_10_49_turns is OR's median cost of a real 10-49-turn
    # coding session, or_toks_served the trailing-week volume behind it.
    or_session_cost_10_49_turns: float | None = field(default=None, kw_only=True)
    or_toks_served: int | None = field(default=None, kw_only=True)
    # Datacenter models: OR's synthetic effective prices in $/M tokens: each
    # endpoint instance's trailing-week mean effective price, weighted by the
    # share of the model's tokens that instance served (see
    # .agents/skills/refresh-models). Free endpoints are excluded; a model with
    # no paid endpoint on OR keeps None and plots at AA's price.
    or_eff_input_price: float | None = field(default=None, kw_only=True)
    or_eff_output_price: float | None = field(default=None, kw_only=True)
    # Local models: measured throughput on `hardware` (None = RTX3090).
    local_speed: LocalSpeed | None = field(default=None, kw_only=True)
    # The rig the model runs on. Local models: the measurement rig
    # (None = RTX3090). Datacenter models: the cheapest local rig that
    # can run the model, from the README's "Larger local models" table;
    # only its price is read, as the hardware-cost plot's x.
    hardware: LocalHardware | None = field(default=None, kw_only=True)
    # One-off USD price of the cheapest hardware that can run the model
    # locally, from the README's "Larger local models" table. Independent
    # of the per-task price (which is electricity only): this is the
    # machine, not the power bill. Overrides hardware.price when present,
    # which is how local models measured on the RTX3090 plot at the price
    # of the cheaper rig they actually run on (a phone, a laptop). None =
    # no table entry, so the model is left off the hardware-cost plot.
    hardware_price_override: int | None = field(default=None, kw_only=True)
    # Not on AA: the point is extrapolated/estimated. Rendered as a hollow
    # circle, with a matching "Estimated (not on AA)" legend entry.
    estimated: bool = field(default=False, kw_only=True)
    # Set to False to indicate that the model is not available from datacenters
    available_datacenter: bool = field(default=True, kw_only=True)
    # Set to False to indicate that model weights can't be downloaded
    available_local: bool = field(default=True, kw_only=True)
    # The provider trains on your prompts. Rendered with the thief mask icon
    # and kept out of the Pareto frontier (same offer, worse provider).
    trains_on_your_data: bool = field(default=False, kw_only=True)

    def __post_init__(self) -> None:
        if self.provider_type is ProviderType.LOCAL:
            if self.local_speed is None:
                raise ValueError(f"{self.name}: local model needs local_speed")
            # The ⚡ marker (ICON_PATHS["bolt"]) and the hardware name are
            # spliced into the label here, unlike the trains_on_your_data and
            # available flags, which are drawn as glyphs at render time.
            if self.hardware is None:
                self.hardware = RTX3090
                self.name += " ⚡"
            else:
                self.name += f" ({self.hardware.name} ⚡)"
        elif self.aa_cost_per_task is None:
            raise ValueError(f"{self.name}: datacenter model needs aa_cost_per_task")

    def aa_total_cost_per_task(self) -> float:
        """AA's posted cost of one task, whether or not it is split by type."""
        assert self.aa_cost_per_task is not None
        if isinstance(self.aa_cost_per_task, PricedTokens):
            return sum(self.aa_cost_per_task)
        return self.aa_cost_per_task

    def aa_token_counts(self) -> PricedTokens | None:
        """The task's tokens per stream, or None if AA publishes no split.

        AA's cost-per-task split divided by the sticker price of each stream
        (both in PricedTokens), in tokens. A 0.0 cache price means AA reports no
        cache reads for the model, so the task burned no cached tokens.
        """
        if not isinstance(self.aa_cost_per_task, PricedTokens):
            return None
        if self.aa_sticker_price is None:
            return None
        cost, sticker = self.aa_cost_per_task, self.aa_sticker_price
        assert min(cost.input_, cost.output, sticker.input_, sticker.output) > 0, (
            f"{self.name}: every price we divide by must be non-zero"
        )
        assert min(cost.cached_input, sticker.cached_input) >= 0, (
            f"{self.name}: cache prices cannot be negative"
        )
        return PricedTokens(
            input_=cost.input_ / sticker.input_ * 1e6,
            output=cost.output / sticker.output * 1e6,
            cached_input=(
                cost.cached_input / sticker.cached_input * 1e6
                if sticker.cached_input
                else 0.0
            ),
        )

    def local_token_counts(self) -> PricedTokens:
        """The task's tokens per stream, for the electricity cost.

        Cached input is deliberately zeroed: on a local rig a cache read is
        free, it only costs the electricity to decode the answer. Models AA
        does not break down fall back to LOCAL_INPUT_TOKEN_RATIO x the output
        token count for the uncached input. The output leg is always AA's
        published aa_tok_per_task rather than the split's, which is the same
        figure before AA's rounding.
        """
        counts = self.aa_token_counts()
        input_ = (
            self.aa_tok_per_task * LOCAL_INPUT_TOKEN_RATIO
            if counts is None
            else counts.input_
        )
        return PricedTokens(
            input_=input_, output=self.aa_tok_per_task, cached_input=0.0
        )

    def price_per_task(self) -> float:
        """Displayed USD cost of one AA Intelligence Index task.

        Datacenter models: AA's token mix for the task (uncached + cached
        input, output) priced at OR's synthetic effective input/output prices
        from the trailing week. A model with no OR price, or no AA breakdown to
        get the token mix from, keeps AA's posted cost per task.

        Local models: the electricity to run the task's uncached input tokens
        through prefill and its output tokens through decode on `hardware`.
        """
        if self.provider_type is ProviderType.LOCAL:
            assert self.local_speed is not None
            assert self.hardware is not None
            tokens = self.local_token_counts()
            sec_per_task = (
                tokens.input_ / self.local_speed.prefill
                + tokens.output / self.local_speed.decode
            )
            kwh_per_task = self.hardware.delta_power_draw * sec_per_task / 3_600_000
            return kwh_per_task * US_ELECTRICITY_PRICE

        tokens = self.aa_token_counts()
        if (
            tokens is None
            or self.or_eff_input_price is None
            or self.or_eff_output_price is None
        ):
            return self.aa_total_cost_per_task()
        return (
            (tokens.input_ + tokens.cached_input) * self.or_eff_input_price
            + tokens.output * self.or_eff_output_price
        ) / 1e6

    def price_delta(self) -> float:
        """Percentage change of the displayed price per task from AA's
        published price. Datacenter rows only: a local model's electricity
        cost has no AA price to be a delta against.
        """
        assert self.provider_type is ProviderType.DATACENTER
        return 100 * (self.price_per_task() / self.aa_total_cost_per_task() - 1)

    def hardware_price(self) -> int | None:
        """One-off USD price of the cheapest local hardware that can run
        the model, for the hardware-cost plot: hardware_price_override overrides
        hardware.price when present. None = no entry, so the model is
        left off that plot.
        """
        if self.hardware_price_override is not None:
            return self.hardware_price_override
        elif self.hardware is not None:
            return self.hardware.price
        else:
            return None


MODELS = [
    # Price per task is derived by Model.price_per_task(). Stored inputs:
    # - intelligence + aa_tok_per_task + aa_cost_per_task + aa_sticker_price:
    #   artificialanalysis.ai (AA model page flight payloads, sub-unit /
    #   sub-cent precision). aa_cost_per_task is AA's cost of one task, either
    #   split by token type (a PricedTokens, whose three streams sum to the
    #   total) or as the bare total (a float, for the models AA publishes no
    #   breakdown for); aa_sticker_price is those same three streams in USD per
    #   1M tokens.
    # - datacenter: or_slug + or_eff_input_price / or_eff_output_price
    #   (openrouter.ai, GET /api/frontend/v1/stats/effective-pricing: each
    #   endpoint instance's mean effective price over the trailing week,
    #   weighted by the share of the model's tokens that instance served,
    #   $/M tokens).
    #   or_session_cost_10_49_turns (GET
    #   /api/frontend/v1/rankings/session-cost: median 10-49-turn session
    #   cost, averaged across the coding harnesses that carry the model) and
    #   or_toks_served (GET /api/frontend/v1/rankings/models?view=week:
    #   prompt + completion tokens served in the trailing week, all variants)
    #   are recorded for reference; neither prices the plot.
    # - local: local_speed, both rates on real hardware. The prefill rate is a
    #   guesstimate at 10x the decode rate until it gets measured.
    # - hardware: datacenter models the README's "Larger local models"
    #   table names get the cheapest rig that fits them
    #   (DGX_SPARK.cluster(...) and friends); their price tags the
    #   hardware-cost plot. Local models get the measurement rig
    #   (None = RTX3090).
    # - hardware_price_override: one-off price override for the same plot, for the
    #   local models whose cheapest rig is not a hardware entry (phones,
    #   laptops, prebuilt PCs); overrides hardware.price when present.
    # See .agents/skills/refresh-models for how to re-fetch these numbers.
    # --- Local models (price per task = electricity) ---
    # Guesstimate - not on AA at all: ArtificialAnalysis has no record of LFM3-1.7B
    # and the free API does not list it, so its index comes from the benchmark table
    # at tinyleague.nunodonato.com plus AA's published indices of its neighbours:
    # Nemotron 3 Nano 4B 1.82, Mistral Small 4 4.30, Granite 4.2 3B 4.72, Granite
    # 4.2 8B 5.10, Nemotron 3.5 Lightning 5.53, Command A+ 6.24, Muse Glimmer 6.34,
    # anchored by the inferred indices of Qwen3.6 35B A3B and K2 Horizon 7B below.
    # Only the columns AA itself measures are comparable across the two sources -
    # tinyleague's HLE, AA-LCR, SciCode, GPQA, Terminal-Bench 2.1 and
    # AutomationBench reproduce AA's own numbers to 1-2 points on every model both
    # list, while tau-bench, BFCL, SWE-bench, MRCR and GDPval are vendor-run or
    # differently scaled and were dropped. LFM3's AA-measured columns (HLE 7.17,
    # AA-LCR 52.8, SciCode 28.5, GPQA 63.4, Terminal-Bench 2.1 12.9, Omniscience Elo
    # +0.5) sit between Granite 4.2 3B and Granite 4.2 8B: a least-squares line on
    # their mean z-scored value, monotone interpolation on it and Gaussian-kernel
    # neighbour averaging all place it at 4.7-6.4, leave-one-out MAE 1.1-1.5.
    # That knowledge-only 5.4 ignores why the model exists: tau3-bench 68.0 is the
    # best score in tinyleague's whole 36-model table, ahead of Qwen3.8-27B's 48.0,
    # and 11.2 of AA's 17.0 constituent weight sits in agentic benchmarks
    # (Terminal-Bench Science 3.0, TerminalBench 4.0 2.6, AA-Briefcase 2.2,
    # FrontierCode 2.0, AutomationBench-AA 1.2) that every AA-scored model of this
    # size measures at exactly 0. Assuming AA would give LFM3 mid-range scores of
    # its class there (AA-Briefcase Elo 310 -> 450, AutomationBench-AA 2% -> 5%,
    # TerminalBench 4.0 0 -> 0.005) and pushing the nine-vector through AA's own
    # linear map gives 7.0. Tau3 is a weak lever, not a strong one: Granite 4.2 8B
    # reports 58.1 on the same column while AA measures it at 2.1% AutomationBench-AA
    # and index 5.10. LFM3's own AutomationBench 41.3% is refused outright - AA
    # measures 20.9% for K2 Horizon 7B and 40.5% for GPT-6 Luna (medium), so that is
    # another harness, not a better model; believed, it reads 11.4. Quote +/-1.5.
    # Tokens per task: AA publishes none for this model, so 34286 is Granite 4.2
    # 3B's - the closest AA-scored model of the class. The other tiny model AA
    # scores, Nemotron 3 Nano 4B, spends only 11648 because it answers lazily.
    # Both speed figures are hand-entered guesstimates; tinyleague's
    # memory-bandwidth estimate for a 1.7B on an RTX 4090 is 148 tok/s decode,
    # 101 tok/s at 128K context.
    Model(
        "Liquid AI",
        "LFM3-1.7B",
        7.0,
        ProviderType.LOCAL,
        aa_tok_per_task=34286,
        local_speed=LocalSpeed(prefill=14000, decode=250),
        hardware_price_override=250,  # 4 GB unified: a phone, see README's table
        estimated=True,
    ),
    # Guesstimate - AA's October 2026 rebuild (v4.3 -> v5) dropped the Intelligence
    # Index of this model and of the one below it, while still publishing their
    # individual benchmark scores. Re-estimated in November 2026 by comparing those
    # scores with GPT-6 Luna (low), GPT-6 Luna (medium), Muse Glimmer and
    # Granite 4.2 3B, the four scored models of the same class:
    # AA's v5 index is a linear function of its nine constituent benchmarks, so
    # regressing the published index on the z-scored constituents of the 91 models
    # AA still scores recovers it exactly - rmse 0.47, R^2 0.999, weights 3.0
    # (Terminal-Bench Science), 2.6 (TerminalBench 4.0), 2.2 (FrontierCode 2.0,
    # AA-Briefcase), 2.0 (HLE-Diamond, CritPT v2 Hard), 1.4 (Omniscience), 1.2
    # (AutomationBench-AA), 0.2 (GDP-PDF), intercept 31.6.
    # AA publishes three of the nine for these models today (four for K2 Horizon
    # 7B): AutomationBench-AA is the record's automationBenchPartialScore,
    # TerminalBench 4.0 its terminalBench40, GDP-PDF its gdpPdfAllPass, and
    # Terminal-Bench Science its terminalBenchScience. AA-Briefcase and
    # Omniscience survive the rebuild unchanged and are read from AA's pre-rebuild
    # record of the same model: across the 88 models AA scored in both revisions,
    # its re-run of AA-Briefcase moved the Elo by sd 7.7 (0.04 index points) and
    # Omniscience by sd 2.9 (0.17 points), so the old number is the current one.
    # What is left to invent is what the rebuild introduced and AA never ran on
    # these models - FrontierCode 2.0, HLE-Diamond, CritPT v2 Hard, and Terminal-
    # Bench Science for the smaller one. Each is imputed by a ridge fit
    # (alpha 0.1) on the eight benchmarks both targets and all four anchors
    # report - GDPVal, HLE, LCR, SciCode, CritPT, AutomationBench-AA, TerminalBench
    # 4.0, GDP-PDF - weighted by a Gaussian kernel of bandwidth 1.25 z so that the
    # neighbouring models, not the frontier ones, set the slope. Pushing the
    # completed nine-vector through the map above gives the index.
    # Anchors as the check: each of the four, sent through the same pipeline with
    # its own constituent scores hidden and itself left out of every fit, comes back
    # at 10.75 (published 9.67), 17.73 (18.30), 7.13 (6.34) and 3.80 (4.72) - MAE
    # 0.84, worst 1.08. Leave-one-out over the 24 scored models below index 22
    # misses by 1.0-1.1 points on average (+0.25 bias), so quote +/-1.1.
    # Robustness: adding GPQA / TerminalBench 2.1 / tau-banking / MMMU-Pro / MLCR
    # moves each estimate by <= 0.3, the bandwidth 0.9-1.8 by <= 0.5; trusting no
    # pre-rebuild data at all reads 12.1 and 10.2 instead.
    # 8.9 sits just under GPT-6 Luna (low) (9.67), whose AutomationBench-AA (0.120
    # vs 0.052), GDPVal (1040 vs 899) and Omniscience (-9.2 vs -22.2) it loses to,
    # and just above Muse Glimmer (6.34) on the strength of AA-Briefcase Elo (708 vs
    # 478), Omniscience (-22.2 vs -32.9) and GDPVal (899 vs 792), even though it
    # trails Glimmer on AutomationBench-AA, LCR, SciCode, GDP-PDF and CritPT.
    Model(
        "Alibaba",
        "Qwen3.6-35B-A3B",
        8.9,
        ProviderType.LOCAL,
        aa_tok_per_task=57589,
        aa_cost_per_task=PricedTokens(0.66, 0.13, 0.0),
        # AA reports no cache reads for this model and publishes no cache-read
        # price; 0.0 in the cached slot of both splits means "never caches"
        aa_sticker_price=PricedTokens(0.375, 2.25, 0.0),
        local_speed=LocalSpeed(prefill=2215, decode=150),
        hardware_price_override=2300,  # RTX 3090 + 32 GB RAM
        estimated=True,
    ),
    # 11.7 clears GPT-6 Luna (low) (9.67) because it beats it on AutomationBench-AA
    # (0.209 vs 0.120), AA-Briefcase Elo (1018 vs 740) and TerminalBench 4.0
    # (0.010 vs 0.000), and stays well under GPT-6 Luna (medium) (18.30), which is
    # ahead of it on every benchmark the two share. Its Terminal-Bench Science of
    # 0.0 is AA's own measurement, not an imputation.
    Model(
        "Institute of Foundation Models",
        "K2 Horizon 7B",
        11.7,
        ProviderType.LOCAL,
        aa_tok_per_task=113872,
        local_speed=LocalSpeed(prefill=3430, decode=104),
        hardware_price_override=1000,
        estimated=True,
    ),
    Model(
        "Alibaba",
        "Qwen3.8-27B",
        20.1068756031705,
        ProviderType.LOCAL,
        aa_tok_per_task=109802,
        aa_cost_per_task=PricedTokens(
            2.0194864181153562, 0.3294063915269094, 1.1806566654988
        ),
        aa_sticker_price=PricedTokens(0.5, 3.0, 0.1),
        local_speed=LocalSpeed(prefill=963, decode=57),
        hardware_price_override=2400,
    ),
    # Not on AA: intelligence = 0.9165 x Qwen3.8-27B, the 91.65% of BF16 that
    # ByteShape measured for this ternary quant (see README note); tokens per
    # task assumed identical to Qwen3.8-27B.
    Model(
        "Prism-ML",
        "Ternary-Bonsai-2",
        0.9165 * 20.1068756031705,
        ProviderType.LOCAL,
        # Verbatim copies from Qwen3.8-27B. This doesn't keep in to consideration that
        # lower quants tend to be more verbose.
        aa_tok_per_task=109802,
        aa_cost_per_task=PricedTokens(
            2.0194864181153562, 0.3294063915269094, 1.1806566654988
        ),
        aa_sticker_price=PricedTokens(0.5, 3.0, 0.1),
        # Actual measure
        local_speed=LocalSpeed(prefill=949, decode=55),
        hardware_price_override=1000,
        estimated=True,
    ),
    # Guesstimate - not on AA: its record (Qwen3.8-Flash-Next) carries no
    # Intelligence Index since the v5 rebuild and the free API does not list it
    # at all, but the page does publish its individual benchmark scores. Those are
    # enough because AA's v5 index is itself just a linear function of its nine
    # constituent benchmarks: regressing the index on the z-scored constituents of
    # the 92 re-benchmarked models (AA names them in each record's
    # intelligenceIndexEvaluations: AA-Briefcase, AutomationBench-AA,
    # TerminalBench 4.0, FrontierCode 2.0, Omniscience, GDP-PDF, HLE-Diamond,
    # CritPT v2 Hard, Terminal-Bench Science) reproduces every published index to
    # rmse 0.47, R^2 0.999. The weights come out as 3.0 (Terminal-Bench Science),
    # 2.6 (TerminalBench 4.0), 2.2 (FrontierCode 2.0), 2.1 (AA-Briefcase),
    # 2.0 (HLE-Diamond, CritPT v2 Hard), 1.4 (Omniscience), 1.2
    # (AutomationBench-AA), 0.2 (GDP-PDF), plus a 31.5 intercept.
    # Qwen3.8-Flash reports six of the nine; FrontierCode 2.0, CritPT v2 Hard and
    # Terminal-Bench Science - the three heaviest-weighted - AA has never run on
    # it, so they are imputed by regressing each of them on the six it reports.
    # Neighbours as the anchor: the recovered map reproduces GLM-5.3-Flash's
    # published 27.75 and MiMo-V2.6-Flash's 26.73 to within 0.02, so pushing it
    # through either of them gives the same number - 27.25 from GLM-5.3-Flash,
    # 27.26 from MiMo-V2.6-Flash. Take 27.2, +/-1.7 (running the same procedure on
    # models that do have an index misses by 1.7 points on average, 1.9 inside the
    # 24-31 band, with the target's whole model family held out of the fit).
    # Reading it: Qwen3.8-Flash lands between its two neighbours because it beats
    # both on AA-Briefcase (+124 Elo over GLM-5.3-Flash) and GDP-PDF, sits between
    # them on TerminalBench 4.0 and HLE-Diamond, and loses to both on
    # AutomationBench and Omniscience. Adding its non-constituent scores (CritPT
    # 0.11, HLE 0.38, MMMU-Pro 0.80) to the imputation reads ~1 point low - that
    # variant misses the 24-31 band by -0.9 when checked against models with a
    # published index, while the six-benchmark version is off by only -0.2.
    Model(
        "Alibaba",
        "Qwen3.8-Flash",
        27.2,
        ProviderType.LOCAL,
        aa_tok_per_task=167387,
        aa_cost_per_task=PricedTokens(0.077, 0.079, 0.42),
        aa_sticker_price=PricedTokens(0.15, 0.47, 0.016),
        local_speed=LocalSpeed(prefill=2000, decode=84),
        hardware_price_override=2700,  # 5060 Ti 16 GiB + 48 GiB RAM (Strata Gyro-M)
        estimated=True,
    ),
    # --- Datacenter models (price per task = AA's token mix priced at OR's
    # effective input/output prices) ---
    # Datacenter twin of the row above; same reconstructed intelligence (see the
    # note there).
    Model(
        "Alibaba",
        "Qwen3.8-Flash",
        27.2,
        ProviderType.DATACENTER,
        aa_tok_per_task=167387,
        aa_cost_per_task=PricedTokens(0.077, 0.079, 0.422),
        aa_sticker_price=PricedTokens(0.15, 0.47, 0.016),
        or_slug="qwen/qwen3.8-flash-20260826",
        or_session_cost_10_49_turns=0.04176418,
        or_toks_served=848911399926,
        or_eff_input_price=0.03395,
        or_eff_output_price=0.4695,
        estimated=True,
    ),
    Model(
        "Alibaba",
        "Qwen3.8 Max (0902)",
        33.6545627367888,
        ProviderType.DATACENTER,
        aa_tok_per_task=170794,
        aa_cost_per_task=PricedTokens(
            3.330158763560127, 1.0247637690848763, 5.668969038896941
        ),
        aa_sticker_price=PricedTokens(2.0, 6.0, 0.25),
        or_slug="qwen/qwen3.8-max-20260902",
        or_session_cost_10_49_turns=0.6514531299999999,
        or_toks_served=172519102623,
        or_eff_input_price=0.3992,
        or_eff_output_price=6.0,
    ),
    # Datacenter version (MXFP4 QAT)
    Model(
        "DeepSeek",
        "DeepSeek V4.1 Flash 1017",
        32.5896393532027,
        ProviderType.DATACENTER,
        aa_tok_per_task=138962,
        aa_cost_per_task=PricedTokens(
            0.2571473726554651, 0.16675404706375513, 0.1782037281167598
        ),
        aa_sticker_price=PricedTokens(0.3, 1.2, 0.006),
        or_slug="deepseek/deepseek-v4.1-flash-20261017",
        or_session_cost_10_49_turns=0.02998322775,
        or_toks_served=56045650403726,
        or_eff_input_price=0.01524,
        or_eff_output_price=0.6597,
    ),
    # Degraded local model
    Model(
        "DeepSeek",
        "DeepSeek V4.1 Flash 1017 (UD-IQ1_M)",
        # https://unsloth.ai/docs/models/deepseek-v4.1-flash
        # Average recovery over 6 benchmarks; Unsloth 3.1
        32.5896393532027 * 85.7 / 87.8,
        ProviderType.LOCAL,
        # https://unsloth.ai/docs/models/deepseek-v4.1-flash#quantization-increases-reasoning-lengths
        aa_tok_per_task=138962 * 2.8 / 1.5,
        aa_cost_per_task=PricedTokens(
            0.2571473726554651, 0.16675404706375513 * 2.8 / 1.5, 0.1782037281167598
        ),
        aa_sticker_price=PricedTokens(0.3, 1.2, 0.006),
        # From https://github.com/Niko1221/Strata @ 2.0bpw; UD-IQ1_M is 1.95bpw
        local_speed=LocalSpeed(prefill=320, decode=21),
        hardware_price_override=5500,  # R9700 + 128 GiB RAM (Strata)
        estimated=True,
    ),
    # Local model with negligible performance loss
    Model(
        "DeepSeek",
        "DeepSeek V4.1 Flash 1017 (UD-Q2_K_XL)",
        # https://unsloth.ai/docs/models/deepseek-v4.1-flash
        # Average recovery over 6 benchmarks; Unsloth 3.1
        32.5896393532027,
        ProviderType.LOCAL,
        # https://unsloth.ai/docs/models/deepseek-v4.1-flash#quantization-increases-reasoning-lengths
        aa_tok_per_task=138962 * 2.1 / 1.5,
        aa_cost_per_task=PricedTokens(
            0.2571473726554651, 0.16675404706375513 * 2.1 / 1.5, 0.1782037281167598
        ),
        aa_sticker_price=PricedTokens(0.3, 1.2, 0.006),
        # From https://github.com/Niko1221/Strata @ 2.0bpw; UD-Q2_K_XL is 2.8bpw
        local_speed=LocalSpeed(prefill=320, decode=21 * 2.0 / 2.8),
        hardware_price_override=6400,  # R9700 + 192 GiB RAM (Strata)
    ),
    Model(
        "Tencent",
        "Hy4",
        34.5552385392508,
        ProviderType.DATACENTER,
        aa_tok_per_task=78058,
        aa_cost_per_task=PricedTokens(
            0.49304540305271105, 0.19522317789568608, 0.34283748969140015
        ),
        # nominal list price; OR's endpoint carries a 0.5 launch discount
        aa_sticker_price=PricedTokens(0.834, 2.501, 0.042),
        or_slug="tencent/hy4-20261104",
        or_toks_served=12800211241293,
        or_eff_input_price=0.07983,
        or_eff_output_price=2.416,
        # GGUFs not available yet; estimated same as GLM @ UD-Q3_K_XL
        hardware=DGX_SPARK.cluster(3, extra_price=300),
        available_local=False,
    ),
    Model(
        "Meta",
        "Muse Spark 1.3",
        36.6172039139368,
        ProviderType.DATACENTER,
        aa_tok_per_task=91114,
        aa_cost_per_task=PricedTokens(
            0.2848644588586805, 0.3872337907751061, 1.7136831361769378
        ),
        aa_sticker_price=PricedTokens(1.25, 4.25, 0.15),
        or_slug="meta/muse-spark-1.3-20260902",
        or_session_cost_10_49_turns=0.5343645,
        or_toks_served=88197751138,
        or_eff_input_price=0.4117,
        or_eff_output_price=4.25,
    ),
    Model(
        "Meta",
        "Muse Spark 1.3",
        36.6172039139368,
        ProviderType.DATACENTER,
        aa_tok_per_task=91114,
        aa_cost_per_task=PricedTokens(
            0.2848644588586805, 0.3872337907751061, 1.7136831361769378
        ),
        aa_sticker_price=PricedTokens(1.25, 4.25, 0.15),
        or_slug="meta/muse-spark-1.3-contributor-20260902",
        or_session_cost_10_49_turns=0.04808478,
        or_toks_served=2798110625445,
        or_eff_input_price=0.009841,
        or_eff_output_price=0.1996,
        trains_on_your_data=True,
    ),
    Model(
        "Meta",
        "Muse Wonder",
        43.5748225998431,
        ProviderType.DATACENTER,
        aa_tok_per_task=96063,
        aa_cost_per_task=PricedTokens(
            1.2403611645930221, 1.921261981835536, 3.790413332177666
        ),
        aa_sticker_price=PricedTokens(4.0, 20.0, 0.2),
        or_slug="meta/muse-wonder-20261012",
        or_toks_served=119260703680,
        or_eff_input_price=0.5932,
        or_eff_output_price=20.0,
    ),
    # Released 2026-11-09: OR's trailing week is a single day so far, and OR has
    # no session-cost or weekly-volume history for the permaslug yet. AA scores
    # only the (Max) record of this ladder.
    Model(
        "Meta",
        "Muse Ember",
        36.0212987864912,
        ProviderType.DATACENTER,
        aa_tok_per_task=84482,
        aa_cost_per_task=PricedTokens(
            0.027216886792402466, 0.042240832668746175, 0.06647789743748768
        ),
        aa_sticker_price=PricedTokens(0.1, 0.5, 0.004),
        or_slug="meta/muse-ember-20261109",
        or_eff_input_price=0.02773,
        or_eff_output_price=0.4998,
    ),
    Model(
        "Z AI",
        "GLM-5.3-Flash (high)",
        # scaled from the (max) record, see the README note
        27.803889160883 * 28.01 / 28.99,
        ProviderType.DATACENTER,
        aa_tok_per_task=round(134205 * 70610 / 138690),
        # the token split scales with the token count; the sticker prices are
        # the model's list prices and do not
        aa_cost_per_task=PricedTokens(
            0.039832743493184115 * 70610 / 138690,
            0.0671024626915514 * 70610 / 138690,
            0.4763751378166728 * 70610 / 138690,
        ),
        aa_sticker_price=PricedTokens(0.15, 0.5, 0.03),
        or_slug="z-ai/glm-5.3-flash-20260826",
        or_session_cost_10_49_turns=0.03630717025,
        or_toks_served=15914499291865,
        or_eff_input_price=0.04489,
        or_eff_output_price=0.409,
        estimated=True,
    ),
    Model(
        "Z AI",
        "GLM-5.3-Flash (max)",
        27.803889160883,
        ProviderType.DATACENTER,
        aa_tok_per_task=134205,
        aa_cost_per_task=PricedTokens(
            0.039832743493184115, 0.0671024626915514, 0.4763751378166728
        ),
        aa_sticker_price=PricedTokens(0.15, 0.5, 0.03),
        or_slug="z-ai/glm-5.3-flash-20260826",
        or_session_cost_10_49_turns=0.0363167315,
        or_toks_served=13226707926719,
        or_eff_input_price=0.04485,
        or_eff_output_price=0.4164,
        hardware=STRIX_HALO,  # IQ3_XXS
    ),
    Model(
        "Z AI",
        "GLM-5.3",
        31.653381843882,
        ProviderType.DATACENTER,
        aa_tok_per_task=118987,
        aa_cost_per_task=PricedTokens(
            0.35155644931635205, 0.5235406213267805, 4.134167531076821
        ),
        aa_sticker_price=PricedTokens(1.4, 4.4, 0.26),
        # 384 GiB unified: 3x DGX Spark
        hardware=DGX_SPARK.cluster(3, extra_price=300),  # Q3_K_XL
        or_slug="z-ai/glm-5.3-20260816",
        or_session_cost_10_49_turns=0.196541875,
        or_toks_served=3648972624074,
        or_eff_input_price=0.2009,
        or_eff_output_price=3.214,
    ),
    Model(
        "Moonshot AI",
        "Kimi K3.1 Pro",
        46.4526236744189,
        ProviderType.DATACENTER,
        aa_tok_per_task=72239,
        aa_cost_per_task=PricedTokens(
            0.8567468935932564, 0.722389702315149, 2.1020636936657997
        ),
        aa_sticker_price=PricedTokens(2.0, 10.0, 0.2),
        # 1 TB VRAM: 8x DGX Spark + 4-ports 400G switch + 2x 400G to 4x100G cables
        hardware=DGX_SPARK.cluster(8, extra_price=1200 + 100 * 2),
        or_slug="moonshotai/kimi-k3.1-pro-20261023",
        or_session_cost_10_49_turns=0.7708645333333334,
        or_toks_served=4370794565301,
        or_eff_input_price=0.3321,
        or_eff_output_price=10.23,
    ),
    # Not on AA anymore. Intelligence extrapolated from Gemini 3.8 Flash and Gemini 3.5
    # Flash-Lite. Tokens per task arbitrarily set the same as Gemini 3.8 Flash.
    Model(
        "Google",
        "Gemini 3.5 Flash",
        21.0,
        ProviderType.DATACENTER,
        aa_tok_per_task=98_940,
        aa_cost_per_task=PricedTokens(7.03 * 2, 0.37 * 9 / 3.75, 0.36 * 2),
        aa_sticker_price=PricedTokens(1.5, 9.0, 0.15),
        or_slug="google/gemini-3.5-flash-20260519",
        or_session_cost_10_49_turns=0.4776147,
        or_toks_served=47630774789,
        or_eff_input_price=0.9748,
        or_eff_output_price=8.956,
        estimated=True,
    ),
    Model(
        "Google",
        "Gemini 3.8 Flash",
        30.8468119957291,
        ProviderType.DATACENTER,
        aa_tok_per_task=98_940,
        aa_cost_per_task=PricedTokens(
            7.030934890491321, 0.3710231777219156, 0.36404810993927716
        ),
        aa_sticker_price=PricedTokens(0.75, 3.75, 0.075),
        or_slug="google/gemini-3.8-flash-20260902",
        or_session_cost_10_49_turns=0.270219105,
        or_toks_served=1583336750082,
        or_eff_input_price=0.3381,
        or_eff_output_price=4.095,
    ),
    Model(
        "Google",
        "Gemini 4 Argon (high)",
        50.4121935905389,
        ProviderType.DATACENTER,
        aa_tok_per_task=116368,
        aa_cost_per_task=PricedTokens(
            5.666941901480672, 2.3273605813566736, 1.9523062287131876
        ),
        aa_sticker_price=PricedTokens(4.0, 20.0, 0.2),
        or_slug="google/gemini-4-argon-20261013",
        or_session_cost_10_49_turns=1.04640594,
        or_toks_served=1171080932024,
        or_eff_input_price=0.6281,
        or_eff_output_price=9.938,
    ),
    # Not released yet: AA has benchmarked it, but OpenRouter carries no price
    # chart for it, so it plots at AA's posted cost and renders greyed out.
    Model(
        "Google",
        "Gemini 4 Carbon",
        60.025738829526,
        ProviderType.DATACENTER,
        aa_tok_per_task=122768,
        aa_cost_per_task=PricedTokens(
            3.2610441869737987, 2.4553559174861244, 2.0391456408266344
        ),
        aa_sticker_price=PricedTokens(4.0, 20.0, 0.2),
        available_datacenter=False,
    ),
    Model(
        "SpaceXAI",
        "Grok 4.7",
        35.4850117067196,
        ProviderType.DATACENTER,
        aa_tok_per_task=104531,
        aa_cost_per_task=PricedTokens(
            2.576886728299715, 0.6271856443241712, 4.567467028963252
        ),
        aa_sticker_price=PricedTokens(2.0, 6.0, 0.5),
        or_slug="x-ai/grok-4.7-20260916",
        or_session_cost_10_49_turns=0.9011245,
        or_toks_served=235375898585,
        or_eff_input_price=0.8426,
        or_eff_output_price=6.261,
    ),
    Model(
        "StepFun",
        "Step 5 Preview",
        28.5283090216016,
        ProviderType.DATACENTER,
        aa_tok_per_task=113473,
        aa_cost_per_task=PricedTokens(
            0.5051213025810452, 0.30637676984880086, 1.5337544881906702
        ),
        aa_sticker_price=PricedTokens(1.0, 2.7, 0.1),
        # 384 GiB unified: 3x DGX Spark
        hardware=DGX_SPARK.cluster(3, extra_price=300),
        or_slug="stepfun/step-5-preview-20261008",
        or_eff_input_price=0.1451,
        or_eff_output_price=2.7,
        or_toks_served=7201181339958,
    ),
    Model(
        "Xiaomi",
        "MiMo-V2.6-Flash",
        26.7276981444568,
        ProviderType.DATACENTER,
        aa_tok_per_task=131806,
        aa_cost_per_task=PricedTokens(
            0.04318792692902057, 0.03690574314577558, 0.0484362007653506
        ),
        aa_sticker_price=PricedTokens(0.14, 0.28, 0.0028),
        hardware=STRIX_HALO,  # 3.0bpw
        or_slug="xiaomi/mimo-v2.6-flash-20260921",
        or_session_cost_10_49_turns=0.030579708666666667,
        or_toks_served=2250334822262,
        or_eff_input_price=0.01746,
        or_eff_output_price=0.2792,
    ),
    Model(
        "Xiaomi",
        "MiMo-V2.6-Pro",
        29.8897833093321,
        ProviderType.DATACENTER,
        aa_tok_per_task=121073,
        aa_cost_per_task=PricedTokens(
            0.10909531014918307, 0.10533362295314275, 0.04275142965643572
        ),
        aa_sticker_price=PricedTokens(0.435, 0.87, 0.0036),
        hardware=DGX_SPARK.cluster(3, extra_price=300),  # 3.0bpw
        or_slug="xiaomi/mimo-v2.6-pro-20260921",
        or_session_cost_10_49_turns=0.08711089,
        or_toks_served=1152194500674,
        or_eff_input_price=0.08991,
        or_eff_output_price=1.592,
    ),
    Model(
        "MiniMax",
        "MiniMax-M3.1-Flash",
        29.0595214290458,
        ProviderType.DATACENTER,
        aa_tok_per_task=138176,
        aa_cost_per_task=PricedTokens(
            0.11282161400286993, 0.08290581497833605, 0.29524324450332606
        ),
        aa_sticker_price=PricedTokens(0.15, 0.6, 0.015),
        or_slug="minimax/minimax-m3.1-flash-20261019",
        or_session_cost_10_49_turns=0.0387672,
        or_toks_served=5006869054401,
        or_eff_input_price=0.02081,
        or_eff_output_price=0.4916,
        hardware=GORGON_HALO,
    ),
    Model(
        "OpenAI",
        "GPT-6 Luna (low)",
        9.66610812336462,
        ProviderType.DATACENTER,
        aa_tok_per_task=3296,
        aa_cost_per_task=PricedTokens(
            0.0023262275757420735, 0.0016479451676949762, 0.001141082445083342
        ),
        aa_sticker_price=PricedTokens(0.1, 0.5, 0.01),
        or_slug="openai/gpt-6-luna-20260922",
        or_session_cost_10_49_turns=0.026637255500000002,
        or_toks_served=7275149316386,
        or_eff_input_price=0.04443,
        or_eff_output_price=0.5012,
    ),
    Model(
        "OpenAI",
        "GPT-6 Luna (medium)",
        18.4542156192416,
        ProviderType.DATACENTER,
        aa_tok_per_task=19823,
        aa_cost_per_task=PricedTokens(
            0.008684756771769251, 0.009911619090008737, 0.0236054602447598
        ),
        aa_sticker_price=PricedTokens(0.1, 0.5, 0.01),
        or_slug="openai/gpt-6-luna-20260922",
        or_session_cost_10_49_turns=0.026637255500000002,
        or_toks_served=7275149316386,
        or_eff_input_price=0.04443,
        or_eff_output_price=0.5012,
    ),
    Model(
        "OpenAI",
        "GPT-6 Luna (high)",
        21.4290170864582,
        ProviderType.DATACENTER,
        aa_tok_per_task=32839,
        aa_cost_per_task=PricedTokens(
            0.013260173743129955, 0.016419570614591268, 0.04326802037422575
        ),
        aa_sticker_price=PricedTokens(0.1, 0.5, 0.01),
        or_slug="openai/gpt-6-luna-20260922",
        or_session_cost_10_49_turns=0.026637255500000002,
        or_toks_served=7275149316386,
        or_eff_input_price=0.04443,
        or_eff_output_price=0.5012,
    ),
    Model(
        "OpenAI",
        "GPT-6 Luna (xhigh)",
        23.4508728089378,
        ProviderType.DATACENTER,
        aa_tok_per_task=43893,
        aa_cost_per_task=PricedTokens(
            0.017039746974944057, 0.021946607051674276, 0.06510685518459125
        ),
        aa_sticker_price=PricedTokens(0.1, 0.5, 0.01),
        or_slug="openai/gpt-6-luna-20260922",
        or_session_cost_10_49_turns=0.026637255500000002,
        or_toks_served=7275149316386,
        or_eff_input_price=0.04443,
        or_eff_output_price=0.5012,
    ),
    Model(
        "OpenAI",
        "GPT-6 Luna (max)",
        27.8552004938153,
        ProviderType.DATACENTER,
        aa_tok_per_task=75296,
        aa_cost_per_task=PricedTokens(
            0.024671461535684842, 0.03764802618281935, 0.08963626960255419
        ),
        aa_sticker_price=PricedTokens(0.1, 0.5, 0.01),
        or_slug="openai/gpt-6-luna-20260922",
        or_session_cost_10_49_turns=0.026637255500000002,
        or_toks_served=7275149316386,
        or_eff_input_price=0.04443,
        or_eff_output_price=0.5012,
    ),
    # GPT-6.1 Terra released 2026-10-26, after OR's session-cost window rolled
    # (windowEnd 2026-10-25), so it carries no session-cost figure yet.
    Model(
        "OpenAI",
        "GPT-6.1 Terra (low)",
        29.4169280792131,
        ProviderType.DATACENTER,
        aa_tok_per_task=4422,
        aa_cost_per_task=PricedTokens(
            0.03421288115320557, 0.022110861337539173, 0.015013819560952543
        ),
        aa_sticker_price=PricedTokens(1.0, 5.0, 0.05),
        or_slug="openai/gpt-6.1-terra-20261026",
        or_session_cost_10_49_turns=0.236225,
        or_toks_served=1054825108400,
        or_eff_input_price=0.2466,
        or_eff_output_price=5.146,
    ),
    Model(
        "OpenAI",
        "GPT-6.1 Terra (medium)",
        38.1652076677975,
        ProviderType.DATACENTER,
        aa_tok_per_task=10875,
        aa_cost_per_task=PricedTokens(
            0.06293867838538772, 0.05437731370395206, 0.051488590294554354
        ),
        aa_sticker_price=PricedTokens(1.0, 5.0, 0.05),
        or_slug="openai/gpt-6.1-terra-20261026",
        or_session_cost_10_49_turns=0.236225,
        or_toks_served=1054825108400,
        or_eff_input_price=0.2466,
        or_eff_output_price=5.146,
    ),
    Model(
        "OpenAI",
        "GPT-6.1 Terra (high)",
        42.4344508712563,
        ProviderType.DATACENTER,
        aa_tok_per_task=16746,
        aa_cost_per_task=PricedTokens(
            0.08723872309895113, 0.08373238661179763, 0.08830853690622348
        ),
        aa_sticker_price=PricedTokens(1.0, 5.0, 0.05),
        or_slug="openai/gpt-6.1-terra-20261026",
        or_session_cost_10_49_turns=0.236225,
        or_toks_served=1054825108400,
        or_eff_input_price=0.2466,
        or_eff_output_price=5.146,
    ),
    Model(
        "OpenAI",
        "GPT-6.1 Terra (xhigh)",
        45.250051789066,
        ProviderType.DATACENTER,
        aa_tok_per_task=26200,
        aa_cost_per_task=PricedTokens(
            0.12107684361417946, 0.1309982959158254, 0.14861386213662026
        ),
        aa_sticker_price=PricedTokens(1.0, 5.0, 0.05),
        or_slug="openai/gpt-6.1-terra-20261026",
        or_session_cost_10_49_turns=0.236225,
        or_toks_served=1054825108400,
        or_eff_input_price=0.2466,
        or_eff_output_price=5.146,
    ),
    Model(
        "OpenAI",
        "GPT-6.1 Terra (max)",
        51.3265188577015,
        ProviderType.DATACENTER,
        aa_tok_per_task=59883,
        aa_cost_per_task=PricedTokens(
            0.23526757467699644, 0.2994158235895208, 0.4264246328151169
        ),
        aa_sticker_price=PricedTokens(1.0, 5.0, 0.05),
        or_slug="openai/gpt-6.1-terra-20261026",
        or_session_cost_10_49_turns=0.236225,
        or_toks_served=1054825108400,
        or_eff_input_price=0.2466,
        or_eff_output_price=5.146,
    ),
    Model(
        "OpenAI",
        "GPT-6.1 Sol (low)",
        34.8029244314559,
        ProviderType.DATACENTER,
        aa_tok_per_task=5838,
        aa_cost_per_task=PricedTokens(
            0.07052627255637522, 0.05837994676258293, 0.03256735899926756
        ),
        aa_sticker_price=PricedTokens(2.0, 10.0, 0.1),
        or_slug="openai/gpt-6.1-sol-20260929",
        or_session_cost_10_49_turns=0.4584023875,
        or_toks_served=4001255063816,
        or_eff_input_price=0.4247,
        or_eff_output_price=10.87,
    ),
    Model(
        "OpenAI",
        "GPT-6.1 Sol (medium)",
        46.3468797067114,
        ProviderType.DATACENTER,
        aa_tok_per_task=13518,
        aa_cost_per_task=PricedTokens(
            0.1275050193232428, 0.13518335935178977, 0.09670148612754065
        ),
        aa_sticker_price=PricedTokens(2.0, 10.0, 0.1),
        or_slug="openai/gpt-6.1-sol-20260929",
        or_session_cost_10_49_turns=0.4584023875,
        or_toks_served=4001255063816,
        or_eff_input_price=0.4247,
        or_eff_output_price=10.87,
    ),
    Model(
        "OpenAI",
        "GPT-6.1 Sol (high)",
        50.3291231936536,
        ProviderType.DATACENTER,
        aa_tok_per_task=22606,
        aa_cost_per_task=PricedTokens(
            0.19572874763810777, 0.22606199071777155, 0.1691946132222032
        ),
        aa_sticker_price=PricedTokens(2.0, 10.0, 0.1),
        or_slug="openai/gpt-6.1-sol-20260929",
        or_session_cost_10_49_turns=0.4584023875,
        or_toks_served=4001255063816,
        or_eff_input_price=0.4247,
        or_eff_output_price=10.87,
    ),
    Model(
        "OpenAI",
        "GPT-6.1 Sol (xhigh)",
        52.458219253291,
        ProviderType.DATACENTER,
        aa_tok_per_task=30241,
        aa_cost_per_task=PricedTokens(
            0.2350380936395193, 0.3024118394747384, 0.22654500250139165
        ),
        aa_sticker_price=PricedTokens(2.0, 10.0, 0.1),
        or_slug="openai/gpt-6.1-sol-20260929",
        or_session_cost_10_49_turns=0.4584023875,
        or_toks_served=4001255063816,
        or_eff_input_price=0.4247,
        or_eff_output_price=10.87,
    ),
    Model(
        "OpenAI",
        "GPT-6.1 Sol (max)",
        54.6087338138148,
        ProviderType.DATACENTER,
        aa_tok_per_task=67824,
        aa_cost_per_task=PricedTokens(
            0.3420645993423942, 0.678236917377782, 0.25176175122452343
        ),
        aa_sticker_price=PricedTokens(2.0, 10.0, 0.1),
        or_slug="openai/gpt-6.1-sol-20260929",
        or_session_cost_10_49_turns=0.4584023875,
        or_toks_served=4001255063816,
        or_eff_input_price=0.4247,
        or_eff_output_price=10.87,
    ),
    Model(
        "OpenAI",
        "GPT-6 Astra (low)",
        43.3349823228524,
        ProviderType.DATACENTER,
        aa_tok_per_task=9126,
        aa_cost_per_task=PricedTokens(
            0.45517619041814145, 0.45629044998074986, 0.5122563231674682
        ),
        aa_sticker_price=PricedTokens(10.0, 50.0, 1.0),
        or_slug="openai/gpt-6-astra-20260903",
        or_session_cost_10_49_turns=2.8218648,
        or_toks_served=330995343765,
        or_eff_input_price=2.95,
        or_eff_output_price=50.96,
    ),
    Model(
        "OpenAI",
        "GPT-6 Astra (medium)",
        51.5548393864113,
        ProviderType.DATACENTER,
        aa_tok_per_task=18052,
        aa_cost_per_task=PricedTokens(
            0.7526100723857124, 0.9025812213565696, 1.0772476392711448
        ),
        aa_sticker_price=PricedTokens(10.0, 50.0, 1.0),
        or_slug="openai/gpt-6-astra-20260903",
        or_session_cost_10_49_turns=2.8218648,
        or_toks_served=330995343765,
        or_eff_input_price=2.95,
        or_eff_output_price=50.96,
    ),
    Model(
        "OpenAI",
        "GPT-6 Astra (high)",
        54.849838322506,
        ProviderType.DATACENTER,
        aa_tok_per_task=23133,
        aa_cost_per_task=PricedTokens(
            0.8835257754891446, 1.1566734818244009, 1.198372711238285
        ),
        aa_sticker_price=PricedTokens(10.0, 50.0, 1.0),
        or_slug="openai/gpt-6-astra-20260903",
        or_session_cost_10_49_turns=2.8218648,
        or_toks_served=330995343765,
        or_eff_input_price=2.95,
        or_eff_output_price=50.96,
    ),
    Model(
        "OpenAI",
        "GPT-6 Astra (xhigh)",
        55.3057468569681,
        ProviderType.DATACENTER,
        aa_tok_per_task=26724,
        aa_cost_per_task=PricedTokens(
            0.9802008654144229, 1.3361879297699635, 1.3425376954282364
        ),
        aa_sticker_price=PricedTokens(10.0, 50.0, 1.0),
        or_slug="openai/gpt-6-astra-20260903",
        or_session_cost_10_49_turns=2.8218648,
        or_toks_served=330995343765,
        or_eff_input_price=2.95,
        or_eff_output_price=50.96,
    ),
    Model(
        "OpenAI",
        "GPT-6 Astra (max)",
        56.8538559596797,
        ProviderType.DATACENTER,
        aa_tok_per_task=51973,
        aa_cost_per_task=PricedTokens(
            1.4712821987090436, 2.598659948821875, 2.1747247960716
        ),
        aa_sticker_price=PricedTokens(10.0, 50.0, 1.0),
        or_slug="openai/gpt-6-astra-20260903",
        or_session_cost_10_49_turns=2.8218648,
        or_toks_served=330995343765,
        or_eff_input_price=2.95,
        or_eff_output_price=50.96,
    ),
    # Released 2026-11-09: OR's trailing week is a single day so far, and OR has
    # no session-cost or weekly-volume history for the permaslug yet. All five
    # effort levels share the one permaslug, so they share one pair of effective
    # prices; each burns a different token mix.
    Model(
        "Anthropic",
        "Claude Fable 5.2 (low)",
        53.9249388799182,
        ProviderType.DATACENTER,
        aa_tok_per_task=35860,
        aa_cost_per_task=PricedTokens(
            1.040665157646971, 1.4240919609594807, 0.6324847856680946
        ),
        aa_sticker_price=PricedTokens(8.0, 40.0, 0.2),
        or_slug="anthropic/claude-fable-5.2-20261109",
        or_eff_input_price=1.404,
        or_eff_output_price=40.32,
    ),
    Model(
        "Anthropic",
        "Claude Fable 5.2 (medium)",
        56.4083157101519,
        ProviderType.DATACENTER,
        aa_tok_per_task=43131,
        aa_cost_per_task=PricedTokens(
            1.228047020697479, 1.7102304491697624, 0.8643275943115216
        ),
        aa_sticker_price=PricedTokens(8.0, 40.0, 0.2),
        or_slug="anthropic/claude-fable-5.2-20261109",
        or_eff_input_price=1.404,
        or_eff_output_price=40.32,
    ),
    Model(
        "Anthropic",
        "Claude Fable 5.2 (high)",
        61.8218785440284,
        ProviderType.DATACENTER,
        aa_tok_per_task=97841,
        aa_cost_per_task=PricedTokens(
            2.59311518192607, 3.8883159624114056, 2.596983633522906
        ),
        aa_sticker_price=PricedTokens(8.0, 40.0, 0.2),
        or_slug="anthropic/claude-fable-5.2-20261109",
        or_eff_input_price=1.404,
        or_eff_output_price=40.32,
    ),
    Model(
        "Anthropic",
        "Claude Fable 5.2 (xhigh)",
        62.2167392403357,
        ProviderType.DATACENTER,
        aa_tok_per_task=161944,
        aa_cost_per_task=PricedTokens(
            4.014762523548346, 6.430695367722222, 4.383129741240549
        ),
        aa_sticker_price=PricedTokens(8.0, 40.0, 0.2),
        or_slug="anthropic/claude-fable-5.2-20261109",
        or_eff_input_price=1.404,
        or_eff_output_price=40.32,
    ),
    Model(
        "Anthropic",
        "Claude Fable 5.2 (max)",
        64.563595756626,
        ProviderType.DATACENTER,
        aa_tok_per_task=273776,
        aa_cost_per_task=PricedTokens(
            7.121424801990003, 10.891777372657142, 8.52535562933377
        ),
        aa_sticker_price=PricedTokens(8.0, 40.0, 0.2),
        or_slug="anthropic/claude-fable-5.2-20261109",
        or_eff_input_price=1.404,
        or_eff_output_price=40.32,
    ),
    Model(
        "Anthropic",
        "Claude Sonnet 5.5 (low)",
        25.649704549789,
        ProviderType.DATACENTER,
        aa_tok_per_task=20856,
        aa_cost_per_task=PricedTokens(
            0.15495829818267776, 0.20856371779750618, 0.14782043040956583
        ),
        aa_sticker_price=PricedTokens(2.0, 10.0, 0.1),
        or_slug="anthropic/claude-sonnet-5.5-20260928",
        or_session_cost_10_49_turns=0.6057009799999999,
        or_toks_served=2373782022158,
        or_eff_input_price=0.5537,
        or_eff_output_price=10.04,
    ),
    Model(
        "Anthropic",
        "Claude Sonnet 5.5 (medium)",
        30.8195522148431,
        ProviderType.DATACENTER,
        aa_tok_per_task=30746,
        aa_cost_per_task=PricedTokens(
            0.2229816006675009, 0.3074589477415119, 0.25021548679343536
        ),
        aa_sticker_price=PricedTokens(2.0, 10.0, 0.1),
        or_slug="anthropic/claude-sonnet-5.5-20260928",
        or_session_cost_10_49_turns=0.6057009799999999,
        or_toks_served=2373782022158,
        or_eff_input_price=0.5537,
        or_eff_output_price=10.04,
    ),
    Model(
        "Anthropic",
        "Claude Sonnet 5.5 (high)",
        41.515109087048,
        ProviderType.DATACENTER,
        aa_tok_per_task=58421,
        aa_cost_per_task=PricedTokens(
            0.42919145427046057, 0.5842121201361333, 0.6227258655718849
        ),
        aa_sticker_price=PricedTokens(2.0, 10.0, 0.1),
        or_slug="anthropic/claude-sonnet-5.5-20260928",
        or_session_cost_10_49_turns=0.6057009799999999,
        or_toks_served=2373782022158,
        or_eff_input_price=0.5537,
        or_eff_output_price=10.04,
    ),
    Model(
        "Anthropic",
        "Claude Sonnet 5.5 (xhigh)",
        49.9240327787242,
        ProviderType.DATACENTER,
        aa_tok_per_task=122205,
        aa_cost_per_task=PricedTokens(
            0.9742263747522919, 1.2220489885977281, 1.895709552971899
        ),
        aa_sticker_price=PricedTokens(2.0, 10.0, 0.1),
        or_slug="anthropic/claude-sonnet-5.5-20260928",
        or_session_cost_10_49_turns=0.6057009799999999,
        or_toks_served=2373782022158,
        or_eff_input_price=0.5537,
        or_eff_output_price=10.04,
    ),
    Model(
        "Anthropic",
        "Claude Sonnet 5.5 (max)",
        54.54133407109,
        ProviderType.DATACENTER,
        aa_tok_per_task=285387,
        aa_cost_per_task=PricedTokens(
            2.2877209769653377, 2.853871133504894, 4.879029232925522
        ),
        aa_sticker_price=PricedTokens(2.0, 10.0, 0.1),
        or_slug="anthropic/claude-sonnet-5.5-20260928",
        or_session_cost_10_49_turns=0.6057009799999999,
        or_toks_served=2373782022158,
        or_eff_input_price=0.5537,
        or_eff_output_price=10.04,
    ),
    Model(
        "Anthropic",
        "Claude Opus 5.5 (low)",
        34.752785007759,
        ProviderType.DATACENTER,
        aa_tok_per_task=19060,
        aa_cost_per_task=PricedTokens(
            0.26830032873330867, 0.38125266891205145, 0.3259763248005245
        ),
        aa_sticker_price=PricedTokens(4.0, 20.0, 0.2),
        or_slug="anthropic/claude-opus-5.5-20260921",
        or_session_cost_10_49_turns=1.663448975,
        or_toks_served=6775387815132,
        or_eff_input_price=0.7688,
        or_eff_output_price=20.41,
    ),
    Model(
        "Anthropic",
        "Claude Opus 5.5 (medium)",
        48.2081024960691,
        ProviderType.DATACENTER,
        aa_tok_per_task=43817,
        aa_cost_per_task=PricedTokens(
            0.5747918883798223, 0.8764203544840845, 0.9998771947972656
        ),
        aa_sticker_price=PricedTokens(4.0, 20.0, 0.2),
        or_slug="anthropic/claude-opus-5.5-20260921",
        or_session_cost_10_49_turns=1.663448975,
        or_toks_served=6775387815132,
        or_eff_input_price=0.7688,
        or_eff_output_price=20.41,
    ),
    Model(
        "Anthropic",
        "Claude Opus 5.5 (high)",
        51.8359958227294,
        ProviderType.DATACENTER,
        aa_tok_per_task=60071,
        aa_cost_per_task=PricedTokens(
            0.751725695304029, 1.201514087031893, 1.5114224997613959
        ),
        aa_sticker_price=PricedTokens(4.0, 20.0, 0.2),
        or_slug="anthropic/claude-opus-5.5-20260921",
        or_session_cost_10_49_turns=1.663448975,
        or_toks_served=6775387815132,
        or_eff_input_price=0.7688,
        or_eff_output_price=20.41,
    ),
    Model(
        "Anthropic",
        "Claude Opus 5.5 (xhigh)",
        57.4447060170507,
        ProviderType.DATACENTER,
        aa_tok_per_task=103395,
        aa_cost_per_task=PricedTokens(
            1.1864324939748008, 2.0680263131828958, 2.7927356836910775
        ),
        aa_sticker_price=PricedTokens(4.0, 20.0, 0.2),
        or_slug="anthropic/claude-opus-5.5-20260921",
        or_session_cost_10_49_turns=1.663448975,
        or_toks_served=6775387815132,
        or_eff_input_price=0.7688,
        or_eff_output_price=20.41,
    ),
    Model(
        "Anthropic",
        "Claude Opus 5.5 (max)",
        59.1060336078904,
        ProviderType.DATACENTER,
        aa_tok_per_task=173313,
        aa_cost_per_task=PricedTokens(
            1.8315729353173493, 3.4664583713189643, 4.738378412876434
        ),
        aa_sticker_price=PricedTokens(4.0, 20.0, 0.2),
        or_slug="anthropic/claude-opus-5.5-20260921",
        or_session_cost_10_49_turns=1.663448975,
        or_toks_served=6775387815132,
        or_eff_input_price=0.7688,
        or_eff_output_price=20.41,
    ),
    Model(
        "Anthropic",
        "Claude Haiku 5.5 (low)",
        18.6180055310996,
        ProviderType.DATACENTER,
        aa_tok_per_task=28161,
        aa_cost_per_task=PricedTokens(
            0.009711032547690896, 0.014080275004595064, 0.01996637882970253
        ),
        aa_sticker_price=PricedTokens(0.1, 0.5, 0.01),
        or_slug="anthropic/claude-haiku-5.5-20261007",
        or_session_cost_10_49_turns=0.06387004574999999,
        or_eff_input_price=0.08047,
        or_eff_output_price=0.5849,
        or_toks_served=2684933075559,
    ),
    Model(
        "Anthropic",
        "Claude Haiku 5.5 (medium)",
        24.6013254343023,
        ProviderType.DATACENTER,
        aa_tok_per_task=53335,
        aa_cost_per_task=PricedTokens(
            0.017755535648584567, 0.026667558259885586, 0.04734146449820697
        ),
        aa_sticker_price=PricedTokens(0.1, 0.5, 0.01),
        or_slug="anthropic/claude-haiku-5.5-20261007",
        or_session_cost_10_49_turns=0.06387004574999999,
        or_eff_input_price=0.08047,
        or_eff_output_price=0.5849,
        or_toks_served=2684933075559,
    ),
    Model(
        "Anthropic",
        "Claude Haiku 5.5 (high)",
        29.874315467743,
        ProviderType.DATACENTER,
        aa_tok_per_task=88064,
        aa_cost_per_task=PricedTokens(
            0.028172450783792408, 0.04403217390244535, 0.08635569488055722
        ),
        aa_sticker_price=PricedTokens(0.1, 0.5, 0.01),
        or_slug="anthropic/claude-haiku-5.5-20261007",
        or_session_cost_10_49_turns=0.06387004574999999,
        or_eff_input_price=0.08047,
        or_eff_output_price=0.5849,
        or_toks_served=2684933075559,
    ),
    Model(
        "Anthropic",
        "Claude Haiku 5.5 (xhigh)",
        34.6387759486628,
        ProviderType.DATACENTER,
        aa_tok_per_task=136177,
        aa_cost_per_task=PricedTokens(
            0.04176397260319096, 0.06808849536432662, 0.1385086332984472
        ),
        aa_sticker_price=PricedTokens(0.1, 0.5, 0.01),
        or_slug="anthropic/claude-haiku-5.5-20261007",
        or_session_cost_10_49_turns=0.06387004574999999,
        or_eff_input_price=0.08047,
        or_eff_output_price=0.5849,
        or_toks_served=2684933075559,
    ),
    Model(
        "Anthropic",
        "Claude Haiku 5.5 (max)",
        36.4845435880142,
        ProviderType.DATACENTER,
        aa_tok_per_task=225966,
        aa_cost_per_task=PricedTokens(
            0.06550207604700437, 0.11298297985798467, 0.23252235946317396
        ),
        aa_sticker_price=PricedTokens(0.1, 0.5, 0.01),
        or_slug="anthropic/claude-haiku-5.5-20261007",
        or_session_cost_10_49_turns=0.06387004574999999,
        or_eff_input_price=0.08047,
        or_eff_output_price=0.5849,
        or_toks_served=2684933075559,
    ),
    # AA dropped the pre-1019 Ling 3.1 Flash record in its 2026-10 methodology
    # change, so its index is regressed from benchmarks (see README), using only
    # the nine benchmarks it shares with its successor -- the 1019 record AA
    # kept, which the same fit puts at 29.9 against its published 29.5. The old
    # revision is a little weaker than the 1019 on seven of those nine, so this
    # must come out below it, and does.
    Model(
        "InclusionAI",
        "Ling 3.1 Flash",
        28.41,
        ProviderType.DATACENTER,
        aa_tok_per_task=158551,
        aa_cost_per_task=PricedTokens(
            0.5426813344872892, 0.1426962308047775, 0.8686886280125741
        ),
        aa_sticker_price=PricedTokens(0.3, 0.9, 0.06),
        or_slug="inclusionai/ling-3.1-flash-20261002",
        or_session_cost_10_49_turns=0.0342785,
        or_toks_served=50780267592,
        or_eff_input_price=0.03414,
        or_eff_output_price=0.2918,
        estimated=True,
    ),
    Model(
        "InclusionAI",
        "Ling 3.1 Flash 1019",
        29.570119369837,
        ProviderType.DATACENTER,
        aa_tok_per_task=224711,
        aa_cost_per_task=PricedTokens(
            1.1261695939526823, 0.20223972528697784, 1.8567356990981327
        ),
        aa_sticker_price=PricedTokens(0.3, 0.9, 0.06),
        # 256 GiB unified: 2x DGX Spark
        hardware=DGX_SPARK.cluster(2, extra_price=100),
        # OpenRouter still only lists the older checkpoint
        available_datacenter=False,
    ),
    Model(
        "Mistral",
        "Mistral Large 4",
        26.1249946754784,
        ProviderType.DATACENTER,
        aa_tok_per_task=81951,
        aa_cost_per_task=PricedTokens(
            1.1078363678887329, 0.172096563210152, 0.7817976957138006
        ),
        aa_sticker_price=PricedTokens(0.7, 2.1, 0.07),
        or_slug="mistralai/mistral-large-4-0-20261006",
        or_session_cost_10_49_turns=0.28279102,
        or_toks_served=130526678338,
        or_eff_input_price=0.1608,
        or_eff_output_price=2.12,
        hardware=DGX_SPARK.cluster(3, extra_price=300),  # UD-Q2_K_XL
    ),
]


class PlotSpec(NamedTuple):
    """One filtered view.

    x_of(m) is the x coordinate, or None for models the view cannot place.
    band_side is the plot edge that
    coincides with the green band's edge ("bottom" for the high-intelligence
    plot, "top" for the low-cost one, None otherwise); band=False drops the
    green band entirely, frontier=False drops the Pareto frontier, and
    zero_line=True draws a solid vertical line at x=0 (the delta plot's AA
    baseline), for views whose x axis is not a price. x_min caps the lower
    bound of a signed x axis (a delta can never be below -100%). bar=True
    renders horizontal bars ordered by intelligence instead of a scatter.
    """

    title: str
    filt: Callable[[Model], bool]
    stem: str
    x_of: Callable[[Model], float | None] = Model.price_per_task
    x_label: str = "Price per Task (USD)"
    xtick_step: float = 5
    xtick_format: str = "$%.0f"
    band_side: str | None = None
    band: bool = True
    frontier: bool = True
    zero_line: bool = False
    x_min: float = -math.inf  # hard lower bound on a signed x axis
    bar: bool = False  # draw horizontal bars (sorted by intelligence) not dots
    one_per_model: bool = False  # collapse a model's effort variants to one
    # bar (the rows OR carries no price for would all draw the same empty bar)
    x_break: float | None = None  # split the x axis here: points at or above
    # the break go into a second panel (see make_broken_axis_plot)
    fig_w: float = FIG_W  # SVG figure width in inches, when this view needs
    # a different one (the height is always FIG_H)
    png_fig_w: float | None = None  # PNG figure width in inches
    png_xtick_step: float | None = None  # PNG-only x tick step, for the same
    # reason: a step that spaces ticks out on a 1488in figure crams them into
    # an unreadable smear on a 26in one
    x_pad: float = 0.03  # right margin on an unsigned x axis, as a fraction
    # of the span. This is the SVG's margin: 0.03 is a few hundred pixels on
    # a 26in figure but 30in of white on a 1000in one, so a wide view tightens
    png_x_pad: float | None = None  # x_pad for the PNG render, when it needs
    # its own margin (a fraction that reads as inches on the wide SVG is a
    # handful of pixels on the 26in PNG, where it clips the rightmost dot)
    png_label_slots: int | None = None  # PNG-only: try just the first N entries
    # of CANDIDATES for every label, and drop a label outright when all N are
    # taken. No leader lines, no far queues: the screen-sized view stays
    # readable by leaving a crowded model unnamed instead of dragging its name
    # across the plot. Only applies to a spec drawn separately for the PNG
    # (see split_renders); a shared png+svg figure can't differ between them.
    easter_eggs: tuple[tuple[float, str], ...] = ()  # (x, slogan) shouts in
    # the background of the SVG only. Newlines are kept, to wrap a long
    # slogan instead of letting it run into its neighbour.
    available_of: Callable[[Model], bool] = lambda m: (
        m.available_datacenter
    )  # which availability flag this view is governed by: the strikethrough,
    # the legend entry and the frontier exclusion all follow it
    label_of: Callable[[Model], str] = lambda m: m.name  # a point's display
    # text. The hardware plot drops the effort/hardware suffixes, since its x
    # is the same rig for every effort variant of a model
    hidden_icons: frozenset[str] = frozenset()  # icon keys this view drops, from
    # its labels and from its legend. The hardware plot hides "bolt": its filter
    # admits only local models, so the ⚡ marks every point and therefore says
    # nothing
    right_xtick_step: float | None = None  # tick step for the x_break panel.
    # None gives it one tick per model, for when the prices are six figures
    # and a fixed step would only produce collisions
    break_xtick_format: str | None = None  # tick format for both panels of an
    # x_break view, in str.format style instead of %-style: only str.format can
    # group thousands, which the hardware axis needs for its six-figure prices
    panel_ratio: float | None = None  # left:right panel width ratio when
    # x_break is set. None uses RIGHT_PANEL_RATIO
    legend_panel: str = "left"  # which panel of an x_break view draws the
    # legend: the one whose lower-right corner covers no model's dot

    def split_renders(self) -> bool:
        return (
            self.png_fig_w is not None
            or self.png_xtick_step is not None
            or self.png_x_pad is not None
            or bool(self.easter_eggs)
        )

    def get_fig_w(self, fmt: Literal["png", "svg", "both"]) -> float:
        """fig_w for this render: the SVG can be much larger"""
        assert fmt in ("png", "svg", "both")
        if fmt == "both":
            assert self.png_fig_w is None
        elif fmt == "png" and self.png_fig_w is not None:
            return self.png_fig_w
        return self.fig_w

    def get_xtick_step(self, fmt: Literal["png", "svg", "both"]) -> float:
        """x tick step for this render: the PNG gets its own coarser step when the
        view has one (see png_xtick_step)."""
        assert fmt in ("png", "svg", "both")
        if fmt == "both":
            assert self.png_xtick_step is None
        elif fmt == "png" and self.png_xtick_step is not None:
            return self.png_xtick_step
        return self.xtick_step

    def get_x_pad(self, fmt: Literal["png", "svg", "both"]) -> float:
        """Right margin on the x axis for this render: the PNG gets its own when
        the view has one (see png_x_pad)."""
        assert fmt in ("png", "svg", "both")
        if fmt == "both":
            assert self.png_x_pad is None
        elif fmt == "png" and self.png_x_pad is not None:
            return self.png_x_pad
        return self.x_pad


# The plots to generate.
PLOTS = [
    PlotSpec(
        "Intelligence vs. Price per Task (High Intelligence)",
        lambda m: m.intelligence >= HIGH_INTELLIGENCE_THRESHOLD,
        "high_intelligence",
        xtick_step=1.0,
        xtick_format="$%.2f",
        band_side="bottom",
        x_break=15.00,
        right_xtick_step=10.0,
        # Its left panel ends at its priciest model, so the corner the legend
        # wants is that model's dot: the legend goes to the right panel, whose
        # bottom half is empty.
        legend_panel="right",
    ),
    PlotSpec(
        "Intelligence vs. Price per Task (Low Cost)",
        lambda m: m.price_per_task() <= LOW_COST_THRESHOLD,
        "low_cost",
        xtick_step=0.05,
        xtick_format="$%.2f",
        band_side="top",
    ),
    PlotSpec(
        "Intelligence vs. Price per Task (All Models)",
        lambda _: True,
        "all_models",
        xtick_step=0.05,
        xtick_format="$%.2f",
        fig_w=2200.0,
        png_fig_w=FIG_W,
        png_xtick_step=2.0,
        png_label_slots=4,
        x_pad=0.002,  # ~$0.14 past Claude Fable 5.2 (max); the PNG needs the
        # default 0.03, where 0.002 would cut that dot in half
        png_x_pad=0.03,
        easter_eggs=(
            (0.45, "Too cheap to care"),
            (1.90, "Need to be careful\nabout your bill"),
            (4.00, "This is EXPENSIVE"),
            (
                6.00,
                "Anything beyond this point\nis unaffordable without\nVC subsidies",
            ),
            (10.00, "Keep scrolling"),
            (16.00, "This is getting ridiculous"),
            (34.00, "Are you still here?"),
            (52.00, "Almost there..."),
            (71.80, "Congratulations,\nyou made it to the end!"),
        ),
    ),
    PlotSpec(
        "Δ from AA's Price per Task",
        lambda m: m.provider_type is ProviderType.DATACENTER,
        "price_delta",
        x_of=Model.price_delta,
        x_label="Δ from AA's price per task (%)",
        xtick_step=20,
        xtick_format="%.0f%%",
        band=False,
        frontier=False,
        zero_line=True,
        x_min=-100,
        bar=True,
        one_per_model=True,
    ),
    # Larger local models: x is the one-off hardware bill, not a per-task
    # price. Kimi K3.1 Pro needs 1 TB of unified memory ($64,600, twice the next
    # most expensive rig), so x_break splits the axis and gives it a panel of its
    # own instead of squashing the other sixteen models into the leftmost 8%.
    PlotSpec(
        "Intelligence vs. Hardware price",
        lambda m: m.hardware_price() is not None,
        "local_hardware",
        x_of=Model.hardware_price,
        x_label="Hardware price, one-off (USD)",
        xtick_step=2_000,
        band=False,
        x_break=35_000,
        break_xtick_format="${x:,.0f}",
        available_of=lambda m: m.available_local,
        label_of=lambda m: _clean_label(m),
        hidden_icons=frozenset({"bolt"}),
    ),
]


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


# Reasoning-effort suffixes AA appends to a model name when one permaslug is
# benchmarked at several thinking levels. Not every parenthetical is an effort
# tag -- "Qwen3.8 Max (0902)" carries a release date -- so only these words get
# dropped when a label is stripped.
EFFORT_TAGS = frozenset({"minimal", "low", "medium", "high", "xhigh", "max"})


def _clean_label(m: Model) -> str:
    """Model name without its reasoning-effort or hardware tag.

    "GLM-5.3-Flash (max)" -> "GLM-5.3-Flash", and "Qwen3.8-Flash (Strix Halo ⚡)" ->
    "Qwen3.8-Flash ⚡": the effort suffix and the hardware name Model.__post_init__
    splices in are dropped, but the ⚡ itself is kept so the local-electricity marker
    still renders (the plots have a legend entry for it). A parenthetical that is part
    of the model's real name -- a release date, or any word outside EFFORT_TAGS --
    stays.
    """
    name, bolt = m.name, ""
    if "⚡" in name:
        head, _, tail = name.partition("⚡")
        name = head.rstrip() + tail.lstrip()  # rejoin around the bolt
        bolt = " ⚡"
        if m.hardware is not None:
            tag = f" ({m.hardware.name})"
            name = name.removesuffix(tag)
    while name.endswith(")"):
        open_ = name.rfind("(")
        if open_ < 0:
            break
        if name[open_ + 1 : -1].strip().lower() not in EFFORT_TAGS:
            break
        name = name[:open_].rstrip()
    return name + bolt


def _split_icon(m, name=None, available=True, hidden=()):
    """Split the display glyphs out of a model's label.

    Returns (left, icon, right, strike): the text before the glyph, the icon
    key (None, "bolt", or "mask"), the text after it, and whether the label is
    unavailable (rendered dark grey with a strikethrough, no icon). The icon is
    drawn between the two text halves, so e.g. "(RTX 3090 ⚡)" keeps its
    parentheses. Only the ⚡ still lives in the name (Model.__post_init__ splices
    it in); the thief mask comes from Model.trains_on_your_data and the
    strikethrough from an availability flag (whichever of available_datacenter
    or available_local the calling view is governed by), so no marker text is
    carried in the data.

    `name` overrides the display text (e.g. with an effort suffix removed).
    `available` overrides the availability the label is rendered under.
    `hidden` lists icon keys this view suppresses: they are neither returned
    nor left in the text, so a hidden ⚡ leaves the name it was spliced into
    unchanged apart from the glyph itself.
    """
    if name is None:
        name = m.name
    strike = not available
    idx = name.find("⚡")
    if idx >= 0:
        left = name[:idx].rstrip()
        right = name[idx + len("⚡") :].lstrip()
        if "bolt" in hidden:
            return (left + " " + right).strip(), None, "", strike
        return left, "bolt", right, strike
    if m.trains_on_your_data and "mask" not in hidden:
        return name, "mask", "", strike
    return name, None, "", strike


def _shift(bb, dy):
    """`bb` (x0, y0, x1, y1) moved vertically by dy."""
    return (bb[0], bb[1] + dy, bb[2], bb[3] + dy)


def _pad(bb, pad=PAD_PX):
    if hasattr(bb, "x0"):  # Bbox
        bb = (bb.x0, bb.y0, bb.x1, bb.y1)
    return (bb[0] - pad, bb[1] - pad, bb[2] + pad, bb[3] + pad)


def _overlap_area(a, b):
    dx = min(a[2], b[2]) - max(a[0], b[0])
    dy = min(a[3], b[3]) - max(a[1], b[1])
    return dx * dy if dx > 0 and dy > 0 else 0.0


def _point_segment_dist(p, seg):
    """Distance from point p to the closed segment seg, in display pixels."""
    x0, y0, x1, y1 = seg
    dx, dy = x1 - x0, y1 - y0
    if dx == 0.0 and dy == 0.0:
        return math.hypot(p[0] - x0, p[1] - y0)
    t = ((p[0] - x0) * dx + (p[1] - y0) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    return math.hypot(p[0] - (x0 + t * dx), p[1] - (y0 + t * dy))


def _segments_cross(a, b):
    """Whether closed line segments a and b intersect or overlap.

    Leader lines sharing space make dense clusters hard to read, so even an
    endpoint touch or a collinear overlap counts. Coordinates are display
    pixels.
    """

    def orient(p, q, r):
        val = (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
        scale = max(1.0, abs(q[0] - p[0]), abs(q[1] - p[1]))
        if abs(val) <= 1e-9 * scale**2:
            return 0
        return 1 if val > 0 else -1

    def on_segment(p, q, r):
        return (
            min(p[0], r[0]) - 1e-9 <= q[0] <= max(p[0], r[0]) + 1e-9
            and min(p[1], r[1]) - 1e-9 <= q[1] <= max(p[1], r[1]) + 1e-9
        )

    p, q = (a[0], a[1]), (a[2], a[3])
    r, s = (b[0], b[1]), (b[2], b[3])
    o1, o2 = orient(p, q, r), orient(p, q, s)
    o3, o4 = orient(r, s, p), orient(r, s, q)
    if o1 != o2 and o3 != o4:
        return True
    return (
        (o1 == 0 and on_segment(p, r, q))
        or (o2 == 0 and on_segment(p, s, q))
        or (o3 == 0 and on_segment(r, p, s))
        or (o4 == 0 and on_segment(r, q, s))
    )


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


def place_labels(
    ax,
    fig,
    points,
    marker_r_px,
    extra_obstacles=(),
    max_candidates=None,
    drop_unplaceable=False,
):
    """points: [(left, right, icon, strike, x, y)] in data coords -- icon is
    None, "bolt", or "mask"; the icon is drawn as a colored vector marker
    between the left and right text halves, so e.g. "(RTX 3090 ⚡)" keeps its
    parens. strike=True (the view's availability flag is False) renders the
    text dark grey with a strikethrough and no icon. Adds annotations,
    auto-placed.

    max_candidates: use only the first N entries of CANDIDATES, skipping the
    crowd-specific reordering. drop_unplaceable: with such a cap, a label with
    no collision-free slot is omitted entirely (no text, no icon, no leader,
    and no obstacle for the others) rather than placed on top of something.

    Placement runs in rounds: a greedy sequential pass, then repair rounds in
    which every label re-chooses its spot around everyone else's position, so
    a crowded plot doesn't end up with cascading overlaps.

    extra_obstacles: additional (x0, y0, x1, y1) display-pixel boxes that
    labels must not overlap (e.g. the legend).
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes_box = _pad(ax.get_window_extent(renderer), 0)
    if os.environ.get("LBLDBG"):
        print(f"=== PLACE {ax.get_title(loc='left') or ax.get_title()} n={len(points)}")

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
    dot_y = [py for _, py in disp]
    dot_x = [px for px, _ in disp]

    def misreads(i, j, box_i, box_j, vc_i, vc_j):
        """True if label i and label j stack in the opposite order to their
        dots: the plot then says "the upper label is the upper dot's" and means
        the opposite, which is how a reader ends up with the wrong model.

        The scope matters. A global "every label must stack in its dot's
        order" rule is unsatisfiable wherever dots sit closer together than a
        label is tall (the high-intelligence plot packs nine models into two
        index points) -- enforcing it there just piles the labels on top of
        each other. So only pairs the eye actually reads as a column count:
        dots within ORDER_TIE_PX of each other vertically are a tie (nobody
        can tell which is higher, so either stacking is honest), dots further
        than ORDER_DOT_DX apart horizontally are paired with their label by
        proximity, and label boxes more than ORDER_X_PX apart horizontally are
        read as separate entries, not as a stack."""
        dy = dot_y[i] - dot_y[j]
        if abs(dy) < ORDER_TIE_PX:
            return False
        # A label centred above or below its dot is anchored by x-alignment; a
        # label beside its dot is anchored by adjacency. A pair with one of each
        # is read through those anchors, not as a stack, so their vertical order
        # says nothing and must not be forced.
        if (box_i[0] <= dot_x[i] <= box_i[2]) != (box_j[0] <= dot_x[j] <= box_j[2]):
            return False
        if abs(dot_x[i] - dot_x[j]) > ORDER_DOT_DX:
            return False
        if max(box_i[0] - box_j[2], box_j[0] - box_i[2]) > ORDER_X_PX:
            return False
        return dy * (vc_i - vc_j) < 0

    def order_violations(state):
        """[(higher dot, lower dot, px)] of the label pairs in `state` that
        misread, in this plot's scope (see misreads)."""
        live = [i for i, s in enumerate(state) if s is not None]
        out = []
        for a, i in enumerate(live):
            for j in live[a + 1 :]:
                if misreads(i, j, state[i][1], state[j][1], state[i][3], state[j][3]):
                    up, lo = (i, j) if dot_y[i] > dot_y[j] else (j, i)
                    out.append((up, lo, abs(state[i][3] - state[j][3])))
        return out

    def crowding(i):
        xi, yi = disp[i]
        return sum(
            1
            for j, (xj, yj) in enumerate(disp)
            if j != i and abs(xi - xj) < CROWD_X and abs(yi - yj) < CROWD_Y
        )

    crowd = [crowding(i) for i in range(len(points))]
    order = sorted(range(len(points)), key=lambda i: (-crowd[i], -points[i][5]))

    def leader_for(i, box):
        """Leader segment from dot i to the nearest edge of `box` (display px),
        or None when the label sits close enough for the dot to speak for
        itself."""
        px, py = disp[i]
        anchor_x = max(box[0], min(px, box[2]))
        anchor_y = max(box[1], min(py, box[3]))
        dist = math.hypot(anchor_x - px, anchor_y - py)
        if not crowd[i] and dist <= marker_r_px + LEADER_MIN:
            return None
        if dist <= 1e-6:
            return (px, py, anchor_x, anchor_y)
        ux, uy = (anchor_x - px) / dist, (anchor_y - py) / dist
        return (px + ux * marker_r_px, py + uy * marker_r_px, anchor_x, anchor_y)

    def leader_dot_hits(i, leader):
        """How many other models' dots the leader line of label i runs too close
        to. A leader grazing a marker reads as if it pointed at that dot -- the
        same misreading the order rule exists to prevent -- so a label has to
        sit far enough above or below its own dot for the line to clear its
        neighbours."""
        if leader is None:
            return 0
        clear = marker_r_px + LEADER_CLEAR_PX
        return sum(
            1
            for j, (px, py) in enumerate(disp)
            if j != i and _point_segment_dist((px, py), leader) < clear
        )

    def clean_at(i, vc, state):
        """True if label i, slid vertically to vc (display px), is as clean as
        the repair rounds demand: inside the axes, off every dot and every
        other label, clear of the legend, not crossing a leader line, and not
        contradicting the dots' height order."""
        box = _shift(state[i][1], vc - state[i][3])
        raw = _pad(box, -PAD_PX)
        if (
            raw[0] < axes_box[0]
            or raw[2] > axes_box[2]
            or raw[1] < axes_box[1]
            or raw[3] > axes_box[3]
        ):
            return False
        if any(_overlap_area(box, o) for o in dot_boxes):
            return False
        if any(_overlap_area(box, o) for o in extra_obstacles):
            return False
        leader = leader_for(i, box)
        for j in range(len(points)):
            if j == i or state[j] is None:
                continue
            if _overlap_area(box, state[j][1]) > 4 * PAD_PX * PAD_PX:
                return False
            if misreads(i, j, box, state[j][1], vc, state[j][3]):
                return False
            if (
                leader is not None
                and state[j][7] is not None
                and _segments_cross(leader, state[j][7])
            ):
                return False
        return not leader_dot_hits(i, leader)

    def choose(i, obstacles, leaders, others):
        """Pick the best candidate position for label i. Returns
        (score, bbox, left_x0, vc, icon_cx, right_x0, w_right, leader); the
        bbox, positions and leader segment are display pixels. others is
        [(j, vc_j, bbox_j)]: the current box and vertical centre of every other
        placed label, for the order rule.
        """
        left, right, icon, _strike, x, y = points[i]
        best = None
        # In a cluster, a label touching its dot is ambiguous no matter what, so
        # only consider the slots that clear it: the horizontal ones, and the
        # mid-height ones that buy a little daylight without running off to a
        # leader-length-away stack. Right of the point is always tried first,
        # then left; the far slots are the fallback once those collide.
        cands = CANDIDATES
        if max_candidates is not None:
            cands = CANDIDATES[:max_candidates]
        elif crowd[i]:
            near = [
                c
                for c in CANDIDATES
                if abs(c[1]) < CROWD_OFFSET
                and (c[3] == "center" or abs(c[1]) >= CROWD_NEAR)
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
            # then how much of it, then leader-line crossings, then overlap
            # with the legend/other labels. Text wins over leader lines only
            # after crossings are exhausted: dotted leaders crossing are
            # readable, overlapping text is not.
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

            leader = leader_for(i, bb)
            lead_dot = leader_dot_hits(i, leader)
            crossings = sum(
                1
                for other in leaders
                if leader is not None and _segments_cross(leader, other)
            )
            # The one mistake the plot cannot recover from is a label that
            # reads as another model's: a stack that contradicts the dots'
            # order, or a leader that grazes a marker. Rank both above
            # leader-line crossings, which only look busy. (Label overlap,
            # last, is the repair rounds' problem -- they drive it to zero.)
            order_cnt, order_px = 0, 0.0
            for j, vj, bj in others:
                if misreads(i, j, bb, bj, vc, vj):
                    order_cnt += 1
                    order_px += abs(vc - vj)
            score = (
                spill,
                dot_pen > 0,
                dot_pen,
                order_cnt > 0,
                order_cnt,
                order_px,
                lead_dot > 0,
                lead_dot,
                crossings > 0,
                crossings,
                pen,
            )
            _dbg = os.environ.get("LBLDBG", "")
            if _dbg and _dbg in left:
                print(
                    f"DBG {left[:30]!r} dx={dx} dy={dy} {ha}/{va} "
                    f"spill={int(spill)} dot={dot_pen:.0f} pen={pen:.0f} "
                    f"cross={crossings} order={order_cnt}/{order_px:.0f} "
                    f"leaddot={lead_dot}"
                )

            result = (
                score,
                bb,
                left_x0,
                vc,
                icon_cx,
                right_x0,
                w_right,
                leader,
            )
            if not any(score):  # nothing to fault: take it
                return result
            if best is None or score < best[0]:
                best = result
        # Capped search: nothing in the allowed slots is collision-free, so the
        # label is either dropped (drop_unplaceable) or placed at the least bad
        # spot.
        return None if drop_unplaceable else best

    # Round 0: greedy sequential pass (already-placed labels and leader lines
    # are obstacles).
    placed = [None] * len(points)
    obstacles = list(extra_obstacles)
    leaders = []
    for i in order:
        seen = [
            (j, placed[j][3], placed[j][1])
            for j in range(len(points))
            if placed[j] is not None
        ]
        b = choose(i, obstacles, leaders, seen)
        placed[i] = b
        if b is None:  # dropped label: reserves no space
            continue
        obstacles.append(b[1])
        if b[7] is not None:
            leaders.append(b[7])

    def _total_cost(state):
        """(order violations, their total px, leader crossings, unpadded
        label-overlap area) of a full layout, in the same priority order as
        choose()'s score."""
        area = 0.0
        crosses = 0
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                if state[i] is None or state[j] is None:
                    continue
                a = tuple(v + PAD_PX * s_ for v, s_ in zip(state[i][1], (1, 1, -1, -1)))
                b2 = tuple(
                    v + PAD_PX * s_ for v, s_ in zip(state[j][1], (1, 1, -1, -1))
                )
                area += _overlap_area(a, b2)
                if (
                    state[i][7] is not None
                    and state[j][7] is not None
                    and _segments_cross(state[i][7], state[j][7])
                ):
                    crosses += 1
        violations = order_violations(state)
        return (len(violations), sum(v[2] for v in violations), crosses, area)

    best_state = [b for b in placed]
    best_cost = _total_cost(best_state)

    # Repair rounds: re-place every label, in crowd order, against the latest
    # positions of the others (Gauss-Seidel style). Updating every label
    # against a stale snapshot instead can end with two labels sitting on top
    # of each other, each having dodged where the other used to be. Iterate
    # until no label collides with anything, or give up after a fixed number
    # of rounds (dense clusters may be unsatisfiable).
    for _round in range(40):
        for i in order:
            others = [j for j in range(len(points)) if j != i]
            obs = list(extra_obstacles) + [
                placed[j][1] for j in others if placed[j] is not None
            ]
            other_leaders = [
                placed[j][7]
                for j in others
                if placed[j] is not None and placed[j][7] is not None
            ]
            seen = [
                (j, placed[j][3], placed[j][1]) for j in others if placed[j] is not None
            ]
            placed[i] = choose(i, obs, other_leaders, seen)
        label_collisions = any(
            _overlap_area(placed[i][1], placed[j][1]) > 4 * PAD_PX * PAD_PX
            for i in range(len(points))
            for j in range(i + 1, len(points))
            if placed[i] is not None and placed[j] is not None
        )
        leader_collisions = any(
            _segments_cross(placed[i][7], placed[j][7])
            for i in range(len(points))
            for j in range(i + 1, len(points))
            if placed[i] is not None
            and placed[j] is not None
            and placed[i][7] is not None
            and placed[j][7] is not None
        )
        order_bad = bool(order_violations(placed))
        lead_dot_bad = any(
            placed[i] is not None and leader_dot_hits(i, placed[i][7])
            for i in range(len(points))
        )
        if (
            not label_collisions
            and not leader_collisions
            and not order_bad
            and not lead_dot_bad
        ):
            break
        cost = _total_cost(placed)
        if cost < best_cost:
            best_cost = cost
            best_state = [b for b in placed]
    else:
        # Gave up without converging (dense cluster): keep the best state seen
        # across all rounds, not whatever the last oscillating round produced.
        if best_cost < _total_cost(placed):
            placed = best_state

    def settle_pass():
        """Slide every label to the clean height nearest its own dot.

        The repair rounds minimise penalties, not distance: a label takes the
        nearest slot that happens to be free at the time and then stays there
        while its neighbours move away, so a crowded column freezes with every
        label parked a slot above its dot -- even though sliding the bottom one
        down would let the next one take the vacated spot, and the one above
        that take its spot, walking the whole column back down toward its dots.
        Sweeping bottom-up (so the space a label vacates is available to the
        next one) performs exactly that shuffle. A move is only ever accepted
        into a spot the repair rounds would call clean, and only if it brings
        the label closer to its dot, so the pass cannot break the layout."""
        for _ in range(3):  # a few sweeps: one move can unblock another label
            moved = False
            for i in sorted(range(len(points)), key=lambda i: dot_y[i]):
                if placed[i] is None:
                    continue
                current = placed[i][3]
                want = dot_y[i]
                best = None
                for step in range(1, SETTLE_MAX_PX + 1):
                    for vc in (want + step, want - step):
                        if clean_at(i, vc, placed):
                            best = vc
                            break
                    if best is not None:
                        break
                if best is None or abs(best - want) >= abs(current - want):
                    continue
                box = _shift(placed[i][1], best - current)
                placed[i] = (
                    placed[i][0],
                    box,
                    placed[i][2],
                    best,
                    placed[i][4],
                    placed[i][5],
                    placed[i][6],
                    leader_for(i, box),
                )
                moved = True
            if not moved:
                break

    settle_pass()

    # Diagnostics toggle, set by --verbose in main().
    if VERBOSE:
        # Print pairwise overlaps between the *unpadded* label extents, in
        # display px. Padded boxes may legally overlap by up to PAD_PX on
        # each side; that is not a real collision.
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                title = ax.get_title(loc="left") or ax.get_title() or "?"
                if placed[i] is None or placed[j] is None:
                    continue
                a = tuple(
                    v + PAD_PX * s_ for v, s_ in zip(placed[i][1], (1, 1, -1, -1))
                )
                b = tuple(
                    v + PAD_PX * s_ for v, s_ in zip(placed[j][1], (1, 1, -1, -1))
                )
                ov = _overlap_area(a, b)
                if ov > 1:
                    print(
                        f"OVERLAP {ov:.0f}px^2 [{title}]: "
                        f"[{points[i][0][:30]}|{points[i][1][:20]}] vs "
                        f"[{points[j][0][:30]}|{points[j][1][:20]}]",
                    )
                if (
                    placed[i][7] is not None
                    and placed[j][7] is not None
                    and _segments_cross(placed[i][7], placed[j][7])
                ):
                    print(
                        f"LEADER CROSS [{title}]: "
                        f"[{points[i][0][:30]}|{points[i][1][:20]}] vs "
                        f"[{points[j][0][:30]}|{points[j][1][:20]}]"
                    )
        # Leader lines that graze a dot they do not belong to: the line then
        # looks like it points at that model. Dense clusters are not always
        # satisfiable, so this is a report, not an assertion.
        for i in range(len(points)):
            if placed[i] is None:
                continue
            hits = leader_dot_hits(i, placed[i][7])
            if hits:
                print(
                    f"LEADER/DOT [{title}]: "
                    f"[{points[i][0][:30]}] grazes {hits} other dot(s)"
                )
        # Label pairs that stack against their dots' order: a reader pairs the
        # upper label with the upper dot and reads the wrong model.
        for up, lo, _ in order_violations(placed):
            print(
                f"ORDER [{title}]: "
                f"[{points[up][0][:30]}] is above [{points[lo][0][:30]}], "
                f"but its label is not"
            )
    inv = ax.transData.inverted()
    for i in order:
        left, right, icon, strike, x, y = points[i]
        if placed[i] is None:  # dropped by the candidate cap: draw nothing
            continue
        _, bb, left_x0, vc, icon_cx, right_x0, _, leader = placed[i]
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
        if leader is not None:
            # Convert back to data coords: pixel-space artists don't survive the
            # SVG renderer's own coordinate space, data coords do.
            (x0, y0), (x1, y1) = inv.transform(
                [(leader[0], leader[1]), (leader[2], leader[3])]
            )
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


def _plot_legend(
    ax,
    models,
    loc="lower right",
    bbox_to_anchor=None,
    available_of=lambda m: True,
    hidden=(),
):
    """Legend for the publishers present in a plot, plus an entry for every
    special marker in use (⚡ local electricity, thief mask for
    trains_on_your_data, hollow dot for estimated points, strikethrough for
    a model unavailable under `available_of`). Icon keys in `hidden` get no
    entry, matching the labels a view suppresses them in.
    Returns (legend, has_unavailable_entry)."""
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
    if "bolt" not in hidden and any("⚡" in m.name for m in models):
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
    if "mask" not in hidden and any(m.trains_on_your_data for m in models):
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
    if any(m.estimated for m in models):
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markersize=DOT_SIZE**0.5,
                markerfacecolor="white",
                markeredgecolor="black",
                markeredgewidth=1.5,
                label="Estimated (not on AA)",
            )
        )
    have_unavailable = any(not available_of(m) for m in models)
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
        loc=loc,
        bbox_to_anchor=bbox_to_anchor,
        fontsize=LABEL_SIZE,
        framealpha=0.9,
        edgecolor="#d0d4da",
        handletextpad=0.6,
        borderpad=0.6,
    )
    return legend, have_unavailable


def pareto_frontier(models, x_of, available_of=lambda m: True):
    """The Pareto staircase: running max of intelligence over ascending x,
    as two parallel lists ready for ax.plot.

    Computed over ALL models, not one plot's filtered view. Each plot is a
    zoom of the same frontier; the axes clip the line, so on the zoomed plots
    it runs off the edge ("continues"), while on the all-models plot it ends at
    the frontier's true last step. Models flagged trains_on_your_data are excluded:
    they are the same offers at providers that train
    on your data, not separate models. Models that are not publicly available
    under `available_of` (the plot's availability flag) are excluded too: their
    cost is an estimate, not a real
    offer. Models the x statistic cannot place are skipped.
    """
    pts = sorted(
        (
            (x, m.intelligence)
            for m in models
            if not m.trains_on_your_data
            and available_of(m)
            and (x := x_of(m)) is not None
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
    return frontier_x, frontier_y


def _scatter_points(ax, models, x_of):
    """Draw the models as publisher-colored dots on ax. Estimated points
    (not on AA) are hollow (white fill) with the publisher-colored border,
    instead of solid."""
    solid = [i for i, m in enumerate(models) if not m.estimated]
    hollow = [i for i, m in enumerate(models) if m.estimated]
    ax.scatter(
        [x_of(models[i]) for i in solid],
        [models[i].intelligence for i in solid],
        s=DOT_SIZE,
        c=[PUBLISHERS[models[i].publisher] for i in solid],
        zorder=3,
    )
    if hollow:
        ax.scatter(
            [x_of(models[i]) for i in hollow],
            [models[i].intelligence for i in hollow],
            s=DOT_SIZE,
            facecolors="white",
            edgecolors=[PUBLISHERS[models[i].publisher] for i in hollow],
            linewidths=1.5,
            zorder=3,
        )


def _draw_easter_eggs(ax, spec: PlotSpec, y_mid) -> None:
    """Shout every slogan across the middle of the plot, centred on its own x
    milestone. zorder 1 keeps them over the grid and under the frontier, the
    dots and the labels, so no model is ever hidden by a joke."""
    for x, text in spec.easter_eggs:
        ax.text(
            x,
            y_mid,
            text,
            ha="center",
            va="center",
            fontsize=EGG_SIZE,
            fontweight="bold",
            color=EGG_COLOR,
            alpha=EGG_ALPHA,
            linespacing=1.15,
            zorder=1,
        )


def make_datacenter_plots(spec: PlotSpec, models, band, y_lim):
    """Draw one view, once per output format it needs.

    A spec with png_fig_w is drawn twice: at png_fig_w for the PNG (what
    GitHub, a README and every browser render inline, so it stays the
    screen-sized figure) and at fig_w for the SVG (vector, so the 57x wider
    view costs nothing but a few hundred KB and stays sharp when zoomed).
    Labels are placed per figure, so the narrow PNG is not a squashed copy of
    the wide one -- it is laid out from scratch at its own size.
    """
    if spec.split_renders():
        _make_datacenter_plot(spec, models, band, y_lim, "png")
        _make_datacenter_plot(spec, models, band, y_lim, "svg")
    else:
        _make_datacenter_plot(spec, models, band, y_lim, "both")


def _make_datacenter_plot(
    spec: PlotSpec, models, band, y_lim, fmt: Literal["png", "svg", "both"]
) -> None:
    """Draw the view and save it"""
    xs = [spec.x_of(m) for m in models]

    fig, ax = plt.subplots(figsize=(spec.get_fig_w(fmt), FIG_H), dpi=DPI)

    _scatter_points(ax, models, spec.x_of)

    # Faint dotted Pareto frontier. The delta plot's x axis is not a price,
    # so it gets no frontier (spec.frontier is False).
    frontier_x, frontier_y = (
        pareto_frontier(MODELS, spec.x_of, spec.available_of)
        if spec.frontier
        else ([], [])
    )
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
    x_lo, x_hi = min(xs), max(xs)
    if x_lo < 0:  # signed axis (delta plot): margin on both sides
        pad = 0.05 * (x_hi - x_lo)
        ax.set_xlim(max(x_lo - pad, spec.x_min), x_hi + pad)
    else:
        ax.set_xlim(0, x_hi * (1 + spec.get_x_pad(fmt)))
    # y_lim: floor of the points with a point of slack (enough for a label
    # centred on the lowest dot) / ceil with 0.2, except on the band
    # side, which snaps exactly to the band edge (computed in main).
    y_lo, y_hi = y_lim
    ax.set_ylim(y_lo, y_hi)
    if spec.easter_eggs:
        assert fmt != "both"
        if fmt == "svg":
            _draw_easter_eggs(ax, spec, (y_lo + y_hi) / 2)
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
    ax.xaxis.set_major_locator(MultipleLocator(spec.get_xtick_step(fmt)))
    ax.xaxis.set_major_formatter(FormatStrFormatter(spec.xtick_format))
    ax.yaxis.set_major_locator(MultipleLocator(1))

    if spec.zero_line:
        # AA's published price is the reference the delta axis is measured
        # from: mark it with a solid vertical line.
        ax.axvline(0, color="#9aa1ad", linewidth=1.2, zorder=1)

    ax.set_xlabel(spec.x_label, fontsize=15, fontweight="bold", labelpad=12)
    ax.set_ylabel(
        "Artificial Analysis Intelligence Index",
        fontsize=15,
        fontweight="bold",
        labelpad=12,
    )
    ax.set_title(
        spec.title,
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
    legend, have_unavailable = _plot_legend(ax, models, available_of=spec.available_of)

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
            (left, right, icon, strike, spec.x_of(m), m.intelligence)
            for m in models
            for left, icon, right, strike in [
                _split_icon(m, available=spec.available_of(m))
            ]
        ],
        marker_r_px,
        extra_obstacles=(legend_box,),
        max_candidates=spec.png_label_slots if fmt == "png" else None,
        drop_unplaceable=spec.png_label_slots is not None and fmt == "png",
    )
    save_plots(fig, spec.stem, fmt)


def save_plots(fig, stem: str, fmt: Literal["png", "svg", "both"] = "both") -> None:
    os.makedirs("plots", exist_ok=True)
    assert fmt in ("png", "svg", "both")
    if fmt in ("svg", "both"):
        print(f"Saving plots/{stem}.svg")
        fig.savefig(
            f"plots/{stem}.svg",
            format="svg",
            bbox_inches="tight",
            # Drop the <dc:date> timestamp so regenerating with unchanged data is a
            # no-op for git.
            metadata={"Date": None},
        )
    if fmt in ("png", "both"):
        print(f"Saving plots/{stem}.png")
        fig.savefig(f"plots/{stem}.png", format="png", bbox_inches="tight")

    plt.close(fig)


BREAK_SLASH_PX = 9  # half-length of the diagonal break slashes, in pixels
BREAK_WSPACE = 0.06  # gap between the two panels, as a fraction of their width
RIGHT_PANEL_RATIO = 4.5  # default left:right width ratio -- the outlier gets
# a narrow panel, so its axis is compressed by roughly this factor


def make_broken_axis_plot(spec, models, band, y_lim):
    """Draw a view whose x span is dominated by a few outliers, on a broken axis.

    One linear axis fails when a handful of models costs an order of magnitude more than
    the rest: either the outliers stretch the scale until every other point piles up in
    the leftmost few percent and no label fits, or the plot spends half its width on
    empty space. So the points at or above spec.x_break get their own narrow panel, with
    a white gap and break slashes between the two, and the right panel's x axis is
    compressed by spec.panel_ratio.

    The hardware view is the extreme case: rigs cost $250 to $24,000, and Kimi K3.1 Pro
    needs 1 TB of unified memory at $64,600 -- twice the next most expensive rig (1T
    models on 3x DGX Spark, $24,000). The high-intelligence price view is the mild one:
    thirty-four models between $0.14 and $12.21 against three between $16.81 and $30.16.
    """
    assert spec.x_break is not None
    left = [m for m in models if spec.x_of(m) < spec.x_break]
    right = [m for m in models if spec.x_of(m) >= spec.x_break]
    if not left or not right:
        raise ValueError(
            f"{spec.stem}: x_break={spec.x_break} needs models on both sides"
        )

    fig, (ax_l, ax_r) = plt.subplots(
        1,
        2,
        figsize=(spec.get_fig_w("both"), FIG_H),
        dpi=DPI,
        sharey=True,
        gridspec_kw={"width_ratios": [spec.panel_ratio or RIGHT_PANEL_RATIO, 1]},
    )

    # Axes limits, set before placing labels (placement works in pixels).
    left_hi = max(spec.x_of(m) for m in left)
    # The usual right-hand padding past the last model the panel holds. When
    # the break falls inside that padded range, round the edge up to a whole
    # tick instead, so the axis ends on a round number with the break a tick
    # inside it: the high-intelligence panel stops at $14, not $13.14.
    left_lim = left_hi * (1 + spec.x_pad)
    if spec.x_break < left_lim:
        left_lim = math.ceil(left_lim / spec.xtick_step) * spec.xtick_step
    ax_l.set_xlim(0, left_lim)
    r_lo = min(spec.x_of(m) for m in right)
    r_hi = max(spec.x_of(m) for m in right)
    r_pad = 0.18 * (r_hi - r_lo) if r_hi > r_lo else 0.18 * r_hi
    # The panel resumes at the break, not below it: its left edge is where the
    # left panel's axis stopped making sense.
    ax_r.set_xlim(max(r_lo - r_pad, spec.x_break), r_hi + r_pad)
    y_lo, y_hi = y_lim
    ax_l.set_ylim(y_lo, y_hi)
    ax_l.yaxis.set_major_locator(MultipleLocator(1))

    # Faint dotted Pareto frontier, computed over ALL models as on every other
    # view and clipped by each panel to its own slice of it: the staircase runs
    # off the left panel's right edge ("continues") and resumes in the right
    # one. A panel with no frontier point inside it draws nothing at all --
    # Kimi K3 sits left of nothing, being dominated by MiMo-V2.6-Pro (more
    # intelligence for 12x less hardware), so the hardware plot's right panel
    # is a bare dot.
    frontier_x, frontier_y = (
        pareto_frontier(MODELS, spec.x_of, spec.available_of)
        if spec.frontier
        else ([], [])
    )
    if len(frontier_x) > 1:
        panels = [ax_l]
        if any(r_lo <= x <= r_hi for x in frontier_x):
            panels.append(ax_r)
        for ax in panels:
            ax.plot(
                frontier_x,
                frontier_y,
                linestyle=":",
                color="#7a7f8a",
                linewidth=1.2,
                alpha=0.9,
                zorder=2,
            )

    _scatter_points(ax_l, left, spec.x_of)
    _scatter_points(ax_r, right, spec.x_of)

    if band is not None:
        # The green band is a per-task price band, so it lives in the left
        # panel only -- and only while its x range is inside the break.
        b_lo = max(band[0], y_lo)
        b_hi = min(band[1], y_hi)
        if b_hi > b_lo and LOW_COST_THRESHOLD < spec.x_break:
            ax_l.add_patch(
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

    # Thousands separators: the right panel's prices can be six figures.
    money = (
        StrMethodFormatter(spec.break_xtick_format)
        if spec.break_xtick_format is not None
        else FormatStrFormatter(spec.xtick_format)
    )
    ax_l.xaxis.set_major_locator(MultipleLocator(spec.xtick_step))
    ax_l.xaxis.set_major_formatter(money)
    # The right panel is a fraction of the figure wide, so a view whose prices
    # are six figures gets one tick per model instead of a fixed step: labels
    # would collide.
    if spec.right_xtick_step is None:
        ax_r.set_xticks(sorted({spec.x_of(m) for m in right}))
    else:
        ax_r.xaxis.set_major_locator(MultipleLocator(spec.right_xtick_step))
    ax_r.xaxis.set_major_formatter(money)

    ax_l.set_xlabel(spec.x_label, fontsize=15, fontweight="bold", labelpad=12)
    ax_l.set_ylabel(
        "Artificial Analysis Intelligence Index",
        fontsize=15,
        fontweight="bold",
        labelpad=12,
    )
    ax_l.set_title(spec.title, fontsize=20, fontweight="bold", loc="left", pad=18)
    # The right panel's own price is on its x axis, so the only thing to say
    # about it is that its x scale is a different one.
    ax_r.set_title(
        "x-axis break:\nnot to scale",
        fontsize=13,
        color="#5b6270",
        loc="left",
        pad=18,
    )

    for ax in (ax_l, ax_r):
        ax.grid(True, color="#e6e8ec", linewidth=1, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(labelsize=12, colors="#5b6270")

    # Legend: one entry per publisher actually present in this plot.
    legend_ax = ax_r if spec.legend_panel == "right" else ax_l
    legend, have_unavailable = _plot_legend(
        legend_ax,
        models,
        loc="lower right",
        available_of=spec.available_of,
        hidden=spec.hidden_icons,
    )
    fig.tight_layout()
    # The gap between the panels, after tight_layout rather than in
    # gridspec_kw: tight_layout gives up ("Axes that are not compatible with
    # tight_layout") on a GridSpec whose params were set at construction.
    fig.subplots_adjust(wspace=BREAK_WSPACE)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    if have_unavailable:
        for text in legend.get_texts():
            if text.get_text() == "Not publicly available":
                # zorder 6: above the legend frame
                _strike_text(legend_ax, text, renderer, zorder=6)
                break

    # Break slashes: short diagonals across the bottom spine where the two
    # panels meet. They live in display pixels (via the axes' inverted
    # transform) because a square step in axes fractions would come out
    # nearly horizontal on a 20-inch-wide panel.
    for ax, x_edge in ((ax_l, 1.0), (ax_r, 0.0)):
        bb = ax.get_window_extent(renderer)
        dx = BREAK_SLASH_PX / bb.width
        dy = BREAK_SLASH_PX / bb.height
        ax.add_line(
            Line2D(
                [x_edge - dx, x_edge + dx],
                [-dy, dy],
                transform=ax.transAxes,
                lw=1.4,
                color="#5b6270",
                zorder=6,
                clip_on=False,
            )
        )

    marker_r_px = (DOT_SIZE**0.5) / 2 / 72 * DPI + 2
    for ax, panel in ((ax_l, left), (ax_r, right)):
        place_labels(
            ax,
            fig,
            [
                (left_, right_, icon, strike, spec.x_of(m), m.intelligence)
                for m in panel
                for left_, icon, right_, strike in [
                    _split_icon(
                        m,
                        spec.label_of(m),
                        available=spec.available_of(m),
                        hidden=spec.hidden_icons,
                    )
                ]
            ],
            marker_r_px,
            extra_obstacles=(_pad(legend.get_window_extent(renderer)),)
            if ax is legend_ax
            else (),
        )

    save_plots(fig, spec.stem)


def make_bar_plot(spec, models):
    """Horizontal bar version of a PlotSpec: one bar per model, x = x_of(m),
    ordered by intelligence (dumbest at the bottom). The model names are the
    y tick labels, so there is no auto-placed text, no Pareto frontier and no
    green band. A label's glyphs (thief mask, ⚡) cannot be embedded in a plain
    tick string: they are stripped from the label and drawn as their vector
    icons just right of the label text (see the icon pass after the draw).

    With one_per_model, a model's effort variants collapse to a single bar: one
    permaslug carries every effort level, and the variants of a model OR carries
    no price for share only their name, so the key falls back to that name with
    the suffix dropped (else five "GPT-6.1 Sol" rows all draw the same empty
    bar). The max-effort row -- the highest intelligence -- stands in and its
    effort suffix is stripped from the label, so one name means one model.
    Two rows of the same name stay separate only if their permaslugs differ:
    Muse Spark 1.3 and its -contributor endpoint are different offers, not
    thinking levels. Each effort level burns a different mix of input and
    output tokens, so a family's Δ really does spread (Sonnet 5.5 is +42% at
    medium and +86% at max); the bar reports the smartest variant's.
    """
    if spec.one_per_model:
        best: dict[str, Model] = {}
        for m in models:
            key = m.or_slug or _clean_label(m)
            if key not in best or m.intelligence > best[key].intelligence:
                best[key] = m
        models = list(best.values())
    ordered = sorted(models, key=lambda m: m.intelligence)
    ys = list(range(len(ordered)))
    xs = [spec.x_of(m) for m in ordered]
    colors = [PUBLISHERS[m.publisher] for m in ordered]
    # The tick labels are plain strings, so a label's glyphs (thief mask, ⚡)
    # cannot be embedded: strip them from the text and record the row, the
    # matching icon is drawn next to the label after the figure is rendered
    # (below).
    labels = []
    icon_rows = []
    for i, m in enumerate(ordered):
        name = _clean_label(m) if spec.one_per_model else m.name
        left, icon, right, _strike = _split_icon(m, name, hidden=spec.hidden_icons)
        labels.append((left + " " + right).strip())
        if icon is not None:
            icon_rows.append((i, icon))

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=DPI)
    ax.barh(ys, xs, height=0.7, color=colors, zorder=3)
    ax.set_yticks(ys, labels, fontsize=LABEL_SIZE)
    ax.set_ylim(-0.6, len(ordered) - 0.4)

    # x axis (set limits before drawing the zero line)
    x_lo, x_hi = min(xs), max(xs)
    pad = 0.05 * (x_hi - x_lo)
    ax.set_xlim(max(x_lo - pad, spec.x_min), x_hi + pad)
    ax.xaxis.set_major_locator(MultipleLocator(spec.xtick_step))
    ax.xaxis.set_major_formatter(FormatStrFormatter(spec.xtick_format))
    if spec.zero_line:
        # AA's published price is the reference the delta axis is measured
        # from: mark it with a solid vertical line over the bars.
        ax.axvline(0, color="#9aa1ad", linewidth=1.2, zorder=4)

    ax.set_xlabel(spec.x_label, fontsize=15, fontweight="bold", labelpad=12)
    ax.set_title(spec.title, fontsize=20, fontweight="bold", loc="left", pad=18)

    # Faint guide line across each row, so a label can be traced to its bar
    # even when the bar is too short to reach it, plus the x grid.
    ax.grid(True, axis="both", color="#e6e8ec", linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    # Room in the tick pad for the row icons, drawn between the label text
    # and the axes (below).
    y_pad = plt.rcParams["ytick.major.pad"]
    if icon_rows:
        y_pad += (ICON_GAP + max(ICON_SIZE[i] for _, i in icon_rows)) * 72 / DPI + 2
    ax.tick_params(axis="y", length=0, pad=y_pad)
    ax.tick_params(axis="x", labelsize=12, colors="#5b6270")

    legend, have_unavailable = _plot_legend(
        ax,
        models,
        loc="center left",
        bbox_to_anchor=(1.005, 0.5),
        available_of=lambda m: m.available_datacenter,
        hidden=spec.hidden_icons,
    )
    fig.tight_layout()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    if have_unavailable:
        for text in legend.get_texts():
            if text.get_text() == "Not publicly available":
                _strike_text(ax, text, renderer, zorder=6)  # above the legend frame
                break

    # The marker icons for the tick labels: each is a separate vector
    # marker, centred just right of its label text (the y pad above leaves
    # room). In data coords, like the leader lines in place_labels, so it
    # survives SVG export.
    if icon_rows:
        inv = ax.transData.inverted()
        tick_labels = ax.get_yticklabels()
        for i, icon in icon_rows:
            bb = tick_labels[i].get_window_extent(renderer)
            s = ICON_SIZE[icon]
            cx = bb.x1 + ICON_GAP + s / 2
            cy = (bb.y0 + bb.y1) / 2
            fill, edge = ICON_COLORS[icon]
            ((ix, iy),) = inv.transform([(cx, cy)])
            ax.plot(
                [ix],
                [iy],
                marker=ICON_PATHS[icon],
                markersize=s * 72 / DPI,
                markerfacecolor=fill,
                markeredgecolor=edge,
                markeredgewidth=1.0,
                linestyle="none",
                zorder=4,
                clip_on=False,
            )

    save_plots(fig, spec.stem)


def _display_width(text: str) -> int:
    """Terminal cell width of text: East Asian wide/fullwidth glyphs (which
    includes the ⚡ spliced into local model names) occupy two cells."""
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)


def _pad_cell(text: str, width: int, right: bool = False) -> str:
    """Pad text to `width` terminal cells (not characters)."""
    gap = " " * max(0, width - _display_width(text))
    return gap + text if right else text + gap


def report_prices() -> None:
    """Print one aligned row per model: AA's sticker price per task, the
    displayed price per task, and how the two compare."""
    rows: list[tuple[str, str, str, str, str]] = []
    for m in MODELS:
        price = m.price_per_task()
        if m.provider_type is ProviderType.LOCAL:
            rows.append((m.name, "-", f"{price:.4f}", "electricity", "-"))
            continue
        aa_total = m.aa_total_cost_per_task()
        tokens = m.aa_token_counts()
        if tokens is None or m.or_eff_input_price is None:
            rows.append((m.name, f"{aa_total:.4f}", f"{price:.4f}", "+0.0%", "-"))
            continue
        rows.append(
            (
                m.name,
                f"{aa_total:.4f}",
                f"{price:.4f}",
                f"{price / aa_total - 1:+.1%}",
                f"{m.or_eff_input_price:.3f}/{m.or_eff_output_price:.3f}",
            )
        )
    headers = ("model", "AA $/task", "price/task", "vs AA", "OR $/M in/out")
    right = (False, True, True, True, True)
    widths = [
        max(_display_width(header), *(_display_width(row[i]) for row in rows))
        for i, header in enumerate(headers)
    ]
    print("  ".join(_pad_cell(h, w, r) for h, w, r in zip(headers, widths, right)))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print("  ".join(_pad_cell(c, w, r) for c, w, r in zip(row, widths, right)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="report residual label overlaps to stderr",
    )
    args = parser.parse_args(argv)
    global VERBOSE
    VERBOSE = args.verbose

    report_prices()

    models_by_stem = {}
    for spec in PLOTS:
        models = [m for m in MODELS if spec.filt(m)]
        if not models:
            raise SystemExit(f"no models match the filter for {spec.stem}")
        models_by_stem[spec.stem] = models

    # Green band (x $0-$LOW_COST_THRESHOLD): the points that appear on both plots (cheap
    # AND smart). Its edges coincide with the band-side axis edge of each plot: the
    # floor of the high-int plot's points and the ceil of the low-cost plot's points.
    # Identical on both plots. Empty (skipped) when no model is both cheap and smart.
    band = (
        math.floor(min(m.intelligence for m in models_by_stem["high_intelligence"])),
        math.ceil(max(m.intelligence for m in models_by_stem["low_cost"])),
    )
    for spec in PLOTS:
        models = models_by_stem[spec.stem]
        if spec.bar:
            print("Generating bar plot:", spec.stem)
            make_bar_plot(spec, models)
            continue
        # Add 1 index point of slack to the bottom: a label centred on the lowest dot is
        # ~18 px tall at this DPI, i.e. ~0.8 index points, and without the room every
        # near slot spills under the axis and the capped PNG label search drops that
        # model's name entirely.
        y_lo = math.floor(min(m.intelligence for m in models)) - 1.0
        y_hi = math.ceil(max(m.intelligence for m in models)) + 0.2
        if y_hi == y_lo:  # degenerate: all points on one level
            y_hi = y_lo + 1
        if spec.band_side == "bottom":
            y_lo = band[0]
        elif spec.band_side == "top":
            y_hi = band[1]
        if spec.x_break is not None:  # two panels, not one
            print("Generating broken-axis plot:", spec.stem)
            make_broken_axis_plot(
                spec, models, band if spec.band else None, (y_lo, y_hi)
            )
            continue
        print("Generating datacenter plot:", spec.stem)
        make_datacenter_plots(spec, models, band if spec.band else None, (y_lo, y_hi))


if __name__ == "__main__":
    main()
