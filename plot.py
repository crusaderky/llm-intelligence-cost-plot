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
the volatile spot quote: for every provider and every day, the first quintile
of the effective price across providers, then the median of those across days
(see .agents/skills/refresh-models). That gives one synthetic OR price pair
per model, in $/M tokens:

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
first that collides with nothing already placed. So MODELS only needs
publisher / name / intelligence / provider type plus the raw cost inputs --
just add rows and re-run.
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
UNAVAILABLE_COLOR = "#4b5563"  # dark grey for available=False labels
LEADER_MIN = 8  # draw a leader once the label sits this far off the dot
CROWD_X = 200  # px window used to decide a point is "in a cluster"
CROWD_Y = 60
CROWD_OFFSET = 23  # clustered labels sit at least this far out (points),
# so their leader lines are long enough to follow

# Bottom of the high-intelligence plot
HIGH_INTELLIGENCE_THRESHOLD = 39
# Right edge of the green band: the cheap cluster tops out at ~$0.15/task on
# the new scale, and the next most expensive model sits at ~$1.4.
LOW_COST_THRESHOLD = 0.25


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
    "Meta": "#0089f4",
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
    peak_power_draw: int  # Watts under max load
    idle_power_draw: int  # Watts when idling


RTX3090 = LocalHardware("RTX 3090", 350, 43)
STRIX_HALO = LocalHardware("Strix Halo 128GB", 170, 11)

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
# count. 3.78 is the median of (uncached input tokens / output tokens) over the
# 34 models AA does break down; the real ratio climbs steeply as output tokens
# fall (2.1 at max effort, 12.9 on a low-effort record), so this is a coarse
# middle guess, not a per-model fit.
LOCAL_INPUT_TOKEN_RATIO = 3.78


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
    # Datacenter models: OR's synthetic effective prices in $/M tokens, the
    # first quintile of provider prices per day, median across the days of the
    # trailing week (see .agents/skills/refresh-models).
    or_eff_input_price: float | None = field(default=None, kw_only=True)
    or_eff_output_price: float | None = field(default=None, kw_only=True)
    # Local models: measured throughput on `hardware` (None = RTX3090).
    local_speed: LocalSpeed | None = field(default=None, kw_only=True)
    hardware: LocalHardware | None = field(default=None, kw_only=True)
    # One-off USD price of the cheapest hardware that can run the model
    # locally, from the README's "Larger local models" table. Independent of
    # the per-task price (which is electricity only): this is the machine, not
    # the power bill. None = no table entry, so the model is left off the
    # hardware-cost plot.
    hardware_cost: float | None = field(default=None, kw_only=True)
    # Not on AA: the point is extrapolated/estimated. Rendered as a hollow
    # circle, with a matching "Estimated (not on AA)" legend entry.
    estimated: bool = field(default=False, kw_only=True)
    # False: the model is not offered to the public yet (it is on AA but has
    # no real price to buy it at). Rendered grey with a strikethrough and kept
    # out of the Pareto frontier, with a "Not publicly available" legend entry.
    available: bool = field(default=True, kw_only=True)
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
            self.name = f"{self.name} ({self.hardware.name} ⚡)"
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
            power_draw = self.hardware.peak_power_draw - self.hardware.idle_power_draw
            kwh_per_task = power_draw * sec_per_task / 3_600_000
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
    #   (openrouter.ai, GET /api/frontend/v1/stats/effective-pricing: first
    #   quintile of the providers' effective prices each day, median across
    #   the days of the trailing week, $/M tokens).
    #   or_session_cost_10_49_turns (GET
    #   /api/frontend/v1/rankings/session-cost: median 10-49-turn session
    #   cost, averaged across the coding harnesses that carry the model) and
    #   or_toks_served (GET /api/frontend/v1/rankings/models?view=week:
    #   prompt + completion tokens served in the trailing week, all variants)
    #   are recorded for reference; neither prices the plot.
    # - local: local_speed, both rates on real hardware. The prefill rate is a
    #   guesstimate at 10x the decode rate until it gets measured.
    # - hardware_cost: one-off price of the cheapest rig in the README's
    #   "Larger local models" table that fits the model; set only for the
    #   models that table names.
    # See .agents/skills/refresh-models for how to re-fetch these numbers.
    # --- Local models (price per task = electricity) ---
    Model(
        "OpenBMB",
        "MiniCPM5-2B",
        12.4634,
        ProviderType.LOCAL,
        aa_tok_per_task=21834,
        local_speed=LocalSpeed(prefill=7460, decode=200),
        hardware_cost=250,
    ),
    Model(
        "Alibaba",
        "Qwen3.6-35B-A3B",
        18.2290,
        ProviderType.LOCAL,
        aa_tok_per_task=34594,
        aa_cost_per_task=PricedTokens(0.3972175434327144, 0.07783591281841616, 0.0),
        # AA reports no cache reads for this model and publishes no cache-read
        # price; 0.0 in the cached slot of both splits means "never caches"
        aa_sticker_price=PricedTokens(0.375, 2.25, 0.0),
        local_speed=LocalSpeed(prefill=2215, decode=150),
        hardware_cost=1500,
    ),
    # Not on AA: intelligence = linear interpolation of median of self-published
    # benchmark scores vs. known intelligence index scores.
    # Tokens per task pixel-peeped from self-published cost per task plot
    Model(
        "Accio",
        "Occamy-1.0",
        29.7,
        ProviderType.LOCAL,
        aa_tok_per_task=34594 * 1.23,
        local_speed=LocalSpeed(prefill=2345, decode=134),
        hardware_cost=1500,
        estimated=True,
    ),
    Model(
        "Meta",
        "Muse Glimmer",
        17.4754,
        ProviderType.LOCAL,
        aa_tok_per_task=13925,
        aa_cost_per_task=PricedTokens(
            0.013893052794893865, 0.0187990159196835, 0.024014277935722587
        ),
        aa_sticker_price=PricedTokens(0.32499999999999996, 1.35, 0.04),
        local_speed=LocalSpeed(prefill=938, decode=124),
        hardware_cost=2300,
    ),
    Model(
        "Institute of Foundation Models",
        "K2 Horizon 7B",
        20.5959,
        ProviderType.LOCAL,
        aa_tok_per_task=72163,
        local_speed=LocalSpeed(prefill=3430, decode=104),
        hardware_cost=1500,
    ),
    Model(
        "Alibaba",
        "Qwen3.8-27B",
        33.6963,
        ProviderType.LOCAL,
        aa_tok_per_task=66797,
        aa_cost_per_task=PricedTokens(
            0.432331423090756, 0.20039241455380952, 0.3745911160120289
        ),
        aa_sticker_price=PricedTokens(0.5, 3.0, 0.1),
        local_speed=LocalSpeed(prefill=963, decode=57),
        hardware_cost=2300,
    ),
    # Not on AA: intelligence = 0.9165 x Qwen3.8-27B, the 91.65% of BF16 that
    # ByteShape measured for this ternary quant (see README note); tokens per
    # task assumed identical to Qwen3.8-27B.
    Model(
        "Prism-ML",
        "Ternary-Bonsai-2",
        30.8827,
        ProviderType.LOCAL,
        aa_tok_per_task=66797,
        local_speed=LocalSpeed(prefill=963, decode=73),
        hardware_cost=1500,
        estimated=True,
    ),
    Model(
        "Alibaba",
        "Qwen3.8-Flash",
        39.8223,
        ProviderType.LOCAL,
        aa_tok_per_task=107885,
        aa_cost_per_task=PricedTokens(
            0.04967406603399954, 0.05070581107333271, 0.2717961254229642
        ),
        aa_sticker_price=PricedTokens(0.15, 0.47, 0.016),
        # https://github.com/peonist-ai/halogen-flash-server
        local_speed=LocalSpeed(prefill=1584, decode=55),
        hardware_cost=3800,
        hardware=STRIX_HALO,
    ),
    # --- Datacenter models (price per task = AA's token mix priced at OR's
    # effective input/output prices) ---
    Model(
        "Alibaba",
        "Qwen3.8-Flash",
        39.8223,
        ProviderType.DATACENTER,
        aa_tok_per_task=107885,
        aa_cost_per_task=PricedTokens(
            0.04967406603399954, 0.05070581107333271, 0.2717961254229642
        ),
        aa_sticker_price=PricedTokens(0.15, 0.47, 0.016),
        or_slug="qwen/qwen3.8-flash-20260826",
        or_session_cost_10_49_turns=0.045219097333333326,
        or_toks_served=514237440996,
        or_eff_input_price=0.04191,
        or_eff_output_price=0.4696,
    ),
    Model(
        "Alibaba",
        "Qwen3.8 Max (0902)",
        45.4152,
        ProviderType.DATACENTER,
        aa_tok_per_task=107730,
        aa_cost_per_task=PricedTokens(
            1.7983995678147318, 0.6463813839388131, 2.9637284766204717
        ),
        aa_sticker_price=PricedTokens(2.0, 6.0, 0.25),
        or_slug="qwen/qwen3.8-max-20260902",
        or_session_cost_10_49_turns=0.76834275,
        or_toks_served=274540525277,
        or_eff_input_price=0.3659,
        or_eff_output_price=6.0,
    ),
    Model(
        "DeepSeek",
        "DeepSeek V4.1 Flash",
        39.4562,
        ProviderType.DATACENTER,
        aa_tok_per_task=88574,
        hardware_cost=15300,
        aa_cost_per_task=PricedTokens(
            0.10051303480135712, 0.10628879819362691, 0.058423437101084455
        ),
        aa_sticker_price=PricedTokens(0.3, 1.2, 0.006),
        or_slug="deepseek/deepseek-v4.1-flash-20260910",
        or_session_cost_10_49_turns=0.058345011,
        or_toks_served=25632936806017,
        or_eff_input_price=0.01795,
        or_eff_output_price=0.5588,
    ),
    Model(
        "Tencent",
        "Hy3",
        25.2973,
        ProviderType.DATACENTER,
        aa_tok_per_task=46161,
        aa_cost_per_task=PricedTokens(
            0.011163402706883093, 0.02561934946572233, 0.03501487864599409
        ),
        aa_sticker_price=PricedTokens(0.136, 0.5549999999999999, 0.034),
        or_slug="tencent/hy3-20260706",
        or_session_cost_10_49_turns=0.04743342,
        or_toks_served=2145970555775,
        or_eff_input_price=0.04681,
        or_eff_output_price=0.5296,
    ),
    # Guesstimate - not on AA. Intelligence is extrapolated from Tencent's own
    # agentic-benchmark chart for Hy4 preview
    # (https://hy.tencent.ai/research/hy4-preview): per benchmark, OLS of the
    # six comparison models' AA Intelligence Index on their chart score,
    # inverted at Hy4 preview's score and averaged over the 12 benchmarks
    # gives 44.3 (leave-one-out over those six models: mean error +0.2, MAE
    # 2.2). The anchors are the versions the chart ran, not their successors
    # (Qwen3.8 Max 0803, GPT-5.6 Sol, Claude Opus 5), so the estimate is
    # "as of the chart". Cross-check: the same fit on Terminal Bench 2.1 with
    # Hy3 added as a wide-range calibration point lands Hy3 at 26.2 against its
    # real 25.3, and gives 43.4 for Hy4 preview.
    # Price per task: no AA breakdown exists for a model AA has not measured,
    # so there is no token mix to reprice; aa_cost_per_task carries the
    # guesstimate below and the delta plot reads 0%, like the other estimated
    # rows. The OR figures are recorded but unused for the same reason. The
    # guess itself: OR's median 10-49-turn session cost for the permaslug
    # averaged $0.195 (Hermes Agent $0.174, Claude Code $0.217), between
    # DeepSeek V4.1 Flash ($0.070/session, $0.40/task) and Gemini 3.8 Flash
    # ($0.263, $1.21/task). Log-interpolating those two displayed prices at
    # $0.195 gives $0.946, which at that session cost is 74,628 output tokens
    # per task: that token count is the guess.
    Model(
        "Tencent",
        "Hy4 preview",
        44.3,
        ProviderType.DATACENTER,
        aa_tok_per_task=74628,
        aa_cost_per_task=0.9461563312,
        or_slug="tencent/hy4-preview-20260827",
        or_session_cost_10_49_turns=0.21568037,
        or_toks_served=7483445110726,
        or_eff_input_price=0.07637364127197355,
        or_eff_output_price=2.500449440012165,
        hardware_cost=21200,
        estimated=True,
    ),
    Model(
        "Meta",
        "Muse Spark 1.3",
        48.0923,
        ProviderType.DATACENTER,
        aa_tok_per_task=60200,
        aa_cost_per_task=PricedTokens(
            0.5058916529996315, 0.25585066991648475, 0.8431508870964701
        ),
        aa_sticker_price=PricedTokens(1.25, 4.25, 0.15),
        or_slug="meta/muse-spark-1.3-20260902",
        or_session_cost_10_49_turns=0.50941202,
        or_toks_served=284849827129,
        or_eff_input_price=0.4159,
        or_eff_output_price=4.25,
    ),
    Model(
        "Meta",
        "Muse Spark 1.3",
        48.0923,
        ProviderType.DATACENTER,
        aa_tok_per_task=60200,
        aa_cost_per_task=PricedTokens(
            0.5058916529996315, 0.25585066991648475, 0.8431508870964701
        ),
        aa_sticker_price=PricedTokens(1.25, 4.25, 0.15),
        or_slug="meta/muse-spark-1.3-contributor-20260902",
        or_session_cost_10_49_turns=0.026878580875,
        or_toks_served=1463544514326,
        or_eff_input_price=0.04726,
        or_eff_output_price=0.1996,
        trains_on_your_data=True,
    ),
    Model(
        "Z AI",
        "GLM-5.3-Flash (high)",
        # scaled from the (max) record, see the README note
        41.8075 * 28.01 / 28.99,
        ProviderType.DATACENTER,
        aa_tok_per_task=round(68673 * 70610 / 138690),
        # the token split scales with the token count; the sticker prices are
        # the model's list prices and do not
        aa_cost_per_task=PricedTokens(
            0.0217437763348598 * 70610 / 138690,
            0.034336576028258216 * 70610 / 138690,
            0.19717920806761985 * 70610 / 138690,
        ),
        aa_sticker_price=PricedTokens(0.15, 0.5, 0.026),
        or_slug="z-ai/glm-5.3-flash-20260826",
        or_session_cost_10_49_turns=0.03966811475,
        or_toks_served=9570478959426,
        or_eff_input_price=0.04117,
        or_eff_output_price=0.4145,
        estimated=True,
    ),
    Model(
        "Z AI",
        "GLM-5.3-Flash (max)",
        41.8075,
        ProviderType.DATACENTER,
        aa_tok_per_task=68673,
        aa_cost_per_task=PricedTokens(
            0.0217437763348598, 0.034336576028258216, 0.19717920806761985
        ),
        aa_sticker_price=PricedTokens(0.15, 0.5, 0.026),
        or_slug="z-ai/glm-5.3-flash-20260826",
        or_session_cost_10_49_turns=0.03966811475,
        or_toks_served=9570478959426,
        or_eff_input_price=0.04117,
        or_eff_output_price=0.4145,
        hardware_cost=10200,
    ),
    Model(
        "Z AI",
        "GLM-5.3",
        44.7774,
        ProviderType.DATACENTER,
        aa_tok_per_task=71128,
        aa_cost_per_task=PricedTokens(
            0.17082790879500323, 0.3129615000082585, 1.5218481062416964
        ),
        aa_sticker_price=PricedTokens(1.4, 4.4, 0.26),
        hardware_cost=21200,
        or_slug="z-ai/glm-5.3-20260816",
        or_session_cost_10_49_turns=0.4667711225,
        or_toks_served=2834146362893,
        or_eff_input_price=0.1872,
        or_eff_output_price=2.881,
    ),
    Model(
        "Moonshot AI",
        "Kimi K3",
        43.5938,
        ProviderType.DATACENTER,
        aa_tok_per_task=48455,
        aa_cost_per_task=PricedTokens(
            0.47145748156727935, 0.7268284815216798, 0.8018463373535905
        ),
        aa_sticker_price=PricedTokens(3.0, 15.0, 0.3),
        hardware_cost=320_000,
        or_slug="moonshotai/kimi-k3-20260715",
        or_session_cost_10_49_turns=0.7484920825,
        or_toks_served=1597559226448,
        or_eff_input_price=0.473,
        or_eff_output_price=12.55,
    ),
    Model(
        "Google",
        "Gemini 3.8 Flash",
        40.9262,
        ProviderType.DATACENTER,
        aa_tok_per_task=71003,
        aa_cost_per_task=PricedTokens(
            0.5700656958516558, 0.2662596260114765, 0.40646943883191045
        ),
        aa_sticker_price=PricedTokens(0.75, 3.75, 0.075),
        or_slug="google/gemini-3.8-flash-20260902",
        or_session_cost_10_49_turns=0.272685015,
        or_toks_served=2155413033520,
        or_eff_input_price=0.2081,
        or_eff_output_price=1.909,
    ),
    # AA badges it "Not publicly available" (released 2026-09-30, no provider serves
    # it), and OpenRouter has no permaslug for it, so there is no session cost to
    # rescale by: the price stays AA's sticker figure unscaled.
    Model(
        "Google",
        "Gemini 4 Argon (high)",
        52.5605655745982,
        ProviderType.DATACENTER,
        aa_tok_per_task=61558,
        aa_cost_per_task=PricedTokens(
            1.0554022379429868, 0.6155762620764799, 0.3193438792795405
        ),
        aa_sticker_price=PricedTokens(2.0, 10.0, 0.1),
        available=False,
    ),
    Model(
        "SpaceXAI",
        "Grok 4.7 (high)",
        46.3321770885625,
        ProviderType.DATACENTER,
        aa_tok_per_task=65901,
        aa_cost_per_task=PricedTokens(
            1.2506133482768427, 0.3954088064888365, 1.0800845370203476
        ),
        aa_sticker_price=PricedTokens(2.0, 6.0, 0.5),
        or_slug="x-ai/grok-4.7-20260916",
        or_session_cost_10_49_turns=0.90376115,
        or_toks_served=346401685126,
        or_eff_input_price=0.8025,
        or_eff_output_price=6.132,
    ),
    Model(
        "SpaceXAI",
        "Grok 4.7 (xhigh)",
        46.4465506302286,
        ProviderType.DATACENTER,
        aa_tok_per_task=80561,
        aa_cost_per_task=PricedTokens(
            1.722124090890538, 0.4833681883711849, 1.5328336728452165
        ),
        aa_sticker_price=PricedTokens(2.0, 6.0, 0.5),
        or_slug="x-ai/grok-4.7-20260916",
        or_session_cost_10_49_turns=0.90376115,
        or_toks_served=346401685126,
        or_eff_input_price=0.8025,
        or_eff_output_price=6.132,
    ),
    # Expected to land on OpenRouter on 2026-10-15
    Model(
        "StepFun",
        "Step 5 Preview",
        43.7343049141614,
        ProviderType.DATACENTER,
        aa_tok_per_task=64144,
        aa_cost_per_task=PricedTokens(
            0.23538385680255072, 0.1731895812858314, 0.30891671284001165
        ),
        aa_sticker_price=PricedTokens(1.0, 2.7, 0.05),
        available=False,
    ),
    Model(
        "Xiaomi",
        "MiMo-V2.6-Flash",
        37.8843590141754,
        ProviderType.DATACENTER,
        aa_tok_per_task=77792,
        aa_cost_per_task=PricedTokens(
            0.022581436951699726, 0.02178167406180075, 0.017951590352690297
        ),
        aa_sticker_price=PricedTokens(0.14, 0.28, 0.0028),
        hardware_cost=6800,
        or_slug="xiaomi/mimo-v2.6-flash-20260921",
        or_session_cost_10_49_turns=0.0333002585,
        or_toks_served=9535285723501,
        or_eff_input_price=0.02109,
        or_eff_output_price=0.2781,
    ),
    Model(
        "Xiaomi",
        "MiMo-V2.6-Pro",
        46.3242065310383,
        ProviderType.DATACENTER,
        aa_tok_per_task=64276,
        aa_cost_per_task=PricedTokens(
            0.057833820951334845, 0.055919859445151064, 0.01946950897564899
        ),
        aa_sticker_price=PricedTokens(0.435, 0.87, 0.0036),
        hardware_cost=27200,
        or_slug="xiaomi/mimo-v2.6-pro-20260921",
        or_session_cost_10_49_turns=0.10763974166666666,
        or_toks_served=1195603370581,
        or_eff_input_price=0.03238,
        or_eff_output_price=0.8662,
    ),
    Model(
        "OpenAI",
        "GPT-6 Luna (low)",
        21.5261980080866,
        ProviderType.DATACENTER,
        aa_tok_per_task=2087,
        aa_cost_per_task=PricedTokens(
            0.003010489994633766, 0.0010436098775824284, 0.0004625076971940582
        ),
        aa_sticker_price=PricedTokens(0.1, 0.5, 0.01),
        or_slug="openai/gpt-6-luna-20260922",
        or_session_cost_10_49_turns=0.029956872,
        or_toks_served=6073943000351,
        or_eff_input_price=0.01837,
        or_eff_output_price=0.3515,
    ),
    Model(
        "OpenAI",
        "GPT-6 Luna (medium)",
        29.9251773515762,
        ProviderType.DATACENTER,
        aa_tok_per_task=11456,
        aa_cost_per_task=PricedTokens(
            0.005698203226871066, 0.005728227888056104, 0.006050474952908773
        ),
        aa_sticker_price=PricedTokens(0.1, 0.5, 0.01),
        or_slug="openai/gpt-6-luna-20260922",
        or_session_cost_10_49_turns=0.029956872,
        or_toks_served=6073943000351,
        or_eff_input_price=0.01837,
        or_eff_output_price=0.3515,
    ),
    Model(
        "OpenAI",
        "GPT-6 Luna (high)",
        32.9282054150837,
        ProviderType.DATACENTER,
        aa_tok_per_task=19703,
        aa_cost_per_task=PricedTokens(
            0.007808019704092541, 0.009851617041137984, 0.011371939658497492
        ),
        aa_sticker_price=PricedTokens(0.1, 0.5, 0.01),
        or_slug="openai/gpt-6-luna-20260922",
        or_session_cost_10_49_turns=0.029956872,
        or_toks_served=6073943000351,
        or_eff_input_price=0.01837,
        or_eff_output_price=0.3515,
    ),
    Model(
        "OpenAI",
        "GPT-6 Luna (xhigh)",
        34.5587320217785,
        ProviderType.DATACENTER,
        aa_tok_per_task=27487,
        aa_cost_per_task=PricedTokens(
            0.009569620806657932, 0.013743548062472534, 0.018870379054820634
        ),
        aa_sticker_price=PricedTokens(0.1, 0.5, 0.01),
        or_slug="openai/gpt-6-luna-20260922",
        or_session_cost_10_49_turns=0.029956872,
        or_toks_served=6073943000351,
        or_eff_input_price=0.01837,
        or_eff_output_price=0.3515,
    ),
    Model(
        "OpenAI",
        "GPT-6 Luna (max)",
        38.1245186869738,
        ProviderType.DATACENTER,
        aa_tok_per_task=50012,
        aa_cost_per_task=PricedTokens(
            0.013840232874372521, 0.025005845215270134, 0.02897687860847939
        ),
        aa_sticker_price=PricedTokens(0.1, 0.5, 0.01),
        or_slug="openai/gpt-6-luna-20260922",
        or_session_cost_10_49_turns=0.029956872,
        or_toks_served=6073943000351,
        or_eff_input_price=0.01837,
        or_eff_output_price=0.3515,
    ),
    # Not on OR: no 10-49-turn session data — falls back to AA's cost per task.
    Model(
        "OpenAI",
        "GPT-6.1 Sol (low)",
        42.0835618555848,
        ProviderType.DATACENTER,
        aa_tok_per_task=3977,
        aa_cost_per_task=PricedTokens(
            0.07543296949664269, 0.03976700974677209, 0.015551929455178966
        ),
        aa_sticker_price=PricedTokens(2.0, 10.0, 0.1),
    ),
    Model(
        "OpenAI",
        "GPT-6.1 Sol (medium)",
        47.7833271274065,
        ProviderType.DATACENTER,
        aa_tok_per_task=8070,
        aa_cost_per_task=PricedTokens(
            0.1014432139069315, 0.08069842557491673, 0.03155924319690711
        ),
        aa_sticker_price=PricedTokens(2.0, 10.0, 0.1),
    ),
    Model(
        "OpenAI",
        "GPT-6.1 Sol (high)",
        50.2377777519769,
        ProviderType.DATACENTER,
        aa_tok_per_task=13192,
        aa_cost_per_task=PricedTokens(
            0.1331377332970313, 0.1319187021982128, 0.054087988261402986
        ),
        aa_sticker_price=PricedTokens(2.0, 10.0, 0.1),
    ),
    Model(
        "OpenAI",
        "GPT-6.1 Sol (xhigh)",
        51.0377679093761,
        ProviderType.DATACENTER,
        aa_tok_per_task=17619,
        aa_cost_per_task=PricedTokens(
            0.1484522739101359, 0.17618833790384872, 0.06822055977452197
        ),
        aa_sticker_price=PricedTokens(2.0, 10.0, 0.1),
    ),
    Model(
        "OpenAI",
        "GPT-6.1 Sol (max)",
        51.8332597011541,
        ProviderType.DATACENTER,
        aa_tok_per_task=38128,
        aa_cost_per_task=PricedTokens(
            0.21968205039306782, 0.3812842891443378, 0.12320072601609774
        ),
        aa_sticker_price=PricedTokens(2.0, 10.0, 0.1),
    ),
    Model(
        "OpenAI",
        "GPT-6 Astra (low)",
        45.7819243120341,
        ProviderType.DATACENTER,
        aa_tok_per_task=4433,
        aa_cost_per_task=PricedTokens(
            0.41051399628740154, 0.22163743983553194, 0.18536249244267217
        ),
        aa_sticker_price=PricedTokens(10.0, 50.0, 1.0),
        or_slug="openai/gpt-6-astra-20260903",
        or_session_cost_10_49_turns=2.8708335,
        or_toks_served=1414081468965,
        or_eff_input_price=1.442,
        or_eff_output_price=35.27,
    ),
    Model(
        "OpenAI",
        "GPT-6 Astra (medium)",
        49.5704363034369,
        ProviderType.DATACENTER,
        aa_tok_per_task=9590,
        aa_cost_per_task=PricedTokens(
            0.58072415872659, 0.4794800326827803, 0.4804451305927465
        ),
        aa_sticker_price=PricedTokens(10.0, 50.0, 1.0),
        or_slug="openai/gpt-6-astra-20260903",
        or_session_cost_10_49_turns=2.8708335,
        or_toks_served=1414081468965,
        or_eff_input_price=1.442,
        or_eff_output_price=35.27,
    ),
    Model(
        "OpenAI",
        "GPT-6 Astra (high)",
        50.9191471636152,
        ProviderType.DATACENTER,
        aa_tok_per_task=11813,
        aa_cost_per_task=PricedTokens(
            0.6308785923021938, 0.5906404999280142, 0.5037339938746381
        ),
        aa_sticker_price=PricedTokens(10.0, 50.0, 1.0),
        or_slug="openai/gpt-6-astra-20260903",
        or_session_cost_10_49_turns=2.8708335,
        or_toks_served=1414081468965,
        or_eff_input_price=1.442,
        or_eff_output_price=35.27,
    ),
    Model(
        "OpenAI",
        "GPT-6 Astra (xhigh)",
        52.3863277108782,
        ProviderType.DATACENTER,
        aa_tok_per_task=16901,
        aa_cost_per_task=PricedTokens(
            0.7400314633137893, 0.8450590482946302, 0.7237054006606568
        ),
        aa_sticker_price=PricedTokens(10.0, 50.0, 1.0),
        or_slug="openai/gpt-6-astra-20260903",
        or_session_cost_10_49_turns=2.8708335,
        or_toks_served=1414081468965,
        or_eff_input_price=1.442,
        or_eff_output_price=35.27,
    ),
    Model(
        "OpenAI",
        "GPT-6 Astra (max)",
        52.6737,
        ProviderType.DATACENTER,
        aa_tok_per_task=27206,
        aa_cost_per_task=PricedTokens(
            0.9159346921901674, 1.3602752458235075, 0.9812903754697416
        ),
        aa_sticker_price=PricedTokens(10.0, 50.0, 1.0),
        or_slug="openai/gpt-6-astra-20260903",
        or_session_cost_10_49_turns=2.8708335,
        or_toks_served=1414081468965,
        or_eff_input_price=1.442,
        or_eff_output_price=35.27,
    ),
    Model(
        "Anthropic",
        "Claude Sonnet 5.5 (medium)",
        40.838652287929,
        ProviderType.DATACENTER,
        aa_tok_per_task=20938,
        aa_cost_per_task=PricedTokens(
            0.16774360885978076, 0.20937627614580473, 0.212287795199691
        ),
        aa_sticker_price=PricedTokens(2.0, 10.0, 0.2),
        or_slug="anthropic/claude-sonnet-5.5-20260928",
        or_toks_served=676140361401,
        or_eff_input_price=0.4823,
        or_eff_output_price=10.0,
    ),
    Model(
        "Anthropic",
        "Claude Sonnet 5.5 (high)",
        46.7356921799024,
        ProviderType.DATACENTER,
        aa_tok_per_task=37327,
        aa_cost_per_task=PricedTokens(
            0.27314221049272563, 0.37327393013630916, 0.476067434943033
        ),
        aa_sticker_price=PricedTokens(2.0, 10.0, 0.2),
        or_slug="anthropic/claude-sonnet-5.5-20260928",
        or_toks_served=676140361401,
        or_eff_input_price=0.4823,
        or_eff_output_price=10.0,
    ),
    Model(
        "Anthropic",
        "Claude Sonnet 5.5 (xhigh)",
        51.8522867021152,
        ProviderType.DATACENTER,
        aa_tok_per_task=74810,
        aa_cost_per_task=PricedTokens(
            0.529775714283959, 0.7480964840807534, 1.4683247635797327
        ),
        aa_sticker_price=PricedTokens(2.0, 10.0, 0.2),
        or_slug="anthropic/claude-sonnet-5.5-20260928",
        or_toks_served=676140361401,
        or_eff_input_price=0.4823,
        or_eff_output_price=10.0,
    ),
    Model(
        "Anthropic",
        "Claude Sonnet 5.5 (max)",
        55.9779549012591,
        ProviderType.DATACENTER,
        aa_tok_per_task=197430,
        aa_cost_per_task=PricedTokens(
            1.280625156107197, 1.974303716101193, 4.411847445202884
        ),
        aa_sticker_price=PricedTokens(2.0, 10.0, 0.2),
        or_slug="anthropic/claude-sonnet-5.5-20260928",
        or_toks_served=676140361401,
        or_eff_input_price=0.4823,
        or_eff_output_price=10.0,
    ),
    Model(
        "Anthropic",
        "Claude Opus 5.5 (low)",
        42.3077781466168,
        ProviderType.DATACENTER,
        aa_tok_per_task=10151,
        aa_cost_per_task=PricedTokens(
            0.22820591953284136, 0.2030255255124665, 0.1199490288638381
        ),
        aa_sticker_price=PricedTokens(4.0, 20.0, 0.2),
        or_slug="anthropic/claude-opus-5.5-20260921",
        or_session_cost_10_49_turns=1.4107879666666667,
        or_toks_served=2316510688375,
        or_eff_input_price=0.6881,
        or_eff_output_price=20.0,
    ),
    Model(
        "Anthropic",
        "Claude Opus 5.5 (medium)",
        51.2434931792768,
        ProviderType.DATACENTER,
        aa_tok_per_task=25745,
        aa_cost_per_task=PricedTokens(
            0.4115624102397983, 0.5149057379088234, 0.4095411957490367
        ),
        aa_sticker_price=PricedTokens(4.0, 20.0, 0.2),
        or_slug="anthropic/claude-opus-5.5-20260921",
        or_session_cost_10_49_turns=1.4107879666666667,
        or_toks_served=2316510688375,
        or_eff_input_price=0.6881,
        or_eff_output_price=20.0,
    ),
    Model(
        "Anthropic",
        "Claude Opus 5.5 (high)",
        53.5831959822067,
        ProviderType.DATACENTER,
        aa_tok_per_task=35584,
        aa_cost_per_task=PricedTokens(
            0.5066628428356482, 0.711681177559682, 0.6041606983763739
        ),
        aa_sticker_price=PricedTokens(4.0, 20.0, 0.2),
        or_slug="anthropic/claude-opus-5.5-20260921",
        or_session_cost_10_49_turns=1.4107879666666667,
        or_toks_served=2316510688375,
        or_eff_input_price=0.6881,
        or_eff_output_price=20.0,
    ),
    Model(
        "Anthropic",
        "Claude Opus 5.5 (xhigh)",
        55.9873505840139,
        ProviderType.DATACENTER,
        aa_tok_per_task=65667,
        aa_cost_per_task=PricedTokens(
            0.7764090415678275, 1.3133479013199578, 1.369353232934504
        ),
        aa_sticker_price=PricedTokens(4.0, 20.0, 0.2),
        or_slug="anthropic/claude-opus-5.5-20260921",
        or_session_cost_10_49_turns=1.4107879666666667,
        or_toks_served=2316510688375,
        or_eff_input_price=0.6881,
        or_eff_output_price=20.0,
    ),
    Model(
        "Anthropic",
        "Claude Opus 5.5 (max)",
        57.6223698102963,
        ProviderType.DATACENTER,
        aa_tok_per_task=119166,
        aa_cost_per_task=PricedTokens(
            1.1744624066957075, 2.383320315146325, 2.4242292976790343
        ),
        aa_sticker_price=PricedTokens(4.0, 20.0, 0.2),
        or_slug="anthropic/claude-opus-5.5-20260921",
        or_session_cost_10_49_turns=1.4107879666666667,
        or_toks_served=2316510688375,
        or_eff_input_price=0.6881,
        or_eff_output_price=20.0,
    ),
    Model(
        "InclusionAI",
        "Ling 3.1 Flash",
        41.0906200397391,
        ProviderType.DATACENTER,
        hardware_cost=15300,
        aa_tok_per_task=100671,
        aa_cost_per_task=PricedTokens(
            0.3441050128378794, 0.09060381742459483, 0.550980668121204
        ),
        aa_sticker_price=PricedTokens(0.3, 0.9, 0.06),
        or_slug="inclusionai/ling-3.1-flash-20261002",
        or_toks_served=77359660361,
        # Currently free on OpenRouter; use AA pricing only
        # or_eff_input_price=0.0,
        # or_eff_output_price=0.0,
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
    # the break go into a second panel (see make_hardware_plot)
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
    easter_eggs: tuple[tuple[float, str], ...] = ()  # (x, slogan) shouts in
    # the background of the SVG only. Newlines are kept, to wrap a long
    # slogan instead of letting it run into its neighbour.

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
        xtick_step=0.5,
        xtick_format="$%.2f",
        band_side="bottom",
    ),
    PlotSpec(
        "Intelligence vs. Price per Task (Low Cost)",
        lambda m: m.price_per_task() <= LOW_COST_THRESHOLD,
        "low_cost",
        xtick_step=0.01,
        xtick_format="$%.2f",
        band_side="top",
    ),
    # Spans $0-14 on the price axis, so at the shared 26in width every model
    # under $0.50 piles up in the leftmost 3% of the plot. 1488in puts
    # roughly $0.01 of price per inch -- the same density as the low-cost
    # view, hard-coded so the width never depends on another view's data.
    # The PNG keeps the 26in figure (an inline <img> scales it to the page
    # anyway, and the SVG is what anyone zooms into), and x_pad is tight
    # because 3% of 1000in is 30in of blank space past the last model.
    PlotSpec(
        "Intelligence vs. Price per Task (All Models)",
        lambda _: True,
        "all_models",
        xtick_step=0.02,
        xtick_format="$%.2f",
        fig_w=1000.0,
        png_fig_w=FIG_W,
        png_xtick_step=0.5,  # 0.02 over 26in is 700+ labels on one axis
        x_pad=0.002,  # ~$0.03 past Sonnet 5.5 (max); the PNG needs the
        # default 0.03, where 0.002 would cut that dot in half
        png_x_pad=0.03,
        easter_eggs=(
            (0.25, "Too cheap to care"),
            (0.75, "Need to be careful\nabout your bill"),
            (1.5, "This is EXPENSIVE"),
            (
                3.0,
                "Anything beyond this point\nis unaffordable without\nVC subsidies",
            ),
            (5.0, "Keep scrolling"),
            (9.0, "Are you still here?"),
            (14.0, "Congratulations,\nyou made it to the end!"),
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
    # price. Kimi K3 needs 2 TB of RAM ($320,000, twelve times the next most
    # expensive rig), so x_break splits the axis and gives it a panel of its
    # own instead of squashing the other ten models into the leftmost 8%.
    PlotSpec(
        "Intelligence vs. Hardware price",
        lambda m: m.hardware_cost is not None,
        "local_hardware",
        x_of=lambda m: m.hardware_cost,
        x_label="Hardware price, one-off (USD)",
        xtick_step=2_000,
        band=False,
        x_break=28_000,
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

    "GLM-5.3-Flash (max)" -> "GLM-5.3-Flash", and "Qwen3.8-Flash (Strix Halo
    128GB ⚡)" -> "Qwen3.8-Flash ⚡": the effort suffix and the hardware name
    Model.__post_init__ splices in are dropped, but the ⚡ itself is kept so
    the local-electricity marker still renders (the plots have a legend entry
    for it). A parenthetical that is part of the model's real name -- a release
    date, or any word outside EFFORT_TAGS -- stays.
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


def _split_icon(m, name=None):
    """Split the display glyphs out of a model's label.

    Returns (left, icon, right, strike): the text before the glyph, the icon
    key (None, "bolt", or "mask"), the text after it, and whether the label is
    unavailable (rendered dark grey with a strikethrough, no icon). The icon is
    drawn between the two text halves, so e.g. "(RTX 3090 ⚡)" keeps its
    parentheses. Only the ⚡ still lives in the name (Model.__post_init__ splices
    it in); the thief mask comes from Model.trains_on_your_data and the
    strikethrough from available=False, so no marker text is carried in the data.

    `name` overrides the display text (e.g. with an effort suffix removed).
    """
    if name is None:
        name = m.name
    strike = not m.available
    idx = name.find("⚡")
    if idx >= 0:
        left = name[:idx].rstrip()
        right = name[idx + len("⚡") :].lstrip()
        return left, "bolt", right, strike
    if m.trains_on_your_data:
        return name, "mask", "", strike
    return name, None, "", strike


def _pad(bb, pad=PAD_PX):
    if hasattr(bb, "x0"):  # Bbox
        bb = (bb.x0, bb.y0, bb.x1, bb.y1)
    return (bb[0] - pad, bb[1] - pad, bb[2] + pad, bb[3] + pad)


def _overlap_area(a, b):
    dx = min(a[2], b[2]) - max(a[0], b[0])
    dy = min(a[3], b[3]) - max(a[1], b[1])
    return dx * dy if dx > 0 and dy > 0 else 0.0


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


def place_labels(ax, fig, points, marker_r_px, extra_obstacles=()):
    """points: [(left, right, icon, strike, x, y)] in data coords -- icon is
    None, "bolt", or "mask"; the icon is drawn as a colored vector marker
    between the left and right text halves, so e.g. "(RTX 3090 ⚡)" keeps its
    parens. strike=True (available=False) renders the text dark grey
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

    def crowding(i):
        xi, yi = disp[i]
        return sum(
            1
            for j, (xj, yj) in enumerate(disp)
            if j != i and abs(xi - xj) < CROWD_X and abs(yi - yj) < CROWD_Y
        )

    crowd = [crowding(i) for i in range(len(points))]
    order = sorted(range(len(points)), key=lambda i: (-crowd[i], -points[i][5]))

    def choose(i, obstacles, leaders):
        """Pick the best candidate position for label i. Returns
        (score, bbox, left_x0, vc, icon_cx, right_x0, w_right, leader); the
        bbox, positions and leader segment are display pixels."""
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

            px, py = disp[i]
            anchor_x = max(bb[0], min(px, bb[2]))
            anchor_y = max(bb[1], min(py, bb[3]))
            dist = math.hypot(anchor_x - px, anchor_y - py)
            leader = None
            if crowd[i] or dist > marker_r_px + LEADER_MIN:
                if dist > 1e-6:
                    ux, uy = (anchor_x - px) / dist, (anchor_y - py) / dist
                    start_x = px + ux * marker_r_px
                    start_y = py + uy * marker_r_px
                else:
                    start_x, start_y = px, py
                leader = (start_x, start_y, anchor_x, anchor_y)
            crossings = sum(
                1
                for other in leaders
                if leader is not None and _segments_cross(leader, other)
            )
            score = (
                spill,
                dot_pen > 0,
                dot_pen,
                crossings > 0,
                crossings,
                pen,
            )
            _dbg = os.environ.get("LBLDBG", "")
            if _dbg and _dbg in left:
                print(
                    f"DBG {left[:30]!r} dx={dx} dy={dy} {ha}/{va} "
                    f"spill={int(spill)} dot={dot_pen:.0f} pen={pen:.0f} "
                    f"cross={crossings}"
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
            if not any(score):  # collision-free position
                return result
            if best is None or score < best[0]:
                best = result
        return best

    # Round 0: greedy sequential pass (already-placed labels and leader lines
    # are obstacles).
    placed = [None] * len(points)
    obstacles = list(extra_obstacles)
    leaders = []
    for i in order:
        b = choose(i, obstacles, leaders)
        placed[i] = b
        obstacles.append(b[1])
        if b[7] is not None:
            leaders.append(b[7])

    def _total_cost(state):
        """(leader crossings, unpadded label-overlap area) of a full layout,
        in the same priority order as choose()'s score."""
        area = 0.0
        crosses = 0
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
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
        return (crosses, area)

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
            obs = list(extra_obstacles) + [placed[j][1] for j in others]
            other_leaders = [placed[j][7] for j in others if placed[j][7] is not None]
            placed[i] = choose(i, obs, other_leaders)
        label_collisions = any(
            _overlap_area(placed[i][1], placed[j][1]) > 4 * PAD_PX * PAD_PX
            for i in range(len(points))
            for j in range(i + 1, len(points))
        )
        leader_collisions = any(
            _segments_cross(placed[i][7], placed[j][7])
            for i in range(len(points))
            for j in range(i + 1, len(points))
            if placed[i][7] is not None and placed[j][7] is not None
        )
        if not label_collisions and not leader_collisions:
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

    # Diagnostics toggle, set by --verbose in main().
    if VERBOSE:
        # Print pairwise overlaps between the *unpadded* label extents, in
        # display px. Padded boxes may legally overlap by up to PAD_PX on
        # each side; that is not a real collision.
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                title = ax.get_title(loc="left") or ax.get_title() or "?"
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
    inv = ax.transData.inverted()
    for i in order:
        left, right, icon, strike, x, y = points[i]
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


def _plot_legend(ax, models, loc="lower right", bbox_to_anchor=None):
    """Legend for the publishers present in a plot, plus an entry for every
    special marker in use (⚡ local electricity, thief mask for
    trains_on_your_data, hollow dot for estimated points, strikethrough for
    available=False).
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
    have_unavailable = any(not m.available for m in models)
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


def pareto_frontier(models, x_of):
    """The Pareto staircase: running max of intelligence over ascending x,
    as two parallel lists ready for ax.plot.

    Computed over ALL models, not one plot's filtered view. Each plot is a
    zoom of the same frontier; the axes clip the line, so on the zoomed plots
    it runs off the edge ("continues"), while on the all-models plot it ends at
    the frontier's true last step. Models flagged trains_on_your_data are excluded:
    they are the same offers at providers that train
    on your data, not separate models. Models that are not publicly available
    (available=False) are excluded too: their cost is an estimate, not a real
    offer. Models the x statistic cannot place are skipped.
    """
    pts = sorted(
        (
            (x, m.intelligence)
            for m in models
            if not m.trains_on_your_data and m.available and (x := x_of(m)) is not None
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
        pareto_frontier(MODELS, spec.x_of) if spec.frontier else ([], [])
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
    # y_lim: floor/ceil of the points with 0.2 of slack, except on the band
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
    legend, have_unavailable = _plot_legend(ax, models)

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
            for left, icon, right, strike in [_split_icon(m)]
        ],
        marker_r_px,
        extra_obstacles=(legend_box,),
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
RIGHT_PANEL_RATIO = 4.5  # width ratio left:right -- Kimi K3 gets a narrow panel


def make_hardware_plot(spec, models, y_lim):
    """Hardware price vs intelligence, on a broken x axis.

    The per-task price plots treat hardware as free, which is defensible only
    for machines nobody buys for AI alone. The larger local models break that
    assumption: the cheapest rig that runs them is $250 to $27,200, and Kimi
    K3 needs 2 TB of RAM at $320,000 -- twelve times the next most expensive
    model, and 1,280x the cheapest. On a single linear axis Kimi K3 would
    stretch the scale so far that every other model piles up in the leftmost
    few percent and no label could be placed, so the points at or above
    spec.x_break get their own panel, with a white gap and break slashes
    between the two.
    """
    assert spec.x_break is not None
    left = [m for m in models if m.hardware_cost < spec.x_break]
    right = [m for m in models if m.hardware_cost >= spec.x_break]
    if not left or not right:
        raise ValueError(
            f"{spec.stem}: x_break={spec.x_break} needs models on both sides"
        )

    fig, (ax_l, ax_r) = plt.subplots(
        1,
        2,
        figsize=(FIG_W, FIG_H),
        dpi=DPI,
        sharey=True,
        gridspec_kw={"width_ratios": [RIGHT_PANEL_RATIO, 1]},
    )

    # Faint dotted Pareto frontier over the unbroken prices only: Kimi K3 sits
    # left of nothing, it is dominated by MiMo-V2.6-Pro (more intelligence for
    # 12x less hardware), so the staircase ends in the left panel and the
    # right panel gets a bare dot.
    if spec.frontier:
        frontier_x, frontier_y = pareto_frontier(left, spec.x_of)
        if len(frontier_x) > 1:
            ax_l.plot(
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

    # Axes limits, set before placing labels (placement works in pixels).
    left_hi = max(m.hardware_cost for m in left)
    ax_l.set_xlim(0, math.ceil(left_hi * 1.06 / spec.xtick_step) * spec.xtick_step)
    r_lo = min(m.hardware_cost for m in right)
    r_hi = max(m.hardware_cost for m in right)
    r_pad = 0.18 * (r_hi - r_lo) if r_hi > r_lo else 0.18 * r_hi
    ax_r.set_xlim(r_lo - r_pad, r_hi + r_pad)
    y_lo, y_hi = y_lim
    ax_l.set_ylim(y_lo, y_hi)
    ax_l.yaxis.set_major_locator(MultipleLocator(1))

    # Thousands separators: the right panel's prices are six figures.
    money = StrMethodFormatter("${x:,.0f}")
    ax_l.xaxis.set_major_locator(MultipleLocator(spec.xtick_step))
    ax_l.xaxis.set_major_formatter(money)
    # The right panel is a fraction of the figure wide, so it gets one tick
    # per model instead of a $20k step: six-figure labels would collide.
    ax_r.set_xticks(sorted({m.hardware_cost for m in right}))
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
    legend, have_unavailable = _plot_legend(ax_l, models, loc="lower right")
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
                _strike_text(ax_l, text, renderer, zorder=6)  # above the legend frame
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
                for left_, icon, right_, strike in [_split_icon(m, _clean_label(m))]
            ],
            marker_r_px,
            extra_obstacles=(_pad(legend.get_window_extent(renderer)),)
            if ax is ax_l
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
        left, icon, right, _strike = _split_icon(m, name)
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
        ax, models, loc="center left", bbox_to_anchor=(1.005, 0.5)
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
        y_lo = math.floor(min(m.intelligence for m in models)) - 0.2
        y_hi = math.ceil(max(m.intelligence for m in models)) + 0.2
        if y_hi == y_lo:  # degenerate: all points on one level
            y_hi = y_lo + 1
        if spec.band_side == "bottom":
            y_lo = band[0]
        elif spec.band_side == "top":
            y_hi = band[1]
        if spec.x_break is not None:  # two panels, not one
            print("Generating hardware plot:", spec.stem)
            make_hardware_plot(spec, models, (y_lo, y_hi))
            continue
        print("Generating datacenter plot:", spec.stem)
        make_datacenter_plots(spec, models, band if spec.band else None, (y_lo, y_hi))


if __name__ == "__main__":
    main()
