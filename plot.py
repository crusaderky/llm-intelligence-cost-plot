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
Index task. Datacenter models are priced from what OpenRouter's customers
actually pay: the median cost of a real agentic session of 10-49 turns (the
"core" bucket on OR's session-cost leaderboard, averaged over the coding
harnesses that carry the model), converted to a per-task figure by a
tokens-per-session constant K:

    K = volume-weighted mean of (AA tokens/task x OR session $ / AA $/task)
    price per task = AA tokens/task x OR session $ / K

AA's cost per task and OR's session cost share the model's real $/token, so
their ratio estimates how many tokens a session burns; calibrating K on AA's
prices pins the volume-weighted average of price-per-task/AA-price to 1. The
plot thus keeps AA's overall dollar level while taking its relative shape
from real spending: a model pricier than OpenRouter's average token plots
above AA's sticker price, and vice versa. A datacenter model with no OR
session data falls back to AA's cost per task, unscaled.

Local (electricity-powered) models have no sticker price: their price per
task is the electricity needed to generate AA's output tokens per task on
consumer hardware.

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
from typing import NamedTuple

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
from matplotlib.ticker import FormatStrFormatter, MultipleLocator

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
    models must have a measured tok_per_sec, DATACENTER models must have
    AA's aa_price_per_task.
    """

    publisher: str
    name: str
    intelligence: float
    provider_type: ProviderType
    # AA benchmark task size (also the tok_per_task input of the electricity
    # cost).
    aa_tok_per_task: float = field(kw_only=True)
    # Datacenter models: AA's posted-price cost of one task, plus the
    # OpenRouter numbers.
    aa_price_per_task: float | None = field(default=None, kw_only=True)
    or_slug: str | None = field(default=None, kw_only=True)
    or_session_cost_10_49_turns: float | None = field(default=None, kw_only=True)
    or_toks_served: int | None = field(default=None, kw_only=True)
    # Local models: decode speed measured on `hardware` (None = RTX3090).
    tok_per_sec: float | None = field(default=None, kw_only=True)
    hardware: LocalHardware | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        if self.provider_type is ProviderType.LOCAL:
            if self.tok_per_sec is None:
                raise ValueError(f"{self.name}: local model needs tok_per_sec")
            # The ⚡ marker (ICON_PATHS["bolt"]) and the hardware name are
            # part of the label, exactly like the explicit [TRAIN] markers.
            if self.hardware is None:
                self.hardware = RTX3090
            self.name = f"{self.name} ({self.hardware.name} ⚡)"
        elif self.aa_price_per_task is None:
            raise ValueError(f"{self.name}: datacenter model needs aa_price_per_task")

    def price_per_task(self) -> float:
        """Displayed USD cost of one AA Intelligence Index task.

        Datacenter models: OR's median 10-49-turn session cost converted to
        a per-task figure by or_tokens_per_session() -- equivalently, AA's
        cost per task times the model's price-level ratio (verbosity x
        session cost / AA cost per task) over the volume-weighted average; a
        model with no OR session data keeps AA's figure unscaled.

        Local models: the electricity to generate AA's output tokens per
        task on `hardware`.
        """
        if self.provider_type is ProviderType.LOCAL:
            assert self.tok_per_sec is not None
            assert self.hardware is not None
            # Weighted by population, May 2026 (USD/KWh)
            # https://www.eia.gov/electricity/monthly/epm_table_grapher.php?t=epmt_5_6_a
            US_ELECTRICITY_PRICE = 0.2049
            sec_per_task = self.aa_tok_per_task / self.tok_per_sec
            power_draw = self.hardware.peak_power_draw - self.hardware.idle_power_draw
            kwh_per_task = power_draw * sec_per_task / 3_600_000
            # Finger-in-the-air overhead to account for prefill
            PREFILL_OVERHEAD = 1.2
            return kwh_per_task * US_ELECTRICITY_PRICE * PREFILL_OVERHEAD

        else:
            assert self.aa_price_per_task is not None
            if self.or_session_cost_10_49_turns is None:
                return self.aa_price_per_task
            # AA's cost per task cancels out of AA $/task x ratio / K; it is
            # kept in the ratio because that is the price-level statistic:
            # the model's real $/token relative to the volume-weighted mean.
            ratio = (
                self.aa_tok_per_task
                * self.or_session_cost_10_49_turns
                / self.aa_price_per_task
            )
            return self.aa_price_per_task * ratio / or_tokens_per_session()

    def price_delta(self) -> float | None:
        """Percentage change of the displayed price per task from AA's
        published price; None for local models, which have no published
        price to compare with.
        """
        if self.aa_price_per_task is None:
            return None
        return 100 * (self.price_per_task() / self.aa_price_per_task - 1)

    @property
    def trains_on_your_data(self) -> bool:
        return "[TRAIN]" in self.name

    @property
    def not_publicly_available(self) -> bool:
        return "[UNAVAILABLE]" in self.name


MODELS = [
    # Price per task is derived by Model.price_per_task(). Stored inputs:
    # - intelligence + aa_tok_per_task: artificialanalysis.ai (AA Data API +
    #   model page flight payloads, sub-unit precision).
    # - datacenter: aa_price_per_task (AA's posted-price cost of one task),
    #   or_slug / or_session_cost_10_49_turns (openrouter.ai, GET
    #   /api/frontend/v1/rankings/session-cost: median 10-49-turn session
    #   cost, averaged across the coding harnesses that carry the model) and
    #   or_toks_served (GET /api/frontend/v1/rankings/models?view=week:
    #   prompt + completion tokens served in the trailing week, all variants).
    # - local: tok_per_sec measured on real hardware.
    # See .agents/skills/refresh-models for how to re-fetch these numbers.
    # --- Local models (price per task = electricity) ---
    Model(
        "OpenBMB",
        "MiniCPM5-2B",
        12.4634,
        ProviderType.LOCAL,
        aa_tok_per_task=21834,
        tok_per_sec=180,
    ),
    Model(
        "Alibaba",
        "Qwen3.6-35B-A3B",
        18.2290,
        ProviderType.LOCAL,
        aa_tok_per_task=34594,
        tok_per_sec=150,
    ),
    Model(
        "Meta",
        "Muse Glimmer",
        17.4754,
        ProviderType.LOCAL,
        aa_tok_per_task=13925,
        tok_per_sec=124,
    ),
    Model(
        "Institute of Foundation Models",
        "K2 Horizon 7B",
        20.5959,
        ProviderType.LOCAL,
        aa_tok_per_task=72163,
        tok_per_sec=100,
    ),
    Model(
        "Alibaba",
        "Qwen3.8-27B (Non-reasoning)",
        20.1502,
        ProviderType.LOCAL,
        aa_tok_per_task=29975,
        tok_per_sec=46,
    ),
    Model(
        "Alibaba",
        "Qwen3.8-27B (low)",
        26.2048,
        ProviderType.LOCAL,
        aa_tok_per_task=45426,
        tok_per_sec=46,
    ),
    Model(
        "Alibaba",
        "Qwen3.8-27B (medium)",
        27.5508,
        ProviderType.LOCAL,
        aa_tok_per_task=51943,
        tok_per_sec=46,
    ),
    Model(
        "Alibaba",
        "Qwen3.8-27B (xhigh)",
        33.6963,
        ProviderType.LOCAL,
        aa_tok_per_task=66797,
        tok_per_sec=46,
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
        tok_per_sec=73,
    ),
    Model(
        "Alibaba",
        "Qwen3.8-Flash-Next",
        39.8223,
        ProviderType.LOCAL,
        aa_tok_per_task=107885,
        tok_per_sec=25,
        hardware=STRIX_HALO,
    ),
    # --- Datacenter models (price per task = AA's cost per task rescaled by
    # OR real-world usage) ---
    Model(
        "Alibaba",
        "Qwen3.8 Max (0902)",
        45.4152,
        ProviderType.DATACENTER,
        aa_tok_per_task=107730,
        aa_price_per_task=5.408509428374016,
        or_slug="qwen/qwen3.8-max-20260902",
        or_session_cost_10_49_turns=0.80500755,
        or_toks_served=400624673646,
    ),
    Model(
        "DeepSeek",
        "DeepSeek V4.1 Flash",
        39.4562,
        ProviderType.DATACENTER,
        aa_tok_per_task=88574,
        aa_price_per_task=0.26522527009606844,
        or_slug="deepseek/deepseek-v4.1-flash-20260910",
        or_session_cost_10_49_turns=0.06962802900000001,
        or_toks_served=17834860326691,
    ),
    Model(
        "Tencent",
        "Hy3",
        25.2973,
        ProviderType.DATACENTER,
        aa_tok_per_task=46161,
        aa_price_per_task=0.07430984031498059,
        or_slug="tencent/hy3-20260706",
        or_session_cost_10_49_turns=0.047079116,
        or_toks_served=4207353041359,
    ),
    Model(
        "Meta",
        "Muse Spark 1.3",
        48.0923,
        ProviderType.DATACENTER,
        aa_tok_per_task=60200,
        aa_price_per_task=1.6048932100125866,
        or_slug="meta/muse-spark-1.3-20260902",
        or_session_cost_10_49_turns=0.5541186,
        or_toks_served=273275622851,
    ),
    Model(
        "Meta",
        "Muse Spark 1.3 [TRAIN]",
        48.0923,
        ProviderType.DATACENTER,
        aa_tok_per_task=60200,
        aa_price_per_task=1.6048932100125866,
        or_slug="meta/muse-spark-1.3-contributor-20260902",
        or_session_cost_10_49_turns=0.025736046749999998,
        or_toks_served=2183144317956,
    ),
    Model(
        "Z AI",
        "GLM-5.3-Flash (high)",
        # scaled from the (max) record, see the README note
        41.8075 * 28.01 / 28.99,
        ProviderType.DATACENTER,
        aa_tok_per_task=round(68673 * 70610 / 138690),
        aa_price_per_task=0.2532595604307378 * 70610 / 138690,
        or_slug="z-ai/glm-5.3-flash-20260826",
        or_session_cost_10_49_turns=0.03686962075,
        or_toks_served=18427117932787,
    ),
    Model(
        "Z AI",
        "GLM-5.3-Flash (max)",
        41.8075,
        ProviderType.DATACENTER,
        aa_tok_per_task=68673,
        aa_price_per_task=0.2532595604307378,
        or_slug="z-ai/glm-5.3-flash-20260826",
        or_session_cost_10_49_turns=0.03686962075,
        or_toks_served=18427117932787,
    ),
    Model(
        "Z AI",
        "GLM-5.3",
        44.7774,
        ProviderType.DATACENTER,
        aa_tok_per_task=71128,
        aa_price_per_task=2.0056375150449584,
        or_slug="z-ai/glm-5.3-20260816",
        or_session_cost_10_49_turns=0.46428977,
        or_toks_served=3141624698164,
    ),
    Model(
        "Moonshot AI",
        "Kimi K3",
        43.5938,
        ProviderType.DATACENTER,
        aa_tok_per_task=48455,
        aa_price_per_task=2.0001323004425493,
        or_slug="moonshotai/kimi-k3-20260715",
        or_session_cost_10_49_turns=0.7604998125,
        or_toks_served=1454487527071,
    ),
    Model(
        "Google",
        "Gemini 3.8 Flash",
        40.9262,
        ProviderType.DATACENTER,
        aa_tok_per_task=71003,
        aa_price_per_task=1.2427947606950427,
        or_slug="google/gemini-3.8-flash-20260902",
        or_session_cost_10_49_turns=0.2633476625,
        or_toks_served=2221665595290,
    ),
    Model(
        "SpaceXAI",
        "Grok 4.7",
        46.4465506302286,
        ProviderType.DATACENTER,
        aa_tok_per_task=80561,
        aa_price_per_task=3.738325952106939,
        or_slug="x-ai/grok-4.7-20260916",
        or_toks_served=70052675398,
    ), 
    # Expected to land on OpenRouter on 2026-10-15
    Model(
        "StepFun",
        "Step 5 Preview",
        43.7343049141614,
        ProviderType.DATACENTER,
        aa_tok_per_task=63974,
        aa_price_per_task=0.715523357458521,
    ),  
    # not on OpenRouter: no permaslug, so the price is AA's cost per task, unscaled
    # Model(
    #     "Xiaomi", "MiMo-V2.6-Flash", 0.0, ProviderType.DATACENTER,
    #     aa_tok_per_task=0.0, or_slug="xiaomi/mimo-v2.6-flash-20260921",
    # ),
    Model(
        "Xiaomi",
        "MiMo-V2.6-Pro",
        46.3242065310383,
        ProviderType.DATACENTER,
        aa_tok_per_task=64276,
        aa_price_per_task=0.13322318937213493,
        or_slug="xiaomi/mimo-v2.6-pro-20260921",
        or_toks_served=201796558953,
    ),
    Model(
        "OpenAI",
        "GPT-6 Luna (low)",
        20.9225480080866,
        ProviderType.DATACENTER,
        aa_tok_per_task=2054,
        aa_price_per_task=0.004483809259539013,
        or_slug="openai/gpt-6-luna-20260922",
        or_toks_served=72935760636,
    ),
    Model(
        "OpenAI",
        "GPT-6 Luna (medium)",
        29.4619523515762,
        ProviderType.DATACENTER,
        aa_tok_per_task=11227,
        aa_price_per_task=0.01725359419426588,
        or_slug="openai/gpt-6-luna-20260922",
        or_toks_served=72935760636,
    ),
    Model(
        "OpenAI",
        "GPT-6 Luna (high)",
        32.1482304150837,
        ProviderType.DATACENTER,
        aa_tok_per_task=19771,
        aa_price_per_task=0.02861964876935845,
        or_slug="openai/gpt-6-luna-20260922",
        or_toks_served=72935760636,
    ),
    Model(
        "OpenAI",
        "GPT-6 Luna (xhigh)",
        33.8845820217785,
        ProviderType.DATACENTER,
        aa_tok_per_task=27189,
        aa_price_per_task=0.041708771594451556,
        or_slug="openai/gpt-6-luna-20260922",
        or_toks_served=72935760636,
    ),
    Model(
        "OpenAI",
        "GPT-6 Luna (max)",
        37.2559686869738,
        ProviderType.DATACENTER,
        aa_tok_per_task=50537,
        aa_price_per_task=0.06809498628701058,
        or_slug="openai/gpt-6-luna-20260922",
        or_toks_served=72935760636,
    ),
    Model(
        "OpenAI",
        "GPT-6 Sol (low)",
        33.9008505585733,
        ProviderType.DATACENTER,
        aa_tok_per_task=3358,
        aa_price_per_task=0.13224085088798104,
        or_slug="openai/gpt-6-sol-20260922",
        or_toks_served=18699869899,
    ),
    Model(
        "OpenAI",
        "GPT-6 Sol (medium)",
        39.7820971749522,
        ProviderType.DATACENTER,
        aa_tok_per_task=6478,
        aa_price_per_task=0.24820209578663968,
        or_slug="openai/gpt-6-sol-20260922",
        or_toks_served=18699869899,
    ),
    Model(
        "OpenAI",
        "GPT-6 Sol (high)",
        42.8215513642985,
        ProviderType.DATACENTER,
        aa_tok_per_task=10232,
        aa_price_per_task=0.3746326907148203,
        or_slug="openai/gpt-6-sol-20260922",
        or_toks_served=18699869899,
    ),
    Model(
        "OpenAI",
        "GPT-6 Sol (xhigh)",
        44.1009705516407,
        ProviderType.DATACENTER,
        aa_tok_per_task=16013,
        aa_price_per_task=0.5318962692390143,
        or_slug="openai/gpt-6-sol-20260922",
        or_toks_served=18699869899,
    ),
    Model(
        "OpenAI",
        "GPT-6 Sol (max)",
        47.5276426437724,
        ProviderType.DATACENTER,
        aa_tok_per_task=31238,
        aa_price_per_task=1.0564240894076389,
        or_slug="openai/gpt-6-sol-20260922",
        or_toks_served=18699869899,
    ),
    Model(
        "OpenAI",
        "GPT-6 Astra (low)",
        45.7819243120341,
        ProviderType.DATACENTER,
        aa_tok_per_task=4433,
        aa_price_per_task=0.8175139285656057,
        or_slug="openai/gpt-6-astra-20260903",
        or_session_cost_10_49_turns=2.9237528,
        or_toks_served=1849344485783,
    ),
    Model(
        "OpenAI",
        "GPT-6 Astra (medium)",
        49.5704363034369,
        ProviderType.DATACENTER,
        aa_tok_per_task=9590,
        aa_price_per_task=1.5406493220021167,
        or_slug="openai/gpt-6-astra-20260903",
        or_session_cost_10_49_turns=2.9237528,
        or_toks_served=1849344485783,
    ),
    Model(
        "OpenAI",
        "GPT-6 Astra (high)",
        50.9191471636152,
        ProviderType.DATACENTER,
        aa_tok_per_task=11813,
        aa_price_per_task=1.7252530861048456,
        or_slug="openai/gpt-6-astra-20260903",
        or_session_cost_10_49_turns=2.9237528,
        or_toks_served=1849344485783,
    ),
    Model(
        "OpenAI",
        "GPT-6 Astra (xhigh)",
        52.3863277108782,
        ProviderType.DATACENTER,
        aa_tok_per_task=16901,
        aa_price_per_task=2.308795912269076,
        or_slug="openai/gpt-6-astra-20260903",
        or_session_cost_10_49_turns=2.9237528,
        or_toks_served=1849344485783,
    ),
    Model(
        "OpenAI",
        "GPT-6 Astra (max)",
        52.6737,
        ProviderType.DATACENTER,
        aa_tok_per_task=27206,
        aa_price_per_task=3.2575003134834164,
        or_slug="openai/gpt-6-astra-20260903",
        or_session_cost_10_49_turns=2.9237528,
        or_toks_served=1849344485783,
    ),
    Model(
        "Anthropic",
        "Claude Sonnet 5",
        38.1639,
        ProviderType.DATACENTER,
        aa_tok_per_task=117787,
        aa_price_per_task=5.091163584815694,
        or_slug="anthropic/claude-sonnet-5-20260630",
        or_session_cost_10_49_turns=0.6632285433333334,
        or_toks_served=1490383785036,
    ),
    Model(
        "Anthropic",
        "Claude Opus 5.5 (low)",
        42.3077781466168,
        ProviderType.DATACENTER,
        aa_tok_per_task=10151,
        aa_price_per_task=0.551180473909146,
        or_slug="anthropic/claude-opus-5.5-20260921",
        or_toks_served=27634285008,
    ),
    Model(
        "Anthropic",
        "Claude Opus 5.5 (medium)",
        51.2434931792768,
        ProviderType.DATACENTER,
        aa_tok_per_task=25745,
        aa_price_per_task=1.3360093438976588,
        or_slug="anthropic/claude-opus-5.5-20260921",
        or_toks_served=27634285008,
    ),
    Model(
        "Anthropic",
        "Claude Opus 5.5 (high)",
        53.5831959822067,
        ProviderType.DATACENTER,
        aa_tok_per_task=35584,
        aa_price_per_task=1.822504718771704,
        or_slug="anthropic/claude-opus-5.5-20260921",
        or_toks_served=27634285008,
    ),
    Model(
        "Anthropic",
        "Claude Opus 5.5 (xhigh)",
        55.9873505840139,
        ProviderType.DATACENTER,
        aa_tok_per_task=65667,
        aa_price_per_task=3.459110175822289,
        or_slug="anthropic/claude-opus-5.5-20260921",
        or_toks_served=27634285008,
    ),
    Model(
        "Anthropic",
        "Claude Opus 5.5 (max)",
        57.6223698102963,
        ProviderType.DATACENTER,
        aa_tok_per_task=119166,
        aa_price_per_task=5.982012019521066,
        or_slug="anthropic/claude-opus-5.5-20260921",
        or_toks_served=27634285008,
    ),
]


def or_tokens_per_session() -> float:
    """Volume-weighted estimate of the tokens OpenRouter serves per
    10-49-turn session.

    AA's cost per task and OR's session cost share the model's real $/token,
    so `AA tokens/task x OR session $ / AA $/task` estimates the tokens a
    session burns. The volume-weighted mean of that ratio (weighted by
    or_toks_served) is the K that converts OR's $/session into $/task:
    calibrating it on AA's prices makes the volume-weighted mean of
    price-per-task/AA-price equal 1. Models missing either OR figure do not
    contribute.
    """
    weighted = [
        (
            m.or_toks_served,
            m.aa_tok_per_task * m.or_session_cost_10_49_turns / m.aa_price_per_task,
        )
        for m in MODELS
        if m.or_toks_served is not None and m.or_session_cost_10_49_turns is not None
    ]
    if not weighted:
        raise ValueError("No model has OpenRouter session data")
    return sum(w * v for w, v in weighted) / sum(w for w, _ in weighted)


# Bottom of the high-intelligence plot
HIGH_INTELLIGENCE_THRESHOLD = 33
# Right edge of the green band: the cheap cluster tops out at ~$0.15/task on
# the new scale, and the next most expensive model sits at ~$1.4.
LOW_COST_THRESHOLD = 0.2


class PlotSpec(NamedTuple):
    """One filtered view.

    x_of(m) is the x coordinate, or None for models the view cannot place
    (local models on the delta plot). band_side is the plot edge that
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
    # bar (they share a permaslug, and with it the x statistic)


# The four plots to generate.
PLOTS = [
    PlotSpec(
        "Intelligence vs. Price per Task (High Intelligence)",
        lambda m: m.intelligence >= HIGH_INTELLIGENCE_THRESHOLD,
        "high_intelligence",
        xtick_step=0.2,
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
    PlotSpec(
        "Intelligence vs. Price per Task (All Models)",
        lambda _: True,
        "all_models",
        xtick_step=0.2,
        xtick_format="$%.2f",
    ),
    PlotSpec(
        "Intelligence vs. Δ from AA's Price per Task",
        lambda m: m.price_delta() is not None,
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
]

# --- knobs -----------------------------------------------------------------
FIG_W, FIG_H = 26, 14  # inches
DPI = 100
VERBOSE = False  # set by --verbose: report residual label overlaps to stderr
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

    # Repair rounds: re-place every label, in crowd order, against the latest
    # positions of the others (Gauss-Seidel style). Updating every label
    # against a stale snapshot instead can end with two labels sitting on top
    # of each other, each having dodged where the other used to be. Iterate
    # until no label collides with anything, or give up after a fixed number
    # of rounds (dense clusters may be unsatisfiable).
    for _round in range(40):
        for i in order:
            obs = list(extra_obstacles) + [
                placed[j][1] for j in range(len(points)) if j != i
            ]
            placed[i] = choose(i, obs)
        if not any(
            _overlap_area(placed[i][1], placed[j][1]) > 4 * PAD_PX * PAD_PX
            for i in range(len(points))
            for j in range(i + 1, len(points))
        ):
            break

    # Diagnostics toggle, set by --verbose in main().
    if VERBOSE:
        # Print pairwise overlaps between the *unpadded* label extents, in
        # display px. Padded boxes may legally overlap by up to PAD_PX on
        # each side; that is not a real collision.
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                a = tuple(
                    v + PAD_PX * s_ for v, s_ in zip(placed[i][1], (1, 1, -1, -1))
                )
                b = tuple(
                    v + PAD_PX * s_ for v, s_ in zip(placed[j][1], (1, 1, -1, -1))
                )
                ov = _overlap_area(a, b)
                if ov > 1:
                    print(
                        f"OVERLAP {ov:.0f}px^2: "
                        f"[{points[i][0][:30]}|{points[i][1][:20]}] vs "
                        f"[{points[j][0][:30]}|{points[j][1][:20]}]",
                    )
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


def _plot_legend(ax, models, loc="lower right", bbox_to_anchor=None):
    """Legend for the publishers present in a plot, plus an entry for every
    special marker in use (⚡ local electricity, [TRAIN] thief mask,
    [UNAVAILABLE]). Returns (legend, has_unavailable_entry)."""
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
        loc=loc,
        bbox_to_anchor=bbox_to_anchor,
        fontsize=LABEL_SIZE,
        framealpha=0.9,
        edgecolor="#d0d4da",
        handletextpad=0.6,
        borderpad=0.6,
    )
    return legend, have_unavailable


def make_plot(spec, models, band, y_lim):
    xs = [spec.x_of(m) for m in models]
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

    # Faint dotted Pareto frontier: max intelligence for each cost, computed
    # over ALL models, not this plot's filtered view. Each plot is a zoom of
    # the same frontier; the axes clip the line, so on the zoomed plots it
    # runs off the edge ("continues"), while on the all-models plot it ends
    # at the frontier's true last step. Models that train on your data
    # ([TRAIN] in the name) are excluded: they are the same offers at
    # providers that train on your data, not separate models. Models that are
    # not publicly available ([UNAVAILABLE]) are excluded too: their cost is
    # an estimate, not a real offer. The delta plot's x axis is not a price,
    # so it gets no frontier (spec.frontier is False).
    pts = []
    if spec.frontier:
        pts = sorted(
            (
                (x, m.intelligence)
                for m in MODELS
                if not m.trains_on_your_data
                and not m.not_publicly_available
                and (x := spec.x_of(m)) is not None
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
    x_lo, x_hi = min(xs), max(xs)
    if x_lo < 0:  # signed axis (delta plot): margin on both sides
        pad = 0.05 * (x_hi - x_lo)
        ax.set_xlim(max(x_lo - pad, spec.x_min), x_hi + pad)
    else:
        ax.set_xlim(0, x_hi * 1.03)
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
    ax.xaxis.set_major_locator(MultipleLocator(spec.xtick_step))
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
            for left, icon, right, strike in [_split_icon(m.name)]
        ],
        marker_r_px,
        extra_obstacles=(legend_box,),
    )

    os.makedirs("plots", exist_ok=True)
    # Drop the <dc:date> timestamp so regenerating with unchanged data is a
    # no-op for git.
    fig.savefig(
        f"plots/{spec.stem}.svg",
        format="svg",
        bbox_inches="tight",
        metadata={"Date": None},
    )
    fig.savefig(f"plots/{spec.stem}.png", format="png", bbox_inches="tight")
    plt.close(fig)


def make_bar_plot(spec, models):
    """Horizontal bar version of a PlotSpec: one bar per model, x = x_of(m),
    ordered by intelligence (dumbest at the bottom). The model names are the
    y tick labels, so there is no auto-placed text, no Pareto frontier and no
    green band.

    With one_per_model, a model's effort variants collapse to a single bar:
    they share a permaslug, and with it the scaling factor behind x, so they
    would all draw the same bar; the max-effort row stands in for the rest.
    """
    if spec.one_per_model:
        best: dict[str, Model] = {}
        for m in models:
            key = m.or_slug or m.name
            if key not in best or m.intelligence > best[key].intelligence:
                best[key] = m
        models = list(best.values())
    ordered = sorted(models, key=lambda m: m.intelligence)
    ys = list(range(len(ordered)))
    xs = [spec.x_of(m) for m in ordered]
    colors = [PUBLISHERS[m.publisher] for m in ordered]

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=DPI)
    ax.barh(ys, xs, height=0.7, color=colors, zorder=3)
    ax.set_yticks(ys, [m.name for m in ordered], fontsize=LABEL_SIZE)
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
    ax.tick_params(axis="y", length=0)
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

    os.makedirs("plots", exist_ok=True)
    # Drop the <dc:date> timestamp so regenerating with unchanged data is a
    # no-op for git.
    fig.savefig(
        f"plots/{spec.stem}.svg",
        format="svg",
        bbox_inches="tight",
        metadata={"Date": None},
    )
    fig.savefig(f"plots/{spec.stem}.png", format="png", bbox_inches="tight")
    plt.close(fig)


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
    print(f"OpenRouter tokens/session (K): {or_tokens_per_session():.0f}\n")
    rows: list[tuple[str, str, str, str]] = []
    for m in MODELS:
        price = m.price_per_task()
        if m.provider_type is ProviderType.LOCAL:
            rows.append((m.name, "-", f"{price:.4f}", "electricity"))
        elif m.or_session_cost_10_49_turns is None:
            rows.append((m.name, f"{price:.4f}", f"{price:.4f}", "unscaled"))
        else:
            rows.append(
                (
                    m.name,
                    f"{m.aa_price_per_task:.4f}",
                    f"{price:.4f}",
                    f"{price / m.aa_price_per_task - 1:+.1%}",
                )
            )
    headers = ("model", "AA $/task", "price/task", "vs AA")
    right = (False, True, True, True)
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
        make_plot(spec, models, band if spec.band else None, (y_lo, y_hi))


if __name__ == "__main__":
    main()
