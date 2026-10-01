"""Refresh the AA + OpenRouter numbers behind every model in plot.py.

Reads the MODELS list from plot.py, re-fetches from:
- artificialanalysis.ai (model page payloads): Intelligence Index, output
  tokens per task, cost per task and its token-type breakdown (uncached input
  incl. cache write / cache read / output), and the sticker prices per 1M
  tokens the breakdown was priced at (input / cache read / output).
- openrouter.ai (GET /api/frontend/v1/rankings/session-cost): median cost of
  a 10-49-turn session ("core" bucket) per permaslug, averaged across the
  OR coding harnesses that carry session data for the model. Collected but no
  longer used to price the plot; it stays as a record of real session spend.
- openrouter.ai (GET /api/frontend/v1/stats/effective-pricing): the effective
  input and output price per provider per day over the last week, reduced to
  one synthetic price pair per permaslug (first quintile across providers per
  day, median across days). This is what prices the plot's datacenter models.
- openrouter.ai (GET /api/frontend/v1/rankings/models?view=week): prompt +
  completion tokens served in the trailing week per permaslug, summed across
  variants.

Prints an old -> new table plus paste-ready constructor lines. It never edits
plot.py itself; the agent applies the edits.

Usage (always through the pixi environment):
  pixi r refresh-models                 # report (cached fetches, 6h)
  pixi r refresh-models --refresh       # force re-fetch of everything
  pixi r refresh-models --new OR_SLUG --aa-slug SLUG --aa-name "AA Record Name" \
        --publisher "Pub" --name "Model Name"
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
SKILLS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SKILLS / "aa-lookup" / "scripts"))
sys.path.insert(0, str(REPO_ROOT))

import aa_query

import plot

OR_SESSION_URL = "https://openrouter.ai/api/frontend/v1/rankings/session-cost"
OR_MODELS_URL = "https://openrouter.ai/api/frontend/v1/rankings/models"
OR_EFFECTIVE_URL = "https://openrouter.ai/api/frontend/v1/stats/effective-pricing"
# range=1w is one of OR's EFFECTIVE_PRICING_RANGES; it returns one point per
# day, per endpoint, of the effective (cache-discounted) price per 1M tokens.
OR_EFFECTIVE_RANGE = "1w"
CACHE_PATH = REPO_ROOT / ".cache" / "or_session_cost.json"
EFFECTIVE_CACHE_PATH = REPO_ROOT / ".cache" / "or_effective_pricing.json"
TOKS_CACHE_PATH = REPO_ROOT / ".cache" / "or_toks_served.json"
CACHE_TTL = 6 * 3600
AUTH_PATHS = [
    Path.home() / ".pi" / "agent" / "auth.json",
    Path.home() / ".pi" / "auth.json",
]

# plot.py MODELS base name -> (AA page slug, AA record name).
# One row per DISTINCT AA record; the trains_on_your_data twins and local-vs-datacenter
# pairs of the same model share a row.
AA_LOOKUPS: dict[str, tuple[str, str, bool]] = {
    # base name: (aa_page_slug, aa_record_name, derived_intelligence?)
    "MiniCPM5-2B": ("minicpm5-2b", "MiniCPM5-2B", False),
    "Muse Glimmer": ("muse-glimmer", "Muse Glimmer (High)", False),
    "Qwen3.6-35B-A3B": ("qwen3-6-35b-a3b", "Qwen3.6 35B A3B (Reasoning)", False),
    "Qwen3.8-27B": ("qwen3-8-27b", "Qwen3.8 27B (Xhigh)", False),
    "Ternary-Bonsai-2": ("", "", False),  # not on AA: manual entry
    "Occamy-1.0": ("", "", False),  # not on AA: manual entry
    "Qwen3.8-Flash": ("qwen3-8-flash-next", "Qwen3.8-Flash-Next", False),
    "K2 Horizon 7B": ("k2-horizon-7b", "K2 Horizon 7B", False),
    "Qwen3.8 Max (0902)": ("qwen3-8-max", "Qwen3.8 Max (0902)", False),
    "DeepSeek V4.1 Flash": (
        "deepseek-v4-1-flash",
        "DeepSeek V4.1 Flash (Max)",
        False,
    ),
    "Hy3": ("hy3", "Hy3", False),
    "Hy4 preview": ("", "", False),  # not on AA yet: manual entry
    "Muse Spark 1.3": ("muse-spark-1-3", "Muse Spark 1.3 (Max)", False),
    # (high) is extrapolated from the (max) record: intelligence x
    # 28.01/28.99, tokens and price x 70610/138690 (ratios from Z.ai's coding
    # scores and AA's GLM-5.3 effort split, see the README note)
    "GLM-5.3-Flash (high)": ("glm-5-3-flash", "GLM 5.3 Flash", True),
    "GLM-5.3-Flash (max)": ("glm-5-3-flash", "GLM 5.3 Flash", False),
    "GLM-5.3": ("glm-5-3", "GLM-5.3 (Max)", False),
    "Kimi K3": ("kimi-k3", "Kimi K3 (Max)", False),
    "Gemini 3.8 Flash": ("gemini-3-8-flash", "Gemini 3.8 Flash (High)", False),
    # not on OpenRouter yet, so the refresh reports it with no session data
    "Gemini 4 Argon (high)": ("gemini-4-argon", "Gemini 4 Argon (High)", False),
    "Grok 4.7 (high)": ("grok-4-7-high", "Grok 4.7 (High)", False),
    "Grok 4.7 (xhigh)": ("grok-4-7", "Grok 4.7 (Xhigh)", False),
    "MiMo-V2.6-Flash": ("mimo-v2-6-flash", "MiMo-V2.6-Flash", False),
    "MiMo-V2.6-Pro": ("mimo-v2-6-pro", "MiMo-V2.6-Pro", False),
    "GPT-6 Luna (low)": ("gpt-6-luna-low", "GPT-6 Luna (Low)", False),
    "GPT-6 Luna (medium)": ("gpt-6-luna-medium", "GPT-6 Luna (Medium)", False),
    "GPT-6 Luna (high)": ("gpt-6-luna-high", "GPT-6 Luna (High)", False),
    "GPT-6 Luna (xhigh)": ("gpt-6-luna-xhigh", "GPT-6 Luna (Xhigh)", False),
    "GPT-6 Luna (max)": ("gpt-6-luna", "GPT-6 Luna (Max)", False),
    "GPT-6.1 Sol (low)": ("gpt-6-1-sol-low", "GPT-6.1 Sol (Low)", False),
    "GPT-6.1 Sol (medium)": ("gpt-6-1-sol-medium", "GPT-6.1 Sol (Medium)", False),
    "GPT-6.1 Sol (high)": ("gpt-6-1-sol-high", "GPT-6.1 Sol (High)", False),
    "GPT-6.1 Sol (xhigh)": ("gpt-6-1-sol-xhigh", "GPT-6.1 Sol (Xhigh)", False),
    "GPT-6.1 Sol (max)": ("gpt-6-1-sol", "GPT-6.1 Sol (Max)", False),
    "GPT-6 Astra (low)": ("gpt-6-astra-low", "GPT-6 Astra (Low)", False),
    "GPT-6 Astra (medium)": ("gpt-6-astra-medium", "GPT-6 Astra (Medium)", False),
    "GPT-6 Astra (high)": ("gpt-6-astra-high", "GPT-6 Astra (High)", False),
    "GPT-6 Astra (xhigh)": ("gpt-6-astra-xhigh", "GPT-6 Astra (Xhigh)", False),
    "GPT-6 Astra (max)": ("gpt-6-astra", "GPT-6 Astra (Max)", False),
    "Step 5 Preview": ("step-5", "Step 5 Preview", False),
    # AA retired its Low Effort record: it is on no Sonnet page any more, so
    # there is nothing to refresh and the row was dropped from plot.py.
    "Claude Sonnet 5.5 (medium)": (
        "claude-sonnet-5-5-medium",
        "Claude Sonnet 5.5 (Adaptive Reasoning, Medium Effort, Default Fallback)",
        False,
    ),
    "Claude Sonnet 5.5 (high)": (
        "claude-sonnet-5-5-high",
        "Claude Sonnet 5.5 (Adaptive Reasoning, High Effort, Default Fallback)",
        False,
    ),
    "Claude Sonnet 5.5 (xhigh)": (
        "claude-sonnet-5-5-xhigh",
        "Claude Sonnet 5.5 (Adaptive Reasoning, Xhigh Effort, Default Fallback)",
        False,
    ),
    "Claude Sonnet 5.5 (max)": (
        "claude-sonnet-5-5",
        "Claude Sonnet 5.5 (Adaptive Reasoning, Max Effort, Default Fallback)",
        False,
    ),
    "Claude Opus 5.5 (low)": (
        "claude-opus-5-5-low",
        "Claude Opus 5.5 (Adaptive Reasoning, Low Effort, Default Fallback)",
        False,
    ),
    "Claude Opus 5.5 (medium)": (
        "claude-opus-5-5-medium",
        "Claude Opus 5.5 (Adaptive Reasoning, Medium Effort, Default Fallback)",
        False,
    ),
    "Claude Opus 5.5 (high)": (
        "claude-opus-5-5-high",
        "Claude Opus 5.5 (Adaptive Reasoning, High Effort, Default Fallback)",
        False,
    ),
    "Claude Opus 5.5 (xhigh)": (
        "claude-opus-5-5-xhigh",
        "Claude Opus 5.5 (Adaptive Reasoning, Xhigh Effort, Default Fallback)",
        False,
    ),
    "Claude Opus 5.5 (max)": (
        "claude-opus-5-5",
        "Claude Opus 5.5 (Adaptive Reasoning, Max Effort, Default Fallback)",
        False,
    ),
}


def get_or_key() -> str:
    for path in AUTH_PATHS:
        if path.exists():
            auth = json.loads(path.read_text())
            key = (auth.get("openrouter") or {}).get("key")
            if key:
                return key
    sys.exit(
        "No OpenRouter key. Expected 'openrouter.key' in "
        " ~/.pi/agent/auth.json (or ~/.pi/auth.json)."
    )


def _or_get(url: str, params: str = "") -> dict:
    req = urllib.request.Request(
        f"{url}{params}", headers={"Authorization": f"Bearer {get_or_key()}"}
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)


def fetch_or_sessions(
    refresh: bool,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """permaslug -> (average 10-49-turn session cost, per-harness medians).

    OR publishes session costs per coding harness; the plotted statistic is
    the average of the "core" bucket (10-49 turns) medians across the
    harnesses that carry session data for the model.
    """
    harnesses = None
    if not refresh and CACHE_PATH.exists():
        entry = json.loads(CACHE_PATH.read_text())
        if time.time() - entry["fetched_at"] < CACHE_TTL:
            harnesses = entry["harnesses"]
    if harnesses is None:
        data = _or_get(OR_SESSION_URL)["data"]
        harnesses = {
            h["label"]: {
                m["model"]: {p["bucket"]: p["medianUsd"] for p in m["points"]}
                for m in h["models"]
            }
            for h in data["harnesses"]
        }
        CACHE_PATH.parent.mkdir(exist_ok=True)
        CACHE_PATH.write_text(
            json.dumps({"fetched_at": time.time(), "harnesses": harnesses})
        )
    detail: dict[str, dict[str, float]] = {}
    for harness, models in harnesses.items():
        for slug, buckets in models.items():
            if "core" in buckets:
                detail.setdefault(slug, {})[harness] = buckets["core"]
    costs = {slug: sum(v.values()) / len(v) for slug, v in detail.items()}
    return costs, detail


def fetch_or_toks_served(refresh: bool) -> dict[str, int]:
    """permaslug -> prompt + completion tokens served in the trailing week.

    GET /api/frontend/v1/rankings/models?view=week returns one daily row per
    (permaslug, variant) for the last few days; summing them gives the weekly
    volume behind plot.py's or_toks_served. Every variant (standard, batch,
    free) counts as served tokens.
    """
    if not refresh and TOKS_CACHE_PATH.exists():
        entry = json.loads(TOKS_CACHE_PATH.read_text())
        if time.time() - entry["fetched_at"] < CACHE_TTL:
            return entry["toks_served"]
    rows = _or_get(OR_MODELS_URL, "?view=week")["data"]
    toks: dict[str, int] = {}
    for row in rows:
        slug = row["model_permaslug"]
        toks[slug] = (
            toks.get(slug, 0)
            + row["total_prompt_tokens"]
            + row["total_completion_tokens"]
        )
    TOKS_CACHE_PATH.parent.mkdir(exist_ok=True)
    TOKS_CACHE_PATH.write_text(
        json.dumps({"fetched_at": time.time(), "toks_served": toks})
    )
    return toks


@dataclass(frozen=True)
class OrEffectivePrice:
    """One synthetic OR price pair per permaslug, in USD per 1M tokens."""

    input: float
    output: float
    days: int  # daily points the medians came from
    providers: int  # distinct providers seen across those days
    cache_hit_rate: float | None  # OR's token-weighted cache hit rate


def _significant(value: float) -> float:
    """Round to 4 significant digits.

    A trailing-week median of a quintile is a noisy statistic at the 1e-5 level
    (the window rolls by a day every refresh), and plot.py stores whatever it is
    given at full precision. Rounding to 4 significant digits -- ~0.01%, far
    finer than the gap between any two real prices, and far coarser than the
    window noise -- keeps an unchanged model byte-identical across refreshes so
    the plots stay reproducible.
    """
    return float(f"{value:.4g}")


def first_quintile(values: list[float]) -> float:
    """Linear-interpolation 20th percentile (the numpy/statistics default)."""
    s = sorted(values)
    if not s:
        raise ValueError("no values")
    if len(s) == 1:
        return s[0]
    pos = 0.2 * (len(s) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def synthesize_effective(chart: dict) -> OrEffectivePrice | None:
    """Reduce OR's effective-pricing chart to one price pair.

    The chart carries one value per day per *endpoint instance* (a provider
    colocation, e.g. "Sail Research (1)" and "(2)"). Instances are collapsed
    to their provider (endpointRawNames) keeping the cheapest one that day, so
    a provider with five colos does not get five votes; the first quintile is
    then taken across providers on each day and the median across days, which
    is what the plot prices with. None when the chart carries no day at all.
    """
    raw_names = chart.get("endpointRawNames") or {}
    days_in = chart.get("inputChartData") or []
    days_out = chart.get("outputChartData") or []
    per_day_in: list[float] = []
    per_day_out: list[float] = []
    providers: set[str] = set()
    for day_in, day_out in zip(days_in, days_out):
        cheapest_in: dict[str, float] = {}
        cheapest_out: dict[str, float] = {}
        for series, cheapest in (
            (day_in["y"], cheapest_in),
            (day_out["y"], cheapest_out),
        ):
            for endpoint_id, price in series.items():
                provider = raw_names.get(endpoint_id, endpoint_id)
                if provider not in cheapest or price < cheapest[provider]:
                    cheapest[provider] = price
        served = sorted(set(cheapest_in) & set(cheapest_out))
        if not served:
            continue
        providers.update(served)
        per_day_in.append(first_quintile([cheapest_in[p] for p in served]))
        per_day_out.append(first_quintile([cheapest_out[p] for p in served]))
    if not per_day_in:
        return None
    return OrEffectivePrice(
        input=_significant(statistics.median(per_day_in)),
        output=_significant(statistics.median(per_day_out)),
        days=len(per_day_in),
        providers=len(providers),
        cache_hit_rate=chart.get("weightedCacheHitRate"),
    )


def fetch_or_effective(refresh: bool, slugs: list[str]) -> dict[str, OrEffectivePrice]:
    """permaslug -> OrEffectivePrice, from the trailing-week price charts.

    Charts are cached per permaslug so adding one model only fetches one
    chart. A slug OR has no chart for is simply absent.
    """
    cached: dict[str, dict] = {}
    if EFFECTIVE_CACHE_PATH.exists():
        cached = json.loads(EFFECTIVE_CACHE_PATH.read_text()).get("charts", {})
    charts: dict[str, dict] = {}
    for slug in slugs:
        entry = cached.get(slug)
        if not refresh and entry and time.time() - entry["fetched_at"] < CACHE_TTL:
            charts[slug] = entry["chart"]
            continue
        url = (
            f"{OR_EFFECTIVE_URL}?permaslug={urllib.parse.quote(slug)}"
            f"&shape=v7&range={OR_EFFECTIVE_RANGE}"
        )
        payload = _or_get(url)
        if "data" not in payload:
            print(
                f"no OR price chart for {slug}: {payload.get('error')}", file=sys.stderr
            )
            continue
        chart = payload["data"]
        if not chart.get("inputChartData") and slug in cached:
            # An empty trailing week is an OR hiccup (the window can come back
            # empty for a model whose traffic just moved permaslug), not the
            # model turning free: keep the last chart that had days in it.
            print(
                f"OR returned an empty price week for {slug}; keeping the "
                f"previously cached chart",
                file=sys.stderr,
            )
            charts[slug] = cached[slug]["chart"]
            continue
        charts[slug] = chart
        cached[slug] = {"fetched_at": time.time(), "chart": chart}
    EFFECTIVE_CACHE_PATH.parent.mkdir(exist_ok=True)
    EFFECTIVE_CACHE_PATH.write_text(
        json.dumps({"fetched_at": time.time(), "charts": cached})
    )
    return {
        slug: price
        for slug, chart in charts.items()
        if (price := synthesize_effective(chart)) is not None
    }


def _fmt_tokens(t: plot.PricedTokens) -> str:
    """A PricedTokens as paste-ready source, full precision."""
    inner = ", ".join(repr(v) for v in t)
    return f"PricedTokens({inner})"


def _same(a: float | None, b: float | None, tol: float = 1e-9) -> bool:
    """Equality within a relative tolerance.

    plot.py derives AA's cost-per-task total by summing the split, while AA
    computes it in one pass, so the two agree to a few ULP rather than exactly.
    A row whose total only moves in the 16th digit is not a data change.
    """
    if a is None or b is None:
        return a is b
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def aa_splits(rec: dict | None) -> tuple[plot.PricedTokens, plot.PricedTokens] | None:
    """AA page record -> (cost per task split, sticker price per 1M tokens).

    Both are PricedTokens(input_, output, cached_input). The cost split is in
    USD per task: `input_` is AA's nonCacheInput + cacheWrite (everything that
    is not a cache read), `output` is reasoning + answer, `cached_input` is the
    cache read. The sticker split is the per-1M-token price each of those three
    streams was billed at, so dividing cost by price recovers the token counts.
    A model with no cache-read price gets 0.0 in that slot, which means "AA
    reports no cache reads for this model". None when the record or its input
    and output prices are missing.
    """
    if not rec or rec.get("cost_per_task_total") is None:
        return None
    uncached = rec["cost_per_task_nonCacheInput"] + (
        rec["cost_per_task_cacheWrite"] or 0
    )
    cost = plot.PricedTokens(
        uncached, rec["cost_per_task_output"], rec["cost_per_task_cacheRead"]
    )
    sticker = plot.PricedTokens(
        rec.get("price_1m_input"),
        rec.get("price_1m_output"),
        rec.get("price_1m_cache_hit") or 0.0,
    )
    if sticker.input_ is None or sticker.output is None:
        return None
    return cost, sticker


def base_name(name: str) -> str:
    """MODELS display name -> key into AA_LOOKUPS (strip the local hardware suffix
    and the ⚡ marker; trains_on_your_data and available are flags, so unlike the
    old name markers they never reach here)."""
    left = name
    # local names carry "(RTX 3090 ⚡)" / "(Strix Halo 128GB ⚡)"
    if "⚡" in name and "(" in left:
        left = left[: left.rfind("(")].strip()
    return left.strip()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--refresh", action="store_true", help="ignore caches")
    ap.add_argument(
        "--new", metavar="OR_SLUG", help="print a constructor row for a new model"
    )
    ap.add_argument("--aa-slug", help="AA page slug for --new")
    ap.add_argument("--aa-name", help="exact AA record name for --new")
    ap.add_argument("--publisher", help="publisher for --new")
    ap.add_argument("--name", help="display name for --new")
    args = ap.parse_args()

    if args.new:
        assert args.aa_slug and args.aa_name and args.publisher and args.name, (
            "--new needs --aa-slug, --aa-name, --publisher, --name"
        )
        session_costs, _detail = fetch_or_sessions(args.refresh)
        toks_served = fetch_or_toks_served(args.refresh)
        effective = fetch_or_effective(args.refresh, [args.new])
        rec = aa_query.get_page_record(
            {"name": args.aa_name, "slug": args.aa_slug}, args.refresh
        )
        if rec is None:
            sys.exit(f"no AA page record for {args.aa_name!r} at {args.aa_slug}")
        if rec.get("cost_per_task_total") is None:
            sys.exit(f"no AA cost per task for {args.aa_name!r} at {args.aa_slug}")
        splits = aa_splits(rec)
        if splits is None:
            sys.exit(f"no AA sticker prices for {args.aa_name!r} at {args.aa_slug}")
        cost, sticker = splits
        eff = effective.get(args.new)
        if eff is None:
            sys.exit(f"no OR effective-pricing chart for {args.new}")
        sess = session_costs.get(args.new)
        toks = toks_served.get(args.new)
        print("AA record:", json.dumps(rec, indent=1))
        if sess is not None:
            print(
                f"OR {args.new}: ${sess:.6f} average 10-49-turn session cost "
                f"across {len(_detail[args.new])} harness(es) (recorded, not plotted)"
            )
        print(
            f"OR {args.new}: ${eff.input:.6f}/Mtok input, ${eff.output:.6f}/Mtok output "
            f"(first quintile over {eff.providers} providers, median over {eff.days} days)"
        )
        print(f"OR {args.new}: {toks} tokens served in the trailing week")
        print(
            "Model(\n"
            f'    "{args.publisher}",\n'
            f'    "{args.name}",\n'
            f"    {rec['intelligence']:.4f},\n"
            "    ProviderType.DATACENTER,\n"
            f"    aa_tok_per_task={round(rec['output_tokens_per_task'])},\n"
            f"    aa_cost_per_task={_fmt_tokens(cost)},\n"
            f"    aa_sticker_price={_fmt_tokens(sticker)},\n"
            f'    or_slug="{args.new}",\n'
            + (
                f"    or_session_cost_10_49_turns={sess!r},\n"
                f"    or_toks_served={toks!r},\n"
                if sess is not None
                else ""
            )
            + f"    or_eff_input_price={eff.input!r},\n"
            f"    or_eff_output_price={eff.output!r},\n"
            "),"
        )
        return

    session_costs, session_detail = fetch_or_sessions(args.refresh)
    toks_served = fetch_or_toks_served(args.refresh)
    effective = fetch_or_effective(
        args.refresh,
        sorted({m.or_slug for m in plot.MODELS if m.or_slug}),
    )
    print("(OR session $: 10-49-turn median, averaged across the coding harnesses;")
    print(" recorded for reference, no longer used to price the plot)")
    print("(OR week toks: prompt + completion tokens served in the trailing week)")
    print("(OR eff $/M: first quintile of provider effective prices per day,")
    print(" median across days of the trailing week)")
    aa_cache: dict[tuple[str, str], dict | None] = {}
    print(
        f"{'model':30s} {'intelligence':>22s} {'tokens/task':>20s} "
        f"{'AA $/task':>20s} {'OR session $':>28s} {'OR week toks':>20s} "
        f"{'OR in $/M':>19s} {'OR out $/M':>19s}"
    )
    changed = 0
    paste: list[str] = []

    def fmt(v, spec=".4f"):
        return format(v, spec) if v is not None else "-"

    def fmt_vol(v):
        if v is None:
            return "-"
        return f"{v / 1e12:.2f}T" if v >= 1e12 else f"{v / 1e9:.1f}B"

    for m in plot.MODELS:
        key = base_name(m.name)
        if key not in AA_LOOKUPS:
            print(f"!! {m.name}: no AA_LOOKUPS row — add one to refresh_models.py")
            continue
        slug, rec_name, derived = AA_LOOKUPS[key]
        if (slug, rec_name) not in aa_cache:
            aa_cache[(slug, rec_name)] = (
                aa_query.get_page_record({"name": rec_name, "slug": slug}, args.refresh)
                if slug
                else None
            )
        rec = aa_cache[(slug, rec_name)]
        if rec is None and not slug:
            name = plot._pad_cell(m.name[:30], 30)
            print(f"{name}   (not on AA: intelligence + tokens are manual)")
            continue
        if rec is not None and rec.get("_matched_name") not in (None, rec_name):
            print(
                f"!! {m.name}: AA_LOOKUPS name {rec_name!r} is stale, AA calls it "
                f"{rec['_matched_name']!r}"
            )
        new_int = rec["intelligence"] if rec else m.intelligence
        new_tok = (
            round(rec["output_tokens_per_task"])
            if rec and rec.get("output_tokens_per_task")
            else None
        )
        old_int, old_tok = m.intelligence, m.aa_tok_per_task
        splits = aa_splits(rec)
        new_cost, new_sticker = splits if splits else (None, None)
        old_cost = (
            m.aa_cost_per_task
            if isinstance(m.aa_cost_per_task, plot.PricedTokens)
            else None
        )
        old_sticker = m.aa_sticker_price
        if m.provider_type is plot.ProviderType.DATACENTER:
            old_aa = m.aa_total_cost_per_task()
            old_sess = m.or_session_cost_10_49_turns
            old_vol = m.or_toks_served
            new_aa = rec.get("cost_per_task_total") if rec else None
            if m.or_slug in session_costs:
                new_sess = session_costs[m.or_slug]
                src = (
                    f" ({len(session_detail[m.or_slug])}/4 harnesses)"
                    if len(session_detail[m.or_slug]) < 4
                    else ""
                )
            else:
                new_sess, src = None, " (no session data)"
            new_vol = toks_served.get(m.or_slug)
            old_eff_in, old_eff_out = m.or_eff_input_price, m.or_eff_output_price
            eff = effective.get(m.or_slug)
            new_eff_in = eff.input if eff else None
            new_eff_out = eff.output if eff else None
            if eff is None:
                src += " (no OR price chart: plots at AA's price)"
        else:  # local: electricity, no OR fields
            old_aa = old_sess = old_vol = new_aa = new_sess = new_vol = None
            old_eff_in = old_eff_out = new_eff_in = new_eff_out = None
            src = ""

        mark = ""
        if not derived:  # derived rows are extrapolations; the reminder explains
            if new_int is not None and abs(new_int - old_int) > 0.05:
                mark += " INT!"
            if (
                new_tok is not None
                and old_tok
                and abs(new_tok - old_tok) > max(2, old_tok * 0.001)
            ):
                mark += " TOK!"
            # exact: plot.py stores full precision, so any nonzero change means
            # the stored value no longer matches the AA page / OR window
            if new_aa is not None and not _same(new_aa, old_aa):
                mark += " AA$!"
            if new_sess is not None and new_sess != old_sess:
                mark += " OR$!"
            # expected on every refresh: the volume is a rolling window
            if new_vol is not None and new_vol != old_vol:
                mark += " VOL!"
            if new_cost != old_cost or new_sticker != old_sticker:
                mark += " SPL!"
            if (new_eff_in, new_eff_out) != (old_eff_in, old_eff_out):
                mark += " ORP!"
        if mark:
            changed += 1
        print(
            f"{plot._pad_cell(m.name[:30], 30)} {fmt(old_int):>9s} "
            f"-> {fmt(new_int):>9s} "
            f"{fmt(old_tok, '.0f'):>8s} -> {fmt(new_tok, '.0f'):>8s} "
            f"{fmt(old_aa):>8s} -> {fmt(new_aa):>8s} "
            f"{fmt(old_sess, '.6f'):>12s} -> {fmt(new_sess, '.6f'):>12s} "
            f"{fmt_vol(old_vol):>8s} -> {fmt_vol(new_vol):>8s} "
            f"{fmt(old_eff_in, '.6f'):>10s} -> {fmt(new_eff_in, '.6f'):>10s} "
            f"{fmt(old_eff_out, '.6f'):>10s} -> {fmt(new_eff_out, '.6f'):>10s}{mark}{src}"
        )
        if mark:
            lines = []
            if new_int is not None and abs(new_int - old_int) > 0.05:
                # intelligence is the 3rd positional argument, so it cannot be
                # pasted as a kwarg line
                lines.append(
                    f"    # 3rd positional arg (intelligence): "
                    f"{old_int!r} -> {new_int!r}"
                )
            if new_tok is not None and new_tok != old_tok:
                lines.append(f"    aa_tok_per_task={new_tok!r},")
            if new_cost is not None and (new_cost, new_sticker) != (
                old_cost,
                old_sticker,
            ):
                # the split replaces the bare total: it is the same money, itemized
                lines.append(f"    aa_cost_per_task={_fmt_tokens(new_cost)},")
                lines.append(f"    aa_sticker_price={_fmt_tokens(new_sticker)},")
            elif new_aa is not None and not _same(new_aa, old_aa):
                lines.append(f"    aa_cost_per_task={new_aa!r},")
            if (new_eff_in, new_eff_out) != (old_eff_in, old_eff_out):
                lines.append(f"    or_eff_input_price={new_eff_in!r},")
                lines.append(f"    or_eff_output_price={new_eff_out!r},")
            if new_sess is not None and new_sess != old_sess:
                lines.append(f"    or_session_cost_10_49_turns={new_sess!r},")
            if new_vol is not None and new_vol != old_vol:
                lines.append(f"    or_toks_served={new_vol!r},")
            if any(f in mark for f in ("SPL!", "ORP!", "AA$!", "INT!", "TOK!")):
                paste.append(m.name + "\n" + "\n".join(lines))
        if derived:
            print(
                "   (derived: values are extrapolated from the (max) record — "
                "the ratios are historical, see the README note)"
            )
        if m.trains_on_your_data:
            print(
                "   (trains_on_your_data twin of the entry above — keep values in sync)"
            )
        if m.provider_type is plot.ProviderType.LOCAL:
            print(
                "   (local: update intelligence + aa_tok_per_task; "
                "local_speed and hardware stay)"
            )

    if paste:
        print("\n--- paste-ready constructor kwargs ---")
        for block in paste:
            print(block)

    print(
        f"\n{changed} model(s) drifted. Apply the changes to plot.py, then:\n"
        "  pixi r plot  (reprints every displayed price per task; inspect the PNGs!)\n"
        "  pixi r lint\n"
        "Check that LOW_COST_THRESHOLD still catches the cheap cluster and leaves\n"
        "the green band non-empty, and that HIGH_INTELLIGENCE_THRESHOLD still\n"
        "floors the top plot. VOL! and ORP! are expected on every refresh: the\n"
        "weekly serving volume is a rolling window and the OR prices are a\n"
        "trailing-week median, so always apply them."
    )


if __name__ == "__main__":
    main()
