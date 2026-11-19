"""Refresh data.toml from Artificial Analysis and OpenRouter.

Reads data.toml, re-fetches from:
- artificialanalysis.ai (model page payloads): Intelligence Index, output
  tokens per task, cost per task and its token-type breakdown (uncached input
  incl. cache write / cache read / output), and the sticker prices per 1M
  tokens the breakdown was priced at (input / cache read / output).
- openrouter.ai (GET /api/frontend/v1/rankings/session-cost): median cost of
  a 10-49-turn session ("core" bucket) per permaslug, averaged across the
  OR coding harnesses that carry session data for the model. Collected but no
  longer used to price the plot; it stays as a record of real session spend.
- openrouter.ai (GET /api/frontend/v1/stats/effective-pricing): the effective
  input and output price per endpoint instance per day over the last week,
  reduced to one synthetic price pair per permaslug: each instance's weekly
  mean price, weighted by the share of tokens it served. This is what prices
  the plot's datacenter models.
- openrouter.ai (GET /api/frontend/v1/rankings/models?view=week): prompt +
  completion tokens served in the trailing week per permaslug, summed across
  variants.

Prints what changed and writes data.toml back in place. tomlkit keeps every
comment, including the "# computed from:" notes on hand-derived values; those
notes are dropped for a field once the refresh replaces its value. Rows with
aa_derived=true and rows pinned to an empty AA slug keep their AA numbers but
still refresh their OpenRouter fields.

Usage (always through the pixi environment):
  pixi r refresh-models                 # cached fetches (6h)
  pixi r refresh-models --force         # force re-fetch of everything
  pixi r refresh-models --new OR_SLUG --aa-slug SLUG --aa-name "AA Record Name" \
        --publisher "Pub" --name "Model Name"    # print a TOML row
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

import aa_query
import plot
import tomlkit

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "data.toml"

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


def or_slug_variants(slug: str) -> list[str]:
    """The two spellings OpenRouter uses for one dated release.

    The stats endpoints (session cost, weekly tokens served) key on the
    *permaslug*, which dashes the dots inside a version name:
    "google/gemini-3-8-flash-20260902". The effective-pricing chart keys on the
    developer slug plus the date and keeps the dots:
    "google/gemini-3.8-flash-20260902". data.toml stores the readable dotted form
    (the only spelling the price chart answers for), so the stats lookups have to
    try the dashed twin. Only that direction is derivable from the stored string;
    a slug already stored dashed gets no twin, and an undated slug gets none.
    """
    head, _, date = slug.rpartition("-")
    if not (date.isdigit() and len(date) == 8 and "." in head):
        return [slug]
    return [slug, f"{head.replace('.', '-')}-{date}"]


def or_lookup_key(mapping: dict[str, object], slug: str | None) -> str | None:
    """The key in mapping that spellings of slug agree on, or None.

    A row can have no OpenRouter slug at all (a model OR does not carry), which
    is a miss, not a lookup to attempt.
    """
    if not slug:
        return None
    return next((v for v in or_slug_variants(slug) if v in mapping), None)


def or_lookup(mapping: dict[str, object], slug: str | None) -> object:
    """mapping[slug], or mapping[its other spelling], or None."""
    key = or_lookup_key(mapping, slug)
    return None if key is None else mapping[key]


def fetch_or_sessions(
    force: bool,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """permaslug -> (average 10-49-turn session cost, per-harness medians).

    OR publishes session costs per coding harness; the plotted statistic is
    the average of the "core" bucket (10-49 turns) medians across the
    harnesses that carry session data for the model.
    """
    harnesses = None
    if not force and CACHE_PATH.exists():
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


def fetch_or_toks_served(force: bool) -> dict[str, int]:
    """permaslug -> prompt + completion tokens served in the trailing week.

    GET /api/frontend/v1/rankings/models?view=week returns one daily row per
    (permaslug, variant) for the last few days; summing them gives the weekly
    volume behind data.toml's or_toks_served. Every variant (standard, batch,
    free) counts as served tokens.
    """
    if not force and TOKS_CACHE_PATH.exists():
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
    days: int  # daily points the per-endpoint means came from
    providers: int  # distinct providers that carried weight
    cache_hit_rate: float | None  # OR's token-weighted cache hit rate


def _significant(value: float) -> float:
    """Round to 4 significant digits.

    A trailing-week token-share-weighted average is a noisy statistic at the
    1e-5 level (the window rolls by a day every refresh), and data.toml stores
    whatever it is given at full precision. Rounding to 4 significant digits --
    ~0.01%, far finer than the gap between any two real prices, and far coarser
    than the window noise -- keeps an unchanged model byte-identical across
    refreshes so the plots stay reproducible.
    """
    return float(f"{value:.4g}")


def _weekly_means(chart: dict, key: str) -> dict[str, float]:
    """endpoint instance -> mean of its daily effective prices."""
    prices: dict[str, list[float]] = {}
    for day in chart.get(key) or []:
        for endpoint_id, price in day["y"].items():
            prices.setdefault(endpoint_id, []).append(price)
    return {
        endpoint_id: statistics.fmean(values) for endpoint_id, values in prices.items()
    }


def synthesize_effective(chart: dict) -> OrEffectivePrice | None:
    """Reduce OR's effective-pricing chart to one token-share-weighted price pair.

    OR prices per *endpoint instance* (a provider colocation, e.g. "Sail
    Research (1)" and "(2)") and reports, per instance, one effective price per
    day plus the tokens that instance served. Each instance's price is averaged
    over its days, then the instances are averaged again weighted by their
    token share: the price an average token of the model's real traffic cost.

    Free ($0) instances are dropped and the weights renormalized over the rest:
    a free tier is not what a paying user buys. None when that leaves nothing
    (the model is free on OR, so it keeps AA's price) or when the chart carries
    no day at all.
    """
    means_in = _weekly_means(chart, "inputChartData")
    means_out = _weekly_means(chart, "outputChartData")
    raw_names = chart.get("endpointRawNames") or {}
    weight = 0.0
    total_in = 0.0
    total_out = 0.0
    providers: set[str] = set()
    for summary in chart.get("providerSummaries") or []:
        endpoint_id = summary["endpointId"]
        tokens = summary["totalTokens"]
        mean_in = means_in.get(endpoint_id)
        mean_out = means_out.get(endpoint_id)
        if (
            not tokens
            or mean_in is None
            or mean_out is None
            or not mean_in
            or not mean_out
        ):
            # no traffic, no price history, or a free instance: no vote
            continue
        weight += tokens
        total_in += tokens * mean_in
        total_out += tokens * mean_out
        providers.add(raw_names.get(endpoint_id, endpoint_id))
    if not weight:
        return None
    return OrEffectivePrice(
        input=_significant(total_in / weight),
        output=_significant(total_out / weight),
        days=len(chart.get("inputChartData") or []),
        providers=len(providers),
        cache_hit_rate=chart.get("weightedCacheHitRate"),
    )


def fetch_or_effective(force: bool, slugs: list[str]) -> dict[str, OrEffectivePrice]:
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
        if not force and entry and time.time() - entry["fetched_at"] < CACHE_TTL:
            charts[slug] = entry["chart"]
            continue
        chart = None
        error = None
        for variant in or_slug_variants(slug):
            url = (
                f"{OR_EFFECTIVE_URL}?permaslug={urllib.parse.quote(variant)}"
                f"&shape=v7&range={OR_EFFECTIVE_RANGE}"
            )
            payload = _or_get(url)
            error = payload.get("error")
            candidate = payload.get("data")
            if candidate and candidate.get("inputChartData"):
                chart = candidate
                break
        if chart is None:
            if slug in cached and cached[slug]["chart"].get("inputChartData"):
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
            print(f"no OR price chart for {slug}: {error}", file=sys.stderr)
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
    """A PricedTokens as a TOML array, full precision."""
    return "[" + ", ".join(repr(v) for v in t) + "]"


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


def _toml_str(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


@dataclass(frozen=True)
class Change:
    """One field updated in data.toml."""

    model: str
    field: str
    old: object
    new: object


def _number(value: object) -> float | None:
    """The value itself, or the total of a numeric array; None for text."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, (list, tuple)) and all(
        isinstance(v, (int, float)) and not isinstance(v, bool) for v in value
    ):
        return float(sum(value))
    return None


def _pct_change(old: object, new: object) -> str:
    """Signed percentage from old to new, rounded to a whole percent."""
    before, after = _number(old), _number(new)
    if before is None or after is None or before == 0:
        return ""
    pct = round((after - before) / abs(before) * 100)
    if pct == 0:
        return "0%"
    return f"{pct:+d}%"


def _print_changes(changes: list[Change]) -> None:
    """Print all field updates as an aligned table."""
    headers = ("Model", "Field", "Old", "New", "Δ%")
    rows = [
        (
            change.model,
            change.field,
            repr(change.old),
            repr(change.new),
            _pct_change(change.old, change.new),
        )
        for change in changes
    ]
    widths = [
        max([len(header)] + [len(row[i]) for row in rows])
        for i, header in enumerate(headers)
    ]

    def render(cells: tuple[str, ...]) -> str:
        return "  ".join(
            cell.ljust(width) if i < len(headers) - 1 else cell.rjust(width)
            for i, (cell, width) in enumerate(zip(cells, widths))
        ).rstrip()

    print(render(headers))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print(render(row))


def _set(
    row,
    key: str,
    value,
    model: str,
    changes: list[Change],
    *,
    field: str | None = None,
) -> None:
    """Update one TOML value, keeping its non-calculation comment."""
    item = row.get(key)
    old = item.unwrap() if hasattr(item, "unwrap") else item
    if old == value:
        return
    comment = getattr(getattr(item, "trivia", None), "comment", "")
    row[key] = list(value) if isinstance(value, tuple) else value
    new_item = row[key]
    trivia = getattr(new_item, "trivia", None)
    if trivia is not None:
        # A "# computed from:" comment describes the old hand calculation,
        # which no longer applies once AA has supplied the value.
        if comment and not comment.lstrip().startswith("# computed from:"):
            trivia.comment = comment
        else:
            trivia.comment = ""
    changes.append(Change(model=model, field=field or key, old=old, new=value))


def _base_of(rows, index: int):
    """The model row an effort row inherits its shared fields from."""
    stem = plot._split_effort(rows[index]["name"])[0]
    for j in range(index - 1, -1, -1):
        if (
            rows[j]["kind"] == "model"
            and plot._split_effort(rows[j]["name"])[0] == stem
        ):
            return rows[j]
    raise ValueError(f"{rows[index]['name']}: no model row above it")


def _validate(doc) -> None:
    """Validate the document before writing it back."""
    data = plot.DataFile.model_validate(doc.unwrap())
    plot.PUBLISHERS = data.publishers
    plot.HARDWARE_PRICES = data.hardware_prices
    for row in data.models:
        if isinstance(row, plot.ModelSpec) and row.hardware_bom is not None:
            plot.parse_hardware_bom(row.hardware_bom)
    plot.expand_model_efforts(row.to_model() for row in data.models)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--force", action="store_true", help="ignore caches")
    ap.add_argument("--new", metavar="OR_SLUG", help="print a TOML row for a new model")
    ap.add_argument("--aa-slug", help="AA page slug for --new")
    ap.add_argument("--aa-name", help="exact AA record name for --new")
    ap.add_argument("--publisher", help="publisher for --new")
    ap.add_argument("--name", help="display name for --new")
    args = ap.parse_args()

    if args.new:
        assert args.aa_slug and args.aa_name and args.publisher and args.name, (
            "--new needs --aa-slug, --aa-name, --publisher, --name"
        )
        session_costs, _detail = fetch_or_sessions(args.force)
        toks_served = fetch_or_toks_served(args.force)
        effective = fetch_or_effective(args.force, [args.new])
        rec = aa_query.get_page_record(
            {"name": args.aa_name, "slug": args.aa_slug}, args.force
        )
        if rec is None:
            sys.exit(f"no AA page record for {args.aa_name!r} at {args.aa_slug}")
        splits = aa_splits(rec)
        if splits is None:
            sys.exit(f"no AA cost split or sticker prices for {args.aa_name!r}")
        cost, sticker = splits
        eff = effective.get(args.new)
        if eff is None:
            sys.exit(f"no OR effective-pricing chart for {args.new}")
        sess = or_lookup(session_costs, args.new)
        toks = or_lookup(toks_served, args.new)
        print(f"# {args.name}: AA record {args.aa_name!r} (paste under [models])")
        print("[[models]]")
        print('kind = "model"')
        print(f"publisher = {_toml_str(args.publisher)}")
        print(f"name = {_toml_str(args.name)}")
        print(f"intelligence = {rec['intelligence']!r}")
        print('provider_type = "datacenter"')
        print(f"aa_page_slug = {_toml_str(args.aa_slug)}")
        print(f"aa_record_name = {_toml_str(args.aa_name)}")
        print("aa_derived = false")
        print(f"aa_cost_per_task = {_fmt_tokens(cost)}")
        print(f"aa_sticker_price = {_fmt_tokens(sticker)}")
        print(f"or_slug = {_toml_str(args.new)}")
        if sess is not None:
            print(f"or_session_cost_10_49_turns = {sess!r}")
        if toks is not None:
            print(f"or_toks_served = {toks!r}")
        print(f"or_eff_input_price = {eff.input!r}")
        print(f"or_eff_output_price = {eff.output!r}")
        return

    doc = tomlkit.parse(DATA_PATH.read_text())
    _validate(doc)
    rows = doc["models"]
    sessions, session_detail = fetch_or_sessions(args.force)
    toks_served = fetch_or_toks_served(args.force)
    effective = fetch_or_effective(
        args.force,
        sorted({row["or_slug"] for row in rows if row.get("or_slug")}),
    )
    aa_cache: dict[tuple[str, str], dict | None] = {}
    warned: set[tuple[str, str]] = set()
    changes: list[Change] = []

    for index, row in enumerate(rows):
        name = row["name"]
        slug = row["aa_page_slug"]
        rec_name = row["aa_record_name"]

        if slug and not row["aa_derived"]:
            key = (slug, rec_name)
            if key not in aa_cache:
                aa_cache[key] = aa_query.get_page_record(
                    {"name": rec_name, "slug": slug}, args.force
                )
            rec = aa_cache[key]
            if rec is None:
                if key not in warned:
                    print(
                        f"!! {name}: no AA index/cost for {rec_name!r} at {slug!r}; "
                        "keeping the stored AA values",
                        file=sys.stderr,
                    )
                    warned.add(key)
            elif rec_name != rec["_matched_name"]:
                actual = rec["_matched_name"]
                print(
                    f"!! {name}: AA record {rec_name!r} is now {actual!r}",
                    file=sys.stderr,
                )
                _set(row, "aa_record_name", actual, name, changes)
            if rec is not None:
                new_int = rec.get("intelligence")
                if new_int is not None:
                    _set(row, "intelligence", new_int, name, changes)
                splits = aa_splits(rec)
                if splits is not None:
                    new_cost, new_sticker = splits
                    sticker_row = row
                    if row["kind"] == "effort":
                        # Rungs inherit the base row's sticker prices.
                        sticker_row = _base_of(rows, index)
                    hand_mix = tuple(sticker_row["aa_sticker_price"]) == (
                        1.0,
                        1.0,
                        1.0,
                    )
                    if hand_mix:
                        # Dummy sticker price: the cost slot carries token
                        # counts, and only AA's published output count is real.
                        cost = list(row["aa_cost_per_task"])
                        new_tok = rec.get("output_tokens_per_task")
                        if new_tok:
                            cost[1] = round(new_tok) / 1e6
                            _set(row, "aa_cost_per_task", cost, name, changes)
                    else:
                        _set(row, "aa_cost_per_task", list(new_cost), name, changes)
                        _set(
                            sticker_row,
                            "aa_sticker_price",
                            list(new_sticker),
                            sticker_row["name"],
                            changes,
                            field="aa_sticker_price (via rung)",
                        )
                elif row["kind"] != "effort" and tuple(row["aa_sticker_price"]) == (
                    1.0,
                    1.0,
                    1.0,
                ):
                    # K2 Horizon 7B: AA publishes output tokens but no split.
                    cost = list(row["aa_cost_per_task"])
                    new_tok = rec.get("output_tokens_per_task")
                    if new_tok:
                        cost[1] = round(new_tok) / 1e6
                        _set(row, "aa_cost_per_task", cost, name, changes)

        if row["kind"] == "effort" or row["provider_type"] != "datacenter":
            continue
        or_slug = row.get("or_slug")
        if not or_slug:
            continue
        eff = effective.get(or_slug)
        if eff is None:
            print(
                f"!! {name}: no usable OR price chart for {or_slug}; "
                "keeping the stored prices",
                file=sys.stderr,
            )
        else:
            _set(row, "or_eff_input_price", eff.input, name, changes)
            _set(row, "or_eff_output_price", eff.output, name, changes)
        sess_key = or_lookup_key(sessions, or_slug)
        if sess_key is not None:
            n = len(session_detail[sess_key])
            if n < 4:
                print(
                    f"!! {name}: OR session cost from {n}/4 harnesses",
                    file=sys.stderr,
                )
            _set(
                row,
                "or_session_cost_10_49_turns",
                sessions[sess_key],
                name,
                changes,
            )
        new_vol = or_lookup(toks_served, or_slug)
        if new_vol is not None:
            _set(row, "or_toks_served", int(new_vol), name, changes)

    if changes:
        _validate(doc)
        DATA_PATH.write_text(tomlkit.dumps(doc))
        print(f"Updated {DATA_PATH}: {len(changes)} field change(s)")
        _print_changes(changes)
    else:
        print(f"No changes to {DATA_PATH}")


if __name__ == "__main__":
    main()
