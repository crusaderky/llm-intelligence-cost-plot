"""Query the Artificial Analysis Data API + website for one or more models.

The free Data API (GET /api/v2/data/llms/models) returns the unrounded
Intelligence Index, per-M-token pricing (input/output/blended) and speed — but
NOT the cost per task, cache prices or per-task token counts. Those only exist
on the website, embedded in each model page's Next.js flight payload. This
script therefore:

1. Fetches the API dataset (cached 6h) for fuzzy name matching, slugs and speed.
2. Fetches the model page for each match (cached 6h per slug) and extracts the
   full per-model record: Intelligence Index (sub-unit precision), cost per
   task with token-type breakdown (sub-cent precision), cache hit/write prices,
   and output tokens per Intelligence Index task.

Usage:
  aa_query.py --refresh                 # force re-fetch of API data + pages
  aa_query.py <name> [<name> ...]       # fuzzy substring match on name/slug
  aa_query.py --list                    # list all model names (from API)

Attribution required: https://artificialanalysis.ai/
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.request

API_URL = "https://artificialanalysis.ai/api/v2/data/llms/models"
CACHE_PATH = os.path.join(".cache", "aa_query.json")
PAGE_CACHE_DIR = os.path.join(".cache", "aa_pages")
CACHE_TTL = 6 * 3600  # hours; AA benchmarks update at most daily

# Fields in the page flight payload are JSON-in-JS escaped: `\"key\"`. This
# file content literal, used verbatim (escaped once more when it ends up in
# regexes via re.escape).
Q = '\\"'


def _plain(text: str) -> str:
    """Flight-payload field sequence -> regex-escaped pattern."""
    return re.escape(text)


def get_key() -> str:
    key = os.environ.get("AA_API_KEY", "").strip()
    if key:
        return key
    sys.exit(
        "No API key. Export AA_API_KEY (free account at "
        "https://artificialanalysis.ai, generate key, then add "
        "`export AA_API_KEY='...'` to your shell profile)"
    )


def load_cache():
    try:
        with open(CACHE_PATH) as f:
            entry = json.load(f)
        if time.time() - entry["fetched_at"] < CACHE_TTL:
            return entry["data"]
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        pass
    return None


def fetch_data(refresh: bool):
    if not refresh:
        cached = load_cache()
        if cached is not None:
            return cached, True
    req = urllib.request.Request(API_URL, headers={"x-api-key": get_key()})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            payload = json.load(resp)
    except urllib.error.HTTPError as e:
        sys.exit(f"API HTTP {e.code}: {e.read().decode(errors='replace')[:300]}")
    except urllib.error.URLError as e:
        sys.exit(f"Network error: {e.reason}")
    data = payload["data"]
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump({"fetched_at": time.time(), "data": data}, f)
    return data, False


# --------------------------------------------------------------------------
# Website scrape: per-model records embedded in the model page flight payload
# --------------------------------------------------------------------------


def _num(pattern: str, text: str):
    m = re.search(pattern, text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def parse_page_records(raw: str) -> dict[str, dict]:
    """Extract per-model records from a model page's flight payload.

    Returns {exact model name: {intelligence, cost_per_task breakdown, prices,
    output_tokens_per_task}}. Every model object in the payload contains the
    escaped sequence `\"name\":\"<name>\",\"shortName\":...` near its start and
    `\"intelligenceIndex\":` a few hundred chars later, so records are bounded
    by consecutive `\"shortName\":` markers. The pricing and cost-per-task
    fields follow the intelligence index, still inside the same record.
    """
    short = _plain(Q + "shortName" + Q + ":")
    locs = [m.start() for m in re.finditer(short, raw)]
    records: dict[str, dict] = {}
    for n, loc in enumerate(locs):
        window_end = locs[n + 1] if n + 1 < len(locs) else len(raw)
        window = raw[loc:window_end]
        # Model display name: the `"name":"..."` immediately before this
        # shortName marker.
        # Include the shortName marker itself, which the name pattern ends
        # on.
        back = raw[max(0, loc - 2000) : loc + 40]
        nm = re.search(
            _plain(Q + "name" + Q + ":" + Q)
            + r'([^"\\\\]{1,300}?)'
            + _plain(Q + "," + Q + "shortName" + Q),
            back,
        )
        if not nm:
            continue
        name = nm.group(1)
        rec: dict = {"label": name}
        ii = _num(
            _plain(Q + "intelligenceIndex" + Q + ":") + r"([0-9.eE+-]+)",
            window,
        )
        if ii is None:
            continue
        rec["intelligence"] = ii
        cpt = _num(
            _plain(
                Q
                + "intelligenceIndexCostPerTask"
                + Q
                + ":{"
                + Q
                + "cost"
                + Q
                + ":{"
                + Q
                + "total"
                + Q
                + ":"
            )
            + r"([0-9.eE+-]+)",
            window,
        )
        if cpt is not None:
            # Scope the breakdown-key search to the costPerTask object only:
            # the same keys appear earlier in the window inside the
            # intelligenceIndexCost (total run cost) object.
            cpt_start = window.index(Q + "intelligenceIndexCostPerTask" + Q + ":{")
            cpt_slice = window[cpt_start : cpt_start + 4000]
            for key in (
                "input",
                "nonCacheInput",
                "cacheRead",
                "cacheWrite",
                "output",
                "reasoning",
                "answer",
            ):
                rec[f"cost_per_task_{key}"] = _num(
                    _plain(Q + key + Q + ":") + r"([0-9.eE+-]+)", cpt_slice
                )
            rec["cost_per_task_total"] = cpt
        for key in ("input", "output"):
            rec[f"price_1m_{key}"] = _num(
                _plain(Q + f"price1m{key.capitalize()}Tokens" + Q + ":")
                + r"([0-9.eE+-]+)",
                window,
            )
        rec["price_1m_cache_hit"] = _num(
            _plain(Q + "cacheHitPrice" + Q + ":") + r"([0-9.eE+-]+)", window
        )
        rec["price_1m_cache_write"] = _num(
            _plain(Q + "cacheWritePrice" + Q + ":") + r"([0-9.eE+-]+)", window
        )
        # Key order inside intelligenceIndexOutputTokensPerTask is
        # reasoning, answer, output -- anchor on the object start and take
        # the third capture group.
        tot_m = re.search(
            _plain(
                Q
                + "intelligenceIndexOutputTokensPerTask"
                + Q
                + ":{"
                + Q
                + "reasoning"
                + Q
                + ":"
            )
            + r"([0-9.eE+-]+|null)"
            + _plain("," + Q + "answer" + Q + ":")
            + r"([0-9.eE+-]+|null)"
            + _plain("," + Q + "output" + Q + ":")
            + r"([0-9.eE+-]+|null)",
            window,
        )
        if tot_m and tot_m.group(3) != "null":
            rec["output_tokens_per_task"] = float(tot_m.group(3))
        records[name] = rec
    return records


def fetch_page(slug: str, refresh: bool) -> str | None:
    """Fetch and cache the model page HTML for one base slug."""
    os.makedirs(PAGE_CACHE_DIR, exist_ok=True)
    path = os.path.join(PAGE_CACHE_DIR, f"{slug}.html")
    if (
        not refresh
        and os.path.exists(path)
        and time.time() - os.path.getmtime(path) < CACHE_TTL
    ):
        with open(path, encoding="utf8", errors="replace") as f:
            return f.read()
    url = f"https://artificialanalysis.ai/models/{slug}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read().decode("utf8", errors="replace")
    except urllib.error.HTTPError as e:
        print(f"page HTTP {e.code} for {slug}", file=sys.stderr)
        return None
    except urllib.error.URLError as e:
        print(f"page fetch failed for {slug}: {e.reason}", file=sys.stderr)
        return None
    with open(path, "w", encoding="utf8") as f:
        f.write(raw)
    return raw


# Page records are keyed by exact display name; effort variants of one model
# all live on the page of the base slug. Derive candidates from an API slug.
_EFFORT_SUFFIXES = [
    "-non-reasoning",
    "-max",
    "-xhigh",
    "-high",
    "-medium",
    "-low",
    "-minimal",
]


def base_slug_candidates(slug: str) -> list[str]:
    cands = [slug]
    for suffix in _EFFORT_SUFFIXES:
        if slug.endswith(suffix):
            cands.append(slug[: -len(suffix)])
    return cands


def get_page_record(api_model: dict, refresh: bool) -> dict | None:
    """Look up the website record for one API model, by exact display name."""
    name = api_model.get("name", "")
    seen: set[str] = set()
    for cand in base_slug_candidates(api_model.get("slug", "")):
        if cand in seen:
            continue
        seen.add(cand)
        raw = fetch_page(cand, refresh)
        if raw is None:
            continue
        records = parse_page_records(raw)
        if name in records:
            return records[name]
        # Fall back to the only record whose label contains the API name's
        # distinctive first segment (before any parenthesised effort info).
        stem = name.split(" (")[0].lower()
        hits = [v for k, v in records.items() if k.lower().startswith(stem.lower())]
        if len(hits) == 1:
            return hits[0]
    return None


def fmt(m: dict) -> str:
    ev = m.get("evaluations") or {}
    pr = m.get("pricing") or {}
    creator = (m.get("model_creator") or {}).get("name", "?")
    lines = [
        f"{creator} — {m.get('name')} (slug: {m.get('slug')})",
        f"  Intelligence Index: {ev.get('artificial_analysis_intelligence_index', '?')}",
    ]
    parts = []
    for label, key in [
        ("input", "price_1m_input_tokens"),
        ("cache_read", "price_1m_cache_read_tokens"),
        ("cache_write", "price_1m_cache_write_tokens"),
        ("output", "price_1m_output_tokens"),
        ("blended_3:1", "price_1m_blended_3_to_1"),
    ]:
        if key in pr:
            parts.append(f"{label}=${pr[key]}/Mtok")
    if parts:
        lines.append("  Pricing: " + ", ".join(parts))
    sp = m.get("median_output_tokens_per_second")
    if sp is not None:
        lines.append(f"  Output speed: {sp} tok/s")
    page = m.get("_page") or {}
    if page:
        lines.append(f"  Intelligence Index (page, sub-unit): {page['intelligence']}")
        if page.get("cost_per_task_total") is not None:
            b = {
                k.removeprefix("cost_per_task_"): v
                for k, v in page.items()
                if k.startswith("cost_per_task_")
            }
            parts = [f"{k}=${v:.4f}" for k, v in b.items()]
            lines.append("  Cost/task breakdown: " + ", ".join(parts))
        if page.get("output_tokens_per_task") is not None:
            lines.append(f"  Output tokens/task: {page['output_tokens_per_task']:.0f}")
        for k in ("price_1m_cache_hit", "price_1m_cache_write"):
            if page.get(k) is not None:
                lines.append(f"  {k}: ${page[k]}/Mtok")
    else:
        lines.append("  [no page record found]")
    return "\n".join(lines)


def main():
    args = sys.argv[1:]
    refresh = "--refresh" in args
    args = [a for a in args if a != "--refresh"]
    data, from_cache = fetch_data(refresh)
    if "--list" in args:
        for m in data:
            print(m.get("name"))
        return
    if not args:
        sys.exit("usage: aa_query.py [--refresh] <name> ... | --list")
    # Substring match, case-insensitive, across name and slug.
    matches = {}
    for q in args:
        ql = q.lower()
        hits = [
            m
            for m in data
            if ql in m.get("name", "").lower() or ql in m.get("slug", "").lower()
        ]
        matches[q] = hits
    for q, hits in matches.items():
        if not hits:
            print(f"no match: {q}", file=sys.stderr)
        for m in hits:
            m["_page"] = get_page_record(m, refresh)
            print(fmt(m))
            print()
    if from_cache:
        print(
            f"[API cached; --refresh to re-fetch; {len(data)} models]",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
