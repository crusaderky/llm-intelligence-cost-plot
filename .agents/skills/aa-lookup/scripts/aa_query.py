"""Query the Artificial Analysis Data API + website for one or more models.

The free Data API (GET /api/v2/language/models/free — the V2 replacement for the
retired /api/v2/data/llms/models) returns the Intelligence Index, per-M-token
pricing (input/output/cache hit/cache write), the headline indices and median
speed — but NOT the cost-per-task token breakdown or per-task token counts.
Those only exist on the website, embedded in each model page's Next.js flight
payload. This script therefore:

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

# V2 free-tier endpoint. The legacy /api/v2/data/llms/models answers with
# `Deprecation` / `Sunset` headers and returns 410 Gone after 2026-11-04.
# It is paginated at a fixed 200 models per page; a page past total_pages is a 500.
API_URL = "https://artificialanalysis.ai/api/v2/language/models/free"
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
        # A cache written by a different endpoint has a different field shape.
        if entry["api"] != API_URL:
            return None
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
    # Walk the pagination to the end (200 models per page, ~4 pages).
    data: list[dict] = []
    try:
        page = 1
        while True:
            req = urllib.request.Request(
                f"{API_URL}?page={page}", headers={"x-api-key": get_key()}
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                payload = json.load(resp)
            data += payload["data"]
            if not payload["pagination"]["has_more"]:
                break
            page += 1
    except urllib.error.HTTPError as e:
        sys.exit(f"API HTTP {e.code}: {e.read().decode(errors='replace')[:300]}")
    except urllib.error.URLError as e:
        sys.exit(f"Network error: {e.reason}")
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump({"fetched_at": time.time(), "api": API_URL, "data": data}, f)
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


def _paren_tags(name: str) -> set[str]:
    """The words inside a record name's parentheses, lowercased.

    Splitting on spaces, commas and dashes keeps "xhigh" and "non-reasoning"
    whole, so a wanted "(High)" never matches a record's "(Xhigh)".
    """
    if "(" not in name:
        return set()
    return {
        w for w in re.split(r"[\s,\-]+", name[name.rfind("(") + 1 : -1].lower()) if w
    }


def pick_record(records: dict[str, dict], wanted: str) -> dict | None:
    """The page record for `wanted`, tolerating AA's naming drift.

    Tries the exact display name, then a case-insensitive match, then the
    records whose name starts with the same stem (the part before the
    parentheses), disambiguated by how many of the wanted name's effort tags
    they repeat. Returns None when nothing matches uniquely, and stamps the
    name it actually matched as "_matched_name" so callers can spot a lookup
    whose stored name no longer matches AA.
    """
    if wanted in records:
        rec = records[wanted]
        rec["_matched_name"] = wanted
        return rec
    lowered = {k.lower(): k for k in records}
    if wanted.lower() in lowered:
        key = lowered[wanted.lower()]
        rec = records[key]
        rec["_matched_name"] = key
        return rec
    stem = wanted.split(" (")[0].lower()
    same_stem = {k: v for k, v in records.items() if k.lower().startswith(stem)}
    if len(same_stem) == 1:
        key, rec = next(iter(same_stem.items()))
        rec["_matched_name"] = key
        return rec
    tags = _paren_tags(wanted)
    # Score = tags the record shares with the wanted name, less tags it adds of
    # its own, so "(Max)" beats "(Non-reasoning)" for a wanted "(Reasoning,
    # Max Effort)" and "(High)" beats "(Xhigh)" for a wanted "(High)".
    scored = [
        (len(_paren_tags(k) & tags) - len(_paren_tags(k) - tags), k, v)
        for k, v in same_stem.items()
    ]
    if tags and scored:
        best = max(s for s, _, _ in scored)
        if best > 0:
            winners = [(k, v) for s, k, v in scored if s == best]
            if len(winners) == 1:
                key, rec = winners[0]
                rec["_matched_name"] = key
                return rec
    return None


def get_page_record(api_model: dict, refresh: bool) -> dict | None:
    """Look up the website record for one API model, by display name."""
    name = api_model.get("name", "")
    seen: set[str] = set()
    for cand in base_slug_candidates(api_model.get("slug", "")):
        if cand in seen:
            continue
        seen.add(cand)
        raw = fetch_page(cand, refresh)
        if raw is None:
            continue
        rec = pick_record(parse_page_records(raw), name)
        if rec is not None:
            return rec
    return None


def fmt(m: dict) -> str:
    ev = m.get("evaluations") or {}
    pr = m.get("pricing") or {}
    perf = m.get("performance") or {}
    creator = (m.get("model_creator") or {}).get("name", "?")
    lines = [
        f"{creator} — {m.get('name')} (slug: {m.get('slug')})",
        f"  Intelligence Index: {ev.get('artificial_analysis_intelligence_index', '?')}",
    ]
    parts = []
    for label, key in [
        ("input", "price_1m_input_tokens"),
        ("cache_read", "price_1m_cache_hit_tokens"),
        ("cache_write", "price_1m_cache_write_tokens"),
        ("output", "price_1m_output_tokens"),
    ]:
        if pr.get(key) is not None:
            parts.append(f"{label}=${pr[key]}/Mtok")
    if parts:
        lines.append("  Pricing: " + ", ".join(parts))
    sp = perf.get("median_output_tokens_per_second")
    if sp is not None:
        lines.append(f"  Output speed: {sp} tok/s")
    # Free tier exposes AA's own cost per Intelligence Index task (total only,
    # no token-type split); the page record below is what plot.py actually uses.
    cpt = (
        (m.get("artificial_analysis_intelligence_index_cost") or {})
        .get("cost_per_task", {})
        .get("total_cost")
    )
    if cpt is not None:
        lines.append(f"  API cost/task (total only): ${cpt}")
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
    if page and page.get("_matched_name") not in (None, m.get("name")):
        lines.append(
            f"  note: matched AA record {page['_matched_name']!r} "
            f"(API name {m.get('name')!r} is not on the page verbatim)"
        )
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
