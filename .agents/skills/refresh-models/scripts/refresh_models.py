"""Refresh the AA + OpenRouter numbers behind every model in plot.py.

Reads the MODELS list from plot.py, re-fetches from:
- artificialanalysis.ai (Data API + model page payloads): Intelligence Index,
  output tokens per task, cost per task.
- openrouter.ai (GET /api/frontend/v1/rankings/session-cost): median cost of
  a 10-49-turn session ("core" bucket) per permaslug, averaged across the
  OR coding harnesses that carry session data for the model. Models with no
  session data on any harness are reported as missing (no fallback
  statistic exists for this metric).
- openrouter.ai (GET /api/frontend/v1/rankings/models?view=week): prompt +
  completion tokens served in the trailing week per permaslug, summed across
  variants. This is the weight behind plot.py's or_tokens_per_session.

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
import sys
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
SKILLS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SKILLS / "aa-lookup" / "scripts"))
sys.path.insert(0, str(REPO_ROOT))

import aa_query

import plot

OR_SESSION_URL = "https://openrouter.ai/api/frontend/v1/rankings/session-cost"
OR_MODELS_URL = "https://openrouter.ai/api/frontend/v1/rankings/models"
CACHE_PATH = REPO_ROOT / ".cache" / "or_session_cost.json"
TOKS_CACHE_PATH = REPO_ROOT / ".cache" / "or_toks_served.json"
CACHE_TTL = 6 * 3600
AUTH_PATHS = [
    Path.home() / ".pi" / "agent" / "auth.json",
    Path.home() / ".pi" / "auth.json",
]

# plot.py MODELS base name -> (AA page slug, AA record name).
# One row per DISTINCT AA record; [TRAIN] twins and local-vs-datacenter pairs
# of the same model share a row.
AA_LOOKUPS: dict[str, tuple[str, str, bool]] = {
    # base name: (aa_page_slug, aa_record_name, derived_intelligence?)
    "MiniCPM5-2B": ("minicpm5-2b", "MiniCPM5-2B", False),
    "Muse Glimmer": ("muse-glimmer", "Muse Glimmer (high)", False),
    "Qwen3.6-35B-A3B": ("qwen3-6-35b-a3b", "Qwen3.6 35B A3B (Reasoning)", False),
    "Qwen3.8-27B (Non-reasoning)": (
        "qwen3-8-27b",
        "Qwen3.8 27B (Non-reasoning)",
        False,
    ),
    "Qwen3.8-27B (low)": ("qwen3-8-27b", "Qwen3.8 27B (low)", False),
    "Qwen3.8-27B (medium)": ("qwen3-8-27b", "Qwen3.8 27B (medium)", False),
    "Qwen3.8-27B (xhigh)": ("qwen3-8-27b", "Qwen3.8 27B (xhigh)", False),
    "Ternary-Bonsai-2": ("", "", False),  # not on AA: manual entry
    "Qwen3.8-Flash-Next": ("qwen3-8-flash-next", "Qwen3.8-Flash-Next", False),
    "K2 Horizon 7B": ("k2-horizon-7b", "K2 Horizon 7B", False),
    "Qwen3.8 Max (0902)": ("qwen3-8-max", "Qwen3.8 Max (0902)", False),
    "DeepSeek V4.1 Flash": (
        "deepseek-v4-1-flash",
        "DeepSeek V4.1 Flash (Reasoning, Max Effort)",
        False,
    ),
    "Hy3": ("hy3", "Hy3", False),
    "Muse Spark 1.3": ("muse-spark-1-3", "Muse Spark 1.3 (max)", False),
    # (high) is extrapolated from the (max) record: intelligence x
    # 28.01/28.99, tokens and price x 70610/138690 (ratios from Z.ai's coding
    # scores and AA's GLM-5.3 effort split, see the README note)
    "GLM-5.3-Flash (high)": ("glm-5-3-flash", "GLM 5.3 Flash", True),
    "GLM-5.3-Flash (max)": ("glm-5-3-flash", "GLM 5.3 Flash", False),
    "GLM-5.3": ("glm-5-3", "GLM-5.3 (max)", False),
    "Kimi K3": ("kimi-k3", "Kimi K3 (max)", False),
    "Gemini 3.8 Flash": ("gemini-3-8-flash", "Gemini 3.8 Flash (high)", False),
    "Grok 4.6": ("grok-4-6", "Grok 4.6 (xhigh)", False),
    "Grok 4.7": ("grok-4-7", "Grok 4.7 (xhigh)", False),
    "MiMo-V2.6-Flash": ("", "", False),  # not on AA yet
    "MiMo-V2.6-Pro": ("mimo-v2-6-pro", "MiMo-V2.6-Pro", False),
    "GPT-5.5 (Apr '26)": ("gpt-5-5", "GPT-5.5 (xhigh)", False),
    "GPT-5.6 Sol (Jul '26)": ("gpt-5-6-sol", "GPT-5.6 Sol (max)", False),
    "GPT-6 Luna (low)": ("gpt-6-luna-low", "GPT-6 Luna (low)", False),
    "GPT-6 Luna (medium)": ("gpt-6-luna-medium", "GPT-6 Luna (medium)", False),
    "GPT-6 Luna (high)": ("gpt-6-luna-high", "GPT-6 Luna (high)", False),
    "GPT-6 Luna (xhigh)": ("gpt-6-luna-xhigh", "GPT-6 Luna (xhigh)", False),
    "GPT-6 Luna (max)": ("gpt-6-luna", "GPT-6 Luna (max)", False),
    "GPT-6 Sol (low)": ("gpt-6-sol-low", "GPT-6 Sol (low)", False),
    "GPT-6 Sol (medium)": ("gpt-6-sol-medium", "GPT-6 Sol (medium)", False),
    "GPT-6 Sol (high)": ("gpt-6-sol-high", "GPT-6 Sol (high)", False),
    "GPT-6 Sol (xhigh)": ("gpt-6-sol-xhigh", "GPT-6 Sol (xhigh)", False),
    "GPT-6 Sol (max)": ("gpt-6-sol", "GPT-6 Sol (max)", False),
    "GPT-6 Astra (low)": ("gpt-6-astra-low", "GPT-6 Astra (low)", False),
    "GPT-6 Astra (medium)": ("gpt-6-astra-medium", "GPT-6 Astra (medium)", False),
    "GPT-6 Astra (high)": ("gpt-6-astra-high", "GPT-6 Astra (high)", False),
    "GPT-6 Astra (xhigh)": ("gpt-6-astra-xhigh", "GPT-6 Astra (xhigh)", False),
    "GPT-6 Astra (max)": ("gpt-6-astra", "GPT-6 Astra (max)", False),
    "Step 5 Preview": ("step-5", "Step 5 Preview", False),
    "Claude Opus 4.8 (May '26)": (
        "claude-opus-4-8",
        "Claude Opus 4.8 (Adaptive Reasoning, Max Effort)",
        False,
    ),
    "Claude Haiku 4.5": (
        "claude-4-5-haiku-reasoning",
        "Claude 4.5 Haiku (Reasoning)",
        False,
    ),
    "Claude Sonnet 5": (
        "claude-sonnet-5",
        "Claude Sonnet 5 (Adaptive Reasoning, Max Effort)",
        False,
    ),
    "Claude Opus 5 (Jul '26)": (
        "claude-opus-5",
        "Claude Opus 5 (Adaptive Reasoning, Max Effort)",
        False,
    ),
    "Claude Fable 5 (Jun '26)": (
        "claude-fable-5",
        "Claude Fable 5 (Adaptive Reasoning, Max Effort, Opus 4.8 Fallback)",
        False,
    ),
    "Claude Fable 5.1 (Aug '26)": (
        "claude-fable-5-1",
        "Claude Fable 5.1 (Adaptive Reasoning, Max Effort, Default Fallback)",
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
    volume that weights plot.py's or_tokens_per_session. Every variant (standard,
    batch, free) counts as served tokens.
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


def base_name(name: str) -> str:
    """MODELS display name -> key into AA_LOOKUPS (strip local hardware suffix,
    [TRAIN] and [UNAVAILABLE] markers)."""
    left, _icon, _right, _strike = plot._split_icon(name)
    left = left.replace(" [TRAIN]", "").replace(" [UNAVAILABLE]", "")
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
        rec = aa_query.get_page_record(
            {"name": args.aa_name, "slug": args.aa_slug}, args.refresh
        )
        if rec is None:
            sys.exit(f"no AA page record for {args.aa_name!r} at {args.aa_slug}")
        if args.new not in session_costs:
            sys.exit(
                f"{args.new} has no 10-49-turn session data on any OR harness; "
                "it cannot be plotted under the current design"
            )
        if rec.get("cost_per_task_total") is None:
            sys.exit(f"no AA cost per task for {args.aa_name!r} at {args.aa_slug}")
        cost = session_costs[args.new]
        toks = toks_served.get(args.new)
        print("AA record:", json.dumps(rec, indent=1))
        print(
            f"OR {args.new}: ${cost:.6f} average 10-49-turn session cost "
            f"across {len(_detail[args.new])} harness(es)"
        )
        print(f"OR {args.new}: {toks} tokens served in the trailing week")
        print(
            "Model(\n"
            f'    "{args.publisher}",\n'
            f'    "{args.name}",\n'
            f"    {rec['intelligence']:.4f},\n"
            "    ProviderType.DATACENTER,\n"
            f"    aa_tok_per_task={round(rec['output_tokens_per_task'])},\n"
            f"    aa_price_per_task={rec['cost_per_task_total']!r},\n"
            f'    or_slug="{args.new}",\n'
            f"    or_session_cost_10_49_turns={cost!r},\n"
            f"    or_toks_served={toks!r},\n"
            "),"
        )
        return

    session_costs, session_detail = fetch_or_sessions(args.refresh)
    toks_served = fetch_or_toks_served(args.refresh)
    print("(OR session $: 10-49-turn median, averaged across the coding harnesses)")
    print("(OR week toks: prompt + completion tokens served in the trailing week)")
    aa_cache: dict[tuple[str, str], dict | None] = {}
    print(
        f"{'model':30s} {'intelligence':>22s} {'tokens/task':>20s} "
        f"{'AA $/task':>20s} {'OR session $':>28s} {'OR week toks':>20s}"
    )
    changed = 0

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
        new_int = rec["intelligence"] if rec else m.intelligence
        new_tok = (
            round(rec["output_tokens_per_task"])
            if rec and rec.get("output_tokens_per_task")
            else None
        )
        old_int, old_tok = m.intelligence, m.aa_tok_per_task
        if m.provider_type is plot.ProviderType.DATACENTER:
            old_aa = m.aa_price_per_task
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
                new_sess, src = None, " (no session data!)"
            new_vol = toks_served.get(m.or_slug)
        else:  # local: electricity, no OR fields
            old_aa = old_sess = old_vol = new_aa = new_sess = new_vol = None
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
            if new_aa is not None and new_aa != old_aa:
                mark += " AA$!"
            if new_sess is not None and new_sess != old_sess:
                mark += " OR$!"
            # expected on every refresh: the volume is a rolling window
            if new_vol is not None and new_vol != old_vol:
                mark += " VOL!"
        if mark:
            changed += 1
        print(
            f"{plot._pad_cell(m.name[:30], 30)} {fmt(old_int):>9s} "
            f"-> {fmt(new_int):>9s} "
            f"{fmt(old_tok, '.0f'):>8s} -> {fmt(new_tok, '.0f'):>8s} "
            f"{fmt(old_aa):>8s} -> {fmt(new_aa):>8s} "
            f"{fmt(old_sess, '.6f'):>12s} -> {fmt(new_sess, '.6f'):>12s} "
            f"{fmt_vol(old_vol):>8s} -> {fmt_vol(new_vol):>8s}{mark}{src}"
        )
        if derived:
            print(
                "   (derived: values are extrapolated from the (max) record — "
                "the ratios are historical, see the README note)"
            )
        if "[TRAIN]" in m.name:
            print("   ([TRAIN] twin of the entry above — keep values in sync)")
        if m.provider_type is plot.ProviderType.LOCAL:
            print(
                "   (local: update intelligence + aa_tok_per_task; "
                "tok/s and hardware stay)"
            )

    print(
        f"\n{changed} model(s) drifted. Apply the changes to plot.py, then:\n"
        "  pixi r plot  (reprints every displayed price per task; inspect the PNGs!)\n"
        "  pixi r lint\n"
        "Check that LOW_COST_THRESHOLD still catches the cheap cluster and leaves\n"
        "the green band non-empty, and that HIGH_INTELLIGENCE_THRESHOLD still\n"
        "floors the top plot. VOL! is expected on every refresh: the weekly\n"
        "serving volume is a rolling window, so always apply it."
    )


if __name__ == "__main__":
    main()
