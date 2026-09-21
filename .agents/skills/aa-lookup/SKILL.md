---
name: aa-lookup
description: Queries artificialanalysis.ai for a model's unrounded Intelligence Index and pricing (input/cache/output per M tokens, cost per task) via the AA free Data API. Use when the user asks for AA intelligence scores, model benchmark scores, or AA pricing for LLMs — replaces pixel-peeping AA's plots.
---

# AA Lookup — Intelligence Index + pricing from Artificial Analysis

## Why this exists

AA's website displays the Intelligence Index rounded to the unit, and cost per task
with aggressive rounding on cheap models. The **free Data API** returns the unrounded
values — this is the robust alternative to pixel-peeping plots:

The free **Data API** returns the unrounded Intelligence Index and basic per-M-token
pricing (input/output/blended) — but NOT cost per task, cache prices or per-task token
counts. Those exist only on the website, embedded in each model page's Next.js flight
payload. The script therefore:

1. Queries the API for name matching, slugs and speed.
1. Fetches the model page (one per base slug, cached 6h) and extracts the full record:
   Intelligence Index (sub-unit precision), cost per task with a token-type breakdown
   (input / nonCacheInput / cacheRead / cacheWrite / output / reasoning / answer, at
   sub-cent precision), cache hit/write prices, and output tokens per task.

- `evaluations.artificial_analysis_intelligence_index` — API-side (may be rounded);
  the page record's `intelligence` is the sub-unit value to use.
- `intelligenceIndexCostPerTask.cost` — per-task cost breakdown. `output` includes
  reasoning + answer; `input` = nonCacheInput + cacheRead + cacheWrite.
- `intelligenceIndexOutputTokensPerTask.output` — total output tokens per task
  (reasoning + answer); use for local-electricity cost calculations.

Docs: https://artificialanalysis.ai/api-reference
Endpoint: `GET https://artificialanalysis.ai/api/v2/data/llms/models` (all models,
one request; 1000 req/day; responses cached locally for 6h — do not hammer it).

## Setup

Key resolution: **`AA_API_KEY` env var only** (no key file). If unset, script exits with
instructions. Export it in your shell profile:

```bash
export AA_API_KEY='...'
```

## Usage

```bash
pixi r aa-query "fable 5.1"   # fuzzy match
pixi r aa-query --refresh     # ignore cache
pixi r aa-query --list        # all names
```

Run through the pixi environment (`pixi r aa-query`) — never via a globally-installed
python. Multiple queries in one call share the cached fetch. The Intelligence Index and
cost-per-task numbers are taken from the model page (sub-unit / sub-cent precision);
query with the full exact name (as printed by `--list`) for a reliable page match.

## Feeding results into plot.py

`plot.py` consumes, per model:

- `intelligence` — the page record's sub-unit value (the API-side
  `evaluations.artificial_analysis_intelligence_index` may be rounded).
- `output_tokens_per_task` — AA's benchmark task size in output tokens; feeds the
  reference-cost formula (`OR session cost x tokens / HOUR_SCALE`) in
  `Model.datacenter` and the electricity calculation in `Model.local`.
- `cost_per_task_total` — no longer plotted, but still needed for the hardcoded
  GPT-5.6 Luna (max) anchor inside `Model.local` and as a sanity cross-check.

The `.agents/skills/refresh-models` skill automates all of this (AA + OpenRouter)
and prints old -> new values plus paste-ready constructor rows; prefer it over
manual entry.

## Attribution

AA free API requires attribution to https://artificialanalysis.ai/ when data is
redistributed (README already attributes; keep it).
