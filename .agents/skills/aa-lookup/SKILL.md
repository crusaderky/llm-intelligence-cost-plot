---
name: aa-lookup
description: Queries artificialanalysis.ai for a model's Intelligence Index and pricing (input/cache/output per M tokens, cost per task) via the AA free Data API and the model page payload. Use when the user asks for AA intelligence scores, model benchmark scores, or AA pricing for LLMs — replaces pixel-peeping AA's plots.
---

# AA Lookup — Intelligence Index + pricing from Artificial Analysis

## Why this exists

AA's website displays the Intelligence Index rounded to the unit, and cost per task
with aggressive rounding on cheap models. The **free Data API** plus each model page's
machine-readable payload give the exact values — this is the robust alternative to
pixel-peeping plots:

The free **Data API** returns the Intelligence Index and per-M-token pricing
(input/output/cache hit/cache write) — but NOT the cost-per-task token breakdown or
per-task token counts. Those exist only on the website, embedded in each model page's Next.js
flight payload. The script therefore:

1. Queries the API for name matching, slugs and speed.
1. Fetches the model page (one per base slug, cached 6h) and extracts the full record:
   Intelligence Index (sub-unit precision), cost per task with a token-type breakdown
   (input / nonCacheInput / cacheRead / cacheWrite / output / reasoning / answer, at
   sub-cent precision), the sticker price per 1M tokens for each stream (input /
   cache read / cache write / output), and output tokens per task.

Record names drift (AA renames effort variants, e.g. `Kimi K3 (max)` → `Kimi K3 (Max)`,
and retires records entirely). `get_page_record` matches exactly, then
case-insensitively, then by name stem plus effort tag, and stamps the name it actually
matched as `_matched_name`; `aa-query` prints a note when that differs from the queried
name. Refresh tooling uses `pick_record` the same way.

- `evaluations.artificial_analysis_intelligence_index` — API-side (may be rounded);
  the page record's `intelligence` is the sub-unit value to use.
- `intelligenceIndexCostPerTask.cost` — per-task cost breakdown. `output` includes
  reasoning + answer; `input` = nonCacheInput + cacheRead + cacheWrite, so plot.py's
  `PricedTokens(input_, output, cached_input)` is
  `(nonCacheInput + cacheWrite, output, cacheRead)`.
- `intelligenceIndexOutputTokensPerTask.output` — total output tokens per task
  (reasoning + answer).
- `price1mInputTokens` / `price1mOutputTokens` / `cacheHitPrice` (keyed
  `price_1m_input`, `price_1m_output`, `price_1m_cache_hit`) — the sticker prices, per
  1M tokens, that the breakdown was billed at. Cost ÷ price recovers the token counts
  of each stream; a missing `cacheHitPrice` means the model is treated as never hitting
  a cache.

Docs: https://artificialanalysis.ai/data-api/docs
Endpoint: `GET https://artificialanalysis.ai/api/v2/language/models/free?page=N`
(V2 free tier; paginated at a fixed 200 models/page — the script walks `pagination.has_more`,
so one refresh is ~4 requests; 1000 req/day; responses cached locally for 6h — do not hammer
it). The legacy `GET /api/v2/data/llms/models` is retired: it already answers with
`Deprecation`/`Sunset` headers and returns `410 Gone` after 2026-11-04. The local cache records
the endpoint it came from and is refetched when the endpoint changes.

Free-tier shape: `evaluations` (headline + capability indices), `pricing`
(`price_1m_input_tokens` / `price_1m_output_tokens` / `price_1m_cache_hit_tokens` /
`price_1m_cache_write_tokens` — no blended field), `performance` (medians only, nested under
`performance`), `model_creator` (no `slug`), and
`artificial_analysis_intelligence_index_cost.cost_per_task.total_cost` — AA's cost per task,
total only and null for most models; `aa_query` prints it as a cross-check against the page
breakdown, which is what plot.py consumes.

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
- `output_tokens_per_task` — AA's benchmark task size in output tokens; feeds
  `aa_tok_per_task`.
- `cost_per_task_*` — feeds `aa_cost_per_task=PricedTokens(input_, output,
  cached_input)`, the per-task cost split by token type.
- `price_1m_input` / `price_1m_output` / `price_1m_cache_hit` — feeds
  `aa_sticker_price=PricedTokens(...)`, the same three streams in USD per 1M tokens.
  Together with the split they give the task's token mix, which is what the OpenRouter
  prices are applied to. A missing `price_1m_cache_hit` becomes `cached_input = 0.0`,
  which reads as "AA reports no cache reads for this model".
- `cost_per_task_total` — the total the split sums to; when AA publishes no split at
  all, store it as a bare float in `aa_cost_per_task`. Either way it is the fallback
  price and the delta plot's baseline.

The `.agents/skills/refresh-models` skill automates all of this (AA + OpenRouter)
and prints old -> new values plus paste-ready constructor rows; prefer it over
manual entry.

## Attribution

AA free API requires attribution to https://artificialanalysis.ai/ when data is
redistributed (README already attributes; keep it).
