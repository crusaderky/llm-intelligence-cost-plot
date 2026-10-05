---
name: refresh-models
description: Refreshes the ArtificialAnalysis (intelligence, output tokens per task, cost per task and its token-type breakdown, sticker prices) and OpenRouter (permaslug, effective input/output prices from the trailing week, 10-49-turn session cost, weekly tokens served) numbers behind every model in plot.py, or prints a ready-to-paste constructor row for a new model. Use when asked to refresh/re-fetch model scores and prices, or to add a model to the plots.
---

# Refresh or add models in plot.py

`plot.py` stores, per model: AA's Intelligence Index (y axis), AA's output tokens per
task, AA's cost per task and its split by token type, the sticker price per 1M tokens
that split was billed at, the OpenRouter permaslug and its synthetic effective
input/output prices, and (recorded but unplotted) OR's median cost of a 10-49-turn
session plus the prompt + completion tokens OR served for the model in the trailing week.
The displayed price per task is derived by `Model.price_per_task()`:

```text
datacenter: price = (input tokens x OR input $/M + output tokens x OR output $/M) / 1e6,
                    input tokens  = AA $/task[input] / AA sticker[input]
                                  + AA $/task[cached] / AA sticker[cached]
                    output tokens = AA $/task[output] / AA sticker[output]
local:      price = electricity to run those tokens on the local rig
                    (uncached input tokens / prefill tok/s + output tokens / decode tok/s)
```

`aa_cost_per_task` holds that split as a `PricedTokens(input_, output, cached_input)`
(USD per task, the three summing to the total) or, for a model AA publishes no split
for, the bare total as a float. `aa_sticker_price` is the same three streams in USD
per 1M tokens; a `cached_input` of 0.0 in either means "AA reports no cache reads
for this model".

A datacenter model with no OR price chart, no paid OR endpoint, or no AA split to get
the token mix from, keeps AA's own cost per task.

## Running the refresh

```bash
pixi r refresh-models             # cached (6h) fetch, prints old -> new table
pixi r refresh-models --refresh   # force re-fetch of AA pages + OR stats
```

Requires `AA_API_KEY` (exported in the shell profile) and the OpenRouter key at
`~/.pi/agent/auth.json` (field `openrouter.key`).

The script prints one row per `MODELS` entry — old vs new intelligence, tokens per task,
AA cost per task, AA token split and sticker prices (`SPL!`), OR effective input/output
prices (`ORP!`), OR session cost, and weekly tokens served — plus a **paste-ready
constructor kwargs** block for every row that drifted, and reminders. It **never edits
`plot.py`**: apply the changes yourself, then follow the AGENTS.md workflow (`pixi r
plot`, visually inspect the PNGs, `pixi r lint`).

## Reading the report

- `INT!` / `TOK!` — intelligence or tokens per task no longer matches `plot.py`
  (both keep small tolerances for AA-side rounding). `AA$!` / `SPL!` / `OR$!` / `ORP!` /
  `VOL!` are exact: the stored values are full precision, so any nonzero difference
  flags the row.
- `SPL!` (AA cost split + sticker prices), `ORP!` (OR effective prices) and `VOL!`
  (weekly volume) are expected on every refresh: they move with every AA page republish
  and every rolling window. Always apply them. The OR prices are rounded to 4
  significant digits (~0.01%) before they are stored, so sub-noise wobble in the
  trailing-week token-share average does not dirty plot.py or the plots.
- If OR ever returns a trailing week with no days at all (it does so transiently for a
  model whose traffic just moved permaslug), the refresh warns on stderr and keeps the
  last chart that had days in it, so a hiccup cannot silently drop a model back to AA's
  price.
- The session-cost statistic is a **30-day trailing median, and OR only rolls its window
  periodically** (observed `windowEnd` frozen for 8+ days at 2026-09-13); between rolls a
  refresh returns byte-identical session values, so an `OR$!` flag signals a real roll,
  not sampling noise. It is no longer used to price the plot — it is kept as a record of
  real session spend.
- The weekly-volume endpoint returns the trailing few days, not a calendar week; the
  window rolls every day, so `VOL!` fires on every datacenter row every time.
- A `!!` line means the AA record name in `AA_LOOKUPS` no longer matches AA verbatim;
  the refresh still resolves it by effort tag, but copy the name it prints.
- Reminder lines cover the special cases:
  - `GLM-5.3-Flash (high)` is extrapolated from the `(max)` record — multiply
    intelligence by 28.01/28.99 and tokens **and the whole `aa_cost_per_task` split** by
    70610/138690 (the sticker prices are the model's list prices and do not scale).
  - The `trains_on_your_data=True` twin of `Muse Spark 1.3` must always get the same AA
    numbers as `Muse Spark 1.3`, but has its own OR permaslug, effective prices, session
    cost and weekly volume. Watch out: two rows can share a display name.
  - Local models: update `intelligence` and `aa_tok_per_task`; keep
    `local_speed=LocalSpeed(prefill=…, decode=…)` and `hardware` (measured on real
    hardware, never from AA), but do add `aa_cost_per_task` / `aa_sticker_price` when
    AA publishes them — the electricity cost uses the real uncached input token count
    instead of `LOCAL_INPUT_TOKEN_RATIO ×` the output count. `Ternary-Bonsai-2` and
    `Occamy-1.0` are not on AA at all — manual entry, no refresh.
  - A model AA has retired (no record on its page any more) cannot be refreshed; drop
    the row rather than freezing its numbers. `Claude Sonnet 5.5 (low)` was dropped for
    this reason.
- `pixi r plot` reprints the recomputed displayed price per task of every model
  (AA's cost per task -> displayed price, the relative delta, and the OR $/M pair), so
  use it to sanity check the numbers.

## Threshold sanity after a refresh

- `LOW_COST_THRESHOLD` must still catch the cheap cluster *and* keep the green
  "cheap AND smart" band non-empty (at least one model with intelligence ≥
  `HIGH_INTELLIGENCE_THRESHOLD` at or below it).
- `HIGH_INTELLIGENCE_THRESHOLD` must still floor the high-intelligence plot.
- If a refresh moves a model across a threshold, update the threshold and say so in the
  commit message.

## Adding a new model

1. Find the model on OpenRouter: the permaslug (dated, e.g. `openai/gpt-6-astra-20260903`)
   from `https://openrouter.ai/<slug>` or the catalog API
   (`/api/frontend/v1/catalog/models`). Verify OR publishes a trailing-week effective
   price chart for it: `pixi r refresh-models --refresh` then check the slug is in
   `.cache/or_effective_pricing.json`. Without one the model plots at AA's own price.
  (A model that is free on OR — every endpoint priced at 0 — also plots at AA's price.)
2. Find it on AA and get the exact record name + page slug: `pixi r aa-query --list`
   (see the `aa-lookup` skill for details).
3. Print the constructor row:

```bash
pixi r refresh-models --new OR_SLUG --aa-slug AA_SLUG --aa-name "AA Record Name" \
    --publisher "Publisher" --name "Display Name"
```

4. Paste the row into `MODELS` (publisher already in `PUBLISHERS` with its
   artificialanalysis.ai color, or add it first). `AA_LOOKUPS` in
   `refresh_models.py` needs a matching row so future refreshes find it.
5. Effort variants are separate rows sharing one permaslug (and therefore one pair of OR
   prices, one session cost and one weekly volume each), like the Luna/Opus/Sonnet/Astra
   ladders. Their price per task still differs: each burns a different mix of input and
   output tokens.
6. A model not on AA (or one AA has retired) needs a hand-written bare total in
   `aa_cost_per_task` and no `aa_sticker_price`; it plots at that price.

## Provenance rules

- Intelligence, tokens per task, cost per task, its split and the sticker prices: AA
  model page payloads, unrounded (`aa-lookup` skill does the fetching).
- AA model names, slugs and speed: `GET /api/v2/language/models/free?page=N` (paginated
  at 200/page). The legacy `/api/v2/data/llms/models` is retired — `410 Gone` after
  2026-11-04 — and its `pricing` has no cache fields and stale speed data anyway.
- OR effective input/output prices: `GET /api/frontend/v1/stats/effective-pricing`
  (`?permaslug=…&shape=v7&range=1w`). One value per day per endpoint instance, plus
  `providerSummaries[]` carrying each instance's latest effective price (identical to the
  last chart day) and the tokens it served. Each instance is priced at the mean of its
  daily values, then the instances are averaged again weighted by their token share
  (`totalTokens`); instances priced at 0 (free tiers) are dropped and the rest
  renormalized, and a model with no paid instance keeps AA's price. Never substitute the
  spot "cheapest provider" price, nor the old per-day quintile/median reduction: it
  priced providers whose traffic is a rounding error.
- OR session cost: `GET /api/frontend/v1/rankings/session-cost`
  (`data.harnesses[].models[].points`, bucket `core` = 10–49 turns), averaged across
  the harnesses that carry session data. Recorded only, never plotted.
- OR weekly tokens served: `GET /api/frontend/v1/rankings/models?view=week` (`data[]`,
  one daily row per permaslug and variant). Sum `total_prompt_tokens` +
  `total_completion_tokens` over every day and every variant (standard, batch, free).
- Effort variants share one permaslug; the ×-tokens-per-task factor is what separates
  them on the plot.
- Keep README.md in sync with any value change (AGENTS.md rule).