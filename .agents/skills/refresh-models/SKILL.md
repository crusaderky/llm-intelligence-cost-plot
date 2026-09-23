---
name: refresh-models
description: Refreshes the ArtificialAnalysis (intelligence, output tokens per task, cost per task) and OpenRouter (permaslug, avg 10-49-turn session cost, weekly tokens served) numbers behind every model in plot.py, or prints a ready-to-paste constructor row for a new model. Use when asked to refresh/re-fetch model scores and prices, or to add a model to the plots.
---

# Refresh or add models in plot.py

`plot.py` stores, per model: AA's Intelligence Index (y axis), AA's output tokens per
task, AA's cost per task, the OpenRouter permaslug, OpenRouter's median cost of a
10-49-turn session (the "core" bucket, averaged across OR's coding harnesses), and the
prompt + completion tokens OpenRouter served for the model in the trailing week. The
displayed price per task is derived by `Model.price_per_task()`:

```text
datacenter: price = AA tokens/task x OR session $ / K,
                    K = volume-weighted mean of
                        (AA tokens/task x OR session $ / AA $/task)
local:      price = electricity to generate AA's output tokens per task on the
                    hardware the decode speed was measured on
```

K is the volume-weighted estimate of the tokens a 10-49-turn session burns (AA's cost
per task and OR's session cost share the model's real $/token, so their ratio isolates
it). Calibrating K on AA's own prices pins the volume-weighted average of
price-per-task/AA-price to 1, so the plot keeps AA's dollar level while taking its
relative shape from what OpenRouter's users actually pay: a model whose real $/token is
above the volume-weighted average plots above AA's sticker price, and vice versa. A
datacenter model with no 10-49-turn session data keeps AA's cost per task unscaled.
Local (⚡ electricity) models store the AA fields and a measured tok/s, but no OR fields.

## Running the refresh

```bash
pixi r refresh-models             # cached (6h) fetch, prints old -> new table
pixi r refresh-models --refresh   # force re-fetch of AA pages + OR stats
```

Requires `AA_API_KEY` (exported in the shell profile) and the OpenRouter key at
`~/.pi/agent/auth.json` (field `openrouter.key`).

The script prints one row per `MODELS` entry — old vs new intelligence, tokens per task,
AA cost per task, OR session cost, and weekly tokens served — plus reminders. It
**never edits `plot.py`**: apply the changes yourself, then follow the AGENTS.md
workflow (`pixi r plot`, visually inspect the PNGs, `pixi r lint`).

## Reading the report

- `INT!` / `TOK!` — intelligence or tokens per task no longer matches `plot.py`
  (both keep small tolerances for AA-side rounding). `AA$!` / `OR$!` / `VOL!` are
  exact: the stored AA cost per task, OR session cost and weekly volume are full
  precision, so any nonzero difference flags the row.
- `VOL!` is expected on every refresh — weekly tokens served is a rolling window, so
  always apply it. The other flags only fire when AA republishes a page or OR rolls its
  session window. The session-cost statistic is a **30-day trailing median, and OR only
  rolls its window periodically** (observed `windowEnd` frozen for 8+ days at
  2026-09-13); between rolls a refresh returns byte-identical session values, so an
  `OR$!` flag signals a real roll, not sampling noise. Conversely, the stat is a lagging
  indicator: new models and price changes take days to weeks to show up. The old
  avg-price-per-100-requests statistic this replaced was recomputed from same-day
  traffic and swung 30-50% within hours.
- The weekly-volume endpoint returns the trailing few days, not a calendar week; the
  window rolls every day, so `VOL!` fires on every datacenter row every time.
- Reminder lines cover the special cases:
  - `GLM-5.3-Flash (high)` is extrapolated from the `(max)` record — multiply
    intelligence by 28.01/28.99 and tokens by 70610/138690 (the ratios themselves are
    historical, from Z.ai's coding scores and AA's GLM-5.3 effort split).
  - `Muse Spark 1.3 [TRAIN]` must always get the same AA numbers as `Muse Spark 1.3`,
    but has its own OR permaslug, session cost and weekly volume.
  - Local models: update `intelligence` and `aa_tok_per_task`; keep tok/s and
    `hardware` (measured on real hardware, never from AA). `Ternary-Bonsai-2` is not on
    AA at all — manual entry, no refresh.
- `pixi r plot` reprints the recomputed displayed price per task of every model
  (AA's cost per task -> displayed price, plus the relative delta), so use it to sanity
  check the new K (tokens/session).

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
   (`/api/frontend/v1/catalog/models`). Verify OR actually publishes a 10-49-turn
   session cost for it on any harness: `pixi r refresh-models --refresh` then
   check the slug is in `.cache/or_session_cost.json`. If no harness carries session
   data for it, the model cannot be plotted under the current design — say so instead
   of inventing a price.
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
5. Effort variants are separate rows sharing one permaslug (and therefore one session
   cost and one weekly volume each), like the Luna/Opus/Fable/Astra ladders.
6. Commented-out rows are not refreshed. Revive one with `--new`; it also needs its
   `aa_price_per_task` filled in (the dataclass constructor enforces it for datacenter
   models), and preferably `or_toks_served`, before the row can be added.

## Provenance rules

- Intelligence, tokens per task and AA cost per task: AA Data API + model page
  payloads, unrounded (`aa-lookup` skill does the fetching).
- OR session cost: `GET /api/frontend/v1/rankings/session-cost`
  (`data.harnesses[].models[].points`, bucket `core` = 10–49 turns). OR splits this per
  coding harness; the plots use the average across the harnesses that carry session
  data for the model (the report marks rows covered by fewer than all 4 harnesses).
  Never substitute listed prices. Bucket labels: `single` = 1 turn, `short` = 2–9, `core` = 10–49, `long` = 50+.
- OR weekly tokens served: `GET /api/frontend/v1/rankings/models?view=week` (`data[]`,
  one daily row per permaslug and variant). Sum `total_prompt_tokens` +
  `total_completion_tokens` over every day and every variant (standard, batch, free).
- Effort variants share one permaslug and therefore one session cost; the
  ×-tokens-per-task factor is what separates them on the plot.
- Keep README.md in sync with any value change (AGENTS.md rule).
