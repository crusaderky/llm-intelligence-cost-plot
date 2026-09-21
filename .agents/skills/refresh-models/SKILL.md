---
name: refresh-models
description: Refreshes the ArtificialAnalysis (intelligence, output tokens per task) and OpenRouter (permaslug, avg 10-49-turn session cost) numbers behind every model in plot.py, or prints a ready-to-paste constructor row for a new model. Use when asked to refresh/re-fetch model scores and prices, or to add a model to the plots.
---

# Refresh or add models in plot.py

`plot.py` stores, per model: AA's Intelligence Index (y axis), AA's output tokens per
task, the OpenRouter permaslug, and OpenRouter's median cost of a 10-49-turn session
(the "core" bucket, averaged across OR's coding harnesses). The reference cost is computed
inside `Model.datacenter` as

```
reference cost = OR session cost (10-49 turns) x AA output tokens per task / HOUR_SCALE
```

Local (⚡ electricity) models store the same AA fields but no OR fields; their reference
cost is electricity × a hardcoded GPT-5.6 Luna (max) anchor (see `Model.local`).

## Running the refresh

```bash
pixi r refresh-models             # cached (6h) fetch, prints old -> new table
pixi r refresh-models --refresh   # force re-fetch of AA pages + OR stats
```

Requires `AA_API_KEY` (exported in the shell profile) and the OpenRouter key at
`~/.pi/agent/auth.json` (field `openrouter.key`).

The script prints one row per `MODELS` entry — old vs new intelligence, tokens per task,
OR session cost, and reference cost — plus reminders. It **never edits `plot.py`**:
apply the changes yourself, then follow the AGENTS.md workflow (`pixi r plot`, visually
inspect the PNGs, `pixi r lint`).

## Reading the report

- `INT!` / `TOK!` / `REF!` — intelligence, tokens per task, or reference cost drifted
  beyond the round-off threshold. Values usually match what is already in `plot.py`;
  only drifted rows need edits.
- The session-cost statistic is a **30-day trailing median, and OR only rolls its
  window periodically** (observed `windowEnd` frozen for 8+ days at 2026-09-13).
  Between window rolls a refresh returns byte-identical values; a `REF!` flag therefore
  signals a real roll, not sampling noise. Conversely, the stat is a lagging indicator:
  new models and price changes take days to weeks to show up. The old
  avg-price-per-100-requests statistic this replaced was recomputed from same-day
  traffic and swung 30-50% within hours.
- Reminder lines cover the special cases:
  - `GLM-5.3-Flash (high)` is extrapolated from the `(max)` record — multiply
    intelligence by 28.01/28.99 and tokens by 70610/138690 (the ratios themselves are
    historical, from Z.ai's coding scores and AA's GLM-5.3 effort split).
  - `Muse Spark 1.3 [TRAIN]` must always get the same numbers as `Muse Spark 1.3`.
  - Local models: update `tok_per_task` and intelligence; keep tok/s and `hardware`
    (measured on real hardware, never from AA). `Occamy-1.0` is not on AA at all —
    manual entry, no refresh.
- The script also prints the **GPT-5.6 Luna (max) anchor** — the three numbers hardcoded
  inside `Model.local`. Update them in the same commit as everything else.
- Commented-out entries (`Qwen3.8 2.4T A95B`, `GPT-5.6 Terra (max)`, `StepFun Step 5
  Preview`) are not refreshed; if reviving one, use `--new` (below). Qwen3.8 2.4T A95B
  and Step 5 have no 10-49-turn session data on any OR harness, so they cannot come
  back under the current cost design.

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
5. Effort variants are separate rows sharing one permaslug (one `costPerRequest`
   value each), like the Luna/Opus/Fable/Astra ladders.

## Provenance rules

- Intelligence + tokens per task: AA Data API + model page payloads, unrounded
  (`aa-lookup` skill does the fetching).
- OR session cost: `GET /api/frontend/v1/rankings/session-cost`
  (`data.harnesses[].models[].points`, bucket `core` = 10–49 turns). OR splits this per
  coding harness; the plots use the average across the harnesses that carry session
  data for the model (the report marks rows covered by fewer than all 4 harnesses).
  Never substitute listed prices. Bucket labels: `single` = 1 turn, `short` = 2–9, `core` = 10–49, `long` = 50+.
- Effort variants share one permaslug and therefore one session cost; the
  ×-tokens-per-task factor is what separates them on the plot.
- Keep README.md in sync with any value change (AGENTS.md rule).
