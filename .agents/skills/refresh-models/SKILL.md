---
name: refresh-models
description: Refreshes the ArtificialAnalysis (intelligence, output tokens per task, cost per task and its token-type breakdown, sticker prices) and OpenRouter (permaslug, effective input/output prices from the trailing week) numbers behind every model in data.toml, writing the file back in place, or prints a ready-to-paste TOML row for a new model. Use when asked to refresh/re-fetch model scores and prices, or to add a model to the plots.
---

# Refresh or add models in data.toml

`data.toml` is the source of truth for every plotted number, the publisher colors and the
hardware prices. `scripts/plot.py` loads and validates it with pydantic. Each `[[models]]` row
stores AA's Intelligence Index (y axis), AA's output tokens per task (indirectly, via
the cost split), AA's cost per task and its split by token type, the sticker price per 1M
tokens that split was billed at, and the OpenRouter permaslug with its synthetic effective
input/output prices. Each row also
carries its own `aa_page_slug` / `aa_record_name` / `aa_derived` — the old `AA_LOOKUPS`
table was merged into the models. The displayed price per task is derived by
`Model.price_per_task()`:

```text
datacenter: price = (input tokens x OR input $/M + output tokens x OR output $/M) / 1e6,
                    input tokens  = AA $/task[input] / AA sticker[input]
                                  + AA $/task[cached] / AA sticker[cached]
                    output tokens = AA $/task[output] / AA sticker[output]
local:      price = electricity to run those tokens on the local rig
                    (uncached input tokens / prefill tok/s + output tokens / decode tok/s)
```

`aa_cost_per_task` is a three-element TOML array `[input, output, cached_input]` in USD
per task (input includes cache writes, output is reasoning + answer, cached is cache
reads; the three sum to the total). `aa_sticker_price` is the same shape in USD per 1M
tokens. A `cached_input` of 0.0 in either means "AA reports no cache reads for this
model". Every row carries both. The one model AA publishes no split for, `K2 Horizon 7B`,
stores its token mix in the cost slot against a dummy `aa_sticker_price = [1.0, 1.0, 1.0]`,
so cost ÷ price × 1e6 is just the token count (0.347 = 347,000 tokens) and its summed
"cost per task" is a fiction no plot reads.

A datacenter model with no OR price chart or no paid OR endpoint keeps AA's own cost per
task.

## Running the refresh

```bash
pixi r refresh-models             # cached (6h) fetch, writes data.toml
pixi r refresh-models --force     # force re-fetch of AA pages + OR stats
```

Requires `AA_API_KEY` (exported in the shell profile) and the OpenRouter key at
`~/.pi/agent/auth.json` (field `openrouter.key`).

`scripts/refresh_models.py` reads `data.toml` with tomlkit so every
comment survives, re-fetches AA's and OR's numbers, changes only the stale fields, prints
an aligned table of changed fields (with a rounded percentage change where the
value is numeric), and writes `data.toml` back. It never rewrites
`scripts/plot.py`.
Run `pixi r plot` afterwards to see the effect on the displayed prices and PNGs, then
`pixi r lint`.

Rows with `aa_derived = true` and rows pinned to `aa_page_slug = ""` keep their AA
numbers: the derived ones are scaled or extrapolated from another record, and the pinned
ones have no AA record at all. Their OpenRouter fields still refresh. A `# computed from:
...` trailing comment records the hand calculation behind a literal; the refresh removes
that comment when it replaces the value, but keeps every other comment. The `--new` mode
prints a ready-to-paste `[[models]]` block instead of modifying the file:

```bash
pixi r refresh-models --new OR_SLUG --aa-slug AA_SLUG --aa-name "AA Record Name" \
    --publisher "Publisher" --name "Display Name"
```

## Double-check the diff

The script prints what it changed; the agent must still read `git diff data.toml` before
accepting a refresh. Expected in a normal run: sub-unit moves in `intelligence`,
`aa_cost_per_task` and `aa_sticker_price` on AA-scored rows, and OR effective prices (a
trailing-week average).

Treat these as abnormal and investigate before running `pixi r plot`:

- a change to `aa_derived`, `aa_page_slug`, `aa_record_name`, `provider_type`,
  `local_speed`, `hardware_bom`, availability flags, `publisher`, `name`, `[publishers]`
  or `[hardware_prices]`;
- an AA-side change on a derived/manual row or a row AA no longer scores;
- a large jump in intelligence or tokens per task on one model while its siblings stay
  put — check the AA record name in the `!!` warnings on stderr;
- a field disappearing, or a comment other than `# computed from:` being lost;
- a changed OR price when the script warned that the chart was empty (it keeps the last
  stored chart on purpose).

Then check the thresholds in `scripts/plot.py`:

- `LOW_COST_THRESHOLD` must still catch the cheap cluster *and* keep the green
  "cheap AND smart" band non-empty (at least one model with intelligence ≥
  `HIGH_INTELLIGENCE_THRESHOLD` at or below it).
- `HIGH_INTELLIGENCE_THRESHOLD` must still floor the high-intelligence plot.
- If a refresh moves a model across a threshold, update the threshold and say so in the
  commit message.

Then re-read the `## Conclusion` section of README.md: it hard-codes example prices and
ratios that drift with every refresh. Take the current values from the `pixi r plot -v`
price/task table, update each dollar figure and recompute every "X times cheaper" claim
from those same numbers. Check the rounded intelligence scores too, in case an AA score
crosses a rounding boundary.

## Special cases

- A free-only model that starts showing a **paid** OR endpoint must be swapped to the
  paid offer: replace the free permaslug and the zero effective prices with the paid
  endpoint's permaslug and trailing-week prices, so the tankard and the -100% delta
  disappear. A paid offer needs an AA price baseline again. If AA publishes no cost
  split, use OpenRouter's nominal (list) price for the developer's own endpoint as the
  sticker price, guesstimate the token mix / price per task from the closest AA-scored
  sibling, record the calculation as a `# computed from:` comment in `data.toml`, set
  `estimated = true` / `aa_derived = true`, and tell the user exactly how you made the
  guesstimate before accepting the change. Never leave the old zero prices in place.
- A model AA scores but OR does not list can still be plotted on the hardware view:
  give it a `hardware_bom` and `hardware_only = true`, with NaN cost/sticker splits.
  Every price view skips it, so it needs no OR prices; its AA index still refreshes.

- Every value is stored verbatim at full precision (intelligence to 12+ digits, OR prices
  rounded to 4 significant digits before storage), so a tiny nonzero move is real and is
  applied. The derived token count from `aa_cost_per_task ÷ aa_sticker_price` lags AA's
  published integer by a few tenths of a percent because AA rounds the dollars; a
  hand-priced `[1.0, 1.0, 1.0]` split stores exact token counts.
- Ladders: each rung has its own row with its own AA record fields. The rung's
  `intelligence` and `aa_cost_per_task` are refreshed from its record; the sticker prices,
  OR fields and hardware live on the `kind = "model"` base row and refresh there once.
  Effort variants share one OR permaslug.
- `!! no AA index/cost for ...` on stderr means AA has no usable record for that name
  (since the 2026-10 index rebuild it drops the index, task cost and token count for
  models it did not re-benchmark). The row keeps its hand-derived numbers and is marked
  `aa_derived = true` / `estimated = true`; its OR side still refreshes. Rows whose AA
  record is gone *and* whose name would stem into another record must stay pinned to
  `aa_page_slug = ""`.
- `!! AA record ... is now ...` updates `aa_record_name` in the file. Check the change is
  the same model, not the stem fallback picking a sibling.
- `Muse Spark 1.3` has two rows with the same AA record (the `trains_on_your_data` twin)
  but different OR permaslugs; keep their AA numbers in sync.
- `K2 Horizon 7B` is local, AA-published but has no cost split: only its `intelligence`
  and the output leg of its dummy-priced `aa_cost_per_task` refresh. Leave the guessed
  input leg alone. `Ternary-Bonsai-2` is not on AA at all (`aa_page_slug = ""`,
  `aa_derived = true`): nothing on its AA side refreshes.
- The two quantised `DeepSeek V4.1 Flash 1017` rows are `aa_derived = true`: their
  numbers are the datacenter record scaled by Unsloth's recovery and reasoning-length
  ratios. Check that the datacenter row itself did not move, then retype the constants in
  those rows if it did.
- Rows with `estimated = true` render as hollow dots and sit out the Pareto frontier; a
  refresh must not flip that flag.
- Models OR carries no chart for are reported on stderr as `(no usable OR price)` and
  keep their stored prices. `Ling 3.1 Flash 1019` is local-only at AA's price and has
  `available_datacenter = false`.
- A model AA has retired cannot be refreshed: drop the row rather than freezing its
  numbers. `Claude Sonnet 5.5 (low)` was dropped for this reason.

## Adding a new model

1. Find the model on OpenRouter: the permaslug (dated, e.g.
   `openai/gpt-6-astra-20260903`) from `https://openrouter.ai/<slug>` or the catalog API
   (`/api/frontend/v1/catalog/models`). Verify OR publishes a trailing-week effective
   price chart for it: `pixi r refresh-models --force` then check the slug is in
   `.cache/or_effective_pricing.json`. Without one the model plots at AA's own price.
2. Find it on AA and get the exact record name + page slug: `pixi r aa-query --list`
   (see the `aa-lookup` skill for details).
3. Print the TOML row with `pixi r refresh-models --new ...` as shown above.
4. Paste the row into the `[[models]]` array in `data.toml`. The publisher must already be
   in `[publishers]` with its artificialanalysis.ai color, or add it first. The printed
   row already includes `aa_page_slug`, `aa_record_name` and `aa_derived = false`.
5. Effort variants are `kind = "effort"` rows listed after the ladder's `kind = "model"`
   base row, each with its own AA record fields, intelligence and `aa_cost_per_task`.
   They inherit the base's sticker prices, OR fields and hardware through `Model.effort()`;
   `local_hardware` draws a ladder once, at its highest effort. See AGENTS.md's "Effort
   ladders".
6. A model free on every OR endpoint has an empty effective-pricing chart. Write
   `or_eff_input_price = 0.0`, `or_eff_output_price = 0.0` and
   `aa_cost_per_task = aa_sticker_price = [nan, nan, nan]` by hand: no token mix is
   needed when nothing is billed, and the model draws a beer tankard at $0/task.
   `refresh-models` keeps those zero prices when the chart stays empty.
7. A model not on AA (or one AA has retired) needs a hand-entered token mix:
   `[tokens/1e6, tokens/1e6, 0.0]` against `aa_sticker_price = [1.0, 1.0, 1.0]`
   (see `K2 Horizon 7B`) if it is priced by electricity, or a hand-derived real split
   (see `Hy4-flash`) if OR prices it. Set `aa_page_slug = ""` and `aa_derived = true`.
8. Run `pixi r refresh-models` once: a correct new row changes nothing, or only refreshes
   the OR fields that moved between the two fetches.

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
- OR spells a dated release two ways: the permaslug dashes the dots
  (`google/gemini-3-8-flash-20260902`), the effective-pricing chart keeps the dots
  (`google/gemini-3.8-flash-20260902`). `or_slug` stores the dotted form and
  `or_slug_variants` tries both.
- Effort variants share one permaslug; the ×-tokens-per-task factor is what separates
  them on the plot.
- Keep README.md in sync with any value change (AGENTS.md rule). Hardware prices are
  re-priced separately: see `.agents/skills/refresh-hardware-bom/SKILL.md` and
  `Hardware_BOM.md`; this refresh must never change `[hardware_prices]`.
