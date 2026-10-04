# AGENTS.md

## Workflow

- **After EVERY change to `plot.py`, no exceptions:**

  a. Run `pixi r plot`
  b. Inspect the generated plots in `plots/` (use `read` tool on the PNG files)
  c. Double-check they look correct: axes, labels, data series, layout, special glyphs.

  Do not consider the task done until the plots have been visually verified.

- Run `pixi r lint` before finishing if you touched any code or markdown. Note: `mdformat`
  rewrites markdown files in place — check `git diff` afterwards, as it may reflow
  README.md more than expected.

## Structure

Single script, `plot.py`. All model data lives in the `MODELS` list; label placement is
automatic, so adding a model = adding one row and re-running. The `PLOTS` list defines
which filtered views get generated.

## Adding or refreshing a model

Follow `.agents/skills/refresh-models/SKILL.md`: it re-fetches the AA and OpenRouter
numbers for every existing model (and prints a ready-to-paste constructor row for a new
one). `.agents/skills/aa-lookup` is the low-level AA query tool it builds on.

Models are `Model(...)` dataclasses: `publisher`, `name`, `intelligence` and
`provider_type` (`ProviderType.LOCAL` or `ProviderType.DATACENTER`) are positional,
every other field is keyword-only. `aa_tok_per_task` is mandatory (AA's *output* tokens
per task). The displayed price per task is derived on demand by
`Model.price_per_task()` (see README for the full rationale):

- `ProviderType.DATACENTER` — AA's token mix for the task, priced at OR's synthetic
  effective prices:

  - `aa_cost_per_task` is AA's cost of one task: a
    `PricedTokens(input_, output, cached_input)` when AA publishes the split
    (`input_` = nonCacheInput + cacheWrite, `output` = reasoning + answer,
    `cached_input` = cacheRead; the three sum to the total), or a bare **float**
    — the total only — for the models AA publishes no breakdown for. A
    `cached_input` of 0.0 means "AA reports no cache reads for this model", which
    is also how the refresh writes the model whose cache price AA leaves unset.
  - `aa_sticker_price=PricedTokens(...)` is the same shape in USD per **1M tokens**: the
    sticker price each stream was billed at. Cost ÷ price recovers the token counts.
  - `or_eff_input_price` / `or_eff_output_price` (USD per 1M tokens) come from OR's
    trailing-week effective-pricing chart: every endpoint instance is priced at its mean
    effective price over the window's days, then instances are averaged weighted by the
    share of the model's tokens they served (free endpoints dropped). Never use OR's
    spot "cheapest provider" price, and not the old quintile/median reduction either —
    it priced providers nobody routed to.
  - The total (`sum()` of the split, or the float) stays as the fallback price when
    either the AA breakdown or the OR price is missing, and as the delta plot's
    baseline; read it with `Model.aa_total_cost_per_task()`.
  - `or_slug`, `or_session_cost_10_49_turns` and `or_toks_served` are still collected
    but no longer price anything; keep them in sync, don't use them.

- `ProviderType.LOCAL` — electricity to generate the task on `hardware` (None =
  RTX3090; `hardware=STRIX_HALO` for the ~120B class). Runtime comes from
  `Model.local_token_counts()`, which zeroes the cached input (free on a local rig)
  and falls back to `LOCAL_INPUT_TOKEN_RATIO ×` output tokens when AA publishes no
  split, times `LocalSpeed(prefill=…, decode=…)`. Both rates are measured on the
  rig; prefill is currently a hand-typed guesstimate at 10× decode — replace it
  with the measured number when there is one. The `⚡` and hardware suffix is
  appended automatically.

A publisher appearing for the first time must be added to `PUBLISHERS` with its
artificialanalysis.ai color, or the script KeyErrors.

## Render markers

Only the `⚡` is still a name string (`__post_init__` splices it into local models) —
do not "clean it up". The other two are boolean fields on `Model`, never text in a
name:

- `⚡` → lightning bolt icon (local electricity cost)
- `trains_on_your_data=True` → thief mask icon (provider trains on your data); excluded
  from the Pareto frontier
- `available=False` → grey strikethrough text; excluded from the Pareto frontier

A publisher appearing for the first time must be added to `PUBLISHERS` with its
artificialanalysis.ai color, or the script KeyErrors.

## Keep README.md in sync

README.md states specific scores, prices, and model lists in prose. Any data change that
contradicts it (new/removed models, changed intelligence or cost) requires updating the
README text, not just the plot.

## Reproducibility gotchas

SVG output is deliberately deterministic (`svg.hashsalt` set, `<dc:date>` stripped) so
unchanged data produces a byte-identical file and git stays clean. Don't remove these
workarounds; a dirty diff on an unchanged-data rerun means something regressed.

matplotlib can't render color emoji — the ⚡ and thief-mask glyphs are hand-built vector
paths (`ICON_PATHS`). If a glyph looks wrong in the output, check the path definition,
not the font.
