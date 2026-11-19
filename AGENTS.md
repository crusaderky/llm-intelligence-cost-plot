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

One plotting script, `plot.py`, and one data file, `data.toml`. All model data, the
publisher colors and the hardware prices live in `data.toml`, loaded and validated with
pydantic at import time; label placement is automatic, so adding a model = adding one row
and re-running. The `PLOTS` list in `plot.py` defines which filtered views get generated.

The `[[models]]` array in `data.toml` holds `kind = "model"` and `kind = "effort"` rows,
converted to `Model | ModelEffort` and then flattened into the `list[Model]` (`MODELS`)
that the plots and the refresh actually read. The AA record name/slug for each row lives
on the row itself (the old `AA_LOOKUPS` table was merged in), with `aa_derived = true`
for rows that must not have their AA values refreshed.

## Label placement

`place_labels` picks every label's slot itself (`CANDIDATES`, in preference order, then
repair rounds and a settle pass). Leave it automatic unless a label genuinely reads as
another model's — then hand-pin that one label in `LABEL_PINS`, keyed by
`(plot stem, drawn label text)`, where the text is the name as that view renders it
_without_ the `⚡`/mask glyph. A pin is hard: that slot is the only one tried, the label
claims it before the automatic ones arrange around it, the settle pass leaves it alone,
and a pinned label is exempt from the height-order rule (its leader line is what asserts
the pairing). Re-check `pixi r plot -- -v` after pinning: it reports residual overlaps,
order and leader/dot grazes, and a pin can force one the other labels cannot escape.

## Adding or refreshing a model

Follow `.agents/skills/refresh-models/SKILL.md`: `pixi r refresh_models` re-fetches the
AA and OpenRouter numbers for every existing model and writes them straight into
`data.toml`. Read the printed change list and `git diff data.toml` before accepting it.
`.agents/skills/aa-lookup` is the low-level AA query tool it builds on.

Each `[[models]]` row holds the fields of one `Model` (or `ModelEffort`) dataclass;
`publisher`, `name`, `intelligence` and `provider_type` are the leading columns. pydantic
validates the file before plotting. The displayed price per task is derived on demand by
`Model.price_per_task()` (see README for the full rationale):

- `ProviderType.DATACENTER` — AA's token mix for the task, priced at OR's synthetic
  effective prices:

  - `aa_cost_per_task` is a three-element TOML array `[input, output, cached_input]`,
    loaded as `PricedTokens(input_, output, cached_input)`
    (`input_` = nonCacheInput + cacheWrite, `output` = reasoning + answer,
    `cached_input` = cacheRead; the three sum to the total). A `cached_input` of
    0.0 means "AA reports no cache reads for this model", which is also how the
    refresh writes the model whose cache price AA leaves unset. It is mandatory:
    the one model AA publishes no breakdown for (K2 Horizon 7B) carries its
    token mix here against a dummy `aa_sticker_price=PricedTokens(1, 1, 1)`, so
    cost ÷ price × 1e6 reads the token count straight back out (0.347 = 347k
    tokens). Its summed total is then a fiction — harmless, because only
    datacenter rows ever fall back to an AA cost.
  - `aa_sticker_price` is the same three-element array in USD per **1M tokens**: the
    sticker price each stream was billed at. Cost ÷ price × 1e6 recovers the token counts.
  - `or_eff_input_price` / `or_eff_output_price` (USD per 1M tokens) come from OR's
    trailing-week effective-pricing chart: every endpoint instance is priced at its mean
    effective price over the window's days, then instances are averaged weighted by the
    share of the model's tokens they served (free endpoints dropped). Never use OR's
    spot "cheapest provider" price, and not the old quintile/median reduction either —
    it priced providers nobody routed to.
  - The total (`sum()` of the split) stays as the fallback price when the OR price
    is missing, and as the delta plot's
    baseline; read it with `Model.aa_total_cost_per_task()`.
  - `or_slug`, `or_session_cost_10_49_turns` and `or_toks_served` are still collected
    but no longer price anything; keep them in sync, don't use them.

- `provider_type = "local"` — electricity to generate the task on the RTX 3090 rig
  (`RTX3090_POWER_DRAW`; every local measurement was taken on that card).
  Runtime comes from `Model.local_token_counts()`, which is `aa_token_counts()`
  with the cached input zeroed (free on a local rig), times
  `local_speed = [prefill, decode]` in tokens/second. Both rates are measured on the
  rig; prefill is currently a hand-typed guesstimate at 10× decode — replace it
  with the measured number when there is one. The `⚡` is appended
  automatically.

- `hardware_bom` — the hardware-cost plot's x, independent of the per-task price: the
  one-off bill of the cheapest rig that runs the model. It is a sympy expression over
  the `[hardware_prices]` keys in `data.toml`, evaluated into `Model.hardware_cost` by
  `__post_init__`:
  `"DGX_SPARK*3+QSFP112_CABLE*3"` is three Sparks cabled together, the same rig as
  `3 * (6000 + 112)`. An unknown name raises instead of pricing at zero. `None` leaves
  the model off that plot.

A publisher appearing for the first time must be added to `[publishers]` in `data.toml`
with its artificialanalysis.ai color, or loading the file raises.

### Effort ladders

A model sold at several thinking levels (low/medium/high/xhigh/max) is one
`kind = "model"` row — any one rung, tag and all — plus one `kind = "effort"` row per
other rung, listed directly after it. An effort row carries only `name`, `intelligence`,
`aa_cost_per_task`, `estimated` and its own AA record fields. `name` is that rung's own
label, `"GPT-6.2 Sol (max)"`:
`expand_model_efforts` splits it with `_split_effort` and matches the stem against the
rows above, so the base row is found by name, not by position. Each rung becomes
`Model.effort()`, which clones the base row swapping the name tag, the intelligence and
the cost split (and `estimated` when passed) and inherits the rest — publisher, provider
type, AA sticker price, OR permaslug and its effective prices, hardware, flags — because a
ladder is one offer burning a different token mix per rung. A rung that differs in
anything `effort()` does not swap is no rung: write it as a full `Model` row, as the
`DeepSeek V4.1 Flash 1017` quantisations are, which change `local_speed` and
`hardware_bom` too. Two rows sharing one stem — the `Qwen3.8-Flash` local/datacenter pair,
the `Muse Spark 1.3` trains-on-your-data pair — leave the base ambiguous, so a
`ModelEffort` naming that stem raises rather than guessing a twin.

A rung inherits `hardware_bom`, so a local ladder would plot the same rig twice.
`local_hardware` sets `one_per_model`, and `_one_per_model` keeps only the highest-effort
rung (rows group by OR permaslug, else by name with the effort tag dropped), which
`label_of=_clean_label` then draws without the tag: one rig, one point, one name. That
dedup is what lets `GLM-5.3-Flash (high)` be a rung at all, next to the `(max)` row that
carries its Strix Halo `hardware_bom`. Only the words in `EFFORT_TAGS` are effort tags,
so the DeepSeek `UD-*` quantisations — different rigs, different points — never collapse.

## Render markers

Only the `⚡` is still a name string (`__post_init__` splices it into local models) —
do not "clean it up". The other two are boolean fields on `Model`, never text in a
name:

- `⚡` → lightning bolt icon (local electricity cost)
- `trains_on_your_data=True` → thief mask icon (provider trains on your data); excluded
  from the Pareto frontier
- `available_datacenter=False` / `available_local=False` → grey strikethrough text in the
  views that flag governs (all price views use `available_datacenter`, the hardware view
  uses `available_local`, via `PlotSpec.available_of`); excluded from the Pareto frontier
  and given a "Not publicly available" legend entry
- `estimated=True` → hollow dot (data not published by AA: extrapolated, or hand-entered)

A view can drop a glyph entirely with `PlotSpec.hidden_icons` (labels and legend entry
alike); the hardware plot hides `bolt` because its filter admits only local models.

A publisher appearing for the first time must be added to `[publishers]` in `data.toml`
with its artificialanalysis.ai color, or loading the file raises.

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
