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

## Adding a model — which constructor to use

Three constructors, each encoding a different cost provenance (see README for rationale):

- `Model(publisher, name, intelligence, cost_per_task)` — datacenter/AA pricing.
- `Model.local(publisher, name, intelligence, tok_per_task, tok_per_sec, hardware=RTX3090)`
  — cost computed as local electricity. Use for sub-35B models; `hardware=STRIX_HALO` for
  the ~120B class. Requires tok/s measured on local hardware, not from AA.
- `Model.reduced_price(...)` — OpenRouter cheapest-provider pricing. Takes the AA
  `nominal_cost_per_task`, the developer's `nominal_price`, and the OpenRouter
  `cheapest_price`; scales each token-type cost by the price ratio. It prints the
  computed cost per task to stdout — sanity-check it.

## Name markers (load-bearing strings)

Markers embedded in model names drive rendering — do not "clean them up":

- `⚡` → lightning bolt icon (local electricity cost)
- `[TRAIN]` → thief mask icon (provider trains on your data); excluded from Pareto frontier
- `[UNAVAILABLE]` → grey strikethrough text; excluded from Pareto frontier

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
