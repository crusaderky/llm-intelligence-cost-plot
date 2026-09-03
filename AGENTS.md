# AGENTS.md

## Workflow

- **After EVERY change to `plot.py`, no exceptions:**

  a. Run `pixi r plot`
  b. Inspect the generated plots in `plots/` (use `read` tool on the PNG files)
  c. Double-check they look correct: axes, labels, data series, layout, special glyphs.

  Do not consider the task done until the plots have been visually verified.

- Run `pixi r lint` before finishing if you touched any code or markdown.
