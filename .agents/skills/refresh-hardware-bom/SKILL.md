---
name: refresh-hardware-bom
description: Re-prices every component in Hardware_BOM.md from its linked retailer, recomputes each build total, and syncs README.md and data.toml. Use when asked to refresh hardware prices, BOM totals, or the rig prices behind the hardware-cost plot.
---

# Refresh the hardware BOM

`Hardware_BOM.md` holds one table per build from README.md's "Larger local models"
section. Every component line links to the page its price came from, and each table's
**Build total** is the headline price in README.md. The same totals, as whole USD
integers, live in `data.toml`'s `[hardware_prices]`; a model's `hardware_bom` is a sympy
expression over those keys and drives the hardware-cost plot. Mac builds appear in the
README table and the BOM only — no model plots on them, so they have no TOML key.

## Re-price the components

1. Update the "Prices checked" date at the top of `Hardware_BOM.md`.

1. Open each linked page and read its current US price. Prefer the linked retailer and
   the exact variant already in the BOM (capacity, SSD size, cable length). Amazon
   buy-box prices are volatile; take the price a buyer would pay today, not a list
   price, a monthly payment, or a "from" price on a different variant.

1. Keep the same component unless it is out of stock or clearly a worse buy. If a
   switch is needed, change the link and the line text too. Never invent a breakdown
   for a complete system (Bosgame mini PCs, Macs, DGX Sparks): one line is correct.

1. Round every unit price to the nearest whole dollar before multiplying by quantity.
   `data.toml` stores integers only, and README prices are the rounded BOM sums.

1. For the used RTX 3090, use the **trailing 30-day median of eBay.com sold prices**,
   not an ask price and not a 7- or 90-day window:

   ```bash
   curl -sSL https://www.pcprice.watch/data/rtx3090_sales_data.json \
     | python -c 'import json,sys; j=json.load(sys.stdin); p=j["periods"]["30days"]; print(p["median_price"], p["total_sales"], j["last_updated"])'
   ```

   Check the value against the eBay sold-results page linked from the BOM. Reject
   parts-only, heatsink/fan-only, box-only, lot/bundle and large "read description"
   listings before accepting a median; `pcprice.watch`'s query already excludes fan,
   laptop, bracket and motherboard keywords, and a clean-sale comp source (gpusold,
   soldcomps) is the cross-check that no junk sale set the number. Round the accepted
   median to the nearest dollar. Update the RTX 3090 note in `Hardware_BOM.md` with the
   new sale count, date and median when it changes.

1. For a DDR4/DDR5 kit above 48 GB, keep the current 4-stick kit unless a real 2-stick
   UDIMM kit becomes available at retail. Do not substitute ECC RDIMMs: they will not
   boot in the AM4/AM5 boards listed.

## Recompute and sync

1. For each table, recompute `qty × unit price` on every line and the **Build total**.
   A spreadsheet or a short Python script is fine; do not hand-add long rows twice.

1. In `data.toml`, change only the values in `[hardware_prices]`. The keys map to BOM
   builds as follows:

   | TOML key | BOM build |
   | --- | --- |
   | `MOBILE_PHONE` | Mobile phone — 4 GB unified |
   | `STRIX_POINT_16GB` / `STRIX_POINT_32GB` | Ryzen AI 300/400 laptop (Strix/Gorgon Point) — 16/32 GB unified |
   | `RTX_3090_PC_32GB` | PC with RTX 3090 — 24 GB VRAM + 32 GB host |
   | `RTX_5060TI_PC_64GB` | PC with RTX 5060 Ti — 16 GB VRAM + 64 GB host |
   | `RTX_3090_PC_128GB` / `RTX_3090_PC_192GB` | PC with RTX 3090 — 128/192 GB host |
   | `STRIX_HALO` / `GORGON_HALO` | Bosgame M5 / M5 Max mini PC |
   | `DGX_SPARK` | single Asus Ascent GX10 |
   | `QSFP112_CABLE` | 400G QSFP112 DAC, 1 m |
   | `QSFP_DD_SWITCH` | MikroTik 4×400G switch |
   | `QSFP_DD_BREAKOUT_CABLE` | 400G to 2×200G breakout, 1 m |

   The multi-Spark totals are computed from those keys by the `hardware_bom`
   expressions in `data.toml` (e.g. `DGX_SPARK*8+QSFP_DD_SWITCH+QSFP_DD_BREAKOUT_CABLE*4`),
   so never write a separate "8-node" key.

1. Update every price in README.md's "Larger local models" table from the BOM totals,
   including the multi-Spark rows. README prose also hard-codes the used RTX 3090 price
   and the price of the 48 GB RTX 5060 Ti gaming PC; update both from the BOM.

1. Verify the model expressions still evaluate to what you expect:

   ```bash
   pixi run python - <<'PY'
   import tomlkit, sympy
   data = tomlkit.parse(open('data.toml').read())
   prices = {k: int(v) for k, v in data['hardware_prices'].items()}
   for row in data['models']:
       expr = row.get('hardware_bom')
       if expr:
           print(f"{row.get('name', ''):35} {sympy.sympify(expr, locals=prices)}")
   PY
   ```

   Then compare each README price against its BOM total and each plotted rig against
   its key/expression. "Total vs total" means exact integer equality after rounding.

## Finish

Run `pixi r plot` and inspect every PNG in `plots/` as required by AGENTS.md, then
`pixi r lint`, `pixi run mdformat Hardware_BOM.md`, and check `git diff`. Do not run
`mdformat` on SKILL.md files: it destroys their YAML frontmatter. A refresh of
hardware prices must not change any model's intelligence, cost per task, label
placement, or publisher colors.
