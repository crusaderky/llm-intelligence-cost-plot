# LLMs: Intelligence vs. cost

**Last updated:** 2026-10-01

[ArtificialAnalysis](https://artificialanalysis.ai) is a website that benchmarks the
intelligence of various LLM models. They publish a headline _Intelligence Index_, which
is calculated as the mean output of the curated selection of benchmarks they run on each
model. It's a decent finger-in-the-air measure of how smart a model is overall.

AA also records useful information — namely, how much it cost them to run the
benchmarks. Since the benchmarks are the same across all models, this offers a good
indicator of how much it will cost a user to run each model, in relative terms.

One of their main plots is the [Intelligence vs. cost
plot](https://artificialanalysis.ai/#intelligence-comparison-tabs), which shows the
Pareto frontier, i.e. the cheapest model that can achieve each intelligence score. This
frontier is important, because using a super-intelligent and super-expensive model to
accomplish menial tasks that could be done by a much dumber and cheaper one is just a
waste of money.

Over time, I've become progressively more irritated by this plot, for a few reasons.

## Why AA's plot is misleading

The first issue I have with it is that it uses a logarithmic scale on the cost axis.
Using a log scale is the only way to make you spot the difference between a model that
costs $0.010 per task and one that costs $0.019, while the same plot contains a model
that costs $7.63 — over 700 times as expensive. However, the net result is that the
viewers can no longer appreciate the immensity of the price difference between the cheap
models and the heavy ones; nor can they realize how inconsequential the price
differences are between the cheap models.

The second thing that irks me is that it calculates the cost of each task using the
official pricing from the model developers' own API offering. Posted prices are
routinely undercut by what customers actually pay: input cache hit rates fluctuate
substantially over time, some providers sell the exact same weights much cheaper, and
open-weights models in particular can be rented for a fraction of the list price.
[OpenRouter](https://openrouter.ai) publishes, for every provider that serves a model,
what its customers _actually_ pay per input and per output token; this is is what my
plots are built on.

The third and final issue is that local models — those that can fit on consumer hardware
— appear on the plot at their datacenter pricing, which is always very expensive in
proportion to the intelligence you buy with it and ultimately not something any real
user will actively want to buy.

## I made my own plots

All intelligence index scores are from ArtificialAnalysis. All points are
benchmarked at maximum thinking effort where not explicitly stated otherwise.

The X axis shows the cost of one Artificial Analysis Intelligence Index task, derived
from AA's benchmark task size and the prices OpenRouter's users actually pay for the
same tokens, rather than from the developers' sticker prices. Later in the document
I explain how I calculated these numbers.

In the first plot we see the current offering with the most intelligent (and expensive)
models.

A good rule of thumb for reading the intelligence axis: a one-point difference is
unlikely to be noticeable by most, while a 5-point gap is substantial. It's important to
point out that an intelligence score of 39, which is the rock bottom in this first plot,
is above what the smartest models in the world could deliver in April 2026 (Opus 4.7 and
GPT-5.5, score 38).

Models marked with a thief mask symbol
(<img src="static/thief_mask.png" alt="[thief mask]" height="15" style="vertical-align: middle">)
train on your data and you should not use them for anything that you would not
want to become publicly available on the internet.

Models in grey with a strikethrough are not offered to the public yet.

Models that ArtificialAnalysis has not measured are drawn as hollow circles; their
positions are extrapolated or estimated (the details are in the list of differences
below).

<a href="https://raw.githubusercontent.com/crusaderky/llm-intelligence-cost-plot/main/plots/high_intelligence.svg"><img src="plots/high_intelligence.png" alt="Intelligence vs. Price per Task (High Intelligence)"></a>

The green area at the bottom left is where models become _extremely_ cheap. Let's zoom
into it and extend the intelligence plot a bit lower, down to what can run today on a
smartphone.

Some models are marked with a yellow lightning-bolt symbol (⚡). It means that the cost
was calculated as the electricity to run the model locally (details on the calculation
below), since the model is so small that it makes no sense to serve it from a
datacenter. When comparing local models against each other, it also offers a scale of
how long each model takes to complete tasks.

<a href="https://raw.githubusercontent.com/crusaderky/llm-intelligence-cost-plot/main/plots/low_cost.svg"><img src="plots/low_cost.png" alt="Intelligence vs. Price per Task (Low Cost)"></a>

Let's merge the two plots together to better visualize the diminishing returns in
performance/cost. Again, the area that's common to all plots is highlighted in green.
**CLICK TO EMBIGGEN** to admire it in all its zoomed-in glory:

<a href="https://raw.githubusercontent.com/crusaderky/llm-intelligence-cost-plot/main/plots/all_models.svg"><img src="plots/all_models.png" alt="Intelligence vs. Price per Task (All Models)"></a>

This is how much each datacenter model's real price per task deviates from the price
ArtificialAnalysis published for it. To the left of zero the OpenRouter reality is
cheaper than AA's sticker price, to the right it is more expensive. Local models are
absent as they have no sticker price to compare with.

<a href="https://raw.githubusercontent.com/crusaderky/llm-intelligence-cost-plot/main/plots/price_delta.svg"><img src="plots/price_delta.png" alt="Intelligence vs. Δ from AA's Price per Task"></a>

## All the differences between AA's plot and mine

- Changed x scale from logarithmic to linear, because people's money is not logarithmic
- Replaced AA's cost-per-task (measured by AA using the developers' first-party infra
  early in the model's lifetime) with the real up-to-date cost of the same task as
  measured on OpenRouter, described below
- Changed sub-200-billion-parameter models from datacenter pricing to cost to run locally
  (read below)
- Extrapolated GLM-5.3-Flash at high reasoning effort, by crossing AA scores at max
  effort with [Z.ai's coding scores](https://z.ai/blog/glm-5.3-flash) at different
  effort levels
- Extrapolated Ternary-Bonsai-2's intelligence as ~92% of Qwen3.8-27B's, as reported
  by [ByteShape](https://byteshape.com/blogs/Qwen3.8-27B/)
- Extrapolated Hy4 preview, which ArtificialAnalysis has not measured yet, from
  [Tencent's own agentic-benchmark chart](https://hy.tencent.ai/research/hy4-preview).
  Its intelligence score was calculated by mapping the delta on the _self-published_
  benchmarks onto the AA Intelligence Index of the models it is compared against.
- Extrapolated [Occamy-1.0](https://huggingface.co/Accio-Lab/occamy-1.0), in the same
  way as Hy4.

## Cost calculation for datacenter models

The **price per task** is AA's own benchmark task, priced at what OpenRouter's
customers actually pay for the same tokens. Two steps: work out what the task
spends in tokens, then price those tokens.

**Step 1 — the task's token mix.** AA publishes the cost of one Intelligence Index task
split by token type — uncached input (including cache writes), cache reads, and output
(reasoning plus the final answer) — and the sticker price per token that each stream
was billed at. Dividing one by the other recovers how many tokens of each kind the
task burns:

```text
input tokens  = AA $/task[input] + AA $/task[cached input], each divided by its own sticker price
output tokens = AA $/task[output] / AA sticker output price
```

An agentic task caches aggressively, so most input tokens are cache hits. Opus 5.5 at
max effort, for example, spends 12.1M cached input tokens against 294k uncached ones.

**Step 2 — OpenRouter's real prices.** The spot price of the cheapest provider is far
too volatile to plot (it moves wildly within hours), so I use a rolling average over the
last week instead. For each day, and for each provider serving the model that day,
OpenRouter's _effective_ price — what that provider's traffic actually pays per input
and per output token, after its cache discounts — is reduced to the first quintile
across providers, and the median across the days of the week gives the model's price.
Finally:

```text
price per task = (input tokens × OR effective input price + output tokens × OR effective output price) / 1e6
```

Note that OR's effective input price is itself already net of that provider's cache hit
rate, so it is applied to the whole input side rather than to cache hits alone. The
result is a real, absolute price for the same task AA benchmarks: **what you pay on
OpenRouter for the tokens that AA's task actually consumes.**.

A model OR carries no price history for, or one AA publishes no breakdown for, keeps
AA's own cost per task.

## Cost calculation for local models

Price per task for models marked with the lightning-bolt symbol (⚡) was crudely
calculated as follows:

- Take the tokens AA's benchmark task uses: always the output tokens per task [from
  artificialanalysis.ai](https://artificialanalysis.ai/#intelligence-comparison-tabs),
  plus the uncached input tokens from the same breakdown the datacenter models use.
  Cache hits are free on your own machine, so they are excluded.
- AA publishes no breakdown for a few of these models (MiniCPM5-2B, K2 Horizon 7B,
  Occamy-1.0, Ternary-Bonsai-2). For those, the uncached input tokens are estimated as
  3.78× the output tokens, which is the median of that ratio over the 34 models AA
  does break down. It is a crude guess: the real ratio is about 2 for a max-effort
  task and about 13 for a low-effort one.
- Crudely observe decode speed (tok/s) on local hardware. Most measurements were taken
  on the same RTX 3090 video card from 2020, which today is relatively affordable at
  ~$1,400 (used). Prefill speed is not published or measured anywhere, so it is
  guesstimated at 10× the decode speed; on these task sizes that makes prefill a minor
  addition to the runtime.
- Measure delta between peak and idle energy draw on said hardware
- Price electricity at $0.2049/kWh, which is the US residential electricity price,
  weighted average by population, as of May 2026.
- Hardware is priced at zero, on the basis that both an RTX 3090 PC and a 64GB Strix
  Halo are desirable gaming/work machines anyways.

Note that there isn't a material difference in electricity costs between different
hardware platforms: a Strix Halo draws less power than an RTX 3090, but it's slower so
it needs to run longer to complete the same tasks.

## Larger local models

Not including the cost of hardware stops being defensible once you upgrade beyond 64 GB
RAM, as almost nobody needs that much RAM if not for AI.

Qwen3.8-Flash needs, as a minimum, a 128GB Strix Halo to run locally. It appears twice
on the plot: once with datacenter pricing and once with the ⚡ symbol for local
electricity cost. The latter hides a substantial hardware expense: a 64 GB Strix Halo,
which is a very desirable general purpose mini PC, costs $2,200; a 128 GB one costs
$3,800 and doesn't enable anything other than AI models in the ~120B-parameter class.

The following models _can_ be run locally, but their up-front hardware cost grows very
rapidly:

| RAM requirements | Hardware | Price | Models |
| --- | --- | --- | --- |
| 4 GB | Mobile phone | ~$250 | MiniCPM5-2B |
| 12 GB | PC with RTX 3080 Ti | ~$1,500 (used) | Ternary-Bonsai-2<br>Occamy-1.0 |
| 24 GB | PC with RTX 3090 | ~$2,300 (used) | Qwen-3.8-27B |
| 96 GB | Strix Halo 128 GB<br>DGX Spark (128 GB)<br>Mac Studio M5 Max 128 GB<br>Mac Studio M5 Ultra 96 GB<br>MacBook Pro M5 Max 128 GB<br>PC with RTX 6000 Pro | $3,800<br>$5,000<br>$5,100<br>$5,400<br>$7,000<br>~$16,000 | Qwen3.8-Flash<br> |
| 192 GB | Gorgon Halo | $6,800 | MiMo-v2.6-Flash |
| 256 GB | 2x DGX Spark<br>Mac Studio M5 Ultra 256 GB | $10,200<br>$11,300 | GLM-5.3-Flash |
| 280 GB | 3x DGX Spark (384 GB) | $15,300 | DeepSeek-V4.1-Flash |
| 512 GB | 4x DGX Spark + QFP28 switch<br>2x Mac Studio M5 Ultra 256 GB<br>Mac studio M5 Ultra 512 GB | $21,200<br>$22,600<br>T.B.A. | GLM-5.3<br>Hy4 preview |
| 640 GB | 5x DGX Spark + 2x QFP28 switch | $27,200 | MiMo-v2.6-Pro |
| 2 TB | 2x TensTorrent Galaxy Blackhole | $320,000 | Kimi K3 |

The same models on a hardware-cost axis instead of a per-task one. Each point
sits at the cheapest rig in the table above that can run the model, so the
electricity prices on the other plots — fractions of a cent a task — vanish next
to the machine. The curve is over almost immediately: $250 of hardware buys
more intelligence than $27,000 does, and Kimi K3 is off the right edge of the
plot entirely, needing $320,000 of hardware for less intelligence than
MiMo-V2.6-Pro delivers for $27,200. The x axis breaks before it, so the left
panel is $0-$30,000 and the right panel holds Kimi K3 alone, on its own scale.

<a href="https://raw.githubusercontent.com/crusaderky/llm-intelligence-cost-plot/main/plots/local_hardware.svg"><img src="plots/local_hardware.png" alt="Intelligence vs. Hardware Price (Larger Local Models)"></a>

## Conclusion

There is an immense difference in cost between the state-of-the-art models from
Anthropic and OpenAI and the much cheaper Chinese models: the former are [too expensive
even for large
corporations](https://www.tomshardware.com/tech-industry/artificial-intelligence/ai-cost-crisis-hits-tech-giants-as-employee-tokenmaxxing-backfires-agentic-ai-eats-up-to-1000x-more-tokens-than-standard-ai-sparks-corporate-pullback-at-microsoft-meta-and-amazon),
while the latter can be as cheap as a mobile phone subscription.

How much extra intelligence emptying the wallet purchases obeys the law of diminishing
returns: while a top-tier engineer or scientist is probably going to be able to
appreciate how much better Claude Opus 5.5 at max effort (intelligence score 58,
$11.13/task) is compared to GPT Sol-6.1 at xhigh effort (intelligence 51, $0.39/task — 28x
cheaper), most people will have a hard time doing so. Going further down, GLM-5.3-Flash
at high effort (intelligence 40, $0.18/task — over _sixty times_ cheaper than Opus
5.5) is visibly less capable when you give it very sophisticated tasks, like
one-shotting a whole coding project on its own, but it remains _enough_ for 90% of what
people actually need. Even the highly specialized engineers and scientists mentioned
above don't actually need the extra intelligence for a lot of what they do. Descending
just a little bit further, an enthusiast gamer can run Qwen3.8-27B (intelligence 34,
$0.06/task in electricity) on a computer they already own.
