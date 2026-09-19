# LLMs: Intelligence vs. cost

**Last updated:** 2026-09-18

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
costs $0.005 per task and one that costs $0.053, while the same plot contains a model
that costs $4.26 — 850 times as expensive. However, the net result is that the
viewers can no longer appreciate the immensity of the price difference between the cheap
models and the heavy ones; nor can they realize how inconsequential the price
differences are between the cheap models.

The second thing that irks me is that it uses the official baseline pricing from the
model developers' own API offering, with input cache hit rate sampled on the very first
days the model was released. This is, in many cases, vastly different from what the
exact same model can be rented either for from third-party API providers or by using
opt-in service tiers (e.g. OpenAI's `flex`, which costs half the baseline).
[OpenRouter](https://openrouter.ai) makes it very easy to switch providers and service
tiers on the fly and always get the cheapest offer.

The third and final issue is that local models — those that can fit on consumer hardware
— appear on the plot at their datacenter pricing, which is always very expensive in
proportion to the intelligence you buy with it and ultimately not something any real
user will actively want to buy.

## I made my own plots

All intelligence index scores are from ArtificialAnalysis. All points are benchmarked at
maximum thinking effort where not explicitly stated otherwise. All cost scores are from
ArtificialAnalysis too, except where noted below.

In the first plot we see the current offering with the most intelligent (and expensive)
models.

A good rule of thumb for reading the intelligence axis: a one-point difference is
unlikely to be noticeable by most, while a 5-point gap is substantial. It's important to
point out that an intelligence score of 32, which is the rock bottom in this first plot,
is roughly what the smartest model in the world could deliver in February 2026 (Opus
4.6).

Models marked with a thief mask symbol
(<img src="static/thief_mask.png" alt="[thief mask]" height="15" style="vertical-align: middle">)
train on your data and you should not use them for anything that you would not
want to become publicly available on the internet.

<a href="https://raw.githubusercontent.com/crusaderky/llm-intelligence-cost-plot/main/plots/high_intelligence.svg"><img src="plots/high_intelligence.png" alt="Intelligence vs. Cost per Task (High Intelligence)"></a>

The green area at the bottom left is where models become _extremely_ cheap. Let's zoom
into it and extend the intelligence plot a bit lower, down to what can run today on a
smartphone.

Some models are marked with a yellow lightning-bolt symbol (⚡). It means that the cost
was calculated as the electricity to run the model locally (details on the calculation
below), since the model is so small that it makes no sense to serve it from a
datacenter. When comparing local models against each other, it also offers a scale of
how long each model takes to complete tasks.

<a href="https://raw.githubusercontent.com/crusaderky/llm-intelligence-cost-plot/main/plots/low_cost.svg"><img src="plots/low_cost.png" alt="Intelligence vs. Cost per Task (Low Cost)"></a>

Finally, let's merge the two plots together to better visualize the diminishing returns
in performance/cost. Again, the area that's common to all plots is highlighted in green:

<a href="https://raw.githubusercontent.com/crusaderky/llm-intelligence-cost-plot/main/plots/all_models.svg"><img src="plots/all_models.png" alt="Intelligence vs. Cost per Task (All Models)"></a>

## All the differences between AA's plot and mine

- Changed x scale from logarithmic to linear, because people's money is not logarithmic
- Recalculated price of all datacenter models to use the cheapest effective price currently
  available on OpenRouter (excessively slow or unreliable providers are excluded).  
- Extrapolated points for GLM-5.3-Flash at high reasoning effort, by crossing AA scores
  at max effort with [Z.ai's coding scores](https://z.ai/blog/glm-5.3-flash) at
  different effort levels
- Changed sub-35-billion-parameter models from datacenter pricing to cost to run locally
  (read below)
- Added [Occamy-1.0](https://huggingface.co/Accio-Lab/occamy-1.0). The intelligence
  score is extrapolated from _self-reported_ benchmark results by the model authors and
  should be taken with a healthy dose of skepticism.

## Cost calculation for datacenter models

Cost per task for models served by datacenters was calculated as follows:

- From ArtificialAnalysis, read the breakdown of the cost per task (input, output, cached
  input) and the pricing from which it was calculated
- From the model's page on OpenRouter, discard anomalously poor performers on

  - GPQA Diamond
  - TAU-Bench
  - Tool Call Error Rate
  - Structured Output Error Rate
  - Uptime

  Include opt-in service tiers, like OpenAI's `flex`. Providers that train on your data
  are excluded and appear on the plot as a separate entry (e.g. Muse Spark).

- Note down Effective in and Effective out.
- Find the cheapest overall provider assuming 99%/1% input/output rate.
- Calculate

  ```python
  input tokens = (
    input price per task / AA input price 
    + cached input price per task / AA cached input price
  )
  output tokens = output price per task / AA output price
  openrouter price per task = (
    input tokens * effective input price
    + output tokens * effective output price
  )
  ```

## Cost calculation for local models

Cost per task for models marked with the lightning-bolt symbol (⚡) was crudely
calculated as follows:

- Take Output tokens per task [from
  artificialanalysis.ai](https://artificialanalysis.ai/#intelligence-comparison-tabs)
- Crudely observe decode speed (tok/s) on local hardware. Most measurements were taken
  on the same RTX 3090 video card from 2020, which today is relatively affordable at
  ~$1,400 (used).
- Measure delta between peak and idle energy draw on said hardware
- Price electricity at $0.2049/kWh, which is the US residential electricity price,
  weighted average by population, as of May 2026.
- Add 15% (finger-in-the-air) for uncached input tokens and waiting for tool calls
- Hardware is priced at zero, on the basis that both an RTX 3090 PC and a 64GB Strix
  Halo are desirable gaming/work machines anyways.

Note that there isn't a material difference in electricity costs between different
hardware platforms: a Strix Halo draws less power than an RTX 3090, but it's slower so
it needs to run longer to complete the same tasks.

## Larger local models

Not including the cost of hardware stops being defensible once you upgrade beyond 64 GB
RAM, as almost nobody needs that much RAM if not for AI.

Qwen3.8-Flash needs, as a minimum, a 128GB Strix Halo; it is shown on the plot as priced
by datacenters as well as the electricity it costs to run it locally; however the latter
already hides a substantial expense for hardware: a 64 GB Strix Halo, which is a very
desirable general purpose mini PC, costs $2,000; a 128 GB one costs $3,600 and doesn't
enable anything other than AI models in the ~120B-parameter class.

The following models _can_ be run locally, but carry a very steep up-front hardware
cost:

| Memory | Hardware | Models |
| --- | --- | --- |
| 128 GB RAM | Strix Halo ($3,600)<br>DGX Spark ($4,300)<br>Mac Studio M5 Max ($5,100)<br>MacBook Pro M5 Max ($7,150) | Qwen3.8-Flash<br> |
| 256 GB RAM | 2x DGX Spark ($8,700)<br>Mac Studio M5 Ultra ($11,300) | GLM-5.3-Flash |
| 384 GB RAM | 3x DGX Spark ($13,200)<br> | DeepSeek-V4.1-Flash |
| 512 GB RAM | 4x DGX Spark + QFP28 switch($18,400)<br>2x Mac Studio M5 Ultra ($22,600) | GLM-5.3 |
| 2 TB RAM | 2x TensTorrent Galaxy Blackhole ($320,000) | Kimi K3<br>Qwen3.8-Max |

## Conclusion

There is an immense difference in cost between the state-of-the-art models from
Anthropic and OpenAI and the much cheaper Chinese models: the former are [too expensive
even for large
corporations](https://www.tomshardware.com/tech-industry/artificial-intelligence/ai-cost-crisis-hits-tech-giants-as-employee-tokenmaxxing-backfires-agentic-ai-eats-up-to-1000x-more-tokens-than-standard-ai-sparks-corporate-pullback-at-microsoft-meta-and-amazon),
while the latter can be as cheap as a mobile phone subscription.

How much extra intelligence emptying the wallet purchases obeys the law of diminishing
returns: while a top-tier engineer or scientist is probably going to be able to
appreciate how much better Fable 5.1 (intelligence score 53, $4.26 per task) is compared
to GLM-5.3 (intelligence 45, $1.46 — almost 3x cheaper), most people will have a hard
time doing so. Going further down, GLM-5.3-Flash at high settings (intelligence 41,
$0.11 — almost _fourty times_ cheaper than Fable) is visibly less capable when you give
it very sophisticated tasks, like one-shotting a whole coding project on its own, but it
remains _enough_ for 90% of what people actually need. Even the highly specialized
engineers and scientists mentioned above don't actually need the extra intelligence for
a lot of what they do. Descending just a little bit further, an enthusiast gamer can run
Qwen3.8-27B (intelligence 34, $0.023 in electricity) on a computer they already own.
