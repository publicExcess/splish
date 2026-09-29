# Splish

[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Release](https://img.shields.io/badge/release-splish--v1.1-green.svg)](https://github.com/publicExcess/splish/releases/tag/splish-v1.1)
[![Ko-fi](https://img.shields.io/badge/Ko--fi-support-ff5e5b?logo=ko-fi&logoColor=white)](https://ko-fi.com/severalviolins)

Splish is an **unofficial** fork of [Inco's Splash](https://github.com/incoai/splash), not affiliated with Inco. We
retune and extend Splash's Metal kernels with one piece of hardware in mind: a 40-core **M5 Max** (128 GB, macOS 27).
That device is what we support; where our findings carry over to other Macs, we share them. At present we focus on
**Qwen3.8-27B** and its fine-tunes. Support for **Qwen3.8 Flash-Next** is planned for a future release.

**Why a fork.** Kernel tuning is specific to a chip and an OS: the same M5 Max on macOS 26.6 already ranks some kernels
differently. Rather than ask Inco to maintain hardware-specific work, we keep it here. The general pieces (kernel
choices loaded from a file, the copy rule, the benchmark tools) are available upstream if Inco wants them.

## Results

**Qwen3.8-27B (4-bit) on a 40-core M5 Max, Splish v1.1:**
- **Decode:** 91–178 tok/s for one request, depending on the workload; up to ~400 tok/s in total across 4 requests;
  63–109 tok/s for one request as the context grows from 2K to 128K tokens.
- **Prefill:** ~700–920 tok/s (2K–128K tokens). Splish does not change prefill; it matches Splash.

**Against Splash 1.1.0 as shipped** (same Mac, same models; decode tok/s, stock → Splish; at 2–4 requests the total
across requests):

| Workload | 1 request | 2 requests | 3 requests | 4 requests |
|---|---:|---:|---:|---:|
| Inco's Qwen3.8-27B, long reasoning | 131 → **178** (+35%) | 223 → **292** (+31%) | 222 → **330** (+48%) | 299 → **392** (+31%) |
| Inco's Qwen3.8-27B, short answers | 78 → **99** (+26%) | 134 → **164** (+23%) | 138 → **171** (+24%) | 183 → **210** (+15%) |
| Swift-1.5 (a Qwen3.8-27B fine-tune), long reasoning | 141 → **179** (+27%) | 224 → **296** (+32%) | 224 → **342** (+52%) | 288 → **400** (+39%) |
| Qwen3.6-35B-A3B, long reasoning | 331 → **348** (+5%) | 486 → **573** (+18%) | 553 → **672** (+22%) | 642 → **754** (+18%) |

*Measured on v1.0. v1.1 adds about 4% at one request on the 27B, 3–6% at 2–4 requests, and a further 10–18% at one
request on Qwen3.6-35B-A3B; see [RESULTS.md](RESULTS.md).*

![Decode by context length](docs/m5/charts/context-decode.svg)
![Prefill by context length](docs/m5/charts/context-prefill.svg)

**Where this places Splish:**

| Engine | Decode vs Splish | Prefill vs Splish | Source |
|---|---|---|---|
| Splash 1.1.0 | Splish 1.15–1.52x faster | same | measured, this Mac ([RESULTS.md](RESULTS.md)) |
| oMLX | Splash 1.0 was 1.2–2.4x faster (widening with context) | about the same | measured on an earlier Splash and Swift-1.0 (22 Sep); not yet repeated with Splish |
| MTPLX | not yet measured on Qwen3.8-27B | not yet measured | — |

## Quality

Splish changes kernels, not the model. Outputs can differ from stock at near-ties (sums added in a different order),
so we measure accuracy directly:

| Test | Stock Splash 1.1.0 | Splish |
|---|---:|---:|
| 95-task set (maths, code, reasoning), Inco's Qwen3.8-27B | 95/95 | 95/95 |
| 95-task set, Swift-1.5 | 95/95 | 95/95 (380/380 with four copies at once) |
| 95-task set, Qwen3.6-35B-A3B | not measured | 95/95 |
| 1,621 benchmark items, greedy, Swift-1.5: MMLU-Pro, CMMLU, TruthfulQA, GSM8K, HumanEval, MBPP, long context | 82.8% | 82.7% (4 items differ; p = 0.63) |

## Quick start

You need an Apple silicon Mac with macOS 26.4+, Xcode 26 or newer, and Python 3.12–3.14.

```sh
git clone https://github.com/publicExcess/splish.git && cd splish
make -j4
./splish serve --model mlx-community/Qwen3.8-27B-4bit
```

The first serve downloads the model and its DFlash2 draft and prepares the weights, as in
Splash. `./splish` picks the tuned kernel choices for the model and turns on the copy rule; it
prints what it chose. Then connect an agent (`./splish opencode`, `claude`, `codex`, `hermes`,
`pi`) or any OpenAI- or Anthropic-compatible client, as with Splash
([upstream README](docs/UPSTREAM_README.md)).

The tuned choices are for a **40-core M5 Max on macOS 27**, plus one Qwen3.8-27B table measured on a
**20-core M5 Pro** (contributed by Michael McCrimmons). An independent tuning run on another 40-core
M5 Max on macOS 26.6 picked different winners for some shapes; the gains were in the same direction
but about half the size, which is why choices should be tuned per machine. On any other Mac,
`./splish` keeps Splash's own defaults and still turns on the copy rule.

**Advanced.** `./splish` only sets these when you have not:

| Variable | Effect |
|---|---|
| `SPLASH_KERNEL_CHOICES=FILE` | Kernel choices to load ([tuning/](tuning/)); unset means Splash's defaults |
| `SPLISH_CHOICES=any` | Apply the tuned choices on a chip they were not measured on |
| `SPLASH_M5_COPY_MIN_MATCH=N` | Copy rule: draft verbatim continuations of N+ repeated tokens (default 16; 0 turns it off) |
| `SPLASH_M5_ACCEPT_LOG`, `SPLASH_M5_TOKEN_LOG` | Diagnostics: acceptance histogram and per-step tokens |

## Recommended settings

| Model | Choices file | Notes |
|---|---|---|
| Qwen3.8-27B family, Splash package (Inco's, Swift-1.5, other fine-tunes) | `tuning/m5max-40c-swift15-v12.choices` | Tuned on a 40-core M5 Max. Other M5 chips: run the tuner (`dev/tuning`). |
| Qwen3.8-27B family on a 20-core M5 Pro | `tuning/m5pro-20c-qwen38-27b-hybrid.choices` | Measured by Michael McCrimmons: decode step −6.8% at one request, −6.8 to −8.4% at two, against v8. Picked automatically on a 20-core M5 Pro. |
| Qwen3.8-27B GGUF Q8_0 | `tuning/m5max-40c-swift15-q80-v2.choices` | v2 tunes the DFlash draft too: decode step −2.8% / −2.8% / −4.9% / −6.6% at 1–4 requests |
| Qwen3.6-35B-A3B | `tuning/m5max-40c-qwen36-35b-v2.choices` | v1 (the tuner's set): +5/+18/+22/+18% at 1–4 requests; v2 adds the one-request shapes the tuner left on defaults: a further +10% (greedy) to +18% (steady state) at one request |
| Qwen3.8-27B GGUF Q4_K_M, Q6_K | `tuning/m5max-40c-swift15-kquant-v2.choices` | v2 tunes the DFlash draft too: Q4_K_M decode step −1.7% to −4.5% at 1–4 requests (the G4a kernel does the rest) |

**Copy rule:** on by default through `./splish` (`SPLASH_M5_COPY_MIN_MATCH=16`). It pays off on
agents that rewrite files, never triggers on prose, and costs at most ~3% when it misfires.

Draft length: keep Splash's 7 drafted tokens. On the long-reasoning load 50–62% of verify
steps accept all 7 (45–48% on Qwen3.6-35B). A draft cut to 5 keeps only ~81% of the tokens
per step, more than a shorter verify could save.

Heat: sustained 3–4-request load peaked at 92–94 °C GPU on both engines, with the default fan
behaviour. For long agent sessions, set a fan curve that reaches full speed by 80–85 °C.

## Credits, license and support

- **[Inco](https://github.com/incoai/splash)** built Splash, its models and its DFlash draft models. Splish only
  changes kernels and tuning.
- **Michael McCrimmons ([@mikebuckets171](https://github.com/mikebuckets171))** for the M5 Pro table and the
  Student-t fix in `step_bench.py`.
- **[ashhart/TensorFold](https://github.com/ashhart/TensorFold)** for the copy rule, its benchmark client and its
  recipe for the same model on the same chip; **u/Erp4759**, **u/SnooPredictions515**,
  **[giveen/ninfer-ext](https://github.com/giveen/ninfer-ext)** and **[harryslimes](https://github.com/harryslimes/ninfer-fast/pull/1)**
  for ideas and write-ups we learned from.

Results from other Macs are very welcome, especially where Splish is slower: `python3 dev/m5/report.py --port 8000
--model mlx-community/Qwen3.8-27B-4bit`, pasted into a
[performance report](https://github.com/publicExcess/splish/issues/new?template=performance-report.md).

Apache 2.0, as Splash ([LICENSE](LICENSE), [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES)). Splish's changes are marked
`splash-m5` in the source. Splash is Inco's; Splish is an independent fork, and Inco does not endorse or support it.
If this is useful: [ko-fi.com/severalviolins](https://ko-fi.com/severalviolins).
