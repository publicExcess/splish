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

**Splish v1.1** (September 2026) builds on v1.0 with newly tuned kernel choices (faster at 2–4 requests on the
Qwen3.8-27B family and at one request on Qwen3.6-35B-A3B), a tuned table for the 20-core M5 Pro, and fixes for
non-Latin tokenization and very thin images ([RESULTS.md](RESULTS.md)).

**Qwen3.8-27B (4-bit) on a 40-core M5 Max, Splish v1.1:**
- **Decode:** 178–191 tok/s for one long-reasoning request; up to ~415 tok/s in total across 4 requests; 62–110 tok/s
  for one request as the context grows from 2K to 128K tokens.
- **Prefill:** ~575–1,030 tok/s (2K–128K tokens). Splish does not change prefill; it matches Splash.

**Against Splash 1.1.0 as shipped** (same Mac, same models; long-reasoning prompts, 4,096 tokens out, sampled at the
recommended settings; decode tok/s, stock → Splish; at 2–4 requests the total across requests; mean of two rounds,
Splish ahead in every round):

| Workload | 1 request | 2 requests | 3 requests | 4 requests |
|---|---:|---:|---:|---:|
| Inco's Qwen3.8-27B | 134 → **178** (+33%) | 221 → **308** (+39%) | 217 → **332** (+53%) | 284 → **385** (+36%) |
| Swift-1.5 (a Qwen3.8-27B fine-tune) | 139 → **191** (+37%) | 227 → **321** (+41%) | 221 → **353** (+60%) | 290 → **415** (+43%) |
| Qwen3.6-35B-A3B | 347 → **397** (+14%) | 511 → **560** (+10%) | 553 → **677** (+22%) | 664 → **769** (+16%) |

One request by context length (Inco's Qwen3.8-27B, a distinct document of each length, 2,048 tokens out): decode
+26% to +31% from 2K to 64K and +10% at 128K; prefill unchanged. Method and every measurement: [RESULTS.md](RESULTS.md).

![Decode by context length](docs/m5/charts/context-decode.svg)
![Prefill by context length](docs/m5/charts/context-prefill.svg)

## Quality

Splish changes kernels, not the model. In deterministic runs (greedy, thinking off) over 1,621 benchmark items
(MMLU-Pro, CMMLU, TruthfulQA, GSM8K, HumanEval, MBPP, long context), Splish and stock Splash 1.1.0 scored within
0.1 points of each other on Swift-1.5: 82.7% vs 82.8%, with 4 items differing (not significant, p = 0.63). Every
model also passes our 95-task smoke set (95/95).

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
