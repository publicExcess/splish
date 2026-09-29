#!/usr/bin/env python3
"""Splish (splash-m5) write-up charts: dependency-free SVG bar charts into docs/m5/charts/.

The data below is copied from our measurement log (every number was measured there). Charts
follow GitHub's light or dark theme through prefers-color-scheme inside the SVG.

  python3 dev/m5/charts.py
"""
import os

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "docs/m5/charts")
STYLE = """<style>
  .bg{fill:#ffffff} .t{fill:#1f2328;font:600 15px -apple-system,Segoe UI,Helvetica,Arial,sans-serif}
  .l{fill:#57606a;font:12px -apple-system,Segoe UI,Helvetica,Arial,sans-serif}
  .v{fill:#1f2328;font:11px -apple-system,Segoe UI,Helvetica,Arial,sans-serif}
  .g{stroke:#d0d7de;stroke-width:1} .gain{fill:#1a7f37;font:600 11px -apple-system,Segoe UI,Helvetica,Arial,sans-serif} .s0{fill:#8c959f} .s1{fill:#0969da} .s2{fill:#1a7f37} .s3{fill:#bf8700}
  @media (prefers-color-scheme: dark){
    .bg{fill:#0d1117} .t,.v{fill:#e6edf3} .l{fill:#8d96a0} .g{stroke:#30363d} .gain{fill:#3fb950}
    .s0{fill:#6e7681} .s1{fill:#4493f8} .s2{fill:#3fb950} .s3{fill:#d29922}}
</style>"""


def bars(name, title, groups, series, values, unit, fmt="{:.0f}", note="", gains=False):
    """Grouped vertical bars: values[series][group]."""
    width, height, left, top, bottom = 760, 380, 60, 56, 70
    plot_w, plot_h = width - left - 20, height - top - bottom
    top_value = max(v for row in values for v in row if v is not None) * 1.1
    # The smallest clean step whose 4 or 5 ticks reach the tallest bar (the axis starts at zero).
    peak, step = min((m * 10 ** e * n, m * 10 ** e) for e in range(-2, 6)
                     for m in (1, 1.5, 2, 2.5, 3, 4, 5, 6, 8) for n in (4, 5) if m * 10 ** e * n >= top_value)
    ticks = round(peak / step)
    gw = plot_w / len(groups)
    bw = min(46, gw * 0.8 / len(series))
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           STYLE, f'<rect class="bg" width="{width}" height="{height}" rx="8"/>',
           f'<text class="t" x="{left}" y="28">{title}</text>']
    for i in range(ticks + 1):
        y = top + plot_h - plot_h * i / ticks
        svg.append(f'<line class="g" x1="{left}" x2="{width - 20}" y1="{y:.1f}" y2="{y:.1f}"/>')
        svg.append(f'<text class="l" x="{left - 8}" y="{y + 4:.1f}" text-anchor="end">{step * i:g}</text>')
    svg.append(f'<text class="l" x="14" y="{top + plot_h / 2}" transform="rotate(-90 14 {top + plot_h / 2})" '
               f'text-anchor="middle">{unit}</text>')
    for g, group in enumerate(groups):
        x0 = left + g * gw + (gw - bw * len(series)) / 2
        for s in range(len(series)):
            v = values[s][g]
            if v is None:
                continue
            h = plot_h * v / peak
            x, y = x0 + s * bw, top + plot_h - h
            svg.append(f'<rect class="s{s % 4}" x="{x:.1f}" y="{y:.1f}" width="{bw - 3:.1f}" height="{h:.1f}" rx="2"/>')
            svg.append(f'<text class="v" x="{x + (bw - 3) / 2:.1f}" y="{y - 4:.1f}" text-anchor="middle">{fmt.format(v)}</text>')
            if gains and s == len(series) - 1 and values[0][g]:
                change = 100 * (v / values[0][g] - 1)
                svg.append(f'<text class="gain" x="{x + (bw - 3) / 2:.1f}" y="{y - 18:.1f}" '
                           f'text-anchor="middle">{change:+.0f}%</text>')
        svg.append(f'<text class="l" x="{left + g * gw + gw / 2:.1f}" y="{top + plot_h + 18}" text-anchor="middle">{group}</text>')
    lx = left
    for s, label in enumerate(series):
        svg.append(f'<rect class="s{s % 4}" x="{lx}" y="{height - 30}" width="12" height="12" rx="2"/>')
        svg.append(f'<text class="l" x="{lx + 18}" y="{height - 20}">{label}</text>')
        lx += 30 + 7 * len(label)
    if note:
        svg.append(f'<text class="l" x="{width - 20}" y="{height - 20}" text-anchor="end">{note}</text>')
    svg.append("</svg>")
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name + ".svg"), "w") as f:
        f.write("\n".join(svg) + "\n")


def main():
    # Official Qwen3.8-27B package, stock 1.1.0 vs fork (v8), greedy, 512 tokens, mean of 2 rounds.
    bars("official-27b-concurrency", "Inco's Qwen3.8-27B: stock Splash 1.1.0 vs Splish (greedy)",
         ["1 request", "2 requests", "3 requests", "4 requests"], ["stock 1.1.0", "Splish"],
         [[78.2, 133.7, 137.8, 182.6], [98.8, 163.9, 170.8, 209.7]], "aggregate tok/s",
         note="512 tokens, mean of 2 rounds", gains=True)
    bars("official-27b-concurrency-sampled", "Inco's Qwen3.8-27B: stock vs Splish (sampled)",
         ["1 request", "2 requests", "3 requests", "4 requests"], ["stock 1.1.0", "Splish"],
         [[73.8, 118.2, 124.9, 163.4], [90.5, 151.3, 171.2, 208.6]], "aggregate tok/s",
         note="temperature 1.0, top_p 0.95, top_k 20", gains=True)
    # Single request across models and workloads, stock 1.1.0 vs Splish (README headline table).
    bars("single-request", "One request: stock Splash 1.1.0 vs Splish, by model",
         ["27B, long reasoning", "27B, short answers", "27B, 128K document", "Swift-1.5, reasoning",
          "35B-A3B, reasoning"], ["stock 1.1.0", "Splish"],
         [[131.3, 78.2, 54.0, 140.5, 330.6], [177.7, 98.8, 62.9, 179.0, 348.3]], "decode tok/s",
         note="27B = Inco's Qwen3.8-27B; 4-bit; 40-core M5 Max", gains=True)
    # By concurrent requests (the steady-state data, titled for the split).
    bars("concurrency", "1-4 concurrent requests: stock Splash 1.1.0 vs Splish",
         ["27B, 1", "27B, 2", "27B, 3", "27B, 4", "Swift, 1", "Swift, 2", "Swift, 3", "Swift, 4"],
         ["stock 1.1.0", "Splish"],
         [[131.3, 223.2, 222.2, 299.0, 140.5, 224.2, 224.1, 287.7],
          [177.7, 292.0, 329.5, 391.8, 179.0, 295.8, 341.5, 400.2]], "total decode tok/s",
         note="long reasoning; model, requests; mean of 2 rounds", gains=True)
    # Long reasoning, steady state (dev/m5/serve_bench.py, 4,096 tokens, sampled), overnight 2026-09-26.
    bars("steady-state", "Long reasoning, steady state: stock 1.1.0 vs Splish",
         ["27B, 1", "27B, 2", "27B, 3", "27B, 4", "Swift, 1", "Swift, 2", "Swift, 3", "Swift, 4"],
         ["stock 1.1.0", "Splish"],
         [[131.3, 223.2, 222.2, 299.0, 140.5, 224.2, 224.1, 287.7],
          [177.7, 292.0, 329.5, 391.8, 179.0, 295.8, 341.5, 400.2]], "aggregate tok/s",
         note="model, concurrent requests; mean of 2 rounds", gains=True)
    # Context breakdown (serve_bench --task document --warm, one request, Inco's 27B, 2 rounds).
    import json
    context = os.path.join(os.path.dirname(OUT), "context.json")
    if os.path.exists(context):
        c = json.load(open(context))
        labels = [f"{k // 1024}K" for k in c["contexts"]]
        bars("context-decode", "Decode speed by context length, one request (Inco's Qwen3.8-27B)",
             labels, ["stock 1.1.0", "Splish"], [c["decode"]["stock"], c["decode"]["splish"]],
             "decode tok/s", note="document summary, 2,048 tokens out; mean of 2 rounds", gains=True)
        bars("context-prefill", "Prefill speed by context length (cold, one request)",
             labels, ["stock 1.1.0", "Splish"], [c["prefill"]["stock"], c["prefill"]["splish"]],
             "prompt tok/s", note="first round, empty prefix cache")
    # Swift-1.5 decode step (decode-profile, 2,048-token prompt): stock 1.0.2, tuned 1.0.2, splash-m5 v8.
    bars("swift-step-time", "Swift-1.5 decode step time (lower is better)",
         ["1 request", "2 requests", "3 requests", "4 requests"],
         ["stock 1.0.2", "1.0.2 + tuned choices", "Splish (v8)"],
         [[49.0, 59.1, 87.2, 87.0], [41.5, 59.4, 88.2, 85.8], [40.5, 44.9, 59.3, 62.9]], "ms per step",
         fmt="{:.1f}", note="decode-profile, 2,048-token prompt")
    # Verify attention probes, Swift 27B, 131K history, 4 lanes (attention-sweep).
    bars("attention-probes", "Long-context attention: what each probe costs (131K context, 4 requests)",
         ["production", "no softmax", "no softmax+barriers", "no PV", "no QK", "A4 QK on 4 sg", "A6 int8 QK"],
         ["split kernel ms"], [[3.20, 2.83, 2.60, 2.06, 1.50, 3.10, 3.41]], "ms per layer", fmt="{:.2f}",
         note="diagnostics give wrong output; timing only")
    # GGUF Q8_0 staged decode tile at 32 rows (kernel-bench --gguf q80).
    bars("gguf-g4a", "GGUF Q8_0 decode at 32 rows: production vs G4a (input prefetch)",
         ["gdn_in", "gate_up", "down", "gdn_out", "attn_qkv"], ["production", "G4a"],
         [[494, 497, 478, 423, 421], [535, 531, 495, 429, 421]], "GB/s (higher is better)",
         note="bit-identical output", gains=True)


if __name__ == "__main__":
    main()
