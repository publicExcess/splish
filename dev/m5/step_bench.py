#!/usr/bin/env python3
"""splash-m5 step bench: whole-model decode step time, A/B, with statistics.

Runs dev/benchmarks/decode_profile (build/engine-tests/decode-profile) alternately
for each configuration (ABAB... over --rounds), and reports for every batch width:

  fused   the real, one-command GPU time of a DFlash decode cycle (the serving path)
  parts   the sum of the same dispatches replayed one per command
  overlap parts - fused: what the fused command gains from kernel overlap at
          boundaries (thesis: large threadgroups lose it; FORK.md)

plus paired differences against the first configuration with a 95% interval, and
per-pipeline attributed time for pipelines that differ. A configuration is
LABEL=CHOICES_FILE[@METALLIB][%VAR=VALUE,...] (CHOICES_FILE "-" for none): configurations share the
decode-profile binary, and may bring their own metallib (kernel-only changes).

  dev/m5/step_bench.py --rounds 4 base=tuning/a.choices new=tuning/b.choices
"""
import argparse, math, os, re, statistics as st, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# The model package or GGUF assembly root: --package, or SPLISH_PACKAGE.
PACKAGE = os.environ.get("SPLISH_PACKAGE", "")


def run_profile(spec, prompt, cycles, binary, metallib, package=PACKAGE):
    spec, _, env_spec = spec.partition("%")
    choices, _, own_lib = spec.partition("@")
    metallib = own_lib or metallib
    env = dict(os.environ)
    for item in filter(None, env_spec.split(",")):
        key, _, value = item.partition("=")
        env[key] = value
    env.pop("SPLASH_KERNEL_CHOICES", None)
    if choices != "-":
        env["SPLASH_KERNEL_CHOICES"] = os.path.abspath(choices)
    out = subprocess.run([binary, metallib, package, "--prompt-tokens", str(prompt), "--cycles", str(cycles)],
                         capture_output=True, text=True, env=env, cwd=ROOT)
    widths = parse(out.stdout)
    # decode-profile can end with "completed request was decoded" once a synthetic
    # request reaches its end during the extra cycles, after every width has
    # already been measured and printed; accept a run only if all four are there.
    if out.returncode and not (len(widths) == 4 and "completed request was decoded" in out.stderr + out.stdout):
        sys.exit(f"decode-profile failed ({spec}):\n{out.stdout[-800:]}{out.stderr[-800:]}")
    return widths


def parse(text):
    widths = {}
    for block in re.finditer(r"== (B\d) decode cycle: ([\d.]+) ms fused, ([\d.]+) ms as \d+ separate dispatches ==\n"
                             r"pipeline.*\n((?:.+\n)+)", text):
        rows = {m[1]: float(m[3]) for m in re.finditer(r"^(\S+)\s+([\d.]+)\s+([\d.]+)\s+[\d.]+%", block[4], re.M)}
        widths[block[1]] = {"fused": float(block[2]), "parts": float(block[3]), "rows": rows}
    medians = {m[1]: float(m[2]) for m in re.finditer(r"(B\d) decode cycle: median fused gpu ([\d.]+) ms", text)}
    for w, v in widths.items():
        v["median"] = medians.get(w, v["fused"])
    # A width whose attributed replay was cut short still has its fused median.
    for w, m in medians.items():
        widths.setdefault(w, {"fused": m, "parts": float("nan"), "rows": {}, "median": m})
    return widths


# Two-sided 95% Student-t critical values: a handful of rounds is far from the normal 1.96.
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262,
        10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 19: 2.093,
        24: 2.064, 29: 2.045}


def interval(values):
    m = st.mean(values)
    if len(values) < 2:
        return m, float("nan")
    df = len(values) - 1
    # The nearest tabulated df at or below this one: conservative apart from rounding the
    # tabulated critical values to three decimal places.
    t = T975[max(k for k in T975 if k <= df)]
    return m, t * st.stdev(values) / math.sqrt(len(values))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="+", help="LABEL=CHOICES_FILE or LABEL=-")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--cycles", type=int, default=9)
    ap.add_argument("--prompt-tokens", type=int, default=2048)
    ap.add_argument("--binary", default=os.path.join(ROOT, "build/engine-tests/decode-profile"))
    ap.add_argument("--package", default=PACKAGE, help="package or GGUF assembly root")
    ap.add_argument("--metallib", default=os.path.join(ROOT, "build/splash.metallib"))
    ap.add_argument("--show", default="", help="comma-separated pipelines always listed with their attributed ms")
    args = ap.parse_args()
    if not args.package:
        sys.exit("step_bench: give --package (or set SPLISH_PACKAGE)")
    configs = [c.split("=", 1) for c in args.configs]
    results = {label: [] for label, _ in configs}
    for r in range(args.rounds):
        order = configs if r % 2 == 0 else configs[::-1]
        for label, choices in order:
            results[label].append(run_profile(choices, args.prompt_tokens, args.cycles, args.binary, args.metallib, args.package))
        print(f"round {r + 1}/{args.rounds}", file=sys.stderr, flush=True)
    base = configs[0][0]
    widths = sorted(results[base][0])
    print(f"step bench: {args.rounds} rounds x {args.cycles} cycles, {args.prompt_tokens}-token prompt; ms per cycle")
    for w in widths:
        print(f"\n{w}")
        print(f"  {'config':14} {'fused (median)':>16} {'parts':>9} {'overlap':>8} {'vs ' + base:>22}")
        for label, _ in configs:
            f = [x[w]["median"] for x in results[label]]
            p = [x[w]["parts"] for x in results[label]]
            o = [x[w]["parts"] - x[w]["fused"] for x in results[label]]
            fm, fh = interval(f)
            line = f"  {label:14} {fm:9.2f} ±{fh:4.2f} {st.mean(p):9.2f} {st.mean(o):8.2f}"
            if label != base:
                d = [a[w]["median"] - b[w]["median"] for a, b in zip(results[label], results[base])]
                dm, dh = interval(d)
                line += f"   {dm:+6.2f} ±{dh:4.2f} ms ({100 * dm / st.mean(x[w]['median'] for x in results[base]):+5.1f}%)"
            print(line)
        # Pipelines whose attributed time differs between configurations.
        names = set()
        for label, _ in configs:
            for x in results[label]:
                names |= set(x[w]["rows"])
        diffs = []
        for n in names:
            means = [st.mean(x[w]["rows"].get(n, 0.0) for x in results[label]) for label, _ in configs]
            if max(means) - min(means) > 0.05 or n in args.show.split(","):
                diffs.append((n, means))
        if diffs:
            print("  attributed ms by pipeline (where configurations differ):")
            for n, means in sorted(diffs, key=lambda d: -max(d[1])):
                print(f"    {n:46} " + "  ".join(f"{m:7.3f}" for m in means))


if __name__ == "__main__":
    main()
