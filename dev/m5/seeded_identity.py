#!/usr/bin/env python3
"""Seeded sampled identity between Splash servers: the same prompt and seed at temperature 1.0
(top_p 0.95, top_k 20) must give the same text on engines whose arithmetic is bit-identical.
Stronger than greedy: any logit difference changes which token gets sampled somewhere.

  dev/m5/seeded_identity.py OUT.json REF=PORT OTHER=PORT [...] [--seeds 3] [--tokens 384]

Reports, per engine, how many (prompt, seed) outputs match the reference byte for byte, where the
first difference falls, and a repeat of the reference's first seed (run-to-run determinism).
"""
import json, os, sys, time, urllib.request

KEY = open(os.path.expanduser("~/.splash/api-key")).read().strip()
MODEL = os.environ.get("MODEL", "local/Swift-1.5-4bit-MLX-Splash")
PROMPTS = [
    "Write a Python function that merges overlapping intervals, with a docstring and three tests.",
    "Explain how vaccines train the immune system, for a teenager.",
    "Write a short story about a cartographer who maps a city that keeps changing.",
    "A tank fills at 12 L/min and drains at 7 L/min from 480 L; when is it 1,000 L? Show your working.",
    "Give ten tips for learning a new language as an adult, one sentence each.",
    "Summarise the main arguments for and against nuclear power in about 250 words.",
]


def call(port, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=900))


def main():
    args = sys.argv[1:]
    seeds, tokens = 3, 384
    if "--seeds" in args:
        i = args.index("--seeds"); seeds = int(args[i + 1]); del args[i:i + 2]
    if "--tokens" in args:
        i = args.index("--tokens"); tokens = int(args[i + 1]); del args[i:i + 2]
    out, engines = args[0], [a.split("=") for a in args[1:]]
    texts = {label: {} for label, _ in engines}
    for p, prompt in enumerate(PROMPTS):
        for seed in range(1, seeds + 1):
            for label, port in engines:
                d = call(port, {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": tokens,
                                "temperature": 1.0, "top_p": 0.95, "top_k": 20, "seed": seed,
                                "reasoning_effort": "none"})
                texts[label][f"{p}:{seed}"] = d["choices"][0]["message"]["content"]
        print(f"prompt {p + 1}/{len(PROMPTS)}", file=sys.stderr, flush=True)
    ref_label, ref_port = engines[0]
    repeat = call(ref_port, {"model": MODEL, "messages": [{"role": "user", "content": PROMPTS[0]}], "max_tokens": tokens,
                             "temperature": 1.0, "top_p": 0.95, "top_k": 20, "seed": 1, "reasoning_effort": "none"})
    run_to_run = repeat["choices"][0]["message"]["content"] == texts[ref_label]["0:1"]
    json.dump(texts, open(out, "w"), indent=1)
    print(f"reference {ref_label}: run-to-run {'identical' if run_to_run else 'DIFFERENT'} (prompt 1, seed 1)")
    for label, _ in engines[1:]:
        same, firsts = 0, []
        for key, ref in texts[ref_label].items():
            other = texts[label][key]
            if other == ref:
                same += 1
            else:
                firsts.append(next((i for i, (a, b) in enumerate(zip(ref, other)) if a != b), min(len(ref), len(other))))
        total = len(texts[ref_label])
        print(f"{label}: identical to {ref_label} {same}/{total}" +
              (f"; first difference at characters {sorted(firsts)}" if firsts else ""))


if __name__ == "__main__":
    main()
