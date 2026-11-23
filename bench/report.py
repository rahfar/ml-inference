"""Experiment results -> markdown table.

uv run python -m bench.report results/strategy.json
"""

import json
import sys

from bench.load import CLIENT_CPU_WARN_PCT

PHASES = ("decode", "wait", "predict", "encode")


def table(result: dict) -> str:
    varied = result["vary"]
    per_row = "samples" in varied  # throughput in samples/s matters when it varies
    header = [*varied, "users", "RPS", *(["samples/s"] if per_row else [])]
    header += [
        "P50 ms",
        "P99 ms",
        *(f"{p} ms" for p in PHASES),
        "batch",
        "errors",
        "client CPU",
    ]

    lines = [
        "Fixed: " + ", ".join(f"{k}={v}" for k, v in result["fixed"].items()),
        "Server-side ms are means per request; batch = rows per predict() call.",
        "",
        "| " + " | ".join(header) + " |",
        "|"
        + "|".join("---" if i < len(varied) else "---:" for i in range(len(header)))
        + "|",
    ]
    for r in result["rows"]:
        cpu = f"{r['client_cpu_pct']:.0f}%" + (
            " !" if r["client_cpu_pct"] >= CLIENT_CPU_WARN_PCT else ""
        )
        cells = [str(r[k]) for k in varied]
        cells += [str(r["concurrency"]), f"{r['rps']:.0f}"]
        cells += [f"{r['rps'] * r['samples']:.0f}"] if per_row else []
        cells += [f"{r['p50_ms']:.2f}", f"{r['p99_ms']:.2f}"]
        cells += [f"{r['server_ms'].get(p, float('nan')):.2f}" for p in PHASES]
        cells += [f"{r['batch_size']:.1f}", str(r["errors"]), cpu]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    for path in sys.argv[1:]:
        with open(path) as f:
            print(table(json.load(f)), end="\n\n")
