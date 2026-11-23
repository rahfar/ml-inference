"""Experiment results -> markdown tables.

uv run python -m bench.report results/strategy.json             # every row
uv run python -m bench.report results/strategy.json --pivot rps # one metric,
    one row per variant, one column per concurrency level
"""

import argparse
import json

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


def _samples(result: dict, r: dict) -> int:
    return r.get("samples", result["fixed"].get("samples", 1))


METRICS = {
    "rps": lambda result, r: f"{r['rps']:,.0f}",
    "samples/s": lambda result, r: f"{r['rps'] * _samples(result, r):,.0f}",
    "p50": lambda result, r: f"{r['p50_ms']:.2f}",
    "p99": lambda result, r: f"{r['p99_ms']:.2f}",
    "codec ms": lambda result, r: (
        f"{r['server_ms']['decode'] + r['server_ms']['encode']:.2f}"
    ),
}


def pivot(
    result: dict, metric: str, columns: tuple[str, ...] = ("concurrency",)
) -> str:
    """One metric: a row per variant, a column per value of `columns`."""
    row_keys = [k for k in result["vary"] if k not in columns]
    cells: dict[tuple, dict[tuple, str]] = {}
    for r in result["rows"]:
        row = tuple(r[k] for k in row_keys)
        col = tuple(r[k] for k in columns)
        cells.setdefault(row, {})[col] = METRICS[metric](result, r)
    cols = list(dict.fromkeys(c for row in cells.values() for c in row))
    names = {"concurrency": "users"}
    col_label = lambda c: ", ".join(
        f"{names.get(k, k)} {v}" for k, v in zip(columns, c)
    )
    lines = [
        "| " + " | ".join([*row_keys, *map(col_label, cols)]) + " |",
        "|" + "|".join(["---"] * len(row_keys) + ["---:"] * len(cols)) + "|",
    ]
    for row, values in cells.items():
        lines.append(
            "| "
            + " | ".join([*map(str, row), *(values.get(c, "") for c in cols)])
            + " |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Print experiment results as markdown")
    p.add_argument("files", nargs="+")
    p.add_argument("--pivot", choices=METRICS, help="show one metric as a pivot table")
    p.add_argument(
        "--columns", nargs="+", default=["concurrency"], help="pivot columns"
    )
    args = p.parse_args()
    for path in args.files:
        with open(path) as f:
            result = json.load(f)
        out = (
            pivot(result, args.pivot, tuple(args.columns))
            if args.pivot
            else table(result)
        )
        print(out, end="\n\n")
