"""Results -> Markdown / HTML / JSON report."""

from __future__ import annotations

import json
from dataclasses import asdict
from itertools import groupby

from benches.base import CLIENT_CPU_WARN_PCT, BenchResult


def _client_cpu(r: BenchResult) -> str:
    flag = "!" if r.client_cpu_pct >= CLIENT_CPU_WARN_PCT else ""
    return f"{r.client_procs}p/{r.client_cpu_pct:.0f}%{flag}"


def render(results: list[BenchResult], fmt: str = "table") -> str:
    if fmt == "json":
        return _render_json(results)
    elif fmt == "html":
        return _render_html(results)
    else:
        return _render_table(results)


# ---------------------------------------------------------------------------
# Table (terminal)
# ---------------------------------------------------------------------------
def _render_table(results: list[BenchResult]) -> str:
    if not results:
        return "No results."
    lines = ["\n" + "=" * 98]
    for bench_name, group in groupby(results, key=lambda r: r.bench_name):
        group_list = list(group)
        lines.append(f"\n  {bench_name.upper()} BENCH")
        lines.append(f"  {'─' * 96}")
        lines.append(
            f"  {'Server':<16} {'Users':>5} {'P50 ms':>8} {'P95 ms':>8} "
            f"{'P99 ms':>8} {'RPS':>8} {'Samples':>8} {'Errors':>7} {'Client CPU':>10}"
        )
        lines.append(f"  {'─' * 96}")
        for r in group_list:
            lines.append(
                f"  {r.server_name:<16} {r.concurrency:>5} "
                f"{r.p50_ms:>8.2f} {r.p95_ms:>8.2f} {r.p99_ms:>8.2f} "
                f"{r.throughput_rps:>8.1f} {r.n_samples:>8} {r.error_rate:>7.1%} "
                f"{_client_cpu(r):>10}"
            )
    lines.append(
        f"\n  Client CPU = busiest client process (% of one core); "
        f"'!' = >= {CLIENT_CPU_WARN_PCT:.0f}%, results may be client-bound."
    )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------
def _render_json(results: list[BenchResult]) -> str:
    data = []
    for r in results:
        d = asdict(r)
        del d["raw_latencies"]
        data.append(d)
    return json.dumps(data, indent=2)


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------
def _render_html(results: list[BenchResult]) -> str:
    rows = ""
    for r in results:
        rows += (
            f"<tr><td>{r.bench_name}</td><td>{r.server_name}</td>"
            f"<td>{r.concurrency}</td>"
            f"<td>{r.p50_ms:.2f}</td><td>{r.p95_ms:.2f}</td><td>{r.p99_ms:.2f}</td>"
            f"<td>{r.throughput_rps:.1f}</td><td>{r.n_samples}</td>"
            f"<td>{r.error_rate:.1%}</td><td>{_client_cpu(r)}</td></tr>\n"
        )
    return f"""<!DOCTYPE html>
<html>
<head>
<title>ML Inference Bench Report</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 2rem; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ border: 1px solid #ddd; padding: 8px; text-align: right; }}
  th {{ background: #f5f5f5; }}
  td:first-child, td:nth-child(2) {{ text-align: left; }}
</style>
</head>
<body>
<h1>ML Inference Bench Report</h1>
<table>
<tr>
  <th>Bench</th><th>Server</th><th>Users</th>
  <th>P50 ms</th><th>P95 ms</th><th>P99 ms</th>
  <th>RPS</th><th>Samples</th><th>Errors</th><th>Client CPU</th>
</tr>
{rows}</table>
<p>Client CPU = busiest client process (% of one core); "!" means the client may be the bottleneck.</p>
</body>
</html>"""


# ---------------------------------------------------------------------------
# CLI: merge JSON results (e.g. one file per server) into one report
# ---------------------------------------------------------------------------
def load_json(paths: list[str]) -> list[BenchResult]:
    results = []
    for path in paths:
        with open(path) as f:
            results.extend(BenchResult(**d) for d in json.load(f))
    return results


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Merge JSON bench results")
    p.add_argument("files", nargs="+")
    p.add_argument("--output", choices=["table", "json", "html"], default="table")
    p.add_argument("--out")
    args = p.parse_args()
    results = sorted(load_json(args.files), key=lambda r: r.bench_name)
    output = render(results, fmt=args.output)
    if args.out:
        with open(args.out, "w") as f:
            f.write(output)
    else:
        print(output)
