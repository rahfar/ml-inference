"""Results -> Markdown / HTML / JSON report."""

from __future__ import annotations

import json
from dataclasses import asdict
from itertools import groupby

from benches.base import BenchResult


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
    lines = ["\n" + "=" * 90]
    for bench_name, group in groupby(results, key=lambda r: r.bench_name):
        group_list = list(group)
        lines.append(f"\n  {bench_name.upper()} BENCH")
        lines.append(f"  {'─' * 84}")
        lines.append(
            f"  {'Server':<16} {'Workers':>7} {'P50 ms':>8} {'P95 ms':>8} "
            f"{'P99 ms':>8} {'RPS':>8} {'Errors':>7}"
        )
        lines.append(f"  {'─' * 84}")
        for r in group_list:
            lines.append(
                f"  {r.server_name:<16} {r.concurrency:>7} "
                f"{r.p50_ms:>8.2f} {r.p95_ms:>8.2f} {r.p99_ms:>8.2f} "
                f"{r.throughput_rps:>8.1f} {r.error_rate:>7.1%}"
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
            f"<td>{r.throughput_rps:.1f}</td><td>{r.error_rate:.1%}</td></tr>\n"
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
  <th>Bench</th><th>Server</th><th>Workers</th>
  <th>P50 ms</th><th>P95 ms</th><th>P99 ms</th>
  <th>RPS</th><th>Errors</th>
</tr>
{rows}</table>
</body>
</html>"""
