"""Latency percentiles (linear interpolation) and result aggregation."""


def percentile(sorted_values: list[float], p: float) -> float:
    """Linear-interpolation percentile. p in [0, 100]."""
    n = len(sorted_values)
    if n == 0:
        return float("nan")
    if n == 1:
        return sorted_values[0]
    idx = (p / 100) * (n - 1)
    lo = int(idx)
    hi = lo + 1
    if hi >= n:
        return sorted_values[-1]
    frac = idx - lo
    return sorted_values[lo] + frac * (sorted_values[hi] - sorted_values[lo])


def compute(
    latencies: list[float],
    errors: int,
    elapsed: float,
    batch_size: int,
) -> dict:
    n = len(latencies)
    rps = n / elapsed if elapsed > 0 else 0.0

    lat_ms = sorted(lat * 1000 for lat in latencies)
    if lat_ms:
        lat_stats = {
            "lat_avg_ms": sum(lat_ms) / len(lat_ms),
            "lat_p50_ms": percentile(lat_ms, 50),
            "lat_p95_ms": percentile(lat_ms, 95),
            "lat_p99_ms": percentile(lat_ms, 99),
            "lat_max_ms": lat_ms[-1],
        }
    else:
        lat_stats = {k: float("nan") for k in
                     ("lat_avg_ms", "lat_p50_ms", "lat_p95_ms", "lat_p99_ms", "lat_max_ms")}

    return {
        "requests": n,
        "errors": errors,
        "elapsed_s": elapsed,
        "rps": rps,
        "vessels_per_sec": rps * batch_size,
        **lat_stats,
    }
