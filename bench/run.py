"""Run one experiment from experiments.yaml.

Every combination of the experiment's `vary` settings is one variant. For each
variant it starts a fresh server process, runs every concurrency level and
stops the server, so no state carries over between variants.

    uv run python -m bench.run strategy
    uv run python -m bench.run codec --duration 3 --levels 1 16     # quick look

Writes results/<experiment>.json and prints the table (see bench/report.py).
Server logs go to results/logs/.
"""

import argparse
import itertools
import json
import subprocess
import sys
import time
import urllib.request
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

import yaml

from bench.load import CLIENT_CPU_WARN_PCT, run_level
from bench.report import table

PORT = 8000
URL = f"http://127.0.0.1:{PORT}"
RESULTS = Path("results")
SERVER_SETTINGS = ("stack", "model", "strategy", "codec", "threads")


def variants(exp: dict) -> list[dict]:
    """{'vary': {'a': [1, 2], 'b': ['x']}} -> [{'a': 1, 'b': 'x'}, {'a': 2, 'b': 'x'}]"""
    names = list(exp["vary"])
    return [
        dict(zip(names, values, strict=True))
        for values in itertools.product(*exp["vary"].values())
    ]


def _healthy() -> bool:
    try:
        with urllib.request.urlopen(f"{URL}/health", timeout=0.5) as resp:
            return resp.status == 200
    except OSError:
        return False


@contextmanager
def server(settings: dict, log_path: Path):
    if _healthy():
        raise RuntimeError(f"something is already serving {URL}; stop it first")
    cmd = [sys.executable, "-m", "serving.app", f"--port={PORT}"]
    cmd += [f"--{k}={settings[k]}" for k in SERVER_SETTINGS]
    with open(log_path, "w") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 60
            while not _healthy():
                if proc.poll() is not None:
                    raise RuntimeError(
                        f"server exited with {proc.returncode}, see {log_path}"
                    )
                if time.monotonic() > deadline:
                    raise RuntimeError(f"server not ready after 60 s, see {log_path}")
                time.sleep(0.25)
            yield
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


def main() -> None:
    p = argparse.ArgumentParser(description="Run one experiment from experiments.yaml")
    p.add_argument("experiment")
    p.add_argument(
        "--duration", type=float, help="override measurement seconds per level"
    )
    p.add_argument("--levels", type=int, nargs="+", help="override concurrency levels")
    p.add_argument("--config", default="experiments.yaml")
    args = p.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    exp = {**config["defaults"], **config["experiments"][args.experiment]}
    if args.duration:
        exp["duration_s"] = args.duration
    if args.levels:
        exp["levels"] = args.levels
    base = {k: v for k, v in exp.items() if k not in ("fixed", "vary")} | exp["fixed"]

    (RESULTS / "logs").mkdir(parents=True, exist_ok=True)
    rows = []
    for varied in variants(exp):
        settings = base | varied
        label = " ".join(f"{k}={v}" for k, v in varied.items())
        print(f"\n== {label}", file=sys.stderr, flush=True)
        with server(settings, RESULTS / "logs" / f"{args.experiment}.log"):
            for level in settings["levels"]:
                r = run_level(
                    URL,
                    settings["codec"],
                    settings["samples"],
                    level,
                    settings["warmup_s"],
                    settings["duration_s"],
                    settings["client_processes"],
                )
                flag = (
                    "  <- client-bound"
                    if r.client_cpu_pct >= CLIENT_CPU_WARN_PCT
                    else ""
                )
                print(
                    f"   c={r.concurrency:<3} {r.rps:8.0f} RPS  P50 {r.p50_ms:7.2f} ms  "
                    f"P99 {r.p99_ms:7.2f} ms  batch {r.batch_size:5.1f}  errors {r.errors}  "
                    f"client CPU {r.client_cpu_pct:3.0f}%{flag}",
                    file=sys.stderr,
                    flush=True,
                )
                rows.append(varied | asdict(r))

    result = {
        "experiment": args.experiment,
        "fixed": {
            k: base[k] for k in (*SERVER_SETTINGS, "samples") if k not in exp["vary"]
        },
        "vary": list(exp["vary"]),
        "rows": rows,
    }
    out = RESULTS / f"{args.experiment}.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"\nWrote {out}\n", file=sys.stderr)
    print(table(result))


if __name__ == "__main__":
    main()
