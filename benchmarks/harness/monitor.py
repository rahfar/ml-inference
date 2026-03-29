"""Background CPU/RSS sampling for one or more processes."""

import threading
import time
from dataclasses import dataclass, field

import psutil


@dataclass
class Sample:
    label: str
    mem_mb: float
    cpu_pct: float


@dataclass
class Monitor:
    _thread: threading.Thread = field(repr=False)
    _stop: threading.Event = field(repr=False)
    samples: list[Sample] = field(default_factory=list)

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=3)

    def stats_for(self, label: str) -> dict:
        vals = [s for s in self.samples if s.label == label]
        if not vals:
            return {"mem_avg_mb": float("nan"), "mem_max_mb": float("nan"),
                    "cpu_avg_pct": float("nan"), "cpu_max_pct": float("nan")}
        mems = [s.mem_mb for s in vals]
        cpus = [s.cpu_pct for s in vals]
        return {
            "mem_avg_mb": sum(mems) / len(mems),
            "mem_max_mb": max(mems),
            "cpu_avg_pct": sum(cpus) / len(cpus),
            "cpu_max_pct": max(cpus),
        }


def start_monitor(pids: dict[str, int], interval: float = 0.5) -> Monitor:
    """pids: {label: pid}. Returns a Monitor; call .stop() when done."""
    stop_evt = threading.Event()
    samples: list[Sample] = []

    def _run():
        procs = {}
        for label, pid in pids.items():
            try:
                p = psutil.Process(pid)
                p.cpu_percent()  # initialise counter
                procs[label] = p
            except psutil.NoSuchProcess:
                pass

        time.sleep(interval)
        while not stop_evt.is_set():
            for label, p in list(procs.items()):
                try:
                    samples.append(Sample(
                        label=label,
                        mem_mb=p.memory_info().rss / 1024 / 1024,
                        cpu_pct=p.cpu_percent(),
                    ))
                except psutil.NoSuchProcess:
                    procs.pop(label, None)
            time.sleep(interval)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    mon = Monitor(_thread=t, _stop=stop_evt, samples=samples)
    return mon
