"""Where and when predict() runs. This is the file that matters.

Everything runs in one process, and every strategy exposes the same coroutine:

    result = await strategy.predict(x)      # x: (n, 150) -> result.y: (n, 30)

inline_loop        call predict() directly in the async handler; the event
                   loop is blocked until it returns
inline_threadpool  run predict() in the server's thread pool, as FastAPI does
                   for a sync `def` handler; many predictions run at once
serial             queue -> one inference thread, one request at a time
batched            queue -> `workers` inference threads (default 1); each
                   one takes everything waiting as a single predict() call

serial vs inline_threadpool shows the effect of limiting concurrency;
batched vs serial shows the effect of batching on top of that.
"""

import asyncio
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import NamedTuple

import numpy as np
from starlette.concurrency import run_in_threadpool

PredictFn = Callable[[np.ndarray], np.ndarray]


class Result(NamedTuple):
    y: np.ndarray
    predict_s: float  # duration of the predict() call that produced y
    batch_size: int  # rows in that predict() call


def timed_predict(predict: PredictFn, x: np.ndarray) -> Result:
    start = time.perf_counter()
    y = predict(x)
    return Result(y, time.perf_counter() - start, len(x))


class Strategy:
    def __init__(self, predict: PredictFn, workers: int = 1):
        self._predict = predict
        self.workers = workers  # inference threads; only batched uses more than 1

    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    async def predict(self, x: np.ndarray) -> Result:
        raise NotImplementedError


class InlineLoop(Strategy):
    async def predict(self, x: np.ndarray) -> Result:
        return timed_predict(self._predict, x)


class InlineThreadpool(Strategy):
    async def predict(self, x: np.ndarray) -> Result:
        # The same pool FastAPI uses for sync handlers (40 threads by default).
        return await run_in_threadpool(timed_predict, self._predict, x)


class Serial(Strategy):
    async def start(self) -> None:
        # A one-thread executor *is* a queue plus one worker.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="predict")

    async def stop(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    async def predict(self, x: np.ndarray) -> Result:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor, timed_predict, self._predict, x
        )


class Batched(Strategy):
    """A queue and `workers` inference threads. Each predict() call takes
    everything that is waiting, up to max_batch rows.

    No added delay at low load: a request reaching an idle worker runs at once,
    as a batch of 1. Under load, requests pile up while the workers are busy
    and go together in the next batch. Several workers run batches in parallel,
    each on its own cores.
    """

    def __init__(self, predict: PredictFn, workers: int = 1, max_batch: int = 64):
        super().__init__(predict, workers)
        self.max_batch = max_batch  # rows per predict() call

    async def start(self) -> None:
        self._executor = ThreadPoolExecutor(self.workers, thread_name_prefix="predict")
        self._queue: asyncio.Queue[tuple[np.ndarray, asyncio.Future]] = asyncio.Queue()
        self._tasks = [asyncio.create_task(self._run()) for _ in range(self.workers)]

    async def stop(self) -> None:
        for task in self._tasks:
            task.cancel()
        self._executor.shutdown(wait=False, cancel_futures=True)

    async def predict(self, x: np.ndarray) -> Result:
        future = asyncio.get_running_loop().create_future()
        self._queue.put_nowait((x, future))
        return await future

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            waiting = [await self._queue.get()]
            rows = len(waiting[0][0])
            while rows < self.max_batch and not self._queue.empty():
                waiting.append(self._queue.get_nowait())
                rows += len(waiting[-1][0])

            batch = np.concatenate([x for x, _ in waiting])
            try:
                result = await loop.run_in_executor(
                    self._executor, timed_predict, self._predict, batch
                )
            except Exception as e:  # noqa: BLE001 -- hand the error to every caller
                for _, future in waiting:
                    if not future.done():
                        future.set_exception(e)
                continue

            start = 0
            for x, future in waiting:
                if not future.done():  # the client may have gone away
                    future.set_result(
                        result._replace(y=result.y[start : start + len(x)])
                    )
                start += len(x)


STRATEGIES: dict[str, type[Strategy]] = {
    "inline_loop": InlineLoop,
    "inline_threadpool": InlineThreadpool,
    "serial": Serial,
    "batched": Batched,
}
