"""Where and when predict() runs. This is the file that matters.

Everything runs in one process, and every strategy exposes the same coroutine:

    result = await strategy.predict(x)      # x: (n, 150) -> result.y: (n, 30)

inline_loop        call predict() directly in the async handler; the event
                   loop is blocked until it returns
inline_threadpool  run predict() in the server's thread pool, as FastAPI does
                   for a sync `def` handler; many predictions run at once
serial             queue -> one inference thread, one request at a time
batched            queue -> one inference thread; everything waiting is
                   concatenated into a single predict() call

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
    def __init__(self, predict: PredictFn):
        self._predict = predict

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


class Batched(Serial):
    """Like serial, but each predict() call takes everything that is waiting.

    No added delay at low load: a request reaching an idle server runs at once,
    as a batch of 1. Under load, requests pile up while a batch is running and
    go together in the next one.
    """

    def __init__(self, predict: PredictFn, max_batch: int = 64):
        super().__init__(predict)
        self.max_batch = max_batch  # rows per predict() call

    async def start(self) -> None:
        await super().start()
        self._queue: asyncio.Queue[tuple[np.ndarray, asyncio.Future]] = asyncio.Queue()
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._task.cancel()
        await super().stop()

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
