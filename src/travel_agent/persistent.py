"""常驻 event loop 基础设施（阶段 6.4）。

当前 CLI 形态每轮一次 ``asyncio.run``：loop 随请求生灭，带来三个已知问题
（详见 schedule 6.4）：

1. OpenAI 客户端缓存的 httpx 连接池挂在已关闭的 loop 上，下一轮报
   "Event loop is closed"，评测 runner 被迫加 workaround；
2. 审批过期清理、流式输出等能力要求 loop 在请求之间存活；
3. Agent/checkpointer 每轮重建。

职责边界：本模块只提供「一个 loop 活到底 + 在它上面跑协程」的机制，
不持有任何业务状态。业务（service）决定什么被复用。
"""

from __future__ import annotations

import asyncio
import queue as queue_module
import threading
from typing import AsyncIterator, Awaitable, Iterator, TypeVar

T = TypeVar("T")


class PersistentLoopRunner:
    """Own one event loop for the whole process lifetime.

    All async work is submitted to that loop, so connections, semaphores and
    cached clients are created and destroyed inside the same loop they are
    used in. This removes the per-turn ``asyncio.run`` that caused the
    cross-loop failures seen in evaluation batches.
    """

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def start(self) -> None:
        with self._lock:
            if self._loop is not None and self._loop.is_running():
                return
            self._loop = asyncio.new_event_loop()
            self._thread = threading.Thread(
                target=self._run_forever, name="travel-agent-loop", daemon=True
            )
            self._thread.start()

    def _run_forever(self) -> None:
        assert self._loop is not None
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def run_coro(self, coro: Awaitable[T]) -> T:
        """Submit a coroutine and block until it finishes."""
        self.start()
        loop = self._loop
        assert loop is not None
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        return future.result()

    def run_coro_with_timeout(self, coro: Awaitable[T], timeout: float) -> T:
        """Submit with a hard ceiling; the coroutine is cancelled on expiry.

        ``timeout`` is the hard ceiling for the whole call. Expiry cancels the
        asyncio task inside the loop - ``concurrent.futures.Future.cancel()``
        alone cannot stop a task that already started - waits for the
        cancellation to finish so no model call is left hanging on the loop,
        then raises ``TimeoutError``.

        Callers wanting a domain-specific error (the service's ``TurnTimeout``)
        wrap their coroutine in ``asyncio.wait_for`` and treat this ceiling
        purely as a stuck-call net; the soft budget then fires first and the
        domain error propagates out of ``future.result()``.
        """
        self.start()
        loop = self._loop
        assert loop is not None

        # 包装协程：把 asyncio.Task 句柄交给外部，超时后才能从另一线程
        # 真正取消它。future.cancel() 只对未开始的任务有效。
        task_holder: dict[str, asyncio.Task] = {}

        async def _managed() -> T:
            task = asyncio.current_task()
            assert task is not None
            task_holder["task"] = task
            return await coro

        future = asyncio.run_coroutine_threadsafe(_managed(), loop)
        try:
            return future.result(timeout)
        except TimeoutError:
            task = task_holder.get("task")
            if task is not None:
                loop.call_soon_threadsafe(task.cancel)
            # 等取消处理完（协程跑完 finally、释放连接）再返回。
            try:
                future.result(timeout=10)
            except Exception:  # noqa: BLE001 - 取消路径上的异常不是关注点
                pass
            raise

    def iter_async(self, agen: AsyncIterator[T]) -> Iterator[T]:
        """Bridge an async generator into a sync iterator.

        Streaming endpoints are synchronous from the server framework's point of
        view (``StreamingResponse`` drains a plain iterator). The async generator
        must still run on the resident loop - that is where the agent, its
        connections and its locks live - so a thread-safe queue carries items
        across the boundary.

        Exceptions raised inside the generator are re-raised in the consuming
        thread after everything already produced has been delivered: a client
        that saw half a turn must also see why the rest never arrived.
        """
        self.start()
        loop = self._loop
        assert loop is not None
        items: queue_module.Queue = queue_module.Queue()
        done = object()

        async def _pump() -> None:
            try:
                async for item in agen:
                    items.put((True, item))
            except Exception as error:  # noqa: BLE001 - 交回消费线程处理
                items.put((False, error))
            finally:
                items.put((True, done))

        asyncio.run_coroutine_threadsafe(_pump(), loop)
        while True:
            ok, value = items.get()
            if value is done:
                return
            if not ok:
                raise value
            yield value

    def stop(self) -> None:
        """Stop the loop thread and release the loop's resources.

        ``run_forever`` returning is not enough: the loop still owns a selector
        with a self-pipe. Leaving it unclosed makes ``BaseEventLoop.__del__``
        raise ``Invalid file descriptor: -1`` at garbage-collection time - the
        very class of shutdown noise this module exists to remove - so the loop
        is closed explicitly once its thread has finished.
        """
        with self._lock:
            if self._loop is None:
                return
            loop, self._loop = self._loop, None
            thread = self._thread
        loop.call_soon_threadsafe(loop.stop)
        if thread is not None:
            thread.join(timeout=5)
        if not loop.is_running():
            loop.close()
