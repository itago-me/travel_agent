"""阶段 6.4：常驻 loop 的生命周期、并发正确性回归。

三个此前无法测试的属性：

1. 同一 loop 复用——跨 loop "Event loop is closed" 从根上消失；
2. 同 thread 串行——并发 chat 不会竞争 checkpoint；
3. 超时真正取消——被打断的模型调用不会留下悬挂任务。
"""

import asyncio
import time

import pytest

from travel_agent.persistent import PersistentLoopRunner
from travel_agent.service import _THREAD_LOCKS, _persistent_runner


def test_runner_reuses_one_loop_across_submissions():
    """所有提交都在同一个 loop 上执行——这是 6.4 的立身之本。"""
    runner = PersistentLoopRunner()

    async def loop_id():
        return id(asyncio.get_running_loop())

    ids = [runner.run_coro(loop_id()) for _ in range(3)]

    assert len(set(ids)) == 1
    runner.stop()


def test_runner_timeout_cancels_the_coroutine():
    """超时不能只是放弃等待——悬挂的协程必须被取消。"""
    runner = PersistentLoopRunner()
    cancelled = []

    async def hang():
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            cancelled.append(True)
            raise

    started = time.monotonic()
    with pytest.raises(TimeoutError):
        runner.run_coro_with_timeout(hang(), timeout=0.2)

    assert cancelled, "协程必须收到取消信号"
    assert time.monotonic() - started < 5
    runner.stop()


def test_thread_lock_serialises_same_thread_concurrently():
    """同一 thread_id 的两个协程必须一先一后，不允许重叠。"""
    import travel_agent.service as service_module

    async def scenario():
        lock = service_module._THREAD_LOCKS.for_thread("thread-a")
        order = []

        async def worker(name, delay):
            async with lock:
                order.append(f"{name}-start")
                await asyncio.sleep(delay)
                order.append(f"{name}-end")

        await asyncio.gather(worker("first", 0.05), worker("second", 0.0))
        return order

    order = service_module._persistent_runner().run_coro(scenario())

    assert order == ["first-start", "first-end", "second-start", "second-end"]


def test_thread_locks_are_independent_across_threads():
    """不同 thread_id 不互斥——这是并发能力的全部意义。"""
    import travel_agent.service as service_module

    async def scenario():
        lock_a = service_module._THREAD_LOCKS.for_thread("thread-a")
        lock_b = service_module._THREAD_LOCKS.for_thread("thread-b")

        async def hold():
            async with lock_a:
                await asyncio.sleep(0.1)

        # b 不等 a：立即拿到锁
        got_b = asyncio.ensure_future(lock_b.acquire())
        await asyncio.sleep(0.02)
        assert got_b.done(), "不同 thread 的锁不应互斥"
        lock_b.release()
        await hold()

    service_module._persistent_runner().run_coro(scenario())


def test_locks_survive_across_turns():
    """同一 thread 的锁要跨轮复用，否则串行化形同虚设。"""
    first = _THREAD_LOCKS.for_thread("thread-x")
    second = _THREAD_LOCKS.for_thread("thread-x")

    assert first is second


def test_persistent_runner_is_process_wide():
    assert _persistent_runner() is _persistent_runner()
